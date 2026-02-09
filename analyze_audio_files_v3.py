import os
import csv
import sys
import yaml
import subprocess
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment

# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path):
    """Load configuration from YAML file with error handling."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Config load error: {e}")
        sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config', 'settings.yaml')
CONFIG = load_config(CONFIG_PATH)

INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['input_dir']))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['output_dir']))
COORDINATES_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['coordinates_file']))
MIN_CONFIDENCE = CONFIG.get('min_confidence', 0.1)

SEGMENT_SETTINGS = CONFIG.get('segment_extraction', {})
EXTRACT_SEGMENTS = SEGMENT_SETTINGS.get('enabled', False)
SEGMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, SEGMENT_SETTINGS.get('output_dir', 'segments'))
THREADS = SEGMENT_SETTINGS.get('threads', 4)

# FIX: Ensure these are available for the extraction logic
MAX_SEGMENTS_PER_SPECIES = SEGMENT_SETTINGS.get('max_segments', 20)
SEG_LENGTH = SEGMENT_SETTINGS.get('segment_length_sec', 3)

BIRDNET_ANALYZER_PATH = CONFIG.get('birdnet_analyzer_path', None)
PROCESSED_LOG_FILE = os.path.join(OUTPUT_DIR, "processed_files.txt")

# ============================================================
# GLOBAL TRACKERS (For Quantity Filtering)
# ============================================================
# This keeps track of how many segments we've saved per species across all threads
segment_counts = {}
segment_lock = threading.Lock()

# ============================================================
# LOGGING & HELPERS
# ============================================================

def log_message(message, log_file):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{ts}] {message}\n")
    log_file.flush()
    print(message)

def get_processed_files():
    return set(open(PROCESSED_LOG_FILE).read().splitlines()) if os.path.exists(PROCESSED_LOG_FILE) else set()

def add_processed_file(path):
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(path + "\n")

def load_coordinates(file_path, log_file):
    audiomoth_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as coord_csvfile:
            reader = csv.DictReader(coord_csvfile)
            for row in reader:
                audiomoth_id = row.get('audiomoth_id')
                if not audiomoth_id: continue
                try:
                    lat = float(row.get('latitude', '')) if row.get('latitude') else None
                    lon = float(row.get('longitude', '')) if row.get('longitude') else None
                    audiomoth_data[audiomoth_id] = {
                        'name': row.get('human_readable_name', 'Unknown'),
                        'lat': lat, 'lon': lon, 'location': row.get('location', '')
                    }
                except ValueError:
                    audiomoth_data[audiomoth_id] = {'name': 'Unknown', 'lat': None, 'lon': None, 'location': ''}
    return audiomoth_data

def get_times(det):
    start = det.get('start') if det.get('start') is not None else det.get('start_time')
    end = det.get('end') if det.get('end') is not None else det.get('end_time')
    return start, end

# ============================================================
# UPDATED SEGMENT EXTRACTION (With Quantity Filter)
# ============================================================

def extract_audio_segment(input_file_path, detection, log_file):
    """Extracts segment ONLY if we haven't hit the max_segments limit for this species."""
    global segment_counts
    if not EXTRACT_SEGMENTS:
        return ""

    common_name = detection.get('common_name', 'unknown')
    
    # 1. CHECK THE QUANTITY FILTER (The "Gatekeeper")
    with segment_lock:
        current_count = segment_counts.get(common_name, 0)
        if current_count >= MAX_SEGMENTS_PER_SPECIES:
            return ""  # Skip extraction for this species
        segment_counts[common_name] = current_count + 1

    # 2. PROCEED WITH EXTRACTION
    try:
        audio = AudioSegment.from_wav(input_file_path)
        start, end = get_times(detection)
        
        # Ensure we use the length from config if start/end is different
        start_ms = int(start * 1000)
        end_ms = start_ms + int(SEG_LENGTH * 1000) 

        seg = audio[start_ms:end_ms]
        safe_name = common_name.replace(' ', '_').replace('/', '_')
        conf = detection.get('confidence', 0)
        fname = f"{safe_name}_{conf:.2f}_{os.path.basename(input_file_path)}"
        
        os.makedirs(SEGMENT_OUTPUT_DIR, exist_ok=True)
        seg.export(os.path.join(SEGMENT_OUTPUT_DIR, fname), format="wav")
        return fname
    except Exception as e:
        log_message(f"Segment export failed for {common_name}: {e}", log_file)
        return ""

# ============================================================
# ANALYZER & WORKERS
# ============================================================

_thread_local = threading.local()

def get_thread_analyzer():
    if not hasattr(_thread_local, "analyzer"):
        _thread_local.analyzer = Analyzer()
    return _thread_local.analyzer

def process_file(file_path, audiomoth_data, log_file):
    results = []
    try:
        analyzer = get_thread_analyzer()
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) < 3: return results

        device_id = parts[0]
        file_dt = datetime.strptime(parts[1] + " " + parts[2].split('.')[0], '%Y%m%d %H%M%S')
        data = audiomoth_data.get(device_id, {'name': 'Unknown', 'lat': None, 'lon': None, 'location': ''})

        rec = Recording(analyzer, file_path, lat=data['lat'], lon=data['lon'], date=file_dt, min_conf=MIN_CONFIDENCE)
        rec.analyze()

        for det in rec.detections:
            start, end = get_times(det)
            # This function now internally respects the MAX_SEGMENTS limit
            segment_filename = extract_audio_segment(file_path, det, log_file)
            
            results.append({
                'filename': filename, 'audiomoth_id': device_id, 'audiomoth_name': data['name'],
                'datetime': file_dt.isoformat(), 'latitude': data['lat'], 'longitude': data['lon'],
                'start_time': start, 'end_time': end, 'common_name': det.get('common_name'),
                'confidence': det.get('confidence'), 'segment_filename': segment_filename
            })
    except Exception as e:
        log_message(f"Error processing {file_path}: {e}", log_file)
    finally:
        add_processed_file(file_path)
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================

def analyze_audio_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT_DIR, f"birdnet_results_{ts}.csv")
    log_file_path = os.path.join(OUTPUT_DIR, f"analysis_{ts}.log")

    with open(log_file_path, 'w') as log_file:
        log_message("Starting BirdNET analysis with Quantity Filtering.", log_file)
        audiomoth_data = load_coordinates(COORDINATES_FILE, log_file)
        processed = get_processed_files()
        
        wavs = [os.path.join(dp, f) for dp, _, fns in os.walk(INPUT_DIR) for f in fns 
                if f.lower().endswith(".wav") and os.path.join(dp, f) not in processed]
        wavs.sort()

        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fields = ['filename', 'audiomoth_id', 'audiomoth_name', 'datetime', 'latitude', 'longitude', 
                      'start_time', 'end_time', 'common_name', 'confidence', 'segment_filename']
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                futures = {ex.submit(process_file, fp, audiomoth_data, log_file): fp for fp in wavs}
                for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    rows = f.result()
                    if rows:
                        writer.writerows(rows)
                        csvfile.flush()

    print(f"Analysis complete. Results in {out_csv}")

if __name__ == "__main__":
    analyze_audio_files()