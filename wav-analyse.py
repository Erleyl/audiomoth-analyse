import os
import csv
import sys
import yaml
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# --- Configuration Section ---
def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML configuration file: {e}")
        sys.exit(1)

# Get the directory of the current script and load the config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes 'config' folder is in the same directory as the script.
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config', 'settings.yaml')
CONFIG = load_config(CONFIG_PATH)

INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['input_dir']))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['output_dir']))
COORDINATES_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, CONFIG['coordinates_file']))
MIN_CONFIDENCE = CONFIG.get('min_confidence', 0.1)

# Path to the file that tracks already processed files
PROCESSED_LOG_FILE = os.path.join(OUTPUT_DIR, "processed_files.txt")

# --- Functions ---
def log_message(message, log_file):
    """Writes a timestamped message to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {message}\n")
    print(message)

def load_coordinates(file_path, log_file):
    """
    Loads device coordinates, human-readable names, and location data from a CSV file.
    """
    audiomoth_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as coord_csvfile:
            reader = csv.DictReader(coord_csvfile)
            for row in reader:
                audiomoth_id = row.get('audiomoth_id')
                if not audiomoth_id:
                    log_message(f"Skipping row with missing audiomoth_id: {row}", log_file)
                    continue
                try:
                    lat_str = row.get('latitude', '').strip()
                    lon_str = row.get('longitude', '').strip()

                    lat = float(lat_str) if lat_str else None
                    lon = float(lon_str) if lon_str else None

                    audiomoth_data[audiomoth_id] = {
                        'name': row.get('human_readable_name', 'Unknown'),
                        'lat': lat,
                        'lon': lon,
                        'location': row.get('location', '')
                    }
                except (ValueError, KeyError) as e:
                    log_message(f"Error reading coordinates for '{audiomoth_id}': {e}. Check for non-numeric values.", log_file)
                    audiomoth_data[audiomoth_id] = {'name': 'Unknown', 'lat': None, 'lon': None, 'location': ''}
    else:
        log_message(f"Warning: Coordinates file '{file_path}' not found. Using default coordinates.", log_file)
    return audiomoth_data

def get_processed_files():
    """Reads the list of processed files from the tracking log."""
    processed = set()
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, 'r') as f:
            for line in f:
                processed.add(line.strip())
    return processed

def add_processed_file(file_path):
    """Adds a file path to the processed files log."""
    with open(PROCESSED_LOG_FILE, 'a') as f:
        f.write(f"{file_path}\n")

# --- Main Analysis Logic ---
def analyze_audio_files():
    """
    Walks through the directory, analyzes audio files, and saves detections to a CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_file = os.path.join(OUTPUT_DIR, f"birdnet_results_{timestamp}.csv")
    log_file_name = os.path.join(OUTPUT_DIR, f"birdnet_analysis_{timestamp}.log")
    
    with open(log_file_name, 'w') as log_file:
        log_message("Starting BirdNET analysis.", log_file)
        
        try:
            analyzer = Analyzer()
            log_message("BirdNET analyzer model loaded successfully.", log_file)
        except Exception as e:
            log_message(f"Error loading BirdNET analyzer: {e}", log_file)
            return

        audiomoth_data = load_coordinates(COORDINATES_FILE, log_file)
        if not audiomoth_data:
            log_message("No audiomoth coordinate data loaded.", log_file)

        processed_files = get_processed_files()

        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'audiomoth_id', 'audiomoth_name', 'datetime', 'latitude', 'longitude', 'location', 'start_time', 'end_time', 'common_name', 'scientific_name', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            log_message("Output CSV file created with headers.", log_file)

            all_wav_files = []
            for dirpath, _, filenames in os.walk(INPUT_DIR):
                for filename in filenames:
                    if filename.endswith(".WAV") and not filename.startswith('.'):
                        file_path = os.path.join(dirpath, filename)
                        all_wav_files.append(file_path)
            
            all_wav_files.sort()
            
            for file_path in all_wav_files:
                if file_path in processed_files:
                    log_message(f"Skipping {file_path}: already processed.", log_file)
                    continue

                filename = os.path.basename(file_path)
                log_message(f"Processing {file_path}", log_file)

                try:
                    parts = filename.split('_')
                    if len(parts) < 3:
                        log_message(f"Skipping {filename}: filename format is incorrect.", log_file)
                        add_processed_file(file_path)
                        continue
                    
                    device_id = parts[0]
                    datetime_str = parts[1] + ' ' + parts[2].split('.')[0]
                    file_datetime = datetime.strptime(datetime_str, '%Y%m%d %H%M%S')
                    data = audiomoth_data.get(device_id, {'name': 'Unknown', 'lat': None, 'lon': None, 'location': ''})
                    
                    recording = Recording(
                        analyzer,
                        file_path,
                        lat=data['lat'],  
                        lon=data['lon'],
                        date=file_datetime,
                        min_conf=MIN_CONFIDENCE
                    )
                    recording.analyze()
                    
                    if recording.detections:
                        for detection in recording.detections:
                            writer.writerow({
                                'filename': filename,
                                'audiomoth_id': device_id,
                                'audiomoth_name': data['name'],
                                'datetime': file_datetime.isoformat(),
                                'latitude': data['lat'],
                                'longitude': data['lon'],
                                'location': data['location'],
                                'start_time': detection.get('start'),
                                'end_time': detection.get('end'),
                                'common_name': detection.get('common_name'),
                                'scientific_name': detection.get('scientific_name'),
                                'confidence': detection.get('confidence'),
                            })
                    else:
                        log_message(f"No detections found for {filename}.", log_file)
                except Exception as e:
                    log_message(f"Error processing {filename}: {e}", log_file)
                
                # Add file to the processed log regardless of success or failure
                add_processed_file(file_path)

        log_message(f"Analysis complete. Results are in {output_csv_file}", log_file)
        
if __name__ == "__main__":
    analyze_audio_files()

