import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- CONFIG ---
MODEL_PATH   = "/Volumes/WD_BLACK/audioacoustics/resources/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
BASE_DIR     = "/Volumes/WD_BLACK/audioacoustics/results/June"
CSV_INPUT    = os.path.join(BASE_DIR, "birdnet_results_20260127_160603.csv")
SEGMENTS_DIR = os.path.join(BASE_DIR, "segments")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "embeddings_output")
OUTPUT_FILE   = os.path.join(OUTPUT_FOLDER, "master_dataset.parquet")

def process_worker(args):
    idx, full_path, in_idx, out_idx = args
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        # Load 3s audio at 48kHz
        sig, _ = librosa.load(full_path, sr=48000, mono=True, duration=3.0)
        
        input_data = np.zeros((1, 144000), dtype=np.float32)
        actual_len = min(len(sig), 144000)
        input_data[0, :actual_len] = sig[:actual_len]
        
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        
        # --- THE FIX ---
        # We grab the tensor and FLATTEN it. 
        # If it was 144,000, the shape would be (144000,)
        # If it is the correct embedding, the shape will be (1024,)
        embedding = interpreter.get_tensor(out_idx).flatten()
        
        return {"index": idx, "embedding": embedding.astype(np.float32)}
    except:
        return None

def run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Identify the 1024 layer once at the start
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]['index']
    
    # We look for ANY output layer that results in 1024 values
    target_out_idx = None
    for out in interp.get_output_details():
        # np.prod calculates the total size (e.g., 1*1*1*1024 = 1024)
        if np.prod(out['shape']) == 1024:
            target_out_idx = out['index']
            print(f"🎯 Target Layer Identified: Index {target_out_idx}")
            break

    if target_out_idx is None:
        # Fallback for V2.4 models where shape is sometimes reported as empty/dynamic
        print("⚠️ Automatic detection failed, using standard Index 1 for V2.4...")
        target_out_idx = interp.get_output_details()[1]['index']

    # 2. File matching
    disk_map = {entry.name: entry.path for entry in os.scandir(SEGMENTS_DIR) 
                if entry.name.lower().endswith('.wav') and not entry.name.startswith('._')}
    
    df = pd.read_csv(CSV_INPUT)
    active_df = df[df['segment_filename'].isin(disk_map.keys())].copy()

    # 3. Execution
    tasks = [(idx, disk_map[row['segment_filename']], in_idx, target_out_idx) 
             for idx, row in active_df.iterrows()]
    
    print(f"🚀 Processing {len(tasks)} files...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_worker, tasks), total=len(tasks)))

    # 4. Save
    valid = [r for r in results if r is not None]
    if valid:
        # Verify the first result is actually 1024 before saving
        first_len = len(valid[0]['embedding'])
        print(f"📊 Verified Embedding Size: {first_len}")
        
        if first_len != 1024:
            print(f"❌ ERROR: Still getting {first_len} instead of 1024. Layer index is wrong.")
            return

        emb_df = pd.DataFrame(valid).set_index('index')
        final_df = active_df.join(emb_df, how='inner')
        final_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
        print(f"✨ SUCCESS: {OUTPUT_FILE}")
    else:
        print("❌ No valid results generated.")

if __name__ == "__main__":
    run()