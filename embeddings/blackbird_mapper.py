import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
INPUT_FILE = "/Volumes/WD_BLACK/audioacoustics/results/June/embeddings_output/master_dataset.parquet"
# Note: Using the name exactly as it appears in your filenames
SPECIES_NAME = "Eurasian_Blackbird" 
OUTPUT_PATH = f"/Volumes/WD_BLACK/audioacoustics/results/June/Blackbird_Location_Map.png"

def run_blackbird_location_analysis():
    print(f"📂 Loading master dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Flexible Filtering
    # We check both 'common_name' and the filename itself to be safe
    spec_df = df[df['segment_filename'].str.contains(SPECIES_NAME)].copy()
    
    if len(spec_df) < 10:
        print(f"❌ Error: Found {len(spec_df)} samples for {SPECIES_NAME}.")
        return

    # 2. Extract Metadata from your specific filename format
    # Eurasian_Blackbird_0.43_24F3190863FAD0D4_20250619_202500.wav
    print("🧬 Extracting Device IDs and Timestamps...")
    
    def parse_filename(name):
        parts = name.replace('.wav', '').split('_')
        # Based on your example: parts[3] is Device ID, parts[4] is Date
        return pd.Series({
            'device_id': parts[3] if len(parts) > 3 else 'Unknown',
            'date_str': parts[4] if len(parts) > 4 else 'Unknown'
        })

    spec_df[['device_id', 'date_str']] = spec_df['segment_filename'].apply(parse_filename)

    # 3. Prepare Features
    embeddings = np.stack(spec_df['embedding'].values)
    
    # 4. UMAP Projection (Tuned for separation)
    print(f"🧠 Mapping {len(spec_df)} samples...")
    reducer = umap.UMAP(
        n_neighbors=20, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    spec_df['x'], spec_df['y'] = coords[:, 0], coords[:, 1]

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # Left: Colored by Device ID (The Location separation)
    sns.scatterplot(
        data=spec_df, x='x', y='y', 
        hue='device_id', palette='Set2', 
        ax=ax1, s=70, alpha=0.8, edgecolor='black', linewidth=0.3
    )
    ax1.set_title(f"Acoustic Clusters by Device (Location)", fontsize=15)
    ax1.legend(title="Device ID", bbox_to_anchor=(1, 1))

    # Right: Colored by Date (Seasonal drift)
    sns.scatterplot(
        data=spec_df, x='x', y='y', 
        hue='date_str', palette='viridis', 
        ax=ax2, s=70, alpha=0.8
    )
    ax2.set_title("Acoustic Clusters by Date", fontsize=15)

    plt.suptitle(f"Eurasian Blackbird: Geographic & Temporal Acoustic Map", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()
    print(f"✨ Map created: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_blackbird_location_analysis()