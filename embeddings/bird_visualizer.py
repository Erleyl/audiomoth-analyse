import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
INPUT_FILE = "/Volumes/WD_BLACK/audioacoustics/results/June/embeddings_output/master_dataset.parquet"
OUTPUT_PLOT = "/Volumes/WD_BLACK/audioacoustics/results/June/bird_acoustic_map.png"

def run_visualization():
    print(f"📂 Loading 71k fingerprints from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # 1. Prepare the data
    # UMAP needs a 2D numpy array: (rows, 1024)
    print("🧠 Preparing feature matrix...")
    embeddings = np.stack(df['embedding'].values)
    
    # 2. Run UMAP
    # n_neighbors: low = local detail, high = global structure
    # min_dist: how tightly points pack together
    print("🗺️ Running UMAP projection (this may take 10-15 minutes)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine', # Cosine is better for high-dimensional fingerprints
        random_state=42
    )
    
    projected_coords = reducer.fit_transform(embeddings)
    
    # 3. Add coordinates back to dataframe
    df['x'] = projected_coords[:, 0]
    df['y'] = projected_coords[:, 1]

    # 4. Create the Plot
    print(f"🎨 Generating map and saving to {OUTPUT_PLOT}...")
    plt.figure(figsize=(16, 10))
    
    # We color by 'common_name' to see if the species cluster together
    # For 71k points, we use a small point size (s=1) and alpha for transparency
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='common_name',
        palette='viridis',
        s=2,
        alpha=0.5,
        edgecolor=None
    )
    
    plt.title("Acoustic Landscape: June BirdNET Embeddings", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✨ SUCCESS! Your acoustic map is ready at: {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_visualization()