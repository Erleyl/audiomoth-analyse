import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
INPUT_FILE = "/Volumes/WD_BLACK/audioacoustics/results/June/embeddings_output/master_dataset.parquet"
# Matches the common_name in your BirdNET results
SPECIES_NAME = "Eurasian Blackbird" 
OUTPUT_PLOT = "/Volumes/WD_BLACK/audioacoustics/results/June/Blackbird_USUV_Baseline.png"
OUTPUT_CSV = "/Volumes/WD_BLACK/audioacoustics/results/June/Location_Diversity_Metrics.csv"

def get_diel_phase(hour):
    """Categorizes activity based on biological timeframes."""
    if 4 <= hour <= 8: return 'Dawn Chorus'
    if 9 <= hour <= 17: return 'Daytime'
    if 18 <= hour <= 21: return 'Dusk / Evening'
    return 'Night'

def calculate_repertoire_diversity(group):
    """
    Calculates Diversity as the average distance of all calls 
    to the 'mean sound' of that location in 1024-D space.
    """
    if len(group) < 5: 
        return pd.Series({'diversity_score': np.nan, 'sample_size': len(group)})
    
    # Convert series of embedding arrays into a matrix
    embeddings = np.stack(group['embedding'].values)
    centroid = embeddings.mean(axis=0)
    
    # Calculate Euclidean distances to the centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    return pd.Series({
        'diversity_score': np.mean(distances),
        'sample_size': len(group)
    })

def run_investigation():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found at {INPUT_FILE}")
        return

    print("📂 Loading master dataset from WD_BLACK...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Filter specifically for Blackbird
    # Using case-insensitive search to catch different naming variations
    spec_df = df[df['common_name'].str.contains("Blackbird", case=False)].copy()
    
    if spec_df.empty:
        print("❌ No Blackbird data found. Please check 'common_name' entries in your parquet.")
        return

    # 2. Extract Diel (Time) Information from your filename format
    # Format: 2453AC0263FA64B4_20250619_185500.wav
    print("🧬 Parsing timestamps for diel patterns...")
    spec_df['time_str'] = spec_df['segment_filename'].apply(lambda x: x.split('_')[-1].replace('.wav', ''))
    spec_df['hour'] = spec_df['time_str'].apply(lambda x: int(x[:2]))
    spec_df['diel_phase'] = spec_df['hour'].apply(get_diel_phase)

    # 3. Calculate Diversity Scores (Populates the right pane)
    print("📊 Calculating Acoustic Diversity per location...")
    diversity_stats = spec_df.groupby('location').apply(calculate_repertoire_diversity).reset_index()
    diversity_stats = diversity_stats.dropna().sort_values('diversity_score', ascending=False)
    diversity_stats.to_csv(OUTPUT_CSV, index=False)

    # 4. Generate UMAP Coordinates
    print(f"🧠 Processing {len(spec_df)} embeddings...")
    embeddings_matrix = np.stack(spec_df['embedding'].values)
    
    # metric='cosine' is best for identifying vocalization structure
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings_matrix)
    spec_df['x'], spec_df['y'] = coords[:, 0], coords[:, 1]

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [1.2, 0.8]})
    
    # Left Pane: Acoustic Manifold (Behavioral Clusters)
    sns.scatterplot(
        data=spec_df, x='x', y='y', 
        hue='diel_phase', hue_order=['Dawn Chorus', 'Daytime', 'Dusk / Evening', 'Night'],
        palette='Spectral', ax=ax1, s=65, alpha=0.7, edgecolor='black', linewidth=0.2
    )
    ax1.set_title(f"Acoustic Clusters: Behavioral Phases (n={len(spec_df)})", fontsize=16)

    # Right Pane: Repertoire Richness by Location
    sns.barplot(
        data=diversity_stats, x='diversity_score', y='location', 
        palette='magma', ax=ax2
    )
    
    # Add sample size labels to bars
    for i, row in enumerate(diversity_stats.itertuples()):
        ax2.text(row.diversity_score, i, f" n={int(row.sample_size)}", va='center', fontsize=11)
        
    ax2.set_title("Acoustic Diversity (Repertoire Richness)", fontsize=16)
    ax2.set_xlabel("Diversity Score (Higher = More Varied/Healthy Repertoire)")

    plt.suptitle(f"Eurasian Blackbird Baseline: {SPECIES_NAME} (June)", fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save results
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Baseline Plot saved to: {OUTPUT_PLOT}")
    print(f"✅ CSV data saved to: {OUTPUT_CSV}")
    
    plt.show()

if __name__ == "__main__":
    run_investigation()