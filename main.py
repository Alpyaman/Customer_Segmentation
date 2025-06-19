from preprocess import load_and_clean_data
from rfm import generate_rfm_table
from clustering import perform_kmeans_clustering
from visualize import plot_rfm_clusters, plot_cluster_profiles

def main():
    # Step 1: Load and clean data
    print("📦 Loading and cleaning data...")
    df = load_and_clean_data("data/data.csv")

    # Step 2: Generate RFM features
    print("📊 Generating RFM table...")
    rfm = generate_rfm_table(df)

    # Step 3: Apply K-Means clustering
    print("🔢 Performing clustering...")
    rfm_clustered = perform_kmeans_clustering(rfm, n_clusters=4)

    # Step 4: Show cluster summary
    print("📈 Cluster summary (mean RFM values):")
    print(rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1))
    print("🧮 Customer count per cluster:")
    print(rfm_clustered["Cluster"].value_counts())

    # Step 5: Visualizations
    print("🎨 Plotting PCA and cluster profiles...")
    plot_rfm_clusters(rfm_clustered)
    plot_cluster_profiles(rfm_clustered)

if __name__ == "__main__":
    main()
