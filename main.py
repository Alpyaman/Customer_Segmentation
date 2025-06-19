from preprocess import load_and_clean_data
from rfm import generate_rfm_table
from clustering import (
    perform_kmeans_clustering,
    cluster_with_dbscan,
    cluster_with_agglomerative
)
from visualize import plot_rfm_clusters, plot_cluster_profiles

def main():
    print("📦 Loading and cleaning data...")
    df = load_and_clean_data("data/data.csv")

    print("📊 Generating RFM table...")
    rfm = generate_rfm_table(df)

    # Choose clustering method
    print("\nSelect Clustering Method:")
    print("1 - KMeans")
    print("2 - DBSCAN")
    print("3 - Agglomerative Clustering")
    choice = input("Enter 1, 2 or 3: ")

    if choice == "1":
        print("🔢 Applying KMeans clustering...")
        rfm_clustered = perform_kmeans_clustering(rfm, n_clusters=4)
    elif choice == "2":
        print("🔍 Applying DBSCAN clustering...")
        rfm_clustered = cluster_with_dbscan(rfm, eps=1.2, min_samples=10)
    elif choice == "3":
        print("🧱 Applying Agglomerative clustering...")
        rfm_clustered = cluster_with_agglomerative(rfm, n_clusters=4)
    else:
        print("Invalid choice. Defaulting to KMeans.")
        rfm_clustered = perform_kmeans_clustering(rfm, n_clusters=4)

    # Show cluster stats
    print("\n📈 Cluster summary (RFM means):")
    print(rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1))

    print("\n🧮 Customer count per cluster:")
    print(rfm_clustered["Cluster"].value_counts())

    # Visualizations
    print("🎨 Visualizing clusters...")
    plot_rfm_clusters(rfm_clustered)
    plot_cluster_profiles(rfm_clustered)

if __name__ == "__main__":
    main()
