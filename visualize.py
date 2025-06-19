import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_rfm_clusters(rfm_clustered: pd.DataFrame):

    features = ['Recency', 'Frequency', 'Monetary']
    rfm_copy = rfm_clustered.copy()

    # Apply PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(rfm_copy[features])
    rfm_copy['PCA1'] = components[:, 0]
    rfm_copy['PCA2'] = components[:, 1]

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm_copy, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80, alpha=0.8)
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

def plot_cluster_profiles(rfm_clustered: pd.DataFrame):
    """
    Plots a heatmap showing average RFM values for each cluster.

    Args:
        rfm_clustered (pd.DataFrame): DataFrame with RFM columns and 'Cluster'.
    """
    profile = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    sns.heatmap(profile, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Cluster Profile Heatmap")
    plt.show()
