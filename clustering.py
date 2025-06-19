import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def perform_kmeans_clustering(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    rfm_copy = rfm.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_copy[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_copy['Cluster'] = kmeans.fit_predict(scaled)
    return rfm_copy

def cluster_with_dbscan(rfm: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    rfm_copy = rfm.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_copy[['Recency', 'Frequency', 'Monetary']])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    rfm_copy['Cluster'] = dbscan.fit_predict(scaled)
    return rfm_copy

def cluster_with_agglomerative(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    rfm_copy = rfm.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_copy[['Recency', 'Frequency', 'Monetary']])
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    rfm_copy['Cluster'] = agg.fit_predict(scaled)
    return rfm_copy
