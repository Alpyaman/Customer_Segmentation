# clustering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_kmeans_clustering(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    
    rfm_copy = rfm.copy()

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_copy[['Recency', 'Frequency', 'Monetary']])

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_copy['Cluster'] = kmeans.fit_predict(scaled_features)

    return rfm_copy
