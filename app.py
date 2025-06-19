# app.py

import streamlit as st
import pandas as pd
from preprocess import load_and_clean_data
from rfm import generate_rfm_table
from clustering import (
    perform_kmeans_clustering,
    cluster_with_dbscan,
    cluster_with_agglomerative
)
from visualize import plot_rfm_clusters, plot_cluster_profiles

def assign_segment_names(rfm_df: pd.DataFrame) -> pd.DataFrame:
    profile = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # Skip DBSCAN noise (-1)
    clusters = [c for c in profile.index if c != -1]

    sorted_clusters = profile.loc[clusters].sort_values(
        by=["Monetary", "Frequency", "Recency"],
        ascending=[False, False, True]
    ).index.tolist()

    segment_names = ["VIP", "Loyal", "At Risk", "Lost"]
    cluster_map = {}

    for i, cluster_id in enumerate(sorted_clusters):
        name = segment_names[i] if i < len(segment_names) else f"Segment {i+1}"
        cluster_map[cluster_id] = name

    if -1 in profile.index:
        cluster_map[-1] = "Noise"

    rfm_df["Segment"] = rfm_df["Cluster"].map(cluster_map)
    return rfm_df



st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation Dashboard")
st.markdown("Segment customers using **RFM** and **Clustering Algorithms**.")

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Clustering Settings")
    method = st.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])
    
    if method == "KMeans" or method == "Agglomerative Clustering":
        k = st.slider("Number of Clusters", 2, 10, 4)
    if method == "DBSCAN":
        eps = st.slider("DBSCAN EPS", 0.1, 5.0, 1.2, 0.1)
        min_samples = st.slider("Min Samples", 1, 20, 10)

    st.markdown("---")
    export = st.checkbox("📤 Enable CSV Export")

# --- Load & Prepare ---
df = load_and_clean_data("data/data.csv")
rfm = generate_rfm_table(df)

# --- Clustering ---
if method == "KMeans":
    rfm_clustered = perform_kmeans_clustering(rfm, n_clusters=k)
elif method == "DBSCAN":
    rfm_clustered = cluster_with_dbscan(rfm, eps=eps, min_samples=min_samples)
else:
    rfm_clustered = cluster_with_agglomerative(rfm, n_clusters=k)

# --- Assign Segment Labels
rfm_clustered = assign_segment_names(rfm_clustered)

# --- Display Segment Summary
st.subheader("📊 Cluster Summary")
summary = (
    rfm_clustered
    .groupby(["Cluster", "Segment"])[["Recency", "Frequency", "Monetary"]]
    .mean()
    .round(1)
    .reset_index()
)

# Add count per segment
count_df = rfm_clustered.groupby(["Cluster", "Segment"]).size().reset_index(name="Count")
summary = pd.merge(summary, count_df, on=["Cluster", "Segment"])
st.dataframe(summary)

# --- Visualizations
st.subheader("📈 PCA Visualization")
plot_rfm_clusters(rfm_clustered)

st.subheader("📉 Cluster Heatmap")
plot_cluster_profiles(rfm_clustered)

# --- Export CSV
if export:
    st.download_button(
        label="Download Clustered Customers CSV",
        data=rfm_clustered.to_csv(index=False).encode(),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )
