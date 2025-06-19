import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_rfm_clusters(rfm_clustered: pd.DataFrame):
    features = ['Recency', 'Frequency', 'Monetary']
    rfm_copy = rfm_clustered.copy()

    # PCA projection
    pca = PCA(n_components=2)
    components = pca.fit_transform(rfm_copy[features])
    rfm_copy['PCA1'] = components[:, 0]
    rfm_copy['PCA2'] = components[:, 1]

    fig = px.scatter(
        rfm_copy,
        x='PCA1',
        y='PCA2',
        color='Segment',
        hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
        title="📈 Customer Segments (PCA Projection)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_profiles(rfm_clustered: pd.DataFrame):
    """
    Plots a log-scaled heatmap showing average RFM values per customer segment.
    """
    profile = rfm_clustered.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Apply log transformation
    profile_log = profile.applymap(lambda x: np.log1p(x)).round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=profile_log.values,
            x=profile_log.columns.tolist(),
            y=profile_log.index.tolist(),
            colorscale='YlGnBu',
            colorbar=dict(title="Log Value"),
            hovertemplate='Segment: %{y}<br>Metric: %{x}<br>Log Value: %{z}<extra></extra>'
        )
    )

    fig.update_layout(
        title="📉 Log-Scaled Average RFM Values by Segment",
        xaxis_title="RFM Feature",
        yaxis_title="Segment",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)