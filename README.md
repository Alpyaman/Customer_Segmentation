
# 🧠 Customer Segmentation with Clustering (RFM Analysis)

## 📌 Objective
Segment customers based on their purchasing behavior using **RFM analysis** and **K-Means clustering** to help businesses personalize marketing strategies.

---

## 📁 Dataset
- **Source**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Description**: ~500K transaction records from a UK-based e-commerce store between 2010–2011.

---

## 🧹 Workflow

### 1. Data Preprocessing (`preprocess.py`)
- Removed missing CustomerIDs
- Removed cancelled orders
- Filtered out non-positive `Quantity` or `UnitPrice`
- Created `TotalPrice = Quantity × UnitPrice`

### 2. RFM Feature Engineering (`rfm.py`)
- **Recency**: Days since last transaction
- **Frequency**: Number of unique purchases
- **Monetary**: Total money spent

### 3. Clustering (`clustering.py`)
- Standardized RFM features
- Applied **K-Means** with `k=4`
- Used **Elbow method** and **Silhouette Score** to validate `k`

### 4. Visualization (`visualize.py`)
- PCA projection for 2D cluster visualization
- Heatmap of average RFM values by cluster

### 5. Pipeline Runner (`main.py`)
- Runs everything end-to-end from raw data to cluster insights

---

## 📈 Visualizations

### Cluster Profile Heatmap
![Cluster Profile Heatmap](Cluster_profile_heatmap.png)

### PCA-Based Cluster Projection
![PCA Projection](Customer_Segments(PCA_Projection).png)

---

## 🧩 Cluster Summary

| Cluster | Description         | Recency | Frequency | Monetary | Count |
|---------|---------------------|---------|-----------|----------|--------|
| **2**   | 🏆 Champions / VIPs | 4.7     | 2566.0    | £126,118 | 13     |
| **0**   | 📈 Loyal Customers  | 20.9    | 135.5     | £2,648   | 2,165  |
| **1**   | 😐 At Risk          | 97.8    | 37.9      | £778     | 1,331  |
| **3**   | ⚠️ Lost / Inactive  | 272.2   | 25.1      | £605     | 830    |

---

## 🚀 Technologies Used
- Python, Pandas, NumPy
- Scikit-learn (KMeans, PCA)
- Matplotlib, Seaborn

---

## 💡 Future Improvements
- Add interactive Streamlit dashboard
- Try other clustering algorithms (DBSCAN, Agglomerative)
- Perform RFM scoring (quartile ranking) before clustering

---

## ▶️ Run the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py
```

---

## 👨‍💻 Author
Alp Yaman
