
# 🧠 Customer Segmentation with Clustering (RFM Analysis)

## 📌 Objective
Segment customers based on their purchasing behavior using **RFM analysis** and **Clustering Algorithms** to help businesses personalize marketing strategies.

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
- Created `TotalSum = Quantity × UnitPrice`

### 2. RFM Feature Engineering (`rfm.py`)
- **Recency**: Days since last transaction
- **Frequency**: Number of unique purchases
- **Monetary**: Total money spent

### 3. Clustering Algorithms (`clustering.py`)
- ✅ KMeans
- ✅ DBSCAN (with noise point detection)
- ✅ Agglomerative Clustering

### 4. Visualization (`visualize.py`)
- Interactive PCA cluster plot
- Log-scaled heatmap of average RFM values

### 5. Streamlit App (`app.py`)
- Upload, segment, visualize, and export customer groups

---

## 🚀 Live App

🔗 **[Try the Live App on Streamlit Cloud](https://customer-segmentation-dashboard.streamlit.app)**  
_(You’ll need to deploy this yourself — see instructions below)_

---

## ▶️ How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit dashboard
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this project to a public GitHub repo
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **"New App"** → Select your repo and `app.py`
4. Add a **Secrets File** with any API keys if needed (not used in this project)
5. Click **Deploy**

---

## 📈 Sample Outputs

### PCA Cluster Projection
![PCA Projection](Customer_Segments(PCA_Projection).png)

### Log-Scaled Heatmap
![Cluster Heatmap](Cluster_profile_heatmap.png)

---

## 👨‍💻 Author
Alp Yaman
