# Social Bot Detection using GCN and Behavioural Metadata

## 🧠 Overview
This project focuses on detecting bot accounts in Twitter data using Graph Neural Networks. We use the **Cresci-RTbust-2019** dataset and apply a **Graph Convolutional Network (GCN)** to classify users as either bots or humans. 

The graph structure is created using a **k-Nearest Neighbors (k-NN)** approach on user metadata features (like followers count, tweet activity, verification status, etc.). The model learns patterns in both node features and their connectivity to predict labels.

---

## 🧾 What the Notebook Covers
All steps are included in a single notebook/script:

### ✅ Data Loading & Cleaning
- Load `merged_data.csv`, `feature_data.csv`, and labels (`cresci-rtbust-2019.tsv`)
- Select and preprocess numeric features (e.g., `followers_count`, `verified`, `tweet_per_follower`)
- Convert labels (0 = human, 1 = bot)

### 🔗 Graph Construction
- Apply **k-NN** using sklearn’s `kneighbors_graph` on feature vectors
- Generate `edge_index` (in COO format) for PyTorch Geometric

### 🧠 GCN Model
- Define a 2-layer Graph Convolutional Network using PyTorch Geometric
- Use ReLU activation, Dropout, and Adam optimizer

### 📊 Training & Evaluation
- Train the GCN on the full graph
- Evaluate using loss and accuracy
- Print results and optionally visualize predictions

---

## 🔧 Technologies Used
- Python
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Pandas, NumPy

---

## 📌 Key Highlights
- Created a graph from tabular data using k-NN (no explicit Twitter network needed)
- Demonstrated node classification with GCN on real-world bot detection
- Easy to extend with alternative edge strategies (cosine similarity, temporal coordination, etc.)

---

## 🚀 How to Run
1. Clone the repository or copy the code locally
2. Place the Cresci dataset CSV/TSV files in the working directory
3. Install dependencies:
```bash
pip install torch torch-geometric scikit-learn pandas
