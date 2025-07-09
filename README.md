# 🌍 Tourism Services – Country Segmentation Using Clustering

## 📌 Problem Statement

### 🧭 Context

Tourism is now recognized as a measurable economic activity, with reliable indicators available to guide policy decisions and business investments. A tours and travel company seeks to understand which countries offer the most potential for tourism services by segmenting them based on key socioeconomic, environmental, and infrastructure indicators.

### 🎯 Objective

The goal of this project is to perform **unsupervised learning (clustering)** to group countries based on multiple factors such as GDP per capita, life expectancy, carbon emissions, internet usage, and more. These insights will help the company prioritize regions for strategic tourism development.

---

## 🧾 Dataset Description

- **File**: `country_stats.csv`
- **Observations**: Countries worldwide
- **Features**:
  - `GDP per Capita`
  - `Life Expectancy`
  - `Carbon Emissions`
  - `Population`
  - `Internet Usage`
  - `Tourist Arrivals` (and other indicators)

---

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (StandardScaler, KMeans, PCA)
- Plotly (optional for interactive visuals)

---

## 📊 Project Workflow

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 2. Load Dataset
df = pd.read_csv('country_stats.csv')

# 3. EDA & Cleaning
df.info()
df.describe()
df.isnull().sum()
df.fillna(df.mean(), inplace=True)

# 4. Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# 5. Elbow Method to Find Optimal Clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# 6. Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

# 7. Visualize Clusters Using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2')
plt.title('Country Clusters Based on Tourism Indicators')
plt.show()
```

---

## 🔍 Key Insights

- **Cluster 0**: High-income, internet-penetrated countries with high tourist arrivals
- **Cluster 1**: Low GDP and life expectancy, low tourism activity
- **Cluster 2**: Developing nations with mid-range tourism potential
- **Cluster 3**: Resource-rich but underdeveloped tourism infrastructure

---

## 📈 Recommendations

- Focus marketing and infrastructure efforts on **Cluster 2** for maximum ROI
- Explore public-private partnerships in **Cluster 3** to develop tourism
- Sustain existing services in **Cluster 0** with premium packages

---

## 📁 Repository Structure

```
├── USL_MLS2_TourismServices.ipynb   # Main notebook
├── country_stats.csv                # Dataset
├── README.md                        # Documentation
```

---

## 👨‍💻 Author

**Suhaiub Khalid**  
AI & ML Practitioner

