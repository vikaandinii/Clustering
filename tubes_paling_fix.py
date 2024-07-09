import os
os.environ["OMP_NUM_THREADS"] = "1"  # Menetapkan variabel lingkungan untuk menghindari peringatan kebocoran memori

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Memuat dataset
df = pd.read_csv(r'C:\clustering\Mall_Customers.csv')

# Mengeksplorasi data
print(df.head())
print(df.info())

# Statistik deskriptif
print(df.describe())

# Distribusi data
plt.figure(figsize=(12, 6))
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Distribusi Pendapatan Tahunan')
plt.xlabel('Pendapatan Tahunan (k$)')
plt.ylabel('Frekuensi')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title('Distribusi Skor Pengeluaran')
plt.xlabel('Skor Pengeluaran (1-100)')
plt.ylabel('Frekuensi')
plt.grid(True)
plt.show()

# Praproses data (memilih fitur yang relevan dan melakukan skala)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster optimal menggunakan Metode Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Menampilkan grafik Metode Elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Jumlah Cluster', fontsize=14)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=14)
plt.title('Metode Elbow', fontsize=16)
plt.grid(True)
plt.show()

# Evaluasi menggunakan Silhouette Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Jumlah Cluster', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.title('Silhouette Score untuk Berbagai Jumlah Cluster', fontsize=16)
plt.grid(True)
plt.show()

# Berdasarkan Metode Elbow dan Silhouette Score, kita pilih jumlah cluster optimal (misalnya, 5)
jumlah_cluster_optimal = 5

# Menerapkan clustering K-Means dengan n_init yang ditentukan untuk menghindari FutureWarning
kmeans = KMeans(n_clusters=jumlah_cluster_optimal, random_state=0, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Menambahkan label cluster ke dataframe asli
df['Cluster'] = labels

# Memvisualisasikan hasil clustering
plt.figure(figsize=(10, 6))

# Plot scatter dari data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', edgecolors='grey', alpha=0.8, s=100, label='Data Points')

# Plotting pusat cluster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.8, marker='X', label='Centroids')

# Label dan judul
plt.xlabel('Pendapatan Tahunan (terstandarisasi)')
plt.ylabel('Skor Pengeluaran (terstandarisasi)')
plt.title('Segmentasi Pelanggan Berdasarkan Pendapatan Tahunan dan Skor Pengeluaran')
plt.legend()

plt.grid(True)
plt.show()

# Menampilkan rincian setiap cluster
print(df.groupby('Cluster').mean(numeric_only=True))

# Analisis Cluster
for i in range(jumlah_cluster_optimal):
    print(f"\nCluster {i+1} Rincian:")
    print(df[df['Cluster'] == i].describe())

# Visualisasi distribusi usia pelanggan dalam setiap cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Age', data=df)
plt.title('Distribusi Usia dalam Setiap Cluster')
plt.xlabel('Cluster')
plt.ylabel('Usia')
plt.grid(True)
plt.show()

# Visualisasi distribusi gender dalam setiap cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', hue='Gender', data=df)
plt.title('Distribusi Gender dalam Setiap Cluster')
plt.xlabel('Cluster')
plt.ylabel('Jumlah')
plt.grid(True)
plt.show()

# Visualisasi cluster dalam 3D berdasarkan Usia, Pendapatan Tahunan, dan Skor Pengeluaran
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels, cmap='viridis', s=100)
ax.set_xlabel('Usia')
ax.set_ylabel('Pendapatan Tahunan (k$)')
ax.set_zlabel('Skor Pengeluaran (1-100)')
plt.title('Cluster Pelanggan dalam 3D')
plt.colorbar(sc)
plt.show()
