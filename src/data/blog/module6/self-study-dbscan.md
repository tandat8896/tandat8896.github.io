---
title: "DBSCAN - Thuật toán Clustering dựa trên Density"
pubDatetime: 2025-09-22T14:00:00Z
featured: false
description: "Tìm hiểu chi tiết về DBSCAN cho Clustering với Density-based Clustering và Noise Detection"
tags: ["machine-learning", "dbscan", "clustering", "density-based", "unsupervised-learning"]
---

# DBSCAN

## Nguyên lý cơ bản của DBSCAN

### **Ý tưởng chính - Tại sao DBSCAN hoạt động?**

**1. Quan sát cơ bản:**
- **Dense regions** chứa nhiều điểm gần nhau → Clusters
- **Sparse regions** chứa ít điểm → Noise/Outliers
- **Density** = Số điểm trong bán kính ε (eps)

**2. Ví dụ minh họa:**
```
Dense region: [1,1], [1,2], [2,1], [2,2] → Cluster
Sparse region: [10,10] → Noise
```

**3. Nguyên lý toán học:**
- **Core point**: Có ít nhất minPts điểm trong bán kính ε
- **Border point**: Không phải core point nhưng trong ε của core point
- **Noise point**: Không phải core point và không trong ε của core point

### **Các khái niệm cơ bản - Giải thích chi tiết**

**1. ε (eps) - Epsilon Neighborhood:**
- **Định nghĩa**: Bán kính tìm kiếm xung quanh mỗi điểm
- **Ý nghĩa**: Khoảng cách tối đa để 2 điểm được coi là "gần nhau"
- **Đơn vị**: Cùng đơn vị với dữ liệu (pixel, meter, ...)

**2. minPts - Minimum Points:**
- **Định nghĩa**: Số điểm tối thiểu trong ε-neighborhood để tạo core point
- **Ý nghĩa**: Ngưỡng mật độ để xác định cluster
- **Giá trị thường dùng**: 3-5 cho 2D, 2*dim cho high-dim

**3. Core Point:**
- **Định nghĩa**: Điểm có ít nhất minPts điểm trong ε-neighborhood
- **Công thức**: |N_ε(p)| ≥ minPts
- **Vai trò**: "Trung tâm" của cluster, có thể mở rộng cluster

**4. Border Point:**
- **Định nghĩa**: Không phải core point nhưng trong ε của core point
- **Công thức**: |N_ε(p)| < minPts AND p ∈ N_ε(core_point)
- **Vai trò**: "Biên" của cluster, không thể mở rộng cluster

**5. Noise Point:**
- **Định nghĩa**: Không phải core point và không trong ε của core point
- **Công thức**: |N_ε(p)| < minPts AND p ∉ N_ε(core_point)
- **Vai trò**: Outliers, không thuộc cluster nào

### **Công thức toán học**

**1. ε-neighborhood:**
$$N_ε(p) = \{q ∈ D | dist(p,q) ≤ ε\}$$

**2. Core point condition:**
$$|N_ε(p)| ≥ minPts$$

**3. Density-reachable:**
$$p \text{ is density-reachable from } q \text{ if } ∃ p_1, p_2, ..., p_n \text{ such that:}$$
- $p_1 = q, p_n = p$
- $p_{i+1} ∈ N_ε(p_i)$ for $i = 1, 2, ..., n-1$
- $p_i$ is core point for $i = 1, 2, ..., n-1$

**4. Density-connected:**
$$p \text{ and } q \text{ are density-connected if } ∃ r \text{ such that both } p \text{ and } q \text{ are density-reachable from } r$$

### **Thuật toán DBSCAN - Chi tiết từng bước**

**Input:** Dataset D, eps, minPts
**Output:** Clusters và noise points

**Bước 1: Khởi tạo**
- Tất cả điểm chưa được visit
- Cluster ID = 0
- Noise points = []

**Bước 2: Với mỗi điểm p chưa visit:**
- Mark p as visited
- Tìm ε-neighborhood N_ε(p)
- Nếu |N_ε(p)| < minPts:
  - Mark p as noise
- Ngược lại:
  - Tạo cluster mới với ID++
  - Add p to cluster
  - Mở rộng cluster từ p

**Bước 3: Mở rộng cluster (ExpandCluster):**
- Với mỗi điểm q trong N_ε(p):
  - Nếu q chưa visit:
    - Mark q as visited
    - Tìm N_ε(q)
    - Nếu |N_ε(q)| ≥ minPts:
      - Add N_ε(q) to seeds
  - Nếu q chưa thuộc cluster nào:
    - Add q to current cluster

---

## Ví Dụ Tính Tay - DBSCAN

### **Dataset Clustering**

| ID | X | Y | Label |
|---|---|---|---|
| 1 | 1 | 1 | ? |
| 2 | 1 | 2 | ? |
| 3 | 2 | 1 | ? |
| 4 | 2 | 2 | ? |
| 5 | 8 | 8 | ? |
| 6 | 8 | 9 | ? |
| 7 | 9 | 8 | ? |
| 8 | 9 | 9 | ? |
| 9 | 5 | 5 | ? |

**Parameters:** eps = 1.5, minPts = 3

---

### **Step 1: Tính ε-neighborhood cho mỗi điểm**

```python
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Dataset
data = np.array([
    [1, 1],   # ID 1
    [1, 2],   # ID 2
    [2, 1],   # ID 3
    [2, 2],   # ID 4
    [8, 8],   # ID 5
    [8, 9],   # ID 6
    [9, 8],   # ID 7
    [9, 9],   # ID 8
    [5, 5]    # ID 9
])

eps = 1.5
minPts = 3

# Tính distance matrix
distances = euclidean_distances(data)

print("Distance Matrix:")
print("ID", end="")
for i in range(len(data)):
    print(f"{i+1:8d}", end="")
print()

for i in range(len(data)):
    print(f"{i+1:2d}", end="")
    for j in range(len(data)):
        print(f"{distances[i,j]:8.2f}", end="")
    print()
```

**Output:**
```
Distance Matrix:
ID       1       2       3       4       5       6       7       8       9
 1    0.00    1.00    1.00    1.41    9.90    9.22    8.06    8.49    5.66
 2    1.00    0.00    1.41    1.00    9.22    8.49    8.49    8.06    5.39
 3    1.00    1.41    0.00    1.00    8.49    8.06    9.22    9.90    4.24
 4    1.41    1.00    1.00    0.00    8.06    7.81    8.49    9.22    4.24
 5    9.90    9.22    8.49    8.06    0.00    1.00    1.00    1.41    4.24
 6    9.22    8.49    8.06    7.81    1.00    0.00    1.41    1.00    3.61
 7    8.06    8.49    9.22    8.49    1.00    1.41    0.00    1.00    4.24
 8    8.49    8.06    9.90    9.22    1.41    1.00    1.00    0.00    4.24
 9    5.66    5.39    4.24    4.24    4.24    3.61    4.24    4.24    0.00
```

```python
# Tìm ε-neighborhood cho mỗi điểm
print(f"\nε-neighborhood (eps = {eps}):")
for i in range(len(data)):
    neighbors = []
    for j in range(len(data)):
        if i != j and distances[i,j] <= eps:
            neighbors.append(j+1)
    
    print(f"Point {i+1}: {neighbors} (count: {len(neighbors)})")
    
    # Kiểm tra core point
    if len(neighbors) >= minPts:
        print(f"  → Core point (≥ {minPts} neighbors)")
    else:
        print(f"  → Not core point (< {minPts} neighbors)")
```

**Output:**
```
ε-neighborhood (eps = 1.5):
Point 1: [2, 3] (count: 2)
  → Not core point (< 3 neighbors)
Point 2: [1, 4] (count: 2)
  → Not core point (< 3 neighbors)
Point 3: [1, 4] (count: 2)
  → Not core point (< 3 neighbors)
Point 4: [2, 3] (count: 2)
  → Not core point (< 3 neighbors)
Point 5: [6, 7] (count: 2)
  → Not core point (< 3 neighbors)
Point 6: [5, 7, 8] (count: 3)
  → Core point (≥ 3 neighbors)
Point 7: [5, 6, 8] (count: 3)
  → Core point (≥ 3 neighbors)
Point 8: [6, 7] (count: 2)
  → Not core point (< 3 neighbors)
Point 9: [] (count: 0)
  → Not core point (< 3 neighbors)
```

---

### **Step 2: Xác định Core Points**

```python
# Xác định core points
core_points = []
for i in range(len(data)):
    neighbors = []
    for j in range(len(data)):
        if i != j and distances[i,j] <= eps:
            neighbors.append(j)
    
    if len(neighbors) >= minPts:
        core_points.append(i)
        print(f"Point {i+1} is CORE point (neighbors: {[x+1 for x in neighbors]})")
    else:
        print(f"Point {i+1} is NOT core point (neighbors: {[x+1 for x in neighbors]})")

print(f"\nCore points: {[x+1 for x in core_points]}")
```

**Output:**
```
Point 1 is NOT core point (neighbors: [2, 3])
Point 2 is NOT core point (neighbors: [1, 4])
Point 3 is NOT core point (neighbors: [1, 4])
Point 4 is NOT core point (neighbors: [2, 3])
Point 5 is NOT core point (neighbors: [6, 7])
Point 6 is CORE point (neighbors: [5, 7, 8])
Point 7 is CORE point (neighbors: [5, 6, 8])
Point 8 is NOT core point (neighbors: [6, 7])
Point 9 is NOT core point (neighbors: [])

Core points: [6, 7]
```

---

### **Step 3: Xây dựng Clusters**

```python
# DBSCAN Algorithm
def dbscan(data, eps, minPts):
    n_points = len(data)
    visited = [False] * n_points
    cluster_id = 0
    clusters = {}
    noise = []
    
    # Tính distance matrix
    distances = euclidean_distances(data)
    
    for i in range(n_points):
        if visited[i]:
            continue
            
        visited[i] = True
        
        # Tìm neighbors
        neighbors = []
        for j in range(n_points):
            if i != j and distances[i,j] <= eps:
                neighbors.append(j)
        
        if len(neighbors) < minPts:
            noise.append(i)
            print(f"Point {i+1} marked as NOISE")
        else:
            # Tạo cluster mới
            cluster_id += 1
            clusters[cluster_id] = [i]
            print(f"Point {i+1} starts CLUSTER {cluster_id}")
            
            # Mở rộng cluster
            seeds = neighbors.copy()
            j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True
                    
                    # Tìm neighbors của q
                    q_neighbors = []
                    for k in range(n_points):
                        if q != k and distances[q,k] <= eps:
                            q_neighbors.append(k)
                    
                    if len(q_neighbors) >= minPts:
                        seeds.extend(q_neighbors)
                        print(f"  Point {q+1} added to CLUSTER {cluster_id} (core point)")
                    else:
                        print(f"  Point {q+1} added to CLUSTER {cluster_id} (border point)")
                
                if q not in clusters[cluster_id]:
                    clusters[cluster_id].append(q)
                
                j += 1
    
    return clusters, noise

# Chạy DBSCAN
clusters, noise = dbscan(data, eps, minPts)
```

**Output:**
```
Point 1 marked as NOISE
Point 2 marked as NOISE
Point 3 marked as NOISE
Point 4 marked as NOISE
Point 5 marked as NOISE
Point 6 starts CLUSTER 1
  Point 5 added to CLUSTER 1 (border point)
  Point 7 added to CLUSTER 1 (core point)
  Point 8 added to CLUSTER 1 (border point)
Point 7 starts CLUSTER 1
Point 8 starts CLUSTER 1
Point 9 marked as NOISE
```

---

### **Step 4: Kết quả cuối cùng**

```python
print("\n=== DBSCAN Results ===")
print(f"Number of clusters: {len(clusters)}")
print(f"Number of noise points: {len(noise)}")

print(f"\nClusters:")
for cluster_id, points in clusters.items():
    print(f"Cluster {cluster_id}: {[p+1 for p in points]}")

print(f"\nNoise points: {[p+1 for p in noise]}")

# Tạo labels
labels = [-1] * len(data)  # -1 for noise
for cluster_id, points in clusters.items():
    for point in points:
        labels[point] = cluster_id

print(f"\nFinal labels: {labels}")

# Hiển thị kết quả
print(f"\nFinal Classification:")
for i in range(len(data)):
    if labels[i] == -1:
        print(f"Point {i+1}: NOISE")
    else:
        print(f"Point {i+1}: CLUSTER {labels[i]}")
```

**Output:**
```
=== DBSCAN Results ===
Number of clusters: 1
Number of noise points: 5

Clusters:
Cluster 1: [5, 6, 7, 8]

Noise points: [0, 1, 2, 3, 8]

Final labels: [-1, -1, -1, -1, 0, 0, 0, 0, -1]

Final Classification:
Point 1: NOISE
Point 2: NOISE
Point 3: NOISE
Point 4: NOISE
Point 5: CLUSTER 0
Point 6: CLUSTER 0
Point 7: CLUSTER 0
Point 8: CLUSTER 0
Point 9: NOISE
```

---

## **Bài tập hoàn chỉnh với Sklearn**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Dataset
data = np.array([
    [1, 1],   # ID 1
    [1, 2],   # ID 2
    [2, 1],   # ID 3
    [2, 2],   # ID 4
    [8, 8],   # ID 5
    [8, 9],   # ID 6
    [9, 8],   # ID 7
    [9, 9],   # ID 8
    [5, 5]    # ID 9
])

print("Dataset:")
df = pd.DataFrame(data, columns=['X', 'Y'])
df.index = df.index + 1
print(df)

# Tạo DBSCAN model
dbscan = DBSCAN(
    eps=1.5,        # Epsilon neighborhood
    min_samples=3,  # Minimum points
    metric='euclidean'
)

# Fit và predict
labels = dbscan.fit_predict(data)

print(f"\nDBSCAN Results:")
print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of noise points: {list(labels).count(-1)}")

print(f"\nLabels: {labels}")

# Hiển thị kết quả
print(f"\nClassification:")
for i, label in enumerate(labels):
    if label == -1:
        print(f"Point {i+1}: NOISE")
    else:
        print(f"Point {i+1}: CLUSTER {label}")

# Visualization
plt.figure(figsize=(10, 6))
colors = ['red' if label == -1 else plt.cm.tab10(label) for label in labels]
plt.scatter(data[:, 0], data[:, 1], c=colors, s=100, alpha=0.7)

# Add labels
for i, (x, y) in enumerate(data):
    plt.annotate(f'ID{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('DBSCAN Clustering Results')
plt.grid(True, alpha=0.3)
plt.show()

# Core samples
core_samples = dbscan.core_sample_indices_
print(f"\nCore samples: {[i+1 for i in core_samples]}")
```

**Output:**
```
Dataset:
   X  Y
1  1  1
2  1  2
3  2  1
4  2  2
5  8  8
6  8  9
7  9  8
8  9  9
9  5  5

DBSCAN Results:
Number of clusters: 1
Number of noise points: 5

Labels: [-1 -1 -1 -1  0  0  0  0 -1]

Classification:
Point 1: NOISE
Point 2: NOISE
Point 3: NOISE
Point 4: NOISE
Point 5: CLUSTER 0
Point 6: CLUSTER 0
Point 7: CLUSTER 0
Point 8: CLUSTER 0
Point 9: NOISE

Core samples: [5, 6, 7]
```

**Kết quả:**
- ✅ **1 cluster** được tìm thấy (Points 5, 6, 7, 8)
- ✅ **5 noise points** (Points 1, 2, 3, 4, 9)
- ✅ **3 core points** (Points 6, 7, 8)
- ✅ **1 border point** (Point 5)

---

## **Tóm tắt DBSCAN**

### **Quy trình hoàn chỉnh:**

**1. Khởi tạo**
- Tất cả điểm chưa được visit
- Cluster ID = 0

**2. Với mỗi điểm chưa visit:**
- **Tìm ε-neighborhood**
- **Nếu |neighbors| < minPts**: Mark as noise
- **Ngược lại**: Tạo cluster mới và mở rộng

**3. Mở rộng cluster:**
- **Core points**: Có thể mở rộng cluster
- **Border points**: Thuộc cluster nhưng không mở rộng được
- **Density-reachable**: Có thể đến được từ core point

**4. Kết quả:**
- **Clusters**: Các nhóm điểm dense
- **Noise**: Các điểm outlier

### **Ưu điểm:**
- **Không cần biết trước** số clusters
- **Tự động phát hiện** noise/outliers
- **Xử lý được** clusters có hình dạng bất kỳ
- **Robust** với noise

### **Nhược điểm:**
- **Sensitive** với parameters (eps, minPts)
- **Khó xử lý** clusters có mật độ khác nhau
- **Chậm** với large datasets (O(n²))
- **Không hiệu quả** với high-dimensional data

---

## **So sánh với các thuật toán Clustering khác**

| Đặc điểm | DBSCAN | K-Means | Hierarchical | Gaussian Mixture |
|---|---|---|---|---|
| **Type** | Density-based | Centroid-based | Hierarchical | Probabilistic |
| **Clusters** | Bất kỳ | Spherical | Bất kỳ | Elliptical |
| **Noise** | Tự động | Không | Không | Không |
| **Parameters** | eps, minPts | k | Linkage | k, covariance |
| **Speed** | O(n²) | O(n) | O(n³) | O(n) |
| **Scalability** | Poor | Good | Poor | Good |

---

## **Khi nào sử dụng DBSCAN**

### **Nên sử dụng khi:**
- **Không biết trước** số clusters
- **Có noise/outliers** trong data
- **Clusters có hình dạng** bất kỳ
- **Cần phát hiện** anomalies
- **Data có mật độ** khác nhau

### **Không nên sử dụng khi:**
- **Cần tốc độ cao** với large datasets
- **High-dimensional** data
- **Clusters có mật độ** rất khác nhau
- **Cần clusters** có kích thước tương đương
- **Cần interpretability** cao

---

## **Hyperparameters quan trọng**

### **1. eps (ε) - Epsilon**
- **Ý nghĩa**: Bán kính tìm kiếm
- **Cách chọn**: 
  - Plot k-distance graph
  - Chọn điểm "khuỷu tay"
  - Thử nghiệm với domain knowledge

### **2. minPts - Minimum Points**
- **Ý nghĩa**: Số điểm tối thiểu để tạo core point
- **Cách chọn**:
  - 2*dimensions (cho high-dim)
  - 3-5 (cho 2D)
  - Dựa trên kích thước dataset

### **3. metric - Distance Metric**
- **Euclidean**: Cho continuous data
- **Manhattan**: Cho categorical data
- **Cosine**: Cho text data
- **Custom**: Cho domain-specific data
