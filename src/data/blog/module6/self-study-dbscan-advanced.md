---
title: "DBSCAN Nâng Cao - Lý thuyết, Thực hành, và Công thức chọn tham số"
pubDatetime: 2025-10-30T14:00:00Z
featured: false
description: "Hướng dẫn DBSCAN toàn diện: trực giác mật độ, công thức, chọn eps/minPts, biến thể, tối ưu hiệu năng, case-study, và bài tập"
tags: ["machine-learning", "dbscan", "clustering", "density-based", "unsupervised-learning", "anomaly-detection", "hdbscan", "optics"]
---

# DBSCAN Nâng Cao

Tài liệu này mở rộng và chi tiết hóa nội dung trong `self-study-dbscan.md`, đi sâu vào trực giác, chứng minh nhẹ nhàng, kỹ thuật chọn tham số, tối ưu hiệu năng, các biến thể (OPTICS, HDBSCAN), công thức đánh giá, thực hành với nhiều kiểu dữ liệu (2D, high-dim, văn bản, chuỗi thời gian, tọa độ địa lý), cùng các bài tập kèm đáp án gợi ý.

## Mục tiêu học tập

- Hiểu sâu cơ chế density-reachability, density-connectivity, và vì sao DBSCAN ổn định trước nhiễu.
- Biết cách chọn `eps` và `minPts` một cách có hệ thống (k-distance, heuristic theo chiều dữ liệu, domain knowledge).
- Vận dụng DBSCAN cho nhiều miền dữ liệu (liên tục, văn bản, địa lý, thời gian) và biết khi nào không nên dùng.
- Biết các biến thể: OPTICS, HDBSCAN; hiểu khác biệt, khi nào nên chọn biến thể nào.
- Nắm kỹ thực hành: scale/normalize, lựa chọn metric, tối ưu hàng xóm gần (k-d tree, ball tree, ANN), đánh giá kết quả khi không có nhãn.

---

## Trực giác và Lý thuyết cốt lõi

### Trực giác

- **Ý tưởng mật độ**: Vùng có nhiều điểm ở khoảng cách ≤ `eps` sẽ tạo nên cụm; vùng thưa thành nhiễu.
- **Core vs Border vs Noise**:
  - Core: có ít nhất `minPts` lân cận trong bán kính `eps` (tính cả chính nó nếu dùng định nghĩa inclusive).
  - Border: gần core nhưng không đủ mật độ để là core.
  - Noise: không thuộc vùng mật độ nào.

### Định nghĩa (nhắc nhanh)

- Neighborhood: \(N_\varepsilon(p) = \{ q \mid dist(p,q) \le \varepsilon \}\)
- Core: \(|N_\varepsilon(p)| \ge \text{minPts}\)
- Density-reachable: tồn tại chuỗi core nối từ \(q\) đến \(p\) với bước nhảy ≤ `eps`.
- Density-connected: \(p\) và \(q\) density-reachable từ một điểm \(r\).

### Tính chất quan trọng

- Bất biến theo phép hoán vị dữ liệu (order-invariant) nếu hiện thực đúng; kết quả không phụ thuộc thứ tự duyệt.
- Không cần biết số cụm trước; tự phát hiện nhiễu.
- Nhạy với lựa chọn metric và scale của dữ liệu.

---

## Chọn tham số eps và minPts

### Heuristic nhanh

- **minPts**:
  - Dữ liệu 2D: 3–5 là khởi điểm hợp lý.
  - Chiều cao hơn: thường chọn \(\text{minPts} \approx 2 \times \text{dim}\) hoặc 5–20 tùy nhiễu.
- **eps**: dùng đồ thị k-distance (k = minPts) và tìm điểm khuỷu tay.

### Quy trình k-distance elbow

1) Chuẩn hóa dữ liệu (z-score/MinMax) nếu các chiều khác đơn vị:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

X_scaled = StandardScaler().fit_transform(X)  # X shape (n_samples, n_features)

k = minPts  # ví dụ 4, 5, hay 2*dim
nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
nn.fit(X_scaled)
dists, _ = nn.kneighbors(X_scaled)

# Khoảng cách tới hàng xóm thứ k (bỏ bản thân)
kth_dists = np.sort(dists[:, -1])

plt.figure(figsize=(7,4))
plt.plot(kth_dists)
plt.ylabel(f"{k}-distance")
plt.xlabel("Điểm (sắp theo khoảng cách)")
plt.title("k-distance graph để chọn eps")
plt.grid(True, alpha=0.3)
plt.show()
```

2) Chọn **eps** tại “khuỷu tay” (điểm mà độ dốc bắt đầu tăng mạnh). Có thể dùng thuật toán phát hiện elbow tự động, nhưng nên kiểm tra trực quan.

### Mẹo chọn tham số theo miền dữ liệu

- **Dữ liệu địa lý (lat, lon)**: dùng metric Haversine, chuyển `eps` sang đơn vị radian (scikit-learn hỗ trợ `metric='haversine'` khi dữ liệu là radian). Chọn `eps` ≈ bán kính cụm mong muốn/Trái Đất.
- **Văn bản (embedding)**: dùng Cosine; chuẩn hóa vector; `eps` nhỏ (0.1–0.3) thường hợp lý sau khi chuyển Cosine similarity sang Cosine distance \(d = 1 - s\).
- **Chuỗi thời gian**: trích đặc trưng (window, DTW distance), scale theo từng đặc trưng; minPts ≥ 5–10 để ổn định.

### Pitfalls thường gặp

- `eps` quá nhỏ → nhiều điểm thành nhiễu, vỡ cụm.
- `eps` quá lớn → gộp cụm khác nhau, mất chi tiết.
- `minPts` quá thấp → cụm dễ bị nhiễu; quá cao → khó tạo core.

---

## Thuật toán và giả mã rõ ràng

```text
Input: D (n điểm), eps, minPts
Output: Nhãn cụm cho mỗi điểm; -1 là noise

visited := [False]*n
labels := [-1]*n
cluster_id := 0

for mỗi điểm p:
  if visited[p]: continue
  visited[p] := True
  neighbors := {q | dist(p,q) ≤ eps}
  if |neighbors| < minPts:
    labels[p] := -1  # tạm coi là noise, có thể đổi nếu về sau được mở rộng
  else:
    cluster_id := cluster_id + 1
    labels[p] := cluster_id
    seedset := neighbors
    for q in seedset:
      if not visited[q]:
        visited[q] := True
        q_neighbors := {r | dist(q,r) ≤ eps}
        if |q_neighbors| ≥ minPts:
          seedset := seedset ∪ q_neighbors
      if labels[q] == -1:
        labels[q] := cluster_id
```

### Độ phức tạp

- Naive: \(O(n^2)\) do tính khoảng cách mọi cặp.
- Dùng cấu trúc chỉ mục không gian (k-d tree, ball tree) có thể giảm xuống gần \(O(n \log n)\) với dữ liệu thấp chiều và metric lân cận.
- High-dimensional: curse of dimensionality khiến chỉ mục kém hiệu quả → cân nhắc giảm chiều (PCA/UMAP) hoặc HDBSCAN/OPTICS.

---

## Thực hành chuẩn: pipeline tiền xử lý

1) Làm sạch outlier thô (nếu là lỗi đo lường) – paradox: DBSCAN tìm outlier, nhưng lỗi thô có thể phá scale.
2) Biến đổi đặc trưng: chuẩn hóa/scaling theo đơn vị.
3) Chọn metric phù hợp (Euclidean/Manhattan/Cosine/Haversine/DTW).
4) Chọn tham số bằng k-distance + thử nghiệm lưới nhỏ quanh elbow.
5) Đánh giá định tính (mắt, miền kiến thức) và định lượng (nếu có nhãn/chi phí domain).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def run_dbscan(X, eps, min_samples, metric='euclidean'):
    Xs = StandardScaler().fit_transform(X)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(Xs)
    return labels, model
```

---

## Đánh giá cụm khi không có nhãn

- **Silhouette score**: hay dùng nhưng không luôn phù hợp cho cụm phi cầu; dùng để so sánh tương đối giữa tham số.
- **Davies–Bouldin, Calinski–Harabasz**: tương tự, so sánh tương đối.
- **Stability checking**: chạy lại với bootstrap/khử nhiễu nhẹ; cụm tốt thường ổn định.
- **Domain utility**: chi phí/giá trị trong nghiệp vụ (ví dụ phát hiện gian lận giảm false positive bao nhiêu?).

```python
from sklearn.metrics import silhouette_score

def evaluate_silhouette(X, labels, metric='euclidean'):
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return np.nan
    return silhouette_score(X[mask], labels[mask], metric=metric)
```

Lưu ý: loại bỏ noise trước khi tính một số chỉ số; luôn báo cáo tỷ lệ noise kèm theo.

---

## Các biến thể và khi sử dụng

### OPTICS (Ordering Points To Identify the Clustering Structure)

- Trả về ordering và reachability-distance; không cần chốt một `eps` duy nhất, thấy cấu trúc cụm ở nhiều thang mật độ.
- Hữu ích khi mật độ biến thiên mạnh; có thể trích cụm bằng ngưỡng trên reachability plot.

```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5, metric='euclidean')
labels = optics.fit_predict(X)
reachability = optics.reachability_
ordering = optics.ordering_
```

### HDBSCAN (Hierarchical DBSCAN)

- Xây cụm phân cấp theo mật độ, tự động chọn độ bền cụm (cluster selection via stability); thường tốt hơn DBSCAN khi mật độ thay đổi.
- Cài qua `hdbscan` (pip). Tham số chủ yếu: `min_cluster_size`, `min_samples`.

```python
# pip install hdbscan
import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
labels = hdb.fit_predict(X)
prob = hdb.probabilities_  # độ tin cậy gán cụm
```

So sánh nhanh:

- **DBSCAN**: đơn giản, cần `eps`, tốt khi mật độ tương đối đồng đều.
- **OPTICS**: không cần `eps` cố định, khám phá nhiều mức mật độ.
- **HDBSCAN**: tự chọn cụm theo độ bền, thường vững hơn trước mật độ thay đổi, ít cần tinh chỉnh `eps`.

---

## Ứng dụng theo miền dữ liệu

### 2D/3D liên tục

- Chuẩn hóa; chọn Euclidean; elbow chọn `eps`.
- Dùng k-d tree/ball tree mặc định của sklearn để tăng tốc.

### Văn bản (embeddings)

- Tạo vector (Sentence-BERT, fastText, TF-IDF → SVD/UMAP).
- Dùng Cosine distance, scale vector (l2-normalize).
- `eps` nhỏ (0.1–0.3) sau khi chuẩn hóa thường hợp lý; kiểm tra tỉ lệ noise.

### Tọa độ địa lý

- Dữ liệu đầu vào theo radian nếu dùng `haversine`.
- `eps` ≈ bán kính mong muốn / bán kính Trái Đất (6371 km).

```python
import numpy as np
from sklearn.cluster import DBSCAN

# lat, lon độ → radian
coords_deg = np.array([[10.77, 106.70], [10.78, 106.69], ...])
coords_rad = np.radians(coords_deg)

eps_km = 0.8  # bán kính cụm 0.8 km
eps = eps_km / 6371.0

labels = DBSCAN(eps=eps, min_samples=5, metric='haversine').fit_predict(coords_rad)
```

### Chuỗi thời gian / IoT

- Trích đặc trưng (rolling mean/std, spectral features) hoặc dùng DTW distance (tslearn).
- Phát hiện đoạn bất thường (anomaly) bằng cách xem nhãn -1.

---

## Tối ưu hiệu năng và khả năng mở rộng

- Giảm chiều (PCA, UMAP) trước khi chạy DBSCAN ở high-dim để tăng tương phản khoảng cách.
- Dùng cấu trúc lân cận: `NearestNeighbors(algorithm='ball_tree'|'kd_tree')` cho Euclidean thấp chiều.
- Dùng approximate nearest neighbors (FAISS, Annoy, HNSWlib) để ước lượng neighborhood khi dữ liệu rất lớn; sau đó áp dụng DBSCAN trên đồ thị k-NN xấp xỉ.
- Batch/mini-batch: chia nhỏ dữ liệu, dựng đồ thị k-NN cục bộ rồi hợp nhất (cần cẩn thận phần biên, thích hợp khi cụm cách xa).

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
nn.fit(X)
graph = nn.kneighbors_graph(mode='distance')  # sparse graph phục vụ tìm lân cận nhanh
```

---

## Case study ngắn: Phân cụm địa điểm cửa hàng

- Bài toán: gom cụm cửa hàng để tối ưu tuyến giao hàng trong bán kính 1 km.
- Dữ liệu: lat/lon ~ 50,000 điểm tại một thành phố.
- Cách làm:
  - Chuyển sang radian, `metric='haversine'`, `eps=1/6371` (~1 km), `min_samples=8`.
  - Loại bỏ cụm đơn lẻ nhỏ (hậu xử lý) hoặc gán -1.
  - Đánh giá: tỉ lệ điểm gán cụm, trực quan hóa trên bản đồ; thử `min_samples` 5–12.
- Kết quả: cụm khu vực trung tâm rõ ràng, ngoại vi nhiều noise → quyết định thêm điểm trung chuyển tại trung tâm.

---

## Công thức và ghi chú thực hành quan trọng

- Bao gồm chính điểm \(p\) trong \(N_\varepsilon(p)\) hay không? Thư viện khác nhau; sklearn tính số lân cận dựa vào khoảng cách đến điểm khác (không tính chính nó) → khi so sánh lý thuyết cần thống nhất.
- Chuẩn hóa rất quan trọng khi đặc trưng có đơn vị khác nhau; nếu không, chiều có variance lớn sẽ lấn át metric.
- Khi so sánh nhiều chạy: báo cáo `tỷ lệ noise`, `số cụm`, và chỉ số đánh giá; đừng chỉ nhìn silhouette.

---

## Bài tập tự luyện (có gợi ý)

1) Vẽ k-distance cho một bộ dữ liệu 2D synthetic gồm 3 cụm bán kính khác nhau. Hỏi: vì sao DBSCAN với một `eps` không bắt trọn cả 3 cụm tốt? Gợi ý: mật độ thay đổi → cân nhắc OPTICS/HDBSCAN.

2) Trên embeddings câu (SBERT) của 5,000 review, chuẩn hóa L2, thử DBSCAN với `eps ∈ {0.15, 0.2, 0.25}` và `min_samples ∈ {5, 10}`. Báo cáo tỷ lệ noise và 5 cụm lớn nhất theo top từ khóa (dùng TF-IDF centroids).

3) Dữ liệu GPS: gom cụm điểm dừng xe buýt trong bán kính 200 m. Chuyển `eps` sang radian; so sánh kết quả khi `min_samples=3` và `min_samples=8`.

4) High-dimensional (d=100): so sánh DBSCAN trên dữ liệu gốc vs PCA 20 chiều; so sánh silhouette và tỷ lệ noise.

5) Thực thi HDBSCAN cho bộ dữ liệu có mật độ biến thiên; so sánh số cụm, tỷ lệ noise, và độ ổn định khi bootstrap 10 lần.

---

## FAQ ngắn

- Hỏi: Vì sao DBSCAN gộp hai cụm gần nhau thành một? → `eps` quá lớn hoặc cầu nối điểm border-core; giảm `eps` hoặc tăng `minPts`.
- Hỏi: Vì sao quá nhiều noise? → `eps` quá nhỏ hoặc dữ liệu chưa chuẩn hóa; tăng `eps`/giảm `minPts`/chuẩn hóa.
- Hỏi: Dữ liệu rất lớn, chạy chậm? → Dùng ANN hoặc giảm chiều; thử OPTICS/HDBSCAN.

---

## Tài liệu tham khảo

- Ester et al., “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise” (KDD, 1996).
- Campello et al., “Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection” (HDBSCAN).
- McInnes et al., `hdbscan` library documentation.
- Scikit-learn `DBSCAN` và `OPTICS` documentation.

---

## Ví dụ tính tay dạng bảng (phong cách như m4-w2-xgboost.md)

Ở phần này, ta dùng cùng dataset 2D nhỏ để minh họa DBSCAN theo bảng, với tham số: `eps = 1.5`, `minPts = 3`.

### Dataset

| ID | X | Y |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 1 | 2 |
| 3 | 2 | 1 |
| 4 | 2 | 2 |
| 5 | 8 | 8 |
| 6 | 8 | 9 |
| 7 | 9 | 8 |
| 8 | 9 | 9 |
| 9 | 5 | 5 |

---

### Ma trận khoảng cách Euclidean (làm tròn 2 chữ số)

| ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 1.00 | 1.41 | 9.90 | 9.22 | 8.06 | 8.49 | 5.66 |
| 2 | 1.00 | 0.00 | 1.41 | 1.00 | 9.22 | 8.49 | 8.49 | 8.06 | 5.39 |
| 3 | 1.00 | 1.41 | 0.00 | 1.00 | 8.49 | 8.06 | 9.22 | 9.90 | 4.24 |
| 4 | 1.41 | 1.00 | 1.00 | 0.00 | 8.06 | 7.81 | 8.49 | 9.22 | 4.24 |
| 5 | 9.90 | 9.22 | 8.49 | 8.06 | 0.00 | 1.00 | 1.00 | 1.41 | 4.24 |
| 6 | 9.22 | 8.49 | 8.06 | 7.81 | 1.00 | 0.00 | 1.41 | 1.00 | 3.61 |
| 7 | 8.06 | 8.49 | 9.22 | 8.49 | 1.00 | 1.41 | 0.00 | 1.00 | 4.24 |
| 8 | 8.49 | 8.06 | 9.90 | 9.22 | 1.41 | 1.00 | 1.00 | 0.00 | 4.24 |
| 9 | 5.66 | 5.39 | 4.24 | 4.24 | 4.24 | 3.61 | 4.24 | 4.24 | 0.00 |

---

### ε-neighborhood và phân loại Core/Border/Noise (eps = 1.5, minPts = 3)

| Point | Neighbors (≤ 1.5, loại trừ bản thân) | Count | Loại |
|:-----:|:--------------------------------------|------:|:-----|
| 1 | [2, 3] | 2 | Not core |
| 2 | [1, 4] | 2 | Not core |
| 3 | [1, 4] | 2 | Not core |
| 4 | [2, 3] | 2 | Not core |
| 5 | [6, 7] | 2 | Not core |
| 6 | [5, 7, 8] | 3 | Core |
| 7 | [5, 6, 8] | 3 | Core |
| 8 | [6, 7] | 2 | Not core |
| 9 | [] | 0 | Not core |

Ghi chú: Điểm border là điểm không phải core nhưng nằm trong ε của ít nhất một core. Theo bảng trên, 5 và 8 là ứng viên border (gần core 6/7).

---

### Các bước xây cụm (DBSCAN)

Khởi tạo: tất cả unvisited, `cluster_id = 0`.

| Bước | Điểm xét | Láng giềng (≤ eps) | Trạng thái | Hành động |
|:----:|:--------:|:-------------------|:----------:|:----------|
| 1 | 1 | [2,3] | < minPts | Mark noise tạm |
| 2 | 2 | [1,4] | < minPts | Mark noise tạm |
| 3 | 3 | [1,4] | < minPts | Mark noise tạm |
| 4 | 4 | [2,3] | < minPts | Mark noise tạm |
| 5 | 5 | [6,7] | < minPts | Mark noise tạm |
| 6 | 6 | [5,7,8] | ≥ minPts | Tạo Cluster 1, seed=[5,7,8] |

Mở rộng Cluster 1 với seedset khởi đầu từ điểm 6:

| Seed idx | q | q là visited? | q_neighbors | |q_neighbors| ≥ minPts? | Hành động | Thành viên cụm |
|:--------:|:--:|:-------------:|:-----------:|:---------------------:|:---------:|:----------------|
| 1 | 5 | False → True | [6,7] | 2 | Border | Thêm 5 |
| 2 | 7 | False → True | [5,6,8] | 3 | Core | Thêm [5,6,8] vào seed (nếu chưa có) + thêm 7 |
| 3 | 8 | False → True | [6,7] | 2 | Border | Thêm 8 |
| … | 6/7/8 lặp lại | True | – | – | Bỏ qua | – |

Kết thúc mở rộng: Cluster 1 = {6, 5, 7, 8}.

Tiếp tục vòng lặp tổng thể:

| Bước | Điểm xét | Trạng thái |
|:----:|:--------:|:----------|
| 7 | 7 | Đã thuộc Cluster 1 |
| 8 | 8 | Đã thuộc Cluster 1 |
| 9 | 9 | Không có láng giềng | Mark noise |

---

### Kết quả cuối cùng

| Cluster | Thành viên (ID) |
|:-------:|:-----------------|
| 1 | [5, 6, 7, 8] |

| Noise (ID) |
|:----------:|
| [1, 2, 3, 4, 9] |

| ID | Nhãn |
|---:|:-----|
| 1 | -1 |
| 2 | -1 |
| 3 | -1 |
| 4 | -1 |
| 5 | 1 |
| 6 | 1 |
| 7 | 1 |
| 8 | 1 |
| 9 | -1 |

Lưu ý: Một số hiện thực đánh dấu tạm noise có thể được “thu hồi” về sau nếu điểm được kết nối mật độ tới cụm thông qua một core.

---

## Phụ lục: Demo nhanh đầy đủ (sklearn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1) Tạo dữ liệu
X, y_true = make_blobs(n_samples=1500, centers=[[0,0],[5,5],[9,0]], cluster_std=[0.3, 0.6, 0.4], random_state=42)

# 2) Chuẩn hóa
Xs = StandardScaler().fit_transform(X)

# 3) DBSCAN
db = DBSCAN(eps=0.25, min_samples=5, metric='euclidean')
labels = db.fit_predict(Xs)

# 4) Đánh giá nhanh
mask = labels != -1
sil = silhouette_score(Xs[mask], labels[mask]) if mask.sum()>1 and len(np.unique(labels[mask]))>1 else np.nan
print({
    'num_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
    'noise_ratio': float((labels==-1).mean()),
    'silhouette': float(sil)
})

# 5) Vẽ
plt.figure(figsize=(6,5))
colors = np.array(['#d3d3d3'] + [plt.cm.tab10(i) for i in range(10)]).astype(object)
color_idx = np.where(labels==-1, 0, labels+1)
plt.scatter(Xs[:,0], Xs[:,1], c=colors[color_idx], s=12, alpha=0.9)
plt.title('DBSCAN (chuẩn hóa, eps=0.25, min_samples=5)')
plt.grid(True, alpha=0.3)
plt.show()
```


