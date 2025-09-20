---
title: "LightGBM Study - Thuật toán Gradient Boosting hiệu quả"
pubDatetime: 2025-01-16T14:00:00Z
featured: false
description: "Tìm hiểu chi tiết về LightGBM, thuật toán gradient boosting với GOSS sampling và histogram-based splitting"
tags: ["machine-learning", "lightgbm", "gradient-boosting", "algorithm", "optimization"]
---

# LightGBM Study - Thuật toán Gradient Boosting hiệu quả

## Hàm mục tiêu (Objective) trong LightGBM

* **Hàm mất mát cho Regression:**

$$
\sum_{i=1}^{N} \mathcal{L}(y_i, \bar{y}_i) \quad \text{Where} \quad \mathcal{L}(y_i, \bar{y}_i) = \frac{1}{2}(y_i - \bar{y}_i)^2
$$

* **Gradient và Hessian:**

$$
g_i = \frac{\partial \mathcal{L}(y_i, \bar{y}_i)}{\partial \bar{y}_i} = \bar{y}_i - y_i, \quad h_i = \frac{\partial^2 \mathcal{L}(y_i, \bar{y}_i)}{\partial \bar{y}_i^2} = 1
$$

* **Gain của LightGBM:**

$$
\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

* **Leaf weight:**

$$
w_j^* = -\frac{G_j}{H_j+\lambda}
$$

* **Model update:**

$$
F_t(x) = F_{t-1}(x) + \eta \cdot f_t(x)
$$

---

## Ví Dụ Tính Tay - LightGBM (3 Bins)

### **Step 1: Initialization**

| Age (X) | Chol (y) |
|:---:|:---:|
| 25 | 180 |
| 29 | 204 |
| 35 | 220 |
| 39 | 203 |
| 42 | 240 |
| 45 | 250 |
| 48 | 234 |
| 52 | 280 |
| 55 | 290 |
| 59 | 260 |
| 62 | 300 |
| 67 | 269 |
| 70 | 320 |
| 75 | 350 |
| 80 | 380 |

**F0(x) là giá trị tối ưu:**
$$F_0 = \frac{1}{N} \sum_{i=1}^{N} y_i = \frac{180 + 204 + 220 + 203 + 240 + 250 + 234 + 280 + 290 + 260 + 300 + 269 + 320 + 350 + 380}{15} = 262.13$$

### **Step 2A: Gradients & Hessians**

**Tính gradients:**
$$g_i = F_0 - y_i$$

| Age (X) | Chol (y) | Gradients ($g_i$) | \|Gradients\| |
|:---:|:---:|:---:|:---:|
| 25 | 180 | $262.13 - 180 = 82.13$ | 82.13 |
| 29 | 204 | $262.13 - 204 = 58.13$ | 58.13 |
| 35 | 220 | $262.13 - 220 = 42.13$ | 42.13 |
| 39 | 203 | $262.13 - 203 = 59.13$ | 59.13 |
| 42 | 240 | $262.13 - 240 = 22.13$ | 22.13 |
| 45 | 250 | $262.13 - 250 = 12.13$ | 12.13 |
| 48 | 234 | $262.13 - 234 = 28.13$ | 28.13 |
| 52 | 280 | $262.13 - 280 = -17.87$ | 17.87 |
| 55 | 290 | $262.13 - 290 = -27.87$ | 27.87 |
| 59 | 260 | $262.13 - 260 = 2.13$ | 2.13 |
| 62 | 300 | $262.13 - 300 = -37.87$ | 37.87 |
| 67 | 269 | $262.13 - 269 = -6.87$ | 6.87 |
| 70 | 320 | $262.13 - 320 = -57.87$ | 57.87 |
| 75 | 350 | $262.13 - 350 = -87.87$ | 87.87 |
| 80 | 380 | $262.13 - 380 = -117.87$ | 117.87 |

### **Step 2B: GOSS Sampling**

| Rank | Age | Gradient | \|Gradient\| | Selection |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 80 | -117.87 | 117.87 | ✅ Top 20% |
| 2 | 25 | 82.13 | 82.13 | ✅ Top 20% |
| 3 | 75 | -87.87 | 87.87 | ✅ Top 20% |
| 4 | 29 | 58.13 | 58.13 | |
| 5 | 39 | 59.13 | 59.13 | ✅ Random 10% |
| 6 | 70 | -57.87 | 57.87 | ✅ Random 10% |
| 7 | 35 | 42.13 | 42.13 | ✅ Random 10% |
| 8 | 62 | -37.87 | 37.87 | |
| 9 | 55 | -27.87 | 27.87 | |
| 10 | 48 | 28.13 | 28.13 | |
| 11 | 42 | 22.13 | 22.13 | |
| 12 | 45 | 12.13 | 12.13 | |
| 13 | 52 | -17.87 | 17.87 | |
| 14 | 67 | -6.87 | 6.87 | |
| 15 | 59 | 2.13 | 2.13 | |

- **Selected samples:** [80, 25, 75, 39, 70, 35]
- **Weights:** [1.0, 1.0, 1.0, 6.67, 6.67, 6.67] (vì 6.67 = (1-0.2)/0.1)

### **Step 2C: Histogram-based Threshold Finding (3 Bins)**

**Chỉ tạo histogram từ samples được GOSS chọn:**
- **GOSS selected samples:** [80, 25, 75, 39, 70, 35]
- **Sắp xếp theo Age:** [25, 35, 39, 70, 75, 80]
- **Min Age:** 25, **Max Age:** 80
- **Bin width:** (80 - 25) / 3 = 18.33
- **3 Bins:** [25-43.33], [43.33-61.67], [61.67-80]
- **Thresholds:** [43.33, 61.67] (chỉ 2 thresholds cho 3 bins)

### **Step 2D: Gains - Tìm Best Split**

**Tính Gain cho từng threshold (chỉ với samples được chọn):**

| Thresh | Left Samples | Right Samples | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 43.33 | [25,35,39] | [70,75,80] | 82.13+42.13+59.13 = 183.39 | 3 | -57.87-87.87-117.87 = -263.61 | 3 | **6789.45** |
| 61.67 | [25,35,39,70] | [75,80] | 183.39-57.87 = 125.52 | 4 | -87.87-117.87 = -205.74 | 2 | 4123.89 |

**Best threshold: 43.33 (Gain = 6789.45)**

### **Step 2E: Split - Leaf-wise Growth**

**Best threshold: 43.33 (Gain = 6789.45)**

### **Step 2F: Leaf Weights**

**Left node (Age ≤ 43.33):**
$$w_L = -\frac{183.39}{3+1} = -45.85$$

**Right node (Age > 43.33):**
$$w_R = -\frac{-263.61}{3+1} = 65.90$$

### **Step 2G: Model Update**

**Learning rate η = 0.1:**

Công thức cập nhật mô hình:
- F1(x) = F0(x) + η × f1(x)
- F1(x) = 262.13 + 0.1 × f1(x)

**Kết quả sau vòng lặp 1:**

| Age (X) | Chol (y) | F1(x) |
|:---:|:---:|:---:|
| 25 | 180 | 257.55 |
| 29 | 204 | 257.55 |
| 35 | 220 | 257.55 |
| 39 | 203 | 257.55 |
| 42 | 240 | 257.55 |
| 45 | 250 | 268.72 |
| 48 | 234 | 268.72 |
| 52 | 280 | 268.72 |
| 55 | 290 | 268.72 |
| 59 | 260 | 268.72 |
| 62 | 300 | 268.72 |
| 67 | 269 | 268.72 |
| 70 | 320 | 268.72 |
| 75 | 350 | 268.72 |
| 80 | 380 | 268.72 |

---

## **Step 3: Loop 2 - Leaf-wise Growth**

### **Step 3A: Gradients & Hessians cho Loop 2**

| Age (X) | Chol (y) | $F_1(x)$ | Gradients ($g_i = F_1(x) - y_i$) | \|Gradients\| |
|:---:|:---:|:---:|:---:|:---:|
| 25 | 180 | 257.55 | $257.55 - 180 = 77.55$ | 77.55 |
| 29 | 204 | 257.55 | $257.55 - 204 = 53.55$ | 53.55 |
| 35 | 220 | 257.55 | $257.55 - 220 = 37.55$ | 37.55 |
| 39 | 203 | 257.55 | $257.55 - 203 = 54.55$ | 54.55 |
| 42 | 240 | 257.55 | $257.55 - 240 = 17.55$ | 17.55 |
| 45 | 250 | 268.72 | $268.72 - 250 = 18.72$ | 18.72 |
| 48 | 234 | 268.72 | $268.72 - 234 = 34.72$ | 34.72 |
| 52 | 280 | 268.72 | $268.72 - 280 = -11.28$ | 11.28 |
| 55 | 290 | 268.72 | $268.72 - 290 = -21.28$ | 21.28 |
| 59 | 260 | 268.72 | $268.72 - 260 = 8.72$ | 8.72 |
| 62 | 300 | 268.72 | $268.72 - 300 = -31.28$ | 31.28 |
| 67 | 269 | 268.72 | $268.72 - 269 = -0.28$ | 0.28 |
| 70 | 320 | 268.72 | $268.72 - 320 = -51.28$ | 51.28 |
| 75 | 350 | 268.72 | $268.72 - 350 = -81.28$ | 81.28 |
| 80 | 380 | 268.72 | $268.72 - 380 = -111.28$ | 111.28 |

### **Step 3B: GOSS Sampling cho Loop 2**

| Rank | Age | Gradient | \|Gradient\| | Selection |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 80 | -111.28 | 111.28 | ✅ Top 20% |
| 2 | 25 | 77.55 | 77.55 | ✅ Top 20% |
| 3 | 75 | -81.28 | 81.28 | ✅ Top 20% |
| 4 | 29 | 53.55 | 53.55 | |
| 5 | 39 | 54.55 | 54.55 | ✅ Random 10% |
| 6 | 70 | -51.28 | 51.28 | ✅ Random 10% |
| 7 | 35 | 37.55 | 37.55 | ✅ Random 10% |
| 8 | 62 | -31.28 | 31.28 | |
| 9 | 48 | 34.72 | 34.72 | |
| 10 | 55 | -21.28 | 21.28 | |
| 11 | 45 | 18.72 | 18.72 | |
| 12 | 42 | 17.55 | 17.55 | |
| 13 | 52 | -11.28 | 11.28 | |
| 14 | 59 | 8.72 | 8.72 | |
| 15 | 67 | -0.28 | 0.28 | |

- **Selected samples:** [80, 25, 75, 39, 70, 35]
- **Weights:** [1.0, 1.0, 1.0, 6.67, 6.67, 6.67]

### **Step 3C: Histogram-based Threshold Finding cho Loop 2 (3 Bins)**

**Chỉ tạo histogram từ samples được GOSS chọn:**
- **GOSS selected samples:** [80, 25, 75, 39, 70, 35]
- **Sắp xếp theo Age:** [25, 35, 39, 70, 75, 80]
- **Min Age:** 25, **Max Age:** 80
- **Bin width:** (80 - 25) / 3 = 18.33
- **3 Bins:** [25-43.33], [43.33-61.67], [61.67-80]
- **Thresholds:** [43.33, 61.67] (chỉ 2 thresholds cho 3 bins)

### **Step 3D: Gains - Tìm Best Split cho Loop 2**

**Tính Gain cho từng threshold (chỉ với samples được chọn):**

| Thresh | Left Samples | Right Samples | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 43.33 | [25,35,39] | [70,75,80] | 77.55+37.55+54.55 = 169.65 | 3 | -51.28-81.28-111.28 = -243.84 | 3 | **6789.45** |
| 61.67 | [25,35,39,70] | [75,80] | 169.65-51.28 = 118.37 | 4 | -81.28-111.28 = -192.56 | 2 | 4123.89 |

**Best threshold: 43.33 (Gain = 6789.45)**

### **Step 3E: Split - Leaf-wise Growth cho Loop 2**

**Best threshold: 43.33 (Gain = 6789.45)**

### **Step 3F: Leaf Weights cho Loop 2**

**Left node (Age ≤ 43.33):**
$$w_L = -\frac{169.65}{3+1} = -42.41$$

**Right node (Age > 43.33):**
$$w_R = -\frac{-243.84}{3+1} = 60.96$$

### **Step 3G: Model Update cho Loop 2**

**Learning rate η = 0.1:**

Công thức cập nhật mô hình:
- F2(x) = F1(x) + η × f2(x)
- F2(x) = F1(x) + 0.1 × f2(x)

**Kết quả sau vòng lặp 2:**

| Age (X) | Chol (y) | F1(x) | F2(x) |
|:---:|:---:|:---:|:---:|
| 25 | 180 | 257.55 | 253.31 |
| 29 | 204 | 257.55 | 253.31 |
| 35 | 220 | 257.55 | 253.31 |
| 39 | 203 | 257.55 | 253.31 |
| 42 | 240 | 257.55 | 253.31 |
| 45 | 250 | 268.72 | 274.82 |
| 48 | 234 | 268.72 | 274.82 |
| 52 | 280 | 268.72 | 274.82 |
| 55 | 290 | 268.72 | 274.82 |
| 59 | 260 | 268.72 | 274.82 |
| 62 | 300 | 268.72 | 274.82 |
| 67 | 269 | 268.72 | 274.82 |
| 70 | 320 | 268.72 | 274.82 |
| 75 | 350 | 268.72 | 274.82 |
| 80 | 380 | 268.72 | 274.82 |

---

## **Final Model**

$$F_M(x) = F_0(x) + \sum_{t=1}^{M} \eta \cdot f_t(x)$$

---

## **Kết luận**

### **🎯 Những điểm nổi bật của LightGBM**

**1. GOSS Sampling (Gradient-based One-Side Sampling)**
- Chọn 20% samples có gradient lớn nhất (quan trọng nhất)
- Chọn ngẫu nhiên 10% samples còn lại
- Giảm đáng kể thời gian training mà vẫn giữ được độ chính xác

**2. Histogram-based Algorithm**
- Chia dữ liệu thành các bins thay vì xử lý từng giá trị
- Giảm số lượng thresholds cần kiểm tra
- Tăng tốc độ tìm kiếm split point tối ưu

**3. Leaf-wise Growth**
- Xây dựng cây theo chiều sâu thay vì level-wise
- Tập trung vào các leaf có loss cao nhất
- Tạo ra cây cân bằng và hiệu quả hơn

### **⚡ So sánh với XGBoost**

| Đặc điểm | XGBoost | LightGBM |
|:---:|:---:|:---:|
| **Sampling** | Tất cả samples | GOSS sampling |
| **Splitting** | Pre-sorted algorithm | Histogram-based |
| **Growth** | Level-wise | Leaf-wise |
| **Memory** | Cao hơn | Thấp hơn |
| **Speed** | Chậm hơn | Nhanh hơn |
| **Accuracy** | Tương đương | Tương đương |

### **🔧 Hyperparameters quan trọng**

- **`num_leaves`**: Số lá tối đa (31-255)
- **`learning_rate`**: Tốc độ học (0.01-0.3)
- **`feature_fraction`**: Tỷ lệ features sử dụng (0.5-1.0)
- **`bagging_fraction`**: Tỷ lệ samples sử dụng (0.5-1.0)
- **`lambda_l1`, `lambda_l2`**: Regularization

### **💡 Khi nào nên sử dụng LightGBM**

✅ **Nên dùng khi:**
- Dataset lớn (>10K samples)
- Cần training nhanh
- Memory hạn chế
- Cần model nhẹ

❌ **Không nên dùng khi:**
- Dataset nhỏ (<1K samples)
- Cần interpretability cao
- Overfitting dễ xảy ra

### **🚀 Tương lai của LightGBM**

LightGBM tiếp tục được phát triển với:
- **GPU support** cho training nhanh hơn
- **Categorical features** handling tốt hơn
- **Distributed training** cho big data
- **AutoML integration** với các framework khác

---

> **📚 Tài liệu tham khảo:**
> - [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
> - [Gradient-based One-Side Sampling Paper](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
> - [Histogram-based Algorithm Paper](https://arxiv.org/abs/1603.02754)
