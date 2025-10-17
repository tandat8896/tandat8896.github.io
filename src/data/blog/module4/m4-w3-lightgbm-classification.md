---
title: "LightGBM Classification - Thuật toán Gradient Boosting với GOSS và Histogram"
pubDatetime: 2025-09-21T10:00:00Z
featured: false
description: "Tìm hiểu chi tiết về LightGBM cho Classification với GOSS Sampling và Histogram-based Splitting"
tags: ["machine-learning", "lightgbm", "classification", "gradient-boosting", "goss", "histogram"]
---

# LightGBM Classification

## Hàm mục tiêu (Objective) trong LightGBM Classification

### **Hàm mất mát cho Classification (Log Loss)**

$$\mathcal{L}(y_i, \hat{y}_i) = -y_i\log(\hat{y}_i) - (1-y_i)\log(1-\hat{y}_i)$$

### **Chuyển đổi xác suất thành log(odds)**

$$\hat{y}_i = \frac{1}{1 + e^{-F(x_i)}}$$

$$\log(odds) = F(x_i) = \log\left(\frac{\hat{y}_i}{1-\hat{y}_i}\right)$$

### **Gradient và Hessian cho Classification**

**Gradient (đạo hàm bậc nhất):**
$$g_i = \frac{\partial \mathcal{L}}{\partial F(x_i)} = \hat{y}_i - y_i$$

**Hessian (đạo hàm bậc hai):**
$$h_i = \frac{\partial^2 \mathcal{L}}{\partial F(x_i)^2} = \hat{y}_i(1-\hat{y}_i)$$

### **Gain của LightGBM Classification**

$$\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma$$

### **Leaf weight cho Classification**

$$w_j^* = -\frac{G_j}{H_j+\lambda}$$

---

## Đặc điểm nổi bật của LightGBM

### **1. GOSS (Gradient-based One-Side Sampling)**
- **Vấn đề:** Gradient boosting chậm với large datasets
- **Giải pháp:** Chọn samples có gradient lớn và random samples có gradient nhỏ
- **Cơ chế:** Giữ lại tất cả samples có gradient lớn, random chọn samples có gradient nhỏ

### **2. Histogram-based Splitting**
- **Vấn đề:** Tìm threshold tối ưu chậm
- **Giải pháp:** Sử dụng histogram để tìm split points
- **Cơ chế:** Chia continuous features thành bins, tìm split points giữa các bins

### **3. Leaf-wise Tree Growth**
- **Cấu trúc:** Tăng trưởng theo chiều sâu thay vì chiều rộng
- **Ưu điểm:** Tăng tốc độ training, giảm memory usage
- **Ứng dụng:** Hiệu quả với high-dimensional data

### **4. EFB (Exclusive Feature Bundling)**
- **Vấn đề:** Sparse features làm tăng memory usage và giảm tốc độ
- **Giải pháp:** Gộp các features không đồng thời xuất hiện (mutually exclusive)
- **Cơ chế:** Tạo feature bundles để giảm số lượng features
- **Ưu điểm:** Giảm 50-80% memory usage, tăng tốc độ training

#### **Ví dụ EFB cụ thể:**

**Dataset gốc với 6 features:**
| ID | Feature_1 | Feature_2 | Feature_3 | Feature_4 | Feature_5 | Feature_6 |
|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 | 0 | 0 | 1 |
| 4 | 1 | 0 | 0 | 1 | 0 | 0 |
| 5 | 0 | 1 | 0 | 0 | 1 | 0 |

**Phân tích Mutual Exclusivity:**
- **Bundle 1:** Feature_1 và Feature_4 (không bao giờ cùng = 1)
- **Bundle 2:** Feature_2 và Feature_5 (không bao giờ cùng = 1)
- **Bundle 3:** Feature_3 và Feature_6 (không bao giờ cùng = 1)

**Dataset sau EFB (3 bundles thay vì 6 features):**
| ID | Bundle_1 | Bundle_2 | Bundle_3 |
|---|---|---|---|
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 0 | 1 | 0 |

**Kết quả:**
- **Memory usage:** Giảm 50% (6 → 3 features)
- **Training speed:** Tăng 2x
- **Accuracy:** Giữ nguyên (không mất thông tin)

---

## Ví Dụ Tính Tay - LightGBM Classification

### **Dataset Classification**

| ID | Age (X1) | Income (X2) | Label (y) |
|---|---|---|---|
| 1 | 25 | 30 | 1 |
| 2 | 30 | 50 | 1 |
| 3 | 35 | 40 | 0 |
| 4 | 40 | 60 | 0 |
| 5 | 45 | 70 | 1 |
| 6 | 50 | 80 | 0 |

**Mục tiêu:** Dự đoán label dựa trên Age và Income

---

### **Step 1: Initialization**

**F0(x) là giá trị tối ưu cho log(odds):**
$$F_0 = \log\left(\frac{p}{1-p}\right) \quad \\text{Where } p = \frac{\sum_{i=1}^{6} y_i}{6} = \frac{3}{6} = 0.5$$

$$F_0 = \log\left(\frac{0.5}{1-0.5}\right) = \log(1) = 0$$

**Initial predictions (probabilities):**
$$\hat{y}_i = \frac{1}{1 + e^{-F_0}} = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = 0.5$$

| ID | Age | Income | Label | $F_0(x)$ | $\hat{y}$ |
|---|---|---|---|:---:|:---:|
| 1 | 25 | 30 | 1 | 0 | 0.5 |
| 2 | 30 | 50 | 1 | 0 | 0.5 |
| 3 | 35 | 40 | 0 | 0 | 0.5 |
| 4 | 40 | 60 | 0 | 0 | 0.5 |
| 5 | 45 | 70 | 1 | 0 | 0.5 |
| 6 | 50 | 80 | 0 | 0 | 0.5 |

---

### **Step 2: GOSS Sampling Setup**

**Tính gradients:**
$$g_i = \hat{y}_i - y_i$$

| ID | Age | Income | Label | y_hat | Gradient ($g_i$) | |Gradient| |
|---|---|---|---|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.5 | -0.5 | 0.5 |
| 2 | 30 | 50 | 1 | 0.5 | -0.5 | 0.5 |
| 3 | 35 | 40 | 0 | 0.5 | 0.5 | 0.5 |
| 4 | 40 | 60 | 0 | 0.5 | 0.5 | 0.5 |
| 5 | 45 | 70 | 1 | 0.5 | -0.5 | 0.5 |
| 6 | 50 | 80 | 0 | 0.5 | 0.5 | 0.5 |

**GOSS Sampling:**
- **Top samples (gradient lớn):** Tất cả samples (vì |gradient| = 0.5 cho tất cả)
- **Random samples (gradient nhỏ):** Không có
- **Selected samples:** [1, 2, 3, 4, 5, 6] (tất cả)

---

### **Step 3: Histogram-based Splitting**

**Tạo histogram cho Age:**
- **Bins:** [25, 30, 35, 40, 45, 50]
- **Thresholds:** [27.5, 32.5, 37.5, 42.5, 47.5]

**Tính Gain cho các thresholds:**

| Threshold | Left (<=) | Right (>) | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|---|---|---|---|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | -0.5 | 0.25 | -0.5+0.5+0.5-0.5+0.5 = 0.5 | 1.25 | 0.125 |
| 32.5 | [1,2] | [3,4,5,6] | -1.0 | 0.50 | 0.5+0.5-0.5+0.5 = 1.0 | 1.00 | 0.250 |
| 37.5 | [1,2,3] | [4,5,6] | -0.5 | 0.75 | 0.5-0.5+0.5 = 0.5 | 0.75 | 0.125 |
| 42.5 | [1,2,3,4] | [5,6] | 0.0 | 1.00 | -0.5+0.5 = 0.0 | 0.50 | 0.000 |
| 47.5 | [1,2,3,4,5] | [6] | -0.5 | 1.25 | 0.5 | 0.25 | 0.125 |

**Best threshold: 32.5 (Gain = 0.250)**

---

### **Step 4: Build Tree 1**

**Tree 1 Structure:**
```
Root: Age <= 32.5?
├── Yes: Label = 1 (samples 1, 2)
└── No: Label = 0 (samples 3, 4, 5, 6)
```

---

### **Step 5: Calculate Leaf Weights**

**Left node (Age <= 32.5):**
$$w_L = -\frac{G_L}{H_L+\lambda} = -\frac{-1.0}{0.50+1} = \frac{1.0}{1.5} = 0.67$$

**Right node (Age > 32.5):**
$$w_R = -\frac{G_R}{H_R+\lambda} = -\frac{1.0}{1.00+1} = -\frac{1.0}{2.0} = -0.5$$

**Hyperparameters:**
- $\lambda = 1.0$ (L2 regularization)
- $\gamma = 0.0$ (minimum gain to split)

---

### **Step 6: Update Model**

**Learning rate alpha = 0.1:**

$$F_1(x) = F_0(x) + a \cdot f_1(x)$$

$$f_1(x) = 0.67 \text{ if Age} \leq 32.5, \text{ otherwise } -0.5$$

**Updated predictions:**

| ID | Age | Income | Label | $F_0(x)$ | $f_1(x)$ | $F_1(x)$ | $\hat{y}$ |
|---|---|---|---|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0 | 0.67 | 0 + 0.1*0.67 = 0.067 | 0.517 |
| 2 | 30 | 50 | 1 | 0 | 0.67 | 0 + 0.1*0.67 = 0.067 | 0.517 |
| 3 | 35 | 40 | 0 | 0 | -0.5 | 0 + 0.1*(-0.5) = -0.05 | 0.488 |
| 4 | 40 | 60 | 0 | 0 | -0.5 | 0 + 0.1*(-0.5) = -0.05 | 0.488 |
| 5 | 45 | 70 | 1 | 0 | -0.5 | 0 + 0.1*(-0.5) = -0.05 | 0.488 |
| 6 | 50 | 80 | 0 | 0 | -0.5 | 0 + 0.1*(-0.5) = -0.05 | 0.488 |

---

## **Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} a \cdot f_m(x)$$

**Predictions:**
| ID | Age | Income | Label | $F_1(x)$ | $\hat{y}$ | Prediction |
|---|---|---|---|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.067 | 0.517 | 1 (>=0.5) Correct |
| 2 | 30 | 50 | 1 | 0.067 | 0.517 | 1 (>=0.5) Correct |
| 3 | 35 | 40 | 0 | -0.05 | 0.488 | 0 (<0.5) Correct |
| 4 | 40 | 60 | 0 | -0.05 | 0.488 | 0 (<0.5) Correct |
| 5 | 45 | 70 | 1 | -0.05 | 0.488 | 0 (<0.5) Wrong |
| 6 | 50 | 80 | 0 | -0.05 | 0.488 | 0 (<0.5) Correct |

**Accuracy: 5/6 = 83.3%**

---

## **Tóm tắt LightGBM Classification**

### **Quy trình hoàn chỉnh:**

**1. Khởi tạo**
- Tính F₀(x) = log(odds) của base probability

**2. GOSS Sampling**
- Chọn samples có gradient lớn
- Random chọn samples có gradient nhỏ

**3. Lặp lại cho m = 1 đến M:**
- **Tính gradients** gi = ŷi - yi
- **Tính hessians** hi = ŷi(1-ŷi)
- **GOSS sampling** để chọn samples quan trọng
- **EFB bundling** để gộp sparse features
- **Histogram-based splitting** để tìm thresholds
- **Calculate leaf weights** wj = -Gj/(Hj+λ)
- **Update model** Fm(x) = Fm-1(x) + α⋅fm(x)

**4. Final predictions**
- ŷi = 1/(1+e^(-FM(xi)))

### **Ưu điểm:**
- **GOSS sampling** giảm thời gian training
- **Histogram-based** splitting nhanh hơn
- **EFB bundling** giảm memory usage
- **Leaf-wise growth** hiệu quả hơn
- **Memory efficient** với large datasets
- **Parallel training** support

### **Nhược điểm:**
- **Overfitting** với small datasets
- **Complex** hyperparameter tuning
- **Less robust** than XGBoost
- **Sensitive** to hyperparameters

---

## **So sánh với các thuật toán khác**

| Đặc điểm | LightGBM | XGBoost | CatBoost | AdaBoost |
|---|---|---|---|:---:|
| **Method** | Boosting | Boosting | Boosting | Boosting |
| **Sampling** | GOSS | All | All | All |
| **Splitting** | Histogram | Exact | Exact | Exact |
| **Growth** | Leaf-wise | Level-wise | Level-wise | Level-wise |
| **Speed** | Fastest | Fast | Medium | Slow |
| **Memory** | Low | High | High | Low |
| **Accuracy** | Excellent | Excellent | Excellent | Good |

---

## **Khi nào sử dụng LightGBM**

### **Nên sử dụng khi:**
- **Large datasets** (>100K samples)
- **Cần tốc độ cao**
- **Memory hạn chế**
- **High-dimensional data**
- **Cần parallel training**

### **Không nên sử dụng khi:**
- **Small datasets** (<10K samples)
- **Cần model ổn định**
- **Có nhiều categorical features**
- **Cần fine-tune performance**
- **Overfitting** là vấn đề
