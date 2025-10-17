---
title: "CatBoost Classification - Thuật toán Gradient Boosting với Categorical Features"
pubDatetime: 2025-09-22T10:00:00Z
featured: false
description: "Tìm hiểu chi tiết về CatBoost cho Classification với Ordered Boosting và Categorical Features Handling"
tags: ["machine-learning", "catboost", "classification", "gradient-boosting", "categorical-features"]
---

# CatBoost Classification

## Hàm mục tiêu (Objective) trong CatBoost Classification

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

### **Gain của CatBoost Classification**

$$\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma$$

### **Leaf weight cho Classification**

$$w_j^* = -\frac{G_j}{H_j+\lambda}$$

---

## Đặc điểm nổi bật của CatBoost

### **1. Ordered Boosting**
- **Vấn đề:** Target leakage trong gradient boosting
- **Giải pháp:** Sử dụng permutation để tránh overfitting
- **Cơ chế:** Mỗi tree chỉ sử dụng samples có index nhỏ hơn để tính gradient

### **2. Categorical Features Handling**
- **One-hot encoding** tự động
- **Target encoding** với regularization
- **Categorical features** được xử lý trực tiếp

### **3. Symmetric Trees**
- **Cấu trúc:** Tất cả nodes ở cùng level có cùng splitting condition
- **Ưu điểm:** Giảm overfitting, tăng tốc độ prediction
- **Ứng dụng:** Đặc biệt hiệu quả với categorical features

---

## Ví Dụ Tính Tay - CatBoost Classification

### **Dataset Classification với Categorical Features**

| ID | Age (X1) | City (X2) | Income (X3) | Label (y) |
|---|---|---|---|---|
| 1 | 25 | A | 30 | 1 |
| 2 | 30 | B | 50 | 1 |
| 3 | 35 | A | 40 | 0 |
| 4 | 40 | C | 60 | 0 |
| 5 | 45 | B | 70 | 1 |
| 6 | 50 | A | 80 | 0 |

**Mục tiêu:** Dự đoán label dựa trên Age, City (categorical), và Income

---

### **Step 1: Initialization**

**F0(x) là giá trị tối ưu cho log(odds):**
$$F_0 = \log\left(\frac{p}{1-p}\right) \quad \text{Where } p = \frac{\sum_{i=1}^{6} y_i}{6} = \frac{3}{6} = 0.5$$

$$F_0 = \log\left(\frac{0.5}{1-0.5}\right) = \log(1) = 0$$

**Initial predictions (probabilities):**
$$\hat{y}_i = \frac{1}{1 + e^{-F_0}} = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = 0.5$$

| ID | Age | City | Income | Label | F0(x) | y_hat |
|---|---|---|---|:---:|:---:|:---:|
| 1 | 25 | A | 30 | 1 | 0 | 0.5 |
| 2 | 30 | B | 50 | 1 | 0 | 0.5 |
| 3 | 35 | A | 40 | 0 | 0 | 0.5 |
| 4 | 40 | C | 60 | 0 | 0 | 0.5 |
| 5 | 45 | B | 70 | 1 | 0 | 0.5 |
| 6 | 50 | A | 80 | 0 | 0 | 0.5 |

---

### **Step 2: Ordered Boosting Setup**

**Permutation:** [3, 1, 5, 2, 4, 6] (random permutation)

**Ordered Boosting Rule:** Tree t chỉ sử dụng samples có index < t trong permutation

---

### **Step 3: Iteration 1 - Gradients & Hessians**

**Tính gradients:**
$$g_i = \hat{y}_i - y_i$$

| ID | Age | City | Income | Label | y_hat | Gradient ($g_i$) |
|---|---|---|---|:---:|:---:|:---:|
| 1 | 25 | A | 30 | 1 | 0.5 | 0.5 - 1 = -0.5 |
| 2 | 30 | B | 50 | 1 | 0.5 | 0.5 - 1 = -0.5 |
| 3 | 35 | A | 40 | 0 | 0.5 | 0.5 - 0 = 0.5 |
| 4 | 40 | C | 60 | 0 | 0.5 | 0.5 - 0 = 0.5 |
| 5 | 45 | B | 70 | 1 | 0.5 | 0.5 - 1 = -0.5 |
| 6 | 50 | A | 80 | 0 | 0.5 | 0.5 - 0 = 0.5 |

**Tính hessians:**
$$
h_i = \hat{y}_i(1-\hat{y}_i) = 0.5 \cdot (1-0.5) = 0.25
$$

| ID | Age | City | Income | Label | y_hat | Hessian ($h_i$) |
|----|-----|------|--------|:-----:|:-----:|:---------------:|
| 1  | 25  | A    | 30     |   1   |  0.5  |      0.25       |
| 2  | 30  | B    | 50     |   1   |  0.5  |      0.25       |
| 3  | 35  | A    | 40     |   0   |  0.5  |      0.25       |
| 4  | 40  | C    | 60     |   0   |  0.5  |      0.25       |
| 5  | 45  | B    | 70     |   1   |  0.5  |      0.25       |
| 6  | 50  | A    | 80     |   0   |  0.5  |      0.25       |


---

### **Step 4: Build Tree 1 - Ordered Boosting**

**Permutation:** [3, 1, 5, 2, 4, 6]

**Tree 1 chỉ sử dụng samples có index < 1 trong permutation:**
- **Available samples:** Không có (index 0 không tồn tại)
- **Fallback:** Sử dụng tất cả samples với regularization

**Tìm threshold tốt nhất cho Age:**

| Threshold | Left (<=) | Right (>) | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|---|---|---|---|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | -0.5 | 0.25 | -0.5+0.5+0.5-0.5+0.5 = 0.5 | 1.25 | 0.125 |
| 32.5 | [1,2] | [3,4,5,6] | -1.0 | 0.50 | 0.5+0.5-0.5+0.5 = 1.0 | 1.00 | 0.250 |
| 37.5 | [1,2,3] | [4,5,6] | -0.5 | 0.75 | 0.5-0.5+0.5 = 0.5 | 0.75 | 0.125 |
| 42.5 | [1,2,3,4] | [5,6] | 0.0 | 1.00 | -0.5+0.5 = 0.0 | 0.50 | 0.000 |
| 47.5 | [1,2,3,4,5] | [6] | -0.5 | 1.25 | 0.5 | 0.25 | 0.125 |

**Best threshold: 32.5 (Gain = 0.250)**

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

| ID | Age | City | Income | Label | F0(x) | f1(x) | F1(x) | y_hat |
|----|-----|------|--------|-------|-------|-------|-------|-------|
| 1  | 25  | A    | 30     | 1     | 0     | 0.67  | 0.067 | 0.517 |
| 2  | 30  | B    | 50     | 1     | 0     | 0.67  | 0.067 | 0.517 |
| 3  | 35  | A    | 40     | 0     | 0     | -0.5  | -0.05 | 0.488 |
| 4  | 40  | C    | 60     | 0     | 0     | -0.5  | -0.05 | 0.488 |
| 5  | 45  | B    | 70     | 1     | 0     | -0.5  | -0.05 | 0.488 |
| 6  | 50  | A    | 80     | 0     | 0     | -0.5  | -0.05 | 0.488 |

**Chi tiết tính toán:**
- ID 1: F1(x) = 0 + 0.1 * 0.67 = 0.067 → y_hat = 1/(1 + e^(-0.067)) = 0.517
- ID 3: F1(x) = 0 + 0.1 * (-0.5) = -0.05 → y_hat = 1/(1 + e^(0.05)) = 0.488

---


## **Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \alpha \cdot f_m(x)$$

**Predictions:**
| ID | Age | City | Income | Label | F1(x) | y_hat | Prediction |
|----|-----|------|--------|:-----:|:-----:|:-----:|:----------|
| 1  | 25  | A    | 30     |   1   | 0.067 | 0.517 | 1 (>=0.5) ✅ Correct |
| 2  | 30  | B    | 50     |   1   | 0.067 | 0.517 | 1 (>=0.5) ✅ Correct |
| 3  | 35  | A    | 40     |   0   | -0.05 | 0.488 | 0 (<0.5) ✅ Correct |
| 4  | 40  | C    | 60     |   0   | -0.05 | 0.488 | 0 (<0.5) ✅ Correct |
| 5  | 45  | B    | 70     |   1   | -0.05 | 0.488 | 0 (<0.5) ❌ Wrong   |
| 6  | 50  | A    | 80     |   0   | -0.05 | 0.488 | 0 (<0.5) ✅ Correct |


**Accuracy: 5/6 = 83.3%**

---

## **Tóm tắt CatBoost Classification**

### **Quy trình hoàn chỉnh:**

**1. Khởi tạo**
- Tính F₀(x) = log(odds) của base probability

**2. Tạo permutation**
- Tạo random permutation cho Ordered Boosting

**3. Lặp lại cho m = 1 đến M:**
- **Tính gradients:** $g_i = \hat{y}_i - y_i$
- **Tính hessians:** $h_i = \hat{y}_i(1-\hat{y}_i)$
- **Ordered Boosting:** Chỉ sử dụng samples có index < m
- **Build symmetric tree:** Với categorical features
- **Calculate leaf weights:** $w_j = -\frac{G_j}{H_j+\lambda}$
- **Update model:** $F_m(x) = F_{m-1}(x) + \alpha \cdot f_m(x)$

**4. Final predictions**
- $\hat{y}_i = \frac{1}{1+e^{-F_M(x_i)}}$

### **Ưu điểm:**
- **Ordered Boosting** tránh target leakage
- **Categorical features** xử lý tự động
- **Symmetric trees** giảm overfitting
- **No hyperparameter tuning** cần thiết
- **Robust** với missing values

### **Nhược điểm:**
- **Slower** than LightGBM và XGBoost
- **Memory intensive** với large datasets
- **Less flexible** than other boosting methods
- **Complex** internal implementation

---

## **So sánh với các thuật toán khác**

| Đặc điểm | CatBoost | XGBoost | LightGBM | AdaBoost |
|---|---|---|---|:---:|
| **Method** | Boosting | Boosting | Boosting | Boosting |
| **Categorical** | Native | Manual | Manual | Manual |
| **Overfitting** | Low | Medium | Medium | High |
| **Speed** | Medium | Fast | Fastest | Slow |
| **Memory** | High | High | Low | Low |
| **Tuning** | Minimal | Complex | Complex | Simple |
| **Accuracy** | Excellent | Excellent | Excellent | Good |

---

## **Khi nào sử dụng CatBoost**

### **Nên sử dụng khi:**
- **Có nhiều categorical features**
- **Cần tránh overfitting**
- **Không muốn tune hyperparameters**
- **Có missing values**
- **Cần model ổn định**

### **Không nên sử dụng khi:**
- **Cần tốc độ cao**
- **Dataset rất lớn**
- **Cần fine-tune performance**
- **Có ít categorical features**
- **Memory hạn chế**
