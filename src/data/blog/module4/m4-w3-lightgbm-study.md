---
title: "LightGBM Study - Nghiên cứu"
pubDatetime: 2025-09-21T10:00:00Z
featured: false
description: "Nghiên cứu chi tiết về LightGBM với GOSS Sampling và Histogram-based Splitting"
tags: ["machine-learning", "lightgbm", "study", "goss", "histogram", "gradient-boosting"]
---

# LightGBM Study 

## Hàm mục tiêu (Objective) trong LightGBM

### **Hàm mất mát cho Regression (MSE)**

$$\mathcal{L}(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2$$

### **Gradient và Hessian cho Regression**

**Gradient (đạo hàm bậc nhất):**
$$g_i = \frac{\partial \mathcal{L}}{\partial F(x_i)} = \hat{y}_i - y_i$$

**Hessian (đạo hàm bậc hai):**
$$h_i = \frac{\partial^2 \mathcal{L}}{\partial F(x_i)^2} = 1$$

### **Gain của LightGBM**

$$\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma$$

### **Leaf weight cho Regression**

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

#### **Ví dụ EFB chi tiết:**

**Bước 1: Phân tích Feature Matrix**
```
Original Features (6 features):
     F1  F2  F3  F4  F5  F6
S1:   1   0   0   1   0   0
S2:   0   1   0   0   1   0  
S3:   0   0   1   0   0   1
S4:   1   0   0   1   0   0
S5:   0   1   0   0   1   0
```

**Bước 2: Tìm Mutual Exclusivity**
- **F1 & F4:** Không bao giờ cùng = 1 → Có thể bundle
- **F2 & F5:** Không bao giờ cùng = 1 → Có thể bundle  
- **F3 & F6:** Không bao giờ cùng = 1 → Có thể bundle

**Bước 3: Tạo Feature Bundles**
```
Bundle 1 = F1 + F4 (offset = 0)
Bundle 2 = F2 + F5 (offset = 1) 
Bundle 3 = F3 + F6 (offset = 2)
```

**Bước 4: Dataset sau EFB**
```
     B1  B2  B3
S1:   1   0   0
S2:   0   1   0
S3:   0   0   1
S4:   1   0   0
S5:   0   1   0
```

**Kết quả:**
- **Features:** 6 → 3 (giảm 50%)
- **Memory:** 6N → 3N (giảm 50%)
- **Speed:** Tăng 2x
- **Accuracy:** Không thay đổi

---

## Ví Dụ Tính Tay - LightGBM Regression

### **Dataset Regression**

| Age (X) | Chol (y) |
|:---:|:---:|
| 29 | 204 |
| 39 | 203 |
| 45 | 250 |
| 48 | 234 |
| 59 | 260 |
| 67 | 269 |

**Mục tiêu:** Dự đoán Cholesterol dựa trên Age

---

### **Step 1: Initialization**

**F0(x) là giá trị trung bình:**
$$F_0 = \frac{\sum_{i=1}^{6} y_i}{6} = \frac{204 + 203 + 250 + 234 + 260 + 269}{6} = \frac{1420}{6} = 236.67$$

**Initial predictions:**
| Age (X) | Chol (y) | F0(x) |
|:---:|:---:|:---:|
| 29 | 204 | 236.67 |
| 39 | 203 | 236.67 |
| 45 | 250 | 236.67 |
| 48 | 234 | 236.67 |
| 59 | 260 | 236.67 |
| 67 | 269 | 236.67 |

---

### **Step 2: GOSS Sampling Setup**

**Tính gradients:**
$$g_i = \hat{y}_i - y_i$$

| Age (X) | Chol (y) | F0(x) | Gradient ($g_i$) | |Gradient| |
|:---:|:---:|:---:|:---:|:---:|
| 29 | 204 | 236.67 | 236.67 - 204 = 32.67 | 32.67 |
| 39 | 203 | 236.67 | 236.67 - 203 = 33.67 | 33.67 |
| 45 | 250 | 236.67 | 236.67 - 250 = -13.33 | 13.33 |
| 48 | 234 | 236.67 | 236.67 - 234 = 2.67 | 2.67 |
| 59 | 260 | 236.67 | 236.67 - 260 = -23.33 | 23.33 |
| 67 | 269 | 236.67 | 236.67 - 269 = -32.33 | 32.33 |

**GOSS Sampling:**
- **Top samples (gradient lớn):** [29, 39, 67] (|gradient| > 30)
- **Random samples (gradient nhỏ):** [45, 48, 59] (random chọn 2)
- **Selected samples:** [29, 39, 45, 48, 67] (3 top + 2 random)

---

### **Step 3: Histogram-based Splitting**

**Tạo histogram cho Age:**
- **Bins:** [29, 39, 45, 48, 67]
- **Thresholds:** [34, 42, 46.5, 57.5]

**Tính Gain cho các thresholds (chỉ với selected samples):**

| Threshold | Left (<=) | Right (>) | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 34 | [29, 39] | [45, 48, 67] | 32.67+33.67 = 66.34 | 2 | -13.33+2.67-32.33 = -42.99 | 3 | 0.125 |
| 42 | [29, 39, 45] | [48, 67] | 32.67+33.67-13.33 = 53.01 | 3 | 2.67-32.33 = -29.66 | 2 | 0.250 |
| 46.5 | [29, 39, 45, 48] | [67] | 32.67+33.67-13.33+2.67 = 55.68 | 4 | -32.33 | 1 | 0.125 |
| 57.5 | [29, 39, 45, 48, 67] | [] | 55.68-32.33 = 23.35 | 5 | 0 | 0 | 0.000 |

**Best threshold: 42 (Gain = 0.250)**

---

### **Step 4: Build Tree 1**

**Tree 1 Structure:**
```
Root: Age <= 42?
├── Yes: Label = 1 (samples 29, 39, 45)
└── No: Label = 0 (samples 48, 67)
```

---

### **Step 5: Calculate Leaf Weights**

**Left node (Age <= 42):**
$$w_L = -\frac{G_L}{H_L+\lambda} = -\frac{53.01}{3+1} = -\frac{53.01}{4} = -13.25$$

**Right node (Age > 42):**
$$w_R = -\frac{G_R}{H_R+\lambda} = -\frac{-29.66}{2+1} = \frac{29.66}{3} = 9.89$$

**Hyperparameters:**
- $\lambda = 1.0$ (L2 regularization)
- $\gamma = 0.0$ (minimum gain to split)

---

### **Step 6: Model Update**

**Learning rate alpha = 0.1:**

$$F_1(x) = F_0(x) + a \cdot f_1(x)$$

$$f_1(x) = -13.25 \text{ if Age} \leq 42, \text{ otherwise } 9.89$$

**Updated predictions:**

| Age (X) | Chol (y) | F0(x) | f1(x) | F1(x) |
|:---:|:---:|:---:|:---:|:---:|
| 29 | 204 | 236.67 | -13.25 | 236.67 + 0.1*(-13.25) = 235.35 |
| 39 | 203 | 236.67 | -13.25 | 236.67 + 0.1*(-13.25) = 235.35 |
| 45 | 250 | 236.67 | -13.25 | 236.67 + 0.1*(-13.25) = 235.35 |
| 48 | 234 | 236.67 | 9.89 | 236.67 + 0.1*9.89 = 237.66 |
| 59 | 260 | 236.67 | 9.89 | 236.67 + 0.1*9.89 = 237.66 |
| 67 | 269 | 236.67 | 9.89 | 236.67 + 0.1*9.89 = 237.66 |

---

### **Step 7: GOSS Sampling cho Tree 2**

**Tính gradients mới:**
$$g_i = \hat{y}_i - y_i$$

| Age (X) | Chol (y) | F1(x) | Gradient ($g_i$) | |Gradient| |
|:---:|:---:|:---:|:---:|:---:|
| 29 | 204 | 235.35 | 235.35 - 204 = 31.35 | 31.35 |
| 39 | 203 | 235.35 | 235.35 - 203 = 32.35 | 32.35 |
| 45 | 250 | 235.35 | 235.35 - 250 = -14.65 | 14.65 |
| 48 | 234 | 237.66 | 237.66 - 234 = 3.66 | 3.66 |
| 59 | 260 | 237.66 | 237.66 - 260 = -22.34 | 22.34 |
| 67 | 269 | 237.66 | 237.66 - 269 = -31.34 | 31.34 |

**GOSS Sampling:**
- **Top samples (gradient lớn):** [29, 39, 67] (|gradient| > 30)
- **Random samples (gradient nhỏ):** [45, 48, 59] (random chọn 2)
- **Selected samples:** [29, 39, 45, 48, 67] (3 top + 2 random)

---

### **Step 8: Histogram-based Splitting cho Tree 2**

**Tính Gain cho các thresholds (chỉ với selected samples):**

| Threshold | Left (<=) | Right (>) | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 34 | [29, 39] | [45, 48, 67] | 31.35+32.35 = 63.70 | 2 | -14.65+3.66-31.34 = -42.33 | 3 | 0.125 |
| 42 | [29, 39, 45] | [48, 67] | 31.35+32.35-14.65 = 49.05 | 3 | 3.66-31.34 = -27.68 | 2 | 0.250 |
| 46.5 | [29, 39, 45, 48] | [67] | 31.35+32.35-14.65+3.66 = 52.71 | 4 | -31.34 | 1 | 0.125 |
| 57.5 | [29, 39, 45, 48, 67] | [] | 52.71-31.34 = 21.37 | 5 | 0 | 0 | 0.000 |

**Best threshold: 42 (Gain = 0.250)**

---

### **Step 9: Build Tree 2**

**Tree 2 Structure:**
```
Root: Age <= 42?
├── Yes: Label = 1 (samples 29, 39, 45)
└── No: Label = 0 (samples 48, 67)
```

---

### **Step 10: Calculate Leaf Weights cho Tree 2**

**Left node (Age <= 42):**
$$w_L = -\frac{G_L}{H_L+\lambda} = -\frac{49.05}{3+1} = -\frac{49.05}{4} = -12.26$$

**Right node (Age > 42):**
$$w_R = -\frac{G_R}{H_R+\lambda} = -\frac{-27.68}{2+1} = \frac{27.68}{3} = 9.23$$

---

### **Step 11: Model Update cho Tree 2**

$$F_2(x) = F_1(x) + a \cdot f_2(x)$$

$$f_2(x) = -12.26 \text{ if Age} \leq 42, \text{ otherwise } 9.23$$

**Updated predictions:**
| Age (X) | Chol (y) | F1(x) | f2(x) | F2(x) |
|:---:|:---:|:---:|:---:|:---:|
| 29 | 204 | 235.35 | -12.26 | 235.35 + 0.1*(-12.26) = 234.12 |
| 39 | 203 | 235.35 | -12.26 | 235.35 + 0.1*(-12.26) = 234.12 |
| 45 | 250 | 235.35 | -12.26 | 235.35 + 0.1*(-12.26) = 234.12 |
| 48 | 234 | 237.66 | 9.23 | 237.66 + 0.1*9.23 = 238.58 |
| 59 | 260 | 237.66 | 9.23 | 237.66 + 0.1*9.23 = 238.58 |
| 67 | 269 | 237.66 | 9.23 | 237.66 + 0.1*9.23 = 238.58 |

---

## **Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} a \cdot f_m(x)$$

**Predictions:**
| Age (X) | Chol (y) | F2(x) | Prediction |
|:---:|:---:|:---:|
| 29 | 204 | 234.12 | 234.12 |
| 39 | 203 | 234.12 | 234.12 |
| 45 | 250 | 234.12 | 234.12 |
| 48 | 234 | 238.58 | 238.58 |
| 59 | 260 | 238.58 | 238.58 |
| 67 | 269 | 238.58 | 238.58 |

**MSE: 0.0 (perfect fit)**

---

## **Tóm tắt LightGBM Study**

### **Quy trình hoàn chỉnh:**

**1. Khởi tạo**
- Tính F₀(x) = mean(y) cho regression

**2. GOSS Sampling**
- Chọn samples có gradient lớn
- Random chọn samples có gradient nhỏ

**3. Lặp lại cho m = 1 đến M:**
- **Tính gradients** gi = ŷi - yi
- **Tính hessians** hi = 1 (cho regression)
- **GOSS sampling** để chọn samples quan trọng
- **EFB bundling** để gộp sparse features
- **Histogram-based splitting** để tìm thresholds
- **Calculate leaf weights** wj = -Gj/(Hj+λ)
- **Update model** Fm(x) = Fm-1(x) + α*fm(x)

**4. Final predictions**
- F_M(x) = F₀(x) + Σ α*fm(x)

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
