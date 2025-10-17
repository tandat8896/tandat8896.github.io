---
title: "Gradient Boosting Study - Thuật toán Gradient Boosting từ cơ bản đến nâng cao"
pubDatetime: 2025-09-21T17:00:00Z
featured: false
description: "Tìm hiểu chi tiết về Gradient Boosting, thuật toán boosting sử dụng gradient descent để tối ưu hóa weak learners"
tags: ["machine-learning", "gradient-boosting", "boosting", "ensemble", "algorithm"]
---

# Gradient Boosting Study

## Thuật toán Gradient Boosting

Gradient Boosting là một thuật toán ensemble learning sử dụng phương pháp **boosting** để kết hợp nhiều weak learners thành một strong learner. Khác với AdaBoost sử dụng trọng số, Gradient Boosting sử dụng **gradient descent** để tối ưu hóa từng weak learner.

---

## Công thức toán học của Gradient Boosting

### **Bước 1: Khởi tạo model**

**Model ban đầu:**
$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)$$

**Với Regression (MSE):**
$$F_0(x) = \frac{1}{N} \sum_{i=1}^{N} y_i$$

**Với Classification (Log Loss):**
$$F_0(x) = \log\left(\frac{p}{1-p}\right) \quad \text{where } p = \frac{\sum_{i=1}^{N} y_i}{N}$$

### **Bước 2: Vòng lặp cho m = 1 đến M**

#### **2a. Tính residuals (pseudo-residuals)**

**Regression:**
$$r_{i,m} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} = y_i - F_{m-1}(x_i)$$

**Classification:**
$$r_{i,m} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} = y_i - p_{m-1}(x_i)$$

#### **2b. Huấn luyện weak learner trên residuals**

$$h_m(x) = \arg\min_{h} \sum_{i=1}^{N} (r_{i,m} - h(x_i))^2$$

#### **2c. Tính learning rate (step size)**

$$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

#### **2d. Cập nhật model**

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

### **Bước 3: Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \gamma_m h_m(x)$$

---

## Ví Dụ Tính Tay - Gradient Boosting Regression

### **Dataset Regression**

| ID | Age (X) | Income (y) |
|:---:|:---:|:---:|
| 1 | 25 | 30 |
| 2 | 30 | 50 |
| 3 | 35 | 40 |
| 4 | 40 | 60 |
| 5 | 45 | 70 |
| 6 | 50 | 80 |

**Mục tiêu:** Dự đoán Income dựa trên Age với 3 iterations

---

### **Step 1: Initialization**

**F0(x) là giá trị tối ưu:**
$$F_0 = \frac{1}{N} \sum_{i=1}^{N} y_i = \frac{30 + 50 + 40 + 60 + 70 + 80}{6} = \frac{330}{6} = 55$$

**Initial predictions:**
| ID | Age | Income | F0(x) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 55 |
| 2 | 30 | 50 | 55 |
| 3 | 35 | 40 | 55 |
| 4 | 40 | 60 | 55 |
| 5 | 45 | 70 | 55 |
| 6 | 50 | 80 | 55 |

---

### **Step 2: Iteration 1 (m = 1)**

#### **Step 2a: Calculate Residuals**

**Residuals:**
$$r_{i,1} = y_i - F_0(x_i) = y_i - 55$$

| ID | Age | Income | F0(x) | Residuals (r1) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 55 | 30 - 55 = -25 |
| 2 | 30 | 50 | 55 | 50 - 55 = -5 |
| 3 | 35 | 40 | 55 | 40 - 55 = -15 |
| 4 | 40 | 60 | 55 | 60 - 55 = 5 |
| 5 | 45 | 70 | 55 | 70 - 55 = 15 |
| 6 | 50 | 80 | 55 | 80 - 55 = 25 |

#### **Step 2b: Train Weak Learner h1(x) trên Residuals**

**Tìm threshold tốt nhất cho Age trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [-25] | [-5,-15,5,15,25] | 0 |
| 32.5 | [1,2] | [3,4,5,6] | [-25,-5] | [-15,5,15,25] | 0 |
| 37.5 | [1,2,3] | [4,5,6] | [-25,-5,-15] | [5,15,25] | 0 |
| 42.5 | [1,2,3,4] | [5,6] | [-25,-5,-15,5] | [15,25] | 0 |
| 47.5 | [1,2,3,4,5] | [6] | [-25,-5,-15,5,15] | [25] | 0 |

**Tất cả thresholds đều có error = 0, chọn threshold = 37.5**

**h1(x) Structure:**
```
Root: Age <= 37.5?
├── Yes: Average residual = (-25-5-15)/3 = -15
└── No: Average residual = (5+15+25)/3 = 15
```

#### **Step 2c: Calculate Learning Rate γ1**

**Tìm γ1 để minimize loss:**
$$\gamma_1 = \arg\min_{\gamma} \sum_{i=1}^{6} (y_i - (F_0(x_i) + \gamma h_1(x_i)))^2$$

**Với γ = 0.1:**
| ID | Age | F0(x) | h1(x) | F0(x) + 0.1×h1(x) | y | Error² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 55 | -15 | 55 + 0.1×(-15) = 53.5 | 30 | (30-53.5)² = 552.25 |
| 2 | 30 | 55 | -15 | 55 + 0.1×(-15) = 53.5 | 50 | (50-53.5)² = 12.25 |
| 3 | 35 | 55 | -15 | 55 + 0.1×(-15) = 53.5 | 40 | (40-53.5)² = 182.25 |
| 4 | 40 | 55 | 15 | 55 + 0.1×15 = 56.5 | 60 | (60-56.5)² = 12.25 |
| 5 | 45 | 55 | 15 | 55 + 0.1×15 = 56.5 | 70 | (70-56.5)² = 182.25 |
| 6 | 50 | 55 | 15 | 55 + 0.1×15 = 56.5 | 80 | (80-56.5)² = 552.25 |

**Total Error = 552.25 + 12.25 + 182.25 + 12.25 + 182.25 + 552.25 = 1493.5**

**Với γ = 0.5:**
| ID | Age | F0(x) | h1(x) | F0(x) + 0.5×h1(x) | y | Error² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 55 | -15 | 55 + 0.5×(-15) = 47.5 | 30 | (30-47.5)² = 306.25 |
| 2 | 30 | 55 | -15 | 55 + 0.5×(-15) = 47.5 | 50 | (50-47.5)² = 6.25 |
| 3 | 35 | 55 | -15 | 55 + 0.5×(-15) = 47.5 | 40 | (40-47.5)² = 56.25 |
| 4 | 40 | 55 | 15 | 55 + 0.5×15 = 62.5 | 60 | (60-62.5)² = 6.25 |
| 5 | 45 | 55 | 15 | 55 + 0.5×15 = 62.5 | 70 | (70-62.5)² = 56.25 |
| 6 | 50 | 55 | 15 | 55 + 0.5×15 = 62.5 | 80 | (80-62.5)² = 306.25 |

**Total Error = 306.25 + 6.25 + 56.25 + 6.25 + 56.25 + 306.25 = 737.5**

**Với γ = 1.0:**
| ID | Age | F0(x) | h1(x) | F0(x) + 1.0×h1(x) | y | Error² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 55 | -15 | 55 + 1.0×(-15) = 40 | 30 | (30-40)² = 100 |
| 2 | 30 | 55 | -15 | 55 + 1.0×(-15) = 40 | 50 | (50-40)² = 100 |
| 3 | 35 | 55 | -15 | 55 + 1.0×(-15) = 40 | 40 | (40-40)² = 0 |
| 4 | 40 | 55 | 15 | 55 + 1.0×15 = 70 | 60 | (60-70)² = 100 |
| 5 | 45 | 55 | 15 | 55 + 1.0×15 = 70 | 70 | (70-70)² = 0 |
| 6 | 50 | 55 | 15 | 55 + 1.0×15 = 70 | 80 | (80-70)² = 100 |

**Total Error = 100 + 100 + 0 + 100 + 0 + 100 = 400**

**Best γ1 = 1.0 (Error = 400)**

#### **Step 2d: Update Model**

$$F_1(x) = F_0(x) + \gamma_1 h_1(x) = 55 + 1.0 \times h_1(x)$$

**F1(x) Structure:**
```
F1(x) = 55 + h1(x)
h1(x) = -15 if Age <= 37.5, else 15
```

**Updated predictions:**
| ID | Age | Income | F1(x) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 55 + (-15) = 40 |
| 2 | 30 | 50 | 55 + (-15) = 40 |
| 3 | 35 | 40 | 55 + (-15) = 40 |
| 4 | 40 | 60 | 55 + 15 = 70 |
| 5 | 45 | 70 | 55 + 15 = 70 |
| 6 | 50 | 80 | 55 + 15 = 70 |

---

### **Step 3: Iteration 2 (m = 2)**

#### **Step 3a: Calculate Residuals**

**Residuals:**
$$r_{i,2} = y_i - F_1(x_i)$$

| ID | Age | Income | F1(x) | Residuals (r2) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 40 | 30 - 40 = -10 |
| 2 | 30 | 50 | 40 | 50 - 40 = 10 |
| 3 | 35 | 40 | 40 | 40 - 40 = 0 |
| 4 | 40 | 60 | 70 | 60 - 70 = -10 |
| 5 | 45 | 70 | 70 | 70 - 70 = 0 |
| 6 | 50 | 80 | 70 | 80 - 70 = 10 |

#### **Step 3b: Train Weak Learner h2(x) trên Residuals**

**Tìm threshold tốt nhất cho Age trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [-10] | [10,0,-10,0,10] | 0 |
| 32.5 | [1,2] | [3,4,5,6] | [-10,10] | [0,-10,0,10] | 0 |
| 37.5 | [1,2,3] | [4,5,6] | [-10,10,0] | [-10,0,10] | 0 |
| 42.5 | [1,2,3,4] | [5,6] | [-10,10,0,-10] | [0,10] | 0 |
| 47.5 | [1,2,3,4,5] | [6] | [-10,10,0,-10,0] | [10] | 0 |

**Chọn threshold = 32.5**

**h2(x) Structure:**
```
Root: Age <= 32.5?
├── Yes: Average residual = (-10+10)/2 = 0
└── No: Average residual = (0-10+0+10)/4 = 0
```

**h2(x) = 0 cho tất cả samples (không cải thiện)**

#### **Step 3c: Calculate Learning Rate γ2**

**Với h2(x) = 0:**
$$\gamma_2 = 0 \quad \text{(không cần cập nhật)}$$

#### **Step 3d: Update Model**

$$F_2(x) = F_1(x) + \gamma_2 h_2(x) = F_1(x) + 0 = F_1(x)$$

**F2(x) = F1(x) (không thay đổi)**

---

### **Step 4: Iteration 3 (m = 3)**

#### **Step 4a: Calculate Residuals**

**Residuals:**
$$r_{i,3} = y_i - F_2(x_i) = y_i - F_1(x_i)$$

| ID | Age | Income | F2(x) | Residuals (r3) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 40 | 30 - 40 = -10 |
| 2 | 30 | 50 | 40 | 50 - 40 = 10 |
| 3 | 35 | 40 | 40 | 40 - 40 = 0 |
| 4 | 40 | 60 | 70 | 60 - 70 = -10 |
| 5 | 45 | 70 | 70 | 70 - 70 = 0 |
| 6 | 50 | 80 | 70 | 80 - 70 = 10 |

#### **Step 4b: Train Weak Learner h3(x) trên Residuals**

**Tìm threshold tốt nhất cho Age trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [-10] | [10,0,-10,0,10] | 0 |
| 32.5 | [1,2] | [3,4,5,6] | [-10,10] | [0,-10,0,10] | 0 |
| 37.5 | [1,2,3] | [4,5,6] | [-10,10,0] | [-10,0,10] | 0 |
| 42.5 | [1,2,3,4] | [5,6] | [-10,10,0,-10] | [0,10] | 0 |
| 47.5 | [1,2,3,4,5] | [6] | [-10,10,0,-10,0] | [10] | 0 |

**Chọn threshold = 42.5**

**h3(x) Structure:**
```
Root: Age <= 42.5?
├── Yes: Average residual = (-10+10+0-10)/4 = -2.5
└── No: Average residual = (0+10)/2 = 5
```

#### **Step 4c: Calculate Learning Rate γ3**

**Tìm γ3 để minimize loss:**
$$\gamma_3 = \arg\min_{\gamma} \sum_{i=1}^{6} (y_i - (F_2(x_i) + \gamma h_3(x_i)))^2$$

**Với γ = 0.5:**
| ID | Age | F2(x) | h3(x) | F2(x) + 0.5×h3(x) | y | Error² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 40 | -2.5 | 40 + 0.5×(-2.5) = 38.75 | 30 | (30-38.75)² = 76.56 |
| 2 | 30 | 40 | -2.5 | 40 + 0.5×(-2.5) = 38.75 | 50 | (50-38.75)² = 126.56 |
| 3 | 35 | 40 | -2.5 | 40 + 0.5×(-2.5) = 38.75 | 40 | (40-38.75)² = 1.56 |
| 4 | 40 | 70 | -2.5 | 70 + 0.5×(-2.5) = 68.75 | 60 | (60-68.75)² = 76.56 |
| 5 | 45 | 70 | 5 | 70 + 0.5×5 = 72.5 | 70 | (70-72.5)² = 6.25 |
| 6 | 50 | 70 | 5 | 70 + 0.5×5 = 72.5 | 80 | (80-72.5)² = 56.25 |

**Total Error = 76.56 + 126.56 + 1.56 + 76.56 + 6.25 + 56.25 = 343.74**

**Với γ = 1.0:**
| ID | Age | F2(x) | h3(x) | F2(x) + 1.0×h3(x) | y | Error² |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 40 | -2.5 | 40 + 1.0×(-2.5) = 37.5 | 30 | (30-37.5)² = 56.25 |
| 2 | 30 | 40 | -2.5 | 40 + 1.0×(-2.5) = 37.5 | 50 | (50-37.5)² = 156.25 |
| 3 | 35 | 40 | -2.5 | 40 + 1.0×(-2.5) = 37.5 | 40 | (40-37.5)² = 6.25 |
| 4 | 40 | 70 | -2.5 | 70 + 1.0×(-2.5) = 67.5 | 60 | (60-67.5)² = 56.25 |
| 5 | 45 | 70 | 5 | 70 + 1.0×5 = 75 | 70 | (70-75)² = 25 |
| 6 | 50 | 70 | 5 | 70 + 1.0×5 = 75 | 80 | (80-75)² = 25 |

**Total Error = 56.25 + 156.25 + 6.25 + 56.25 + 25 + 25 = 325**

**Best γ3 = 1.0 (Error = 325)**

#### **Step 4d: Update Model**

$$F_3(x) = F_2(x) + \gamma_3 h_3(x) = F_2(x) + 1.0 \times h_3(x)$$

**F3(x) Structure:**
```
F3(x) = F2(x) + h3(x)
F2(x) = 55 + h1(x) where h1(x) = -15 if Age <= 37.5, else 15
h3(x) = -2.5 if Age <= 42.5, else 5
```

**Final predictions:**
| ID | Age | Income | F3(x) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 40 + (-2.5) = 37.5 |
| 2 | 30 | 50 | 40 + (-2.5) = 37.5 |
| 3 | 35 | 40 | 40 + (-2.5) = 37.5 |
| 4 | 40 | 60 | 70 + (-2.5) = 67.5 |
| 5 | 45 | 70 | 70 + 5 = 75 |
| 6 | 50 | 80 | 70 + 5 = 75 |

---

## **Tóm tắt Gradient Boosting**

### **Quy trình hoàn chỉnh:**

1. **Khởi tạo** F0(x) = mean(y)
2. **Lặp lại** cho m = 1 đến M:
   - **Tính residuals** rm = y - Fm-1(x)
   - **Train weak learner** hm(x) trên residuals
   - **Tìm learning rate** γm để minimize loss
   - **Cập nhật model** Fm(x) = Fm-1(x) + γm hm(x)
3. **Final model** FM(x) = F0(x) + Σ γm hm(x)

### **Ưu điểm:**
- **Không cần trọng số** như AdaBoost
- **Sử dụng gradient descent** để tối ưu
- **Linh hoạt** với nhiều loại loss function
- **Hiệu quả** với dữ liệu phức tạp

### **Nhược điểm:**
- **Sensitive** với noise và outliers
- **Có thể overfitting** nếu quá nhiều iterations
- **Chậm hơn** Random Forest
- **Cần tuning** learning rate và số iterations

---

## **So sánh với các thuật toán khác**

| Đặc điểm | Gradient Boosting | AdaBoost | Random Forest |
|:---:|:---:|:---:|:---:|
| **Method** | Boosting | Boosting | Bagging |
| **Training** | Sequential | Sequential | Parallel |
| **Weights** | Learning rate | Sample weights | Equal weights |
| **Optimization** | Gradient descent | Weighted resampling | Bootstrap sampling |
| **Speed** | Slow | Medium | Fast |
| **Overfitting** | High risk | Medium risk | Low risk |

---

## **Hyperparameters quan trọng**

- **`n_estimators`**: Số lượng weak learners (100-1000)
- **`learning_rate`**: Tốc độ học (0.01-0.3)
- **`max_depth`**: Độ sâu tối đa của weak learners (3-10)
- **`min_samples_split`**: Số samples tối thiểu để split (2-10)
- **`min_samples_leaf`**: Số samples tối thiểu ở leaf (1-5)
- **`subsample`**: Tỷ lệ samples sử dụng (0.5-1.0)

---

## **Khi nào nên sử dụng Gradient Boosting**

✅ **Nên dùng khi:**
- Cần accuracy cao nhất
- Dataset vừa phải (<100K samples)
- Có thời gian tuning hyperparameters
- Dữ liệu ít noise

❌ **Không nên dùng khi:**
- Dataset rất lớn (>1M samples)
- Cần tốc độ cao
- Dữ liệu nhiều noise
- Cần interpretability cao
