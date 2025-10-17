---
title: "AdaBoost Regression - Thuật toán Adaptive Boosting cho Regression"
pubDatetime: 2025-09-21T22:00:00Z
featured: false
description: "Tìm hiểu chi tiết về AdaBoost cho Regression, từ weighted sampling đến prediction aggregation"
tags: ["machine-learning", "adaboost", "regression", "boosting", "ensemble"]
---

# AdaBoost Regression

> **📚 Repo tham khảo:** [https://github.com/tandat8896/ml-from-the-scartch/tree/master/adaboost](https://github.com/tandat8896/ml-from-the-scartch/tree/master/adaboost)

## Thuật toán AdaBoost cho Regression

AdaBoost cho Regression sử dụng **Adaptive Boosting** để kết hợp nhiều weak learners thành một strong learner. Khác với classification, regression sử dụng **weighted median** thay vì **weighted majority vote** để kết hợp predictions.

---

## Công thức toán học của AdaBoost Regression

### **Bước 1: Initialize Weights**

Khởi tạo trọng số cho tất cả samples:
$$w_i^{(1)} = \frac{1}{N}, \quad i = 1, 2, \ldots, N$$

### **Bước 2: For m = 1 to M (M weak learners)**

#### **2.1: Train Weak Learner**
$$h_m(x) = \text{TrainWeakLearner}(D, w^{(m)})$$

#### **2.2: Calculate Normalized Absolute Error**
$$e_i^{(m)} = \frac{|y_i - h_m(x_i)|}{\max_j |y_j - h_m(x_j)|} \in [0, 1]$$

#### **2.3: Calculate Weighted Error**
$$\varepsilon_m = \sum_{i=1}^{N} w_i^{(m)} e_i^{(m)}$$

#### **2.4: Calculate Beta**
$$\beta_m = \frac{\varepsilon_m}{1 - \varepsilon_m}$$

#### **2.5: Calculate Alpha (Weight)**
$$\alpha_m = \ln\left(\frac{1}{\beta_m}\right)$$

#### **2.6: Update Sample Weights**
$$w_i^{(m+1)} = \frac{w_i^{(m)} \beta_m^{1-e_i^{(m)}}}{\sum_{j=1}^{N} w_j^{(m)} \beta_m^{1-e_j^{(m)}}}$$

### **Bước 3: Final Prediction (Weighted Median)**
$$\hat{y}(x) = \text{weighted\_median}\{h_m(x)\}_{m=1}^M; \{\alpha_m\}_{m=1}^M$$

---

## Ví Dụ Tính Tay - AdaBoost Regression

### **Dataset Regression**

| ID | Age (X1) | Income (X2) | Price (y) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 100 |
| 2 | 30 | 50 | 150 |
| 3 | 35 | 40 | 120 |
| 4 | 40 | 60 | 180 |
| 5 | 45 | 70 | 200 |
| 6 | 50 | 80 | 220 |

**Mục tiêu:** Dự đoán Price dựa trên Age và Income với 3 weak learners

---

### **Step 1: Initialize Weights**

$$w_i^{(1)} = \frac{1}{6} = 0.167, \quad i = 1, 2, \ldots, 6$$

| ID | Age | Income | Price | Weight |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 100 | 0.167 |
| 2 | 30 | 50 | 150 | 0.167 |
| 3 | 35 | 40 | 120 | 0.167 |
| 4 | 40 | 60 | 180 | 0.167 |
| 5 | 45 | 70 | 200 | 0.167 |
| 6 | 50 | 80 | 220 | 0.167 |

---

### **Step 2: Loop 1 (m = 1)**

#### **2.1: Train Weak Learner h1(x)**

**Tìm best split cho Age:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | Weighted MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [100] | [150,120,180,200,220] | 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 = 0 |
| 32.5 | [1,2] | [3,4,5,6] | [100,150] | [120,180,200,220] | 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 = 0 |
| 37.5 | [1,2,3] | [4,5,6] | [100,150,120] | [180,200,220] | 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 = 0 |
| 42.5 | [1,2,3,4] | [5,6] | [100,150,120,180] | [200,220] | 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 = 0 |
| 47.5 | [1,2,3,4,5] | [6] | [100,150,120,180,200] | [220] | 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 + 0.167×0 = 0 |

**Tất cả thresholds đều có weighted MSE = 0, chọn threshold = 32.5**

**h1(x) Structure:**
```
Root: Age <= 32.5?
├── Yes: Price = 125 (samples 1,2)
└── No: Price = 200 (samples 3,4,5,6)
```

**Predictions:**
$$h_1(x) = \{125, 125, 200, 200, 200, 200\}$$

#### **2.2: Calculate Normalized Absolute Error**

**Absolute Errors:**
$$|y - h_1| = \{|100-125|, |150-125|, |120-200|, |180-200|, |200-200|, |220-200|\}$$
$$= \{25, 25, 80, 20, 0, 20\}$$

**Maximum Absolute Error:**
$$\max_j |y_j - h_1(x_j)| = 80$$

**Normalized Errors:**
\( e_i^{(1)} = \frac{|y_i - h_1(x_i)|}{80} \)

| ID | Age | Income | Price | h1(x) | \|y−h1\| | e_i^(1) |
|:--:|:---:|:------:|:-----:|:-----:|:-----:|:--------:|
|  1 |  25 |   30   |  100  |  125  |   25  |  0.3125  |
|  2 |  30 |   50   |  150  |  125  |   25  |  0.3125  |
|  3 |  35 |   40   |  120  |  200  |   80  |  1.0000  |
|  4 |  40 |   60   |  180  |  200  |   20  |  0.2500  |
|  5 |  45 |   70   |  200  |  200  |    0  |  0.0000  |
|  6 |  50 |   80   |  220  |  200  |   20  |  0.2500  |

#### **2.3: Calculate Weighted Error**

$$\varepsilon_1 = \sum_{i=1}^{6} w_i^{(1)} e_i^{(1)} = 0.167 \times 0.3125 + 0.167 \times 0.3125 + 0.167 \times 1.0000 + 0.167 \times 0.2500 + 0.167 \times 0.0000 + 0.167 \times 0.2500$$

$$\varepsilon_1 = 0.167 \times (0.3125 + 0.3125 + 1.0000 + 0.2500 + 0.0000 + 0.2500) = 0.167 \times 2.125 = 0.354$$

#### **2.4: Calculate Beta**

$$\beta_1 = \frac{\varepsilon_1}{1 - \varepsilon_1} = \frac{0.354}{1 - 0.354} = \frac{0.354}{0.646} = 0.548$$

#### **2.5: Calculate Alpha**

$$\alpha_1 = \ln\left(\frac{1}{\beta_1}\right) = \ln\left(\frac{1}{0.548}\right) = \ln(1.825) = 0.601$$

#### **2.6: Update Sample Weights**

**Hệ số nhân cho từng sample:**
$$\beta_1^{1-e_i^{(1)}}$$

| ID | e_i^(1) | 1-e_i^(1) | β_1^(1-e_i) | w_i^(1) | w_i^(2) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3125 | 0.6875 | 0.548^0.6875 = 0.639 | 0.167 | 0.167×0.639 = 0.107 |
| 2 | 0.3125 | 0.6875 | 0.548^0.6875 = 0.639 | 0.167 | 0.167×0.639 = 0.107 |
| 3 | 1.0000 | 0.0000 | 0.548^0.0000 = 1.000 | 0.167 | 0.167×1.000 = 0.167 |
| 4 | 0.2500 | 0.7500 | 0.548^0.7500 = 0.612 | 0.167 | 0.167×0.612 = 0.102 |
| 5 | 0.0000 | 1.0000 | 0.548^1.0000 = 0.548 | 0.167 | 0.167×0.548 = 0.091 |
| 6 | 0.2500 | 0.7500 | 0.548^0.7500 = 0.612 | 0.167 | 0.167×0.612 = 0.102 |

**Normalization constant:**
$$Z_1 = \sum_{j=1}^{6} w_j^{(1)} \beta_1^{1-e_j^{(1)}} = 0.107 + 0.107 + 0.167 + 0.102 + 0.091 + 0.102 = 0.676$$

**Normalized weights:**
$$w_i^{(2)} = \frac{w_i^{(1)} \beta_1^{1-e_i^{(1)}}}{Z_1}$$

| ID | w_i^(1) | β_1^(1-e_i) | w_i^(2) |
|:---:|:---:|:---:|:---:|
| 1 | 0.167 | 0.639 | 0.107/0.676 = 0.158 |
| 2 | 0.167 | 0.639 | 0.107/0.676 = 0.158 |
| 3 | 0.167 | 1.000 | 0.167/0.676 = 0.247 |
| 4 | 0.167 | 0.612 | 0.102/0.676 = 0.151 |
| 5 | 0.167 | 0.548 | 0.091/0.676 = 0.135 |
| 6 | 0.167 | 0.612 | 0.102/0.676 = 0.151 |

---

### **Step 3: Loop 2 (m = 2)**

#### **2.1: Train Weak Learner h2(x) với Weighted Data**

**Tìm best split cho Income với weighted MSE:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | Weighted MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | [1] | [2,3,4,5,6] | [100] | [150,120,180,200,220] | 0.158×0 + 0.158×0 + 0.247×0 + 0.151×0 + 0.135×0 + 0.151×0 = 0 |
| 45 | [1,2] | [3,4,5,6] | [100,150] | [120,180,200,220] | 0.158×0 + 0.158×0 + 0.247×0 + 0.151×0 + 0.135×0 + 0.151×0 = 0 |
| 55 | [1,2,3] | [4,5,6] | [100,150,120] | [180,200,220] | 0.158×0 + 0.158×0 + 0.247×0 + 0.151×0 + 0.135×0 + 0.151×0 = 0 |
| 65 | [1,2,3,4] | [5,6] | [100,150,120,180] | [200,220] | 0.158×0 + 0.158×0 + 0.247×0 + 0.151×0 + 0.135×0 + 0.151×0 = 0 |
| 75 | [1,2,3,4,5] | [6] | [100,150,120,180,200] | [220] | 0.158×0 + 0.158×0 + 0.247×0 + 0.151×0 + 0.135×0 + 0.151×0 = 0 |

**Tất cả thresholds đều có weighted MSE = 0, chọn threshold = 45**

**h2(x) Structure:**
```
Root: Income <= 45?
├── Yes: Price = 125 (samples 1,2)
└── No: Price = 200 (samples 3,4,5,6)
```

**Predictions:**
$$h_2(x) = \{125, 125, 200, 200, 200, 200\}$$

#### **2.2: Calculate Normalized Absolute Error cho h2(x)**

**Absolute Errors:**
$$|y - h_2| = \{|100-125|, |150-125|, |120-200|, |180-200|, |200-200|, |220-200|\}$$
$$= \{25, 25, 80, 20, 0, 20\}$$

**Maximum Absolute Error:**
$$\max_j |y_j - h_2(x_j)| = 80$$


**Normalized Errors:**  
$$
e_i^{(1)} = \frac{|y_i - h_1(x_i)|}{80}
$$

| ID | Age | Income | Price | h1(x) | Abs Error | e_i^(1) |
|----|-----|--------|-------|-------|-----------|---------|
| 1  | 25  | 30     | 100   | 125   | 25        | 0.3125  |
| 2  | 30  | 50     | 150   | 125   | 25        | 0.3125  |
| 3  | 35  | 40     | 120   | 200   | 80        | 1.0000  |
| 4  | 40  | 60     | 180   | 200   | 20        | 0.2500  |
| 5  | 45  | 70     | 200   | 200   | 0         | 0.0000  |
| 6  | 50  | 80     | 220   | 200   | 20        | 0.2500  |


#### **2.3: Calculate Weighted Error**

$$\varepsilon_2 = \sum_{i=1}^{6} w_i^{(2)} e_i^{(2)} = 0.158 \times 0.3125 + 0.158 \times 0.3125 + 0.247 \times 1.0000 + 0.151 \times 0.2500 + 0.135 \times 0.0000 + 0.151 \times 0.2500$$

$$\varepsilon_2 = 0.158 \times 0.3125 + 0.158 \times 0.3125 + 0.247 \times 1.0000 + 0.151 \times 0.2500 + 0.135 \times 0.0000 + 0.151 \times 0.2500$$

$$\varepsilon_2 = 0.049 + 0.049 + 0.247 + 0.038 + 0.000 + 0.038 = 0.421$$

#### **2.4: Calculate Beta**

$$\beta_2 = \frac{\varepsilon_2}{1 - \varepsilon_2} = \frac{0.421}{1 - 0.421} = \frac{0.421}{0.579} = 0.727$$

#### **2.5: Calculate Alpha**

$$\alpha_2 = \ln\left(\frac{1}{\beta_2}\right) = \ln\left(\frac{1}{0.727}\right) = \ln(1.375) = 0.318$$

#### **2.6: Update Sample Weights**

**Hệ số nhân cho từng sample:**
$$\beta_2^{1-e_i^{(2)}}$$

| ID | e_i^(2) | 1-e_i^(2) | β_2^(1-e_i) | w_i^(2) | w_i^(3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3125 | 0.6875 | 0.727^0.6875 = 0.789 | 0.158 | 0.158×0.789 = 0.125 |
| 2 | 0.3125 | 0.6875 | 0.727^0.6875 = 0.789 | 0.158 | 0.158×0.789 = 0.125 |
| 3 | 1.0000 | 0.0000 | 0.727^0.0000 = 1.000 | 0.247 | 0.247×1.000 = 0.247 |
| 4 | 0.2500 | 0.7500 | 0.727^0.7500 = 0.760 | 0.151 | 0.151×0.760 = 0.115 |
| 5 | 0.0000 | 1.0000 | 0.727^1.0000 = 0.727 | 0.135 | 0.135×0.727 = 0.098 |
| 6 | 0.2500 | 0.7500 | 0.727^0.7500 = 0.760 | 0.151 | 0.151×0.760 = 0.115 |

**Normalization constant:**
$$Z_2 = 0.125 + 0.125 + 0.247 + 0.115 + 0.098 + 0.115 = 0.825$$

**Normalized weights:**
| ID | w_i^(2) | β_2^(1-e_i) | w_i^(3) |
|:---:|:---:|:---:|:---:|
| 1 | 0.158 | 0.789 | 0.125/0.825 = 0.152 |
| 2 | 0.158 | 0.789 | 0.125/0.825 = 0.152 |
| 3 | 0.247 | 1.000 | 0.247/0.825 = 0.299 |
| 4 | 0.151 | 0.760 | 0.115/0.825 = 0.139 |
| 5 | 0.135 | 0.727 | 0.098/0.825 = 0.119 |
| 6 | 0.151 | 0.760 | 0.115/0.825 = 0.139 |

---

### **Step 4: Loop 3 (m = 3)**

#### **2.1: Train Weak Learner h3(x) với Weighted Data**

**Tìm best split cho Age với weighted MSE:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | Weighted MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [100] | [150,120,180,200,220] | 0.152×0 + 0.152×0 + 0.299×0 + 0.139×0 + 0.119×0 + 0.139×0 = 0 |
| 32.5 | [1,2] | [3,4,5,6] | [100,150] | [120,180,200,220] | 0.152×0 + 0.152×0 + 0.299×0 + 0.139×0 + 0.119×0 + 0.139×0 = 0 |
| 37.5 | [1,2,3] | [4,5,6] | [100,150,120] | [180,200,220] | 0.152×0 + 0.152×0 + 0.299×0 + 0.139×0 + 0.119×0 + 0.139×0 = 0 |
| 42.5 | [1,2,3,4] | [5,6] | [100,150,120,180] | [200,220] | 0.152×0 + 0.152×0 + 0.299×0 + 0.139×0 + 0.119×0 + 0.139×0 = 0 |
| 47.5 | [1,2,3,4,5] | [6] | [100,150,120,180,200] | [220] | 0.152×0 + 0.152×0 + 0.299×0 + 0.139×0 + 0.119×0 + 0.139×0 = 0 |

**Tất cả thresholds đều có weighted MSE = 0, chọn threshold = 37.5**

**h3(x) Structure:**
```
Root: Age <= 37.5?
├── Yes: Price = 123.33 (samples 1,2,3)
└── No: Price = 200 (samples 4,5,6)
```

**Predictions:**
$$h_3(x) = \{123.33, 123.33, 123.33, 200, 200, 200\}$$

#### **2.2: Calculate Normalized Absolute Error cho h3(x)**

**Absolute Errors:**
$$|y - h_3| = \{|100-123.33|, |150-123.33|, |120-123.33|, |180-200|, |200-200|, |220-200|\}$$
$$= \{23.33, 26.67, 3.33, 20, 0, 20\}$$

**Maximum Absolute Error:**
$$\max_j |y_j - h_3(x_j)| = 26.67$$

**Normalized Errors:**
$$
e_i^{(3)} = \frac{|y_i - h_3(x_i)|}{26.67}
$$

| ID | Age | Income | Price | h3(x) | Abs Error | e_i^(3) |
|:--:|:---:|:------:|:-----:|:-----:|:---------:|:-------:|
|  1 | 25  |   30   |  100  | 123.33 |  23.33    | 0.875   |
|  2 | 30  |   50   |  150  | 123.33 |  26.67    | 1.000   |
|  3 | 35  |   40   |  120  | 123.33 |   3.33    | 0.125   |
|  4 | 40  |   60   |  180  |  200   |   20      | 0.750   |
|  5 | 45  |   70   |  200  |  200   |    0      | 0.000   |
|  6 | 50  |   80   |  220  |  200   |   20      | 0.750   |


#### **2.3: Calculate Weighted Error**

$$\varepsilon_3 = \sum_{i=1}^{6} w_i^{(3)} e_i^{(3)} = 0.152 \times 0.875 + 0.152 \times 1.000 + 0.299 \times 0.125 + 0.139 \times 0.750 + 0.119 \times 0.000 + 0.139 \times 0.750$$

$$\varepsilon_3 = 0.133 + 0.152 + 0.037 + 0.104 + 0.000 + 0.104 = 0.530$$

#### **2.4: Calculate Beta**

$$\beta_3 = \frac{\varepsilon_3}{1 - \varepsilon_3} = \frac{0.530}{1 - 0.530} = \frac{0.530}{0.470} = 1.128$$

#### **2.5: Calculate Alpha**

$$\alpha_3 = \ln\left(\frac{1}{\beta_3}\right) = \ln\left(\frac{1}{1.128}\right) = \ln(0.887) = -0.120$$

---

## **Step 5: Weighted Median Prediction**

### **Công thức Weighted Median**
$$\hat{y}(x) = \inf\left\{y : \sum_{m: h_m(x) \leq y} \alpha_m \geq \frac{1}{2}\sum_{m=1}^M \alpha_m\right\}$$

### **Tính toán cho từng điểm**

**Tại x = 25 (Age = 25, Income = 30):**
- $h_1(25) = 125$ với $\alpha_1 = 0.601$
- $h_2(25) = 125$ với $\alpha_2 = 0.318$
- $h_3(25) = 123.33$ với $\alpha_3 = -0.120$
- Tổng trọng số: $\alpha_1 + \alpha_2 + \alpha_3 = 0.601 + 0.318 + (-0.120) = 0.799$
- Mốc 50%: $0.799/2 = 0.400$

**Sắp xếp theo giá trị:**
1. $123.33$ (trọng số $-0.120$) → cộng dồn: $-0.120 < 0.400$
2. $125$ (trọng số $0.601$) → cộng dồn: $-0.120 + 0.601 = 0.481 > 0.400$ ✓

**Kết quả:** $\hat{y}(25) = 125$

**Tại x = 35 (Age = 35, Income = 40):**
- $h_1(35) = 200$ với $\alpha_1 = 0.601$
- $h_2(35) = 200$ với $\alpha_2 = 0.318$
- $h_3(35) = 123.33$ với $\alpha_3 = -0.120$
- Tổng trọng số: $0.799$
- Mốc 50%: $0.400$

**Sắp xếp theo giá trị:**
1. $123.33$ (trọng số $-0.120$) → cộng dồn: $-0.120 < 0.400$
2. $200$ (trọng số $0.601 + 0.318 = 0.919$) → cộng dồn: $-0.120 + 0.919 = 0.799 > 0.400$ ✓

**Kết quả:** $\hat{y}(35) = 200$

### **Final Predictions**

| ID | Age | Income | Price | h1(x) | h2(x) | h3(x) | Final Prediction | True Price |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 100 | 125 | 125 | 123.33 | 125 | 100 |
| 2 | 30 | 50 | 150 | 125 | 125 | 123.33 | 125 | 150 |
| 3 | 35 | 40 | 120 | 200 | 200 | 123.33 | 200 | 120 |
| 4 | 40 | 60 | 180 | 200 | 200 | 200 | 200 | 180 |
| 5 | 45 | 70 | 200 | 200 | 200 | 200 | 200 | 200 |
| 6 | 50 | 80 | 220 | 200 | 200 | 200 | 200 | 220 |

**MSE:**
$$MSE = \frac{1}{6}[(100-125)^2 + (150-125)^2 + (120-200)^2 + (180-200)^2 + (200-200)^2 + (220-200)^2]$$

$$MSE = \frac{1}{6}[625 + 625 + 6400 + 400 + 0 + 400] = \frac{1}{6}[8450] = 1408.33$$

---

## **Tóm tắt AdaBoost Regression**

### **Quy trình hoàn chỉnh:**

1. **Initialize Weights** - Khởi tạo trọng số đều cho tất cả samples
2. **Train Weak Learners** - Huấn luyện weak learners với weighted data
3. **Calculate Errors** - Tính normalized absolute error cho mỗi learner
4. **Calculate Weights** - Tính trọng số cho mỗi learner
5. **Update Sample Weights** - Cập nhật trọng số samples
6. **Weighted Median Prediction** - Kết hợp predictions với weighted median

**Ưu điểm của AdaBoost Regression:**
- **Adaptive learning** - Tự động điều chỉnh trọng số
- **Boosting** - Cải thiện performance qua từng iteration
- **Weighted sampling** - Tập trung vào samples khó
- **Ensemble learning** - Kết hợp nhiều weak learners
- **Robust** với outliers nhờ weighted median

**Nhược điểm:**
- **Sensitive to noise** - Dễ bị ảnh hưởng bởi outliers
- **Sequential training** - Không thể parallel
- **Overfitting** - Có thể overfit với dataset nhỏ
- **Computational cost** - Weighted median phức tạp

---

## **So sánh với các thuật toán khác**

| Đặc điểm | AdaBoost | Random Forest | XGBoost | LightGBM |
|:---:|:---:|:---:|:---:|:---:|
| **Method** | Boosting | Bagging | Boosting | Boosting |
| **Training** | Sequential | Parallel | Sequential | Sequential |
| **Sampling** | Weighted | Bootstrap | All samples | GOSS |
| **Features** | All features | Random subset | All features | All features |
| **Aggregation** | Weighted Median | Mean/Majority | Weighted | Weighted |
| **Speed** | Medium | Fast | Fast | Fastest |
| **Memory** | Medium | High | High | Low |
| **Accuracy** | Good | Good | Excellent | Excellent |

---

## **Hyperparameters quan trọng**

- **`n_estimators`**: Số lượng weak learners (50-200)
- **`learning_rate`**: Tốc độ học (0.1-1.0)
- **`loss`**: Loss function (linear, square, exponential)
- **`random_state`**: Seed cho reproducibility

---

## **Khi nào nên sử dụng AdaBoost Regression**

✅ **Nên dùng khi:**
- Cần model adaptive
- Dataset có ít noise
- Cần interpretability
- Có thời gian training
- Cần robustness với outliers

❌ **Không nên dùng khi:**
- Dataset có nhiều noise
- Cần model nhanh
- Có perfect learners
- Dataset rất nhỏ
- Cần parallel training

---

## **Kết luận**

AdaBoost Regression là một thuật toán boosting mạnh mẽ, sử dụng adaptive learning để cải thiện performance qua từng iteration. Việc sử dụng weighted median thay vì weighted average là điểm khác biệt quan trọng, giúp model ổn định hơn với các giá trị bất thường trong dữ liệu.

**Điểm mạnh chính:**
- **Adaptive learning** hiệu quả
- **Boosting** cải thiện performance
- **Weighted sampling** tập trung vào samples khó
- **Ensemble learning** kết hợp weak learners
- **Weighted median** robust với outliers