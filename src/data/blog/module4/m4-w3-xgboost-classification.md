---
title: "XGBoost Classification - Thuật toán Gradient Boosting cho Classification"
pubDatetime: 2025-09-21T18:00:00Z
featured: false
description: "Tìm hiểu chi tiết về XGBoost cho Classification, từ hàm mục tiêu đến quá trình xây dựng cây quyết định"
tags: ["machine-learning", "xgboost", "classification", "gradient-boosting", "algorithm"]
---

# XGBoost Classification

> **📚 Repo tham khảo:** [https://github.com/tandat8896/ml-from-the-scartch/tree/master/xgboost](https://github.com/tandat8896/ml-from-the-scartch/tree/master/xgboost)

## Hàm mục tiêu (Objective) trong XGBoost Classification

### **Hàm mất mát cho Classification (Log Loss)**

$$\mathcal{L}(y_i, \hat{y}_i) = -y_i\log(\hat{y}_i) - (1-y_i)\log(1-\hat{y}_i)$$

### **Chuyển đổi xác suất thành log(odds)**

$$\hat{y}_i = \frac{1}{1 + e^{-F(x_i)}}$$

### **Gradient và Hessian cho Classification**

**Gradient:**
$$g_i = \frac{\partial \mathcal{L}}{\partial F(x_i)} = \hat{y}_i - y_i$$

**Hessian:**
$$h_i = \frac{\partial^2 \mathcal{L}}{\partial F(x_i)^2} = \hat{y}_i(1-\hat{y}_i)$$

### **Hàm mục tiêu XGBoost với Regularization**

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \mathcal{L}(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

**Trong đó:**
- $\mathcal{L}(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))$: Loss function
- $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$: Regularization term
- $T$: Số leaves trong tree
- $w_j$: Output value của leaf $j$
- $\gamma$: Minimum gain to split
- $\lambda$: L2 regularization

---

## **Ví dụ chi tiết: Dự đoán Credit Risk**

### **Dataset**
| ID | Age | Income | Label |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 |
| 2 | 30 | 50 | 1 |
| 3 | 35 | 40 | 0 |
| 4 | 40 | 60 | 0 |
| 5 | 45 | 70 | 1 |
| 6 | 50 | 80 | 0 |

**Mục tiêu:** Dự đoán credit risk (1 = high risk, 0 = low risk)

---

## **Bước 1: Khởi tạo**

### **Initial Prediction**
$$F_0(x) = \log\left(\frac{p}{1-p}\right) = \log\left(\frac{3/6}{3/6}\right) = \log(1) = 0$$

**Initial probabilities:**
$$\hat{y}_i = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = 0.5 \text{ cho tất cả samples}$$

---

## **Bước 2: Iteration 1 - Gradients & Hessians**

**Tính gradients:**
$$g_i = \hat{y}_i - y_i = 0.5 - y_i$$

**Tính hessians:**
$$h_i = \hat{y}_i(1-\hat{y}_i) = 0.5 \times (1-0.5) = 0.25$$

| ID | Age | Income | Label | F0(x) | $\hat{y}_i$ | $g_i$ | $h_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0 | 0.5 | -0.5 | 0.25 |
| 2 | 30 | 50 | 1 | 0 | 0.5 | -0.5 | 0.25 |
| 3 | 35 | 40 | 0 | 0 | 0.5 | 0.5 | 0.25 |
| 4 | 40 | 60 | 0 | 0 | 0.5 | 0.5 | 0.25 |
| 5 | 45 | 70 | 1 | 0 | 0.5 | -0.5 | 0.25 |
| 6 | 50 | 80 | 0 | 0 | 0.5 | 0.5 | 0.25 |

---

## **Bước 3: Tìm Best Split**

### **Candidates cho Age**
- Threshold = 27.5: Left = {1}, Right = {2,3,4,5,6}
- Threshold = 32.5: Left = {1,2}, Right = {3,4,5,6}
- Threshold = 37.5: Left = {1,2,3}, Right = {4,5,6}
- Threshold = 42.5: Left = {1,2,3,4}, Right = {5,6}
- Threshold = 47.5: Left = {1,2,3,4,5}, Right = {6}

### **Tính Gain cho từng threshold**

**Threshold = 27.5:**
- Left: {1} → $G_L = -0.5$, $H_L = 0.25$
- Right: {2,3,4,5,6} → $G_R = -0.5 + 0.5 + 0.5 + (-0.5) + 0.5 = 0.5$, $H_R = 0.25 + 0.25 + 0.25 + 0.25 + 0.25 = 1.25$

**Gain calculation với $\lambda = 1.0$:**
$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-0.5)^2}{0.25 + 1.0} + \frac{(0.5)^2}{1.25 + 1.0} - \frac{(0)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.25}{1.25} + \frac{0.25}{2.25} - 0\right] = \frac{1}{2}[0.2 + 0.111] = 0.156$$

**Threshold = 32.5:**
- Left: {1,2} → $G_L = -0.5 + (-0.5) = -1.0$, $H_L = 0.25 + 0.25 = 0.5$
- Right: {3,4,5,6} → $G_R = 0.5 + 0.5 + (-0.5) + 0.5 = 1.0$, $H_R = 0.25 + 0.25 + 0.25 + 0.25 = 1.0$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-1.0)^2}{0.5 + 1.0} + \frac{(1.0)^2}{1.0 + 1.0} - \frac{(0)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{1}{1.5} + \frac{1}{2.0} - 0\right] = \frac{1}{2}[0.667 + 0.5] = 0.583$$

**Threshold = 37.5:**
- Left: {1,2,3} → $G_L = -0.5 + (-0.5) + 0.5 = -0.5$, $H_L = 0.25 + 0.25 + 0.25 = 0.75$
- Right: {4,5,6} → $G_R = 0.5 + (-0.5) + 0.5 = 0.5$, $H_R = 0.25 + 0.25 + 0.25 = 0.75$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-0.5)^2}{0.75 + 1.0} + \frac{(0.5)^2}{0.75 + 1.0} - \frac{(0)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.25}{1.75} + \frac{0.25}{1.75} - 0\right] = \frac{1}{2}[0.143 + 0.143] = 0.143$$

**Threshold = 42.5:**
- Left: {1,2,3,4} → $G_L = -0.5 + (-0.5) + 0.5 + 0.5 = 0$, $H_L = 0.25 + 0.25 + 0.25 + 0.25 = 1.0$
- Right: {5,6} → $G_R = -0.5 + 0.5 = 0$, $H_R = 0.25 + 0.25 = 0.5$

$$\text{Gain} = \frac{1}{2}\left[\frac{(0)^2}{1.0 + 1.0} + \frac{(0)^2}{0.5 + 1.0} - \frac{(0)^2}{1.5 + 1.0}\right] = 0$$

**Threshold = 47.5:**
- Left: {1,2,3,4,5} → $G_L = -0.5 + (-0.5) + 0.5 + 0.5 + (-0.5) = -0.5$, $H_L = 0.25 + 0.25 + 0.25 + 0.25 + 0.25 = 1.25$
- Right: {6} → $G_R = 0.5$, $H_R = 0.25$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-0.5)^2}{1.25 + 1.0} + \frac{(0.5)^2}{0.25 + 1.0} - \frac{(0)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.25}{2.25} + \frac{0.25}{1.25} - 0\right] = \frac{1}{2}[0.111 + 0.2] = 0.156$$

**Kết quả:** Threshold tối ưu = 32.5 (Gain = 0.583)

---

## **Bước 4: Tính Output Values**

**Left leaf (Age ≤ 32.5):**
$$w_L = -\frac{G_L}{H_L + \lambda} = -\frac{-1.0}{0.5 + 1.0} = \frac{1.0}{1.5} = 0.667$$

**Right leaf (Age > 32.5):**
$$w_R = -\frac{G_R}{H_R + \lambda} = -\frac{1.0}{1.0 + 1.0} = -\frac{1.0}{2.0} = -0.5$$

### **Tree Structure**
```
Root: Age ≤ 32.5?
├── Yes: 0.667
└── No: -0.5
```

---

## **Bước 5: Update Model**

**Learning rate $\eta = 0.1$:**

$$F_1(x) = F_0(x) + \eta \times f_1(x)$$

**f1(x) function:**
- Nếu Age ≤ 32.5 thì f1(x) = 0.667
- Nếu Age > 32.5 thì f1(x) = -0.5

**Updated predictions:**
| ID | Age | Income | Label | F0(x) | f1(x) | F1(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0 | 0.667 | 0 + 0.1×0.667 = 0.067 | 0.517 |
| 2 | 30 | 50 | 1 | 0 | 0.667 | 0 + 0.1×0.667 = 0.067 | 0.517 |
| 3 | 35 | 40 | 0 | 0 | -0.5 | 0 + 0.1×(-0.5) = -0.05 | 0.488 |
| 4 | 40 | 60 | 0 | 0 | -0.5 | 0 + 0.1×(-0.5) = -0.05 | 0.488 |
| 5 | 45 | 70 | 1 | 0 | -0.5 | 0 + 0.1×(-0.5) = -0.05 | 0.488 |
| 6 | 50 | 80 | 0 | 0 | -0.5 | 0 + 0.1×(-0.5) = -0.05 | 0.488 |

**Tính xác suất mới:**
- ID 1: $\hat{y}_1 = \frac{1}{1 + e^{-0.067}} = \frac{1}{1 + 0.935} = 0.517$
- ID 2: $\hat{y}_2 = \frac{1}{1 + e^{-0.067}} = 0.517$
- ID 3: $\hat{y}_3 = \frac{1}{1 + e^{-(-0.05)}} = \frac{1}{1 + 1.051} = 0.488$
- ID 4: $\hat{y}_4 = 0.488$
- ID 5: $\hat{y}_5 = 0.488$
- ID 6: $\hat{y}_6 = 0.488$

---

## **Bước 6: Iteration 2 - Gradients & Hessians**

**Tính gradients cho iteration 2:**
$$g_i = \hat{y}_i - y_i$$

**Tính hessians cho iteration 2:**
$$h_i = \hat{y}_i(1-\hat{y}_i)$$

| ID | Age | Income | Label | F1(x) | $\hat{y}_i$ | $g_i$ | $h_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.067 | 0.517 | -0.483 | 0.250 |
| 2 | 30 | 50 | 1 | 0.067 | 0.517 | -0.483 | 0.250 |
| 3 | 35 | 40 | 0 | -0.05 | 0.488 | 0.488 | 0.250 |
| 4 | 40 | 60 | 0 | -0.05 | 0.488 | 0.488 | 0.250 |
| 5 | 45 | 70 | 1 | -0.05 | 0.488 | -0.512 | 0.250 |
| 6 | 50 | 80 | 0 | -0.05 | 0.488 | 0.488 | 0.250 |

**Tính hessians chi tiết:**
- ID 1: $h_1 = 0.517 \times (1-0.517) = 0.517 \times 0.483 = 0.250$
- ID 2: $h_2 = 0.250$
- ID 3: $h_3 = 0.488 \times (1-0.488) = 0.488 \times 0.512 = 0.250$
- ID 4: $h_4 = 0.250$
- ID 5: $h_5 = 0.250$
- ID 6: $h_6 = 0.250$

---

## **Bước 7: Tìm Best Split cho Income**

### **Candidates cho Income**
- Threshold = 35: Left = {1,3}, Right = {2,4,5,6}
- Threshold = 45: Left = {1,2,3}, Right = {4,5,6}
- Threshold = 55: Left = {1,2,3,4}, Right = {5,6}
- Threshold = 75: Left = {1,2,3,4,5}, Right = {6}

### **Tính Gain cho từng threshold**

**Threshold = 35:**
- Left: {1,3} → $G_L = -0.483 + 0.488 = 0.005$, $H_L = 0.250 + 0.250 = 0.5$
- Right: {2,4,5,6} → $G_R = -0.483 + 0.488 + (-0.512) + 0.488 = -0.019$, $H_R = 0.250 + 0.250 + 0.250 + 0.250 = 1.0$

$$\text{Gain} = \frac{1}{2}\left[\frac{(0.005)^2}{0.5 + 1.0} + \frac{(-0.019)^2}{1.0 + 1.0} - \frac{(-0.014)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.000025}{1.5} + \frac{0.000361}{2.0} - \frac{0.000196}{2.5}\right] = \frac{1}{2}[0.000017 + 0.000181 - 0.000078] = 0.000060$$

**Threshold = 45:**
- Left: {1,2,3} → $G_L = -0.483 + (-0.483) + 0.488 = -0.478$, $H_L = 0.250 + 0.250 + 0.250 = 0.75$
- Right: {4,5,6} → $G_R = 0.488 + (-0.512) + 0.488 = 0.464$, $H_R = 0.250 + 0.250 + 0.250 = 0.75$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-0.478)^2}{0.75 + 1.0} + \frac{(0.464)^2}{0.75 + 1.0} - \frac{(-0.014)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.228}{1.75} + \frac{0.215}{1.75} - \frac{0.000196}{2.5}\right] = \frac{1}{2}[0.130 + 0.123 - 0.000078] = 0.126$$

**Threshold = 55:**
- Left: {1,2,3,4} → $G_L = -0.483 + (-0.483) + 0.488 + 0.488 = 0.010$, $H_L = 0.250 + 0.250 + 0.250 + 0.250 = 1.0$
- Right: {5,6} → $G_R = -0.512 + 0.488 = -0.024$, $H_R = 0.250 + 0.250 = 0.5$

$$\text{Gain} = \frac{1}{2}\left[\frac{(0.010)^2}{1.0 + 1.0} + \frac{(-0.024)^2}{0.5 + 1.0} - \frac{(-0.014)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.0001}{2.0} + \frac{0.000576}{1.5} - \frac{0.000196}{2.5}\right] = \frac{1}{2}[0.00005 + 0.000384 - 0.000078] = 0.000178$$

**Threshold = 75:**
- Left: {1,2,3,4,5} → $G_L = -0.483 + (-0.483) + 0.488 + 0.488 + (-0.512) = -0.502$, $H_L = 0.250 + 0.250 + 0.250 + 0.250 + 0.250 = 1.25$
- Right: {6} → $G_R = 0.488$, $H_R = 0.250$

$$\text{Gain} = \frac{1}{2}\left[\frac{(-0.502)^2}{1.25 + 1.0} + \frac{(0.488)^2}{0.250 + 1.0} - \frac{(-0.014)^2}{1.5 + 1.0}\right]$$

$$\text{Gain} = \frac{1}{2}\left[\frac{0.252}{2.25} + \frac{0.238}{1.25} - \frac{0.000196}{2.5}\right] = \frac{1}{2}[0.112 + 0.190 - 0.000078] = 0.151$$

**Kết quả:** Threshold tối ưu = 45 (Gain = 0.126)

---

## **Bước 8: Tính Output Values cho Tree 2**

**Left leaf (Income ≤ 45):**
$$w_L = -\frac{G_L}{H_L + \lambda} = -\frac{-0.478}{0.75 + 1.0} = \frac{0.478}{1.75} = 0.273$$

**Right leaf (Income > 45):**
$$w_R = -\frac{G_R}{H_R + \lambda} = -\frac{0.464}{0.75 + 1.0} = -\frac{0.464}{1.75} = -0.265$$

### **Tree 2 Structure**
```
Root: Income ≤ 45?
├── Yes: 0.273
└── No: -0.265
```

---

## **Bước 9: Update Model lần 2**

**Learning rate $\eta = 0.1$:**

$$F_2(x) = F_1(x) + \eta \times f_2(x)$$

**f2(x) function:**
- Nếu Income ≤ 45 thì f2(x) = 0.273
- Nếu Income > 45 thì f2(x) = -0.265

**Updated predictions:**
| ID | Age | Income | Label | F1(x) | f2(x) | F2(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.067 | 0.273 | 0.067 + 0.1×0.273 = 0.094 | 0.524 |
| 2 | 30 | 50 | 1 | 0.067 | -0.265 | 0.067 + 0.1×(-0.265) = 0.041 | 0.510 |
| 3 | 35 | 40 | 0 | -0.05 | 0.273 | -0.05 + 0.1×0.273 = -0.023 | 0.494 |
| 4 | 40 | 60 | 0 | -0.05 | -0.265 | -0.05 + 0.1×(-0.265) = -0.077 | 0.481 |
| 5 | 45 | 70 | 1 | -0.05 | -0.265 | -0.05 + 0.1×(-0.265) = -0.077 | 0.481 |
| 6 | 50 | 80 | 0 | -0.05 | -0.265 | -0.05 + 0.1×(-0.265) = -0.077 | 0.481 |

**Tính xác suất mới:**
- ID 1: $\hat{y}_1 = \frac{1}{1 + e^{-0.094}} = \frac{1}{1 + 0.910} = 0.524$
- ID 2: $\hat{y}_2 = \frac{1}{1 + e^{-0.041}} = \frac{1}{1 + 0.960} = 0.510$
- ID 3: $\hat{y}_3 = \frac{1}{1 + e^{-(-0.023)}} = \frac{1}{1 + 1.023} = 0.494$
- ID 4: $\hat{y}_4 = \frac{1}{1 + e^{-(-0.077)}} = \frac{1}{1 + 1.080} = 0.481$
- ID 5: $\hat{y}_5 = 0.481$
- ID 6: $\hat{y}_6 = 0.481$

---

## **Bước 10: Final Predictions**

**Threshold = 0.5:**
- Nếu $\hat{y}_i \geq 0.5$ thì prediction = 1 (high risk)
- Nếu $\hat{y}_i < 0.5$ thì prediction = 0 (low risk)

| ID | Age | Income | Label | F2(x) | $\hat{y}_i$ | Prediction | Correct? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.094 | 0.524 | 1 | ✅ |
| 2 | 30 | 50 | 1 | 0.041 | 0.510 | 1 | ✅ |
| 3 | 35 | 40 | 0 | -0.023 | 0.494 | 0 | ✅ |
| 4 | 40 | 60 | 0 | -0.077 | 0.481 | 0 | ✅ |
| 5 | 45 | 70 | 1 | -0.077 | 0.481 | 0 | ❌ |
| 6 | 50 | 80 | 0 | -0.077 | 0.481 | 0 | ✅ |

**Accuracy:** 5/6 = 83.3%

---

## **Tóm tắt XGBoost Classification**

### **Quy trình hoàn chỉnh:**

1. **Initialize** - Khởi tạo prediction với log(odds)
2. **Calculate Gradients** - Tính gradients cho từng sample
3. **Calculate Hessians** - Tính hessians cho từng sample
4. **Find Best Split** - Tìm threshold tối ưu dựa trên gain
5. **Calculate Output** - Tính output values cho leaves
6. **Update Model** - Cập nhật model với learning rate
7. **Repeat** - Lặp lại cho đến khi đạt convergence

**Ưu điểm của XGBoost Classification:**
- **Gradient boosting** hiệu quả
- **Regularization** tránh overfitting
- **Parallel processing** nhanh chóng
- **Handles missing values** tốt
- **Feature importance** có thể tính được
- **Robust** với noise và outliers

**Nhược điểm:**
- **Memory usage** cao
- **Hyperparameter tuning** phức tạp
- **Black box** model
- **Sensitive to outliers**

---

## **So sánh với các thuật toán khác**

| Đặc điểm | XGBoost | Random Forest | AdaBoost | LightGBM |
|:---:|:---:|:---:|:---:|:---:|
| **Method** | Boosting | Bagging | Boosting | Boosting |
| **Training** | Sequential | Parallel | Sequential | Sequential |
| **Speed** | Fast | Fast | Medium | Fastest |
| **Memory** | High | High | Medium | Low |
| **Accuracy** | Excellent | Good | Good | Excellent |
| **Regularization** | Yes | No | No | Yes |

---

## **Hyperparameters quan trọng**

- **`n_estimators`**: Số lượng trees (100-1000)
- **`learning_rate`**: Tốc độ học (0.01-0.3)
- **`max_depth`**: Độ sâu tối đa (3-10)
- **`min_child_weight`**: Trọng số tối thiểu (1-10)
- **`subsample`**: Tỷ lệ samples (0.6-1.0)
- **`colsample_bytree`**: Tỷ lệ features (0.6-1.0)
- **`reg_alpha`**: L1 regularization (0-1)
- **`reg_lambda`**: L2 regularization (0-1)

---

## **Khi nào nên sử dụng XGBoost Classification**

✅ **Nên dùng khi:**
- Cần accuracy cao
- Có dataset lớn
- Cần xử lý missing values
- Có thời gian tuning hyperparameters
- Cần feature importance

❌ **Không nên dùng khi:**
- Dataset rất nhỏ
- Cần model interpretable
- Memory hạn chế
- Cần model rất nhanh
- Có ít thời gian tuning

---

## **Kết luận**

XGBoost Classification là một thuật toán gradient boosting mạnh mẽ, kết hợp sức mạnh của gradient boosting với các kỹ thuật regularization và optimization tiên tiến. Với khả năng xử lý tốt các vấn đề về missing values, feature importance và cho kết quả accuracy cao, XGBoost là lựa chọn tuyệt vời cho nhiều bài toán classification thực tế.

**Điểm mạnh chính:**
- **Gradient boosting** hiệu quả
- **Regularization** tránh overfitting
- **Parallel processing** nhanh chóng
- **Handles missing values** tốt
- **Feature importance** có thể tính được
- **Robust** với noise và outliers