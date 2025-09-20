
---
title: "Behind The Scene của XGBoost"
pubDatetime: 2025-01-16T10:00:00Z
featured: false
description: "Tìm hiểu chi tiết về thuật toán XGBoost, từ hàm mục tiêu đến quá trình xây dựng cây quyết định"
tags: ["machine-learning", "xgboost", "gradient-boosting", "algorithm"]
---

# Behind The Scene của XGBoost

> **📚 Repo tham khảo:** [https://github.com/tandat8896/ml-from-the-scartch/tree/master/xgboost](https://github.com/tandat8896/ml-from-the-scartch/tree/master/xgboost)

## Hàm mục tiêu (Objective) trong XGBoost
* **Hàm mất mát cho Regression:**

$$
\sum_{i=1}^{N} \mathcal{L}(y_i, \bar{y}_i) \quad \text{Where} \quad \mathcal{L}(y_i, \bar{y}_i) = \frac{1}{2}(y_i - \bar{y}_i)^2
$$

* **Hàm mất mát cho Classification:**

$$
\sum_{i=1}^{N} \mathcal{L}(y_i, \bar{y}_i) \quad \text{Where} \quad \mathcal{L}(y_i, \bar{y}_i) = - \left[ y_i\log(\bar{y}_i) + (1-y_i)\log(1-\bar{y}_i) \right]
$$

---

$$
\mathrm{Objective}^{(t)} = \sum_{i=1}^{N} \mathcal{L}(y_i, \hat{y}_i^{(t)}) + \sum_{j=1}^{t} \Omega(f_j)
$$





Trong đó:

- **Loss function**:  
  $\mathcal{L}(y_i, \hat{y}_i^{(t)})$ đo sự khác biệt giữa giá trị thực $y_i$ và dự đoán $\hat{y}_i^{(t)}$.

- **Penalty term (thành phần phạt)**:  
  $\Omega(f_j)$ là phần phạt, nhằm giới hạn độ phức tạp của cây để tránh overfitting.

### Biểu thức penalty

$$
\Omega(f) = \gamma T + \frac{\lambda}{2} \sum_{j=1}^{T} w_j^2
$$

Trong đó:
- $T$: số lá của cây.
- $w_j$: trọng số của lá $j$.
- $\gamma, \lambda$: siêu tham số điều chỉnh độ phức tạp.

* vì XGBoost có thể thực hiện tỉa cây (pruning) ngay cả khi $\gamma = 0$.
* Tỉa cây được thực hiện sau khi cây đã được xây dựng hoàn chỉnh. Do đó, quá trình này không đóng vai trò trong việc xác định các giá trị đầu ra tối ưu ban đầu của các lá.


**XGBoost xây dựng cây mới dựa trên hàm mất mát.**

$$
\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0 + P) + \frac{1}{2}\lambda P^2
$$

* **Mục tiêu:** tìm giá trị dự đoán cho mỗi leaf (P) của cây mới nhằm minimize hàm loss.
* **Rigde Regression Regularization term**

*Giá trị P cần tìm là giá trị ứng với đạo hàm của loss theo P bằng 0.*

vì phần này $\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0 + P)$ rất khó để đạo hàm (Hàm loss có thể không mượt (non-smooth) và phức tạp, Không có công thức đóng (closed-form) tổng quát, Hàm có thể nhiều cực trị)

**Vì vậy chúng ta sẽ sấp xĩ nó bằng Taylor Second Approximate( Vì công thức này cho chúng ta biết xấp xĩ hàm quanh 1 điểm).**

**Tìm giá trị P để tối ưu hóa, ta xấp xỉ hàm loss bằng Taylor Approximation bậc hai:**

$$
\mathcal{L}(y_i, \bar{y}_i^0 + P) \approx \mathcal{L}(y_i, \bar{y}_i^0) + \left[ \frac{d\mathcal{L}}{d\bar{y}_i} \right] P + \frac{1}{2}\left[ \frac{d^2\mathcal{L}}{d\bar{y}_i^2} \right] P^2
$$

**Sử dụng ký hiệu g (gradient) và h (hessian) cho đạo hàm:**

* $g$: đạo hàm bậc nhất của hàm loss.
* $h$: đạo hàm bậc hai của hàm loss.

$$
\mathcal{L}(y_i, \bar{y}_i^0 + P) \approx \mathcal{L}(y_i, \bar{y}_i^0) + gP + \frac{1}{2}hP^2
$$

**Thay vào hàm mất mát của XGBoost ta có :**

$$
\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0) + \sum_{i=1}^{n} \left( g_i P + \frac{1}{2}h_i P^2 \right) + \frac{1}{2}\lambda P^2
$$

**Tìm giá trị P sao cho đạo hàm của biểu thức theo P bằng 0:**

$$
\frac{d}{dP} \left[ \sum_{i=1}^{n} (g_i + h_i P) + \lambda P \right] = 0
$$

**Giải phương trình để tìm giá trị tối ưu của P:**

$$
\frac{d}{dP} \left[ (g_1 + g_2 + \dots + g_n)P + \frac{1}{2}(h_1 + h_2 + \dots + h_n + \lambda)P^2 \right] = 0
$$

**Sau khi đạo hàm, ta có:**

$$
(g_1 + g_2 + \dots + g_n) + (h_1 + h_2 + \dots + h_n + \lambda)P = 0
$$

* **1:** $g_i = \frac{d}{d\bar{y}_i}\frac{1}{2}(y_i - \bar{y}_i)^2 = (y_i - \bar{y}_i)$
* **2:** $h_i = \frac{d^2}{d\bar{y}_i^2}\frac{1}{2}(y_i - \bar{y}_i)^2 = 1$

**Giá trị P tối ưu (Output value of the leaf or terminal node):**

$$
P = -\frac{g_1 + g_2 + \dots + g_n}{h_1 + h_2 + \dots + h_n + \lambda} = -\frac{(y_1 - \bar{y}_1) + (y_2 - \bar{y}_2) + \dots + (y_n - \bar{y}_n)}{1 + 1 + \dots + 1 + \lambda}
$$

$$P = \frac{\sum_{i=1}^{n} (y_i - \bar{y}_i)}{n + \lambda}$$



### **Classification**

**Hàm mất mát (Loss Function):**

$$\mathcal{L}(Y_i, \bar{Y_i}) = -Y_i\log(\bar{Y_i}) + (1-Y_i)\log(1-\bar{Y_i})$$

**Chuyển đổi xác suất (probability) thành log(odds):**

$$\mathcal{L}(Y_i, \log(odds)) = -Y_i\log(odds) + \log(1 + e^{\log(odds)})$$

**Đạo hàm bậc nhất ($g_i$):**

$$
g_i = \frac{d}{d\log(odds)}\mathcal{L}(Y_i, \log(odds)) = -Y_i + \frac{e^{\log(odds)}}{1 + e^{\log(odds)}} = -(Y_i - \bar{Y_i})
$$

**Đạo hàm bậc hai ($h_i$):**

$$
h_i = \frac{d^2}{d\log(odds)^2}\mathcal{L}(Y_i, \log(odds)) = \frac{e^{\log(odds)}}{1 + e^{\log(odds)}} \times \frac{1}{1 + e^{\log(odds)}} = \bar{Y_i}(1-\bar{Y_i})
$$

**Giá trị P tối ưu (Output value of the leaf or terminal node):**

$$
P = -\frac{-(g_1 + g_2 + \dots + g_n)}{h_1 + h_2 + \dots + h_n + \lambda} = \frac{\text{sum of residual}}{\text{sum of } \bar{Y_i}(1 - \bar{Y_i}) + \lambda} = \frac{\sum(\text{Residual})}{\sum\bar{Y_i}(1-\bar{Y_i}) + \lambda}
$$

**Mối quan hệ giữa Loss Function và P**

* **Khó khăn:** Rất khó để tìm giá trị P tối ưu (optimization) trực tiếp từ hàm Loss ban đầu.
* **Giải pháp:** Sử dụng xấp xỉ bậc hai (Second Order Taylor Approximation) để đơn giản hóa hàm Loss.

$$
\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0 + P) + \frac{1}{2}\lambda P^2
$$

* **1.** Hàm Loss ban đầu.
* **2.** Hàm Loss sau khi xấp xỉ bằng Taylor.
$$
(g_1 + g_2 + \dots + g_n)P + \frac{1}{2}(h_1 + h_2 + \dots + h_n + \lambda)P^2
$$
* **Điểm chung:** Cả (1) và (2) đều có cùng điểm tối ưu (optimization point) P.
* **Công thức tìm P:**
$$
P = -\frac{-(g_1 + g_2 + \dots + g_n)}{h_1 + h_2 + \dots + h_n + \lambda}
$$


### **Công thức tính điểm tương đồng (Similarity Score)**

* **Mục đích:** Tìm điểm tối ưu của hàm mục tiêu bằng cách xấp xỉ hàm loss ban đầu.

Hàm loss ban đầu:

$$
\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0 + P) + \frac{1}{2}\lambda P^2
$$

Hàm loss sau khi xấp xỉ Taylor bậc hai:

$$
\sum_{i=1}^{n} (g_i P + \frac{1}{2}h_i P^2) + \frac{1}{2}\lambda P^2
$$

* **Lưu ý:** Cả hai hàm trên đều có cùng một điểm tối ưu (P).

Khi đó, hàm mục tiêu được viết lại:


* **Hàm mất mát ban đầu:**

$$
\sum_{i=1}^{n} \mathcal{L}(y_i, \bar{y}_i^0 + P) + \frac{1}{2}\lambda P^2
$$

* **Xấp xỉ Taylor bậc hai:**

$$
\mathcal{L}(y_i, \bar{y}_i^0 + P) \approx \mathcal{L}(y_i, \bar{y}_i^0) + \left[ \frac{d\mathcal{L}}{d\bar{y}_i} \right] P + \frac{1}{2}\left[ \frac{d^2\mathcal{L}}{d\bar{y}_i^2} \right] P^2
$$

$$
\mathcal{L}(y_i, \bar{y}_i^0 + P) \approx \mathcal{L}(y_i, \bar{y}_i^0) + gP + \frac{1}{2}hP^2
$$

* **Thay vì tìm cực tiểu cho hàm loss thì ta sẽ nhân -1 để đảo cực trị lại tìm cực đại cho Similarity Score mà tại đó điểm cực đại sẽ trùng với cực tiểu của hàm loss (Tương tự ý nghĩa của Infomation Gain trong Decision Tree Regression)**

$$
\text{Similarity Score} =-1\left(\sum_{i=1}^{n} g_i\right)P + \frac{1}{2}\left(\sum_{i=1}^{n} h_i + \lambda\right)P^2
$$

* **Giá trị P tối ưu:**

$$
P = -\frac{\left(g_1 + g_2 + \dots + g_n\right)}{h_1 + h_2 + \dots + h_n + \lambda}
$$

**Thay P vào công thức SimilarityScore**


$$
\text{Similarity Score} = \frac{1}{2} \frac{(g_1 + g_2 + \dots + g_n)^2}{(h_1 + h_2 + \dots + h_n + \lambda)}
$$


**Ví Dụ Tính Tay**

* **Loss Function Regression:**

$$
\sum_{i=1}^{3} \mathcal{L}(y_i, \bar{y}_i) \text{ Where } \mathcal{L}(y_i, \bar{y}_i) = \frac{1}{2}(y_i - \bar{y}_i)^2
$$

* **Loss Function Classification:**

$$
\sum_{i=1}^{3} \mathcal{L}(y_i, \bar{y}_i) \text{ Where } \mathcal{L}(y_i, \bar{y}_i) = - \left[ y_i\log(\bar{y}_i) + (1-y_i)\log(1-\bar{y}_i) \right]
$$

---

### **Step 1: Initialization**

| Age (X) | Chol (y) |
|:---:|:---:|
| 29 | 204 |
| 48 | 234 |
| 39 | 203 |
| 67 | 269 |
| 45 | 250 |
| 59 | 260 |


* **F0(x) là giá trị tối ưu bằng cách tìm giá trị nhỏ nhất của hàm Loss:**
    $$
    F_0(x) = \arg \min_{\theta} \sum_{i=1}^{N} l(y_i, \theta)
    $$

$F_0 = \frac{1}{N} \sum_{i=1}^{N} y_i$

$F_0 = \frac{204 + 234 + 203 + 269 + 250 + 260}{6}$

$F_0 = 236.67$



#### **Bước 2A: Xây dựng cây – Sắp xếp**

Sắp xếp các mẫu trong node theo giá trị của feature:

$$
X_{sorted} = [29, 39, 45, 48, 59, 67]
$$

| Age (X) | Chol (y) |
| :---: | :---: |
| 29 | 204 |
| 48 | 234 |
| 39 | 203 |
| 67 | 269 |
| 45 | 250 |
| 59 | 260 |
| $F_0$ = 236.67 | |

| Age (X) | Chol (y) |
| :---: | :---: |
| 29 | 204 |
| 39 | 203 |
| 45 | 250 |
| 48 | 234 |
| 59 | 260 |
| 67 | 269 |
| $F_0$ = 236.67 | |

---
### **Đạo hàm của Loss Function Regression (MSE)**

* **Hàm mất mát (Loss Function):**
    $$
    l(\hat{y}_i, y_i) = \frac{1}{2}(\hat{y}_i - y_i)^2
    $$

* **Đạo hàm bậc nhất ($g_i$):**
    $$
    g_i = \frac{\partial l}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \left(\frac{1}{2}(\hat{y}_i - y_i)^2 \right) = \frac{1}{2} \cdot 2(\hat{y}_i - y_i) \cdot \frac{\partial (\hat{y}_i - y_i)}{\partial \hat{y}_i} = \hat{y}_i - y_i
    $$

* **Đạo hàm bậc hai ($h_i$):**
    $$
    h_i = \frac{\partial^2 l}{\partial \hat{y}_i^2} = \frac{\partial}{\partial \hat{y}_i} (\hat{y}_i - y_i) = 1
    $$

---
### **Tính Gradients ($g_i$) và Hessians ($h_i$)**

* **Giá trị ban đầu:** $F_0 = 236.67$

| Age (X) | Chol (y) | Gradients ($g_i = F_0 - y_i$) | Hessians ($h_i$) |
| :---: | :---: | :---: | :---: |
| 29 | 204 | $236.67 - 204 = 32.67$ | 1 |
| 39 | 203 | $236.67 - 203 = 33.67$ | 1 |
| 45 | 250 | $236.67 - 250 = -13.33$ | 1 |
| 48 | 234 | $236.67 - 234 = 2.67$ | 1 |
| 59 | 260 | $236.67 - 260 = -23.33$ | 1 |
| 67 | 269 | $236.67 - 269 = -32.33$ | 1 |

### **4. Xây dựng cây và các ngưỡng**


  ### **Bước 2B: Xây dựng cây - Tìm ngưỡng ứng cử viên**


| Thresh |              
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

### **Bước 2B: Xây dựng cây - Gradients & Hessians cho các node con**

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

---


### **Bước 2B: Xây dựng cây - Gradients & Hessians cho các node con**

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |

* **Ngưỡng phân chia:** Thresh = 34.0
* **Node con bên trái (Age <= 34.0):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 32.67 | 1 |

* **Node con bên phải (Age > 34.0):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 39 | 33.67 | 1 |
| 45 | -13.33 | 1 |
| 48 | 2.67 | 1 |
| 59 | -23.33 | 1 |
| 67 | -32.33 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | 32.67 | 1 |
| **Bên phải** | $33.67 - 13.33 + 2.67 - 23.33 - 32.33 = -32.65$ | $1+1+1+1+1 = 5$ |

---

### **Tiếp tục chọn ngưỡng**

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 34.0):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 32.67 | 1 |
| 39 | 33.67 | 1 |

* **Right Node (Age > 34.0):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 45 | -13.33 | 1 |
| 48 | 2.67 | 1 |
| 59 | -23.33 | 1 |
| 67 | -32.33 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $32.67 + 33.67 = 66.34$ | $1 + 1 = 2$ |
| **Bên phải** | $-13.33 + 2.67 - 23.33 - 32.33 = -66.32$ | $1 + 1 + 1 + 1 = 4$ |

* **Formulas:**

$$G_L = \sum_{i \in I_L} g_i \quad H_L = \sum_{i \in I_L} h_i$$
$$G_R = \sum_{i \in I_R} g_i \quad H_R = \sum_{i \in I_R} h_i$$

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 46.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 53.01 | 3 |
| 39 | | |
| 45 | | |

* **Right Node (Age > 46.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 48 | -52.99 | 3 |
| 59 | | |
| 67 | | |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $32.67 + 33.67 - 13.33 = 53.01$ | $1 + 1 + 1 = 3$ |
| **Bên phải** | $2.67 - 23.33 - 32.33 = -52.99$ | $1 + 1 + 1 = 3$ |

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 46.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 55.68 | 4 |
| 39 | | |
| 45 | | |
| 48 | | |

* **Right Node (Age > 46.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 59 | -55.66 | 2 |
| 67 | | |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $32.67 + 33.67 + (-13.33) + 2.67 = 55.68$ | $1 + 1 + 1 + 1 = 4$ |
| **Bên phải** | $-23.33 + (-32.33) = -55.66$ | $1 + 1 = 2$ |



| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 53.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 32.35 | 5 |
| 39 | | |
| 45 | | |
| 48 | | |
| 59 | | |

* **Right Node (Age > 53.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 67 | -32.33 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $32.67 + 33.67 - 13.33 + 2.67 - 23.33 = 32.35$ | $1 + 1 + 1 + 1 + 1 = 5$ |
| **Bên phải** | $-32.33$ | $1$ |


### **Step 2B: Build Tree – Gain**

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 34.0 | 32.67 | 1 | -32.65 | 5 | |
| 42.0 | 66.34 | 2 | -66.32 | 4 | |
| 46.5 | 53.01 | 3 | -52.99 | 3 | |
| 53.5 | 55.68 | 4 | -55.66 | 2 | |
| 63.0 | 32.35 | 5 | -32.33 | 1 | |

$$
\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

$$
= \frac{1}{2}\left( \frac{32.67^2}{1+1} + \frac{(-32.65)^2}{5+1} - \frac{(32.67-32.65)^2}{1+5+1} \right) - 0
$$

$$
= \frac{1}{2}\left( \frac{1067.3289}{2} + \frac{1065.0225}{6} - \frac{0.0004}{7} \right) \approx 355.67
$$

* **Hyperparameters**

| | |
| :---: | :---: |
| $\eta = 0.5$ | |
| $\lambda = 1.0$ | |
| $\gamma = 0.0$ | |



| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 34.0 | 32.67 | 1 | -32.65 | 5 | 355.67 |
| 42.0 | 66.34 | 2 | -66.32 | 4 | 1173.34 |
| 46.5 | 53.01 | 3 | -52.99 | 3 | 702.26 |
| 53.5 | 55.68 | 4 | -55.66 | 2 | 826.37 |
| 63.0 | 32.35 | 5 | -32.33 | 1 | 348.52 |

$$
\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

### **Step 2B: Build Tree – Split**

Sau khi tính toán gain cho tất cả các threshold, chúng ta chọn threshold có gain cao nhất để split:

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 34.0 | 32.67 | 1 | -32.65 | 5 | 355.67 |
| 42.0 | 66.34 | 2 | -66.32 | 4 | 1173.34 |
| 46.5 | 53.01 | 3 | -52.99 | 3 | 702.26 |
| 53.5 | 55.68 | 4 | -55.66 | 2 | 826.37 |
| 63.0 | 32.35 | 5 | -32.33 | 1 | 348.52 |

The best threshold is **42.0** with a gain of **1173.34**. Since the gain is greater than $\gamma$, we split the node into left and right children.

### **Step 2C: Build Tree – Leaf Weights**

| Age (X) | Chol (y) | Gradients | Hessians |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 32.67 | 1 |
| 39 | 203 | 33.67 | 1 |
| 45 | 250 | -13.33 | 1 |
| 48 | 234 | 2.67 | 1 |
| 59 | 260 | -23.33 | 1 |
| 67 | 269 | -32.33 | 1 |
| $F_0 = 236.67$ | | | |

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 42.0 | 66.34 | 2 | -66.32 | 4 | 1173.34 |


For left node:
$$w_L = - \frac{G_L}{H_L+\lambda} = -\frac{66.34}{2+1} = -22.11$$
For right node:
$$w_R = -\frac{-66.33}{4+1} = 13.27$$


Model update:
$$F_t(x) = F_{t-1}(x) + \eta f_t(x)$$
$$
\eta f_1(x) = \begin{cases}
    \eta w_L & , x \le 42 \\
    \eta w_R & , x > 42
\end{cases}
$$

$$
\eta f_1(x) = \begin{cases}
    0.5 * -22.11= -11.06 & , x \le 42 \\
    0.5 * 13.27= 6.64 & , x > 42
\end{cases}
$$
Update 
$$
F_1(x) = \begin{cases}
    236.67 + (-11.06) = 225.61 & , x \le 42 \\
    236.67 + 6.64 = 243.30 & , x > 42
\end{cases}
$$

### **Step 2C: Build Tree – Model Update**

| Age (X) | Chol (y) | $F_1(X)$ |
| :---: | :---: | :---: |
| 29 | 204 | 225.61 |
| 39 | 203 | 225.61 |
| 45 | 250 | 243.30 |
| 48 | 234 | 243.30 |
| 59 | 260 | 243.30 |
| 67 | 269 | 243.30 |
| $F_0 = 236.67$ | | |

Final model after 1 iteration: $$F_1(x) = \begin{cases}
    225.61 & , x \le 42 \\
    243.30 & , x > 42
\end{cases}$$

---

## **Step 2A (Loop 2): Gradients & Hessians**

Bây giờ chúng ta sẽ tiếp tục với vòng lặp thứ 2, sử dụng kết quả từ vòng lặp 1 để tính toán gradients và hessians mới.

### **Tính Gradients ($g_i$) và Hessians ($h_i$) cho Loop 2**

* **Giá trị dự đoán hiện tại:** $F_1(x)$ từ vòng lặp 1

| Age (X) | Chol (y) | $F_1(x)$ | Gradients ($g_i = F_1(x) - y_i$) | Hessians ($h_i$) |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | $225.61 - 204 = 21.61$ | 1 |
| 39 | 203 | 225.61 | $225.61 - 203 = 22.61$ | 1 |
| 45 | 250 | 243.30 | $243.30 - 250 = -6.70$ | 1 |
| 48 | 234 | 243.30 | $243.30 - 234 = 9.30$ | 1 |
| 59 | 260 | 243.30 | $243.30 - 260 = -16.70$ | 1 |
| 67 | 269 | 243.30 | $243.30 - 269 = -25.70$ | 1 |

### **Bước 2B: Xây dựng cây - Tìm ngưỡng ứng cử viên cho Loop 2**

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

### **Bước 2B: Xây dựng cây - Gradients & Hessians cho các node con (Loop 2)**

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

* **Ngưỡng phân chia:** Thresh = 34.0
* **Node con bên trái (Age <= 34.0):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 21.61 | 1 |

* **Node con bên phải (Age > 34.0):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 39 | 22.61 | 1 |
| 45 | -6.70 | 1 |
| 48 | 9.30 | 1 |
| 59 | -16.70 | 1 |
| 67 | -25.70 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | 21.61 | 1 |
| **Bên phải** | $22.61 - 6.70 + 9.30 - 16.70 - 25.70 = -17.19$ | $1+1+1+1+1 = 5$ |

---

### **Tiếp tục chọn ngưỡng (Loop 2)**

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 34.0):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 21.61 | 1 |
| 39 | 22.61 | 1 |

* **Right Node (Age > 34.0):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 45 | -6.70 | 1 |
| 48 | 9.30 | 1 |
| 59 | -16.70 | 1 |
| 67 | -25.70 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $21.61 + 22.61 = 44.22$ | $1 + 1 = 2$ |
| **Bên phải** | $-6.70 + 9.30 - 16.70 - 25.70 = -39.80$ | $1 + 1 + 1 + 1 = 4$ |

---

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 46.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 37.52 | 3 |
| 39 | | |
| 45 | | |

* **Right Node (Age > 46.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 48 | -33.10 | 3 |
| 59 | | |
| 67 | | |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $21.61 + 22.61 - 6.70 = 37.52$ | $1 + 1 + 1 = 3$ |
| **Bên phải** | $9.30 - 16.70 - 25.70 = -33.10$ | $1 + 1 + 1 = 3$ |

---

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 46.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 44.12 | 4 |
| 39 | | |
| 45 | | |
| 48 | | |

* **Right Node (Age > 46.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 59 | -42.40 | 2 |
| 67 | | |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $21.61 + 22.61 + (-6.70) + 9.30 = 46.82$ | $1 + 1 + 1 + 1 = 4$ |
| **Bên phải** | $-16.70 + (-25.70) = -42.40$ | $1 + 1 = 2$ |

---

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh |
| :---: |
| 34.0 |
| 42.0 |
| 46.5 |
| 53.5 |
| 63.0 |

* **Left Node (Age <= 53.5):**

| Left Id | $G_L$ | $H_L$ |
| :---: | :---: | :---: |
| 29 | 30.12 | 5 |
| 39 | | |
| 45 | | |
| 48 | | |
| 59 | | |

* **Right Node (Age > 53.5):**

| Right Id | $G_R$ | $H_R$ |
| :---: | :---: | :---: |
| 67 | -25.70 | 1 |

* **Tổng hợp cho các node:**

| Node | $\sum G$ | $\sum H$ |
| :---: | :---: | :---: |
| **Bên trái** | $21.61 + 22.61 - 6.70 + 9.30 - 16.70 = 30.12$ | $1 + 1 + 1 + 1 + 1 = 5$ |
| **Bên phải** | $-25.70$ | $1$ |

### **Step 2B: Build Tree – Gain (Loop 2)**

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 34.0 | 21.61 | 1 | -17.19 | 5 | |
| 42.0 | 44.22 | 2 | -39.80 | 4 | |
| 46.5 | 37.52 | 3 | -33.10 | 3 | |
| 53.5 | 46.82 | 4 | -42.40 | 2 | |
| 63.0 | 30.12 | 5 | -25.70 | 1 | |

$$
\text{Gain} = \frac{1}{2}\left( \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right) - \gamma
$$

$$
= \frac{1}{2}\left( \frac{21.61^2}{1+1} + \frac{(-17.19)^2}{5+1} - \frac{(21.61-17.19)^2}{1+5+1} \right) - 0
$$

$$
= \frac{1}{2}\left( \frac{466.9921}{2} + \frac{295.4961}{6} - \frac{19.5364}{7} \right) \approx 150.25
$$

* **Hyperparameters**

| | |
| :---: | :---: |
| $\eta = 0.5$ | |
| $\lambda = 1.0$ | |
| $\gamma = 0.0$ | |

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 34.0 | 21.61 | 1 | -17.19 | 5 | 150.25 |
| 42.0 | 44.22 | 2 | -39.80 | 4 | 485.67 |
| 46.5 | 37.52 | 3 | -33.10 | 3 | 312.45 |
| 53.5 | 46.82 | 4 | -42.40 | 2 | 412.89 |
| 63.0 | 30.12 | 5 | -25.70 | 1 | 198.34 |

### **Step 2B: Build Tree – Split (Loop 2)**

Ngưỡng tốt nhất là **42.0** với gain là **485.67**. Vì gain lớn hơn $\gamma$, chúng ta chia node thành các node con trái và phải.

### **Step 2C: Build Tree – Leaf Weights (Loop 2)**

| Age (X) | Chol (y) | $F_1(x)$ | Gradients | Hessians |
| :---: | :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 21.61 | 1 |
| 39 | 203 | 225.61 | 22.61 | 1 |
| 45 | 250 | 243.30 | -6.70 | 1 |
| 48 | 234 | 243.30 | 9.30 | 1 |
| 59 | 260 | 243.30 | -16.70 | 1 |
| 67 | 269 | 243.30 | -25.70 | 1 |

| Thresh | $G_L$ | $H_L$ | $G_R$ | $H_R$ | Gain |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 42.0 | 44.22 | 2 | -39.80 | 4 | 485.67 |

For left node:
$$w_L = - \frac{G_L}{H_L+\lambda} = -\frac{44.22}{2+1} = -14.74$$

For right node:
$$w_R = -\frac{G_R}{H_R+\lambda} = -\frac{-39.80}{4+1} = 7.96$$

Model update:
$$F_t(x) = F_{t-1}(x) + \eta f_t(x)$$

$$
\eta f_2(x) = \begin{cases}
    \eta w_L & , x \le 42 \\
    \eta w_R & , x > 42
\end{cases}
$$

$$
\eta f_2(x) = \begin{cases}
    0.5 * -14.74 = -7.37 & , x \le 42 \\
    0.5 * 7.96 = 3.98 & , x > 42
\end{cases}
$$

Update 
$$
F_2(x) = \begin{cases}
    225.61 + (-7.37) = 218.24 & , x \le 42 \\
    243.30 + 3.98 = 247.28 & , x > 42
\end{cases}
$$

### **Step 2C: Build Tree – Model Update (Loop 2)**

| Age (X) | Chol (y) | $F_1(x)$ | $F_2(x)$ |
| :---: | :---: | :---: | :---: |
| 29 | 204 | 225.61 | 218.24 |
| 39 | 203 | 225.61 | 218.24 |
| 45 | 250 | 243.30 | 247.28 |
| 48 | 234 | 243.30 | 247.28 |
| 59 | 260 | 243.30 | 247.28 |
| 67 | 269 | 243.30 | 247.28 |

Final model after 2 iterations: $$F_2(x) = \begin{cases}
    218.24 & , x \le 42 \\
    247.28 & , x > 42
\end{cases}$$

---

## **Tiếp tục với các vòng lặp tiếp theo...**

Quá trình này sẽ tiếp tục cho đến khi đạt được số vòng lặp mong muốn hoặc khi không còn cải thiện đáng kể nào trong hàm mất mát. Mỗi vòng lặp sẽ:

1. **Tính gradients và hessians** dựa trên model hiện tại
2. **Tìm ngưỡng tối ưu** để chia node
3. **Tính leaf weights** cho các node con
4. **Cập nhật model** với learning rate $\eta$

Điều này cho thấy cách XGBoost xây dựng từng cây một cách tuần tự, mỗi cây mới sẽ học từ lỗi của các cây trước đó để cải thiện dự đoán.

<!-- ### **Đạo hàm của Loss Function Classification**

* **Hàm mất mát (Loss Function):**
    $$
    \mathcal{L}(Y_i, \log(odds)) = -Y_i\log(odds) + \log(1 + e^{\log(odds)})
    $$

* **Đạo hàm bậc nhất ($g_i$):**
    $$
    g_i = \frac{d\mathcal{L}}{d\log(odds)} = -Y_i + \frac{e^{\log(odds)}}{1 + e^{\log(odds)}} = -Y_i + \bar{Y}_i = -(Y_i - \bar{Y}_i)
    $$

* **Đạo hàm bậc hai ($h_i$):**
    $$
    h_i = \frac{d^2\mathcal{L}}{d\log(odds)^2} = \frac{d}{d\log(odds)}\left(-Y_i + \frac{e^{\log(odds)}}{1 + e^{\log(odds)}}\right) = \frac{e^{\log(odds)}(1 + e^{\log(odds)}) - e^{\log(odds)}e^{\log(odds)}}{(1 + e^{\log(odds)})^2} = \frac{e^{\log(odds)}}{(1 + e^{\log(odds)})^2} = \bar{Y}_i(1-\bar{Y}_i)
    $$ -->
