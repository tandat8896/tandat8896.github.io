---
title: "Gradient Boosting Classification - Thuật toán Gradient Boosting cho Classification"
pubDatetime: 2025-09-21T20:00:00Z
featured: false
description: "Tìm hiểu chi tiết về Gradient Boosting cho Classification, từ hàm mục tiêu đến quá trình xây dựng weak learners"
tags: ["machine-learning", "gradient-boosting", "classification", "boosting", "algorithm"]
---

# Gradient Boosting Classification

## Thuật toán Gradient Boosting cho Classification

Gradient Boosting cho Classification sử dụng **log loss** làm hàm mất mát và **log(odds)** làm target để tối ưu hóa. Khác với regression, classification cần chuyển đổi giữa probabilities và log(odds).

---

## Chuyển đổi Log-Odds sang Xác suất và Hàm Sigmoid

Trong Gradient Boosting Classification, chúng ta làm việc với **log-odds** (log tỷ lệ cược) thay vì xác suất trực tiếp. Việc chuyển đổi giữa log-odds và xác suất được thực hiện thông qua hàm sigmoid (logistic function).

### 1. Định nghĩa Log-Odds

**Log-odds** được định nghĩa là:

$$\log\left(\frac{p}{1-p}\right) = \log(odds)$$

Trong đó:
- $p$ là xác suất của sự kiện xảy ra
- $odds = \frac{p}{1-p}$ là tỷ lệ cược
- $\log(odds)$ là log-odds

### 2. Chuyển đổi từ Log-Odds sang Xác suất

Để chuyển đổi từ log-odds sang xác suất, chúng ta thực hiện các bước sau:

**Bước 1: Lũy thừa hóa cả hai vế**
$$\Rightarrow \frac{p}{1-p} = e^{\log(odds)}$$

**Bước 2: Nhân cả hai vế với $(1-p)$**
$$\Rightarrow p = (1-p) \cdot e^{\log(odds)}$$

**Bước 3: Nhân phân phối**
$$\Rightarrow p = e^{\log(odds)} - p \cdot e^{\log(odds)}$$

**Bước 4: Cộng $p \cdot e^{\log(odds)}$ vào cả hai vế**
$$\Rightarrow p + p \cdot e^{\log(odds)} = e^{\log(odds)}$$

**Bước 5: Đặt $p$ làm nhân tử chung**
$$\Rightarrow p \cdot (1 + e^{\log(odds)}) = e^{\log(odds)}$$

**Bước 6: Chia cả hai vế cho $(1 + e^{\log(odds)})$**
$$\Rightarrow p = \frac{e^{\log(odds)}}{1 + e^{\log(odds)}}$$

### 3. Hàm Sigmoid (Logistic Function)

Kết quả cuối cùng là công thức của hàm sigmoid:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó $z = \log(odds)$.

**Tính chất của hàm sigmoid:**
- **Miền giá trị:** $(0, 1)$
- **Điểm uốn:** tại $z = 0$ (khi đó $\sigma(0) = 0.5$)
- **Đối xứng:** $\sigma(-z) = 1 - \sigma(z)$
- **Đạo hàm:** $\sigma'(z) = \sigma(z)(1-\sigma(z))$

---

## Log-Likelihood Function và Optimization

Trong Logistic Regression và Gradient Boosting Classification, chúng ta sử dụng **log-likelihood** làm hàm mục tiêu để tối ưu hóa.

### 1. Log-Likelihood Function

**Log-likelihood của dữ liệu quan sát được:**

$$\sum_{i=1}^{N} y_i \times \log(p) + (1 - y_i) \times \log(1 - p)$$

Trong đó:
- $N$ là tổng số mẫu dữ liệu
- $y_i$ là nhãn thực của mẫu $i$ (0 hoặc 1)
- $p$ là xác suất dự đoán của lớp dương (ví dụ: $P(Y=1|X)$)

**Ví dụ tính toán:**
Nếu $p = 0.67$ và có 3 mẫu với $y = [1, 1, 0]$:
$$= \log(0.67) + \log(0.67) + \log(1 - 0.67)$$

### 2. Chuyển đổi từ Maximization sang Minimization

**Mục tiêu:** Trong Logistic Regression, mục tiêu là **tối đa hóa** log-likelihood.

**Chuyển đổi:** Để sử dụng gradient descent, chúng ta chuyển thành bài toán **tối thiểu hóa** bằng cách lấy âm của log-likelihood:

$$(-1) \sum_{i=1}^{N} y_i \times \log(p) + (1 - y_i) \times \log(1 - p)$$

**Lý do:** Gradient descent tìm điểm cực tiểu, nên chúng ta cần đảo dấu để chuyển từ maximization sang minimization.

---

## Đạo hàm Logistic Loss Function theo Log-Odds

Để tối ưu hóa trong Gradient Boosting, chúng ta cần biểu diễn logistic loss function theo **log-odds** thay vì xác suất.

### 1. Định nghĩa Log-Odds

$$\log(p) - \log(1-p) = \log\left(\frac{p}{1-p}\right) = \log(odds)$$

### 2. Đạo hàm $\log(1-p)$ theo Log-Odds

**Bước 1:** Biểu diễn $p$ theo log-odds
$$p = \frac{e^{\log(odds)}}{1 + e^{\log(odds)}}$$

**Bước 2:** Tính $\log(1-p)$
$$\log(1-p) = \log\left(1 - \frac{e^{\log(odds)}}{1 + e^{\log(odds)}}\right)$$

**Bước 3:** Tìm mẫu số chung
$$= \log\left(\frac{1 + e^{\log(odds)} - e^{\log(odds)}}{1 + e^{\log(odds)}}\right)$$

**Bước 4:** Đơn giản hóa tử số
$$= \log\left(\frac{1}{1 + e^{\log(odds)}}\right)$$

**Bước 5:** Sử dụng tính chất $\log(A/B) = \log(A) - \log(B)$
$$= \log(1) - \log(1 + e^{\log(odds)})$$

**Bước 6:** Vì $\log(1) = 0$
$$\log(1-p) = -\log(1 + e^{\log(odds)})$$

### 3. Đạo hàm Logistic Loss theo Log-Odds

**Bước 1:** Logistic loss cho một mẫu $i$
$$(-1)[y_i \times \log(p) + (1 - y_i) \times \log(1-p)]$$

**Bước 2:** Thay $y_i$ bằng $Observed$
$$\Rightarrow (-1)[Observed \times \log(p) + (1 - Observed) \times \log(1-p)]$$

**Bước 3:** Phân phối $(-1)$
$$\Rightarrow -Observed \times \log(p) - (1 - Observed) \times \log(1-p)$$

**Bước 4:** Khai triển số hạng thứ hai
$$\Rightarrow -Observed \times \log(p) - \log(1-p) + Observed \times \log(1-p)$$

**Bước 5:** Đặt $Observed$ làm nhân tử chung
$$\Rightarrow -Observed[\log(p) - \log(1-p)] - \log(1-p)$$

**Bước 6:** Thay $[\log(p) - \log(1-p)]$ bằng $\log(odds)$
$$\Rightarrow -Observed \times \log(odds) - \log(1-p)$$

**Bước 7:** Thay $-\log(1-p)$ bằng $\log(1 + e^{\log(odds)})$
$$\Rightarrow -Observed \times \log(odds) + \log(1 + e^{\log(odds)})$$

### 4. Công thức cuối cùng

**Logistic Loss Function theo Log-Odds:**
$$L(y, \log(odds)) = -y \times \log(odds) + \log(1 + e^{\log(odds)})$$

Đây là công thức chuẩn được sử dụng trong Gradient Boosting cho Classification, cho phép tối ưu hóa trực tiếp trên log-odds thay vì xác suất.

---

## Taylor Approximation và Newton-Raphson Method

Trong Gradient Boosting Classification, việc tối ưu hóa trực tiếp hàm logistic loss phức tạp là rất khó khăn. Chúng ta sử dụng **Taylor Approximation** và **Newton-Raphson Method** để đơn giản hóa quá trình tối ưu hóa.

### 1. Vấn đề tối ưu hóa phức tạp

**Bước 2 (C): Tính toán $\gamma_{jm}$**

Đối với mỗi leaf node $j = 1, \ldots, J_m$, chúng ta cần tính:

$$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_j} L(y_i, F_{m-1}(x_i) + \gamma)$$

Trong đó:
- $R_j$ là vùng dữ liệu thuộc về leaf node $j$
- $L(y_i, F_{m-1}(x_i) + \gamma)$ là logistic loss function
- $\gamma$ là giá trị output của leaf node

**Hàm logistic loss phức tạp:**
$$L(y_i, F_{m-1}(x_i) + \gamma) = -y_i \times [F_{m-1}(x_i) + \gamma] + \log(1 + e^{F_{m-1}(x_i)+\gamma})$$

**Vấn đề:** Việc lấy đạo hàm của hàm loss này rất khó khăn. Chúng ta cần **xấp xỉ hàm loss bằng Taylor polynomial bậc hai**.

### 2. Taylor Approximation bậc hai

**Taylor expansion của hàm loss:**
$$L(y_i, F_{m-1}(x_i) + \gamma) \approx L(y_i, F_{m-1}(x_i)) + \frac{d}{dF}L(y_i, F_{m-1}(x_i)) \cdot \gamma + \frac{1}{2} \frac{d^2}{dF^2}L(y_i, F_{m-1}(x_i)) \cdot \gamma^2$$

**Đạo hàm của approximation:**
$$\frac{d}{d\gamma}L(y_i, F_{m-1}(x_i) + \gamma) \approx \frac{d}{dF}L(y_i, F_{m-1}(x_i)) + \frac{d^2}{dF^2}L(y_i, F_{m-1}(x_i)) \cdot \gamma$$

**Đặt đạo hàm bằng 0 để tối ưu:**
$$\Rightarrow \frac{d}{dF}L(y_i, F_{m-1}(x_i)) + \frac{d^2}{dF^2}L(y_i, F_{m-1}(x_i)) \cdot \gamma = 0$$

### 3. Đạo hàm bậc nhất và bậc hai của Logistic Loss

**Đạo hàm bậc nhất:**
$$\frac{d}{dF}L(y, F) = \frac{d}{dF}[-y \times F + \log(1 + e^F)] = -y + \frac{e^F}{1 + e^F} = -y + p$$

Trong đó $p = \frac{e^F}{1 + e^F}$ là xác suất dự đoán.

**Đạo hàm bậc hai:**
$$\frac{d^2}{dF^2}L(y, F) = \frac{d}{dF}[-y + p] = \frac{d}{dF}\left[\frac{e^F}{1 + e^F}\right]$$

**Tính toán chi tiết đạo hàm bậc hai:**

Sử dụng quy tắc đạo hàm tích: $\frac{d}{dF}[u \cdot v] = u'v + uv'$

Với $u = e^F$ và $v = (1 + e^F)^{-1}$:

- $u' = e^F$
- $v' = -1 \cdot (1 + e^F)^{-2} \cdot e^F = -\frac{e^F}{(1 + e^F)^2}$

$$\frac{d^2}{dF^2}L(y, F) = e^F \cdot (1 + e^F)^{-1} + e^F \cdot \left(-\frac{e^F}{(1 + e^F)^2}\right)$$

$$= \frac{e^F}{1 + e^F} - \frac{e^{2F}}{(1 + e^F)^2}$$

$$= \frac{e^F(1 + e^F) - e^{2F}}{(1 + e^F)^2}$$

$$= \frac{e^F + e^{2F} - e^{2F}}{(1 + e^F)^2}$$

$$= \frac{e^F}{(1 + e^F)^2}$$

$$= \frac{e^F}{1 + e^F} \cdot \frac{1}{1 + e^F}$$

$$= p \cdot (1 - p)$$

### 4. Công thức Newton-Raphson cho Optimal $\gamma$

**Từ phương trình tối ưu:**
$$\sum_{x_i \in R_j} \frac{d}{dF}L(y_i, F_{m-1}(x_i)) + \sum_{x_i \in R_j} \frac{d^2}{dF^2}L(y_i, F_{m-1}(x_i)) \cdot \gamma = 0$$

**Thay thế các đạo hàm:**
$$\sum_{x_i \in R_j} (y_i - p_i) + \sum_{x_i \in R_j} p_i(1 - p_i) \cdot \gamma = 0$$

**Giải cho $\gamma$:**
$$\gamma_{jm} = -\frac{\sum_{x_i \in R_j} (y_i - p_i)}{\sum_{x_i \in R_j} p_i(1 - p_i)}$$

**Công thức cuối cùng:**
$$\gamma_{jm} = \frac{\sum_{x_i \in R_j} \text{Residual}_i}{\sum_{x_i \in R_j} p_i(1 - p_i)}$$

Trong đó:
- $\text{Residual}_i = y_i - p_i$ là pseudo-residual
- $p_i(1 - p_i)$ là Hessian (đạo hàm bậc hai)

### 5. Ví dụ tính toán cụ thể

**Giả sử có 2 điểm dữ liệu trong leaf node:**
- Điểm 1: $y_1 = 1$, $p_1 = 0.67$ → $\text{Residual}_1 = 1 - 0.67 = 0.33$
- Điểm 2: $y_2 = 0$, $p_2 = 0.67$ → $\text{Residual}_2 = 0 - 0.67 = -0.67$

**Tính toán:**
$$\gamma = \frac{0.33 + (-0.67)}{0.67 \times (1-0.67) + 0.67 \times (1-0.67)}$$

$$= \frac{-0.34}{0.67 \times 0.33 + 0.67 \times 0.33}$$

$$= \frac{-0.34}{0.2211 + 0.2211}$$

$$= \frac{-0.34}{0.4422}$$

$$= -0.77$$

### 6. Công thức tổng quát

**Công thức chung cho optimal leaf value:**
$$\gamma_{jm} = \frac{\sum_{x_i \in R_j} \text{Residual}_i}{\sum_{x_i \in R_j} p_i(1 - p_i)}$$

**Ý nghĩa:**
- **Tử số:** Tổng các pseudo-residual (lỗi dự đoán)
- **Mẫu số:** Tổng các Hessian (độ cong của hàm loss)
- **Kết quả:** Giá trị tối ưu cho leaf node để giảm thiểu loss

---

## Công thức toán học của Gradient Boosting Classification

### **Bước 1: Khởi tạo model**

**Model ban đầu:**
$$F_0(x) = \log\left(\frac{p}{1-p}\right) \quad \text{where } p = \frac{\sum_{i=1}^{N} y_i}{N}$$

**Chuyển đổi log(odds) thành probability:**
$$\hat{y}_i = \frac{1}{1 + e^{-F_0(x)}}$$

### **Bước 2: Vòng lặp cho m = 1 đến M**

#### **2a. Tính residuals (pseudo-residuals)**

**Residuals cho classification:**
$$r_{i,m} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} = y_i - \hat{y}_{i,m-1}$$

**Với log loss:**
$$L(y_i, \hat{y}_i) = -y_i\log(\hat{y}_i) - (1-y_i)\log(1-\hat{y}_i)$$

#### **2b. Huấn luyện weak learner trên residuals**

$$h_m(x) = \arg\min_{h} \sum_{i=1}^{N} (r_{i,m} - h(x_i))^2$$

#### **2c. Tính learning rate (step size)**

$$\gamma_m = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \sigma(F_{m-1}(x_i) + \gamma h_m(x_i)))$$

**Với sigmoid function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

#### **2d. Cập nhật model**

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

### **Bước 3: Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \gamma_m h_m(x)$$

**Final predictions:**
$$\hat{y}_i = \frac{1}{1 + e^{-F_M(x_i)}}$$

---

## Ví Dụ Tính Tay - Gradient Boosting Classification

### **Dataset Classification**

| ID | Age (X) | Income (X2) | Label (y) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 |
| 2 | 30 | 50 | 1 |
| 3 | 35 | 40 | 0 |
| 4 | 40 | 60 | 0 |
| 5 | 45 | 70 | 1 |
| 6 | 50 | 80 | 0 |

**Mục tiêu:** Dự đoán label dựa trên Age và Income với 3 iterations

---

### **Step 1: Initialization**

**F0(x) là giá trị tối ưu cho log(odds):**
$$F_0 = \log\left(\frac{p}{1-p}\right) \quad \text{where } p = \frac{\sum_{i=1}^{6} y_i}{6} = \frac{3}{6} = 0.5$$

$$F_0 = \log\left(\frac{0.5}{1-0.5}\right) = \log(1) = 0$$

**Initial predictions (probabilities):**
$$\hat{y}_i = \frac{1}{1 + e^{-F_0}} = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = 0.5$$

| ID | Age | Income | Label | F0(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0 | 0.5 |
| 2 | 30 | 50 | 1 | 0 | 0.5 |
| 3 | 35 | 40 | 0 | 0 | 0.5 |
| 4 | 40 | 60 | 0 | 0 | 0.5 |
| 5 | 45 | 70 | 1 | 0 | 0.5 |
| 6 | 50 | 80 | 0 | 0 | 0.5 |

---

### **Step 2: Iteration 1 (m = 1)**

#### **Step 2a: Calculate Residuals**

**Residuals:**
$$r_{i,1} = y_i - \hat{y}_{i,0} = y_i - 0.5$$

| ID | Age | Income | Label | $\hat{y}_i$ | Residuals (r1) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.5 | 1 - 0.5 = 0.5 |
| 2 | 30 | 50 | 1 | 0.5 | 1 - 0.5 = 0.5 |
| 3 | 35 | 40 | 0 | 0.5 | 0 - 0.5 = -0.5 |
| 4 | 40 | 60 | 0 | 0.5 | 0 - 0.5 = -0.5 |
| 5 | 45 | 70 | 1 | 0.5 | 1 - 0.5 = 0.5 |
| 6 | 50 | 80 | 0 | 0.5 | 0 - 0.5 = -0.5 |

#### **Step 2b: Train Weak Learner h1(x) trên Residuals**

**Tìm threshold tốt nhất cho Age trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [0.5] | [0.5,-0.5,-0.5,0.5,-0.5] | 0 |
| 32.5 | [1,2] | [3,4,5,6] | [0.5,0.5] | [-0.5,-0.5,0.5,-0.5] | 0 |
| 37.5 | [1,2,3] | [4,5,6] | [0.5,0.5,-0.5] | [-0.5,0.5,-0.5] | 0 |
| 42.5 | [1,2,3,4] | [5,6] | [0.5,0.5,-0.5,-0.5] | [0.5,-0.5] | 0 |
| 47.5 | [1,2,3,4,5] | [6] | [0.5,0.5,-0.5,-0.5,0.5] | [-0.5] | 0 |

**Tất cả thresholds đều có error = 0, chọn threshold = 32.5**

**h1(x) Structure:**
```
Root: Age <= 32.5?
├── Yes: Average residual = (0.5+0.5)/2 = 0.5
└── No: Average residual = (-0.5-0.5+0.5-0.5)/4 = -0.25
```

#### **Step 2c: Calculate Learning Rate γ1**

**Tìm γ1 để minimize log loss:**
$$\gamma_1 = \arg\min_{\gamma} \sum_{i=1}^{6} L(y_i, \sigma(F_0(x_i) + \gamma h_1(x_i)))$$

**Với γ = 0.1:**
| ID | Age | F0(x) | h1(x) | F0(x) + 0.1×h1(x) | σ(F0 + 0.1×h1) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 0 | 0.5 | 0 + 0.1×0.5 = 0.05 | 0.512 | 1 | -log(0.512) = 0.669 |
| 2 | 30 | 0 | 0.5 | 0 + 0.1×0.5 = 0.05 | 0.512 | 1 | -log(0.512) = 0.669 |
| 3 | 35 | 0 | -0.25 | 0 + 0.1×(-0.25) = -0.025 | 0.494 | 0 | -log(1-0.494) = 0.669 |
| 4 | 40 | 0 | -0.25 | 0 + 0.1×(-0.25) = -0.025 | 0.494 | 0 | -log(1-0.494) = 0.669 |
| 5 | 45 | 0 | -0.25 | 0 + 0.1×(-0.25) = -0.025 | 0.494 | 1 | -log(0.494) = 0.705 |
| 6 | 50 | 0 | -0.25 | 0 + 0.1×(-0.25) = -0.025 | 0.494 | 0 | -log(1-0.494) = 0.669 |

**Total Log Loss = 0.669 + 0.669 + 0.669 + 0.669 + 0.705 + 0.669 = 4.050**

**Với γ = 0.5:**
| ID | Age | F0(x) | h1(x) | F0(x) + 0.5×h1(x) | σ(F0 + 0.5×h1) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 0 | 0.5 | 0 + 0.5×0.5 = 0.25 | 0.562 | 1 | -log(0.562) = 0.576 |
| 2 | 30 | 0 | 0.5 | 0 + 0.5×0.5 = 0.25 | 0.562 | 1 | -log(0.562) = 0.576 |
| 3 | 35 | 0 | -0.25 | 0 + 0.5×(-0.25) = -0.125 | 0.469 | 0 | -log(1-0.469) = 0.576 |
| 4 | 40 | 0 | -0.25 | 0 + 0.5×(-0.25) = -0.125 | 0.469 | 0 | -log(1-0.469) = 0.576 |
| 5 | 45 | 0 | -0.25 | 0 + 0.5×(-0.25) = -0.125 | 0.469 | 1 | -log(0.469) = 0.756 |
| 6 | 50 | 0 | -0.25 | 0 + 0.5×(-0.25) = -0.125 | 0.469 | 0 | -log(1-0.469) = 0.576 |

**Total Log Loss = 0.576 + 0.576 + 0.576 + 0.576 + 0.756 + 0.576 = 3.636**

**Với γ = 1.0:**
| ID | Age | F0(x) | h1(x) | F0(x) + 1.0×h1(x) | σ(F0 + 1.0×h1) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 0 | 0.5 | 0 + 1.0×0.5 = 0.5 | 0.622 | 1 | -log(0.622) = 0.474 |
| 2 | 30 | 0 | 0.5 | 0 + 1.0×0.5 = 0.5 | 0.622 | 1 | -log(0.622) = 0.474 |
| 3 | 35 | 0 | -0.25 | 0 + 1.0×(-0.25) = -0.25 | 0.438 | 0 | -log(1-0.438) = 0.474 |
| 4 | 40 | 0 | -0.25 | 0 + 1.0×(-0.25) = -0.25 | 0.438 | 0 | -log(1-0.438) = 0.474 |
| 5 | 45 | 0 | -0.25 | 0 + 1.0×(-0.25) = -0.25 | 0.438 | 1 | -log(0.438) = 0.826 |
| 6 | 50 | 0 | -0.25 | 0 + 1.0×(-0.25) = -0.25 | 0.438 | 0 | -log(1-0.438) = 0.474 |

**Total Log Loss = 0.474 + 0.474 + 0.474 + 0.474 + 0.826 + 0.474 = 3.196**

**Best γ1 = 1.0 (Log Loss = 3.196)**

#### **Step 2d: Update Model**

$$F_1(x) = F_0(x) + \gamma_1 h_1(x) = 0 + 1.0 \times h_1(x)$$

**F1(x) Structure:**
```
F1(x) = h1(x)
h1(x) = 0.5 if Age <= 32.5, else -0.25
```

**Updated predictions:**
| ID | Age | Income | Label | F1(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.5 | 0.622 |
| 2 | 30 | 50 | 1 | 0.5 | 0.622 |
| 3 | 35 | 40 | 0 | -0.25 | 0.438 |
| 4 | 40 | 60 | 0 | -0.25 | 0.438 |
| 5 | 45 | 70 | 1 | -0.25 | 0.438 |
| 6 | 50 | 80 | 0 | -0.25 | 0.438 |

---

### **Step 3: Iteration 2 (m = 2)**

#### **Step 3a: Calculate Residuals**

**Residuals:**
$$r_{i,2} = y_i - \hat{y}_{i,1}$$

| ID | Age | Income | Label | $\hat{y}_i$ | Residuals (r2) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.622 | 1 - 0.622 = 0.378 |
| 2 | 30 | 50 | 1 | 0.622 | 1 - 0.622 = 0.378 |
| 3 | 35 | 40 | 0 | 0.438 | 0 - 0.438 = -0.438 |
| 4 | 40 | 60 | 0 | 0.438 | 0 - 0.438 = -0.438 |
| 5 | 45 | 70 | 1 | 0.438 | 1 - 0.438 = 0.562 |
| 6 | 50 | 80 | 0 | 0.438 | 0 - 0.438 = -0.438 |

#### **Step 3b: Train Weak Learner h2(x) trên Residuals**

**Tìm threshold tốt nhất cho Income trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | [1] | [2,3,4,5,6] | [0.378] | [0.378,-0.438,-0.438,0.562,-0.438] | 0 |
| 45 | [1,2,3] | [4,5,6] | [0.378,0.378,-0.438] | [-0.438,0.562,-0.438] | 0 |
| 55 | [1,2,3,4] | [5,6] | [0.378,0.378,-0.438,-0.438] | [0.562,-0.438] | 0 |
| 65 | [1,2,3,4,5] | [6] | [0.378,0.378,-0.438,-0.438,0.562] | [-0.438] | 0 |
| 75 | [1,2,3,4,5,6] | [] | [0.378,0.378,-0.438,-0.438,0.562,-0.438] | [] | 0 |

**Chọn threshold = 45**

**h2(x) Structure:**
```
Root: Income <= 45?
├── Yes: Average residual = (0.378+0.378-0.438)/3 = 0.106
└── No: Average residual = (-0.438+0.562-0.438)/3 = -0.105
```

#### **Step 3c: Calculate Learning Rate γ2**

**Tìm γ2 để minimize log loss:**
$$\gamma_2 = \arg\min_{\gamma} \sum_{i=1}^{6} L(y_i, \sigma(F_1(x_i) + \gamma h_2(x_i)))$$

**Với γ = 0.5:**
| ID | Age | Income | F1(x) | h2(x) | F1(x) + 0.5×h2(x) | σ(F1 + 0.5×h2) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 0.5 | 0.106 | 0.5 + 0.5×0.106 = 0.553 | 0.635 | 1 | -log(0.635) = 0.455 |
| 2 | 30 | 50 | 0.5 | 0.106 | 0.5 + 0.5×0.106 = 0.553 | 0.635 | 1 | -log(0.635) = 0.455 |
| 3 | 35 | 40 | -0.25 | 0.106 | -0.25 + 0.5×0.106 = -0.197 | 0.451 | 0 | -log(1-0.451) = 0.455 |
| 4 | 40 | 60 | -0.25 | -0.105 | -0.25 + 0.5×(-0.105) = -0.303 | 0.425 | 0 | -log(1-0.425) = 0.455 |
| 5 | 45 | 70 | -0.25 | -0.105 | -0.25 + 0.5×(-0.105) = -0.303 | 0.425 | 1 | -log(0.425) = 0.856 |
| 6 | 50 | 80 | -0.25 | -0.105 | -0.25 + 0.5×(-0.105) = -0.303 | 0.425 | 0 | -log(1-0.425) = 0.455 |

**Total Log Loss = 0.455 + 0.455 + 0.455 + 0.455 + 0.856 + 0.455 = 3.131**

**Với γ = 1.0:**
| ID | Age | Income | F1(x) | h2(x) | F1(x) + 1.0×h2(x) | σ(F1 + 1.0×h2) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 0.5 | 0.106 | 0.5 + 1.0×0.106 = 0.606 | 0.647 | 1 | -log(0.647) = 0.435 |
| 2 | 30 | 50 | 0.5 | 0.106 | 0.5 + 1.0×0.106 = 0.606 | 0.647 | 1 | -log(0.647) = 0.435 |
| 3 | 35 | 40 | -0.25 | 0.106 | -0.25 + 1.0×0.106 = -0.144 | 0.464 | 0 | -log(1-0.464) = 0.435 |
| 4 | 40 | 60 | -0.25 | -0.105 | -0.25 + 1.0×(-0.105) = -0.355 | 0.412 | 0 | -log(1-0.412) = 0.435 |
| 5 | 45 | 70 | -0.25 | -0.105 | -0.25 + 1.0×(-0.105) = -0.355 | 0.412 | 1 | -log(0.412) = 0.886 |
| 6 | 50 | 80 | -0.25 | -0.105 | -0.25 + 1.0×(-0.105) = -0.355 | 0.412 | 0 | -log(1-0.412) = 0.435 |

**Total Log Loss = 0.435 + 0.435 + 0.435 + 0.435 + 0.886 + 0.435 = 3.061**

**Best γ2 = 1.0 (Log Loss = 3.061)**

#### **Step 3d: Update Model**

$$F_2(x) = F_1(x) + \gamma_2 h_2(x) = F_1(x) + 1.0 \times h_2(x)$$

**F2(x) Structure:**
```
F2(x) = F1(x) + h2(x)
F1(x) = 0.5 if Age <= 32.5, else -0.25
h2(x) = 0.106 if Income <= 45, else -0.105
```

**Updated predictions:**
| ID | Age | Income | Label | F2(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.5 + 0.106 = 0.606 | 0.647 |
| 2 | 30 | 50 | 1 | 0.5 + 0.106 = 0.606 | 0.647 |
| 3 | 35 | 40 | 0 | -0.25 + 0.106 = -0.144 | 0.464 |
| 4 | 40 | 60 | 0 | -0.25 + (-0.105) = -0.355 | 0.412 |
| 5 | 45 | 70 | 1 | -0.25 + (-0.105) = -0.355 | 0.412 |
| 6 | 50 | 80 | 0 | -0.25 + (-0.105) = -0.355 | 0.412 |

---

### **Step 4: Iteration 3 (m = 3)**

#### **Step 4a: Calculate Residuals**

**Residuals:**
$$r_{i,3} = y_i - \hat{y}_{i,2}$$

| ID | Age | Income | Label | $\hat{y}_i$ | Residuals (r3) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.647 | 1 - 0.647 = 0.353 |
| 2 | 30 | 50 | 1 | 0.647 | 1 - 0.647 = 0.353 |
| 3 | 35 | 40 | 0 | 0.464 | 0 - 0.464 = -0.464 |
| 4 | 40 | 60 | 0 | 0.412 | 0 - 0.412 = -0.588 |
| 5 | 45 | 70 | 1 | 0.412 | 1 - 0.412 = 0.588 |
| 6 | 50 | 80 | 0 | 0.412 | 0 - 0.412 = -0.588 |

#### **Step 4b: Train Weak Learner h3(x) trên Residuals**

**Tìm threshold tốt nhất cho Age trên residuals:**

| Threshold | Left (≤) | Right (>) | Left Residuals | Right Residuals | Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6] | [0.353] | [0.353,-0.464,-0.588,0.588,-0.588] | 0 |
| 32.5 | [1,2] | [3,4,5,6] | [0.353,0.353] | [-0.464,-0.588,0.588,-0.588] | 0 |
| 37.5 | [1,2,3] | [4,5,6] | [0.353,0.353,-0.464] | [-0.588,0.588,-0.588] | 0 |
| 42.5 | [1,2,3,4] | [5,6] | [0.353,0.353,-0.464,-0.588] | [0.588,-0.588] | 0 |
| 47.5 | [1,2,3,4,5] | [6] | [0.353,0.353,-0.464,-0.588,0.588] | [-0.588] | 0 |

**Chọn threshold = 42.5**

**h3(x) Structure:**
```
Root: Age <= 42.5?
├── Yes: Average residual = (0.353+0.353-0.464-0.588)/4 = -0.087
└── No: Average residual = (0.588-0.588)/2 = 0.0
```

#### **Step 4c: Calculate Learning Rate γ3**

**Tìm γ3 để minimize log loss:**
$$\gamma_3 = \arg\min_{\gamma} \sum_{i=1}^{6} L(y_i, \sigma(F_2(x_i) + \gamma h_3(x_i)))$$

**Với γ = 0.5:**
| ID | Age | Income | F2(x) | h3(x) | F2(x) + 0.5×h3(x) | σ(F2 + 0.5×h3) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 0.606 | -0.087 | 0.606 + 0.5×(-0.087) = 0.563 | 0.637 | 1 | -log(0.637) = 0.451 |
| 2 | 30 | 50 | 0.606 | -0.087 | 0.606 + 0.5×(-0.087) = 0.563 | 0.637 | 1 | -log(0.637) = 0.451 |
| 3 | 35 | 40 | -0.144 | -0.087 | -0.144 + 0.5×(-0.087) = -0.188 | 0.453 | 0 | -log(1-0.453) = 0.451 |
| 4 | 40 | 60 | -0.355 | -0.087 | -0.355 + 0.5×(-0.087) = -0.399 | 0.401 | 0 | -log(1-0.401) = 0.451 |
| 5 | 45 | 70 | -0.355 | 0.0 | -0.355 + 0.5×0.0 = -0.355 | 0.412 | 1 | -log(0.412) = 0.886 |
| 6 | 50 | 80 | -0.355 | 0.0 | -0.355 + 0.5×0.0 = -0.355 | 0.412 | 0 | -log(1-0.412) = 0.451 |

**Total Log Loss = 0.451 + 0.451 + 0.451 + 0.451 + 0.886 + 0.451 = 3.141**

**Với γ = 1.0:**
| ID | Age | Income | F2(x) | h3(x) | F2(x) + 1.0×h3(x) | σ(F2 + 1.0×h3) | y | Log Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 0.606 | -0.087 | 0.606 + 1.0×(-0.087) = 0.519 | 0.627 | 1 | -log(0.627) = 0.467 |
| 2 | 30 | 50 | 0.606 | -0.087 | 0.606 + 1.0×(-0.087) = 0.519 | 0.627 | 1 | -log(0.627) = 0.467 |
| 3 | 35 | 40 | -0.144 | -0.087 | -0.144 + 1.0×(-0.087) = -0.231 | 0.442 | 0 | -log(1-0.442) = 0.467 |
| 4 | 40 | 60 | -0.355 | -0.087 | -0.355 + 1.0×(-0.087) = -0.442 | 0.391 | 0 | -log(1-0.391) = 0.467 |
| 5 | 45 | 70 | -0.355 | 0.0 | -0.355 + 1.0×0.0 = -0.355 | 0.412 | 1 | -log(0.412) = 0.886 |
| 6 | 50 | 80 | -0.355 | 0.0 | -0.355 + 1.0×0.0 = -0.355 | 0.412 | 0 | -log(1-0.412) = 0.451 |

**Total Log Loss = 0.467 + 0.467 + 0.467 + 0.467 + 0.886 + 0.451 = 3.205**

**Best γ3 = 0.5 (Log Loss = 3.141)**

#### **Step 4d: Update Model**

$$F_3(x) = F_2(x) + \gamma_3 h_3(x) = F_2(x) + 0.5 \times h_3(x)$$

**F3(x) Structure:**
```
F3(x) = F2(x) + 0.5×h3(x)
F2(x) = F1(x) + h2(x)
F1(x) = 0.5 if Age <= 32.5, else -0.25
h2(x) = 0.106 if Income <= 45, else -0.105
h3(x) = -0.087 if Age <= 42.5, else 0.0
```

**Final predictions:**
| ID | Age | Income | Label | F3(x) | $\hat{y}_i$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.606 + 0.5×(-0.087) = 0.563 | 0.637 |
| 2 | 30 | 50 | 1 | 0.606 + 0.5×(-0.087) = 0.563 | 0.637 |
| 3 | 35 | 40 | 0 | -0.144 + 0.5×(-0.087) = -0.188 | 0.453 |
| 4 | 40 | 60 | 0 | -0.355 + 0.5×(-0.087) = -0.399 | 0.401 |
| 5 | 45 | 70 | 1 | -0.355 + 0.5×0.0 = -0.355 | 0.412 |
| 6 | 50 | 80 | 0 | -0.355 + 0.5×0.0 = -0.355 | 0.412 |

---

## **Final Model**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \gamma_m h_m(x)$$

**Predictions:**
| ID | Age | Income | Label | F3(x) | $\hat{y}_i$ | Prediction |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.563 | 0.637 | 1 (≥0.5) ✓ |
| 2 | 30 | 50 | 1 | 0.563 | 0.637 | 1 (≥0.5) ✓ |
| 3 | 35 | 40 | 0 | -0.188 | 0.453 | 0 (<0.5) ✓ |
| 4 | 40 | 60 | 0 | -0.399 | 0.401 | 0 (<0.5) ✓ |
| 5 | 45 | 70 | 1 | -0.355 | 0.412 | 0 (<0.5) ✗ |
| 6 | 50 | 80 | 0 | -0.355 | 0.412 | 0 (<0.5) ✓ |

**Accuracy: 5/6 = 83.3%**

---

## **Tóm tắt Gradient Boosting Classification**

### **Quy trình hoàn chỉnh:**

1. **Khởi tạo** F0(x) = log(odds) của base probability
2. **Lặp lại** cho m = 1 đến M:
   - **Tính residuals** rm = y - ŷm-1
   - **Train weak learner** hm(x) trên residuals
   - **Tìm learning rate** γm để minimize log loss
   - **Cập nhật model** Fm(x) = Fm-1(x) + γm hm(x)
3. **Final predictions** ŷi = 1/(1+e^(-FM(xi)))

### **Ưu điểm:**
- **Không cần trọng số** như AdaBoost
- **Sử dụng log loss** phù hợp với classification
- **Linh hoạt** với nhiều loại weak learners
- **Hiệu quả** với dữ liệu phức tạp

### **Nhược điểm:**
- **Sensitive** với noise và outliers
- **Có thể overfitting** nếu quá nhiều iterations
- **Chậm hơn** Random Forest
- **Cần tuning** learning rate và số iterations

---

## **So sánh với các thuật toán khác**

| Đặc điểm | Gradient Boosting | AdaBoost | Random Forest | XGBoost |
|:---:|:---:|:---:|:---:|:---:|
| **Method** | Boosting | Boosting | Bagging | Boosting |
| **Training** | Sequential | Sequential | Parallel | Sequential |
| **Weights** | Learning rate | Sample weights | Equal weights | Learning rate |
| **Optimization** | Gradient descent | Weighted resampling | Bootstrap sampling | 2nd order |
| **Speed** | Slow | Medium | Fast | Fast |
| **Overfitting** | High risk | Medium risk | Low risk | Medium risk |
