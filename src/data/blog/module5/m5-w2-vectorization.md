---
title: "Vectorization trong Linear Regression"
description: "Học cách vectorize Linear Regression để tính toán hiệu quả với ma trận và vector"
pubDatetime: 2025-01-27T20:00:00Z
heroImage: "/assets/images/vectorization-hero.jpg"
tags: ["linear-regression", "vectorization", "matrix", "optimization"]
---

# Vectorization trong Linear Regression

## **Tổng quan**

Vectorization là kỹ thuật chuyển đổi các phép toán từ vòng lặp sang phép toán ma trận/vector, giúp:
- **Tăng tốc tính toán** (10-100x nhanh hơn)
- **Tận dụng tối đa CPU/GPU**
- **Code ngắn gọn và dễ đọc**

## **Nội dung chính**

**Với theta được định nghĩa trước**
## Xét từng sample từng 1 feature 

ta có 

$$
\theta = \begin{bmatrix}
b\\
w
\end{bmatrix}
$$

**và x là feature là một cột**

$$
x=
\begin{bmatrix}
1\\
x_1
\end{bmatrix}
$$

với cái hành động tự nhiên là dotproduct(tích vô hướng)

$$
\theta^T *x = \begin{bmatrix}
b,w
\end{bmatrix}* \begin{bmatrix}
1\\
x_1
\end{bmatrix} = b*1 + w * x_1 = \hat{y}
$$

và ta nhận xét nó rất giống với phương trình predict sample của linear regression 

$$
\mathcal{L}(\theta) = (\hat{y} - y)^2 \quad \text{với } \hat{y} \text{ là một scalar và } y \text{ cũng là một scalar}
$$
nên ỡ bước này ta không cần vectorization gì cả 

- compute gradient

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b} &= 2(\hat{y} - y) \\
\frac{\partial \mathcal{L}}{\partial w} &= 2(\hat{y} - y)x_1
\end{aligned}
$$

mà \(x_1\) là feature đầu tiên của sample.





### **1. Từ vòng lặp đến Vector**
- Chuyển đổi từ `for loop` sang ma trận
- So sánh hiệu suất tính toán
- Ví dụ thực tế với Python

### **2. Vectorization cho Linear Regression**
- Ma trận thiết kế (Design Matrix)
- Vector hóa công thức gradient descent
- Batch processing

### **3. Tối ưu hóa Performance**
- Memory layout và cache efficiency
- Parallel processing
- GPU acceleration với NumPy/CuPy

## **Kết quả học được**

Sau bài này, bạn sẽ:
- ✅ Hiểu cách vectorize Linear Regression
- ✅ Viết code hiệu quả với ma trận
- ✅ Tối ưu performance cho dataset lớn
- ✅ Áp dụng vào các thuật toán ML khác

---

*Bài học này là nền tảng cho việc implement các thuật toán ML hiệu quả trong thực tế.*