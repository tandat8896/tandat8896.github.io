---
title: "M5-W1: Bridge to Linear Regression - Từ Gradient Descent đến Deep Learning"
pubDatetime: 2025-01-27T10:00:00Z
featured: false
description: "Khám phá Linear Regression như cầu nối giữa Machine Learning cơ bản và Deep Learning, với focus vào Gradient Descent và optimization"
tags: ["machine-learning", "linear-regression", "gradient-descent", "optimization", "deep-learning"]
---
# Bridge to Linear Regression - Từ Gradient Descent đến Deep Learning

Linear Regression không chỉ là thuật toán cơ bản mà còn là **cầu nối quan trọng** giữa Machine Learning truyền thống và Deep Learning hiện đại. Trong bài này, chúng ta sẽ khám phá cách Linear Regression hoạt động thông qua **Gradient Descent** và cách nó chuẩn bị nền tảng cho các mô hình phức tạp hơn.

---

## 1. Hàm Loss Function - Square Function
 
![Image](https://github.com/user-attachments/assets/1e0fcb16-5d80-4634-9fbe-a3c90c5678ab)

### **Mục tiêu của Linear Regression:**
Tìm đường thẳng tốt nhất để dự đoán giá trị y từ x:

$$y = wx + b$$

Trong đó:
- **w**: weight (hệ số góc)
- **b**: bias (hệ số chặn)

### **Loss Function - Mean Squared Error (MSE):**
$$L(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(h_w(x^{(i)}) - y^{(i)})^2$$

$$L(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(wx^{(i)} + b - y^{(i)})^2$$

**Đặc điểm của Square Function:**
- **Convex function**: Có một điểm minimum duy nhất
- **Smooth**: Có thể tính đạo hàm tại mọi điểm
- **Quadratic**: Tăng nhanh khi xa khỏi minimum

---

## 2. Gradient Descent với Learning Rate Nhỏ

![Image](https://github.com/user-attachments/assets/866b520e-69c9-47e1-b5fd-6d7df52a8fef)

### **Gradient Descent Algorithm:**
$$w := w - \alpha \frac{\partial L}{\partial w}$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$

### **Với Learning Rate Nhỏ (α = 0.01):**

**Ưu điểm:**
- **Ổn định**: Không bị overshoot
- **Chính xác**: Tiến dần đến minimum
- **An toàn**: Không bị diverge

**Nhược điểm:**
- **Chậm**: Cần nhiều iterations
- **Có thể bị stuck**: Ở local minimum (nếu có)

**Quan sát từ hình:**
- Các bước nhỏ, đều đặn
- Tiến dần về phía minimum
- Convergence chậm nhưng ổn định

---

## 3. Gradient Descent với Learning Rate Lớn

![Image](https://github.com/user-attachments/assets/7a600214-c8c1-4472-b550-a757188be8e9)

### **Với Learning Rate Lớn (α = 0.1):**

**Ưu điểm:**
- **Nhanh**: Tiến nhanh về phía minimum
- **Hiệu quả**: Ít iterations hơn

**Nhược điểm:**
- **Oscillation**: Dao động quanh minimum
- **Khó converge**: Có thể không bao giờ đạt được chính xác minimum

**Quan sát từ hình:**
- Các bước lớn hơn
- Có hiện tượng "zigzag" quanh minimum
- Convergence nhanh nhưng không ổn định

---

## 4. Gradient Descent với Learning Rate Quá Lớn

![Image](https://github.com/user-attachments/assets/d4e2a51f-bdc0-4445-9924-c315fc07ed38)

### **Với Learning Rate Quá Lớn (α = 0.5):**

**Vấn đề nghiêm trọng:**
- **Divergence**: Không thể converge
- **Overshooting**: Vượt quá minimum
- **Instability**: Mất ổn định hoàn toàn

**Quan sát từ hình:**
- Các bước rất lớn
- Bouncing qua lại minimum
- Loss function tăng lên thay vì giảm
- **Không thể tìm được solution**

---

## 5. Partial Derivatives - Fix w, Differentiate b

![Image](https://github.com/user-attachments/assets/4c935f58-a4e7-42c6-867f-01667efc8d5b)

### **Mathematical Foundation:**

Khi **fix w**, ta chỉ optimize b:

$$\frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(wx^{(i)} + b - y^{(i)})$$

$$\frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(h_w(x^{(i)}) - y^{(i)})$$

**Ý nghĩa:**
- **Gradient theo b**: Đo độ lệch trung bình của predictions
- **Direction**: Hướng để điều chỉnh bias
- **Magnitude**: Độ lớn của adjustment cần thiết

---

## 6. Partial Derivatives - Fix b, Differentiate w

![Image](https://github.com/user-attachments/assets/f3b7c8f8-a804-433e-a175-a95541c3e544)

### **Khi fix b, optimize w:**

$$\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(wx^{(i)} + b - y^{(i)}) \cdot x^{(i)}$$

$$\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(h_w(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

**Ý nghĩa:**
- **Gradient theo w**: Đo độ nhạy cảm của loss với weight
- **x^{(i)} factor**: Features có giá trị lớn sẽ ảnh hưởng nhiều hơn
- **Weighted error**: Lỗi được weighted bởi feature values

---

## 7. Simultaneous Optimization - Cả w và b

![Image](https://github.com/user-attachments/assets/1e9e7204-7855-4239-8b67-4e1466cf628e)

### **Complete Gradient Descent:**

**Update Rules:**
$$w := w - \alpha \frac{\partial L}{\partial w}$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$

**Vectorized Form:**
$$\theta := \theta - \alpha \nabla_\theta L(\theta)$$

Trong đó $\theta = [w, b]^T$

### **Tại sao đây là "Bridge to Deep Learning"?**

1. **Gradient Descent**: Nền tảng của tất cả neural networks
2. **Backpropagation**: Mở rộng của gradient descent cho multiple layers
3. **Optimization**: Cùng các kỹ thuật (Adam, RMSprop, etc.)
4. **Loss Functions**: MSE → Cross-entropy, etc.

---

## 8. Kết Luận - Từ Linear Regression đến Deep Learning

### **Linear Regression dạy chúng ta:**

1. **Loss Function Design**: Cách thiết kế objective function
2. **Gradient Computation**: Tính toán derivatives
3. **Optimization**: Tìm minimum của complex functions
4. **Learning Rate**: Tầm quan trọng của hyperparameter tuning

### **Chuyển tiếp sang Deep Learning:**

- **Neural Networks**: Multiple linear layers + activation functions
- **Backpropagation**: Chain rule cho multiple layers
- **Advanced Optimizers**: Adam, RMSprop, AdaGrad
- **Regularization**: Dropout, BatchNorm, L1/L2

### **Key Takeaways:**

✅ **Linear Regression** là foundation của Deep Learning
✅ **Gradient Descent** là core algorithm
✅ **Learning Rate** là hyperparameter quan trọng nhất
✅ **Optimization** là skill cần thiết cho AI/ML

---

## 9. Next Steps

Sau khi hiểu rõ Linear Regression và Gradient Descent, chúng ta sẽ tiến tới:

1. **Multiple Linear Regression**
2. **Logistic Regression** (Classification)
3. **Neural Networks** (Single Layer)
4. **Deep Neural Networks** (Multiple Layers)
5. **Advanced Architectures** (CNN, RNN, Transformer)

**Linear Regression** không chỉ là thuật toán đơn giản - nó là **cầu nối vững chắc** dẫn chúng ta vào thế giới phức tạp và thú vị của Deep Learning! 🚀
