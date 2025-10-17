---
title: "Road into Linear Regression - Computational Graph Journey"
pubDatetime: 2025-10-05T16:00:00Z
featured: false
description: "Hành trình từ computational graph đơn giản đến Linear Regression hoàn chỉnh với mini-batch training"
tags: ["machine-learning", "linear-regression", "computational-graph", "gradient-descent", "deep-learning"]
---

# Road into Linear Regression - Computational Graph Journey

Khi tôi bắt đầu học về Linear Regression, điều đầu tiên tôi nhận ra là: **mọi thứ đều bắt đầu từ một computational graph đơn giản**. Hôm nay tôi sẽ chia sẻ hành trình của mình từ một sample đơn lẻ đến một mô hình Linear Regression hoàn chỉnh với mini-batch training.

## 1. Computational Graph - One Sample

Đây là nơi mọi thứ bắt đầu - computational graph cho **một sample duy nhất**:

<img width="914" height="500" alt="Image" src="https://github.com/user-attachments/assets/fcd6ceed-1029-45e9-b797-0f00dd66505e" />

### **Giải thích từng bước:**

**Input Layer:**
- `x1`, `x2`, `x3`: Các features đầu vào
- `b`: Bias term (hằng số)

**Weight Layer:**
- `w1`, `w2`, `w3`: Các trọng số (weights) cần học

**Computation:**
- `w1*x1`, `w2*x2`, `w3*x3`: Nhân từng feature với weight tương ứng
- `w1*x1 + w2*x2 + w3*x3 + b`: Tổng có trọng số + bias

**Output:**
- `y_pred`: Dự đoán của mô hình
- `y_true`: Giá trị thực tế

**Loss:**
- `MSE = (y_pred - y_true)²`: Mean Squared Error

### **Công thức toán học:**

$$\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + b$$

$$Loss = (\hat{y} - y_{\text{true}})^2$$

### **Ví dụ cụ thể:**

Giả sử tôi muốn dự đoán giá nhà dựa trên:
- `x1 = 100` (diện tích m²)
- `x2 = 3` (số phòng)
- `x3 = 5` (khoảng cách đến trung tâm km)

Với weights ban đầu:
- `w1 = 0.1`, `w2 = 0.05`, `w3 = -0.02`, `b = 10`

**Tính toán:**
```
y_pred = 0.1*100 + 0.05*3 + (-0.02)*5 + 10
       = 10 + 0.15 - 0.1 + 10
       = 20.05
```

Nếu `y_true = 25` (giá thực tế):
```
Loss = (20.05 - 25)² = (-4.95)² = 24.50
```

---

## 2. Loss với N Epochs và M Samples

Khi tôi hiểu được một sample, bước tiếp theo là **mở rộng ra nhiều samples và nhiều epochs**:

<img width="552" height="413" alt="Image" src="https://github.com/user-attachments/assets/6f1bf8d2-1ab6-4145-bae2-093e10bd39ac" />

### **Giải thích:**

**N Epochs (Vòng lặp):**
- Mỗi epoch = một lần duyệt qua toàn bộ dataset
- N epochs = lặp lại quá trình training N lần

**M Samples:**
- Mỗi sample có loss riêng: `L1`, `L2`, ..., `LM`
- Total Loss = Tổng tất cả losses

**Công thức:**

$$\text{TotalLoss} = \sum_{i=1}^{M} L_i = \sum_{i=1}^{M} (\hat{y}_i - y_{\text{true},i})^2$$

$$\text{AverageLoss} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_{\text{true},i})^2$$

### **Ví dụ với 3 samples:**

| Sample | x1 | x2 | x3 | y_true | y_pred | Loss |
|--------|----|----|----|---------|---------|------|
| 1 | 100 | 3 | 5 | 25 | 20.05 | 24.50 |
| 2 | 80 | 2 | 3 | 18 | 16.10 | 3.61 |
| 3 | 120 | 4 | 8 | 30 | 24.20 | 33.64 |

**Total Loss = 24.50 + 3.61 + 33.64 = 61.75**
**Average Loss = 61.75 / 3 = 20.58**

---

## 3. Zoom Loss - Chi tiết Loss Function

Để hiểu sâu hơn về loss, tôi cần **zoom vào** chi tiết:

<img width="559" height="415" alt="Image" src="https://github.com/user-attachments/assets/dce574a4-014b-41f9-8034-b631ddbb28b7" />

### **Phân tích Loss Function:**

**MSE (Mean Squared Error):**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Tại sao MSE?**
- **Penalize large errors**: Sai số lớn bị phạt nặng hơn (bình phương)
- **Differentiable**: Có thể tính gradient
- **Convex**: Có global minimum

**Gradient của Loss:**
$$\frac{\partial Loss}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \cdot x_{i,j}$$

$$\frac{\partial Loss}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

### **Ví dụ tính gradient:**

Với sample đầu tiên:
- `y_true = 25`, `y_pred = 20.05`
- `error = 25 - 20.05 = 4.95`

**Gradients:**
```
∂Loss/∂w1 = 2 * 4.95 * 100 = 990
∂Loss/∂w2 = 2 * 4.95 * 3 = 29.7
∂Loss/∂w3 = 2 * 4.95 * 5 = 49.5
∂Loss/∂b = 2 * 4.95 * 1 = 9.9
```

**Update weights (với learning rate α = 0.001):**
```
w1_new = w1 - α * ∂Loss/∂w1 = 0.1 - 0.001 * 990 = -0.89
w2_new = w2 - α * ∂Loss/∂w2 = 0.05 - 0.001 * 29.7 = 0.02
w3_new = w3 - α * ∂Loss/∂w3 = -0.02 - 0.001 * 49.5 = -0.07
b_new = b - α * ∂Loss/∂b = 10 - 0.001 * 9.9 = 9.99
```

---

## 4. Computational Graph - Mini Batch (Batch = 2)

Bây giờ tôi hiểu được **mini-batch training** - cách hiệu quả để train với nhiều samples:

<img width="829" height="503" alt="Image" src="https://github.com/user-attachments/assets/ecd33b27-5973-4a53-9326-83a154b81979" />

### **Mini-Batch Training:**

**Batch Size = 2:**
- Thay vì train với 1 sample hoặc toàn bộ dataset
- Train với 2 samples cùng lúc

**Lợi ích:**
- **Stable gradients**: Ít noisy hơn single sample
- **Efficient**: Nhanh hơn full batch
- **Memory efficient**: Không cần load toàn bộ data

### **Công thức cho Mini-Batch:**

**Batch Loss:**
$$\text{BatchLoss} = \frac{1}{\text{batchsize}} \sum_{i=1}^{\text{batchsize}} (y_i - \hat{y}_i)^2$$

**Batch Gradients:**
$$\frac{\partial \text{BatchLoss}}{\partial w_j} = \frac{1}{\text{batchsize}} \sum_{i=1}^{\text{batchsize}} \frac{\partial \text{Loss}_i}{\partial w_j}$$

### **Ví dụ với Batch Size = 2:**

**Batch 1:**
| Sample | x1 | x2 | x3 | y_true | y_pred | Loss |
|--------|----|----|----|---------|---------|------|
| 1 | 100 | 3 | 5 | 25 | 20.05 | 24.50 |
| 2 | 80 | 2 | 3 | 18 | 16.10 | 3.61 |

**Batch Loss = (24.50 + 3.61) / 2 = 14.06**

**Batch Gradients:**
```
∂Batch_Loss/∂w1 = (990 + 420) / 2 = 705
∂Batch_Loss/∂w2 = (29.7 + 12.6) / 2 = 21.15
∂Batch_Loss/∂w3 = (49.5 + 18.9) / 2 = 34.2
∂Batch_Loss/∂b = (9.9 + 3.8) / 2 = 6.85
```

**Update weights:**
```
w1_new = 0.1 - 0.001 * 705 = -0.605
w2_new = 0.05 - 0.001 * 21.15 = 0.029
w3_new = -0.02 - 0.001 * 34.2 = -0.054
b_new = 10 - 0.001 * 6.85 = 9.993
```

---

## 5. Computational Graph - N Sample Training

Cuối cùng, đây là **toàn bộ quá trình training** với N samples:

<img width="860" height="389" alt="Image" src="https://github.com/user-attachments/assets/da5d8395-1a52-495c-9ee8-92916ea6dafe" />

### **Quy trình Training hoàn chỉnh:**

**1. Data Preparation:**
- Chia data thành batches
- Mỗi batch có batch_size samples

**2. Forward Pass:**
- Tính predictions cho tất cả samples trong batch
- Tính batch loss

**3. Backward Pass:**
- Tính gradients cho tất cả weights
- Update weights

**4. Repeat:**
- Lặp lại cho tất cả batches
- Lặp lại cho tất cả epochs

### **Algorithm:**

```python
def train_linear_regression(X, y, epochs=100, batch_size=32, learning_rate=0.001):
    # Initialize weights
    n_features = X.shape[1]
    w = np.random.normal(0, 0.01, n_features)
    b = 0
    
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        for batch in range(n_batches):
            # Get batch
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = X_batch @ w + b
            batch_loss = np.mean((y_batch - y_pred) ** 2)
            epoch_loss += batch_loss
            
            # Backward pass
            error = y_batch - y_pred
            dw = -2 * np.mean(error[:, np.newaxis] * X_batch, axis=0)
            db = -2 * np.mean(error)
            
            # Update weights
            w -= learning_rate * dw
            b -= learning_rate * db
        
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return w, b
```

---

## Ví dụ hoàn chỉnh với Python

### **Dataset thực tế:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Tạo dataset
np.random.seed(42)
n_samples = 1000

# Features: diện tích, số phòng, khoảng cách
X = np.random.randn(n_samples, 3)
X[:, 0] = X[:, 0] * 50 + 100  # Diện tích: 50-150 m²
X[:, 1] = np.random.randint(1, 6, n_samples)  # Số phòng: 1-5
X[:, 2] = np.abs(X[:, 2]) * 10  # Khoảng cách: 0-10 km

# Target: giá nhà (triệu VND)
y = 0.1 * X[:, 0] + 0.05 * X[:, 1] - 0.02 * X[:, 2] + 10 + np.random.normal(0, 2, n_samples)

print("Dataset shape:", X.shape, y.shape)
print("Sample data:")
print("Diện tích | Số phòng | Khoảng cách | Giá nhà")
for i in range(5):
    print(f"{X[i,0]:8.1f} | {X[i,1]:8.0f} | {X[i,2]:10.1f} | {y[i]:7.1f}")
```

### **Training:**

```python
# Train model
w, b = train_linear_regression(X, y, epochs=50, batch_size=32, learning_rate=0.001)

print(f"\nFinal weights: {w}")
print(f"Final bias: {b:.4f}")

# Predictions
y_pred = X @ w + b
mse = np.mean((y - y_pred) ** 2)
print(f"Final MSE: {mse:.4f}")
```

### **Visualization:**

```python
# Plot predictions vs actual
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.plot(y[:100], label='Actual', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('First 100 Samples')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Tóm tắt hành trình

Qua hành trình này, tôi đã học được:

**1. Từ đơn giản đến phức tạp:**
- Bắt đầu với 1 sample → N samples → Mini-batch → Full training

**2. Computational Graph:**
- Mỗi bước đều có thể visualize
- Forward pass: tính predictions
- Backward pass: tính gradients

**3. Loss Function:**
- MSE cho regression
- Gradient descent để minimize loss
- Learning rate ảnh hưởng đến convergence

**4. Mini-batch Training:**
- Cân bằng giữa stability và efficiency
- Batch size là hyperparameter quan trọng

**5. Practical Implementation:**
- Code từ đầu giúp hiểu sâu hơn
- Visualization giúp debug và validate

**Kết luận:**
Linear Regression có vẻ đơn giản, nhưng khi hiểu sâu về computational graph và training process, tôi thấy nó là nền tảng cho tất cả các mô hình ML phức tạp hơn. Mỗi bước đều có lý do và mục đích riêng!

---

## So sánh các phương pháp Training

| Method | Stability | Speed | Memory | Use Case |
|--------|-----------|-------|--------|----------|
| **Single Sample** | Low | Fast | Low | Online learning |
| **Full Batch** | High | Slow | High | Small datasets |
| **Mini-batch** | Medium | Medium | Medium | Most cases |
| **Stochastic** | Low | Fast | Low | Large datasets |

**Recommendation:**
- **Batch size = 32-128** cho hầu hết trường hợp
- **Learning rate = 0.001-0.01** cho Linear Regression
- **Epochs = 50-200** tùy thuộc vào dataset size