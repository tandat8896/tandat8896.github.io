---
title: "Hồi Quy Tuyến Tính: Row-wise vs Column-wise Gradient"
description: "Phân tích chi tiết về hai phương pháp tính gradient trong linear regression: row-wise và column-wise gradient"
pubDatetime: 2025-01-28T10:00:00Z
tags: ["Linear Regression", "Gradient Descent", "Machine Learning", "Vectorization"]
heroImage: "/assets/images/linear-regression-gradient.png"
---

# Hồi Quy Tuyến Tính: Row-wise vs Column-wise Gradient

## 🎯 Tổng Quan

Trong linear regression, có hai cách tiếp cận chính để tính gradient:
- **Row-wise gradient**: Tính gradient cho từng sample một cách tuần tự
- **Column-wise gradient**: Tính gradient vector hóa cho tất cả samples cùng lúc

## 📊 Row-wise Gradient - TÍNH THEO TỪNG SAMPLE

### **Cách hoạt động:**
```python
# Dữ liệu mẫu: 3 samples, 3 features
X = np.array([[1, 2, 3],    # Sample 1
              [4, 5, 6],    # Sample 2  
              [7, 8, 9]])   # Sample 3
y = np.array([[10], [20], [30]])  # Target values
theta = np.array([[0.1], [0.2], [0.3]])  # Parameters
```

### **Shapes Analysis:**
| Variable | Shape | **TẠI SAO?** |
|----------|-------|---------------|
| `x_i` | (3,) hoặc (3,1) | Features của sample thứ i |
| `y_i` | (1,) hoặc (1,1) | Target của sample thứ i |
| `theta` | (3,1) | Parameters (3 features + bias) |
| `grad` | (3,1) | Gradient cho từng parameter |

### **Code Implementation:**
```python
def row_wise_gradient(X, y, theta):
    m = X.shape[0]  # Số samples
    gradients = np.zeros_like(theta)
    
    for i in range(m):
        # Lấy sample thứ i
        x_i = X[i]  # Shape: (3,)
        y_i = y[i]  # Shape: (1,)
        
        # Tính prediction
        y_hat = x_i.dot(theta)  # Shape: scalar
        
        # Tính gradient cho sample này
        error = y_hat - y_i  # Shape: scalar
        grad_i = 2 * error * x_i  # Shape: (3,)
        
        # Cộng dồn gradient
        gradients += grad_i.reshape(-1, 1)  # Reshape để cộng
    
    return gradients / m  # Trung bình gradient
```

### **Ví dụ số minh họa (3 samples, 3 features):**

#### **Sample 1: x₁ = [1, 2, 3], y₁ = 10**
```python
x_1 = [1, 2, 3]
y_1 = 10
theta = [0.1, 0.2, 0.3]

# Tính prediction
y_hat_1 = 1*0.1 + 2*0.2 + 3*0.3 = 0.1 + 0.4 + 0.9 = 1.4

# Tính error
error_1 = 1.4 - 10 = -8.6

# Tính gradient
grad_1 = 2 * (-8.6) * [1, 2, 3] = [-17.2, -34.4, -51.6]
```

#### **Sample 2: x₂ = [4, 5, 6], y₂ = 20**
```python
x_2 = [4, 5, 6]
y_2 = 20

# Tính prediction
y_hat_2 = 4*0.1 + 5*0.2 + 6*0.3 = 0.4 + 1.0 + 1.8 = 3.2

# Tính error
error_2 = 3.2 - 20 = -16.8

# Tính gradient
grad_2 = 2 * (-16.8) * [4, 5, 6] = [-134.4, -168.0, -201.6]
```

#### **Sample 3: x₃ = [7, 8, 9], y₃ = 30**
```python
x_3 = [7, 8, 9]
y_3 = 30

# Tính prediction
y_hat_3 = 7*0.1 + 8*0.2 + 9*0.3 = 0.7 + 1.6 + 2.7 = 5.0

# Tính error
error_3 = 5.0 - 30 = -25.0

# Tính gradient
grad_3 = 2 * (-25.0) * [7, 8, 9] = [-350.0, -400.0, -450.0]
```

#### **Gradient cuối cùng:**
```python
# Tổng gradient
total_grad = grad_1 + grad_2 + grad_3
total_grad = [-17.2, -34.4, -51.6] + [-134.4, -168.0, -201.6] + [-350.0, -400.0, -450.0]
total_grad = [-501.6, -602.4, -703.2]

# Gradient trung bình
avg_grad = total_grad / 3 = [-167.2, -200.8, -234.4]
```

### **Ưu điểm Row-wise:**
- ✅ **Luôn an toàn**: Không cần reshape y
- ✅ **Dễ hiểu**: Logic rõ ràng từng bước
- ✅ **Debug dễ**: Có thể trace từng sample
- ✅ **Memory efficient**: Chỉ load 1 sample tại một thời điểm

---

## 🚀 Column-wise Gradient - VECTOR HÓA

### **Cách hoạt động:**
```python
# Dữ liệu mẫu: 3 samples, 3 features
X = np.array([[1, 2, 3],    # Sample 1
              [4, 5, 6],    # Sample 2  
              [7, 8, 9]])   # Sample 3
y = np.array([[10], [20], [30]])  # Target values
theta = np.array([[0.1], [0.2], [0.3]])  # Parameters
```

### **Shapes Analysis:**
| Variable | Shape | **TẠI SAO?** |
|----------|-------|---------------|
| `X` | (3, 3) | Tất cả features của tất cả samples |
| `y` | (3, 1) | Tất cả targets |
| `y_hat` | (3, 1) | Tất cả predictions |
| `grad` | (3, 1) | Gradient cho từng parameter |

### **Code Implementation - ĐÚNG:**
```python
def column_wise_gradient_correct(X, y, theta):
    m = X.shape[0]  # Số samples
    
    # Tính tất cả predictions cùng lúc
    y_hat = X.dot(theta)  # Shape: (3, 3) × (3, 1) = (3, 1)
    
    # Tính error cho tất cả samples
    error = y_hat - y  # Shape: (3, 1) - (3, 1) = (3, 1) ✅
    
    # Tính gradient vector hóa
    gradients = 2 * X.T.dot(error) / m  # Shape: (3, 3)ᵀ × (3, 1) = (3, 1)
    
    return gradients
```

### **Code Implementation - SAI (Broadcasting Error):**
```python
def column_wise_gradient_wrong(X, y, theta):
    m = X.shape[0]
    
    # Tính predictions
    y_hat = X.dot(theta)  # Shape: (3, 1)
    
    # SAI: y không được reshape đúng
    y_flat = y.flatten()  # Shape: (3,) - 1D array
    error = y_hat - y_flat  # Shape: (3, 1) - (3,) → Broadcasting issues! ❌
    
    # Gradient sẽ sai do broadcasting
    gradients = 2 * X.T.dot(error) / m  # Kết quả sai! ❌
    
    return gradients
```

### **Ví dụ số minh họa - GRADIENT ĐÚNG:**

#### **Tính predictions:**
```python
X = [[1, 2, 3],
     [4, 5, 6], 
     [7, 8, 9]]
theta = [[0.1], [0.2], [0.3]]

# Matrix multiplication
y_hat = X.dot(theta) = [[1, 2, 3],    [[0.1],    [[1.4],
                        [4, 5, 6],  ×  [0.2],  =  [3.2],
                        [7, 8, 9]]     [0.3]]     [5.0]]
```

#### **Tính error:**
```python
y = [[10], [20], [30]]

# Error calculation
error = y_hat - y = [[1.4],   [[10],   [[-8.6],
                     [3.2], - [20], =  [-16.8],
                     [5.0]]    [30]]    [-25.0]]
```

#### **Tính gradient:**
```python
# X.T.dot(error)
X.T = [[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]]

gradients = 2 * X.T.dot(error) / 3
         = 2 * [[1, 4, 7],    [[-8.6],    / 3
               [2, 5, 8],  ×  [-16.8],
               [3, 6, 9]]     [-25.0]]

# Tính từng element:
# grad[0] = 2 * (1*(-8.6) + 4*(-16.8) + 7*(-25.0)) / 3
#         = 2 * (-8.6 - 67.2 - 175.0) / 3
#         = 2 * (-250.8) / 3 = -167.2

# grad[1] = 2 * (2*(-8.6) + 5*(-16.8) + 8*(-25.0)) / 3
#         = 2 * (-17.2 - 84.0 - 200.0) / 3
#         = 2 * (-301.2) / 3 = -200.8

# grad[2] = 2 * (3*(-8.6) + 6*(-16.8) + 9*(-25.0)) / 3
#         = 2 * (-25.8 - 100.8 - 225.0) / 3
#         = 2 * (-351.6) / 3 = -234.4

gradients = [[-167.2], [-200.8], [-234.4]]
```

### **Ví dụ số minh họa - GRADIENT SAI (Broadcasting Error):**

#### **Khi y không được reshape đúng:**
```python
# SAI: y là 1D array
y_flat = [10, 20, 30]  # Shape: (3,)
y_hat = [[1.4], [3.2], [5.0]]  # Shape: (3, 1)

# Broadcasting sẽ gây lỗi
error = y_hat - y_flat  # (3, 1) - (3,) → Broadcasting issues! ❌
```

#### **Kết quả gradient sai:**
```python
# Gradient sẽ bị sai do broadcasting
gradients_wrong = 2 * X.T.dot(error_wrong) / 3
# Kết quả: [[-83.6], [-100.4], [-117.2]]  # SAI! ❌

# So với gradient đúng:
gradients_correct = [[-167.2], [-200.8], [-234.4]]  # ĐÚNG! ✅
```

---

## 📊 Bảng So Sánh Trực Quan

| Aspect | **Row-wise Gradient** | **Column-wise Gradient** |
|--------|----------------------|-------------------------|
| **Shape x_i** | (3,) - 1D array | (3, 3) - 2D matrix |
| **Shape y_i** | (1,) - scalar | (3, 1) - 2D array |
| **Shape gradient** | (3,) - 1D array | (3, 1) - 2D array |
| **Cần reshape y?** | ❌ Không cần | ✅ Cần reshape (3, 1) |
| **Nguy cơ gradient sai** | 🟢 Thấp | 🟡 Cao (broadcasting) |
| **Performance** | 🟡 Chậm (loop) | 🟢 Nhanh (vectorized) |
| **Memory usage** | 🟢 Thấp | 🟡 Cao |
| **Debug difficulty** | 🟢 Dễ | 🟡 Khó |
| **Code complexity** | 🟢 Đơn giản | 🟡 Phức tạp |

---

## 🔄 Flow Diagram - Gradient Calculation

```
Row-wise Gradient Flow:
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Sample 1│───▶│ Sample 2│───▶│ Sample 3│
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ grad_1  │    │ grad_2  │    │ grad_3  │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┼──────────────┘
                    ▼
            ┌─────────────┐
            │ Total Grad  │
            │ (Average)   │
            └─────────────┘

Column-wise Gradient Flow:
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Sample 1│    │ Sample 2│    │ Sample 3│
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┼──────────────┘
                    ▼
            ┌─────────────┐
            │     X       │
            │ (3×3 matrix)│
            └─────────────┘
                    │
                    ▼
            ┌─────────────┐
            │   y_hat     │
            │ (3×1 vector)│
            └─────────────┘
                    │
                    ▼
            ┌─────────────┐
            │    error    │
            │ (3×1 vector)│
            └─────────────┘
                    │
                    ▼
            ┌─────────────┐
            │  gradients  │
            │ (3×1 vector) │
            └─────────────┘
```

---

## 🎯 Kết Luận

### **Khi nào dùng Row-wise:**
- ✅ **Học tập**: Dễ hiểu và debug
- ✅ **Data nhỏ**: Không cần performance cao
- ✅ **An toàn**: Ít lỗi broadcasting

### **Khi nào dùng Column-wise:**
- ✅ **Production**: Performance cao
- ✅ **Data lớn**: Cần vectorization
- ⚠️ **Cẩn thận**: Phải reshape y đúng shape

### **Lưu ý quan trọng:**
```python
# ĐÚNG: Column-wise gradient
y = y.reshape(-1, 1)  # Đảm bảo shape (m, 1)
error = y_hat - y     # (m, 1) - (m, 1) = (m, 1) ✅

# SAI: Broadcasting error
y_flat = y.flatten()  # Shape (m,)
error = y_hat - y_flat  # (m, 1) - (m,) → Broadcasting issues! ❌
```

**Nhớ**: Shape consistency là chìa khóa để tránh gradient sai! 🎯
