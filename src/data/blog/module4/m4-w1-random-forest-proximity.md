---
title: "Random Forest Proximity - Quy trình đầy đủ từ Missing Values đến Proximity Matrix"
pubDatetime: 2025-09-21T16:00:00Z
featured: false
description: "Tìm hiểu Random Forest Proximity với quy trình hoàn chỉnh: fill missing values → train Random Forest → tính proximity matrix"
tags: ["machine-learning", "random-forest", "proximity", "missing-values", "imputation"]
---

# Random Forest Proximity - Quy trình hoàn chỉnh

## Khái niệm Random Forest Proximity

**Random Forest Proximity** đo độ tương đồng giữa các samples dựa trên việc chúng rơi vào cùng leaf nodes trong các cây quyết định. Proximity càng cao nghĩa là hai samples càng tương đồng.

---

## Công thức toán học

### **Định nghĩa Proximity**

$$P(x_i, x_j) = \frac{1}{T} \sum_{t=1}^{T} I(\text{leaf}_t(x_i) = \text{leaf}_t(x_j))$$

Trong đó:
- $T$: số lượng trees trong Random Forest
- $\text{leaf}_t(x_i)$: leaf node mà sample $x_i$ rơi vào ở tree $t$
- $I(\cdot)$: indicator function (1 nếu điều kiện đúng, 0 nếu sai)

---

## Ví Dụ Tính Tay - Quy trình hoàn chỉnh

### **Step 1: Dataset gốc có Missing Values**

| ID | Age | Income | Education | Label |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 2 | 1 |
| 2 | 30 | ? | 3 | 1 |
| 3 | ? | 40 | 2 | 0 |
| 4 | 40 | 60 | ? | 0 |
| 5 | 45 | 70 | 4 | 1 |

**Missing values:**
- Sample 2: Income = ?
- Sample 3: Age = ?  
- Sample 4: Education = ?

### **Step 2: Fill Missing Values bằng Mean**

**Tính mean cho từng feature:**

**Age:**
- Có giá trị: [25, 30, 40, 45]
- Mean Age = (25 + 30 + 40 + 45) / 4 = 140 / 4 = 35

**Income:**
- Có giá trị: [30, 60, 70]
- Mean Income = (30 + 60 + 70) / 3 = 160 / 3 = 53.33

**Education:**
- Có giá trị: [2, 3, 2, 4]
- Mean Education = (2 + 3 + 2 + 4) / 4 = 11 / 4 = 2.75

**Dataset sau khi fill missing values:**

| ID | Age | Income | Education | Label | Filled Values |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 2 | 1 | - |
| 2 | 30 | **53.33** | 3 | 1 | Income filled |
| 3 | **35** | 40 | 2 | 0 | Age filled |
| 4 | 40 | 60 | **2.75** | 0 | Education filled |
| 5 | 45 | 70 | 4 | 1 | - |

### **Step 3: Train Random Forest với 3 Trees**

**Tree 1: Features [Age, Income]**
```
Root: Age <= 32.5?
├── Yes: Label = 1 (samples 1, 2)
└── No: Label = 0 (samples 3, 4, 5)
```

**Tree 2: Features [Age, Education]**
```
Root: Education <= 2.5?
├── Yes: Label = 1 (samples 1, 3)
└── No: Label = 0 (samples 2, 4, 5)
```

**Tree 3: Features [Income, Education]**
```
Root: Income <= 45?
├── Yes: Label = 1 (samples 1, 3)
└── No: Label = 0 (samples 2, 4, 5)
```

### **Step 4: Leaf Node Assignment**

| Sample | Tree 1 Leaf | Tree 2 Leaf | Tree 3 Leaf |
|:---:|:---:|:---:|:---:|
| 1 | Left (Age ≤ 32.5) | Left (Edu ≤ 2.5) | Left (Income ≤ 45) |
| 2 | Left (Age ≤ 32.5) | Right (Edu > 2.5) | Right (Income > 45) |
| 3 | Right (Age > 32.5) | Left (Edu ≤ 2.5) | Left (Income ≤ 45) |
| 4 | Right (Age > 32.5) | Right (Edu > 2.5) | Right (Income > 45) |
| 5 | Right (Age > 32.5) | Right (Edu > 2.5) | Right (Income > 45) |

### **Step 5: Tính Proximity Matrix (5×5)**

**Công thức:** $P(x_i, x_j) = \frac{1}{3} \sum_{t=1}^{3} I(\text{leaf}_t(x_i) = \text{leaf}_t(x_j))$

**Tính toán chi tiết:**

#### **Hàng 1: Sample 1 với tất cả samples**

**P(1,1): Sample 1 với Sample 1**
- Tree 1: Left = Left → 1
- Tree 2: Left = Left → 1
- Tree 3: Left = Left → 1
- $P(1,1) = \frac{1+1+1}{3} = 1.00$

**P(1,2): Sample 1 với Sample 2**
- Tree 1: Left = Left → 1
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(1,2) = \frac{1+0+0}{3} = 0.33$

**P(1,3): Sample 1 với Sample 3**
- Tree 1: Left ≠ Right → 0
- Tree 2: Left = Left → 1
- Tree 3: Left = Left → 1
- $P(1,3) = \frac{0+1+1}{3} = 0.67$

**P(1,4): Sample 1 với Sample 4**
- Tree 1: Left ≠ Right → 0
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(1,4) = \frac{0+0+0}{3} = 0.00$

**P(1,5): Sample 1 với Sample 5**
- Tree 1: Left ≠ Right → 0
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(1,5) = \frac{0+0+0}{3} = 0.00$

#### **Hàng 2: Sample 2 với tất cả samples**

**P(2,1): Sample 2 với Sample 1**
- Tree 1: Left = Left → 1
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(2,1) = \frac{1+0+0}{3} = 0.33$

**P(2,2): Sample 2 với Sample 2**
- Tree 1: Left = Left → 1
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(2,2) = \frac{1+1+1}{3} = 1.00$

**P(2,3): Sample 2 với Sample 3**
- Tree 1: Left ≠ Right → 0
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(2,3) = \frac{0+0+0}{3} = 0.00$

**P(2,4): Sample 2 với Sample 4**
- Tree 1: Left ≠ Right → 0
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(2,4) = \frac{0+1+1}{3} = 0.67$

**P(2,5): Sample 2 với Sample 5**
- Tree 1: Left ≠ Right → 0
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(2,5) = \frac{0+1+1}{3} = 0.67$

#### **Hàng 3: Sample 3 với tất cả samples**

**P(3,1): Sample 3 với Sample 1**
- Tree 1: Right ≠ Left → 0
- Tree 2: Left = Left → 1
- Tree 3: Left = Left → 1
- $P(3,1) = \frac{0+1+1}{3} = 0.67$

**P(3,2): Sample 3 với Sample 2**
- Tree 1: Right ≠ Left → 0
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(3,2) = \frac{0+0+0}{3} = 0.00$

**P(3,3): Sample 3 với Sample 3**
- Tree 1: Right = Right → 1
- Tree 2: Left = Left → 1
- Tree 3: Left = Left → 1
- $P(3,3) = \frac{1+1+1}{3} = 1.00$

**P(3,4): Sample 3 với Sample 4**
- Tree 1: Right = Right → 1
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(3,4) = \frac{1+0+0}{3} = 0.33$

**P(3,5): Sample 3 với Sample 5**
- Tree 1: Right = Right → 1
- Tree 2: Left ≠ Right → 0
- Tree 3: Left ≠ Right → 0
- $P(3,5) = \frac{1+0+0}{3} = 0.33$

#### **Hàng 4: Sample 4 với tất cả samples**

**P(4,1): Sample 4 với Sample 1**
- Tree 1: Right ≠ Left → 0
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(4,1) = \frac{0+0+0}{3} = 0.00$

**P(4,2): Sample 4 với Sample 2**
- Tree 1: Right ≠ Left → 0
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(4,2) = \frac{0+1+1}{3} = 0.67$

**P(4,3): Sample 4 với Sample 3**
- Tree 1: Right = Right → 1
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(4,3) = \frac{1+0+0}{3} = 0.33$

**P(4,4): Sample 4 với Sample 4**
- Tree 1: Right = Right → 1
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(4,4) = \frac{1+1+1}{3} = 1.00$

**P(4,5): Sample 4 với Sample 5**
- Tree 1: Right = Right → 1
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(4,5) = \frac{1+1+1}{3} = 1.00$

#### **Hàng 5: Sample 5 với tất cả samples**

**P(5,1): Sample 5 với Sample 1**
- Tree 1: Right ≠ Left → 0
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(5,1) = \frac{0+0+0}{3} = 0.00$

**P(5,2): Sample 5 với Sample 2**
- Tree 1: Right ≠ Left → 0
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(5,2) = \frac{0+1+1}{3} = 0.67$

**P(5,3): Sample 5 với Sample 3**
- Tree 1: Right = Right → 1
- Tree 2: Right ≠ Left → 0
- Tree 3: Right ≠ Left → 0
- $P(5,3) = \frac{1+0+0}{3} = 0.33$

**P(5,4): Sample 5 với Sample 4**
- Tree 1: Right = Right → 1
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(5,4) = \frac{1+1+1}{3} = 1.00$

**P(5,5): Sample 5 với Sample 5**
- Tree 1: Right = Right → 1
- Tree 2: Right = Right → 1
- Tree 3: Right = Right → 1
- $P(5,5) = \frac{1+1+1}{3} = 1.00$

### **Step 6: Proximity Matrix Hoàn Chỉnh (5×5)**

| | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.00 | 0.33 | 0.67 | 0.00 | 0.00 |
| **2** | 0.33 | 1.00 | 0.00 | 0.67 | 0.67 |
| **3** | 0.67 | 0.00 | 1.00 | 0.33 | 0.33 |
| **4** | 0.00 | 0.67 | 0.33 | 1.00 | 1.00 |
| **5** | 0.00 | 0.67 | 0.33 | 1.00 | 1.00 |

**Kiểm tra tính chất:**
- ✅ **Symmetric**: P(i,j) = P(j,i) cho mọi i,j
- ✅ **Diagonal = 1**: P(i,i) = 1.00 cho mọi i
- ✅ **Range [0,1]**: Tất cả giá trị trong khoảng [0,1]

---

## Ứng dụng của Proximity Matrix

### **1. Outlier Detection**

**Công thức Outlier Score:**
$$\text{OutlierScore}(x_i) = 1 - \frac{1}{n-1} \sum_{j \neq i} P(x_i, x_j)$$

**Tính toán chi tiết:**

**Sample 1:**
- Proximity với samples khác: [0.33, 0.67, 0.00, 0.00]
- Average proximity: $\frac{0.33 + 0.67 + 0.00 + 0.00}{4} = 0.25$
- Outlier score: $1 - 0.25 = 0.75$

**Sample 2:**
- Proximity với samples khác: [0.33, 0.00, 0.67, 0.67]
- Average proximity: $\frac{0.33 + 0.00 + 0.67 + 0.67}{4} = 0.42$
- Outlier score: $1 - 0.42 = 0.58$

**Sample 3:**
- Proximity với samples khác: [0.67, 0.00, 0.33, 0.33]
- Average proximity: $\frac{0.67 + 0.00 + 0.33 + 0.33}{4} = 0.33$
- Outlier score: $1 - 0.33 = 0.67$

**Sample 4:**
- Proximity với samples khác: [0.00, 0.67, 0.33, 1.00]
- Average proximity: $\frac{0.00 + 0.67 + 0.33 + 1.00}{4} = 0.50$
- Outlier score: $1 - 0.50 = 0.50$

**Sample 5:**
- Proximity với samples khác: [0.00, 0.67, 0.33, 1.00]
- Average proximity: $\frac{0.00 + 0.67 + 0.33 + 1.00}{4} = 0.50$
- Outlier score: $1 - 0.50 = 0.50$

**Ranking Outliers:**
| Sample | Outlier Score | Rank | Interpretation |
|:---:|:---:|:---:|:---:|
| 1 | 0.75 | 1 | **Most Outlier** |
| 3 | 0.67 | 2 | **Outlier** |
| 2 | 0.58 | 3 | **Moderate** |
| 4,5 | 0.50 | 4 | **Normal** |

### **2. Missing Value Imputation với Proximity**

**Công thức Imputation:**
$$\hat{x}_{i,j} = \frac{\sum_{k=1}^{n} P(x_i, x_k) \cdot x_{k,j}}{\sum_{k=1}^{n} P(x_i, x_k)}$$

**Ví dụ: Imputation cho Sample 2 (Income = 53.33)**

Từ Proximity Matrix:
- P(2,1) = 0.33
- P(2,3) = 0.00  
- P(2,4) = 0.67
- P(2,5) = 0.67

**Imputation Income cho Sample 2:**
$$\hat{x}_{2,Income} = \frac{0.33 \times 30 + 0.00 \times 40 + 0.67 \times 60 + 0.67 \times 70}{0.33 + 0.00 + 0.67 + 0.67}$$

$$\hat{x}_{2,Income} = \frac{9.9 + 0 + 40.2 + 46.9}{1.67} = \frac{97.0}{1.67} = 58.1$$

**So sánh:**
- **Mean imputation**: 53.33
- **Proximity imputation**: 58.1
- **Proximity imputation** gần với samples 4,5 (proximity cao) hơn

---

## Kết luận

**Random Forest Proximity** cung cấp:

1. **Quy trình hoàn chỉnh**: Missing values → Fill → Train → Proximity
2. **Ma trận proximity n×n** đo độ tương đồng giữa tất cả cặp samples
3. **Tính toán chi tiết** từng bước để hiểu rõ cách thức hoạt động
4. **Ứng dụng thực tế** như outlier detection và imputation

**Ưu điểm:**
- Dựa trên **cấu trúc dữ liệu** thực tế từ Random Forest
- **Robust** với noise và outliers
- **Interpretable** và dễ hiểu
- **Flexible** cho nhiều loại dữ liệu

**Nhược điểm:**
- **Computational cost** cao với dataset lớn
- **Phụ thuộc** vào chất lượng Random Forest
- **Memory usage** cao cho ma trận n×n
