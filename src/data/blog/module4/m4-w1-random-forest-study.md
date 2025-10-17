---
title: "Random Forest Study - Thuật toán Ensemble Learning với Bootstrap Aggregating"
pubDatetime: 2025-09-21T10:00:00Z
featured: false
description: "Tìm hiểu chi tiết về Random Forest, thuật toán ensemble learning sử dụng bootstrap aggregating và feature bagging"
tags: ["machine-learning", "random-forest", "ensemble", "bootstrap", "bagging", "algorithm"]
---

# Random Forest Study

> **📚 Repo tham khảo:** [https://github.com/tandat8896/ml-from-the-scartch/tree/master/random_forest](https://github.com/tandat8896/ml-from-the-scartch/tree/master/random_forest)

## Thuật toán Random Forest

Random Forest là một thuật toán **ensemble learning** kết hợp nhiều **decision trees** thông qua:
- **Bootstrap Aggregating (Bagging)**: Tạo nhiều subset ngẫu nhiên từ dataset
- **Feature Bagging**: Chọn ngẫu nhiên subset features cho mỗi tree
- **Majority Vote**: Kết hợp predictions từ tất cả trees

---

## Công thức toán học của Random Forest

### **Bước 1: Bootstrap Sampling**

Tạo N bootstrap samples từ dataset gốc D:
$$D_i = \mathrm{Bootstrap}(D), \quad i = 1, 2, \ldots, n_{estimators}$$

Mỗi bootstrap sample có kích thước bằng dataset gốc, được tạo bằng cách:
- **Sampling with replacement** (có hoàn lại)
- Một số samples có thể xuất hiện nhiều lần
- Một số samples có thể không xuất hiện (out-of-bag samples)

### **Bước 2: Feature Bagging**

Cho mỗi tree i, chọn ngẫu nhiên subset features:
$$\text{max\_features} = \sqrt{\text{total\_features}} \quad \text{(default)}$$

$$\text{selected\_features}_i = \text{RandomChoice}(\text{all\_features}, \text{max\_features})$$

### **Bước 3: Tree Training**

Mỗi tree được huấn luyện trên:
- **Bootstrap sample** $D_i$
- **Selected features** $\text{selected\_features}_i$

$$T_i = \mathrm{TrainDecisionTree}(D_i, \mathrm{selected\_features}_i)$$

### **Bước 4: Prediction Aggregation**

**Classification (Majority Vote):**
$$\hat{y} = \mathrm{mode}\{T_1(x), T_2(x), \ldots, T_n(x)\}$$

**Regression (Mean):**
$$\hat{y} = \frac{1}{n} \sum_{i=1}^{n} T_i(x)$$

---

## Ví Dụ Tính Tay - Random Forest Classification

### **Dataset Classification**

| ID | Age (X1) | Income (X2) | Label (y) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 |
| 2 | 30 | 50 | 1 |
| 3 | 35 | 40 | 0 |
| 4 | 40 | 60 | 0 |
| 5 | 45 | 70 | 1 |
| 6 | 50 | 80 | 0 |

**Mục tiêu:** Dự đoán Label dựa trên Age và Income với 3 trees

---

### **Step 1: Bootstrap Sampling**

**Tạo 3 bootstrap samples từ dataset gốc (6 samples):**

| Tree | Bootstrap Sample | Selected Samples |
|:---:|:---:|:---:|
| 1 | D1 | [1, 2, 3, 4, 5, 6] → [1, 1, 3, 4, 5, 6] |
| 2 | D2 | [1, 2, 3, 4, 5, 6] → [1, 2, 2, 4, 5, 6] |
| 3 | D3 | [1, 2, 3, 4, 5, 6] → [1, 2, 3, 4, 6, 6] |

**Giải thích Bootstrap Sampling:**
- Mỗi sample có thể được chọn nhiều lần (replacement)
- Một số samples có thể không được chọn (out-of-bag)
- Kích thước mỗi bootstrap sample = 6 (bằng dataset gốc)

---

### **Step 2: Feature Bagging**

**Tính max_features:**
$$\text{max\_features} = \sqrt{2} = 1.41 \approx 1$$

**Chọn features cho từng tree:**

| Tree | Selected Features | Feature Indices |
|:---:|:---:|:---:|
| 1 | [Age] | [0] |
| 2 | [Income] | [1] |
| 3 | [Age] | [0] |

---

### **Step 3: Tree Training**

#### **Tree 1: Features [Age]**

**Bootstrap Sample D1:**
| ID | Age | Label |
|:---:|:---:|:---:|
| 1 | 25 | 1 |
| 1 | 25 | 1 |
| 3 | 35 | 0 |
| 4 | 40 | 0 |
| 5 | 45 | 1 |
| 6 | 50 | 0 |

**Tìm best split cho Age:**
| Threshold | Left (≤) | Right (>) | Left Label | Right Label | Gini | Split Info | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1,1] | [3,4,5,6] | [1,1] | [0,0,1,0] | 0.0 | 0.0 | 0.0 |
| 32.5 | [1,1,3] | [4,5,6] | [1,1,0] | [0,1,0] | 0.0 | 0.0 | 0.0 |
| 37.5 | [1,1,3,4] | [5,6] | [1,1,0,0] | [1,0] | 0.0 | 0.0 | 0.0 |
| 42.5 | [1,1,3,4,5] | [6] | [1,1,0,0,1] | [0] | 0.0 | 0.0 | 0.0 |
| 47.5 | [1,1,3,4,5,6] | [] | [1,1,0,0,1,0] | [] | 0.0 | 0.0 | 0.0 |

**Tất cả thresholds đều có Gini = 0, chọn threshold = 32.5**

**Tree 1 Structure:**
```
Root: Age <= 32.5?
├── Yes: Label = 1 (samples 1,1,3)
└── No: Label = 0 (samples 4,5,6)
```

#### **Tree 2: Features [Income]**

**Bootstrap Sample D2:**
| ID | Income | Label |
|:---:|:---:|:---:|
| 1 | 30 | 1 |
| 2 | 50 | 1 |
| 2 | 50 | 1 |
| 4 | 60 | 0 |
| 5 | 70 | 1 |
| 6 | 80 | 0 |

**Tìm best split cho Income:**
| Threshold | Left (≤) | Right (>) | Left Label | Right Label | Gini | Split Info | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | [1] | [2,2,4,5,6] | [1] | [1,1,0,1,0] | 0.0 | 0.0 | 0.0 |
| 45 | [1,2,2] | [4,5,6] | [1,1,1] | [0,1,0] | 0.0 | 0.0 | 0.0 |
| 55 | [1,2,2,4] | [5,6] | [1,1,1,0] | [1,0] | 0.0 | 0.0 | 0.0 |
| 65 | [1,2,2,4,5] | [6] | [1,1,1,0,1] | [0] | 0.0 | 0.0 | 0.0 |
| 75 | [1,2,2,4,5,6] | [] | [1,1,1,0,1,0] | [] | 0.0 | 0.0 | 0.0 |

**Tất cả thresholds đều có Gini = 0, chọn threshold = 45**

**Tree 2 Structure:**
```
Root: Income <= 45?
├── Yes: Label = 1 (samples 1,2,2)
└── No: Label = 0 (samples 4,5,6)
```

#### **Tree 3: Features [Age]**

**Bootstrap Sample D3:**
| ID | Age | Label |
|:---:|:---:|:---:|
| 1 | 25 | 1 |
| 2 | 30 | 1 |
| 3 | 35 | 0 |
| 4 | 40 | 0 |
| 6 | 50 | 0 |
| 6 | 50 | 0 |

**Tìm best split cho Age:**
| Threshold | Left (≤) | Right (>) | Left Label | Right Label | Gini | Split Info | Gain |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,6,6] | [1] | [1,0,0,0,0] | 0.0 | 0.0 | 0.0 |
| 32.5 | [1,2] | [3,4,6,6] | [1,1] | [0,0,0,0] | 0.0 | 0.0 | 0.0 |
| 37.5 | [1,2,3] | [4,6,6] | [1,1,0] | [0,0,0] | 0.0 | 0.0 | 0.0 |
| 42.5 | [1,2,3,4] | [6,6] | [1,1,0,0] | [0,0] | 0.0 | 0.0 | 0.0 |
| 47.5 | [1,2,3,4,6,6] | [] | [1,1,0,0,0,0] | [] | 0.0 | 0.0 | 0.0 |

**Tất cả thresholds đều có Gini = 0, chọn threshold = 37.5**

**Tree 3 Structure:**
```
Root: Age <= 37.5?
├── Yes: Label = 1 (samples 1,2,3)
└── No: Label = 0 (samples 4,6,6)
```

---

### **Step 4: Prediction Aggregation**

**Dự đoán cho từng sample (Majority Vote):**

| ID | Age | Income | T1 | T2 | T3 | Majority Vote | True Label |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 1 | 1 | 1 | 1 |
| 2 | 30 | 50 | 1 | 1 | 1 | 1 | 1 |
| 3 | 35 | 40 | 0 | 1 | 1 | 1 | 0 |
| 4 | 40 | 60 | 0 | 0 | 0 | 0 | 0 |
| 5 | 45 | 70 | 0 | 0 | 0 | 0 | 1 |
| 6 | 50 | 80 | 0 | 0 | 0 | 0 | 0 |

**Giải thích:**
- **T1**: Age ≤ 32.5 → 1, Age > 32.5 → 0
- **T2**: Income ≤ 45 → 1, Income > 45 → 0
- **T3**: Age ≤ 37.5 → 1, Age > 37.5 → 0
- **Majority Vote**: Lấy giá trị xuất hiện nhiều nhất

**Accuracy:** 4/6 = 66.7%

---

## **Tóm tắt Random Forest**

### **Quy trình hoàn chỉnh:**

1. **Bootstrap Sampling** - Tạo nhiều subset ngẫu nhiên từ dataset gốc
2. **Feature Bagging** - Chọn ngẫu nhiên subset features cho mỗi tree
3. **Parallel Training** - Huấn luyện các decision trees độc lập
4. **Majority Vote** - Kết hợp predictions bằng majority vote

**Ưu điểm của Random Forest:**
- **Giảm overfitting** nhờ bootstrap sampling
- **Tăng accuracy** nhờ ensemble learning
- **Xử lý missing values** tốt
- **Feature importance** có thể tính được
- **Parallel training** nhanh chóng
- **Robust** với outliers

**Nhược điểm:**
- **Memory usage** cao với nhiều trees
- **Interpretability** thấp hơn single tree
- **Training time** tăng theo số trees
- **Có thể overfitting** với dataset nhỏ

---

## **So sánh với các thuật toán khác**

| Đặc điểm | Random Forest | AdaBoost | XGBoost | LightGBM |
|:---:|:---:|:---:|:---:|:---:|
| **Method** | Bagging | Boosting | Boosting | Boosting |
| **Training** | Parallel | Sequential | Sequential | Sequential |
| **Sampling** | Bootstrap | Weighted | All samples | GOSS |
| **Features** | Random subset | All features | All features | All features |
| **Aggregation** | Majority Vote | Weighted | Weighted | Weighted |
| **Speed** | Fast | Medium | Fast | Fastest |
| **Memory** | High | Medium | High | Low |
| **Accuracy** | Good | Good | Excellent | Excellent |

---

## **Hyperparameters quan trọng**

- **`n_estimators`**: Số lượng trees (100-1000)
- **`max_features`**: Số features mỗi tree (sqrt, log2, hoặc số cố định)
- **`max_depth`**: Độ sâu tối đa của tree (None, 10-20)
- **`min_samples_split`**: Số samples tối thiểu để split (2-10)
- **`min_samples_leaf`**: Số samples tối thiểu ở leaf (1-5)
- **`bootstrap`**: Có sử dụng bootstrap sampling (True/False)
- **`random_state`**: Seed cho reproducibility

---

## **Khi nào nên sử dụng Random Forest**

✅ **Nên dùng khi:**
- Cần model ổn định và robust
- Dataset có nhiều features
- Cần feature importance
- Muốn giảm overfitting
- Cần model interpretable một phần
- Có outliers trong dữ liệu

❌ **Không nên dùng khi:**
- Dataset rất nhỏ (<100 samples)
- Cần accuracy cao nhất
- Memory hạn chế
- Cần model rất nhanh
- Cần interpretability cao

---

## **Kết luận**

Random Forest là một thuật toán ensemble learning mạnh mẽ, kết hợp sức mạnh của nhiều decision trees thông qua bootstrap aggregating và feature bagging. Với khả năng giảm overfitting, xử lý tốt các vấn đề về features và cho kết quả ổn định, Random Forest là lựa chọn tuyệt vời cho nhiều bài toán classification và regression thực tế.

**Điểm mạnh chính:**
- **Ensemble learning** hiệu quả
- **Bootstrap sampling** giảm overfitting  
- **Feature bagging** tăng đa dạng
- **Parallel training** nhanh chóng
- **Robust** với noise và outliers
- **Majority vote** đơn giản và hiệu quả