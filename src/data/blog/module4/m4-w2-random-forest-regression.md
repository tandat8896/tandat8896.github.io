---
title: "Random Forest Regression - Thuật toán Ensemble Learning cho Regression"
pubDatetime: 2025-09-21T21:00:00Z
featured: false
description: "Tìm hiểu chi tiết về Random Forest cho Regression, từ bootstrap sampling đến prediction aggregation"
tags: ["machine-learning", "random-forest", "regression", "ensemble", "bootstrap"]
---

# Random Forest Regression

## Thuật toán Random Forest cho Regression

Random Forest cho Regression sử dụng **Bootstrap Aggregating (Bagging)** để kết hợp nhiều regression trees thành một model mạnh mẽ. Khác với classification, regression sử dụng **mean** thay vì **majority vote** để kết hợp predictions.

---

## Công thức toán học của Random Forest Regression

### **Bước 1: Bootstrap Sampling**

Tạo N bootstrap samples từ dataset gốc D:

$$D_i = \text{Bootstrap}(D), \quad i = 1, 2, \ldots, n\_estimators$$

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

$$T_i = \text{TrainRegressionTree}(D_i, \text{selected\_features}_i)$$

### **Bước 4: Prediction Aggregation**

**Regression (Mean):**
$$\hat{y} = \frac{1}{n} \sum_{i=1}^{n} T_i(x)$$

**Lưu ý quan trọng:**
- **Không có trọng số** cho từng tree trong Random Forest
- Tất cả trees có **trọng số bằng nhau** (1/n)
- Khác với AdaBoost có trọng số phụ thuộc vào error rate

---

## Ví Dụ Tính Tay - Random Forest Regression

### **Dataset Regression**

| ID | Age (X1) | Income (X2) | Price (y) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 100 |
| 2 | 30 | 50 | 150 |
| 3 | 35 | 40 | 120 |
| 4 | 40 | 60 | 180 |
| 5 | 45 | 70 | 200 |
| 6 | 50 | 80 | 220 |

**Mục tiêu:** Dự đoán Price dựa trên Age và Income với 3 trees

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
| ID | Age | Price |
|:---:|:---:|:---:|
| 1 | 25 | 100 |
| 1 | 25 | 100 |
| 3 | 35 | 120 |
| 4 | 40 | 180 |
| 5 | 45 | 200 |
| 6 | 50 | 220 |

**Tìm best split cho Age:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1,1] | [3,4,5,6] | [100,100] | [120,180,200,220] | 0 |
| 32.5 | [1,1,3] | [4,5,6] | [100,100,120] | [180,200,220] | 0 |
| 37.5 | [1,1,3,4] | [5,6] | [100,100,120,180] | [200,220] | 0 |
| 42.5 | [1,1,3,4,5] | [6] | [100,100,120,180,200] | [220] | 0 |
| 47.5 | [1,1,3,4,5,6] | [] | [100,100,120,180,200,220] | [] | 0 |

**Tất cả thresholds đều có MSE = 0, chọn threshold = 32.5**

**Tree 1 Structure:**
```
Root: Age <= 32.5?
├── Yes: Price = 100 (samples 1,1,3)
└── No: Price = 200 (samples 4,5,6)
```

#### **Tree 2: Features [Income]**

**Bootstrap Sample D2:**
| ID | Income | Price |
|:---:|:---:|:---:|
| 1 | 30 | 100 |
| 2 | 50 | 150 |
| 2 | 50 | 150 |
| 4 | 60 | 180 |
| 5 | 70 | 200 |
| 6 | 80 | 220 |

**Tìm best split cho Income:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 35 | [1] | [2,2,4,5,6] | [100] | [150,150,180,200,220] | 0 |
| 45 | [1,2,2] | [4,5,6] | [100,150,150] | [180,200,220] | 0 |
| 55 | [1,2,2,4] | [5,6] | [100,150,150,180] | [200,220] | 0 |
| 65 | [1,2,2,4,5] | [6] | [100,150,150,180,200] | [220] | 0 |
| 75 | [1,2,2,4,5,6] | [] | [100,150,150,180,200,220] | [] | 0 |

**Tất cả thresholds đều có MSE = 0, chọn threshold = 45**

**Tree 2 Structure:**
```
Root: Income <= 45?
├── Yes: Price = 133.33 (samples 1,2,2)
└── No: Price = 200 (samples 4,5,6)
```

#### **Tree 3: Features [Age]**

**Bootstrap Sample D3:**
| ID | Age | Price |
|:---:|:---:|:---:|
| 1 | 25 | 100 |
| 2 | 30 | 150 |
| 3 | 35 | 120 |
| 4 | 40 | 180 |
| 6 | 50 | 220 |
| 6 | 50 | 220 |

**Tìm best split cho Age:**
| Threshold | Left (≤) | Right (>) | Left Price | Right Price | MSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,6,6] | [100] | [150,120,180,220,220] | 0 |
| 32.5 | [1,2] | [3,4,6,6] | [100,150] | [120,180,220,220] | 0 |
| 37.5 | [1,2,3] | [4,6,6] | [100,150,120] | [180,220,220] | 0 |
| 42.5 | [1,2,3,4] | [6,6] | [100,150,120,180] | [220,220] | 0 |
| 47.5 | [1,2,3,4,6,6] | [] | [100,150,120,180,220,220] | [] | 0 |

**Tất cả thresholds đều có MSE = 0, chọn threshold = 37.5**

**Tree 3 Structure:**
```
Root: Age <= 37.5?
├── Yes: Price = 123.33 (samples 1,2,3)
└── No: Price = 220 (samples 4,6,6)
```

---

### **Step 4: Prediction Aggregation**

**Dự đoán cho từng sample (Mean - Không có trọng số):**

| ID | Age | Income | T1 | T2 | T3 | Mean | True Price |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 100 | 133.33 | 123.33 | (100+133.33+123.33)/3 = 118.89 | 100 |
| 2 | 30 | 50 | 100 | 133.33 | 123.33 | (100+133.33+123.33)/3 = 118.89 | 150 |
| 3 | 35 | 40 | 100 | 133.33 | 123.33 | (100+133.33+123.33)/3 = 118.89 | 120 |
| 4 | 40 | 60 | 200 | 200 | 220 | (200+200+220)/3 = 206.67 | 180 |
| 5 | 45 | 70 | 200 | 200 | 220 | (200+200+220)/3 = 206.67 | 200 |
| 6 | 50 | 80 | 200 | 200 | 220 | (200+200+220)/3 = 206.67 | 220 |

**Giải thích:**
- **Không có trọng số** cho từng tree
- **Mean** đơn giản: trung bình cộng của tất cả trees
- **Tie breaking**: nếu bằng nhau, chọn giá trị đầu tiên (hoặc random)

**MSE:**
$$MSE = \frac{1}{6} \sum_{i=1}^{6} (y_i - \hat{y}_i)^2 = \frac{1}{6}[(100-118.89)^2 + (150-118.89)^2 + (120-118.89)^2 + (180-206.67)^2 + (200-206.67)^2 + (220-206.67)^2]$$

$$MSE = \frac{1}{6}[356.79 + 970.12 + 1.23 + 711.11 + 44.44 + 177.78] = \frac{1}{6}[2261.47] = 377.08$$

---

## **Tóm tắt Random Forest Regression**

### **Quy trình hoàn chỉnh:**

1. **Bootstrap Sampling** - Tạo nhiều subset ngẫu nhiên từ dataset gốc
2. **Feature Bagging** - Chọn ngẫu nhiên subset features cho mỗi tree
3. **Parallel Training** - Huấn luyện các regression trees độc lập
4. **Mean Aggregation** - Kết hợp predictions bằng trung bình cộng

**Ưu điểm của Random Forest Regression:**
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
| **Aggregation** | Mean | Weighted | Weighted | Weighted |
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

## **Khi nào nên sử dụng Random Forest Regression**

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

Random Forest Regression là một thuật toán ensemble learning mạnh mẽ, kết hợp sức mạnh của nhiều regression trees thông qua bootstrap aggregating và feature bagging. Với khả năng giảm overfitting, xử lý tốt các vấn đề về features và cho kết quả ổn định, Random Forest Regression là lựa chọn tuyệt vời cho nhiều bài toán regression thực tế.

**Điểm mạnh chính:**
- **Ensemble learning** hiệu quả
- **Bootstrap sampling** giảm overfitting  
- **Feature bagging** tăng đa dạng
- **Parallel training** nhanh chóng
- **Robust** với noise và outliers
- **Mean aggregation** đơn giản và hiệu quả
