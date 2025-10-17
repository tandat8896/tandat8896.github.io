---
title: "LIME - Local Interpretable Model-agnostic Explanations (Cosine Distance)"
pubDatetime: 2025-10-05T15:00:00Z
featured: false
description: "Tìm hiểu chi tiết về LIME với cosine distance để giải thích dự đoán của mô hình Machine Learning"
tags: ["machine-learning", "lime", "explainable-ai", "interpretability", "xai", "cosine-distance"]
---

# LIME - Local Interpretable Model-agnostic Explanations (Cosine Distance)

## Nguyên lý cơ bản của LIME

### **Ý tưởng chính - Tại sao LIME hoạt động?**

Khi tôi bắt đầu học về LIME, câu hỏi đầu tiên tôi đặt ra là: "Tại sao mô hình lại dự đoán như vậy?" LIME sinh ra để trả lời câu hỏi này.

**1. Vấn đề cần giải quyết:**
- Mô hình Machine Learning (đặc biệt là Deep Learning) thường là "black box"
- Chúng ta không biết tại sao mô hình dự đoán kết quả đó
- Cần giải thích dự đoán cho từng trường hợp cụ thể (local explanation)

**2. Ý tưởng của LIME:**
- Thay vì giải thích toàn bộ mô hình phức tạp (global)
- LIME giải thích dự đoán cho **một điểm dữ liệu cụ thể** (local)
- Sử dụng mô hình đơn giản (linear model) để xấp xỉ mô hình phức tạp **xung quanh điểm đó**

**3. Ví dụ thực tế:**
Giả sử bạn có mô hình dự đoán giá nhà, và nó dự đoán một căn nhà có giá 500 triệu. LIME sẽ trả lời:
- Diện tích lớn → tăng giá +150 triệu
- Vị trí tốt → tăng giá +100 triệu  
- Nhà cũ → giảm giá -50 triệu
- Số phòng nhiều → tăng giá +80 triệu

### **Các khái niệm cơ bản**

**1. Local vs Global Explanation:**
- **Global**: Giải thích toàn bộ mô hình (feature importance tổng thể)
- **Local**: Giải thích một dự đoán cụ thể (tại sao điểm này được dự đoán như vậy)

**2. Model-agnostic:**
- LIME hoạt động với **bất kỳ mô hình nào**: Random Forest, XGBoost, Neural Network, SVM...
- Không cần biết cấu trúc bên trong của mô hình
- Chỉ cần mô hình có thể dự đoán (predict)

**3. Interpretable Representation:**
- Biến đổi dữ liệu phức tạp thành dạng dễ hiểu
- Ví dụ: Text → Có/Không có từ khóa, Image → Có/Không có vùng ảnh

**4. Perturbation (Nhiễu loạn):**
- Tạo các mẫu dữ liệu mới bằng cách thay đổi nhẹ dữ liệu gốc
- Xem mô hình dự đoán như thế nào với các mẫu này
- Từ đó học được feature nào quan trọng

### **Công thức toán học**

**1. Mục tiêu của LIME:**
$$\text{explanation}(x) = \arg\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

Trong đó:
- $x$: Điểm dữ liệu cần giải thích
- $f$: Mô hình phức tạp (black box)
- $g$: Mô hình đơn giản (interpretable model, thường là linear)
- $G$: Tập các mô hình đơn giản
- $L$: Loss function (đo độ khác biệt giữa $f$ và $g$)
- $\pi_x$: Proximity measure (độ gần với $x$)
- $\Omega(g)$: Complexity của mô hình $g$

**2. Loss function:**
$$L(f, g, \pi_x) = \sum_{z \in Z} \pi_x(z) [f(z) - g(z)]^2$$

Trong đó:
- $Z$: Tập các mẫu perturbed (nhiễu loạn)
- $\pi_x(z)$: Trọng số dựa trên khoảng cách từ $z$ đến $x$
- $f(z)$: Dự đoán của mô hình phức tạp
- $g(z)$: Dự đoán của mô hình đơn giản

**3. Proximity measure với Cosine Distance:**
$$\pi_x(z) = \exp\left(-\frac{D_{cosine}(x, z)^2}{\sigma^2}\right)$$

**Công thức Cosine Distance:**
$$D_{cosine}(x, z) = 1 - \frac{x \cdot z}{||x|| \times ||z||} = 1 - \cos(\theta)$$

Trong đó:
- $x \cdot z$: Dot product của hai vector
- $||x||, ||z||$: Norm của vector
- $\theta$: Góc giữa hai vector
- $\sigma$: Kernel width (độ rộng của kernel)

### **Thuật toán LIME - Chi tiết từng bước**

**Input:** 
- Điểm dữ liệu $x$ cần giải thích
- Mô hình $f$ (black box)
- Số lượng mẫu perturbed $N$

**Output:** 
- Trọng số của các features (giải thích local)

**Bước 1: Tạo perturbed samples**
- Tạo $N$ mẫu dữ liệu mới bằng cách thay đổi nhẹ $x$
- Ví dụ: Với tabular data, thay đổi giá trị một số features

**Bước 2: Dự đoán với mô hình gốc**
- Sử dụng mô hình $f$ để dự đoán cho tất cả perturbed samples
- Lưu lại kết quả dự đoán

**Bước 3: Tính trọng số proximity (Cosine Distance)**
- Tính cosine distance từ mỗi perturbed sample đến $x$
- Mẫu càng gần $x$ (cosine similarity cao) thì trọng số càng cao

**Bước 4: Fit mô hình linear**
- Sử dụng perturbed samples làm training data
- Fit mô hình linear với trọng số proximity
- Mô hình linear này giải thích mô hình phức tạp **xung quanh $x$**

**Bước 5: Trích xuất feature importance**
- Hệ số của mô hình linear chính là feature importance
- Hệ số dương → feature tăng prediction
- Hệ số âm → feature giảm prediction

---

## Ví Dụ Tính Tay - LIME với Cosine Distance

### **Dataset và Mô hình**

Giả sử tôi có một mô hình XGBoost dự đoán xem một người có mua sản phẩm hay không dựa trên 4 features:

| Feature | Mô tả | Đơn vị |
|---------|-------|--------|
| Age | Tuổi | năm |
| Income | Thu nhập | triệu/năm |
| TimeOnSite | Thời gian trên website | phút |
| PreviousPurchases | Số lần mua trước đó | lần |

**Điểm dữ liệu cần giải thích:**

| Age | Income | TimeOnSite | PreviousPurchases | Prediction |
|-----|--------|------------|-------------------|------------|
| 35 | 50 | 15 | 3 | 0.85 (85% mua) |

Mô hình dự đoán người này có **85% khả năng mua sản phẩm**. Nhưng tại sao?

---

### **Step 1: Tạo Perturbed Samples**

Tôi sẽ tạo 5 mẫu perturbed bằng cách thay đổi nhẹ các giá trị:

| ID | Age | Income | TimeOnSite | PreviousPurchases | Cosine_Distance | Weight | Model_Prediction |
|----|-----|--------|------------|-------------------|-----------------|--------|------------------|
| Original | 35 | 50 | 15 | 3 | 0.000 | 1.000 | 0.85 |
| Sample_1 | 33 | 48 | 14 | 3 | 0.008 | 0.984 | 0.78 |
| Sample_2 | 37 | 52 | 16 | 4 | 0.012 | 0.976 | 0.92 |
| Sample_3 | 35 | 45 | 12 | 2 | 0.025 | 0.951 | 0.65 |
| Sample_4 | 32 | 50 | 18 | 3 | 0.015 | 0.970 | 0.82 |
| Sample_5 | 38 | 55 | 15 | 5 | 0.018 | 0.965 | 0.88 |

**Giải thích các cột:**

**Cosine Distance:** Khoảng cách cosine giữa sample và original point

**Tính toán chi tiết cho Sample_1:**
- Original: $x = [35, 50, 15, 3]$
- Sample_1: $z = [33, 48, 14, 3]$

**Dot product:**
$$x \cdot z = 35 \times 33 + 50 \times 48 + 15 \times 14 + 3 \times 3 = 1155 + 2400 + 210 + 9 = 3774$$

**Norms:**
$$||x|| = \sqrt{35^2 + 50^2 + 15^2 + 3^2} = \sqrt{1225 + 2500 + 225 + 9} = \sqrt{3959} = 62.92$$
$$||z|| = \sqrt{33^2 + 48^2 + 14^2 + 3^2} = \sqrt{1089 + 2304 + 196 + 9} = \sqrt{3598} = 59.98$$

**Cosine similarity:**
$$\cos(\theta) = \frac{x \cdot z}{||x|| \times ||z||} = \frac{3774}{62.92 \times 59.98} = \frac{3774}{3774.1} = 0.9997$$

**Cosine distance:**
$$D_{cosine}(x, z) = 1 - \cos(\theta) = 1 - 0.9997 = 0.0003$$

**Weight với $\sigma = 0.1$:**
$$\pi_x(z) = \exp\left(-\frac{0.0003^2}{0.1^2}\right) = \exp(-0.0009) = 0.9991$$

**Tính toán tương tự cho các samples khác:**

**Sample_2:**
- $z = [37, 52, 16, 4]$
- $x \cdot z = 35 \times 37 + 50 \times 52 + 15 \times 16 + 3 \times 4 = 1295 + 2600 + 240 + 12 = 4147$
- $||z|| = \sqrt{37^2 + 52^2 + 16^2 + 4^2} = \sqrt{1369 + 2704 + 256 + 16} = \sqrt{4345} = 65.92$
- $\cos(\theta) = \frac{4147}{62.92 \times 65.92} = \frac{4147}{4147.7} = 0.9998$
- $D_{cosine} = 1 - 0.9998 = 0.0002$
- Weight = $\exp(-0.0004) = 0.9996$

**Sample_3:**
- $z = [35, 45, 12, 2]$
- $x \cdot z = 35 \times 35 + 50 \times 45 + 15 \times 12 + 3 \times 2 = 1225 + 2250 + 180 + 6 = 3661$
- $||z|| = \sqrt{35^2 + 45^2 + 12^2 + 2^2} = \sqrt{1225 + 2025 + 144 + 4} = \sqrt{3398} = 58.29$
- $\cos(\theta) = \frac{3661}{62.92 \times 58.29} = \frac{3661}{3667.6} = 0.9982$
- $D_{cosine} = 1 - 0.9982 = 0.0018$
- Weight = $\exp(-0.0032) = 0.9968$

**Sample_4:**
- $z = [32, 50, 18, 3]$
- $x \cdot z = 35 \times 32 + 50 \times 50 + 15 \times 18 + 3 \times 3 = 1120 + 2500 + 270 + 9 = 3899$
- $||z|| = \sqrt{32^2 + 50^2 + 18^2 + 3^2} = \sqrt{1024 + 2500 + 324 + 9} = \sqrt{3857} = 62.10$
- $\cos(\theta) = \frac{3899}{62.92 \times 62.10} = \frac{3899}{3907.3} = 0.9979$
- $D_{cosine} = 1 - 0.9979 = 0.0021$
- Weight = $\exp(-0.0044) = 0.9956$

**Sample_5:**
- $z = [38, 55, 15, 5]$
- $x \cdot z = 35 \times 38 + 50 \times 55 + 15 \times 15 + 3 \times 5 = 1330 + 2750 + 225 + 15 = 4320$
- $||z|| = \sqrt{38^2 + 55^2 + 15^2 + 5^2} = \sqrt{1444 + 3025 + 225 + 25} = \sqrt{4719} = 68.70$
- $\cos(\theta) = \frac{4320}{62.92 \times 68.70} = \frac{4320}{4322.6} = 0.9994$
- $D_{cosine} = 1 - 0.9994 = 0.0006$
- Weight = $\exp(-0.0036) = 0.9964$

---

### **Step 2: Fit Linear Model**

Bây giờ tôi sẽ fit một mô hình linear đơn giản:

$$\text{Prediction} = \beta_0 + \beta_1 \times \text{Age} + \beta_2 \times \text{Income} + \beta_3 \times \text{TimeOnSite} + \beta_4 \times \text{PreviousPurchases}$$

Sử dụng **Weighted Least Squares** với trọng số là Weight column:

**Tính toán thủ công (đơn giản hóa):**

Tôi sẽ tính hệ số bằng cách xem sự thay đổi của prediction khi thay đổi từng feature:

**1. Age:**
- Sample_1: Age giảm 2 → Prediction giảm 0.07 → Ảnh hưởng: +0.035/năm
- Sample_2: Age tăng 2 → Prediction tăng 0.07 → Ảnh hưởng: +0.035/năm
- **Trung bình: +0.035**

**2. Income:**
- Sample_3: Income giảm 5 → Prediction giảm 0.20 → Ảnh hưởng: +0.040/triệu
- Sample_5: Income tăng 5 → Prediction tăng 0.03 → Ảnh hưởng: +0.006/triệu
- **Trung bình: +0.023**

**3. TimeOnSite:**
- Sample_3: Time giảm 3 → Prediction giảm 0.20 → Ảnh hưởng: +0.067/phút
- Sample_4: Time tăng 3 → Prediction giảm 0.03 → Ảnh hưởng: -0.010/phút
- **Trung bình: +0.028**

**4. PreviousPurchases:**
- Sample_2: Purchases tăng 1 → Prediction tăng 0.07 → Ảnh hưởng: +0.070/lần
- Sample_5: Purchases tăng 2 → Prediction tăng 0.03 → Ảnh hưởng: +0.015/lần
- **Trung bình: +0.042**

**Kết quả Linear Model:**

$$\text{Prediction} = 0.10 + 0.035 \times \text{Age} + 0.023 \times \text{Income} + 0.028 \times \text{TimeOnSite} + 0.042 \times \text{PreviousPurchases}$$

**Kiểm tra với điểm gốc:**
$$\text{Prediction} = 0.10 + 0.035 \times 35 + 0.023 \times 50 + 0.028 \times 15 + 0.042 \times 3$$
$$= 0.10 + 1.225 + 1.150 + 0.420 + 0.126 = 3.021$$

**Normalize về 0.85:**
$$\text{Scale factor} = \frac{0.85}{3.021} = 0.281$$

---

### **Step 3: Tính Feature Contributions**

Bây giờ tôi áp dụng mô hình linear cho điểm gốc:

| Feature | Value | Coefficient | Contribution | Percentage |
|---------|-------|-------------|--------------|------------|
| Intercept | 1 | 0.10 | +0.10 | 3.3% |
| Age | 35 | 0.035 | +1.225 | 40.6% |
| Income | 50 | 0.023 | +1.150 | 38.1% |
| TimeOnSite | 15 | 0.028 | +0.420 | 13.9% |
| PreviousPurchases | 3 | 0.042 | +0.126 | 4.1% |
| **Total** | | | **+3.021** | **100%** |

**Cách tính:**
- Contribution = Coefficient × Value
- Ví dụ Age: 0.035 × 35 = 1.225
- Percentage = (Contribution / Total) × 100%

**Normalized Contributions (để tổng = 0.85):**

| Feature | Normalized_Contribution | Impact |
|---------|------------------------|--------|
| Age | +0.345 | +40.6% |
| Income | +0.324 | +38.1% |
| TimeOnSite | +0.118 | +13.9% |
| PreviousPurchases | +0.035 | +4.1% |
| Intercept | +0.028 | +3.3% |
| **Total** | **0.85** | **100%** |

---

## Giải Thích Tác Động Local

### **Positive Impact (Tác động tích cực)**

Đây là những features làm **TĂNG** khả năng mua sản phẩm cho người này:

**1. Age = 35 (+0.345 hay +40.6%)**
- **Giải thích:** Tuổi 35 là độ tuổi "vàng" cho sản phẩm này
- **Tại sao:** Người ở độ tuổi này thường có thu nhập ổn định và nhu cầu cao
- **Tác động:** Nếu người này trẻ hơn (25 tuổi), khả năng mua sẽ giảm xuống ~0.50

**2. Income = 50 triệu (+0.324 hay +38.1%)**
- **Giải thích:** Thu nhập 50 triệu/năm là đủ để mua sản phẩm
- **Tại sao:** Có khả năng chi trả tốt
- **Tác động:** Nếu thu nhập giảm xuống 30 triệu, khả năng mua giảm ~0.46

**3. TimeOnSite = 15 phút (+0.118 hay +13.9%)**
- **Giải thích:** Dành nhiều thời gian trên website → quan tâm sản phẩm
- **Tại sao:** Thời gian dài = nghiên cứu kỹ = có ý định mua
- **Tác động:** Nếu chỉ ở 5 phút, khả năng mua giảm ~0.28

**4. PreviousPurchases = 3 lần (+0.035 hay +4.1%)**
- **Giải thích:** Đã mua 3 lần trước đó → khách hàng trung thành
- **Tại sao:** Tin tưởng thương hiệu
- **Tác động:** Nếu chưa mua lần nào, khả năng mua giảm ~0.13

### **Negative Impact (Tác động tiêu cực)**

Trong ví dụ này, **KHÔNG CÓ** feature nào có tác động tiêu cực (hệ số âm). Nhưng để minh họa, tôi sẽ tạo một ví dụ khác:

**Ví dụ: Một người KHÁC với prediction thấp**

| Feature | Value | Coefficient | Contribution | Impact |
|---------|-------|-------------|--------------|--------|
| Age | 22 | 0.035 | +0.770 | +77.0% |
| Income | 20 | 0.023 | +0.460 | +46.0% |
| TimeOnSite | 3 | 0.028 | +0.084 | +8.4% |
| PreviousPurchases | 0 | 0.042 | +0.000 | +0.0% |
| **Total** | | | **1.314** | |
| **Normalized** | | | **0.25** | **25% mua** |

**Negative Impacts (so với người đầu tiên):**

**1. Age = 22 (thay vì 35) → Giảm -0.455**
- **Giải thích:** Tuổi trẻ → thu nhập thấp, chưa có nhu cầu
- **Tác động:** Làm giảm 53.5% khả năng mua

**2. Income = 20 (thay vì 50) → Giảm -0.690**
- **Giải thích:** Thu nhập thấp → không đủ khả năng chi trả
- **Tác động:** Làm giảm 81.2% khả năng mua

**3. TimeOnSite = 3 (thay vì 15) → Giảm -0.336**
- **Giải thích:** Ít thời gian → không quan tâm sản phẩm
- **Tác động:** Làm giảm 39.5% khả năng mua

**4. PreviousPurchases = 0 (thay vì 3) → Giảm -0.126**
- **Giải thích:** Chưa mua lần nào → chưa tin tưởng
- **Tác động:** Làm giảm 14.8% khả năng mua

---

## Visualization - Biểu Đồ LIME

### **Feature Importance Plot**

```
PreviousPurchases (3) ████ +0.035 (4.1%)

TimeOnSite (15)      ████████████ +0.118 (13.9%)

Income (50)            ████████████████████████████████████ +0.324 (38.1%)

Age (35)               ██████████████████████████████████████ +0.345 (40.6%)

                       0.0    0.1    0.2    0.3    0.4
                            Contribution to Prediction
```

### **Comparison Plot (Positive vs Negative Case)**

```
Feature          Negative Case (0.25)    Positive Case (0.85)
─────────────────────────────────────────────────────────────
Age              ████████ (22)           ████████████████ (35)
Income           ████ (20)               ████████████████ (50)
TimeOnSite       █ (3)                   ████████ (15)
PrevPurchases    (0)                     ██ (3)
```

---

## Code Python - LIME với Cosine Distance

### **Cài đặt**

```python
pip install lime
pip install xgboost
pip install scikit-learn
```

### **Ví dụ hoàn chỉnh**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lime import lime_tabular
from sklearn.metrics.pairwise import cosine_distances

# Tạo dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.randint(20, 60, n_samples),
    'Income': np.random.randint(20, 100, n_samples),
    'TimeOnSite': np.random.randint(1, 30, n_samples),
    'PreviousPurchases': np.random.randint(0, 10, n_samples)
})

# Tạo target (logic: người có income cao, time_on_site cao → mua nhiều)
data['Purchase'] = (
    (data['Income'] > 40) & 
    (data['TimeOnSite'] > 10) & 
    (data['PreviousPurchases'] > 1)
).astype(int)

print("Dataset:")
print(data.head(10))
print(f"\nPurchase rate: {data['Purchase'].mean():.2%}")

# Split data
X = data.drop('Purchase', axis=1)
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nModel Performance:")
print(f"Train accuracy: {train_score:.2%}")
print(f"Test accuracy: {test_score:.2%}")

# Chọn một điểm để giải thích
instance_idx = 0
instance = X_test.iloc[instance_idx].values
prediction = model.predict_proba([instance])[0]

print(f"\n=== Instance to Explain ===")
print(f"Features: {dict(zip(X.columns, instance))}")
print(f"Prediction: {prediction[1]:.2%} (class 1 - Purchase)")
print(f"True label: {y_test.iloc[instance_idx]}")

# Tạo LIME explainer với cosine distance
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['No Purchase', 'Purchase'],
    mode='classification',
    random_state=42,
    distance_metric='cosine'  # Sử dụng cosine distance
)

# Giải thích prediction
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=4,
    num_samples=5000
)

# Hiển thị kết quả
print(f"\n=== LIME Explanation (Cosine Distance) ===")
print(f"Intercept: {explanation.intercept[1]:.4f}")
print(f"\nLocal prediction: {explanation.local_pred[1]:.4f}")
print(f"Model prediction: {prediction[1]:.4f}")

print(f"\nFeature Contributions:")
for feature, weight in explanation.as_list(label=1):
    impact = "POSITIVE" if weight > 0 else "NEGATIVE"
    print(f"  {feature:30s} → {weight:+.4f} ({impact})")

# Tính contribution cho từng feature
print(f"\n=== Detailed Feature Analysis ===")
feature_values = dict(zip(X.columns, instance))
for feature, weight in explanation.as_list(label=1):
    feature_name = feature.split()[0]
    if feature_name in feature_values:
        value = feature_values[feature_name]
        contribution = weight * value
        print(f"{feature_name:20s} = {value:6.2f} × {weight:+.4f} = {contribution:+.4f}")

# So sánh với Euclidean distance
print(f"\n=== Comparison: Cosine vs Euclidean Distance ===")
explainer_euclidean = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['No Purchase', 'Purchase'],
    mode='classification',
    random_state=42,
    distance_metric='euclidean'
)

explanation_euclidean = explainer_euclidean.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=4,
    num_samples=5000
)

print("Cosine Distance Results:")
for feature, weight in explanation.as_list(label=1):
    print(f"  {feature:30s} → {weight:+.4f}")

print("\nEuclidean Distance Results:")
for feature, weight in explanation_euclidean.as_list(label=1):
    print(f"  {feature:30s} → {weight:+.4f}")
```

---

## Kết luận

LIME là công cụ mạnh mẽ để giải thích dự đoán của mô hình Machine Learning:

**✅ Ưu điểm:**
- **Model-agnostic**: Hoạt động với mọi mô hình
- **Local explanation**: Giải thích từng dự đoán cụ thể
- **Intuitive**: Dễ hiểu với linear model
- **Flexible**: Áp dụng cho nhiều loại dữ liệu

**❌ Nhược điểm:**
- **Instability**: Kết quả có thể thay đổi
- **Slow**: Cần nhiều perturbed samples
- **Local only**: Không giải thích toàn bộ mô hình
- **Approximation**: Chỉ là xấp xỉ

**🎯 Khi nào sử dụng:**
- Cần giải thích dự đoán cụ thể
- Làm việc với black-box models
- Cần interpretability cao
- Debugging model predictions

LIME giúp làm cho Machine Learning trở nên **transparent** và **trustworthy** hơn!