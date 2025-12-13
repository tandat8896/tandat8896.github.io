---
title: "Loss Functions cho Classification: Hành trình từ \"Zero-One\" đến \"Cross-Entropy\""
pubDatetime: 2025-01-15T12:00:00Z
featured: false
description: "Tìm hiểu chi tiết về các loại loss functions cho classification: Zero-One Loss, Hinge Loss, Logistic Loss, Cross-Entropy, Focal Loss, Label Smoothing với ví dụ cụ thể và hướng dẫn lựa chọn phù hợp"
tags: ["Machine Learning", "Classification", "Loss Functions", "Cross-Entropy", "Deep Learning"]
---

# Loss Functions cho Classification: Hành trình từ "Zero-One" đến "Cross-Entropy"

Xin chào các bạn! Hôm nay mình muốn chia sẻ về một chủ đề mà mình đã dành rất nhiều thời gian để tìm hiểu: **Loss Functions cho bài toán Classification**.

Khi mới bắt đầu với Machine Learning, mình đã từng nghĩ: "Loss function thì có gì đâu, cứ dùng Cross-Entropy là xong!" Nhưng sau nhiều lần "đau đầu" với các bài toán khác nhau - từ binary classification đến multi-label - mình mới nhận ra rằng việc chọn đúng loss function không hề đơn giản như vậy.

Trong bài viết này, mình sẽ cùng các bạn khám phá các loại loss functions cho classification, từ những cái cơ bản nhất như Zero-One Loss đến những kỹ thuật hiện đại như Label Smoothing. Mỗi loss function đều có những ưu điểm và nhược điểm riêng, và mình sẽ chia sẻ những trải nghiệm thực tế mà mình đã gặp phải.

## I. Nhóm Hàm Loss Cơ bản - Bắt đầu từ những điều đơn giản nhất

### 1. Zero-One Loss Function - Đơn giản nhưng... không thể dùng được!

Mình muốn bắt đầu với Zero-One Loss vì đây là loss function đơn giản nhất mà bạn có thể nghĩ đến. Nó chỉ trả về 0 nếu dự đoán đúng và 1 nếu dự đoán sai. Nghe có vẻ hoàn hảo, phải không? Nhưng thực tế thì...

**Công thức:**

$$L_{0-1}(y, f(x)) = \begin{cases}
    0 & \text{nếu } f(x) \cdot y \geq 0 \\
    1 & \text{nếu } f(x) \cdot y < 0
\end{cases}$$

Trong đó:
- `y`: Nhãn thực tế (1 hoặc -1)
- `f(x)`: Điểm số dự đoán của mô hình
- `f(x) · y ≥ 0`: Dự đoán đúng (cùng dấu)
- `f(x) · y < 0`: Dự đoán sai (khác dấu)

**Ví dụ cụ thể:**

Giả sử chúng ta có 5 mẫu phân loại nhị phân:

| Mẫu | Nhãn thực tế (y) | Điểm số dự đoán f(x) | f(x) · y | Zero-One Loss |
|-----|-----------------|---------------------|----------|---------------|
| 1   | 1               | 0.8                 | 0.8      | 0 (đúng)      |
| 2   | 1               | -0.3                | -0.3     | 1 (sai)       |
| 3   | -1              | -0.9                | 0.9      | 0 (đúng)      |
| 4   | -1              | 0.2                 | -0.2     | 1 (sai)       |
| 5   | 1               | 0.05                | 0.05     | 0 (đúng)      |

**Giải thích:** 
- Mẫu 1, 3, 5: Dự đoán đúng → Loss = 0
- Mẫu 2, 4: Dự đoán sai → Loss = 1
- Tổng Loss = 0 + 1 + 0 + 1 + 0 = **2**

**Ví dụ về thiếu tính linh hoạt:**

**Trường hợp 1: Dự đoán gần đúng (gần ngưỡng 0)**
- Nhãn thực tế: y = 1
- Điểm số dự đoán: f(x) = 0.01 (rất gần 0, nhưng vẫn đúng)
- Zero-One Loss = **0**

**Trường hợp 2: Dự đoán sai nhưng rất gần ngưỡng**
- Nhãn thực tế: y = 1
- Điểm số dự đoán: f(x) = -0.01 (rất gần 0, nhưng sai)
- Zero-One Loss = **1**

**Trường hợp 3: Dự đoán sai rất xa**
- Nhãn thực tế: y = 1
- Điểm số dự đoán: f(x) = -10 (sai rất xa)
- Zero-One Loss = **1**

Đây chính là vấn đề lớn nhất của Zero-One Loss! Cả trường hợp 2 và 3 đều có Loss = 1, mặc dù mức độ sai khác nhau rất nhiều! Zero-One Loss không phân biệt được điều này. Mình đã từng thử dùng nó và nhận ra rằng nó không thể sử dụng trong gradient descent vì không có gradient. Đây là lý do tại sao chúng ta cần các loss functions khác!

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Đơn giản, dễ hiểu | Thiếu tính linh hoạt trong việc đặt ngưỡng |
| Trực tiếp phản ánh độ chính xác | Bỏ qua khoảng cách: không phân biệt sai gần hay sai xa |
| Không cần gradient (không thể tối ưu hóa trực tiếp) | Không thể sử dụng trong gradient descent |
| Phù hợp cho đánh giá cuối cùng | Không cung cấp thông tin về độ tự tin của dự đoán |
| Dễ tính toán | Không khuyến khích mô hình cải thiện dần |

---

### 2. Exponential Loss Function - Khắc phục Zero-One nhưng lại gặp vấn đề khác

Sau khi nhận ra Zero-One Loss không thể dùng được, mình đã tìm đến Exponential Loss như một giải pháp. Nó khắc phục nhược điểm của Zero-One bằng cách giảm Loss từ từ khi tiệm cận 0. Nhưng như bạn sẽ thấy, nó lại gặp một vấn đề khác...

**Công thức:**

$$L_{exp}(y, f(x)) = e^{-f(x) \cdot y}$$

Trong đó:
- `y`: Nhãn thực tế (1 hoặc -1)
- `f(x)`: Điểm số dự đoán của mô hình
- `e`: Số Euler (≈ 2.718)

**Ví dụ cụ thể:**

| Mẫu | Nhãn thực tế (y) | Điểm số dự đoán f(x) | f(x) · y | Exponential Loss |
|-----|-----------------|---------------------|----------|------------------|
| 1   | 1               | 2.0                 | 2.0      | e^(-2.0) ≈ 0.135 |
| 2   | 1               | 1.0                 | 1.0      | e^(-1.0) ≈ 0.368 |
| 3   | 1               | 0.5                 | 0.5      | e^(-0.5) ≈ 0.607 |
| 4   | 1               | -0.5                | -0.5     | e^(0.5) ≈ 1.649  |
| 5   | 1               | -2.0                | -2.0     | e^(2.0) ≈ 7.389  |

**Giải thích:** 
- Khi dự đoán đúng và tự tin (f(x) · y lớn): Loss giảm nhanh (mẫu 1: 0.135)
- Khi dự đoán sai và tự tin (f(x) · y âm lớn): Loss tăng rất nhanh (mẫu 5: 7.389)

**Ví dụ về độ nhạy cảm với Outliers:**

**Trường hợp 1: Không có outlier**
| Mẫu | y | f(x) | f(x)·y | Loss |
|-----|---|------|--------|------|
| 1   | 1 | 1.0  | 1.0    | 0.368|
| 2   | 1 | 0.8  | 0.8    | 0.449|
| 3   | 1 | 0.6  | 0.6    | 0.549|
| 4   | -1| -0.5 | 0.5    | 0.607|
| 5   | -1| -0.8 | 0.8    | 0.449|

Tổng Loss ≈ 2.422

**Trường hợp 2: Có 1 outlier**
| Mẫu | y | f(x) | f(x)·y | Loss |
|-----|---|------|--------|------|
| 1   | 1 | 1.0  | 1.0    | 0.368|
| 2   | 1 | 0.8  | 0.8    | 0.449|
| 3   | 1 | 0.6  | 0.6    | 0.549|
| 4   | -1| -0.5 | 0.5    | 0.607|
| 5   | 1 | -5.0 | -5.0   | 148.4| ← Outlier

Tổng Loss ≈ 150.4 (tăng 62 lần!)

Đây chính là vấn đề mà mình đã gặp phải! Một điểm outlier có thể làm tăng tổng Loss lên rất nhiều (tăng 62 lần!), khiến mô hình tập trung quá mức vào điểm đó và dẫn đến overfitting. Mình đã từng dùng Exponential Loss cho một dataset có nhiều noise và kết quả là mô hình của mình bị "ám ảnh" bởi những điểm khó phân loại nhất, bỏ qua những điểm dễ hơn.

**So sánh với Zero-One Loss:**

| f(x)·y | Zero-One Loss | Exponential Loss |
|--------|---------------|------------------|
| 2.0    | 0             | 0.135            |
| 1.0    | 0             | 0.368            |
| 0.1    | 0             | 0.905            |
| -0.1   | 1             | 1.105            |
| -1.0   | 1             | 2.718            |
| -2.0   | 1             | 7.389            |

Exponential Loss giảm dần khi tiến về 0 (không đột ngột như Zero-One), nhưng tăng rất nhanh khi sai lệch lớn.

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Khắc phục hạn chế của Zero-One Loss | Rất nhạy cảm với outliers |
| Giảm Loss từ từ khi tiệm cận 0 | Dễ dẫn đến overfitting |
| Có gradient liên tục, có thể tối ưu hóa | Tăng theo hàm mũ khi sai lệch lớn |
| Phản ánh độ tự tin của dự đoán | Mô hình có thể tập trung quá mức vào điểm khó |
| Được sử dụng trong AdaBoost | Không phù hợp khi có nhiều noise trong dữ liệu |

--------------------------------------------------------------------------------

## II. Nhóm Hàm Loss Dựa trên Margin (Lề) - Tối đa hóa khoảng cách an toàn

Sau khi "vỡ mộng" với Exponential Loss, mình đã tìm hiểu về Hinge Loss - một loss function được sử dụng rộng rãi trong SVM. Điều thú vị là nó giải quyết vấn đề của Exponential Loss bằng cách tăng Loss theo hướng tuyến tính thay vì hàm mũ.

### Hinge Loss - "Người bạn" của SVM

**Công thức:**

$$L_{hinge}(y, f(x)) = \max(0, 1 - f(x) \cdot y)$$

Trong đó:
- `y`: Nhãn thực tế (1 hoặc -1)
- `f(x)`: Điểm số dự đoán của mô hình
- Margin = 1 (có thể điều chỉnh)

**Ví dụ cụ thể:**

| Mẫu | Nhãn thực tế (y) | Điểm số dự đoán f(x) | f(x) · y | 1 - f(x)·y | Hinge Loss |
|-----|-----------------|---------------------|----------|------------|------------|
| 1   | 1               | 2.0                 | 2.0      | -1.0       | 0 (đúng, margin > 1) |
| 2   | 1               | 1.0                 | 1.0      | 0.0        | 0 (đúng, margin = 1) |
| 3   | 1               | 0.5                 | 0.5      | 0.5        | 0.5 (đúng nhưng margin < 1) |
| 4   | 1               | -0.5                | -0.5     | 1.5        | 1.5 (sai) |
| 5   | -1              | 0.3                 | -0.3     | 1.3        | 1.3 (sai) |

**Giải thích:**
- Mẫu 1, 2: Dự đoán đúng với margin đủ lớn → Loss = 0
- Mẫu 3: Dự đoán đúng nhưng margin nhỏ → Loss = 0.5 (vẫn bị phạt nhẹ)
- Mẫu 4, 5: Dự đoán sai → Loss > 0

**Ví dụ về chấp nhận sai số trong vùng margin:**

**Trường hợp 1: Dự đoán đúng với margin lớn**
- y = 1, f(x) = 3.0
- f(x)·y = 3.0
- Hinge Loss = max(0, 1 - 3.0) = **0**

**Trường hợp 2: Dự đoán đúng nhưng margin nhỏ**
- y = 1, f(x) = 0.8
- f(x)·y = 0.8
- Hinge Loss = max(0, 1 - 0.8) = **0.2**

**Trường hợp 3: Dự đoán sai**
- y = 1, f(x) = -0.5
- f(x)·y = -0.5
- Hinge Loss = max(0, 1 - (-0.5)) = **1.5**

Hinge Loss chấp nhận một phạm vi sai số trong vùng margin (0 < margin < 1) mà không phạt nặng, tập trung vào việc tạo khoảng cách an toàn giữa các lớp.

**So sánh với Exponential Loss:**

| f(x)·y | Hinge Loss | Exponential Loss |
|--------|------------|------------------|
| 2.0    | 0          | 0.135            |
| 1.0    | 0          | 0.368            |
| 0.5    | 0.5        | 0.607            |
| 0.0    | 1.0        | 1.000            |
| -0.5   | 1.5        | 1.649            |
| -1.0   | 2.0        | 2.718            |
| -2.0   | 3.0        | 7.389            |

Đây là điểm khác biệt quan trọng! Hinge Loss tăng tuyến tính (linear) khi sai lệch lớn, trong khi Exponential Loss tăng theo hàm mũ. Điều này làm cho Hinge Loss ít nhạy cảm với outliers hơn, nhưng nó cũng có những hạn chế riêng - đặc biệt là nó không phù hợp với Deep Learning hiện đại.

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Ít nhạy cảm với outliers hơn Exponential Loss | Chấp nhận sai số trong vùng margin |
| Tăng tuyến tính, không tăng quá nhanh | Tập trung vào margin thay vì xác suất chính xác |
| Phù hợp với SVM | Không phù hợp với Deep Learning tiêu chuẩn |
| Giúp tối đa hóa margin giữa các lớp | Không cung cấp xác suất đầu ra |
| Robust hơn Exponential Loss | Ít được sử dụng trong các mô hình hiện đại |

--------------------------------------------------------------------------------

## III. Nhóm Hàm Loss Dựa trên Phân bố (Entropy & KL Divergence) - "Trái tim" của Classification

Đây là nhóm loss functions mà mình nghĩ là quan trọng nhất cho classification. Chúng dựa trên khái niệm entropy và khoảng cách giữa các phân bố xác suất. Khi mình hiểu được cách chúng hoạt động, mọi thứ bỗng trở nên rõ ràng hơn rất nhiều!

### 1. KL Divergence (Kullback-Leibler Divergence)

**Công thức:**

$$D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

Trong đó:
- `P`: Phân bố thực tế (ground truth)
- `Q`: Phân bố dự đoán của mô hình
- `P(i)`: Xác suất của sự kiện i trong phân bố P

**Ví dụ cụ thể:**

So sánh phân bố doanh số năm 2023 và 2024:

| Sản phẩm | P (2023) | Q (2024) | P/Q | log(P/Q) | P × log(P/Q) |
|----------|---------|---------|-----|----------|--------------|
| A        | 0.5     | 0.4     | 1.25| 0.223    | 0.112        |
| B        | 0.3     | 0.4     | 0.75| -0.288   | -0.086       |
| C        | 0.2     | 0.2     | 1.00| 0.000    | 0.000        |

KL Divergence = 0.112 + (-0.086) + 0.000 = **0.026**

**Giải thích:** KL Divergence = 0.026 cho thấy hai phân bố khá giống nhau.

**Ví dụ về tính bất đối xứng:**

**Trường hợp 1: D_KL(P || Q)**
- P = [0.5, 0.3, 0.2]
- Q = [0.4, 0.4, 0.2]
- D_KL(P || Q) = 0.5×log(0.5/0.4) + 0.3×log(0.3/0.4) + 0.2×log(0.2/0.2)
- D_KL(P || Q) ≈ **0.026**

**Trường hợp 2: D_KL(Q || P)**
- Q = [0.4, 0.4, 0.2]
- P = [0.5, 0.3, 0.2]
- D_KL(Q || P) = 0.4×log(0.4/0.5) + 0.4×log(0.4/0.3) + 0.2×log(0.2/0.2)
- D_KL(Q || P) ≈ **0.031**

Đây là một điều thú vị mà mình đã phát hiện ra: D_KL(P || Q) ≠ D_KL(Q || P)! KL Divergence là bất đối xứng, nghĩa là khoảng cách từ P đến Q khác với khoảng cách từ Q đến P. Điều này có thể gây nhầm lẫn nếu bạn không biết, nhưng nó lại có ý nghĩa thống kê sâu sắc.

**Ví dụ về sai lệch khi tính trung bình:**

**Tỷ lệ thay đổi P/Q:**
- Tỷ lệ 1: P/Q = 1.0
- Tỷ lệ 2: P/Q = 0.25
- Tỷ lệ 3: P/Q = 4.0

**Trung bình cộng:**
- Trung bình = (1.0 + 0.25 + 4.0) / 3 = **1.75**

**Trung bình log (sử dụng log để khắc phục):**
- log(1.0) = 0
- log(0.25) = -1.386
- log(4.0) = 1.386
- Trung bình log = (0 + (-1.386) + 1.386) / 3 = **0**

Trung bình cộng bị kéo lệch bởi giá trị 4.0, trong khi trung bình log phản ánh chính xác hơn (tỷ lệ trung bình thực sự là 1.0).

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Đo lường khoảng cách giữa hai phân bố | Bất đối xứng: D_KL(P\|\|Q) ≠ D_KL(Q\|\|P) |
| Có ý nghĩa thống kê rõ ràng | Không phải là metric (không thỏa bất đẳng thức tam giác) |
| Được sử dụng rộng rãi trong thống kê | Cần cả hai phân bố đều có xác suất > 0 |
| Phản ánh sự khác biệt thông tin | Có thể bị sai lệch khi tính trung bình đơn giản |
| Liên quan chặt chẽ với Cross-Entropy | Khó diễn giải trực tiếp |

---

### 2. Cross-Entropy Loss - "Ngôi sao" của Multi-class Classification

Đây là loss function mà mình sử dụng nhiều nhất trong các dự án classification của mình. Cross-Entropy Loss (hay còn gọi là Log Loss) là "ngôi sao" của multi-class classification. Nhưng như mọi "ngôi sao", nó cũng có những hạn chế riêng...

**Công thức:**

$$L_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Trong đó:
- `C`: Số lượng lớp
- `y_i`: Nhãn thực tế (one-hot encoding)
- `ŷ_i`: Xác suất dự đoán của lớp i

**Ví dụ cụ thể - Phân loại 3 lớp:**

**Mẫu 1: Dự đoán đúng và tự tin**
| Lớp | Nhãn thực tế (y) | Xác suất dự đoán (ŷ) | y × log(ŷ) |
|-----|-----------------|---------------------|------------|
| Mèo | 1               | 0.9                 | 1 × log(0.9) = -0.046 |
| Chó | 0               | 0.05                | 0 × log(0.05) = 0 |
| Chim| 0               | 0.05                | 0 × log(0.05) = 0 |

CE Loss = -(-0.046) = **0.046**

**Mẫu 2: Dự đoán đúng nhưng không tự tin**
| Lớp | Nhãn thực tế (y) | Xác suất dự đoán (ŷ) | y × log(ŷ) |
|-----|-----------------|---------------------|------------|
| Mèo | 1               | 0.4                 | 1 × log(0.4) = -0.916 |
| Chó | 0               | 0.3                 | 0 × log(0.3) = 0 |
| Chim| 0               | 0.3                 | 0 × log(0.3) = 0 |

CE Loss = -(-0.916) = **0.916**

**Mẫu 3: Dự đoán sai**
| Lớp | Nhãn thực tế (y) | Xác suất dự đoán (ŷ) | y × log(ŷ) |
|-----|-----------------|---------------------|------------|
| Mèo | 1               | 0.1                 | 1 × log(0.1) = -2.303 |
| Chó | 0               | 0.7                 | 0 × log(0.7) = 0 |
| Chim| 0               | 0.2                 | 0 × log(0.2) = 0 |

CE Loss = -(-2.303) = **2.303**

**Ví dụ về không phù hợp với Multi-label:**

**Bài toán Multi-label: Gán nhãn cho phim**
- Nhãn có thể: "kinh dị", "hình sự", "hài"

**Với CE Loss + Softmax:**
| Nhãn | Xác suất (tổng = 1) |
|------|---------------------|
| Kinh dị | 0.6 |
| Hình sự | 0.3 |
| Hài    | 0.1 |

Đây chính là vấn đề mà mình đã gặp phải khi làm bài toán multi-label! Nếu phim vừa là "kinh dị" vừa là "hình sự", CE Loss không thể xử lý vì Softmax buộc tổng = 1, chỉ cho phép một lớp có xác suất cao. Mình đã từng "vỡ mộng" khi thấy mô hình của mình không thể học được rằng một bộ phim có thể có nhiều thể loại cùng lúc. Đây là lúc mình nhận ra rằng cần phải dùng BCE Loss hoặc Pairwise Ranking Loss cho multi-label!

**Giải pháp:** Sử dụng BCE Loss hoặc Pairwise Ranking Loss cho Multi-label.

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Phù hợp với Multi-class Classification | Không phù hợp với Multi-label Classification |
| Phổ biến nhất trong phân loại đa lớp | Cần kết hợp với Softmax (tổng xác suất = 1) |
| Trừng phạt nặng các lỗi tự tin | Không thể xử lý nhiều nhãn cùng lúc |
| Có gradient tốt, dễ tối ưu hóa | Có thể dẫn đến overconfidence |
| Liên quan chặt chẽ với KL Divergence | Không tính đến mối quan hệ giữa các lớp |

---

### 3. Binary Cross-Entropy Loss (BCE Loss)

**Công thức:**

$$L_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

Trong đó:
- `N`: Số lượng mẫu
- `y_i`: Nhãn thực tế (0 hoặc 1)
- `ŷ_i`: Xác suất dự đoán (0 đến 1)

**Ví dụ cụ thể - Phân loại nhị phân:**

| Mẫu | Nhãn thực tế (y) | Xác suất dự đoán (ŷ) | Loss |
|-----|-----------------|---------------------|------|
| 1   | 1               | 0.9                 | -log(0.9) = 0.105 |
| 2   | 1               | 0.7                 | -log(0.7) = 0.357 |
| 3   | 1               | 0.3                 | -log(0.3) = 1.204 |
| 4   | 0               | 0.2                 | -log(0.8) = 0.223 |
| 5   | 0               | 0.1                 | -log(0.9) = 0.105 |

Tổng Loss = (0.105 + 0.357 + 1.204 + 0.223 + 0.105) / 5 = **0.399**

**Ví dụ về Multi-label Classification:**

**Bài toán: Gán nhãn cho ảnh (có thể có nhiều nhãn)**
- Nhãn: "mèo", "chó", "chim"

**Với BCE Loss (One vs. All):**
| Ảnh | Mèo (y) | Mèo (ŷ) | Chó (y) | Chó (ŷ) | Chim (y) | Chim (ŷ) |
|-----|---------|---------|---------|---------|----------|----------|
| 1   | 1       | 0.9     | 1       | 0.8     | 0        | 0.1      |
| 2   | 0       | 0.2     | 0       | 0.1     | 1        | 0.7      |

**Tính Loss cho từng nhãn độc lập:**
- Loss_mèo = -[1×log(0.9) + 0×log(0.2)] / 2 = 0.053
- Loss_chó = -[1×log(0.8) + 0×log(0.1)] / 2 = 0.112
- Loss_chim = -[0×log(0.1) + 1×log(0.7)] / 2 = 0.178

Đây là một hạn chế lớn của BCE Loss mà mình đã phát hiện ra: BCE Loss không biết rằng "mèo" và "chó" thường xuất hiện cùng nhau trong ảnh 1. Nó xem mỗi nhãn là độc lập, như thể việc phân loại "mèo" không liên quan gì đến việc phân loại "chó". Điều này có thể không phải là vấn đề với các bài toán đơn giản, nhưng với các bài toán phức tạp hơn, bạn có thể cần đến Pairwise Ranking Loss.

**So sánh với CE Loss:**

| Tình huống | CE Loss | BCE Loss |
|------------|---------|----------|
| Binary Classification | Có thể dùng | Phù hợp nhất |
| Multi-class (1 nhãn) | Phù hợp nhất | Không phù hợp |
| Multi-label (nhiều nhãn) | Không phù hợp | Có thể dùng (One vs. All) |

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Phù hợp với Binary Classification | Bỏ qua mối quan hệ giữa các nhãn trong Multi-label |
| Có thể áp dụng cho Multi-label (One vs. All) | Xem mỗi nhãn là độc lập |
| Trừng phạt nặng các lỗi tự tin | Không tối ưu hóa mối quan hệ giữa nhãn |
| Có gradient tốt | Có thể không hiệu quả với Multi-label phức tạp |
| Được sử dụng rộng rãi | Cần nhiều tham số hơn CE Loss |

--------------------------------------------------------------------------------

## IV. Các Kỹ thuật Cải tiến Khác - Những "bí kíp" từ thực tế

Sau khi làm việc với các loss functions cơ bản, mình đã tìm hiểu về các kỹ thuật cải tiến được sử dụng trong thực tế. Đây là những "bí kíp" mà mình đã học được từ các dự án thực tế và các cuộc thi Machine Learning.

### 1. Sparse Categorical Cross-Entropy (SCCE)

**Công thức:**

SCCE có cùng công thức với CE Loss, nhưng sử dụng nhãn số nguyên thay vì one-hot encoding.

$$L_{SCCE} = -\log(\hat{y}_{y_{true}})$$

Trong đó:
- `y_true`: Nhãn số nguyên (ví dụ: 0, 1, 2)
- `ŷ_{y_true}`: Xác suất dự đoán của lớp y_true

**Ví dụ cụ thể:**

**Với CE Loss (One-hot encoding):**
- Nhãn: [0, 1, 0] (lớp 1)
- Vector one-hot: 3 chiều

**Với SCCE (Integer label):**
- Nhãn: 1 (lớp 1)
- Chỉ cần 1 số nguyên

**So sánh bộ nhớ:**

**Bài toán 1000 lớp:**
- CE Loss: Mỗi mẫu cần vector 1000 chiều → 1000 số float
- SCCE: Mỗi mẫu cần 1 số nguyên → 1 số int

**Tiết kiệm:** 
- Bộ nhớ: Giảm ~1000 lần
- Tính toán: Nhanh hơn đáng kể

**Ví dụ tính toán:**

| Mẫu | Nhãn thực tế (y) | Xác suất dự đoán [Lớp 0, Lớp 1, Lớp 2] | SCCE Loss |
|-----|-----------------|----------------------------------------|-----------|
| 1   | 1               | [0.1, 0.8, 0.1]                        | -log(0.8) = 0.223 |
| 2   | 0               | [0.9, 0.05, 0.05]                      | -log(0.9) = 0.105 |
| 3   | 2               | [0.2, 0.1, 0.7]                        | -log(0.7) = 0.357 |

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Tiết kiệm bộ nhớ đáng kể | Chỉ khác CE Loss về cách mã hóa nhãn |
| Nhanh hơn với số lượng lớp lớn | Không có hạn chế cụ thể |
| Phù hợp với bài toán nhiều lớp (1000+) | Ít được biết đến hơn CE Loss |
| Giảm độ phức tạp tính toán | Cần đảm bảo nhãn là số nguyên |
| Tương đương với CE Loss về mặt toán học | |

---

### 2. Label Smoothing (LS)

**Công thức:**

Thay vì nhãn cứng (hard label) [1, 0, 0], sử dụng nhãn mềm (soft label):

$$y_{smooth} = (1 - \alpha) \times y_{hard} + \alpha \times \frac{1}{C}$$

Trong đó:
- `α`: Hệ số smoothing (thường 0.1)
- `C`: Số lượng lớp
- `y_hard`: Nhãn cứng (one-hot)

**Ví dụ cụ thể:**

**Bài toán 3 lớp: Mèo, Chó, Chim**

**Không có Label Smoothing:**
| Lớp | Nhãn cứng (y) |
|-----|--------------|
| Mèo | 1.0          |
| Chó | 0.0          |
| Chim| 0.0          |

**Với Label Smoothing (α = 0.1):**
| Lớp | Nhãn mềm (y_smooth) |
|-----|-------------------|
| Mèo | 0.9 + 0.1/3 = 0.933 |
| Chó | 0.0 + 0.1/3 = 0.033 |
| Chim| 0.0 + 0.1/3 = 0.033 |

**Ví dụ về giảm overconfidence:**

**Mô hình không có Label Smoothing:**
- Dự đoán: [0.99, 0.005, 0.005] (quá tự tin)
- Mô hình nghĩ chắc chắn là "Mèo"

**Mô hình có Label Smoothing:**
- Nhãn mềm: [0.933, 0.033, 0.033]
- Mô hình học được rằng có thể có một chút không chắc chắn
- Dự đoán: [0.95, 0.03, 0.02] (tự tin nhưng không quá mức)

**Ví dụ về mất cân bằng dữ liệu:**

**Trường hợp:** Mô hình thấy quá nhiều "chuối vàng" trong tập train
- Không có LS: Mô hình nghĩ tất cả chuối đều vàng → Overfitting
- Có LS: Mô hình học được rằng có thể có chuối xanh → Tổng quát hóa tốt hơn

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Giảm overfitting | Cần điều chỉnh hệ số α |
| Giúp mô hình tổng quát hóa tốt hơn | Có thể làm giảm độ chính xác một chút |
| Giảm overconfidence | Không phù hợp với tất cả bài toán |
| Phù hợp với dữ liệu mất cân bằng | Cần thử nghiệm để tìm α tối ưu |
| Được sử dụng trong các mô hình hiện đại | Có thể làm chậm quá trình hội tụ |

---

### 3. Pairwise Ranking Loss

**Công thức:**

$$L_{ranking} = \sum_{i \in P} \sum_{j \in N} \max(0, \gamma - (s_i - s_j))$$

Trong đó:
- `P`: Tập các nhãn tích cực (positive labels)
- `N`: Tập các nhãn tiêu cực (negative labels)
- `s_i`: Điểm số của nhãn tích cực i
- `s_j`: Điểm số của nhãn tiêu cực j
- `γ`: Margin (thường = 1)

**Ví dụ cụ thể - Multi-label Classification:**

**Bài toán:** Gán nhãn cho ảnh
- Nhãn có thể: "mèo", "chó", "chim", "cá"

**Ảnh 1:** Có "mèo" và "chó"
- Nhãn tích cực (P): ["mèo", "chó"]
- Nhãn tiêu cực (N): ["chim", "cá"]

**Điểm số dự đoán:**
| Nhãn | Điểm số (s) |
|------|------------|
| Mèo  | 0.9        |
| Chó  | 0.8        |
| Chim | 0.3        |
| Cá   | 0.2        |

**Tính Loss:**
- Cặp (Mèo, Chim): max(0, 1 - (0.9 - 0.3)) = max(0, 0.4) = 0.4
- Cặp (Mèo, Cá): max(0, 1 - (0.9 - 0.2)) = max(0, 0.3) = 0.3
- Cặp (Chó, Chim): max(0, 1 - (0.8 - 0.3)) = max(0, 0.5) = 0.5
- Cặp (Chó, Cá): max(0, 1 - (0.8 - 0.2)) = max(0, 0.4) = 0.4

Tổng Loss = 0.4 + 0.3 + 0.5 + 0.4 = **1.6**

**Ví dụ về học mối quan hệ giữa nhãn:**

**Với BCE Loss:**
- "Mèo" và "Chó" được xử lý độc lập
- Không biết rằng chúng thường xuất hiện cùng nhau

**Với Pairwise Ranking Loss:**
- So sánh điểm số giữa nhãn tích cực và tiêu cực
- Học được thứ tự ưu tiên: "Mèo" > "Chim", "Chó" > "Chim"
- Có thể học được mối quan hệ: nếu có "Mèo" thì thường có "Chó"

**So sánh với BCE Loss:**

| Đặc điểm | BCE Loss | Pairwise Ranking Loss |
|---------|----------|----------------------|
| Xử lý Multi-label | Có (One vs. All) | Có (tối ưu hóa margin) |
| Học mối quan hệ nhãn | Không | Có |
| Thứ tự ưu tiên | Không | Có |
| Tính toán | Đơn giản | Phức tạp hơn (O(P×N)) |
| Phù hợp | Binary, Multi-label đơn giản | Multi-label phức tạp |

**Bảng Ưu điểm và Nhược điểm:**

| Ưu điểm | Nhược điểm |
|---------|------------|
| Tối ưu hóa dựa trên margin | Tính toán phức tạp hơn BCE Loss |
| Học được thứ tự ưu tiên giữa nhãn | Cần nhiều cặp (P×N) để tính toán |
| Có thể học mối quan hệ giữa nhãn | Không phổ biến bằng BCE Loss |
| Phù hợp với Multi-label phức tạp | Cần điều chỉnh margin γ |
| Giải quyết hạn chế của BCE Loss | Có thể chậm với số lượng nhãn lớn |

--------------------------------------------------------------------------------

## Kết luận: Những bài học đã học được

Sau hành trình tìm hiểu về các loss functions cho classification, mình đã rút ra được nhiều bài học quý giá. Việc lựa chọn hàm Loss phù hợp không chỉ là một quyết định kỹ thuật, mà còn ảnh hưởng đến cách mô hình của bạn học và hoạt động.

### Bảng Hướng dẫn Lựa chọn

| Tình huống | Loss Function được khuyến nghị | Lý do |
|------------|-------------------------------|-------|
| Binary Classification | BCE Loss | Phù hợp nhất, trừng phạt nặng lỗi tự tin |
| Multi-class (1 nhãn) | Cross-Entropy Loss | Phổ biến nhất, kết hợp với Softmax |
| Multi-class (nhiều lớp, 1000+) | Sparse Categorical Cross-Entropy | Tiết kiệm bộ nhớ và tính toán |
| Multi-label (nhiều nhãn) | BCE Loss hoặc Pairwise Ranking Loss | BCE cho đơn giản, Pairwise cho phức tạp |
| SVM, tối đa hóa margin | Hinge Loss | Phù hợp với mô hình truyền thống |
| Giảm overconfidence | Cross-Entropy + Label Smoothing | Giúp mô hình tổng quát hóa tốt hơn |
| AdaBoost | Exponential Loss | Được sử dụng trong boosting algorithms |
| Đánh giá cuối cùng | Zero-One Loss | Đơn giản, trực tiếp phản ánh độ chính xác |

### Nguyên tắc vàng

1. **Hiểu rõ bài toán:** Binary, Multi-class, hay Multi-label sẽ quyết định hàm Loss phù hợp.

2. **Xem xét đặc điểm dữ liệu:** Outliers, mất cân bằng, noise sẽ ảnh hưởng đến lựa chọn.

3. **Cân nhắc mối quan hệ nhãn:** Nếu các nhãn có mối quan hệ, cần hàm Loss phù hợp.

4. **Tối ưu hóa tài nguyên:** Với số lượng lớp lớn, ưu tiên SCCE thay vì CE Loss.

5. **Kết hợp kỹ thuật:** Label Smoothing có thể kết hợp với CE Loss để cải thiện hiệu suất.

Cuối cùng, mình muốn nhấn mạnh rằng: việc lựa chọn đúng hàm Loss, giống như việc chọn đúng la bàn, sẽ giúp mô hình của bạn đi đúng hướng và đạt được mục tiêu mong muốn. Mỗi hàm Loss là một công cụ khác nhau, và việc hiểu rõ chúng sẽ giúp bạn xây dựng mô hình hiệu quả hơn.

Mình hy vọng bài viết này sẽ giúp các bạn tránh được những "cạm bẫy" mà mình đã từng vấp phải. Nếu các bạn có câu hỏi hoặc muốn chia sẻ kinh nghiệm của mình, đừng ngại comment bên dưới nhé! Chúng ta cùng học hỏi lẫn nhau! 🚀

