---
title: "Đại Số Tuyến Tính Ứng Dụng Trong Machine Learning"
description: "Tổng quan đại số tuyến tính: vector, ma trận, hyperplane, các phép toán cơ bản và ứng dụng trong Machine Learning."
pubDatetime: 2025-07-05T01:00:00Z
tags:
  - linear-algebra
  - machine-learning
  - math
  - week2
  - module2
draft: false
---

> **Cập nhật: 1h sáng ngày 5/7/2025**

# Hành Trình Đại Số Tuyến Tính Trong Machine Learning: Từ Dữ Liệu Đến Ứng Dụng

## 1. Mở đầu: Vì sao mọi thứ đều thành vector?

Khi học machine learning, mình nhận ra một điều thú vị: dữ liệu gì rồi cũng chuyển thành vector. Bất kể là số liệu, hình ảnh, chữ viết hay âm thanh – tất cả đều được "nén" lại thành một dãy số. Mô hình học máy không cần biết bạn đang xử lý ảnh mèo hay giá nhà, nó chỉ cần vector, và xử lý theo thuật toán.

---

## 2. Từ dữ liệu thực tế đến vector

**Vector là gì?**
Nó đơn giản là một dãy số. Ví dụ:
- Một căn nhà có diện tích 80m², 2 phòng ngủ → `[80, 2]`
- Một quảng cáo tiêu 150$ cho TV, 25$ cho radio → `[150, 25]`

Chỉ vậy thôi. Mọi thứ biến thành số, rồi sắp xếp lại thành vector.

**Ý nghĩa domain knowledge:**
Nếu bạn muốn hiểu ý nghĩa thật sự của từng con số thì cần chuyên môn ngành – ví dụ bác sĩ, chuyên gia tài chính, nhà báo... Còn model? Nó chỉ cần bạn ném vector vào, thế là xong.

---

## 3. Đại số tuyến tính giúp gì?

Khi đã có vector, bạn áp dụng những gì học trong đại số tuyến tính:
- Nhân ma trận
- Tính dot product
- Tìm hệ số tuyến tính
- Dự đoán $Y = f(X)$

Nói ngắn gọn, mô hình chỉ cần:
> "Input là vector → Xử lý → Output là số hoặc vector khác."

---

## 4. Hyperplane (Siêu phẳng): Đường thẳng – Mặt phẳng – Siêu phẳng

Một khái niệm rất hay gặp trong học máy (đặc biệt là SVM – máy vector hỗ trợ), đó là hyperplane – siêu phẳng.
- Trong không gian 2D: một đường thẳng chia mặt phẳng làm hai phần.
- Trong 3D: một mặt phẳng chia không gian làm hai nửa.
- Trong không gian nhiều chiều hơn (n chiều): ta gọi là siêu phẳng (hyperplane).

Hiểu đơn giản, siêu phẳng là "rào chắn" để phân loại các điểm dữ liệu. Nếu bạn có dữ liệu 2 loại, mô hình sẽ cố gắng tìm ra siêu phẳng để chia chúng ra sao cho hợp lý nhất.

**Ví dụ:**
Bạn có dữ liệu khách hàng gồm `[tuổi, thu nhập, điểm tín dụng]`. Model sẽ tìm một siêu phẳng trong không gian 3D để phân biệt ai là người "có khả năng vay được" và "không đủ điều kiện".

---

## 5. Vector và Ma trận: Định nghĩa, ví dụ, tính chất

### 5.1. Vector là gì?
- Vector là một dãy các con số sắp theo thứ tự.
- Mỗi số trong vector thuộc tập số thực $\mathbb{R}$.
- Ví dụ: Vector 3 chiều: $\vec{v} = [2, -1, 5] \in \mathbb{R}^3$
- Vector có khái niệm độ dài (norm), biểu diễn độ "mạnh" hay "xa" của vector trong không gian.

### 5.2. Ma trận là gì?
- Ma trận là tập hợp các số sắp xếp theo hàng và cột.
- Ma trận kích thước $m \times n$ nghĩa là có $m$ dòng, $n$ cột.
- Vector cột là ma trận $m \times 1$, vector hàng là $1 \times n$.
- Ma trận là cách tổng quát hóa vector, dùng để biểu diễn hệ phương trình, ánh xạ tuyến tính, hoặc lưu trữ dữ liệu nhiều chiều.

---

## 6. Các phép toán cơ bản với vector và ma trận

### 6.1. Cộng và trừ vector
Cộng hoặc trừ từng phần tử tương ứng giữa 2 vector cùng kích thước.

**Ví dụ:**
$\vec{a} = [1, 2], \vec{b} = [3, 4] \Rightarrow \vec{a} + \vec{b} = [4, 6]$

```python
import numpy as np
a = np.array([1,2])
b = np.array([3,4])
print(a + b)  # [4 6]
```

**Tính chất:**
- Giao hoán: $\vec{a} + \vec{b} = \vec{b} + \vec{a}$
- Kết hợp: $(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c})$
- Phần tử trung hòa: $\vec{a} + \vec{0} = \vec{a}$
- Phần tử đối: $\vec{a} + (-\vec{a}) = \vec{0}$

### 6.2. Nhân vector với vô hướng (scalar)
Nhân từng phần tử của vector với một số thực $\lambda$.

**Ví dụ:**
$\lambda \cdot \vec{a} = \lambda \cdot [x_1, x_2] = [\lambda x_1, \lambda x_2]$

```python
import numpy as np
a = np.array([2, 3])
lambda_ = 5
print(lambda_ * a)  # [10 15]
```

**Tính chất:**
- Phân phối: $\lambda(\vec{a} + \vec{b}) = \lambda\vec{a} + \lambda\vec{b}$
- Gộp hệ số: $(\lambda + \mu)\vec{a} = \lambda\vec{a} + \mu\vec{a}$

### 6.3. Tích vô hướng (dot product)
Nhân từng phần tử tương ứng rồi cộng lại:
$\vec{a} \cdot \vec{b} = \sum a_i b_i = a_1b_1 + a_2b_2 + \dots + a_n b_n$

**Code:**
```python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a, b))  # 32
```

**Tính chất:**
- Dùng để tính góc giữa hai vector: $\vec{a} \cdot \vec{b} = \|\vec{a}\| \|\vec{b}\| \cos(\theta)$
- Nếu tích vô hướng = 0 → hai vector vuông góc

**Giải thích hình học & chứng minh:**
> 📐 **Hiểu dot product theo hình học: Góc – Chiếu – Hướng**
>
> ✴️ **1. Dot product là gì?**  
> Với 2 vector $\vec{a}$, $\vec{b}$, dot product là:
> $$
> \vec{a} \cdot \vec{b} = \|\vec{a}\| \cdot \|\vec{b}\| \cdot \cos(\theta)
> $$
> Đây là tích của độ dài vector $a$, độ dài vector $b$, và cos của góc giữa chúng.
>
> 🧭 **2. Nghĩa hình học của từng thành phần**
> - $\|\vec{a}\| \cos(\theta)$: là độ dài hình chiếu của $\vec{a}$ lên hướng của $\vec{b}$
> - $\|\vec{b}\|$: là độ dài của vector $\vec{b}$
>
> => **Dot product chính là độ dài hình chiếu của $a$ lên $b$, rồi nhân với độ dài $b$**
>
> 📌 **Tư duy hình học:**
> - Cùng hướng → dot > 0
> - Ngược hướng → dot < 0
> - Vuông góc → dot = 0
>
> 🔍 **Chi tiết hơn:**
> Nếu bạn có $\vec{a} = a_1 \vec{v}$, $\vec{b} = b_1 \vec{v}$ cùng nằm trên hướng $\vec{v}$ thì:
> $$
> \vec{a} \cdot \vec{b} = a_1 b_1 \|\vec{v}\|^2
> $$
> Cùng hướng → chỉ khác độ dài → dot là tích độ dài.

**Code:**
```python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a, b))  # 32
```

**Tính chất:**
- Giao hoán: $\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$
- Phân phối với phép cộng: $\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}$
- Tương thích với nhân vô hướng: $(\lambda \vec{a}) \cdot \vec{b} = \lambda (\vec{a} \cdot \vec{b})$
- Chuẩn hóa vector: $\vec{a} \cdot \vec{a} = \|\vec{a}\|^2$
- Dot = 0 ↔ 2 vector vuông góc: $\vec{a} \cdot \vec{b} = 0 \Leftrightarrow \vec{a} \perp \vec{b}$

### 6.4. Tích Hadamard (element-wise)
Nhân từng phần tử tương ứng:
$[1, 2, 3] \circ [4, 5, 6] = [4, 10, 18]$

```python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a * b)  # [ 4 10 18]
```

**Tính chất:**
- Không phải là dot product!
- Phải cùng kích thước
- Dùng nhiều trong deep learning (ReLU, attention...)

### 6.5. Nhân ma trận – vector
Mỗi dòng của ma trận dot với vector.

**Ví dụ:**
$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, x = \begin{bmatrix} 5 \\ 6 \end{bmatrix} \Rightarrow Ax = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$

```python
A = np.array([[1,2],[3,4]])
x = np.array([5,6])
print(A @ x)  # [17 39]
```

**Tính chất:**
- Là một ánh xạ tuyến tính: $A: \mathbb{R}^n \rightarrow \mathbb{R}^m$
- Không giao hoán: $Ax \ne xA$

### 6.6. Nhân ma trận – ma trận
Nhân hàng của $A$ với cột của $B$:
$A_{m \times n}, B_{n \times p} \Rightarrow C_{m \times p}$

```python
A = np.array([[1,2],[3,4]])
B = np.array([[2,0],[1,2]])
print(A @ B)  # [[4 4]
              #  [10 8]]
```

**Tính chất:**
- Kết hợp: $A(BC) = (AB)C$
- Không giao hoán: $AB \ne BA$
- Ma trận đơn vị: $AI = IA = A$

### 6.7. Chuyển vị (Transpose)
Hoán đổi hàng ↔ cột:
$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \Rightarrow A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$

```python
A = np.array([[1,2],[3,4]])
print(A.T)  # [[1 3]
            #  [2 4]]
```

**Tính chất:**
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$

### 6.8. Định thức (Determinant)
Số đặc trưng cho ma trận vuông – cho biết ma trận có khả nghịch không:
$\det(A) = 0 \Rightarrow A$ suy biến (không khả nghịch)

```python
A = np.array([[1,2],[3,4]])
print(np.linalg.det(A))  # -2.0
```

**Tính chất:**
- $\det(AB) = \det(A)\det(B)$
- $\det(I) = 1$
- Liên quan đến thể tích biến đổi tuyến tính

### 6.9. Ma trận nghịch đảo (Inverse)
Ma trận $A$ khả nghịch nếu tồn tại $A^{-1}$ sao cho:
$AA^{-1} = A^{-1}A = I$

```python
A = np.array([[1,2],[3,4]])
A_inv = np.linalg.inv(A)
print(A_inv)
```

**Tính chất:**
- Dùng để giải nhanh hệ phương trình: $Ax = b \Rightarrow x = A^{-1}b$
- Không phải ma trận nào cũng có nghịch đảo

### 6.10. Hadamard Division (Chia từng phần tử)
Là phép chia từng phần tử tương ứng của hai vector (hoặc hai ma trận) cùng kích thước:
$[10, 20, 30] \div [2, 4, 5] = [5, 5, 6]$

```python
a = np.array([10,20,30])
b = np.array([2,4,5])
print(a / b)  # [5. 5. 6.]
```

Trong ma trận:
$A = \begin{bmatrix} 8 & 9 \\ 12 & 6 \end{bmatrix}, B = \begin{bmatrix} 2 & 3 \\ 4 & 2 \end{bmatrix} \Rightarrow A \div B = \begin{bmatrix} 4 & 3 \\ 3 & 3 \end{bmatrix}$

**Tính chất:**
- Phải cùng kích thước
- Không liên quan đến phép nhân ma trận hay nghịch đảo
- Dùng nhiều trong deep learning, normalization, attention
- Không định nghĩa được nếu chia cho 0 (phải xử lý hoặc clip)

---

## 7. Giải hệ phương trình $Ax = b$

Một trong những mục tiêu quan trọng nhất của đại số tuyến tính là giải hệ phương trình dạng $Ax = b$ một cách nhanh, gọn, rõ.
- $A$: ma trận hệ số (các hệ số trong phương trình)
- $x$: vector ẩn số (cần tìm)
- $b$: vector kết quả (vế phải)

### Có bao nhiêu lời giải?
Tùy vào số phương trình (hàng) và số biến (cột) mà hệ có thể có:

| Trường hợp                  | Kết quả                                                                 |
|----------------------------|------------------------------------------------------------------------|
| Số phương trình = số biến  | ✅ Thường có 1 nghiệm duy nhất (nếu không suy biến)                    |
| Số phương trình < số biến  | 🔄 Có vô số nghiệm – vì thiếu điều kiện ràng buộc                      |
| Số phương trình > số biến  | ⚠ Có thể vô nghiệm (nếu mâu thuẫn), hoặc vẫn có nghiệm nếu dư thừa     |

**Code minh họa:**
```python
import numpy as np
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)
print(x)  # [3.  2.]
```

---

## 8. Ứng dụng thực tế: Xử lý ảnh, tách nền, threshold, chuẩn hóa, overflow

### 8.1. Kiểu dữ liệu uint8 chỉ lưu được từ 0 đến 255
Nếu bạn cộng hoặc nhân pixel → giá trị có thể vượt 255. Khi đó, NumPy kiểu uint8 sẽ tràn số (overflow):
```python
a = np.array([250], dtype=np.uint8)
print(a + 10)  # 👉 ra 4 vì 260 % 256 = 4
```

### 8.2. Dùng float để tính chính xác hơn
- Không tràn số, giữ được phần thập phân
- Thích hợp khi cần chuẩn hóa ảnh: chia cho 255 để về [0,1]
- Phù hợp cho các phép toán như: làm mờ, tăng sáng, tăng tương phản, tính trung bình, dot product, convolution với ma trận lọc

### 8.3. Chuyển về uint8 để hiển thị hoặc lưu ảnh
- OpenCV, Matplotlib, PIL… yêu cầu ảnh có dtype là uint8
- Khi xử lý xong bằng float → cần clip và ép kiểu lại:
```python
img = np.clip(img_float, 0, 255).astype(np.uint8)
```

### 8.4. Ảnh grayscale từ ảnh màu – tại sao chia 3?
- Ảnh màu RGB có shape: (height, width, 3)
- Nếu bạn muốn gộp 3 kênh lại thành 1 giá trị độ sáng:
```python
gray = (R + G + B) / 3
```
- Chia 3 để giữ giá trị nằm trong [0,255] → tránh tràn số.

### 8.5. Dot product và transpose trong ảnh
- Khi thực hiện dot product để chuyển ảnh màu thành xám:
```python
gray = img @ [0.2989, 0.5870, 0.1140]  # hoặc np.dot(img, weights)
```
- Kết quả shape sẽ là (H, W) hoặc (H, W, 1) tùy cách viết
- Nếu bạn transpose: `img.transpose((1, 0, 2))` → chỉ đổi chiều rộng ↔ chiều cao, còn channel giữ nguyên

### 8.6. cv2.threshold() – Tách nền đơn giản bằng ngưỡng

**Cú pháp:**
```python
_, output = cv2.threshold(input, threshold, max_value, cv2.THRESH_BINARY)
```
| Tham số      | Ý nghĩa                                 |
|--------------|-----------------------------------------|
| input        | Ảnh đầu vào (grayscale hoặc ma trận số) |
| threshold    | Ngưỡng cắt                              |
| max_value    | Giá trị gán nếu > ngưỡng                |
| THRESH_BINARY| Nếu pixel > threshold → gán max_value, ngược lại → gán 0 |

**Ví dụ:**
```python
data = np.array([
    [0, 63, 174],
    [30, 205, 132],
    [52, 178, 210]
], dtype=np.uint8)

_, out = cv2.threshold(data, 100, 255, cv2.THRESH_BINARY)
```
- Ngưỡng 100
- Pixel nào > 100 → gán 255, còn lại gán 0

**Lưu ý:**
- Việc ép kiểu về np.uint8 là bắt buộc vì cv2.threshold() yêu cầu input là ảnh kiểu uint8 hoặc float32.
- Nếu không ép kiểu, OpenCV sẽ báo lỗi: Unsupported depth of input image

---

## 9. So sánh các phép đo ảnh: abs diff, cosine, correlation, outer product vs dot product

### 9.1. Dùng Cosine Similarity thay vì Absolute Difference

| Phép đo                | Đặc điểm                                 | Khi nào dùng?                       |
|------------------------|------------------------------------------|-------------------------------------|
| abs(img1 - img2)       | Nhạy cảm với độ lệch sáng/tương phản     | Khi cần độ chính xác pixel          |
| cosine similarity      | Đo độ song song về hướng, bỏ qua độ dài  | Khi ảnh giống về hình dáng/chung    |
| correlation            | Tổng quát hơn cosine, tính cả offset/scale| Khi có lệch tuyến tính nhẹ         |

- 👉 Cosine không cộng được, chỉ dùng để đo góc
- 👉 Correlation = Cosine tổng quát hơn, có thể tính cho cả chuẩn hóa trung bình và phương sai

### 9.2. Vấn đề về shape – dot product và outer product

| Ma trận A | Ma trận B | A @ B  | Ý nghĩa                        |
|-----------|-----------|--------|-------------------------------|
| (2×1)     | (1×2)     | (2×2)  | Outer Product → tạo ma trận    |
| (1×2)     | (2×1)     | (1×1)  | Dot Product → 1 số duy nhất    |

---

## 10. Tổng kết & trải nghiệm cá nhân

Học machine learning, bạn sẽ thấy mọi thứ đều quy về vector và ma trận. Đại số tuyến tính là nền tảng để hiểu, xử lý và tối ưu dữ liệu. Còn ý nghĩa thực sự của dữ liệu? Đó là câu chuyện của domain knowledge – và là hành trình học hỏi không ngừng!

> *Bài viết này là tổng hợp trải nghiệm học tập của mình trong một ngày học đại số tuyến tính ứng dụng cho ML. Nếu bạn thấy hữu ích, hãy thử áp dụng các ví dụ code vào bài toán của bạn nhé!* 