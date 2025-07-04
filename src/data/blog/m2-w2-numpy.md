---
title: "Numpy array cơ bản: Bước đầu với mảng trong Python và ứng dụng AI"
description: "Giới thiệu về Numpy array, cách khởi tạo, thao tác cơ bản, ví dụ thực tiễn và vai trò trong AI/ML. Bài viết dành cho người mới bắt đầu."
pubDatetime: 2025-07-02T01:00:00Z
tags:
  - numpy
  - python
  - ai
  - week1
  - module2
draft: false
--- 
# Hành Trình Khám Phá NumPy: Thư Viện Quyền Năng cho Dữ Liệu và AI

Chào các bạn,

Trong quá trình học về khoa học dữ liệu và Trí tuệ Nhân tạo (AI), tôi đã có cơ hội tìm hiểu sâu về một thư viện vô cùng mạnh mẽ của Python, đó là **NumPy**. Đây không chỉ là một công cụ mà còn là nền tảng cho rất nhiều thư viện khác, đặc biệt trong lĩnh vực tính toán khoa học và xử lý dữ liệu. Hôm nay, tôi muốn chia sẻ hành trình khám phá và những kiến thức quan trọng mà tôi đã thu thập được.

## 1. Giới Thiệu Về NumPy: Hơn Cả Một List Thông Thường

**NumPy là gì?** NumPy là một thư viện Python chuyên dùng cho các tính toán khoa học. Nó cung cấp một cấu trúc dữ liệu đặc biệt gọi là `ndarray` (N-dimensional array), hay còn gọi là mảng đa chiều. Có thể hình dung `ndarray` như là "tensor" - một cấu trúc dữ liệu có thể có bất kỳ số chiều nào. Phiên bản ổn định mà tôi đang học là 2.3.0.

**Tại sao cần NumPy khi đã có list?** Đây là câu hỏi đầu tiên tôi từng thắc mắc. Python list rất linh hoạt, có thể chứa nhiều kiểu dữ liệu khác nhau (ví dụ: tên khách hàng, mã khách hàng, giá tiền) và phù hợp cho các tác vụ lập trình hệ thống nói chung. Nhưng khi làm việc với dữ liệu số lớn, đặc biệt là các phép toán ma trận, xử lý ảnh, thống kê... thì list lại bộc lộ nhiều hạn chế:

- **List lưu trữ "tạp" nhiều kiểu dữ liệu:**
  ```python
  data = ["Nguyen Van A", 123, 4.5]
  # List này chứa cả chuỗi, số nguyên, số thực
  print(data)
  # Output: ['Nguyen Van A', 123, 4.5]
  ```
  Điều này rất tiện cho lưu trữ thông tin tổng hợp, nhưng lại không tối ưu cho tính toán số học.

- **NumPy chỉ lưu trữ một kiểu dữ liệu duy nhất:**
  ```python
  import numpy as np
  arr = np.array([1, 2, 3], dtype=float)
  print(arr)
  print(arr.dtype)
  # Output:
  # [1. 2. 3.]
  # float64
  ```
  Điều này giúp NumPy tối ưu bộ nhớ và tốc độ xử lý, đặc biệt khi thao tác với dữ liệu lớn.

- **Tốc độ vượt trội:** Các phép toán trên list phải lặp từng phần tử, còn NumPy "vector hóa" thao tác, tận dụng tối đa CPU (và có 1 số phiên bản hỗ trợ cả GPU ).
  ```python
  # Cộng từng phần tử với 2
  data = [1, 2, 3]
  result = [x + 2 for x in data]  # List
  print(result)
  # Output: [3, 4, 5]

  arr = np.array([1, 2, 3])
  result_np = arr + 2  # NumPy, nhanh và gọn hơn
  print(result_np)
  # Output: [3 4 5]
  ```

- **Tiết kiệm bộ nhớ:** NumPy lưu trữ dữ liệu liên tục trong bộ nhớ (contiguous), còn list là các object rời rạc, tốn nhiều overhead.

- **Hỗ trợ thao tác ma trận, broadcasting, slicing mạnh mẽ:** Đây là những thứ list không thể làm được hoặc làm rất phức tạp.

**Kết luận nhỏ:** List phù hợp cho dữ liệu "tạp", còn NumPy là lựa chọn số 1 cho dữ liệu số, đặc biệt trong khoa học dữ liệu và AI.

## 2. Cấu Trúc Dữ Liệu ndarray và Các Khái Niệm Cơ Bản

Mảng NumPy có thể có 1 chiều (1D), 2 chiều (2D), 3 chiều (3D) hoặc nhiều chiều hơn nữa. Để xác định kích thước và hình dạng của mảng, chúng ta dùng thuộc tính `shape`:

```python
arr1 = np.array([1, 2, 3])
print(arr1.shape)
# Output: (3,)

arr2 = np.array([[1, 2], [3, 4], [5, 6]])
print(arr2.shape)
# Output: (3, 2)

arr3 = np.zeros((3, 3, 2))
print(arr3.shape)
# Output: (3, 3, 2)
```

Một điểm thú vị là NumPy dùng tuple để biểu diễn shape (ví dụ: `(3,)` thay vì chỉ là `3`). Điều này giúp tránh lỗi khi kiểm tra số chiều của ndarray bằng hàm `len()`.

**Trục (Axis):**
- Axis 0: trục dọc (đi xuống)
- Axis 1: trục ngang (đi ngang)
- Axis 2: chiều sâu (với mảng 3D)

## 3. Các Hàm Phổ Biến Trong NumPy

NumPy có rất nhiều hàm tiện lợi, dưới đây là những hàm mình thấy dùng nhiều nhất trong quá trình học:

### Tạo mảng
- `np.zeros(shape)`: Tạo mảng toàn số 0
  ```python
  print(np.zeros((2, 3)))
  # Output:
  # [[0. 0. 0.]
  #  [0. 0. 0.]]
  ```
- `np.ones(shape)`: Tạo mảng toàn số 1
  ```python
  print(np.ones((2, 3)))
  # Output:
  # [[1. 1. 1.]
  #  [1. 1. 1.]]
  ```
- `np.arange(start, end, step)`: Tạo mảng giá trị liên tiếp
  ```python
  print(np.arange(0, 5, 1))
  # Output: [0 1 2 3 4]
  ```
- `np.linspace(start, stop, num)`: Chia đều khoảng giá trị
  ```python
  print(np.linspace(0, 1, 5))
  # Output: [0.   0.25 0.5  0.75 1.  ]
  ```
- `np.eye(N)`: Ma trận đơn vị
  ```python
  print(np.eye(3))
  # Output:
  # [[1. 0. 0.]
  #  [0. 1. 0.]
  #  [0. 0. 1.]]
  ```
- `np.random.rand(shape)`, `np.random.randn(shape)`, `np.random.randint(low, high, size)`: Sinh số ngẫu nhiên
  ```python
  print(np.random.rand(2, 2))  # Số thực [0,1)
  print(np.random.randint(0, 10, (2, 3)))  # Số nguyên từ 0 đến 9
  # Output ví dụ:
  # [[0.1 0.7]
  #  [0.3 0.2]]
  # [[2 5 7]
  #  [1 0 9]]
  ```

### Thay đổi hình dạng mảng
- `reshape()`: Thay đổi shape mà không đổi số phần tử
  ```python
  a = np.arange(6)
  print(a.reshape(2, 3))
  # Output:
  # [[0 1 2]
  #  [3 4 5]]
  ```
- `flatten()`, `ravel()`: Làm phẳng mảng thành 1D
  ```python
  b = np.array([[1, 2], [3, 4]])
  print(b.flatten())
  # Output: [1 2 3 4]
  print(b.ravel())
  # Output: [1 2 3 4]
  ```
- `repeat()`: Lặp lại phần tử theo trục
  ```python
  data = np.array([[1, 2], [3, 4]])
  print(np.repeat(data, 2, axis=0))  # Lặp hàng
  # Output:
  # [[1 2]
  #  [1 2]
  #  [3 4]
  #  [3 4]]
  print(np.repeat(data, 2, axis=1))  # Lặp cột
  # Output:
  # [[1 1 2 2]
  #  [3 3 4 4]]
  ```

### Thao tác số học, thống kê
- `arr.sum()`, `arr.mean()`, `arr.std()`, `arr.min()`, `arr.max()`, `arr.argmin()`, `arr.argmax()`
  ```python
  arr = np.array([1, 2, 3, 4])
  print(arr.sum())    # Output: 10
  print(arr.mean())   # Output: 2.5
  print(arr.std())    # Output: 1.118...
  print(arr.min())    # Output: 1
  print(arr.argmax()) # Output: 3 (vị trí giá trị lớn nhất)
  ```
- `np.unique(arr)`: Lấy các giá trị duy nhất
  ```python
  arr = np.array([1, 2, 2, 3])
  print(np.unique(arr))
  # Output: [1 2 3]
  ```

### Thao tác logic, điều kiện
- `arr > 2`, `arr[arr > 2]`, `np.where(condition, x, y)`
  ```python
  arr = np.array([1, 2, 3, 4])
  print(arr > 2)
  # Output: [False False  True  True]
  print(arr[arr > 2])
  # Output: [3 4]
  print(np.where(arr > 2, 100, 0))
  # Output: [  0   0 100 100]
  ```

### Một số hàm khác rất hữu ích
- `np.dot(a, b)`: Tích vô hướng/ma trận
- `np.matmul(a, b)`: Nhân ma trận
- `np.transpose(a)`, `a.T`: Chuyển vị
- `np.clip(arr, min, max)`: Giới hạn giá trị
- `arr.astype(dtype)`: Đổi kiểu dữ liệu
- `np.char.replace(arr, old, new)`: Thay thế chuỗi trong mảng ký tự

## 4. Lập Chỉ Mục (Indexing) và Broadcasting

**Indexing:**
- Slicing: `arr[1:3, :]` lấy các hàng từ 1 đến 2, tất cả các cột
- Lấy hàng/cột: `arr[0, :]` (hàng đầu), `arr[:, 1]` (cột thứ hai)
- Dùng list: `arr[[0, 2], :]` lấy hàng 0 và 2
- Dùng boolean: `arr[arr > 2]` lấy phần tử lớn hơn 2

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1, :])
# Output: [4 5 6]
print(arr[:, 0])
# Output: [1 4]
print(arr[arr > 2])
# Output: [3 4 5 6]
```

**Lưu ý:** Slicing thường trả về view, sửa slice sẽ làm thay đổi mảng gốc.

**Broadcasting:**
- Phép toán giữa mảng và số: `arr + 2`
- Phép toán giữa mảng và mảng khác shape (nếu phù hợp quy tắc broadcasting)

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 10)
# Output:
# [[11 12 13]
#  [14 15 16]]
```

**Hàm Softmax:**
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(np.array([1.0, 2.0, 3.0])))
# Output: [0.09003057 0.24472847 0.66524096]
```

## 5. Ứng Dụng AI: Xử Lý Dữ Liệu Hình Ảnh và Hơn Thế Nữa

**Dữ liệu hình ảnh:**
- Ảnh grayscale: mỗi pixel là số từ 0-255, shape=(H, W)
- Ảnh màu: mỗi pixel là (R, G, B), shape=(H, W, 3)

**OpenCV và NumPy:**
- OpenCV đọc ảnh theo kênh BGR, Matplotlib hiển thị theo RGB. Nếu không chuyển đổi, ảnh sẽ bị sai màu.
- Chuyển đổi: `image_rgb = image[:, :, ::-1]` hoặc `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`

**Thay đổi độ sáng ảnh:**
- Khi cộng/trừ giá trị vào pixel kiểu uint8, dễ bị tràn giá trị (overflow). Dùng `np.clip()` để giới hạn giá trị trong [0, 255].

```python
import cv2
img = cv2.imread('image1.png', 1)
brighter = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
print(brighter.shape)
# Output: (chiều cao, chiều rộng, 3)
```

**Các ứng dụng khác:**
- Dự báo thời tiết: reshape dữ liệu nhiệt độ
- One-hot encoding: chuyển nhãn số thành vector
  ```python
  labels = np.array([0, 1, 2])
  one_hot = np.eye(3)[labels]
  print(one_hot)
  # Output:
  # [[1. 0. 0.]
  #  [0. 1. 0.]
  #  [0. 0. 1.]]
  ```
- Xử lý văn bản: `np.genfromtxt()`, `np.unique()`, `np.char.replace()`, `astype()`
  ```python
  arr = np.array(['cat', 'dog', 'cat'])
  print(np.unique(arr))
  # Output: ['cat' 'dog']
  print(np.char.replace(arr, 'cat', 'mouse'))
  # Output: ['mouse' 'dog' 'mouse']
  ```

## Tổng Kết

Hành trình khám phá NumPy đã mang lại cho tôi rất nhiều kiến thức giá trị. Từ việc hiểu bản chất của ndarray và các trục, đến việc sử dụng các hàm tạo, thay đổi hình dạng, lập chỉ mục mạnh mẽ (bao gồm cả slicing với cơ chế view/copy), và các phép toán trên mảng. Đặc biệt, tôi đã thấy rõ vai trò không thể thiếu của NumPy trong các ứng dụng AI, từ xử lý ảnh (với những lưu ý về kênh màu và kiểu dữ liệu uint8) cho đến các bài toán dự báo và tiền xử lý dữ liệu.

NumPy thực sự là một công cụ nền tảng, giúp chúng ta thao tác với dữ liệu số một cách hiệu quả và linh hoạt, mở ra cánh cửa cho nhiều ứng dụng phức tạp trong khoa học dữ liệu và trí tuệ nhân tạo.

Hy vọng bài chia sẻ này hữu ích cho các bạn! 