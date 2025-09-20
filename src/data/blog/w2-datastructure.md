---
title: "Khám phá Sức mạnh của Cấu trúc Dữ liệu trong Python: Từ Cơ bản đến Nâng cao"
description: "Bài viết chuyên sâu về cấu trúc dữ liệu Python: List, Tuple, Set, Dictionary, mutable vs immutable, IoU, NMS, và ứng dụng thực tiễn."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - python
  - data-structure
  - week2
draft: false
---

# Khám phá Sức mạnh của Cấu trúc Dữ liệu trong Python: Từ Cơ bản đến Nâng cao

Bạn có bao giờ tự hỏi làm thế nào máy tính có thể tổ chức và quản lý một lượng lớn thông tin một cách hiệu quả không? Câu trả lời nằm ở **Cấu trúc dữ liệu**. Trong Python, cấu trúc dữ liệu là những cách thức để lưu trữ và sắp xếp dữ liệu, giúp chúng ta truy cập và cập nhật thông tin một cách nhanh chóng và hiệu quả. Việc hiểu rõ và lựa chọn đúng cấu trúc dữ liệu là chìa khóa để viết mã nguồn sạch, tối ưu và dễ bảo trì.

Hãy cùng tìm hiểu sâu hơn về các cấu trúc dữ liệu quan trọng trong Python: **List**, **Tuple**, **Set** và **Dictionary**.

## 1. List: Sức mạnh của Danh sách có thứ tự và Khả biến

List (Danh sách) là một trong những cấu trúc dữ liệu được sử dụng phổ biến nhất trong Python. Nó cho phép bạn lưu trữ một chuỗi các phần tử theo thứ tự.

### Thách thức của Danh sách 1 chiều (1D List)

Hãy hình dung một giáo viên muốn tìm tên học sinh dựa trên vị trí ngồi của họ trong lớp học. Nếu lớp học được tổ chức như một danh sách 1 chiều, việc tìm kiếm sẽ trở nên khá khó khăn, đặc biệt với người mới. Chẳng hạn, để tìm học sinh ở hàng 3, cột 1 trong một danh sách 1 chiều, chúng ta cần một công thức tổng quát để chuyển đổi vị trí 2D sang chỉ số 1D. Công thức có thể là `index = (row - 1) * số_cột + (col - 1)`.

### Giải pháp với Danh sách 2 chiều (2D List)

Để giải quyết bài toán trên, danh sách 2 chiều (2D List) trở thành một lựa chọn tự nhiên và trực quan hơn. Nó giúp chúng ta biểu diễn dữ liệu dưới dạng bảng hoặc ma trận, như vị trí ngồi của học sinh trong lớp.

- **Góc nhìn của Giáo viên (Teacher View)**: Dữ liệu được sắp xếp theo hàng (Row 1, Row 2, Row 3) và cột (Column 1, Column 2). Ví dụ: "Vinh An" ở Hàng 1, Cột 1.
- **Góc nhìn của Lập trình viên (Programmer View)**: Các hàng và cột được đánh chỉ số bắt đầu từ 0. Ví dụ: "Vinh An" có thể được truy cập bằng `student_list[0][0]`. "An" là `student_list[0][1]`.

Danh sách 2 chiều đặc biệt hữu ích khi làm việc với dữ liệu có cấu trúc lưới, chẳng hạn như ảnh RGB (mà có thể được biểu diễn như một tensor 3D).

## 2. Mutable vs. Immutable: Ai có thể thay đổi, ai thì không?

Đây là một khái niệm cực kỳ quan trọng trong Python. Mọi thứ trong Python đều được coi là một đối tượng (object), và mỗi đối tượng có ba thuộc tính chính:

- **Identity (Danh tính)**: Địa chỉ của đối tượng trong bộ nhớ máy tính.
- **Type (Kiểu)**: Loại của đối tượng được tạo ra (ví dụ: int, str, list).
- **Value (Giá trị)**: Giá trị mà đối tượng đó lưu trữ.

### Mutable (Khả biến)
Các đối tượng mà chúng ta có thể thay đổi giá trị đã lưu trữ của chúng sau khi được tạo ra. Điều quan trọng là ID (địa chỉ bộ nhớ) và Type của đối tượng không đổi khi giá trị bên trong được cập nhật. Ví dụ: **List**, **Dictionaries**, **Set**.

> Nếu bạn thay đổi một phần tử trong List, List đó vẫn là cùng một đối tượng nhưng nội dung của nó đã thay đổi.

### Immutable (Bất biến)
Các đối tượng mà chúng ta không thể thay đổi giá trị đã lưu trữ của chúng sau khi được tạo ra. Nếu bạn muốn "thay đổi" một đối tượng bất biến, thực chất Python sẽ tạo ra một đối tượng mới với giá trị mong muốn và biến của bạn sẽ trỏ đến địa chỉ bộ nhớ mới này. Ví dụ: **String**, **Tuples**.

> ID và Type không thể thay đổi đối với các đối tượng đã được tạo.

### Tối ưu bộ nhớ với Object Interning

Python có một kỹ thuật tối ưu bộ nhớ gọi là **Object Interning** (tái sử dụng đối tượng). Kỹ thuật này giúp Python tái sử dụng các đối tượng bất biến đã có sẵn để tiết kiệm bộ nhớ, đặc biệt là với các chuỗi hoặc số nguyên trong một phạm vi nhất định. Điều này có nghĩa là nhiều biến có thể trỏ đến cùng một địa chỉ bộ nhớ nếu chúng có cùng giá trị và là kiểu bất biến.

## 3. Tuple: Dữ liệu "Bất biến" và được Bảo vệ

### Tại sao cần Tuple?

Hãy tưởng tượng bạn có một hàm trả về thông tin cá nhân (tên, tuổi, nghề nghiệp). Nếu bạn dùng List để trả về, bất kỳ ai cũng có thể dễ dàng thay đổi các giá trị này sau khi hàm trả về. Ví dụ: `person[0] = "Your name has been hacked"`. **Tuple ra đời để giải quyết vấn đề này!**

Tuple là một cấu trúc dữ liệu bất biến (immutable), có nghĩa là dữ liệu được bảo vệ và không thể thay đổi sau khi nó được tạo ra. Nó giống như một "chế độ chỉ đọc" cho dữ liệu.

### Cấu trúc và cách sử dụng

```python
# Tạo tuple cơ bản
t = (1, 2, 3)
print(t[0])  # 1
print(t[1])  # 2
print(t[2])  # 3

# Tạo tuple không cần dấu ngoặc đơn
t = 1, 2
print(t)  # (1, 2)
print(type(t))  # <class 'tuple'>

# Tuple với một phần tử (cần dấu phẩy)
singleton = (1,)
var1 = (1 + 2) * 5  # <class 'int'> 15
var2 = (1)          # <class 'int'> 1
var3 = (1,)         # <class 'tuple'> (1,)

# Lặp lại tuple
t = (1,) * 5
print(t)  # (1, 1, 1, 1, 1)

# Nối tuple
t1 = (1, 0)
t1 += (2,)
print(t1)  # (1, 0, 2)
```

### Các thao tác với Tuple

```python
t = (1, 2, 3, 1)
count = t.count(1)  # 2
index = t.index(2)  # 1

# Sử dụng tuple để tạo dictionary
d = dict([('jan', 1), ('feb', 2), ('march', 3)])
print(d['jan'])  # 1

# Unpacking tuple
x1, y1, z1 = ('a', 'b', 'c')
(x2, y2, z2) = ('a', 'b', 'c')
(x3, y3, z3) = range(3)
```

### Tính toàn vẹn của Tuple

Khi nói Tuple là bất biến, điều đó có nghĩa là tập hợp các tham chiếu (references) mà tuple nắm giữ sẽ không bao giờ thay đổi sau khi nó được tạo ra. Nếu các tham chiếu này là các đối tượng bất biến khác (như số nguyên, float, chuỗi), thì tính toàn vẹn tham chiếu đồng nghĩa với tính toàn vẹn giá trị.

### Ứng dụng của Tuple

#### Tính toán IoU (Intersection over Union)

Tuple có thể được sử dụng để lưu trữ tọa độ của các hộp giới hạn, giúp tính toán diện tích giao nhau và hợp nhất của các hộp trong các bài toán thị giác máy tính.

```python
def computeIoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Sử dụng tuple để lưu trữ tọa độ hộp
boxA = (0, 0, 5, 5)
boxB = (2.5, 2.5, 7.5, 7.5)
iou = computeIoU(boxA, boxB)
print(iou)  # 0.20502092050209206
```

#### Non-Maxima Suppression (NMS)

Tuple và List được dùng trong thuật toán NMS để lọc bỏ các hộp dự đoán trùng lặp, chỉ giữ lại những hộp có độ tự tin cao nhất.

```python
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    c = boxes[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(c)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")
```

## 4. Set: Tập hợp Độc đáo và Không theo thứ tự

### Set là gì?

Set (Tập hợp) là một cấu trúc dữ liệu lưu trữ các giá trị, tương tự như List và Tuple, nhưng có hai điểm khác biệt lớn:

- **Không thể có các phần tử trùng lặp**. Set đảm bảo mọi phần tử bên trong nó là duy nhất.
- **Lưu trữ các giá trị không theo thứ tự (unordered)**. Điều này có nghĩa là bạn không thể truy cập các phần tử bằng chỉ số (index).

### Cấu trúc và cách sử dụng

```python
# Tạo set
animals = {"cat", "dog", "tiger"}
print(type(animals))  # <class 'set'>
print(animals)  # {'cat', 'dog', 'tiger'}

# Set với các kiểu dữ liệu khác nhau
a_set = {"cat", 5, True, 40.0}
print(a_set)  # {True, 5, 40.0, 'cat'}

# Không có phần tử trùng lặp
animals = {"cat", "dog", "tiger"}
animals.add("cat")  # Không thêm được vì đã có
print(animals)  # {'cat', 'dog', 'tiger'}
```

### Các thao tác với Set

```python
# Thêm phần tử
animals = {"cat", "dog", "tiger"}
animals.add("bear")
print(animals)  # {'cat', 'dog', 'tiger', 'bear'}

# Thêm một set khác
animals.update({"chicken", "Duck"})
print(animals)  # {'cat', 'dog', 'tiger', 'bear', 'chicken', 'Duck'}

# Gộp hai set
set1 = {"cat", "dog"}
set2 = {"duck", "tiger"}
set3 = set1.union(set2)
print(set3)  # {'cat', 'dog', 'duck', 'tiger'}

# Xóa phần tử
animals.remove("dog")  # Báo lỗi nếu không có
animals.discard("tiger")  # Không báo lỗi nếu không có

# Phép toán tập hợp
set1 = {"apple", "banana", "cherry"}
set2 = {"pineapple", "apple"}
set3 = set1.difference(set2)
print(set3)  # {'banana', 'cherry'}
```

### Không thể chứa các kiểu không băm được

Set không cho phép chứa các phần tử là List hoặc các kiểu dữ liệu khả biến khác vì chúng không thể băm được (unhashable). Lý do là set cần gán một mã băm (hash) cho mỗi phần tử để kiểm tra tính duy nhất và hỗ trợ việc tìm kiếm nhanh.

```python
# Lỗi: không thể chứa list trong set
a_list = [1, 2, 3]
# a_set = {"cat", a_list}  # TypeError: unhashable type: 'list'
```

### Ứng dụng của Set

#### Phân loại văn bản (Text Classification)

Set được sử dụng để xây dựng từ vựng (vocabulary) từ một corpus văn bản. Vì set không chứa các từ trùng lặp và hỗ trợ tìm kiếm nhanh, nó rất hiệu quả trong việc tạo ra một danh sách các từ duy nhất để biểu diễn văn bản dưới dạng các đặc trưng số.

```python
data = [(1, "apple"), (2, "banana"), (3, "apple")]
unique_fruits = set(item[1] for item in data)
print(unique_fruits)  # {'apple', 'banana'}
```

## 5. Dictionary: Ánh xạ Giá trị bằng Khóa có ý nghĩa

### Tại sao cần Dictionary?

Liệu có cách nào dễ hơn để truy cập các phần tử thay vì phải nhớ chỉ số số (index) khó hiểu như trong List? **Dictionary (Từ điển) ra đời để giải quyết vấn đề này!** Thay vì dùng chỉ số số, Dictionary cho phép bạn truy cập giá trị bằng một khóa (key) có ý nghĩa.

### Cấu trúc và đặc điểm

- Dictionary lưu trữ dữ liệu dưới dạng cặp khóa-giá trị (key-value pairs).
- Mỗi khóa phải là duy nhất.
- Các khóa phải là các kiểu dữ liệu bất biến (immutable) như chuỗi, số, hoặc tuple. Các giá trị (value) có thể là bất kỳ kiểu dữ liệu nào, kể cả List, Tuple, hay Set.
- Dictionary được tạo bằng cách sử dụng dấu ngoặc nhọn `{}` với các cặp khóa-giá trị cách nhau bởi dấu hai chấm (`:`) và các cặp cách nhau bởi dấu phẩy (`,`).
- Dictionary là một kiểu dữ liệu khả biến (mutable).

### Các thao tác với Dictionary

```python
# Tạo dictionary
parameters = {
    'learning_rate': 0.1,
    'optimizer': 'Adam',
    'metric': 'Accuracy'
}
print(parameters)  # {'learning_rate': 0.1, 'optimizer': 'Adam', 'metric': 'Accuracy'}

# Truy cập giá trị
value = parameters.get('learning_rate')  # 0.1
value = parameters.get('algorithm')  # None (không báo lỗi)

# Thêm/Cập nhật giá trị
parameters['batch_size'] = 32
parameters.setdefault('epochs', 100)  # Chỉ thêm nếu chưa có

# Xóa phần tử
value = parameters.pop('learning_rate')  # Lấy và xóa
print(parameters)  # {'optimizer': 'Adam', 'metric': 'Accuracy', 'batch_size': 32, 'epochs': 100}

# Lấy keys, values, items
keys = parameters.keys()
values = parameters.values()
items = parameters.items()

for key, value in items:
    print(key, value)
```

### Shallow Copy vs. Deep Copy

Khi sao chép các cấu trúc dữ liệu phức tạp (chứa các đối tượng lồng nhau như list lồng list), cần phân biệt giữa shallow copy và deep copy.

- **Shallow copy** (ví dụ: `a.copy()` cho list, hoặc `dict.copy()`): Chỉ sao chép container bên ngoài. Các phần tử bên trong (nested objects) vẫn trỏ đến cùng một vùng nhớ với đối tượng gốc.
- **Deep copy** (sử dụng `copy.deepcopy()`): Sao chép toàn bộ đối tượng, bao gồm cả các đối tượng lồng nhau. Mọi thay đổi trên bản sao sâu sẽ không ảnh hưởng đến đối tượng gốc.

### Ứng dụng của Dictionary

#### Histogram ảnh (Image Histogram)

Dictionary được sử dụng để tính toán histogram của một ảnh. Trong histogram, mỗi "khóa" là một giá trị pixel (ví dụ: 0-255 cho ảnh grayscale), và "giá trị" là số lần xuất hiện của pixel đó trong ảnh. Điều này giúp phân tích phân bố cường độ sáng của ảnh.

```python
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh grayscale
cat_image = cv2.imread("binary_cat.png", 0)

# Tính histogram
counts = dict()
height, width = cat_image.shape
for row in range(height):
    for col in range(width):
        counts[cat_image[row, col]] = counts.get(cat_image[row, col], 0) + 1

print('Counts', counts)  # {255: 54497, 0: 89983}

# Vẽ histogram
names = list(counts.keys())
values = list(counts.values())
plt.bar(range(len(counts)), values, tick_label=names)
plt.show()
```

## Tóm tắt các Cấu trúc Dữ liệu chính trong Python

| Cấu trúc Dữ liệu | Có thứ tự (Ordered) | Khả biến (Mutable) | Constructor | Ví dụ |
|------------------|---------------------|-------------------|-------------|-------|
| **List** | Có (Yes) | Có (Yes) | `[]` hoặc `list()` | `[5, 7, 'yes', 5.7]` |
| **Tuple** | Có (Yes) | Không (No) | `()` hoặc `tuple()` | `(5.7, 'yes', 5.7)` |
| **Set** | Không (No) | Có (Yes) | `{}` hoặc `set()` | `{5.7, 'yes'}` |
| **Dictionary** | Không (No) | Có (Yes) | `{}` hoặc `dict()` | `{'key': value}` |

> **Lưu ý**: Python 3.7+ đảm bảo thứ tự chèn cho Dictionary, nhưng về mặt lý thuyết, chúng vẫn thường được coi là không có thứ tự trong các định nghĩa cơ bản.

## Kết luận

Hiểu và lựa chọn đúng cấu trúc dữ liệu là nền tảng quan trọng trong lập trình Python. Mỗi loại cấu trúc dữ liệu (List, Tuple, Set, Dictionary) đều có những ưu và nhược điểm riêng, phù hợp với các loại bài toán và yêu cầu khác nhau:

- **List** linh hoạt cho các chuỗi có thứ tự và thay đổi được
- **Tuple** lý tưởng để bảo vệ dữ liệu bất biến
- **Set** tối ưu cho việc lưu trữ các phần tử duy nhất và kiểm tra sự tồn tại nhanh chóng
- **Dictionary** mạnh mẽ khi bạn cần ánh xạ các giá trị bằng các khóa có ý nghĩa

Bằng cách nắm vững những kiến thức này, bạn sẽ có thể viết mã nguồn hiệu quả hơn, dễ đọc hơn và giải quyết các vấn đề phức tạp trong lập trình một cách tự tin!

---

*Bài viết này là một phần của series học tập về Python và cấu trúc dữ liệu. Hãy theo dõi để cập nhật những bài viết tiếp theo về các chủ đề nâng cao khác!* 