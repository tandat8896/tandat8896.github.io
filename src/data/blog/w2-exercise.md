---
title: "Week 2 Day 6: Bài Tập Thực Hành Python - Cấu Trúc Dữ Liệu Cơ Bản"
description: "4 bài tập thực hành Python: sliding window, character counting, word counting, và ứng dụng cấu trúc dữ liệu trong NLP."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - python
  - exercise
  - data-structure
  - week2
draft: false
---

# Week 2 Day 6: Bài Tập Thực Hành Python - Cấu Trúc Dữ Liệu Cơ Bản

## Giới Thiệu

Chào mừng bạn đến với buổi ôn tập quan trọng về các cấu trúc dữ liệu cơ bản trong Python! Hôm nay chúng ta sẽ khám phá 4 bài tập thực hành quan trọng giúp bạn nắm vững List, Tuple, Set, Dictionary và các kỹ thuật xử lý dữ liệu cơ bản. Những bài tập này không chỉ là thử thách để củng cố kiến thức mà còn là nền tảng vững chắc cho những dự án phức tạp hơn, đặc biệt trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP).

---

## Exercise 1: Getting Max Over Kernel (Tìm giá trị lớn nhất trong cửa sổ trượt)

### 🎯 Motivation (Động lực)
Bài toán này mô phỏng các tình huống thực tế trong:
- **Phân tích dữ liệu**: Tìm kiếm các mẫu hoặc giá trị cực đại trong một "cửa sổ" dữ liệu di động
- **Xử lý tín hiệu số**: Phân tích tín hiệu theo từng đoạn thời gian
- **Thuật toán xử lý ảnh**: Tìm kiếm các đặc điểm trong từng vùng ảnh

Nó giúp bạn làm quen với kỹ thuật **"sliding window"** (cửa sổ trượt) – một kỹ thuật rất hiệu quả để giải quyết các vấn đề liên quan đến dãy con hoặc đoạn con.

### 🚀 Challenging Aspects (Thử thách)
- Hiểu rõ cách cửa sổ k di chuyển trên danh sách `num_list`
- Làm thế nào để truy cập các phần tử trong mỗi cửa sổ con một cách hiệu quả
- Sử dụng các phương thức của List và hàm `max()` tích hợp sẵn của Python một cách tối ưu

### 📊 Input/Output
**Input**: 
- Một danh sách số nguyên (`num_list`)
- Một số nguyên `k` (kích thước cửa sổ)

**Output**: 
- Một danh sách mới chứa giá trị lớn nhất của mỗi cửa sổ trượt

**Ví dụ**:
```python
num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
# Output: [5, 5, 5, 5, 10, 12, 33, 33]
```

### 🔍 Manual Debugging/Walkthrough (Gỡ lỗi/Thực hành thủ công)

Hãy hình dung `num_list = [3, 4, 5, 1, -44, 5]` và `k = 3`:

1. **Cửa sổ 1**: `[3, 4, 5]` → Max là **5**. Danh sách kết quả: `[5]`
2. **Cửa sổ 2**: Dịch sang phải 1 vị trí. `[4, 5, 1]` → Max là **5**. Danh sách kết quả: `[5, 5]`
3. **Cửa sổ 3**: Dịch sang phải 1 vị trí. `[5, 1, -44]` → Max là **5**. Danh sách kết quả: `[5, 5, 5]`
4. **Cửa sổ 4**: Dịch sang phải 1 vị trí. `[1, -44, 5]` → Max là **5**. Danh sách kết quả: `[5, 5, 5, 5]`

### 💡 Giải Pháp
```python
def max_kernel(num_list, k):
    """
    Trả về danh sách các giá trị lớn nhất trong mỗi cửa sổ con (window) kích thước k
    chạy trượt trên danh sách num_list.
    """
    result = []
    
    # Duyệt qua tất cả các vị trí bắt đầu có thể của cửa sổ
    for i in range(len(num_list) - k + 1):
        # Lấy cửa sổ con từ vị trí i đến i+k
        window = num_list[i:i+k]
        # Tìm giá trị lớn nhất trong cửa sổ
        max_value = max(window)
        result.append(max_value)
    
    return result
```

### 🎯 Kỹ thuật Slicing trong List
Sử dụng kỹ thuật slicing `num_list[i:i+k]` để dễ dàng trích xuất các cửa sổ con:
- `i` là vị trí bắt đầu
- `i+k` là vị trí kết thúc (không bao gồm)
- Kết quả là một danh sách con có k phần tử

---

## Exercise 2: Character Counting (Đếm ký tự)

### 🎯 Motivation (Động lực)
Đây là bài tập khởi đầu cho các tác vụ phân tích văn bản cơ bản. Việc đếm tần suất xuất hiện của ký tự giúp bạn:
- Hiểu được "đặc điểm" của một từ hoặc một chuỗi
- Là bước đệm quan trọng cho việc xử lý ngôn ngữ tự nhiên sau này
- Ứng dụng trong việc phân tích tần suất ký tự, mã hóa thông tin

### 🚀 Challenging Aspects (Thử thách)
- Sử dụng Dictionary một cách hiệu quả để lưu trữ và cập nhật số lượng ký tự
- Kiểm tra xem một ký tự đã là "key" trong dictionary hay chưa
- Xử lý các trường hợp chữ hoa/chữ thường khác nhau (ví dụ: 'B' và 'b')

### 📊 Input/Output
**Input**: Một từ (chuỗi ký tự)

**Output**: Một Dictionary với key là các ký tự và value là số lần xuất hiện của chúng

**Ví dụ**:
```python
Input: "smiles"
Output: {"e": 1, "i": 1, "l": 1, "m": 1, "s": 2}
```

### 🔍 Manual Debugging/Walkthrough (Gỡ lỗi/Thực hành thủ công)

Giả sử bạn có từ `word = "baby"`:

1. **Khởi tạo**: `character_statistic = {}`

2. **Duyệt qua từng ký tự**:
   - **Ký tự 'b'**: 'b' chưa có trong `character_statistic`. Thêm nó vào: `{'b': 1}`
   - **Ký tự 'a'**: 'a' chưa có. Thêm nó vào: `{'b': 1, 'a': 1}`
   - **Ký tự 'b'**: 'b' đã có. Tăng giá trị của nó lên 1: `{'b': 2, 'a': 1}`
   - **Ký tự 'y'**: 'y' chưa có. Thêm nó vào: `{'b': 2, 'a': 1, 'y': 1}`

3. **Kết thúc**: `character_statistic = {'b': 2, 'a': 1, 'y': 1}`

### 💡 Giải Pháp
```python
def count_character(word):
    """
    Đếm số lần xuất hiện của từng ký tự trong chuỗi đầu vào.
    """
    character_statistic = {}
    
    # Duyệt qua từng ký tự trong chuỗi
    for char in word:
        # Kiểm tra xem ký tự đã có trong dictionary chưa
        if char in character_statistic:
            # Nếu có, tăng số đếm lên 1
            character_statistic[char] += 1
        else:
            # Nếu chưa có, thêm vào với giá trị 1
            character_statistic[char] = 1
    
    return character_statistic
```

### 🎯 Cách Sử Dụng Dictionary Hiệu Quả
- Sử dụng `in` để kiểm tra key có tồn tại
- Sử dụng `get()` method với giá trị mặc định: `character_statistic.get(char, 0) + 1`
- Hoặc sử dụng `defaultdict` từ collections module

---

## Exercise 3: Word Counting (Đếm từ)

### 🎯 Motivation (Động lực)
Bài tập này mở rộng từ việc đếm ký tự sang đếm từ, là một kỹ năng cốt lõi trong NLP để:
- **Phân tích tần suất từ**: Hiểu từ nào xuất hiện nhiều nhất trong văn bản
- **Xây dựng từ điển**: Tạo vocabulary cho các mô hình học máy
- **Chuẩn bị dữ liệu**: Tiền xử lý văn bản cho các thuật toán NLP

### 🚀 Challenging Aspects (Thử thách)
- **Đọc file**: Làm thế nào để mở và đọc nội dung từ một file .txt
- **Tiền xử lý văn bản**:
  - Chuyển đổi toàn bộ văn bản sang chữ thường để coi "The" và "the" là cùng một từ
  - Loại bỏ các dấu câu như dấu chấm (.) và dấu phẩy (,) để đảm bảo các từ được đếm chính xác
  - Tách văn bản thành các từ riêng lẻ
- **Sử dụng Dictionary** để đếm tần suất xuất hiện của từng từ

### 📊 Input/Output
**Input**: Đường dẫn đến một file .txt

**Output**: Một Dictionary với key là các từ và value là số lần xuất hiện của chúng

**Ví dụ**: 
```python
{"a": 7, "again": 1, "and": 1, "are": 1, "at": 1, "be": 1, "become": 2, ...}
```

### 🔍 Manual Debugging/Walkthrough (Gỡ lỗi/Thực hành thủ công)

Giả sử bạn có một dòng từ file: `"Hello, world! How are you?"`

1. **Chuyển về chữ thường**: `"hello, world! how are you?"`

2. **Loại bỏ dấu câu**: `"hello world how are you"` (sau khi loại bỏ `,` và `!`)

3. **Tách thành các từ**: `["hello", "world", "how", "are", "you"]`

4. **Đếm bằng Dictionary**:
   - Khởi tạo `counter = {}`
   - Duyệt qua danh sách từ:
     - `"hello"`: `{'hello': 1}`
     - `"world"`: `{'hello': 1, 'world': 1}`
     - `"how"`: `{'hello': 1, 'world': 1, 'how': 1}`
     - `"are"`: `{'hello': 1, 'world': 1, 'how': 1, 'are': 1}`
     - `"you"`: `{'hello': 1, 'world': 1, 'how': 1, 'are': 1, 'you': 1}`

### 💡 Giải Pháp
```python
def preprocess_text(sentence):
    """
    Tiền xử lý một câu bằng cách:
    - Chuyển tất cả các ký tự thành chữ thường
    - Loại bỏ dấu chấm (.) và dấu phẩy (,)
    - Tách câu thành danh sách các từ
    """
    # Chuyển về chữ thường
    sentence = sentence.lower()
    
    # Loại bỏ dấu câu
    sentence = sentence.replace('.', '').replace(',', '')
    
    # Tách thành các từ
    words = sentence.split()
    
    return words

def count_word(file_path):
    """
    Đếm số lần xuất hiện của từng từ trong văn bản đầu vào sau khi tiền xử lý.
    """
    counter = {}
    
    # Đọc file
    with open(file_path, 'r') as f:
        document = f.read()
    
    # Tiền xử lý văn bản
    words = preprocess_text(document)
    
    # Đếm từ
    for word in words:
        if word in counter:
            counter[word] += 1
        else:
            counter[word] = 1
    
    return counter
```

### 🎯 Kỹ Thuật Xử Lý File và Văn Bản
- Sử dụng `with open()` để đảm bảo file được đóng đúng cách
- Sử dụng `replace()` để loại bỏ dấu câu
- Sử dụng `split()` để tách văn bản thành từ
- Sử dụng `lower()` để chuẩn hóa chữ hoa/thường

---

## Exercise 4: Levenshtein Distance (Khoảng cách chỉnh sửa Levenshtein)

### 🎯 Motivation (Động lực)
Khoảng cách Levenshtein là một thước đo quan trọng cho sự khác biệt giữa hai chuỗi ký tự. Nó được ứng dụng rộng rãi trong:
- **Kiểm tra chính tả**: Như spellchecker của Word
- **Tìm kiếm thông tin**: Tìm kiếm fuzzy search
- **Sinh học**: So sánh chuỗi DNA
- **Gợi ý từ**: Đề xuất từ gần đúng khi người dùng gõ sai

Bài toán này giới thiệu bạn đến với thuật toán **quy hoạch động (dynamic programming)**.

### 🚀 Challenging Aspects (Thử thách)
- Hiểu bản chất của ba phép biến đổi cơ bản: xoá (delete), thêm (insert), thay thế (substitute)
- Xây dựng và khởi tạo ma trận (matrix) D để lưu trữ khoảng cách chỉnh sửa
- Nắm vững công thức đệ quy để tính toán giá trị của mỗi ô D[i, j]
- Đọc hiểu và hình dung quá trình "backtrace" để tìm ra chuỗi các phép biến đổi tối thiểu

### 📊 Input/Output
**Input**: Hai chuỗi ký tự (ví dụ: `token1` và `token2`)

**Output**: Một số nguyên biểu thị khoảng cách Levenshtein tối thiểu

**Ví dụ**:
```python
Input: "kitten", "sitting"
Output: 3 (vì cần 3 bước: k->s, e->i, thêm g)

Input: "hola", "hello"
Output: 3
```

### 🔍 Manual Debugging/Walkthrough (Gỡ lỗi/Thực hành thủ công)

Hãy lấy ví dụ `"yu"` (source) và `"you"` (target):

#### Bước 1: Xây dựng ma trận D
Tạo ma trận D với M = len(source) + 1 hàng và N = len(target) + 1 cột.
Trong trường hợp này là D[3][4]:

```
    y  o  u
  ┌───┬───┬───┬───┐
  │   │ y │ o │ u │
──┼───┼───┼───┼───┤
y │   │   │   │   │
──┼───┼───┼───┼───┤
u │   │   │   │   │
──┴───┴───┴───┴───┘
```

#### Bước 2: Hoàn thiện hàng và cột đầu tiên
- **Hàng đầu tiên** (biến đổi từ chuỗi rỗng # thành target): D[0,0]=0, D[0,1]=1, D[0,2]=2, D[0,3]=3
- **Cột đầu tiên** (biến đổi từ source thành chuỗi rỗng #): D[0,0]=0, D[1,0]=1, D[2,0]=2

```
    y  o  u
  ┌───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │
──┼───┼───┼───┼───┤
y │ 1 │   │   │   │
──┼───┼───┼───┼───┤
u │ 2 │   │   │   │
──┴───┴───┴───┴───┘
```

#### Bước 3: Tính toán các giá trị còn lại
Sử dụng công thức: D[i, j] = min(D[i-1, j] + del_cost, D[i, j-1] + ins_cost, D[i-1, j-1] + sub_cost)

- **D[1,1]** (so sánh source='y' và target='y'):
  - Xóa: D[0,1] + 1 = 1 + 1 = 2
  - Thêm: D[1,0] + 1 = 1 + 1 = 2
  - Thay thế/Giữ nguyên: D[0,0] + 0 = 0 + 0 = 0 (vì 'y' == 'y')
  - D[1,1] = min(2, 2, 0) = **0**

- **D[1,2]** (so sánh source='y' và target='o'):
  - Xóa: D[0,2] + 1 = 2 + 1 = 3
  - Thêm: D[1,1] + 1 = 0 + 1 = 1
  - Thay thế: D[0,1] + 1 = 1 + 1 = 2
  - D[1,2] = min(3, 1, 2) = **1**

- **D[2,1]** (so sánh source='u' và target='y'):
  - Xóa: D[1,1] + 1 = 0 + 1 = 1
  - Thêm: D[2,0] + 1 = 2 + 1 = 3
  - Thay thế: D[1,0] + 1 = 1 + 1 = 2
  - D[2,1] = min(1, 3, 2) = **1**

Ma trận hoàn chỉnh:
```
    y  o  u
  ┌───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │
──┼───┼───┼───┼───┤
y │ 1 │ 0 │ 1 │ 2 │
──┼───┼───┼───┼───┤
u │ 2 │ 1 │ 1 │ 1 │
──┴───┴───┴───┴───┘
```

#### Bước 4: Kết quả
Giá trị cuối cùng ở ô D[2,3] = **1** chính là khoảng cách Levenshtein.

### 💡 Giải Pháp
```python
def levenshtein_distance(token1, token2):
    """
    Tính khoảng cách Levenshtein (edit distance) giữa hai chuỗi.
    """
    # Khởi tạo ma trận khoảng cách
    distances = [[0] * (len(token2) + 1) for _ in range(len(token1) + 1)]
    
    # Khởi tạo hàng đầu tiên và cột đầu tiên
    for i in range(len(token1) + 1):
        distances[i][0] = i
    for j in range(len(token2) + 1):
        distances[0][j] = j
    
    # Tính toán các khoảng cách
    for i in range(1, len(token1) + 1):
        for j in range(1, len(token2) + 1):
            # Chi phí thay thế: 0 nếu giống nhau, 1 nếu khác nhau
            substitution_cost = 0 if token1[i-1] == token2[j-1] else 1
            
            # Công thức đệ quy
            distances[i][j] = min(
                distances[i-1][j] + 1,      # Xóa
                distances[i][j-1] + 1,      # Thêm
                distances[i-1][j-1] + substitution_cost  # Thay thế
            )
    
    return distances[len(token1)][len(token2)]
```

### 🎯 Thuật Toán Quy Hoạch Động
- **Ý tưởng**: Chia bài toán lớn thành các bài toán con nhỏ hơn
- **Lưu trữ kết quả**: Tránh tính toán lặp lại
- **Công thức đệ quy**: Dựa trên kết quả của các bài toán con đã giải

---

## Kết Luận

Các bài tập này đã giúp bạn:

1. **Nắm vững cấu trúc dữ liệu cơ bản**: List, Dictionary, Set, Tuple
2. **Hiểu kỹ thuật Sliding Window**: Ứng dụng trong xử lý dữ liệu tuần tự
3. **Thực hành xử lý văn bản**: Tiền xử lý, đếm tần suất, phân tích
4. **Làm quen với Dynamic Programming**: Thuật toán Levenshtein Distance

Những kỹ năng này sẽ là nền tảng vững chắc cho việc học các thuật toán phức tạp hơn và ứng dụng trong các dự án NLP thực tế.

### 🚀 Bài Tập Mở Rộng
1. **Tối ưu hóa Sliding Window**: Sử dụng Deque để giảm độ phức tạp từ O(nk) xuống O(n)
2. **Xử lý văn bản nâng cao**: Loại bỏ stop words, stemming, lemmatization
3. **Cải thiện Levenshtein**: Thêm trọng số cho các phép biến đổi khác nhau
4. **Ứng dụng thực tế**: Xây dựng spell checker đơn giản

Chúc bạn học tốt và thành công trong hành trình chinh phục Python! 🐍✨ 