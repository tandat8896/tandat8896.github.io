---
title: "Thuật Toán Di Truyền: Hành Trình Từ Ngẫu Nhiên Đến Tiến Hóa Thông Minh"
description: "Khám phá cách thuật toán di truyền mô phỏng quá trình tiến hóa tự nhiên để giải quyết các bài toán phức tạp mà con người không thể tính toán"
pubDatetime: 2025-01-28T14:00:00Z
heroImage: "/assets/images/genetic-algorithms-hero.jpg"
tags: ["genetic-algorithms", "optimization", "evolutionary-computing", "storytelling", "machine-learning"]
---

# Thuật Toán Di Truyền: Hành Trình Từ Ngẫu Nhiên Đến Tiến Hóa Thông Minh

## **Chương 1: Câu chuyện bắt đầu - Nhiệm vụ bí mật**

### **1.1. Bài toán "Đồng hồ bí mật"**

Hãy tưởng tượng bạn là một điệp viên được giao nhiệm vụ tìm vị trí tối ưu trong một khu vực rộng lớn. Bạn được trang bị một **"đồng hồ đặc biệt"**:

- Khi bạn đứng ở một vị trí bất kỳ và nhấn nút, đồng hồ sẽ hiển thị một con số
- Con số này càng lớn càng tốt
- **Nhưng bạn không biết:**
  - Đồng hồ hoạt động như thế nào?
  - Yếu tố nào quyết định con số?
  - Công thức tính toán là gì?
  - Làm sao để đạt được số cao nhất?

**Đây chính là bài toán Blackbox Optimization!**

### **1.2. Câu chuyện "Khu rừng tiến hóa"**

Hãy tưởng tượng một **khu rừng** có rất nhiều cá thể sinh vật:

- Mỗi cá thể có **gen riêng** (x, y) - tọa độ trong không gian
- Cá thể nào có **fitness cao** sẽ sống sót và sinh sản
- Cá thể nào **yếu** sẽ bị đào thải
- **Mục tiêu:** Tìm cá thể có gen tốt nhất để đạt được **cực tiểu toàn cục**

**Giống như trong tự nhiên:**
- **Ghép đôi** giúp mình mạnh hơn
- **Chống chịu** điều kiện thời tiết, survive sống sót
- **Tạo ra quần thể mới** có khả năng chống chịu tốt hơn

### **1.3. Ví dụ thực tế: Phát hiện mèo trong ảnh**

**Bài toán:** Tìm bounding box chứa con mèo trong ảnh

**Cách tiếp cận:**
1. **Tạo 1000 bounding box** với kích cỡ khác nhau
2. **Mỗi bounding box = 1 cá thể** (candidate solution)
3. **Chromosome:** [x, y, w, h] - tọa độ và kích thước
4. **Fitness function:** Đưa ảnh crop vào model nhận dạng mèo
5. **Kết quả:** Xác suất model nhận dạng là mèo

**Quá trình tiến hóa:**
- Bounding box nào **chứa mèo** → fitness cao → sống sót
- Bounding box nào **không chứa mèo** → fitness thấp → bị đào thải
- **Lai tạo:** 2 bounding box tốt → tạo ra bounding box con tốt hơn
- **Đột biến:** Thay đổi kích thước, vị trí để khám phá vùng mới

**Kết quả:** Sau nhiều thế hệ → tìm được bounding box chính xác nhất!

### **1.2. Tại sao gọi là "Blackbox"?**

Giống như một công ty lớn giao dự án cho nhiều công ty nhỏ:
- Công ty lớn: Giao từng phần nhỏ của dự án
- Công ty nhỏ: Chỉ biết **input → output** của phần mình làm
- Không ai biết toàn bộ bức tranh lớn

**Trong Thuật toán Di truyền:**
- Chúng ta có một hàm `f(x, y)` trả về giá trị
- Chúng ta chỉ biết: Cho input → nhận output
- Không biết công thức bên trong
- Mục tiêu: Tìm `x, y` để `f(x, y)` tối ưu

---

## **Chương 2: Tại sao không dùng Gradient Descent?**

### **2.1. Hạn chế của Gradient Descent**

Gradient Descent giống như một người đi bộ trên núi:
- ✅ **Hoạt động tốt:** Khi núi chỉ có 1 đỉnh (unimodal)
- ✅ **Cần đạo hàm:** Biết hướng dốc nhất để đi
- ✅ **Hàm liên tục:** Không có vách đứng

**Nhưng thực tế:**
- ❌ Nhiều hàm không có đạo hàm
- ❌ Hàm đa cực trị (multimodal) - nhiều đỉnh núi
- ❌ Hàm rời rạc, không liên tục
- ❌ Dễ bị kẹt ở cực trị cục bộ

### **2.2. Ví dụ thực tế**

**Bài toán phân ca công nhân:**
- 100 công nhân
- Mỗi người có thể làm ca ngày HOẶC ca đêm HOẶC nghỉ
- Số tổ hợp: 3^100 ≈ 5 × 10^47 khả năng!
- **Không thể tính tay được!**
- **Không có đạo hàm!**

**Gradient Descent không giúp được gì ở đây!**

---

## **Chương 3: Bắt đầu với Ngẫu nhiên - Naïve Random Search**

### **3.1. Bài toán OneMax - Đơn giản nhưng mạnh mẽ**

Trước khi đi vào phức tạp, hãy bắt đầu với bài toán đơn giản:

**Đề bài:**
- Cho một vector có 10 vị trí
- Mỗi vị trí chỉ nhận giá trị 0 hoặc 1
- Mục tiêu: Tìm vector có **nhiều số 1 nhất**

**Ví dụ:**
```
Vector: [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
Số lượng số 1: 7
```

**Câu hỏi:** Làm sao tìm được vector `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`?

### **3.2. Hiểu về Tính ngẫu nhiên (Randomness)**

Trước khi code, hãy hiểu về tính ngẫu nhiên:

**Ví dụ 1: Điểm thi Địa lý**
- Chọn ngẫu nhiên 1 thí sinh tốt nghiệp 2021
- Xem điểm Địa lý của thí sinh đó
- Lặp lại 10,000 lần
- Vẽ histogram → Hình dạng giống Gaussian!

**Ví dụ 2: Màu pixel trong ảnh**
- Chọn ngẫu nhiên 1 pixel trong ảnh
- Lấy giá trị màu (0-255) của 3 kênh RGB
- Tính mean của 3 giá trị
- Lặp lại 10,000 lần
- Vẽ histogram → Hình dạng giống Gaussian! (Central Limit Theorem)

**Kết luận:**
- Mặc dù từng cá thể không kiểm soát được
- Nhưng với hệ thống lớn → tuân theo quy luật
- **Chúng ta sẽ quản lý tính ngẫu nhiên này để đạt mục đích!**

### **3.3. Thử nghiệm 1: Thuê 1 người**

Hãy tưởng tượng bạn thuê 1 người để tìm vị trí tốt nhất:

**Câu chuyện:**
- Bạn thả dù người đó từ máy bay xuống vị trí ngẫu nhiên
- Người đó lật đồng hồ ra xem số
- Báo cáo lại cho bạn
- **Nhiệm vụ bí mật:** Chỉ thuê 1 lần rồi ngừng hợp tác (bảo mật cao!)

**Code Python:**

```python
import random

# Bước 1: Thiết lập bài toán
problem_size = 10  # Vector có 10 vị trí

# Bước 2: Tạo vector ngẫu nhiên
# Mỗi vị trí nhận giá trị 0 hoặc 1 với xác suất 50-50
vector = [random.randint(0, 1) for _ in range(problem_size)]

print("Vector ngẫu nhiên:", vector)
```

**Giải thích từng dòng:**

`problem_size = 10`: Kích thước bài toán (10 vị trí)
`random.randint(0, 1)`: Sinh số ngẫu nhiên 0 hoặc 1
   - Xác suất sinh 0: 50%
   - Xác suất sinh 1: 50%
   - **Tại sao quan trọng?** Vì phải công bằng, không thiên vị
`for _ in range(problem_size)`: Lặp 10 lần để tạo 10 vị trí
`[...]`: List comprehension - cách viết ngắn gọn

**Kết quả có thể:**
```
Vector ngẫu nhiên: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
```

### **3.4. Hàm "Đồng hồ bí mật" - get_signal()**

Bây giờ tạo hàm đồng hồ:

```python
def get_signal(vector):
    """
    Hàm 'đồng hồ bí mật' - Blackbox function
    
    Input: vector có 10 vị trí (mỗi vị trí = 0 hoặc 1)
    Output: Số lượng số 1 trong vector
    
    Lưu ý: Đừng dựa vào bản thân hàm này!
    Trong thực tế, bạn không biết nó hoạt động thế nào.
    """
    return sum(vector)

# Test
fitness = get_signal(vector)
print(f"Fitness (số lượng số 1): {fitness}")
```

**Giải thích:**

1. `def get_signal(vector)`: Định nghĩa hàm
2. `sum(vector)`: Tính tổng các phần tử
   - `[1, 0, 1, 1, 0]` → `1 + 0 + 1 + 1 + 0 = 3`
3. **Tại sao gọi là "secret"?**
   - Trong thực tế, bạn không biết công thức
   - Chỉ biết: Cho input → nhận output
   - Giống như đồng hồ bí mật!

**Kết quả:**
```
Fitness (số lượng số 1): 5
```

### **3.5. Vấn đề với "Thuê 1 người"**

**Thử nghiệm:**

```python
# Chạy 5 lần
for i in range(5):
    vector = [random.randint(0, 1) for _ in range(10)]
    fitness = get_signal(vector)
    print(f"Lần {i+1}: {vector} → Fitness: {fitness}")
```

**Kết quả:**
```
Lần 1: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1] → Fitness: 5
Lần 2: [1, 0, 1, 0, 0, 1, 1, 0, 1, 0] → Fitness: 5
Lần 3: [0, 0, 1, 1, 0, 1, 0, 1, 0, 1] → Fitness: 5
Lần 4: [1, 1, 0, 0, 1, 0, 1, 0, 1, 1] → Fitness: 6
Lần 5: [0, 1, 1, 1, 0, 0, 1, 1, 0, 1] → Fitness: 6
```

**Vấn đề:**
- ❌ Cơ hội thấp (chỉ 1 người)
- ❌ Không học hỏi từ lần trước
- ❌ Mỗi lần thử hoàn toàn độc lập

---

## **Chương 4: Cải tiến - Thuê nhiều người**

### **4.1. Ý tưởng: Tăng số lượng worker**

**Câu hỏi:** Nếu muốn tăng hiệu suất lên 4 lần thì sao?

**Trả lời:** Thuê 4 người!

**Lý do:**
- 1 người → khảo sát 1 điểm
- 4 người → khảo sát 4 điểm
- Cơ hội tìm được vị trí tốt tăng lên!

**Nhưng:**
- Trong bài toán tổng quát, có **vô cùng điểm** để khảo sát
- 4 điểm / ∞ ≈ 0
- Cần **vô cùng worker** để đảm bảo tìm được tối ưu!

**Giải pháp:** Giả lập sự vô cùng bằng cách thông minh!

### **4.2. Code: Thuê 8 người**

```python
def create_member(problem_size):
    """
    Tạo 1 cá thể (member) - Gửi 1 người tới vị trí ngẫu nhiên
    
    Input: problem_size (kích thước bài toán)
    Output: vector ngẫu nhiên (lời giải)
    """
    return [random.randint(0, 1) for _ in range(problem_size)]

# Thiết lập
problem_size = 10
num_of_members = 8  # Số lượng worker (kích thước quần thể)

# Tạo quần thể
population = [create_member(problem_size) for _ in range(num_of_members)]

# In ra
for i, member in enumerate(population):
    fitness = get_signal(member)
    print(f"Người {i+1}: {member} → Fitness: {fitness}")
```

**Giải thích chi tiết:**

   
`num_of_members = 8`:
   - Thuê 8 người
   - Trong thuật ngữ: **Population size = 8**
   
`population = [create_member(...) for _ in range(8)]`:
   - Tạo 8 người
   - Mỗi người ở vị trí ngẫu nhiên
   
`enumerate(population)`:
   - Lặp qua từng người và lấy index

**Kết quả:**
```
=== QUẦN THỂ BAN ĐẦU ===
Người 1: [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] → Fitness: 5
Người 2: [0, 1, 1, 1, 0, 0, 1, 1, 0, 1] → Fitness: 6
Người 3: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0] → Fitness: 6
Người 4: [0, 0, 1, 1, 1, 0, 1, 1, 0, 1] → Fitness: 6
Người 5: [1, 0, 0, 1, 1, 0, 1, 0, 1, 1] → Fitness: 6
Người 6: [0, 1, 1, 0, 1, 1, 0, 1, 0, 1] → Fitness: 6
Người 7: [1, 1, 1, 0, 0, 1, 0, 1, 1, 0] → Fitness: 6
Người 8: [0, 0, 0, 1, 1, 1, 1, 0, 1, 1] → Fitness: 6
```

### **4.3. Vấn đề vẫn còn**

**Quan sát:**
- Đã khảo sát được 8 vị trí
- Fitness tốt nhất: 6
- **Nhưng:** Mỗi lần chạy lại → tạo 8 người MỚI hoàn toàn
- **Không có tính kế thừa!**
- **Không học hỏi từ lần trước!**

**Câu hỏi:** Làm sao để cải tiến?

---

## **Chương 5: Đột phá - Học hỏi và Tiến hóa**

### **5.1. Ý tưởng: 2 người giỏi học chung**

**Câu chuyện:**
- Có 2 học sinh giỏi: A và B
- A giỏi Toán, B giỏi Văn
- Nếu 2 người học chung:
  - A học được Văn từ B
  - B học được Toán từ A
  - **Cả 2 đều tiến bộ!**

**Áp dụng vào thuật toán:**
- Có 2 cá thể tốt: Vector 1 và Vector 2
- Vector 1: `[1, 1, 0, 1, 0, 1, 1, 0, 1, 0]` (fitness: 6)
- Vector 2: `[0, 1, 1, 1, 0, 0, 1, 1, 0, 1]` (fitness: 6)
- **Trao đổi thông tin** → Tạo ra con cái tốt hơn!

### **5.2. Crossover (Lai tạo) - Trao đổi thông tin**

**Ý tưởng:** Giống như 2 người giỏi học chung!

**Câu chuyện:**
- **Bạn A:** Giỏi Toán, yếu Văn
- **Bạn B:** Giỏi Văn, yếu Toán  
- **Học chung:** A học Văn từ B, B học Toán từ A
- **Kết quả:** Cả 2 đều tiến bộ!

**Ví dụ cụ thể:**

```
Cha:  [1, 1, 0, | 1, 0, 1, 1, 0, 1, 0]  ← Giỏi phần đầu
Mẹ:   [0, 1, 1, | 1, 0, 0, 1, 1, 0, 1]  ← Giỏi phần sau
       ↑↑↑↑↑↑↑↑↑   ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
       Giữ nguyên   Trao đổi

Con 1: [1, 1, 0, | 1, 0, 0, 1, 1, 0, 1]  ← Học được phần sau từ mẹ
Con 2: [0, 1, 1, | 1, 0, 1, 1, 0, 1, 0]  ← Học được phần đầu từ cha
```

**Các loại Crossover:**

1. **One Point Crossover:** 1 điểm cắt
2. **Two Point Crossover:** 2 điểm cắt  
3. **Uniform Crossover:** Trao đổi từng gen riêng lẻ

**Ví dụ Uniform Crossover:**
```
Cha:  [1, 0, 1, 0, 1]
Mẹ:   [0, 1, 0, 1, 0]
Mask: [1, 0, 1, 0, 1]  ← Random 0/1 cho từng vị trí

Con 1: [1, 1, 1, 1, 1]  ← Lấy từ cha khi mask=1, từ mẹ khi mask=0
Con 2: [0, 0, 0, 0, 0]  ← Ngược lại
```

**Code Python:**

```python
def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Lai tạo 2 cá thể cha mẹ
    
    Input:
        - parent1: Vector cha
        - parent2: Vector mẹ
        - crossover_rate: Xác suất lai tạo (0.8 = 80%)
    
    Output:
        - child1, child2: 2 con cái
    """
    # Bước 1: Quyết định có lai tạo không?
    if random.random() < crossover_rate:
        # Bước 2: Chọn điểm cắt ngẫu nhiên
        crossover_point = random.randint(1, len(parent1) - 1)
        
        print(f"  Điểm cắt: vị trí {crossover_point}")
        print(f"  Cha: {parent1}")
        print(f"  Mẹ: {parent2}")
        
        # Bước 3: Tạo con cái
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        print(f"  Con 1: {child1}")
        print(f"  Con 2: {child2}")
        
        return child1, child2
    else:
        # Không lai tạo - giữ nguyên
        print("  Không lai tạo - giữ nguyên")
        return parent1, parent2

# Test
parent1 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
parent2 = [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]

child1, child2 = crossover(parent1, parent2)
```

**Giải thích chi tiết:**

1. `random.random() < crossover_rate`:
   - `random.random()`: Sinh số ngẫu nhiên từ 0 đến 1
   - Nếu < 0.8 → Lai tạo (80% cơ hội)
   - Nếu >= 0.8 → Không lai tạo (20% cơ hội)
   - **Tại sao không 100%?** Để giữ đa dạng!

2. `crossover_point = random.randint(1, len(parent1) - 1)`:
   - Chọn điểm cắt từ 1 đến 9 (không phải 0 hoặc 10)
   - **Tại sao?** Để đảm bảo cả 2 phần đều có ít nhất 1 phần tử

3. `parent1[:crossover_point]`:
   - Lấy phần đầu của cha (từ 0 đến điểm cắt)
   
4. `parent2[crossover_point:]`:
   - Lấy phần sau của mẹ (từ điểm cắt đến cuối)

**Kết quả:**
```
  Điểm cắt: vị trí 3
  Cha: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  Mẹ: [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
  Con 1: [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
  Con 2: [0, 1, 1, 1, 0, 1, 1, 0, 1, 0]
```

**Phân tích:**
- Con 1: `[1, 1, 0, 1, 0, 0, 1, 1, 0, 1]` → Fitness: 6
- Con 2: `[0, 1, 1, 1, 0, 1, 1, 0, 1, 0]` → Fitness: 6
- **Chưa tốt hơn cha mẹ?** Đừng lo, sẽ tốt hơn sau nhiều thế hệ!

### **5.3. Edge Case 1: Điểm cắt ở đầu hoặc cuối**

**Câu hỏi:** Nếu điểm cắt = 0 hoặc 10 thì sao?

**Trả lời:**
- Điểm cắt = 0: Con 1 = Mẹ, Con 2 = Cha (không trao đổi gì!)
- Điểm cắt = 10: Con 1 = Cha, Con 2 = Mẹ (không trao đổi gì!)

**Giải pháp:** `random.randint(1, len(parent1) - 1)` → Chỉ chọn từ 1 đến 9

### **5.4. Mutation (Đột biến) - Khám phá mới**

**Câu chuyện:** Giống như **"kịch bản tận thế"**!

**Vấn đề:**
- Trong quần thể hiện tại, tất cả đều có vị trí 5 = 0
- Nếu chỉ lai tạo → Con cái cũng có vị trí 5 = 0
- **Không bao giờ có vị trí 5 = 1!**
- **Cần cơ chế "đột biến" để khám phá!**

**Ví dụ cụ thể:**
```
Trước: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
                ↑
            Đột biến (0 → 1)
Sau:   [1, 1, 0, 1, 1, 1, 1, 0, 1, 0]
```

**Tại sao cần Mutation?**

**Kịch bản tận thế:** Nếu random xui, tất cả vị trí đều bị đột biến thành 0?
- **Xác suất:** 0.1^10 ≈ 10^-10 (cực kỳ nhỏ!)
- **Giải pháp:** Selection đã chọn cá thể tốt ("Môn đăng hộ đối")
- **Elitism:** Giữ lại cá thể tốt nhất

**Các loại Mutation:**

1. **Bit Flip:** Đảo bit (0→1, 1→0)
2. **Gaussian:** Thêm noise ngẫu nhiên
3. **Uniform:** Thay đổi giá trị trong khoảng

**Ví dụ Gaussian Mutation:**
```
Trước: [1.2, 3.4, 5.6]
Noise: [0.1, -0.2, 0.3]  ← Random Gaussian
Sau:   [1.3, 3.2, 5.9]
```

**Code Python:**

```python
def mutation(individual, mutation_rate=0.1):
    """
    Đột biến cá thể
    
    Input:
        - individual: Vector cần đột biến
        - mutation_rate: Xác suất đột biến mỗi vị trí (0.1 = 10%)
    
    Output:
        - mutated: Vector sau đột biến
    """
    mutated = individual.copy()  # Copy để không thay đổi bản gốc
    changes = []
    
    # Duyệt qua từng vị trí
    for i in range(len(mutated)):
        # Quyết định có đột biến không?
        if random.random() < mutation_rate:
            # Đảo bit: 0 → 1, 1 → 0
            mutated[i] = 1 - mutated[i]
            changes.append(i)
    
    # In ra thông tin
    if changes:
        print(f"  Đột biến tại vị trí: {changes}")
        print(f"  Trước: {individual}")
        print(f"  Sau:  {mutated}")
    else:
        print(f"  Không có đột biến: {individual}")
    
    return mutated

# Test
individual = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
mutated = mutation(individual, mutation_rate=0.1)
```

**Giải thích chi tiết:**

1. `individual.copy()`:
   - Copy để không thay đổi bản gốc
   - **Tại sao?** Để so sánh trước/sau

2. `for i in range(len(mutated))`:
   - Duyệt qua từng vị trí
   - Mỗi vị trí có cơ hội đột biến

3. `random.random() < mutation_rate`:
   - Mỗi vị trí có 10% cơ hội đột biến
   - **Tại sao 10%?** Cân bằng giữa khám phá và khai thác

4. `mutated[i] = 1 - mutated[i]`:
   - Đảo bit: 0 → 1, 1 → 0
   - **Công thức thông minh!**

**Kết quả có thể:**
```
  Đột biến tại vị trí: [3, 7]
  Trước: [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  Sau:   [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
```

### **5.5. Edge Case 2: Mutation rate quá cao hoặc quá thấp**

**Mutation rate = 0 (0%):**
- Không có đột biến
- Chỉ lai tạo
- **Vấn đề:** Không khám phá được vùng mới
- **Kết quả:** Bị kẹt ở cực trị cục bộ

**Mutation rate = 1 (100%):**
- Mọi vị trí đều đột biến
- **Vấn đề:** Quá ngẫu nhiên, mất thông tin tốt
- **Kết quả:** Giống Random Search

**Mutation rate = 0.1 (10%):**
- Cân bằng giữa khám phá và khai thác
- **Thường dùng trong thực tế**

---

## **Chương 6: Selection (Chọn lọc) - "Môn đăng hộ đối"**

### **6.1. Ý tưởng: Chỉ người giỏi mới được chọn**

**Câu chuyện:**
- Có 8 người trong quần thể
- Fitness khác nhau: 4, 5, 6, 6, 7, 5, 6, 5
- **Câu hỏi:** Chọn ai để lai tạo?

**Trả lời:**
- ❌ **Không chọn:** Người có fitness thấp (4, 5)
- ✅ **Chọn:** Người có fitness cao (6, 7)
- **Lý do:** "Môn đăng hộ đối" - Người giỏi mới sinh con giỏi!

### **6.2. Roulette Wheel Selection (Chọn lọc bánh xe roulette)**

**Ý tưởng:** Giống như quay bánh xe roulette!

**Ví dụ cụ thể:**
```
Quần thể:
Người 1: fitness = 10 → 10/30 = 33.3% diện tích
Người 2: fitness = 9  → 9/30 = 30.0% diện tích  
Người 3: fitness = 7  → 7/30 = 23.3% diện tích
Người 4: fitness = 5  → 5/30 = 16.7% diện tích
Người 5: fitness = 0  → 0/30 = 0.0% diện tích

Tổng fitness = 30
```

**Cách hoạt động:**
1. **Vẽ bánh xe** với diện tích tỷ lệ với fitness
2. **Quay bánh xe** ngẫu nhiên
3. **Người nào có diện tích lớn** → xác suất được chọn cao hơn
4. **Người fitness = 0** → không có diện tích → không bao giờ được chọn

**Ưu điểm:** Công bằng, người giỏi có cơ hội cao
**Nhược điểm:** Phức tạp tính toán, có thể chọn người yếu

### **6.3. Tournament Selection (Chọn lọc theo giải đấu)**

**Ý tưởng:** Giống như thi đấu thể thao!

**Ví dụ cụ thể:**

```
Quần thể:
Người 1: fitness = 5
Người 2: fitness = 6
Người 3: fitness = 6
Người 4: fitness = 6
Người 5: fitness = 7  ← Tốt nhất!
Người 6: fitness = 5
Người 7: fitness = 6
Người 8: fitness = 5

Giải đấu 1: Chọn ngẫu nhiên 3 người: [1, 5, 7]
  → Fitness: [5, 7, 6]
  → Người thắng: Người 5 (fitness = 7)

Giải đấu 2: Chọn ngẫu nhiên 3 người: [2, 4, 8]
  → Fitness: [6, 6, 5]
  → Người thắng: Người 2 hoặc 4 (fitness = 6)
```

**Ưu điểm:** Đơn giản, nhanh, đảm bảo chọn người tốt
**Nhược điểm:** Có thể chọn người yếu nếu xui

**Code Python:**

```python
def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Chọn lọc theo giải đấu
    
    Input:
        - population: Quần thể hiện tại
        - fitness_scores: Fitness của từng cá thể
        - tournament_size: Số người tham gia giải đấu
    
    Output:
        - selected: Quần thể được chọn
    """
    selected = []
    
    print("=== CHỌN LỌC - GIẢI ĐẤU ===")
    
    # Lặp lại cho đến khi đủ số lượng
    for round_num in range(len(population)):
        # Bước 1: Chọn ngẫu nhiên tournament_size người
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Bước 2: Lấy fitness của những người này
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Bước 3: Tìm người có fitness cao nhất
        max_fitness = max(tournament_fitness)
        winner_index_in_tournament = tournament_fitness.index(max_fitness)
        winner_index = tournament_indices[winner_index_in_tournament]
        
        # Bước 4: Thêm người thắng vào danh sách
        selected.append(population[winner_index])
        
        print(f"Giải đấu {round_num + 1}:")
        print(f"  Người tham gia: {[i+1 for i in tournament_indices]}")
        print(f"  Fitness: {tournament_fitness}")
        print(f"  Người thắng: Người {winner_index + 1} (fitness: {fitness_scores[winner_index]})")
    
    return selected

# Test
population = [
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],  # fitness: 5
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],  # fitness: 6
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # fitness: 6
    [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # fitness: 6
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],  # fitness: 8 ← Tốt nhất!
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],  # fitness: 6
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],  # fitness: 6
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 1],  # fitness: 6
]

fitness_scores = [get_signal(individual) for individual in population]
selected = tournament_selection(population, fitness_scores)
```

**Giải thích chi tiết:**

1. `random.sample(range(len(population)), tournament_size)`:
   - Chọn ngẫu nhiên 3 index từ 0-7
   - **Không lặp lại!** (khác với `random.choice`)
   
2. `tournament_fitness = [fitness_scores[i] for i in tournament_indices]`:
   - Lấy fitness của 3 người được chọn
   
3. `max(tournament_fitness)`:
   - Tìm fitness cao nhất trong giải đấu
   
4. `tournament_fitness.index(max_fitness)`:
   - Tìm vị trí của người thắng trong giải đấu
   
5. `winner_index = tournament_indices[winner_index_in_tournament]`:
   - Chuyển từ vị trí trong giải đấu → vị trí trong quần thể

**Kết quả:**
```
=== CHỌN LỌC - GIẢI ĐẤU ===
Giải đấu 1:
  Người tham gia: [1, 5, 7]
  Fitness: [5, 8, 6]
  Người thắng: Người 5 (fitness: 8)

Giải đấu 2:
  Người tham gia: [2, 4, 8]
  Fitness: [6, 6, 6]
  Người thắng: Người 2 (fitness: 6)
...
```

### **6.3. Edge Case 3: Tournament size**

**Tournament size = 1:**
- Chỉ chọn 1 người
- **Vấn đề:** Chọn ngẫu nhiên, không có áp lực chọn lọc
- **Kết quả:** Giống Random Search

**Tournament size = len(population):**
- Chọn tất cả
- **Vấn đề:** Luôn chọn người tốt nhất
- **Kết quả:** Mất đa dạng, hội tụ quá nhanh

**Tournament size = 3:**
- Cân bằng giữa áp lực chọn lọc và đa dạng
- **Thường dùng trong thực tế**

---

## **Chương 7: Thuật toán hoàn chỉnh**

### **7.1. Tổng hợp tất cả các bước**

```python
import random

def create_vector(problem_size):
    """Tạo vector ngẫu nhiên cho OneMax problem"""
    return [random.randint(0, 1) for _ in range(problem_size)]

def compute_fitness(vector):
    """Tính fitness - số lượng số 1 (càng nhiều càng tốt)"""
    return sum(vector)

def exchange(vector1, vector2, problem_size):
    """Lai tạo 2 vector - trao đổi thông tin"""
    # Chọn điểm cắt ngẫu nhiên
    crossover_point = random.randint(1, problem_size - 1)
    
    # Tạo con cái
    child1 = vector1[:crossover_point] + vector2[crossover_point:]
    child2 = vector2[:crossover_point] + vector1[crossover_point:]
    
    return child1, child2

def select_better_vector(sorted_vectors, nums_of_members):
    """Chọn vector tốt hơn từ quần thể đã sắp xếp"""
    # Chọn ngẫu nhiên từ nửa trên (tốt hơn)
    upper_half = sorted_vectors[nums_of_members//2:]
    return random.choice(upper_half)

# === THUẬT TOÁN DI TRUYỀN HOÀN CHỈNH ===
problem_size = 10        # Kích thước cá thể (chromosome)
nums_of_members = 8     # Kích thước quần thể
n_generations = 30      # Số thế hệ

# Để vẽ biểu đồ quá trình tối ưu
fitnesses = []

# 1. Tạo quần thể ban đầu (CHỈ 1 LẦN)
print("🧬 === THUẬT TOÁN DI TRUYỀN === 🧬")
print(f"Kích thước bài toán: {problem_size}")
print(f"Kích thước quần thể: {nums_of_members}")
print(f"Số thế hệ: {n_generations}")
print("=" * 50)

print("=== KHỞI TẠO QUẦN THỂ BAN ĐẦU ===")
vectors = [create_vector(problem_size) for _ in range(nums_of_members)]

# In ra quần thể ban đầu
for i, vector in enumerate(vectors):
    fitness = compute_fitness(vector)
    print(f"Cá thể {i+1}: {vector} → Fitness: {fitness}")

# Vòng lặp thế hệ
for i in range(n_generations):
    print(f"\n🔄 THẾ HỆ {i + 1}")
    print("-" * 30)
    
    # 2. Sắp xếp vectors theo fitness (tốt nhất ở cuối)
    sorted_vectors = sorted(vectors, key=compute_fitness)
    
    # Debug - in fitness tốt nhất
    best_fitness = compute_fitness(sorted_vectors[nums_of_members-1])
    fitnesses.append(best_fitness)
    print(f"Fitness tốt nhất: {best_fitness}")
    print(f"Vector tốt nhất: {sorted_vectors[nums_of_members-1]}")
    
    # 3. Tạo quần thể mới bằng vòng while
    new_vectors = []
    print("\n=== TẠO QUẦN THỂ MỚI ===")
    
    while len(new_vectors) < nums_of_members:
        print(f"Đang tạo cá thể {len(new_vectors) + 1}/{nums_of_members}")
        
        # Bước 1: Chọn lọc - chọn 2 vector tốt
        vector1 = select_better_vector(sorted_vectors, nums_of_members)
        vector2 = select_better_vector(sorted_vectors, nums_of_members)
        
        print(f"  Chọn vector1: {vector1} (fitness: {compute_fitness(vector1)})")
        print(f"  Chọn vector2: {vector2} (fitness: {compute_fitness(vector2)})")
        
        # Bước 2: Lai tạo - trao đổi thông tin
        child1, child2 = exchange(vector1, vector2, problem_size)
        
        print(f"  Con 1: {child1} (fitness: {compute_fitness(child1)})")
        print(f"  Con 2: {child2} (fitness: {compute_fitness(child2)})")
        
        # Bước 3: Lưu 2 con cái
        new_vectors.append(child1)
        new_vectors.append(child2)
        
        # Kiểm tra nếu đã đủ
        if len(new_vectors) >= nums_of_members:
            # Cắt bớt nếu thừa
            new_vectors = new_vectors[:nums_of_members]
            break
    
    # 4. Cập nhật quần thể
    vectors = new_vectors
    
    print(f"\nQuần thể mới có {len(vectors)} cá thể")
    
    # Kiểm tra cải thiện
    new_best = max([compute_fitness(v) for v in vectors])
    if new_best > best_fitness:
        print(f "Cải thiện! {best_fitness} → {new_best}")
    elif new_best == best_fitness:
        print(f"Giữ nguyên: {best_fitness}")
    else:
        print(f" Giảm: {best_fitness} → {new_best}")

# Kết quả cuối cùng
final_fitness = [compute_fitness(v) for v in vectors]
best_index = final_fitness.index(max(final_fitness))

```

### **7.2. Phân tích kết quả**

**Lịch sử fitness có thể:**
```
Thế hệ 1: 6
Thế hệ 2: 6
Thế hệ 3: 7
Thế hệ 4: 7
Thế hệ 5: 8
Thế hệ 6: 8
Thế hệ 7: 8
Thế hệ 8: 9
Thế hệ 9: 9
Thế hệ 10: 9
Thế hệ 11: 9
Thế hệ 12: 10  ← HOÀN HẢO!
```

**Quan sát:**
- ✅ Fitness tăng dần qua các thế hệ
- ✅ Có lúc giữ nguyên (thế hệ 1-2, 3-4, ...)
- ✅ Cuối cùng đạt được 10 (tối ưu!)

---

## **Chương 8: Giải quyết vấn đề thực tế**

### **8.1. Vấn đề: Tại sao Random Search không đủ?**

**Câu chuyện thực tế:**
Bạn đang tối ưu hóa lịch làm việc cho 100 công nhân:

**Random Search:**
- Mỗi lần tạo lịch mới hoàn toàn ngẫu nhiên
- Không học hỏi từ lịch trước
- **Vấn đề:** Có thể tạo ra lịch tệ hơn lịch hiện tại!

**Genetic Algorithm:**
- Học hỏi từ lịch tốt trước đó
- Trao đổi ca giữa các lịch tốt
- **Kết quả:** Lịch mới thường tốt hơn lịch cũ

### **8.2. Vấn đề: Mất cá thể tốt nhất**

**Câu chuyện:** 
Thế hệ 5 có lịch tốt nhất (fitness = 9), nhưng thế hệ 6 chỉ có lịch tệ hơn (fitness = 8)

**Nguyên nhân:**
- Selection có thể chọn nhầm
- Crossover có thể tạo ra lịch xấu
- Mutation có thể phá hỏng lịch tốt

**Giải pháp: Elitism (Chủ nghĩa tinh hoa)**
- Giữ lại 1-2 lịch tốt nhất từ thế hệ trước
- Đảm bảo không bao giờ mất thông tin tốt

**Code thực tế:**

```python
def genetic_algorithm_with_elitism(problem_size=10, pop_size=8, generations=20, 
                                   crossover_rate=0.8, mutation_rate=0.1, 
                                   tournament_size=3, elite_size=2):
    """
    Thuật toán di truyền với Elitism - Giải quyết vấn đề mất cá thể tốt
    """
    # ... (giống như trước)
    
    for generation in range(generations):
        # ... (các bước như trước)
        
        # Bước mới: Giữ lại elite_size lịch tốt nhất
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], 
                              reverse=True)[:elite_size]
        elite = [population[i] for i in elite_indices]
        
        # ... (lai tạo, đột biến)
        
        # Thay thế lịch xấu nhất bằng elite
        new_fitness = [get_signal(individual) for individual in population]
        worst_indices = sorted(range(len(new_fitness)), 
                              key=lambda i: new_fitness[i])[:elite_size]
        
        for i, elite_individual in enumerate(elite):
            population[worst_indices[i]] = elite_individual
    
    return population, best_fitness_history
```

### **8.3. Vấn đề: "Kịch bản tận thế"**

**Câu hỏi thực tế:**
- Nếu random xui, tất cả vị trí đều bị đột biến thành 0?
- Ví dụ: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` → `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

**Trả lời thực tế:**
- ❌ **Khó xảy ra** vì:
  1. Xác suất đột biến chỉ 10%
  2. Xác suất tất cả 10 vị trí đều đột biến: 0.1^10 ≈ 10^-10 (cực kỳ nhỏ!)
  3. Selection đã chọn cá thể tốt ("Môn đăng hộ đối")
  4. Elitism giữ lại cá thể tốt nhất

### **8.4. So sánh thực tế: Random Search vs GA**

| Vấn đề | Random Search | Genetic Algorithm |
|--------|---------------|-------------------|
| **Học hỏi** | ❌ Không | ✅ Có (từ thế hệ trước) |
| **Trao đổi thông tin** | ❌ Không | ✅ Crossover |
| **Khám phá** | ❌ Chỉ random | ✅ Mutation + Crossover |
| **Hiệu quả** | Thấp | Cao hơn nhiều |
| **Hội tụ** | Chậm | Nhanh hơn |
| **Ứng dụng** | Chỉ bài toán đơn giản | Mọi bài toán phức tạp |

**Ví dụ cụ thể:**
- **Random Search:** Như mỗi lần đánh lại từ đầu
- **GA:** Như học hỏi từ kinh nghiệm trước

---

## **Chương 9: Ứng dụng thực tế**

### **9.1. Bài toán Sphere Function**

**Mô tả:**
- Hàm: $$f(x) = \sum_{i=1}^{n} x_i^2$$
- Mục tiêu: Tìm `x` để `f(x)` nhỏ nhất
- Tối ưu: `x = [0, 0, 0, ..., 0]` → `f(x) = 0`

**Code:**

```python
# === THUẬT TOÁN DI TRUYỀN CHO SPHERE FUNCTION ===
def create_vector_sphere(problem_size, lower_bound=-50, upper_bound=50):
    """Tạo vector ngẫu nhiên cho Sphere problem"""
    return [random.randint(lower_bound, upper_bound) for _ in range(problem_size)]

def compute_fitness_sphere(vector):
    """Tính fitness cho Sphere - càng nhỏ càng tốt"""
    return sum([value ** 2 for value in vector])

def exchange_sphere(vector1, vector2, problem_size):
    """Lai tạo 2 vector cho Sphere"""
    # Chọn điểm cắt ngẫu nhiên
    crossover_point = random.randint(1, problem_size - 1)
    
    # Tạo con cái
    child1 = vector1[:crossover_point] + vector2[crossover_point:]
    child2 = vector2[:crossover_point] + vector1[crossover_point:]
    
    return child1, child2

def select_better_vector_sphere(sorted_vectors, nums_of_members):
    """Chọn vector tốt hơn cho Sphere (fitness nhỏ hơn = tốt hơn)"""
    # Chọn ngẫu nhiên từ nửa dưới (tốt hơn cho Sphere)
    lower_half = sorted_vectors[:nums_of_members//2]
    return random.choice(lower_half)

# === THUẬT TOÁN DI TRUYỀN SPHERE HOÀN CHỈNH ===
problem_size = 20         # Kích thước cá thể
nums_of_members = 40      # Kích thước quần thể  
n_generations = 30        # Số thế hệ

# Để vẽ biểu đồ quá trình tối ưu
fitnesses = []

# 1. Tạo quần thể ban đầu (CHỈ 1 LẦN)
print("🧬 === THUẬT TOÁN DI TRUYỀN - SPHERE === 🧬")
print(f"Kích thước bài toán: {problem_size}")
print(f"Kích thước quần thể: {nums_of_members}")
print(f"Số thế hệ: {n_generations}")
print("=" * 50)

print("=== KHỞI TẠO QUẦN THỂ BAN ĐẦU ===")
vectors = [create_vector_sphere(problem_size) for _ in range(nums_of_members)]

# In ra quần thể ban đầu
for i, vector in enumerate(vectors):
    fitness = compute_fitness_sphere(vector)
    print(f"Cá thể {i+1}: {vector} → Fitness: {fitness}")

# Vòng lặp thế hệ
for i in range(n_generations):
    print(f"\n🔄 THẾ HỆ {i + 1}")
    print("-" * 30)
    
    # 2. Sắp xếp vectors theo fitness (nhỏ nhất ở đầu cho Sphere)
    sorted_vectors = sorted(vectors, key=compute_fitness_sphere)
    
    # Debug - in fitness tốt nhất
    best_fitness = compute_fitness_sphere(sorted_vectors[0])  # Nhỏ nhất = tốt nhất
    fitnesses.append(best_fitness)
    print(f"Fitness tốt nhất: {best_fitness}")
    print(f"Vector tốt nhất: {sorted_vectors[0]}")
    
    # 3. Tạo quần thể mới bằng vòng while
    new_vectors = []
    print("\n=== TẠO QUẦN THỂ MỚI ===")
    
    while len(new_vectors) < nums_of_members:
        print(f"Đang tạo cá thể {len(new_vectors) + 1}/{nums_of_members}")
        
        # Bước 1: Chọn lọc - chọn 2 vector tốt
        vector1 = select_better_vector_sphere(sorted_vectors, nums_of_members)
        vector2 = select_better_vector_sphere(sorted_vectors, nums_of_members)
        
        print(f"  Chọn vector1: {vector1} (fitness: {compute_fitness_sphere(vector1)})")
        print(f"  Chọn vector2: {vector2} (fitness: {compute_fitness_sphere(vector2)})")
        
        # Bước 2: Lai tạo - trao đổi thông tin
        child1, child2 = exchange_sphere(vector1, vector2, problem_size)
        
        print(f"  Con 1: {child1} (fitness: {compute_fitness_sphere(child1)})")
        print(f"  Con 2: {child2} (fitness: {compute_fitness_sphere(child2)})")
        
        # Bước 3: Lưu 2 con cái
        new_vectors.append(child1)
        new_vectors.append(child2)
        
        # Kiểm tra nếu đã đủ
        if len(new_vectors) >= nums_of_members:
            # Cắt bớt nếu thừa
            new_vectors = new_vectors[:nums_of_members]
            break
    
    # 4. Cập nhật quần thể
    vectors = new_vectors
    
    print(f"\nQuần thể mới có {len(vectors)} cá thể")
    
    # Kiểm tra cải thiện
    new_best = min([compute_fitness_sphere(v) for v in vectors])
    if new_best < best_fitness:
        print(f" Cải thiện! {best_fitness} → {new_best}")
    elif new_best == best_fitness:
        print(f"Giữ nguyên: {best_fitness}")
    else:
        print(f" Tăng: {best_fitness} → {new_best}")

# Kết quả cuối cùng
print(f"\n KẾT QUẢ CUỐI CÙNG:")
final_fitness = [compute_fitness_sphere(v) for v in vectors]
best_index = final_fitness.index(min(final_fitness))
print(f"Vector tốt nhất: {vectors[best_index]}")
print(f"Fitness cuối cùng: {min(final_fitness)}")
print(f"Lịch sử fitness: {fitnesses}")
```

### **9.2. Bài toán Hyperparameter Tuning**

**Mô tả:**
- Tìm bộ hyperparameters tốt nhất cho Neural Network
- Không gian tìm kiếm: hàng triệu tổ hợp
- Không có đạo hàm

**Ví dụ thực tế:**
```
Hyperparameters cần tối ưu:
- Learning rate: 0.0001 → 0.1
- Batch size: 16, 32, 64, 128
- Số layers: 2 → 10
- Số neurons: 32 → 512

Không gian tìm kiếm: 1000 × 4 × 9 × 481 = 17,316,000 tổ hợp!
```

**Cách GA giải quyết:**
1. **Tạo quần thể** các bộ hyperparameters ngẫu nhiên
2. **Train model** với từng bộ → lấy accuracy
3. **Chọn lọc** bộ có accuracy cao
4. **Lai tạo** để tạo bộ mới
5. **Đột biến** để khám phá vùng mới
6. **Lặp lại** cho đến khi tìm được bộ tối ưu

### **9.3. Bài toán Phân ca Công nhân**

**Mô tả:** Tối ưu hóa lịch làm việc cho 100 công nhân

**Ví dụ thực tế:**
```
Công nhân A: Làm ca ngày tốt hơn ca đêm
Công nhân B: Làm ca đêm tốt hơn ca ngày  
Công nhân C: Đang nghỉ phép
Công nhân D: Có kinh nghiệm cao

Mục tiêu: Tối ưu hiệu suất cả dây chuyền
```

**Cách GA giải quyết:**
1. **Chromosome:** [ca_ngày, ca_đêm, nghỉ_phép, ...] cho 100 người
2. **Fitness:** Hiệu suất tổng thể của dây chuyền
3. **Selection:** Chọn lịch có hiệu suất cao
4. **Crossover:** Trao đổi ca giữa các lịch tốt
5. **Mutation:** Thay đổi ca ngẫu nhiên để khám phá

### **9.4. Bài toán Thiết kế Mạng Neural**

**Mô tả:** Tìm kiến trúc mạng tối ưu

**Ví dụ thực tế:**
```
Cần thiết kế mạng cho bài toán phân loại ảnh:
- Số layers: 3 → 20
- Số neurons mỗi layer: 32 → 1024
- Activation function: ReLU, Sigmoid, Tanh
- Optimizer: Adam, SGD, RMSprop

Không gian tìm kiếm: 18 × 993 × 3 × 3 = 161,406 tổ hợp!
```

**Cách GA giải quyết:**
1. **Chromosome:** [layers, neurons, activation, optimizer]
2. **Fitness:** Accuracy trên validation set
3. **Selection:** Chọn kiến trúc có accuracy cao
4. **Crossover:** Lai tạo kiến trúc từ 2 mạng tốt
5. **Mutation:** Thay đổi số layers, neurons ngẫu nhiên

**Code:**

```python
def create_member_hyperparameters():
    """Tạo bộ hyperparameters ngẫu nhiên"""
    return {
        'learning_rate': random.uniform(0.0001, 0.1),
        'batch_size': random.choice([16, 32, 64, 128]),
        'num_layers': random.randint(2, 10),
        'num_neurons': random.randint(32, 512)
    }

def get_signal_hyperparameters(hyperparams):
    """Đánh giá hyperparameters - train model và trả về accuracy"""
    # Train model với hyperparams
    # Return accuracy
    pass
```

---

## **Kết luận**

Thuật toán Di truyền là một **cỗ máy tiến hóa thông minh** giúp giải quyết các bài toán phức tạp mà con người không thể tính toán bằng tay.

**Điểm mạnh:**
- ✅ Không cần đạo hàm
- ✅ Tránh cực trị cục bộ
- ✅ Song song hóa dễ dàng
- ✅ Áp dụng được nhiều bài toán

**Điểm yếu:**
- ❌ Chậm (cần nhiều đánh giá)
- ❌ Không đảm bảo tối ưu tuyệt đối
- ❌ Nhiều tham số cần điều chỉnh

**Khi nào dùng GA:**
- ✅ Hàm không có đạo hàm
- ✅ Hàm đa cực trị
- ✅ Không gian tìm kiếm rất lớn
- ✅ Cần tìm lời giải gần tối ưu

**"Đừng tính toán, hãy để nó tiến hóa!"** 🧬🚀
