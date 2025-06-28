---
title: "Tuần 3: Lập Trình Hướng Đối Tượng (OOP) - Từ Nguyên Tắc Cơ Bản đến Ứng Dụng PyTorch"
description: "Khám phá toàn diện OOP: Class, Object, Encapsulation, Abstraction, Inheritance, Polymorphism, áp dụng trong PyTorch, và implement các data structures Stack/Queue. Hành trình học tập từ lý thuyết đến thực hành."
pubDatetime: 2025-01-28T10:00:00Z
tags:
  - oop
  - python
  - pytorch
  - programming
  - education
  - vietnamese
  - week3
  - object-oriented
  - inheritance
  - polymorphism
  - encapsulation
  - abstraction
  - data-structures
draft: false
---

# Blog Quá Trình Học Tập Lập Trình Hướng Đối Tượng (OOP) - Module 1, Tuần 3

## Giới Thiệu

Chào mừng các bạn đến với blog ghi lại hành trình học tập Lập Trình Hướng Đối Tượng (Object-Oriented Programming - OOP) của tôi trong Module 1, Tuần 3. Đây là một chặng đường thú vị khi tôi được tiếp cận với những khái niệm cơ bản và nâng cao của OOP thông qua các bài tập thực hành.

## Môi Trường Học Tập

Trước khi bắt đầu, tôi đã chuẩn bị môi trường học tập với:
- **Conda Environment**: Cài đặt và cấu hình môi trường Python
- **PyTorch**: Thư viện machine learning để thực hành các khái niệm OOP
- **Jupyter Notebook**: Môi trường tương tác để viết và chạy code

---

## SECTION 1: OOP Review - Các Nguyên Tắc Cơ Bản

### Mục Tiêu Học Tập
Mục tiêu chính là hiểu sâu về **Class, Object, Encapsulation, Abstraction, Inheritance, và Polymorphism**. Ngoài ra còn học về các khái niệm PyTorch như `nn.Module`, `Sigmoid`, và các cấu trúc dữ liệu như Stack và Queue.

### 1. Class và Object

#### Khái Niệm Cơ Bản
- **Object (Đối tượng)**: Bất kỳ thực thể nào có thuộc tính (attributes) và hành vi (behaviors)
- **Class (Lớp)**: Template để tạo ra các object, định nghĩa thuộc tính và hành vi chung

#### Ví Dụ Thực Tế: Dog Class
```python
class Dog:
    def __init__(self, name, size, age, color):
        # Attributes (thuộc tính)
        self.name = name
        self.size = size
        self.age = age
        self.color = color
    
    # Behaviors (hành vi)
    def eat(self):
        if self.age < 2:
            return f"{self.name} ăn thức ăn cho chó con"
        else:
            return f"{self.name} ăn thức ăn cho chó trưởng thành"
    
    def sleep(self):
        return f"{self.name} đang ngủ"
    
    def sit(self):
        return f"{self.name} đang ngồi"
    
    def run(self):
        return f"{self.name} đang chạy"
```

**Phân tích:**
- `__init__()` là constructor để khởi tạo các thuộc tính
- Các method như `eat()`, `sleep()`, `sit()`, `run()` định nghĩa hành vi
- Method `eat()` có logic điều kiện dựa trên tuổi của chó

### 2. Encapsulation (Tính Đóng Gói)

#### Khái Niệm
Encapsulation là cơ chế ẩn thông tin và giới hạn truy cập vào trạng thái nội bộ của object.

#### Access Modifiers
- **Public**: Có thể truy cập từ bất kỳ đâu
- **Protected**: Chỉ truy cập được từ class con (sử dụng `_`)
- **Private**: Chỉ truy cập được từ trong class (sử dụng `__`)

#### Getter và Setter Methods
```python
class Person:
    def __init__(self, name, age):
        self.__name = name  # Private attribute
        self.__age = age    # Private attribute
    
    # Getter methods
    def get_name(self):
        return self.__name
    
    def get_age(self):
        return self.__age
    
    # Setter methods
    def set_name(self, name):
        self.__name = name
    
    def set_age(self, age):
        if age >= 0:  # Validation
            self.__age = age
```

### 3. Abstraction (Tính Trừu Tượng)

#### Khái Niệm
Abstraction tập trung vào dữ liệu liên quan của object, ẩn các chi tiết phức tạp và nhấn mạnh các điểm dữ liệu cần thiết.

#### Ví Dụ: Shape Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def compute_area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side
    
    def compute_area(self):
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def compute_area(self):
        return 3.14159 * self.radius ** 2
```

**Phân tích:**
- Cả `Square` và `Circle` đều có method `compute_area()` chung
- Implementation khác nhau nhưng interface giống nhau
- Ẩn chi tiết tính toán, chỉ expose kết quả

### 4. Inheritance (Tính Kế Thừa)

#### Khái Niệm
Inheritance là cơ chế mạnh mẽ để tạo class mới bằng cách tái sử dụng chi tiết của class hiện có mà không cần sửa đổi nó.

#### Thuật Ngữ
- **Base Class (Parent Class)**: Class gốc được kế thừa
- **Derived Class (Child Class)**: Class mới kế thừa từ base class

#### Ví Dụ: Employee và Manager
```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def compute_salary(self):
        return self.salary

class Manager(Employee):
    def __init__(self, name, salary, bonus):
        super().__init__(name, salary)  # Gọi constructor của class cha
        self.bonus = bonus
    
    def compute_salary(self):
        return self.salary + self.bonus  # Override method
```

#### Các Loại Inheritance

1. **Single Inheritance**: Một derived class kế thừa từ một base class
```python
class Child(Parent):
    pass
```

2. **Multilevel Inheritance**: Derived class kế thừa từ base class, mà base class đó cũng là derived class
```python
class GrandChild(Child):  # Child kế thừa từ Parent
    pass
```

3. **Hierarchical Inheritance**: Nhiều derived classes kế thừa từ một base class
```python
class Child1(Parent):
    pass

class Child2(Parent):
    pass
```

4. **Multiple Inheritance**: Một derived class kế thừa từ nhiều base classes
```python
class Child(Parent1, Parent2):
    pass
```

### 5. Polymorphism (Tính Đa Hình)

#### Khái Niệm
Polymorphism cho phép một entity (method, operator, object) đơn lẻ đại diện cho các loại khác nhau trong các tình huống khác nhau.

#### Method Overriding
```python
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"
```

#### Method Overloading
**Lưu ý**: Python không hỗ trợ method overloading như các ngôn ngữ khác, nhưng có thể sử dụng default parameters hoặc `*args`, `**kwargs`.

---

## SECTION 2: OOP trong PyTorch

### torch.nn.Module
`torch.nn.Module` là base class cho tất cả neural network modules và activation functions trong PyTorch.

#### Đặc Điểm Quan Trọng
- Classes kế thừa từ `nn.Module` thường implement method `forward()`
- PyTorch sử dụng `forward()` để thực hiện forward pass

### Bài 1: Làm Quen Với PyTorch và Hàm Sigmoid

#### Khám Phá PyTorch
```python
import torch
x = torch.Tensor([5.0, 3.0])
output = torch.sigmoid(x)
print(output)  # tensor([0.9933, 0.9526])
```

Đây là bước đầu tiên làm quen với PyTorch - một thư viện machine learning mạnh mẽ. Tôi học được cách:
- Tạo tensor từ dữ liệu số
- Sử dụng hàm sigmoid có sẵn
- Hiểu được kết quả trả về dưới dạng tensor

#### Tự Implement Hàm Sigmoid
```python
import torch
import torch.nn as nn

class MySigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1/(1 + torch.exp(-x))
```

**Công thức Sigmoid**: σ(x) = 1 / (1 + e^(-x))

Đây là lần đầu tiên tôi tạo một class trong PyTorch! Tôi học được:
- **Inheritance (Kế thừa)**: Class `MySigmoid` kế thừa từ `nn.Module`
- **Constructor**: Phương thức `__init__()` để khởi tạo
- **Method Override**: Ghi đè phương thức `forward()` để định nghĩa logic riêng

### Bài Toán: Softmax và Stable Softmax

#### Softmax Function
```python
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)
```

#### Stable Softmax Function
```python
class StableSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        c = torch.max(x)  # Tìm giá trị lớn nhất để ổn định số học
        exp_x = torch.exp(x - c)  # Trừ c để tránh overflow
        return exp_x / torch.sum(exp_x)
```

**Lưu ý**: Stable softmax sử dụng c = max(x) để cải thiện tính ổn định số học.

---

## SECTION 3: Đặc Điểm của OOP - Bài Toán Thực Tế

### Mô Tả Bài Toán
Bài toán mô hình hóa một Ward (khu vực) có tên và danh sách các Person objects. Một Person có thể là Student, Doctor, hoặc Teacher.

#### Yêu Cầu Thiết Kế:
- **Person**: có name (string) và yob (year of birth - int)
- **Student**: kế thừa từ Person, thêm grade (string)
- **Teacher**: kế thừa từ Person, thêm subject (string)
- **Doctor**: kế thừa từ Person, thêm specialist (string)
- **Ward**: có name và list để chứa tất cả people

### Is-a Relationship
Bài toán nhấn mạnh mối quan hệ "Is-a":
- A Student **is a** Person
- A Teacher **is a** Person  
- A Doctor **is a** Person

Các derived classes chia sẻ thuộc tính chung (name, yob) từ base class nhưng có thuộc tính riêng.

### Implementation Chi Tiết

#### Abstract Base Class
```python
from abc import ABC, abstractmethod

class Person(ABC):
    def __init__(self, name:str, yob:int):
        self.__name = name  # Private attribute
        self.__yob = yob    # Private attribute
    
    def get_name(self):
        return self.__name
    
    def get_yob(self):
        return self.__yob
    
    @abstractmethod
    def describe(self):
        pass
```

#### Derived Classes
```python
class Student(Person):
    def __init__(self, name: str, yob: int, grade:int):
        super().__init__(name, yob)  # Gọi constructor của class cha
        self.grade = grade 

    def describe(self):
        print(f"Student Name:{self.get_name()}, Yob:{self.get_yob()}, Grade: {self.grade}")

class Teacher(Person):
    def __init__(self, name: str, yob: int, subject:str):
        super().__init__(name, yob)
        self.subject = subject

    def describe(self):
        print(f"Teacher-Name:{self.get_name()}, YoB:{self.get_yob()} , Subject:{self.subject}")

class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist:str):
        super().__init__(name, yob)
        self.specialist = specialist
    
    def describe(self):
        print(f"Doctor-Name:{self.get_name()},YoB:{self.get_yob()},specialist:{self.specialist}")
```

#### Ward Class với Composition
```python
class Ward:
    def __init__(self, name:str) -> None:
        self.__name = name
        self.__listpeople = list()  # Composition: Ward chứa danh sách Person

    def add_person(self, person:Person):
        self.__listpeople.append(person)

    def describe(self):
        print(f"Ward Name: {self.__name}")
        for p in self.__listpeople:
            p.describe()  # Polymorphism: mỗi Person có describe() riêng

    def CountDoctor(self):
        counter = 0 
        for p in self.__listpeople:
            if isinstance(p, Doctor):  # Type checking
                counter+=1
        return counter 
```

### Demo Composition và Polymorphism
```python
student1 = Student(name="StudentA", yob=2010, grade=7)
teacher1 = Teacher(name="teacherA", yob=1969, subject="math")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor1 = Doctor(name="DoctorA", yob=1945 , specialist="Endocrinologist")
doctor2 = Doctor(name="DoctorB", yob=1975, specialist="Cardiologist")

ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)

print(ward1.CountDoctor())  # Kết quả: 2
```

---

## SECTION 4: Data Structures - Stack và Queue

### Stack (Ngăn Xếp)

#### Khái Niệm
Stack hoạt động theo nguyên tắc **Last In First Out (LIFO)** - phần tử cuối cùng được thêm vào sẽ là phần tử đầu tiên được lấy ra.

#### Đặc Điểm
- Có kích thước xác định hoặc giới hạn (capacity)
- Sử dụng TOP pointer để chỉ phần tử cuối cùng

#### Operations
- **Push**: Thêm element vào top của stack
- **Pop**: Lấy element từ top của stack
- **Overflow**: Xảy ra khi push vào stack đầy
- **Underflow**: Xảy ra khi pop từ stack rỗng
- **Top()**: Trả về element ở top mà không xóa

#### Implementation
```python
class MyStack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = list()

    def get_capacity(self):
        return self.__capacity
    
    def get_stack(self):
        return self.__stack
    
    def is_full(self):
        return len(self.__stack) == self.__capacity
    
    def is_empty(self):
        return len(self.__stack) == 0
    
    def push(self, value):
        if not self.is_full():
            self.__stack.append(value)
        else:
            print("Stackoverflow")
    
    def top(self):
        return self.__stack.pop()  # LIFO - Last In First Out
```

### Queue (Hàng Đợi)

#### Khái Niệm
Queue hoạt động theo nguyên tắc **First In First Out (FIFO)** - phần tử đầu tiên được thêm vào sẽ là phần tử đầu tiên được lấy ra.

#### Đặc Điểm
- Có kích thước xác định hoặc giới hạn (capacity)
- Sử dụng Front Pointer và Rear Pointer để quản lý elements

#### Operations
- **Enqueue**: Thêm element vào cuối (rear) của queue
- **Dequeue**: Lấy element từ đầu (front) của queue
- **Overflow**: Xảy ra khi enqueue vào queue đầy
- **Underflow**: Xảy ra khi dequeue từ queue rỗng
- **Front()**: Trả về element ở front mà không xóa

#### Implementation
```python
class MyQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = list()
    
    def is_full(self):
        return len(self.queue) == self.capacity
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def push(self,value):
        self.queue.append(value)  # FIFO - First In First Out
```

---

## Tổng Kết Kiến Thức OOP Đã Học

### 1. **Encapsulation (Tính đóng gói)**
- Sử dụng private attributes (`__name`, `__yob`)
- Cung cấp public methods để truy cập (`get_name()`, `get_yob()`)
- Bảo vệ dữ liệu khỏi truy cập trực tiếp từ bên ngoài

### 2. **Inheritance (Tính kế thừa)**
- `Student`, `Teacher`, `Doctor` kế thừa từ `Person`
- Tái sử dụng code từ class cha
- Mở rộng functionality cho class con
- Học được 4 loại inheritance: Single, Multilevel, Hierarchical, Multiple

### 3. **Polymorphism (Tính đa hình)**
- Cùng method `describe()` nhưng có hành vi khác nhau
- Runtime polymorphism thông qua method override
- Cho phép xử lý các object khác nhau một cách thống nhất

### 4. **Abstraction (Tính trừu tượng)**
- Abstract class `Person` định nghĩa interface chung
- Ẩn implementation details
- Tập trung vào behavior thay vì implementation

### 5. **Composition**
- `Ward` sử dụng composition để chứa danh sách `Person`
- Loose coupling giữa các class
- Linh hoạt hơn inheritance trong một số trường hợp

### 6. **Data Structures**
- **Stack (LIFO)**: Phần tử cuối cùng được thêm vào sẽ được lấy ra trước
- **Queue (FIFO)**: Phần tử đầu tiên được thêm vào sẽ được lấy ra trước
- **Encapsulation**: Ẩn implementation details, chỉ expose các method cần thiết

## Những Thách Thức và Bài Học

### Thách Thức Gặp Phải:
1. **Hiểu Abstract Classes**: Ban đầu khó hiểu tại sao cần abstract class
2. **Type Checking**: Cần thời gian để làm quen với `isinstance()`
3. **Composition vs Inheritance**: Phân biệt khi nào dùng cái nào
4. **PyTorch Integration**: Hiểu cách OOP hoạt động trong framework ML
5. **Data Structures**: Nắm vững logic LIFO vs FIFO

### Bài Học Rút Ra:
1. **Practice Makes Perfect**: Code nhiều giúp hiểu sâu hơn
2. **Real-world Examples**: Các ví dụ thực tế giúp dễ hiểu hơn
3. **Step-by-step Learning**: Học từng khái niệm một cách có hệ thống
4. **Framework Understanding**: OOP là nền tảng cho các framework hiện đại
5. **Problem-solving Approach**: Áp dụng OOP để giải quyết bài toán thực tế

## Kết Luận

Tuần học OOP này đã mở ra cho tôi một thế giới mới trong lập trình. Từ những khái niệm cơ bản như class, object đến những khái niệm nâng cao như abstract classes, polymorphism, tôi đã có cái nhìn tổng quan về cách tổ chức code theo hướng đối tượng.

Đặc biệt, việc áp dụng OOP trong PyTorch đã cho tôi thấy tầm quan trọng của các nguyên tắc này trong thực tế. Các data structures như Stack và Queue cũng giúp tôi hiểu sâu hơn về cách tổ chức và quản lý dữ liệu.

Những kiến thức này không chỉ hữu ích cho việc học lập trình mà còn là nền tảng quan trọng cho việc phát triển các ứng dụng phức tạp trong tương lai. Tôi mong đợi được áp dụng những kiến thức này vào các dự án thực tế và tiếp tục học hỏi thêm về các design patterns và best practices trong OOP.

---


