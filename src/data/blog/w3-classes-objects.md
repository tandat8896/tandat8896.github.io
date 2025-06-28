---
title: "Khám phá Lập trình Hướng đối tượng trong Python: Hành trình học tập và thực hành"
description: "Bài viết chuyên sâu về OOP trong Python: từ lý thuyết cơ bản đến thực hành nâng cao, bao gồm Classes, Objects, Inheritance, Encapsulation và các ứng dụng thực tế."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - python
  - oop
  - classes
  - objects
  - inheritance
  - encapsulation
  - week3
draft: false
---

# Khám phá Lập trình Hướng đối tượng trong Python: Hành trình học tập và thực hành

Bước vào thế giới Lập trình Hướng đối tượng (Object-Oriented Programming - OOP) đã là một trải nghiệm thú vị, thay đổi cơ bản cách tôi tiếp cận phát triển phần mềm. Chuyển từ tư duy thủ tục sang hướng đối tượng giống như có được một siêu năng lực để mô hình hóa thế giới thực trong code.

## Từ Tư duy Thủ tục đến Hướng đối tượng

Ban đầu, tôi hiểu lập trình như một chuỗi các bước và hàm thao tác với dữ liệu – được gọi là Lập trình Thủ tục (Procedural Programming). Mặc dù hiệu quả cho các vấn đề nhỏ, nhưng rõ ràng việc quản lý các hệ thống lớn hơn, phức tạp hơn như thư viện hoặc nền tảng mạng xã hội đòi hỏi một cách tiếp cận khác. Đây là nơi **Trừu tượng hóa (Abstraction)** trở nên quan trọng.

Trừu tượng hóa trong OOP có nghĩa là mô hình hóa các thực thể thế giới thực và hành vi của chúng vào thế giới số. Thay vì chỉ có dữ liệu và hàm, chúng ta nghĩ về các đối tượng, đặc điểm của chúng (thuộc tính) và những gì chúng có thể làm (phương thức).

Ví dụ, trong hệ thống Quản lý Thư viện, thay vì các danh sách dữ liệu và hàm riêng biệt, chúng ta nghĩ về Book, Reader và Librarian như các thực thể riêng biệt, mỗi thực thể có thuộc tính và hành động riêng.

## Nền tảng: Classes và Objects

Các khái niệm cơ bản trong OOP là **Classes** và **Objects**.

- **Class** về cơ bản là một bản thiết kế hoặc template để tạo ra các đối tượng. Hãy nghĩ về nó như thiết kế cho một chiếc xe hơi.
- **Object** (còn gọi là instance) là một hiện thực cụ thể của bản thiết kế đó. Vì vậy, nếu class "car" là bản thiết kế, thì "my blue sedan" và "your red truck" là các đối tượng cụ thể được tạo từ nó, mỗi đối tượng có dữ liệu duy nhất nhưng chia sẻ cùng cấu trúc và hành vi đã định nghĩa.

Ví dụ, một class User cho mạng xã hội có thể có các thuộc tính như Name, Birthday, Gender, Phone, Email và Friends. Nó cũng sẽ có các phương thức (hành vi) như Post, Message, Like và Comment. Tương tự, một class Book cho thư viện sẽ chỉ chứa thông tin thực sự cần thiết, như title, author, rm_time và is_borrow.

## Tạo sự sống cho Objects: Constructors và từ khóa self

Tạo một đối tượng liên quan đến việc gọi **Constructor** của nó, thường là hàm `__init__()` trong Python. Hàm này được tự động gọi mỗi khi một đối tượng mới được tạo từ một class. Mục đích chính của nó là khởi tạo các thuộc tính của đối tượng mới với các giá trị cụ thể. Không phải tất cả thuộc tính đều phải được khởi tạo ở đây; một số có thể được thêm vào trong các phương thức khác sau này.

Một trong những khía cạnh đặc biệt nhất tôi học được là việc sử dụng từ khóa **self**. Từ khóa self luôn đại diện cho instance của class. Nó phải là tham số đầu tiên trong mọi phương thức, mặc dù bạn không truyền nó một cách rõ ràng khi gọi phương thức. Các biến có tiền tố self là các thuộc tính của class, trong khi những biến khác chỉ là biến cục bộ trong một phương thức.

Một điểm quan trọng về self bắt nguồn từ "The Zen of Python": "Explicit is better than implicit". Các nhà phát triển Python thích rằng self được hiển thị rõ ràng, cho thấy rằng một phương thức thuộc về một instance cụ thể, thay vì là một từ khóa "ma thuật" ẩn như `this` trong một số ngôn ngữ khác. Thú vị là bạn thực sự có thể thay thế self bằng một từ khác, vì Python tự động hiểu tham số đầu tiên là biến instance.

## Tăng cường tương tác Object: Phương thức __call__

Một "hàm đặc biệt" thú vị tôi gặp phải là phương thức `__call__()`. Phương thức này cho phép một instance của class hoạt động như một hàm và được gọi trực tiếp bằng dấu ngoặc đơn, giống như một hàm thông thường. Nếu bạn muốn một đối tượng có thể gọi được (ví dụ: `my_object()`), nó phải có phương thức `__call__()` được định nghĩa. Nếu không có nó, bạn chỉ có thể tương tác với các thuộc tính (`obj.attribute`) hoặc phương thức (`obj.method()`).

## Thực hành tốt: Quy ước đặt tên

Áp dụng quy ước đặt tên tốt là rất quan trọng cho code dễ đọc và dễ bảo trì. Các nguồn tài liệu đã làm nổi bật một số quy tắc rõ ràng:

- **Tên Class**: Sử dụng PascalCase (ví dụ: SuperCat), trong đó mỗi từ bắt đầu bằng chữ hoa.
- **Tên thuộc tính**: Sử dụng snake_case (ví dụ: cat_name), với các từ được phân tách bằng dấu gạch dưới và tất cả chữ thường. Thuộc tính thường nên là danh từ hoặc cụm danh từ.
- **Tên phương thức**: Ưu tiên sử dụng động từ hoặc cụm động từ và cũng sử dụng snake_case.

## Xây dựng trên cấu trúc hiện có: Kế thừa (Inheritance)

Kế thừa là một cơ chế OOP mạnh mẽ cho phép một class mới (SubClass) kế thừa thuộc tính và phương thức từ một class hiện có (SuperClass). Điều này được mô hình hóa như một mối quan hệ "is-a", ví dụ, một Dog "là một" Animal.

Lợi ích của kế thừa rất đáng kể:

- **Tái sử dụng code**: Tôi có thể tái sử dụng các đoạn code đã được viết trong SuperClass, tránh sự trùng lặp. Ví dụ, Dog và Cat có thể kế thừa name, eat() và speak() từ Animal.
- **Khả năng mở rộng**: Dễ dàng mở rộng chức năng bằng cách sửa đổi SuperClass, và SubClasses tự động được hưởng lợi.

Một khái niệm quan trọng trong kế thừa là **Ghi đè (Overriding)**. Điều này xảy ra khi một SubClass cung cấp implementation riêng cho một phương thức đã được định nghĩa trong SuperClass của nó. Ví dụ, cả Dog và Cat có thể kế thừa phương thức speak() từ Animal, nhưng mỗi loài sẽ ghi đè nó để tạo ra âm thanh đặc trưng của mình. Điều này khác với **Overloading**, liên quan đến việc có nhiều hàm cùng tên nhưng khác tham số trong cùng một class, một tính năng không được hỗ trợ trực tiếp trong Python như trong Java.

Khi một class con không định nghĩa constructor riêng (`__init__`), nó tự động kế thừa constructor từ class cha. Từ khóa `super()` rất hữu ích để gọi các phương thức từ class cha trong class con.

## Bảo vệ dữ liệu: Đóng gói và Access Modifiers

**Đóng gói (Encapsulation)** (còn được gọi là ẩn dữ liệu hoặc bảo vệ) là về việc đóng gói dữ liệu (thuộc tính) và các phương thức hoạt động trên dữ liệu trong một đơn vị duy nhất (class), và hạn chế truy cập trực tiếp vào một số thành phần của đối tượng.

Python hỗ trợ ba loại **Access Modifiers**:

- **Dữ liệu Public**: Có thể truy cập ở bất kỳ đâu, cả bên trong và bên ngoài class. Đây là mặc định trong Python (ví dụ: `public_attribute`).
- **Dữ liệu Protected**: Được thiết kế để có thể truy cập trong class và các class con của nó. Trong Python, điều này được chỉ ra theo quy ước bằng một dấu gạch dưới đầu (`_attribute_name`). Mặc dù nó không ngăn chặn truy cập một cách nghiêm ngặt, nhưng đây là một quy ước mạnh mẽ cho các nhà phát triển rằng biến này dành cho "sử dụng nội bộ".
- **Dữ liệu Private**: Được thiết kế để chỉ có thể truy cập trong class nơi nó được định nghĩa. Trong Python, điều này được thực hiện bằng cách sử dụng hai dấu gạch dưới đầu (`__attribute_name`). Điều này kích hoạt "name mangling", nơi Python đổi tên thuộc tính một cách nội bộ để ngăn chặn việc ghi đè ngẫu nhiên từ các class con (ví dụ: `_ClassName__var`). Điều này có nghĩa là truy cập trực tiếp như `obj.__x` sẽ không hoạt động, nhưng `obj._Class__x` có thể, làm cho nó trở thành một hình thức riêng tư mềm.

## Ví dụ thực tế: Hệ thống Quản lý Thư viện

Dựa trên các file trong thư mục của tôi, tôi đã thực hành OOP thông qua việc xây dựng một hệ thống quản lý thư viện. Đây là một ví dụ minh họa:

```python
class Person:
    def __init__(self, name: str, yob: int) -> None:
        self.name = name
        self.yob = yob
    
    def describe(self):
        print(f"Name: {self.name}, Yob: {self.yob}")

class Student(Person):
    def __init__(self, name, yob, grade: str):
        super().__init__(name, yob)
        self.grade = grade
    
    def describe(self):
        print(f"Thông tin của sinh viên\nName: {self.name}, Yob: {self.yob}, Grade: {self.grade}")

class Teacher(Person):
    def __init__(self, name, yob, subject) -> None:
        super().__init__(name, yob)
        self.subject = subject
    
    def describe(self):
        return f"Name: {self.name}, YoB: {self.yob}, Subject: {self.subject}"

class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self.specialist = specialist
    
    def describe(self):
        return f"Name: {self.name}, YoB: {self.yob}, Specialist: {self.specialist}"

class Ward:
    def __init__(self, name: str):
        self.__name = name  # Private attribute
        self.__ListPeople = list()  # Private attribute
    
    def Add_Person(self, person: Person):  # Composition: Ward chứa collection của Person
        self.__ListPeople.append(person)
    
    def describe(self):
        print(f"Name: {self.__name}")
        for person in self.__ListPeople:  # Delegation: Ward ủy thác cho từng Person
            person.describe()
    
    def CountDoctor(self):
        count = 0
        for person in self.__ListPeople:
            if isinstance(person, Doctor):  # Kiểm tra kiểu đối tượng
                count += 1
        return count
```

## Kết luận

Hành trình vào OOP đã mở ra những khái niệm mạnh mẽ giúp đơn giản hóa phát triển và thúc đẩy tổ chức code tốt hơn. Từ việc hiểu các bản thiết kế của classes đến việc tạo ra các đối tượng duy nhất, xử lý việc khởi tạo của chúng với constructors và self, tận dụng kế thừa để tái sử dụng code, và quản lý truy cập dữ liệu thông qua đóng gói, mỗi bước đã làm sâu sắc thêm sự đánh giá của tôi về mô hình lập trình này.

Tôi rất hào hứng để áp dụng những nguyên tắc này để xây dựng phần mềm mạnh mẽ và có khả năng mở rộng hơn. OOP không chỉ là một kỹ thuật lập trình, mà còn là một cách tư duy mới về cách tổ chức và giải quyết các vấn đề phức tạp trong thế giới thực.

---

*Bài viết này là một phần của hành trình học tập Module 1 - Week 3 về Lập trình Hướng đối tượng trong Python.* 