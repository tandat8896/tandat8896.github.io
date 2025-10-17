---
title: "Khám phá Lập trình hướng đối tượng (OOP): Từ Nền Tảng đến Các Khái Niệm Nâng Cao"
description: "Hành trình khám phá OOP trong Python: từ khái niệm cơ bản về Class và Object đến các nguyên lý nâng cao như Encapsulation, Inheritance, Polymorphism và Delegation."
pubDatetime: 2025-01-15T10:00:00Z
tags:
  - python
  - oop
  - week3
  - programming
draft: false
---

# Khám phá Lập trình hướng đối tượng (OOP): Từ Nền Tảng đến Các Khái Niệm Nâng Cao

## Mở đầu: Hành trình từ cơ bản đến nâng cao

Hôm trước, chúng ta đã tìm hiểu những khái niệm cơ bản về Lập trình hướng đối tượng (OOP) - một phương pháp lập trình mạnh mẽ đã thay đổi cách chúng ta xây dựng phần mềm. Hôm nay, chúng ta sẽ đi sâu hơn vào các nguyên lý cốt lõi và khám phá cách OOP hoạt động trong Python một cách chi tiết và thực tế.

## 1. OOP Là Gì và Tại Sao Nó Quan Trọng?

Lập trình hướng đối tượng là một mô hình lập trình tập trung vào việc tạo ra các đối tượng chứa cả dữ liệu và các hàm (hành vi). Thay vì viết các hàm riêng lẻ thực hiện thao tác trên dữ liệu (như trong lập trình thủ tục), OOP kết hợp chúng lại thành một "thực thể" duy nhất – đó chính là đối tượng.

### Mục tiêu chính của OOP bao gồm:

- **Đóng gói (Encapsulation)**: Kết hợp dữ liệu và các phương thức xử lý dữ liệu vào một thể thống nhất.
- **Ủy quyền (Delegation)**: Cho phép một đối tượng sử dụng các chức năng của một đối tượng khác để thực hiện công việc.
- **Kế thừa (Inheritance)**: Tái sử dụng mã nguồn bằng cách cho phép một lớp kế thừa các thuộc tính và phương thức từ một lớp khác.
- **Đa hình (Polymorphism)**: Khả năng một đối tượng có thể mang nhiều hình thái hoặc một phương thức có thể được thực hiện theo nhiều cách khác nhau.

So với lập trình thủ tục (tiếp cận từ trên xuống), OOP mang lại tiếp cận từ dưới lên. Điều này cải thiện đáng kể các khía cạnh như bảo mật, bảo trì và khả năng tái sử dụng mã.

## 2. Đối Tượng và Lớp (Objects and Classes) – Nền Tảng của OOP

Trong OOP, **Lớp (Class)** được xem như một khuôn mẫu trừu tượng. Nó là một bản thiết kế để tạo ra các đối tượng. Còn **Đối tượng (Object)** là một thể hiện cụ thể (instance) của một lớp. Ví dụ, "John" là một đối tượng của lớp "Student".

### Mỗi lớp được định nghĩa bởi:

- **Thuộc tính (Attributes)**: Là dữ liệu đặc trưng cho đối tượng, ví dụ như màu lông, màu mắt, chiều cao của một con chó. Trong lập trình, các thuộc tính thường được gọi là biến thành viên.
- **Phương thức (Methods)**: Là các hành vi hoặc hành động mà đối tượng có thể thực hiện. Ví dụ, các hoạt động hàng ngày của một con mèo có thể là ngủ, đi bộ và ăn.

### Class Diagram - Ngôn ngữ chung của thiết kế phần mềm

Để hình dung cấu trúc của một hệ thống, người ta sử dụng **Class Diagram (Biểu đồ lớp)**. Đây là một ngôn ngữ chung cho những người thiết kế phần mềm, không dành riêng cho bất kỳ ngôn ngữ lập trình nào.

Ví dụ về Class Diagram cho lớp Cat:
- **Tên lớp**: Cat
- **Thuộc tính**: `+ name: string`, `+ color: string`, `+ age: float`
- **Phương thức**: `+ sleep(): void`, `+ walk(): void`, `+ eat(): void`

### Các ký hiệu truy cập (Access modifiers):

- **+ (dấu cộng)**: Biểu thị public (công khai), có thể truy cập tự do từ bên ngoài.
- **- (dấu trừ)**: Biểu thị private (riêng tư), chỉ được sử dụng bên trong lớp đó.
- **# (dấu thăng)**: Biểu thị protected (bảo vệ), có thể truy cập được trong lớp đó và các lớp con của nó.

## 3. Từ Khóa `self` trong Python

Khi tạo một phương thức trong một lớp trong Python, bạn phải thêm từ khóa `self` vào vị trí đầu tiên của các tham số. `self` là một tham chiếu đến thể hiện hiện tại của lớp.

### Đặc điểm của `self`:

- `self` không có ý nghĩa cố định nào ngoài việc nó là tham số đầu tiên; bạn có thể đặt tên biến khác nhưng `self` là quy ước chung.
- Khi bạn muốn tiếp cận dữ liệu bên trong (thuộc tính) hoặc gọi các phương thức khác của đối tượng, bạn phải thông qua từ khóa `self` (ví dụ: `self.name`, `self.age`).
- `self` cũng hỗ trợ khởi tạo hoặc cập nhật thuộc tính: nếu `self.name` chưa có, nó sẽ tự động tạo; nếu đã có, nó sẽ cập nhật giá trị.

## 4. Các Phương Thức Đặc Biệt: `__init__` và `__call__`

### `__init__()` (Constructor):
Đây là một phương thức đặc biệt được gọi tự động mỗi khi một đối tượng mới được tạo ra từ lớp. Nó dùng để khởi tạo các thuộc tính của đối tượng.

**Lưu ý quan trọng**: Trong Python, bạn chỉ có thể có duy nhất một phương thức `__init__` về mặt cú pháp. Nếu bạn định nghĩa một `__init__` tường minh với các tham số, thì constructor mặc định (không có tham số) sẽ bị hủy.

### `__call__()`:
Phương thức này cho phép các thể hiện của lớp hoạt động như một hàm và có thể được gọi trực tiếp như một hàm.

## 5. Đóng Gói (Encapsulation) và Kiểm Soát Truy Cập

Encapsulation là một trong những nguyên lý cốt lõi của OOP, giúp che giấu thông tin và kiểm soát quyền truy cập vào dữ liệu nội bộ của một đối tượng.

### Các mức độ truy cập:

- **Public**: Trong Python, các thuộc tính và phương thức mặc định là public. Chúng có thể truy cập tự do từ bên ngoài lớp.
- **Private**: Để biểu thị một thuộc tính là private, chúng ta sử dụng hai dấu gạch dưới (`__`) trước tên thuộc tính (ví dụ: `__name`). Các thuộc tính private chỉ được sử dụng bên trong lớp đó.
- **Protected**: Một dấu gạch dưới (`_`) thường được dùng để chỉ thuộc tính/phương thức có thể truy cập bên trong lớp và các lớp con.

### Getter và Setter:

Để truy cập hoặc sửa đổi các thuộc tính private từ bên ngoài lớp, chúng ta sử dụng các hàm Getter và Setter.

- **Getter (lấy dữ liệu)**: Hàm này dùng để đọc giá trị của một thuộc tính.
- **Setter (gán dữ liệu)**: Hàm này dùng để gán (sửa) giá trị cho một thuộc tính.

Việc sử dụng Getter và Setter giúp chúng ta:
1. Kiểm soát truy cập
2. Kiểm tra tính hợp lệ dữ liệu
3. Bao đóng và che giấu thông tin

## 6. Ủy Quyền (Delegation) – Mối Quan Hệ "Has-A"

Ủy quyền hay còn gọi là Composition (Tổng hợp), mô tả mối quan hệ "có" (has-a) giữa các lớp. Thay vì tự thực hiện một công việc, một lớp có thể "ủy thác" hoàn toàn hành động đó cho một đối tượng khác mà nó sở hữu.

**Ví dụ**: Một Person (người) "có" một Date (ngày sinh).

Delegation giúp tái sử dụng mã nguồn và tổ chức code chặt chẽ hơn bằng cách tận dụng các chức năng đã có sẵn của các lớp khác.

## 7. Kế Thừa (Inheritance) – Mối Quan Hệ "Is-A"

Kế thừa là một cơ chế cho phép một lớp (lớp con) thừa hưởng các thuộc tính và phương thức từ một lớp khác (lớp cha). Nó thể hiện mối quan hệ "là một loại" (is-a).

### Các khái niệm quan trọng:

- **Lớp Cha (Superclass/Base Class/Parent Class)**: Là lớp mà các tính năng của nó được kế thừa.
- **Lớp Con (Subclass/Derived Class/Child Class)**: Là lớp thừa hưởng từ lớp cha.

### Quy tắc truy cập trong Kế thừa:

- Các lớp con có thể sử dụng các thuộc tính và phương thức public và protected của lớp cha.
- Các thuộc tính private của lớp cha không thể được truy cập trực tiếp từ lớp con.
- Khi một lớp con kế thừa từ lớp cha, nếu lớp con cần gọi `__init__` của lớp cha, nó có thể dùng `super().__init__()`.

### Ghi đè phương thức (Method Overriding):

Kế thừa cho phép lớp con tái sử dụng các phương thức của lớp cha. Tuy nhiên, đôi khi, lớp con cần một triển khai khác cho một phương thức đã có trong lớp cha. Trong trường hợp này, lớp con có thể "ghi đè" (override) phương thức đó bằng cách định nghĩa lại nó.

### Kế thừa như một khuôn mẫu:

Một lớp cha có thể hoạt động như một "khuôn mẫu" hoặc abstract class, định nghĩa các phương thức mà các lớp con bắt buộc phải triển khai. Trong Python, chúng ta thường sử dụng `@abstractmethod` để đánh dấu các phương thức cần được triển khai bởi lớp con.

## 8. Quy ước và Lưu ý quan trọng

### Quy ước đặt tên:
- **Tên lớp**: Thường là các từ viết hoa chữ cái đầu và nối liền nhau (ví dụ: `Cat`, `JapaneseBobtail`).
- **Tên thuộc tính**: Thường là danh từ và các từ được nối với nhau bằng dấu gạch dưới (ví dụ: `name`, `date_of_birth`).

### Các lưu ý khác:
- **`__main__`**: Khi bạn chạy trực tiếp một file `.py`, Python sẽ tự động gán tên module của file đó là `__main__`.
- **Tạo đối tượng rỗng**: Nếu bạn định nghĩa một lớp mà không có phương thức `__init__` nào, Python sẽ cung cấp một `__init__` mặc định rỗng.
- **Đặt tên có ý nghĩa**: Luôn cố gắng đặt tên cho các biến, lớp, phương thức một cách có ý nghĩa để code dễ đọc và hiểu.

## Kết luận

OOP không chỉ là một phương pháp lập trình, mà còn là một cách tư duy về việc tổ chức và xây dựng phần mềm. Thông qua việc hiểu sâu về các nguyên lý cốt lõi như Encapsulation, Inheritance, Polymorphism và Delegation, chúng ta có thể tạo ra những ứng dụng mạnh mẽ, dễ bảo trì và có thể mở rộng.

Trong bài viết tiếp theo, chúng ta sẽ khám phá các ứng dụng thực tế của OOP và cách áp dụng những kiến thức này vào các dự án thực tế. Hãy cùng tiếp tục hành trình khám phá thế giới lập trình hướng đối tượng!

---

*Hy vọng bài viết này giúp bạn có cái nhìn toàn diện hơn về OOP và các khái niệm quan trọng của nó trong Python!* 