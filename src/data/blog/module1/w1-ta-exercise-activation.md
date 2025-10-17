---
title: "Khám phá Python và Machine Learning: Từ Cơ bản đến Thực hành"
description: "Bài viết chuyên sâu về Python và ứng dụng trong Machine Learning: lý thuyết cơ bản, các bài tập thực tế về F-Score, activation functions, loss functions và ước lượng hàm lượng giác."
pubDatetime: 2025-01-28T10:00:00Z
tags:
  - python
  - machine-learning
  - activation-functions
  - loss-functions
  - week1
draft: false
---

# Khám phá Python và Machine Learning: Từ Cơ bản đến Thực hành

Tôi muốn chia sẻ những hiểu biết từ một module học tập gần đây tập trung vào lập trình Python, đặc biệt là ứng dụng của nó trong các khái niệm cơ bản của machine learning. Hành trình này bao gồm cú pháp Python cơ bản, điều khiển luồng, hàm, và đi sâu vào các bài tập thực tế như tính toán F-Score, activation functions, loss functions, và thậm chí ước lượng các hàm lượng giác.

## Xây dựng Nền tảng Python Vững chắc

Việc học tập của tôi bắt đầu với những kiến thức cơ bản của Python. Tôi nhanh chóng khám phá ra rằng Python dễ học và sử dụng, hỗ trợ nhiều mô hình lập trình khác nhau như lập trình thủ tục, hướng đối tượng, hoặc lập trình hàm. Nó cũng được nhấn mạnh là hoàn hảo cho các dự án phức tạp và nhanh chóng.

### Các khái niệm cơ bản chính bao gồm:

**Biến (Variables):** Hiểu rằng biến hoạt động như một container hoặc vùng lưu trữ cho dữ liệu, được gán bằng toán tử `=`. Python hỗ trợ các kiểu dữ liệu cơ bản như Boolean, string (str), integer (int), float, và số phức.

**Chuyển đổi kiểu dữ liệu (Type Conversion):** Học về việc chuyển đổi dữ liệu từ một kiểu sang kiểu khác, có thể là ngầm định (tự động) hoặc rõ ràng (thủ công). Điều quan trọng là phải chú ý đến kiểu dữ liệu khi khai báo biến và lưu trữ dữ liệu.

**Input và Output:** Thành thạo hàm `print()` cho output và hàm `input()` để nhận input từ người dùng.

**Toán tử (Operators):** Làm quen với các toán tử khác nhau:
- **Số học:** `+`, `-`, `*`, `//` (chia lấy phần nguyên), `%` (chia lấy dư), `**` (lũy thừa)
- **Gán:** `+=`, `-=`, `*=` v.v.
- **So sánh:** `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logic:** `and`, `or`, `not`

**Điều khiển luồng (Flow Control):** Hiểu cách điều khiển thực thi chương trình với:
- **Câu lệnh if-else:** Thực thi các khối code chỉ khi các điều kiện cụ thể được đáp ứng, bao gồm `if`, `if...else`, và `if...elif...else`
- **Vòng lặp:** Sử dụng vòng lặp `for` để lặp qua các chuỗi (như list hoặc string) và vòng lặp `while` để lặp lại code cho đến khi điều kiện không còn đúng. Thú vị là, một khối `else` cũng có thể kết hợp với vòng lặp `for`

## Hàm Python - Sức mạnh của Modular Programming

Một trong những khía cạnh mạnh mẽ nhất được giới thiệu là **Python Functions**. Hàm cho phép chúng ta đóng gói một khối code để "làm gì đó với input" và tạo ra output. Điều này bao gồm cả các hàm có sẵn như `print()` và `type()`, và các hàm do người dùng định nghĩa.

Khi định nghĩa hàm, điều quan trọng là phải hiểu **parameters** (được định nghĩa trong function signature) và **arguments** (giá trị được truyền khi gọi hàm). Cấu trúc chung bao gồm một keyword (`def`), dấu ngoặc đơn cho parameters, dấu hai chấm, và thụt đầu dòng cho function body.

## Đi sâu vào các Bài tập Machine Learning Thực tế

Các bài tập thực hành thực sự củng cố hiểu biết của tôi về cách Python được áp dụng trong machine learning. Mỗi bài tập đều liên quan đến việc viết các hàm để tính toán các metrics hoặc giá trị cụ thể.

### 1. Tính toán Metric F-Score

Bài tập thực tế đầu tiên tập trung vào **F1-score**, một metric quan trọng được sử dụng để đánh giá độ chính xác của một mô hình phân loại. Điều này liên quan đến việc hiểu **confusion matrix**, giúp ích trong các tác vụ phân loại nhị phân (ví dụ: dự đoán mèo vs chó).

Các thuật ngữ chính trong confusion matrix:
- **TP (True Positive):** Mô hình dự đoán Dog, và nó thực sự là Dog
- **FP (False Positive):** Mô hình dự đoán Dog, nhưng nó là Cat
- **FN (False Negative):** Mô hình dự đoán Cat, nhưng nó là Dog
- **TN (True Negative):** Mô hình dự đoán Cat, và nó thực sự là Cat

F1-score là trung bình điều hòa của Precision và Recall:
- **Precision** đo lường "Có bao nhiêu dự đoán dương tính thực sự đúng?" Được tính bằng `TP / (TP + FP)`
- **Recall** đo lường "Có bao nhiêu dương tính thực tế được dự đoán đúng?" Được tính bằng `TP / (TP + FN)`
- **Công thức F1-score:** `2 * (Precision * Recall) / (Precision + Recall)`

Bài tập bao gồm việc viết một hàm nhận `tp`, `fp`, và `fn` làm input và trả về Precision, Recall, và F1-Score. Một điều kiện quan trọng cho hàm này là xử lý phép chia cho không.

### 2. Tính toán Activation Functions

Tiếp theo, tôi khám phá **activation functions**, là các thành phần cơ bản trong neural networks. Bài tập yêu cầu viết một hàm để tính toán activation function cho một input x và tên của hàm. Hàm nên nhận x như một float và tên (sigmoid, relu, elu) như một string.

Ba activation functions cụ thể được đề cập:

**Sigmoid Function:** `sigmoid(x) = 1 / (1 + e^(-x))`
- Ví dụ: `sigmoid(3) = 0.95`

**ReLU (Rectified Linear Unit) Function:** `relu(x) = 0 if x <= 0 else x if x > 0`
- Ví dụ: `relu(-4) = 0` và `relu(5) = 5`

**ELU (Exponential Linear Unit) Function:** `ELU(x) = alpha * (e^x - 1) if x <= 0 else x if x > 0`
- Giá trị alpha phổ biến là 0.01
- Ví dụ: `ELU(-4) = 0.01 * (e^(-4) - 1) = -0.0098`

### 3. Tính toán Loss Functions

Loss functions được sử dụng để định lượng lỗi của các dự đoán của mô hình. Bài tập này bao gồm việc viết một hàm để tính toán giá trị loss dựa trên số lượng data points n và tên của loss function. Hàm mong đợi n như một integer và tên loss (mae, mse, rmse).

Với y_i và y_hat_i là giá trị thực tế và dự đoán qua i lần lặp từ 0 đến n:

**MAE (Mean Absolute Error):** `MAE = (1/n) * sum(|y_i - y_hat_i|)`

**MSE (Mean Squared Error):** `MSE = (1/n) * sum((y_i - y_hat_i)^2)`

**RMSE (Root Mean Squared Error):** `RMSE = sqrt(MSE)`

### 4. Ước lượng Hàm Lượng giác

Bài tập cuối cùng đi vào việc ước lượng các hàm lượng giác sử dụng chuỗi vô hạn. Cụ thể, tập trung vào việc xấp xỉ `sin(x)`. Hàm được thiết kế cho việc này nhận x (một số) và n (số lượng terms trong chuỗi) làm input và trả về `sin(x)`.

Xấp xỉ cho `sin(x)` được cho bởi chuỗi Taylor:
`sin(x) ≈ sum from i=0 to n of ((-1)^i * x^(2i+1)) / (2i + 1)!`

Điều này mở rộng thành: `x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - ....`

Một thành phần quan trọng cho việc tính toán này là hàm factorial, `factorial_fcn(value)`, tính toán giai thừa của một số cho trước.

Các hàm lượng giác khác được đề cập để ước lượng bao gồm `cos(x)`, `sinh(x)`, và `cosh(x)`, mỗi hàm có chuỗi mở rộng tương ứng của chúng.

## Suy ngẫm và Bước tiếp theo

Module học tập này cung cấp một cái nhìn toàn diện về các kiến thức cơ bản của lập trình Python và ứng dụng thực tế của chúng trong machine learning. Từ việc hiểu các kiểu dữ liệu cơ bản và điều khiển luồng đến việc triển khai các hàm toán học phức tạp, các bài tập đòi hỏi sự chú ý cẩn thận đến chi tiết, đặc biệt là về các điều kiện input (ví dụ: x như float, n như int) và xử lý các trường hợp đặc biệt như phép chia cho không.

Việc nhấn mạnh vào việc xây dựng các hàm do người dùng định nghĩa cho mỗi tác vụ là vô cùng có giá trị, củng cố khái niệm về code modular và có thể tái sử dụng. Nó làm nổi bật tầm quan trọng của việc không chỉ biết các công thức mà còn biết cách dịch chúng thành code mạnh mẽ, có chức năng.

Tôi mong đợi được áp dụng những kỹ năng nền tảng này vào các chủ đề nâng cao hơn trong machine learning! 