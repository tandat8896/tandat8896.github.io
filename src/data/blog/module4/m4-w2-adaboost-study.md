---
title: "AdaBoost Study - Thuật toán Adaptive Boosting"
pubDatetime: 2025-09-19T10:00:00Z
featured: false
description: "Tìm hiểu chi tiết về AdaBoost, thuật toán adaptive boosting sử dụng weak learners để tạo strong learner"
tags: ["machine-learning", "adaboost", "boosting", "ensemble", "algorithm"]
---

# Adaboost (Adaptive Boosting)

## Thuật toán AdaBoost

AdaBoost là một thuật toán ensemble learning sử dụng phương pháp boosting để kết hợp nhiều weak learners thành một strong learner. Thuật toán hoạt động bằng cách:

1. **Khởi tạo trọng số** cho tất cả các mẫu dữ liệu
2. **Lặp lại** cho mỗi weak learner:
   - **Resample** dataset dựa trên trọng số hiện tại
   - Huấn luyện weak learner trên dataset đã resample
   - Tính toán error rate và trọng số cho weak learner
   - Cập nhật trọng số cho các mẫu dữ liệu
3. **Kết hợp** tất cả weak learners thành strong learner

---

## Nghiên cứu AdaBoost: Hàm giả thuyết và Hàm lỗi mũ

Trong thuật toán AdaBoost, chúng ta xây dựng một bộ phân loại mạnh (strong classifier) bằng cách kết hợp nhiều bộ phân loại yếu (weak learners) một cách tuần tự. Các công thức dưới đây mô tả cách hàm giả thuyết (hypothesis function) được định nghĩa và cách hàm lỗi mũ (exponential loss function) được sử dụng để tối ưu hóa.

### 1. Hàm giả thuyết `f(x)`

Hàm giả thuyết cuối cùng `f(x)` của AdaBoost được định nghĩa là tổng có trọng số của `M` bộ phân loại yếu `G_m(x)`:

$$
f(x) = \sum_{m=1}^{M} \alpha_m \cdot G_m(x)
$$

**Giải thích chi tiết:**
*   $f(x)$: Đây là hàm giả thuyết cuối cùng, hay còn gọi là bộ phân loại mạnh. Nó sẽ đưa ra dự đoán cho một mẫu đầu vào $x$.
*   $M$: Tổng số bộ phân loại yếu (hay còn gọi là "cây" - trees) được kết hợp. AdaBoost là một thuật toán ensemble, nơi nhiều mô hình đơn giản được kết hợp để tạo ra một mô hình mạnh hơn.
*   $G_m(x)$: Là bộ phân loại yếu thứ $m$. Đây thường là một mô hình đơn giản, ví dụ như một cây quyết định có độ sâu bằng 1 (decision stump).
*   $\alpha_m$ (alpha_m): Là trọng số (weight) của bộ phân loại yếu $G_m(x)$. Trọng số này được tính toán dựa trên hiệu suất của $G_m(x)$ trong việc phân loại các mẫu. Các bộ phân loại yếu hoạt động tốt hơn sẽ có trọng số $\alpha_m$ lớn hơn, đóng góp nhiều hơn vào quyết định cuối cùng của $f(x)$.
*   $Label: \{-1, 1\}$: Cho biết các nhãn lớp (class labels) mà mô hình dự đoán. Trong AdaBoost, các nhãn thường được mã hóa là -1 và 1.
*   $N$ Samples: Tổng số mẫu dữ liệu trong tập huấn luyện.

### 2. Hàm lỗi mũ (Exponential Loss Function)

AdaBoost sử dụng hàm lỗi mũ để đo lường mức độ sai lệch của hàm giả thuyết $f(x)$ so với nhãn thực tế $y_i$ của các mẫu dữ liệu.

$$
Err(f) = \sum_{i=1}^{N} e^{-y_i \cdot f(x_i)}
$$

**Giải thích chi tiết:**
*   $Err(f)$: Giá trị lỗi tổng thể của hàm giả thuyết $f$. Mục tiêu của AdaBoost là tối thiểu hóa hàm lỗi này.
*   $N$: Tổng số mẫu dữ liệu.
*   $i$: Chỉ số của từng mẫu dữ liệu, chạy từ 1 đến $N$.
*   $y_i$: Nhãn thực tế của mẫu dữ liệu thứ $i$ (có giá trị -1 hoặc 1).
*   $f(x_i)$: Giá trị dự đoán của hàm giả thuyết $f$ cho mẫu dữ liệu $x_i$.
*   $e^{-y_i \cdot f(x_i)}$: Đây là thành phần lỗi cho từng mẫu.
    *   Nếu $y_i$ và $f(x_i)$ có cùng dấu (dự đoán đúng), thì $y_i \cdot f(x_i)$ sẽ dương, và $e^{-y_i \cdot f(x_i)}$ sẽ nhỏ (gần 0 nếu $f(x_i)$ lớn).
    *   Nếu $y_i$ và $f(x_i)$ có dấu khác nhau (dự đoán sai), thì $y_i \cdot f(x_i)$ sẽ âm, và $e^{-y_i \cdot f(x_i)}$ sẽ lớn, cho thấy lỗi lớn.
    *   Hàm lỗi mũ có đặc điểm là phạt rất nặng các lỗi phân loại, đặc biệt là các mẫu bị phân loại sai với độ tin cậy cao.

### 3. Giả định về quá trình xây dựng hàm giả thuyết

AdaBoost xây dựng hàm giả thuyết một cách tuần tự. Ở mỗi bước $m$, một bộ phân loại yếu mới $G_m(x)$ được thêm vào hàm giả thuyết hiện tại $f_{m-1}(x)$:

$$
f_m(x) = f_{m-1}(x) + \alpha_m \cdot G_m(x)
$$

**Giải thích chi tiết:**
*   $f_m(x)$: Hàm giả thuyết sau khi đã thêm bộ phân loại yếu thứ $m$.
*   $f_{m-1}(x)$: Hàm giả thuyết được xây dựng từ $m-1$ bộ phân loại yếu trước đó.
*   $\alpha_m \cdot G_m(x)$: Đóng góp của bộ phân loại yếu thứ $m$ vào hàm giả thuyết tổng thể, với $\alpha_m$ là trọng số của nó.

### 4. Mở rộng hàm lỗi để tối ưu hóa $\alpha_m$ và $G_m$

Để tìm $\alpha_m$ và $G_m$ tốt nhất ở mỗi bước $m$, chúng ta cần tối thiểu hóa hàm lỗi mũ. Bằng cách thay thế $f(x_i)$ trong công thức hàm lỗi mũ bằng $f_m(x_i)$, chúng ta có thể mở rộng hàm lỗi như sau:

$$
Err(f) = \sum_{i=1}^{N} e^{-y_i \cdot f_m(x_i)}
$$

Thay $f_m(x_i) = f_{m-1}(x_i) + \alpha_m \cdot G_m(x_i)$ vào:

$$
Err(f) = \sum_{i=1}^{N} e^{-y_i \cdot (f_{m-1}(x_i) + \alpha_m \cdot G_m(x_i))}
$$

Sử dụng tính chất $e^{A+B} = e^A \cdot e^B$, ta có:

$$
Err(f) = \sum_{i=1}^{N} e^{-y_i \cdot f_{m-1}(x_i)} \cdot e^{-y_i \cdot \alpha_m \cdot G_m(x_i)}
$$

Từ đó, hàm lỗi có thể được viết lại để thể hiện sự phụ thuộc vào $\alpha_m$ và $G_m$ ở bước hiện tại:

$$
\Rightarrow Err(\alpha_m, G_m) = \sum_{i=1}^{N} e^{-y_i \cdot f_{m-1}(x_i)} \cdot e^{-y_i \cdot \alpha_m \cdot G_m(x_i)}
$$

**Giải thích chi tiết:**
*   $Err(\alpha_m, G_m)$: Hàm lỗi mà chúng ta cần tối thiểu hóa để tìm ra $\alpha_m$ và $G_m$ tốt nhất cho bước hiện tại.
*   $w_i = e^{-y_i \cdot f_{m-1}(x_i)}$: Đây là trọng số của mẫu dữ liệu thứ $i$.
    *   **Quan trọng:** $w_i$ **không phụ thuộc vào** $\alpha_m$ và $G_m$. Giá trị $f_{m-1}(x_i)$ đã được cố định từ các bước huấn luyện trước đó. Điều này có nghĩa là $w_i$ là một hằng số đối với quá trình tối ưu hóa $\alpha_m$ và $G_m$ ở bước hiện tại.
    *   Các mẫu $x_i$ mà $f_{m-1}(x_i)$ đã phân loại sai (tức là $y_i \cdot f_{m-1}(x_i)$ âm) sẽ có $w_i$ lớn hơn. Điều này có nghĩa là AdaBoost sẽ tập trung nhiều hơn vào việc huấn luyện bộ phân loại yếu $G_m(x)$ để phân loại đúng các mẫu mà các bộ phân loại trước đó đã gặp khó khăn.

Tóm lại, ở mỗi bước của AdaBoost, thuật toán sẽ tìm một bộ phân loại yếu $G_m(x)$ và trọng số $\alpha_m$ của nó để tối thiểu hóa hàm lỗi mũ có trọng số, trong đó trọng số của mỗi mẫu $w_i$ được cập nhật dựa trên hiệu suất của các bộ phân loại yếu trước đó.

---

## Phân tích và Đạo hàm Hàm Lỗi trong AdaBoost

Trong thuật toán AdaBoost, chúng ta tìm cách kết hợp nhiều bộ phân loại yếu (weak classifiers) thành một bộ phân loại mạnh (strong classifier). Mỗi bộ phân loại yếu `G_m(x)` sẽ được gán một trọng số `α_m` để thể hiện mức độ đóng góp của nó vào bộ phân loại cuối cùng. Mục tiêu là tối thiểu hóa hàm lỗi.

### 1. Hàm Lỗi Ban Đầu

Hàm lỗi tổng thể `Err(f)` thường được định nghĩa dựa trên hàm mất mát mũ (exponential loss function):

$$
Err(f) = \sum_{i=1}^{N} e^{-y_i f_m(x_i)}
$$

Trong đó $f_m(x_i)$ là bộ phân loại mạnh sau $m$ vòng lặp. $f_m(x_i)$ có thể được viết lại là $f_{m-1}(x_i) + \alpha_m G_m(x_i)$, với $f_{m-1}(x_i)$ là bộ phân loại mạnh từ $m-1$ vòng lặp trước và $\alpha_m G_m(x_i)$ là đóng góp của bộ phân loại yếu hiện tại.

Khi đó, hàm lỗi cho vòng lặp thứ $m$, tập trung vào việc tìm $\alpha_m$ và $G_m$, có thể được viết lại như sau:

$$
Err(\alpha_m, G_m) = \sum_{i=1}^{N} e^{-y_i (f_{m-1}(x_i) + \alpha_m G_m(x_i))}
$$

$$
Err(\alpha_m, G_m) = \sum_{i=1}^{N} e^{-y_i f_{m-1}(x_i)} \cdot e^{-y_i \alpha_m G_m(x_i)}
$$

Ở đây:
*   $N$ là tổng số mẫu dữ liệu.
*   $y_i$ là nhãn thực của mẫu $x_i$ (có giá trị $\{-1, 1\}$).
*   $f_{m-1}(x_i)$ là tổng hợp các bộ phân loại yếu từ các vòng lặp trước.
*   $G_m(x_i)$ là bộ phân loại yếu hiện tại (thường là một cây quyết định nhỏ).
*   $\alpha_m$ là trọng số của bộ phân loại yếu $G_m(x_i)$.

### 2. Định nghĩa Trọng số Mẫu $w_i$

Để đơn giản hóa biểu thức, chúng ta định nghĩa trọng số $w_i$ cho mỗi mẫu $x_i$ dựa trên hiệu suất của bộ phân loại mạnh trước đó:

$$
w_i = e^{-y_i f_{m-1}(x_i)}
$$

Trọng số này phản ánh mức độ khó của mẫu $x_i$. Các mẫu bị phân loại sai bởi $f_{m-1}$ sẽ có trọng số lớn hơn, khiến thuật toán tập trung hơn vào chúng trong vòng lặp hiện tại.

Thay $w_i$ vào hàm lỗi, ta có:

$$
Err(\alpha_m, G_m) = \sum_{i=1}^{N} w_i \cdot e^{-y_i \alpha_m G_m(x_i)}
$$

### 3. Đơn giản hóa Biểu thức $-y_i \alpha_m G_m(x_i)$

Biểu thức $-y_i \alpha_m G_m(x_i)$ có thể được đơn giản hóa dựa trên việc mẫu $x_i$ được phân loại đúng hay sai bởi $G_m(x_i)$:

*   **Nếu $y_i = G_m(x_i)$ (phân loại đúng):**
    Khi đó, $y_i G_m(x_i) = 1$.
    Suy ra, $-y_i \alpha_m G_m(x_i) = -\alpha_m$.

*   **Nếu $y_i \neq G_m(x_i)$ (phân loại sai):**
    Khi đó, $y_i G_m(x_i) = -1$.
    Suy ra, $-y_i \alpha_m G_m(x_i) = -\alpha_m(-1) = \alpha_m$.

### 4. Mở rộng Hàm Lỗi dựa trên Phân loại Đúng/Sai

Sử dụng sự đơn giản hóa trên, chúng ta có thể tách tổng hàm lỗi thành hai phần: một cho các mẫu được phân loại đúng và một cho các mẫu bị phân loại sai bởi $G_m(x_i)$:

$$
Err = \sum_{y_i=G_m(x_i)} w_i \cdot e^{-\alpha_m} + \sum_{y_i \neq G_m(x_i)} w_i \cdot e^{\alpha_m}
$$

$$
Err = e^{-\alpha_m} \sum_{y_i=G_m(x_i)} w_i + e^{\alpha_m} \sum_{y_i \neq G_m(x_i)} w_i
$$

### 5. Định nghĩa Tổng trọng số và Trọng số Lỗi

Để tiếp tục đơn giản hóa, chúng ta định nghĩa:

*   **Tổng trọng số $T_w$:** Tổng trọng số của tất cả các mẫu.
    $$
    T_w = \sum_{i=1}^{N} w_i
    $$

*   **Trọng số lỗi $E_w$:** Tổng trọng số của các mẫu bị phân loại sai bởi $G_m(x_i)$.
    $$
    E_w = \sum_{y_i \neq G_m(x_i)} w_i
    $$

Từ đó, tổng trọng số của các mẫu được phân loại đúng là $T_w - E_w$.

### 6. Hàm Lỗi Đơn giản hóa

Thay các định nghĩa $T_w$ và $E_w$ vào biểu thức hàm lỗi đã mở rộng:

$$
Err = e^{-\alpha_m} (T_w - E_w) + e^{\alpha_m} E_w
$$

Đây là hàm lỗi mà chúng ta cần tối thiểu hóa để tìm $\alpha_m$ tối ưu.

### 7. Đạo hàm Hàm Lỗi để tìm $\alpha_m$ tối ưu

Để tìm giá trị $\alpha_m$ tối ưu, chúng ta lấy đạo hàm của $Err$ theo $\alpha_m$ và đặt nó bằng 0.

$$
\frac{dErr}{d\alpha_m} = \frac{d}{d\alpha_m} [e^{-\alpha_m} (T_w - E_w) + e^{\alpha_m} E_w]
$$

Áp dụng quy tắc đạo hàm $\frac{d}{dx} (e^{ax}) = a e^{ax}$:

$$
\frac{dErr}{d\alpha_m} = -e^{-\alpha_m} (T_w - E_w) + e^{\alpha_m} E_w
$$

Đặt đạo hàm bằng 0 để tìm điểm cực trị:

$$
-e^{-\alpha_m} (T_w - E_w) + e^{\alpha_m} E_w = 0
$$

$$
e^{\alpha_m} E_w = e^{-\alpha_m} (T_w - E_w)
$$

Nhân cả hai vế với `e^{\alpha_m}`:

$$
e^{2\alpha_m} E_w = (T_w - E_w)
$$

$$
e^{2\alpha_m} = \frac{T_w - E_w}{E_w}
$$

Lấy logarit tự nhiên (ln) cả hai vế:

$$
2\alpha_m = \ln\left(\frac{T_w - E_w}{E_w}\right)
$$

$$
\alpha_m = \frac{1}{2} \ln\left(\frac{T_w - E_w}{E_w}\right)
$$

Chúng ta có thể biểu diễn $\alpha_m$ theo tỷ lệ lỗi có trọng số của bộ phân loại yếu $G_m$.
Gọi $error\_rate = \frac{E_w}{T_w}$ là tỷ lệ lỗi có trọng số của $G_m$.

Khi đó, $\frac{T_w - E_w}{E_w} = \frac{T_w/T_w - E_w/T_w}{E_w/T_w} = \frac{1 - error\_rate}{error\_rate}$.

Vậy, công thức cuối cùng cho $\alpha_m$ là:

$$
\alpha_m = \frac{1}{2} \ln\left(\frac{1 - error\_rate}{error\_rate}\right)
$$

### Kết luận

Công thức này cho phép chúng ta tính toán trọng số $\alpha_m$ tối ưu cho mỗi bộ phân loại yếu $G_m$ trong thuật toán AdaBoost. Trọng số này càng lớn khi $G_m$ có tỷ lệ lỗi $error\_rate$ thấp (tức là phân loại tốt), và ngược lại. Điều này đảm bảo rằng các bộ phân loại yếu hoạt động tốt sẽ có ảnh hưởng lớn hơn đến quyết định cuối cùng của bộ phân loại mạnh.

---

## Đạo hàm và Công thức cập nhật trọng số trong AdaBoost

Trong thuật toán AdaBoost, chúng ta cần tìm một giá trị $\alpha_m$ (alpha) để cập nhật trọng số của các mẫu dữ liệu. Giá trị này được tối ưu hóa bằng cách giảm thiểu một hàm lỗi $Err$.

### 1. Hàm lỗi ban đầu

Hàm lỗi $Err$ được định nghĩa như sau:

$$
Err = e^{-\alpha_m(T_w - E_w)} + e^{\alpha_m E_w}
$$

Trong đó:
*   $\alpha_m$ là trọng số của bộ phân loại yếu thứ $m$.
*   $T_w$ là tổng trọng số của tất cả các mẫu dữ liệu.
*   $E_w$ là tổng trọng số của các mẫu dữ liệu bị phân loại sai bởi bộ phân loại yếu thứ $m$.

### 2. Đạo hàm của hàm lỗi theo αm

Để tìm giá trị $\alpha_m$ tối ưu, chúng ta lấy đạo hàm của $Err$ theo $\alpha_m$ và đặt nó bằng 0.

$$
\frac{dErr}{d\alpha_m} = -(T_w - E_w)e^{-\alpha_m(T_w - E_w)} + E_w e^{\alpha_m E_w}
$$

### 3. Giải phương trình để tìm αm

Đặt đạo hàm bằng 0 để tìm điểm cực tiểu:

$$
-(T_w - E_w)e^{-\alpha_m(T_w - E_w)} + E_w e^{\alpha_m E_w} = 0
$$

Từ đây, chúng ta có thể đơn giản hóa các bước như sau:

*   Chuyển vế:
    $$
    E_w e^{\alpha_m E_w} = (T_w - E_w)e^{-\alpha_m(T_w - E_w)}
    $$

*   Chia cả hai vế cho `e^{-\alpha_m(T_w - E_w)}` và `E_w`:
    $$
    \frac{e^{\alpha_m E_w}}{e^{-\alpha_m(T_w - E_w)}} = \frac{T_w - E_w}{E_w}
    $$

*   Sử dụng tính chất $\frac{e^A}{e^B} = e^{A-B}$:
    $$
    e^{\alpha_m E_w + \alpha_m(T_w - E_w)} = \frac{T_w - E_w}{E_w}
    $$

*   Đơn giản hóa số mũ:
    $$
    e^{\alpha_m E_w + \alpha_m T_w - \alpha_m E_w} = \frac{T_w - E_w}{E_w}
    $$
    $$
    e^{\alpha_m T_w} = \frac{T_w - E_w}{E_w}
    $$

*   Lấy logarit tự nhiên cả hai vế:
    $$
    \alpha_m T_w = \ln\left(\frac{T_w - E_w}{E_w}\right)
    $$

*   Chia cả hai vế cho `T_w`:
    $$
    \alpha_m = \frac{1}{T_w} \ln\left(\frac{T_w - E_w}{E_w}\right)
    $$

### 4. Định nghĩa err_m

Trong AdaBoost, `err_m` (tỷ lệ lỗi) được định nghĩa là tổng trọng số của các mẫu bị phân loại sai chia cho tổng trọng số của tất cả các mẫu:

$$
err_m = \frac{E_w}{T_w} = \frac{\sum_{y_i \neq G_m(x_i)} w_i}{\sum w_i}
$$

Trong đó:
*   `E_w` là tổng trọng số của các mẫu bị phân loại sai.
*   `T_w` là tổng trọng số của tất cả các mẫu.
*   `y_i` là nhãn thực của mẫu `i`.
*   `G_m(x_i)` là dự đoán của bộ phân loại yếu thứ `m` cho mẫu `i`.
*   `w_i` là trọng số của mẫu `i`.

### 5. Công thức αm cuối cùng với err_m

Thay $\frac{E_w}{T_w}$ bằng $err_m$ vào công thức của $\alpha_m$:

$$
\alpha_m = \frac{1}{T_w} \ln\left(\frac{T_w - E_w}{E_w}\right)
$$

$$
\alpha_m = \frac{1}{T_w} \ln\left(\frac{T_w - err_m \cdot T_w}{err_m \cdot T_w}\right)
$$

$$
\alpha_m = \frac{1}{T_w} \ln\left(\frac{T_w(1 - err_m)}{err_m \cdot T_w}\right)
$$

$$
\alpha_m = \frac{1}{T_w} \ln\left(\frac{1 - err_m}{err_m}\right)
$$

Tuy nhiên, trong thực tế, công thức chuẩn của AdaBoost thường được viết là:

$$
\alpha_m = \frac{1}{2} \ln\left(\frac{1 - err_m}{err_m}\right)
$$

Sự khác biệt này có thể do cách định nghĩa hàm lỗi hoặc cách chuẩn hóa trọng số trong các phiên bản khác nhau của thuật toán.

### 6. Cập nhật trọng số mẫu

Sau khi tính được $\alpha_m$, chúng ta cập nhật trọng số cho các mẫu dữ liệu:

$$
w_i^{(m+1)} = \frac{w_i^{(m)} \exp(-\alpha_m y_i G_m(x_i))}{Z_m}
$$

Trong đó:
*   `w_i^{(m+1)}` là trọng số mới của mẫu `i` cho vòng lặp tiếp theo.
*   `w_i^{(m)}` là trọng số hiện tại của mẫu `i`.
*   `Z_m` là hệ số chuẩn hóa để đảm bảo tổng trọng số bằng 1:
    $$
    Z_m = \sum_{i=1}^{N} w_i^{(m)} \exp(-\alpha_m y_i G_m(x_i))
    $$

### 7. Công thức cuối cùng cho bộ phân loại mạnh

Bộ phân loại mạnh cuối cùng được kết hợp từ tất cả các bộ phân loại yếu:

$$
H(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m G_m(x)\right)
$$

Trong đó:
*   $H(x)$ là bộ phân loại mạnh cuối cùng.
*   $M$ là tổng số bộ phân loại yếu.
*   $\alpha_m$ là trọng số của bộ phân loại yếu thứ $m$.
*   $G_m(x)$ là bộ phân loại yếu thứ $m$.
*   $\text{sign}(\cdot)$ là hàm dấu, trả về +1 nếu giá trị dương, -1 nếu giá trị âm.

---

## Công thức toán học của AdaBoost

### **Bước 1: Khởi tạo trọng số**

$$
w_i^{(1)} = \frac{1}{N}, \quad i = 1, 2, \ldots, N
$$

### **Bước 2: Vòng lặp cho m = 1 đến M**

#### **2a. Resample Dataset**

Tính cumulative distribution:
$$C_i = \sum_{j=1}^{i} w_j^{(m)}$$

Resample N mẫu:
- Tạo random number $r \sim U(0,1)$
- Chọn mẫu $i$ nếu $C_{i-1} < r \leq C_i$

#### **2b. Huấn luyện weak learner trên dataset đã resample**

$$
G_m(x) = \arg\min_{G} \sum_{i=1}^{N} w_i^{(m)} I(y_i \neq G(x_i))
$$

#### **2c. Tính error rate trên dataset gốc**

$$
err_m = \frac{\sum_{i=1}^{N} w_i^{(m)} I(y_i \neq G_m(x_i))}{\sum_{i=1}^{N} w_i^{(m)}}
$$

#### **2d. Tính trọng số cho weak learner**

$$
\text{weight}_m = \frac{1}{2} \log\left(\frac{1 - err_m}{err_m}\right)
$$

#### **2e. Cập nhật trọng số cho các mẫu**

$$
w_i^{(m+1)} = \frac{w_i^{(m)} \exp(\text{weight}_m \cdot I(y_i \neq G_m(x_i)))}{Z_m}
$$

Trong đó $Z_m$ là normalization factor:

$$
Z_m = \sum_{i=1}^{N} w_i^{(m)} \exp(\text{weight}_m \cdot I(y_i \neq G_m(x_i)))
$$

### **Bước 3: Kết hợp weak learners**

$$
G(x) = \text{sign}\left(\sum_{m=1}^{M} \text{weight}_m G_m(x)\right)
$$

---

## Ví Dụ Tính Tay - AdaBoost Classification

### **Dataset Classification**

| ID | Age (X) | Income (X2) | Label (y) |
|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 |
| 2 | 30 | 50 | 1 |
| 3 | 35 | 40 | 0 |
| 4 | 40 | 60 | 0 |
| 5 | 45 | 70 | 1 |
| 6 | 50 | 80 | 0 |
| 7 | 55 | 90 | 1 |
| 8 | 60 | 100 | 0 |

**Mục tiêu:** Dự đoán label dựa trên Age và Income

---

### **Step 1: Initialization**

**Khởi tạo trọng số cho tất cả mẫu:**

$$
w_i^{(1)} = \frac{1}{N} = \frac{1}{8} = 0.125
$$

| ID | Age | Income | Label | $w_i^{(1)}$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 0.125 |
| 2 | 30 | 50 | 1 | 0.125 |
| 3 | 35 | 40 | 0 | 0.125 |
| 4 | 40 | 60 | 0 | 0.125 |
| 5 | 45 | 70 | 1 | 0.125 |
| 6 | 50 | 80 | 0 | 0.125 |
| 7 | 55 | 90 | 1 | 0.125 |
| 8 | 60 | 100 | 0 | 0.125 |

---

### **Step 2: Iteration 1 (m = 1)**

#### **Step 2a: Resample Dataset**

**Tính cumulative distribution:**

| ID | $w_i^{(1)}$ | Cumulative | Range |
|:---:|:---:|:---:|:---:|
| 1 | 0.125 | 0.125 | (0.000, 0.125] |
| 2 | 0.125 | 0.250 | (0.125, 0.250] |
| 3 | 0.125 | 0.375 | (0.250, 0.375] |
| 4 | 0.125 | 0.500 | (0.375, 0.500] |
| 5 | 0.125 | 0.625 | (0.500, 0.625] |
| 6 | 0.125 | 0.750 | (0.625, 0.750] |
| 7 | 0.125 | 0.875 | (0.750, 0.875] |
| 8 | 0.125 | 1.000 | (0.875, 1.000] |

**Resample 8 mẫu (ví dụ với random numbers):**

| Random | Selected ID | Sample |
|:---:|:---:|:---:|
| 0.05 | 1 | (25, 30, 1) |
| 0.20 | 2 | (30, 50, 1) |
| 0.35 | 3 | (35, 40, 0) |
| 0.45 | 4 | (40, 60, 0) |
| 0.60 | 5 | (45, 70, 1) |
| 0.70 | 6 | (50, 80, 0) |
| 0.80 | 7 | (55, 90, 1) |
| 0.95 | 8 | (60, 100, 0) |

**Dataset sau resample cho Iteration 1:**
- Tất cả mẫu đều xuất hiện 1 lần (trọng số đều)

#### **Step 2b: Train Weak Learner G1(x) trên Resampled Data**

**Tìm threshold tốt nhất cho Age trên resampled data:**

| Threshold | Left (<=) | Right (>) | Error |
|:---:|:---:|:---:|:---:|
| 27.5 | [1] | [2,3,4,5,6,7,8] | 3/8 = 0.375 |
| 32.5 | [1,2] | [3,4,5,6,7,8] | 2/8 = 0.25 |
| 37.5 | [1,2,3] | [4,5,6,7,8] | 3/8 = 0.375 |
| 42.5 | [1,2,3,4] | [5,6,7,8] | 2/8 = 0.25 |
| 47.5 | [1,2,3,4,5] | [6,7,8] | 3/8 = 0.375 |
| 52.5 | [1,2,3,4,5,6] | [7,8] | 2/8 = 0.25 |
| 57.5 | [1,2,3,4,5,6,7] | [8] | 1/8 = 0.125 |

**Best threshold: 57.5 (Error = 0.125)**

**Công thức G1(x):**
- Nếu Age <= 57.5 thì G1(x) = 1
- Nếu Age > 57.5 thì G1(x) = 0

#### **Step 2c: Calculate Error Rate trên Original Data**

$$err_1 = \frac{\sum_{i=1}^{8} w_i^{(1)} I(y_i \neq G_1(x_i))}{\sum_{i=1}^{8} w_i^{(1)}}$$

| ID | Age | Label | $G_1(x)$ | $I(y_i \neq G_1(x_i))$ | $w_i^{(1)}$ | $w_i^{(1)} \cdot I(\cdot)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 1 | 1 | 0 | 0.125 | 0 |
| 2 | 30 | 1 | 1 | 0 | 0.125 | 0 |
| 3 | 35 | 0 | 1 | 1 | 0.125 | 0.125 |
| 4 | 40 | 0 | 1 | 1 | 0.125 | 0.125 |
| 5 | 45 | 1 | 1 | 0 | 0.125 | 0 |
| 6 | 50 | 0 | 1 | 1 | 0.125 | 0.125 |
| 7 | 55 | 1 | 1 | 0 | 0.125 | 0 |
| 8 | 60 | 0 | 0 | 0 | 0.125 | 0 |

$$err_1 = \frac{0 + 0 + 0.125 + 0.125 + 0 + 0.125 + 0 + 0}{1} = 0.375$$

#### **Step 2d: Calculate Weight for G1(x)**

$$\text{weight}_1 = \frac{1}{2} \log\left(\frac{1 - err_1}{err_1}\right) = \frac{1}{2} \log\left(\frac{1 - 0.375}{0.375}\right)$$

$$\text{weight}_1 = \frac{1}{2} \log\left(\frac{0.625}{0.375}\right) = \frac{1}{2} \log(1.667) = \frac{1}{2} \times 0.511 = 0.256$$

#### **Step 2e: Update Sample Weights**

$$w_i^{(2)} = \frac{w_i^{(1)} \exp(\text{weight}_1 \cdot I(y_i \neq G_1(x_i)))}{Z_1}$$

| ID | Age | Label | $G_1(x)$ | $I(y_i \neq G_1(x_i))$ | $\exp(\text{weight}_1 \cdot I(\cdot))$ | $w_i^{(1)} \cdot \exp(\cdot)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 1 | 1 | 0 | 1.000 | 0.125 |
| 2 | 30 | 1 | 1 | 0 | 1.000 | 0.125 |
| 3 | 35 | 0 | 1 | 1 | 1.292 | 0.161 |
| 4 | 40 | 0 | 1 | 1 | 1.292 | 0.161 |
| 5 | 45 | 1 | 1 | 0 | 1.000 | 0.125 |
| 6 | 50 | 0 | 1 | 1 | 1.292 | 0.161 |
| 7 | 55 | 1 | 1 | 0 | 1.000 | 0.125 |
| 8 | 60 | 0 | 0 | 0 | 1.000 | 0.125 |

$$Z_1 = 0.125 + 0.125 + 0.161 + 0.161 + 0.125 + 0.161 + 0.125 + 0.125 = 1.108$$

| ID | $w_i^{(2)}$ |
|:---:|:---:|
| 1 | 0.125/1.108 = 0.113 |
| 2 | 0.125/1.108 = 0.113 |
| 3 | 0.161/1.108 = 0.145 |
| 4 | 0.161/1.108 = 0.145 |
| 5 | 0.125/1.108 = 0.113 |
| 6 | 0.161/1.108 = 0.145 |
| 7 | 0.125/1.108 = 0.113 |
| 8 | 0.125/1.108 = 0.113 |

---

### **Step 3: Iteration 2 (m = 2)**

#### **Step 3a: Resample Dataset cho Iteration 2**

**Tính cumulative distribution:**

| ID | $w_i^{(2)}$ | Cumulative | Range |
|:---:|:---:|:---:|:---:|
| 1 | 0.113 | 0.113 | (0.000, 0.113] |
| 2 | 0.113 | 0.226 | (0.113, 0.226] |
| 3 | 0.145 | 0.371 | (0.226, 0.371] |
| 4 | 0.145 | 0.516 | (0.371, 0.516] |
| 5 | 0.113 | 0.629 | (0.516, 0.629] |
| 6 | 0.145 | 0.774 | (0.629, 0.774] |
| 7 | 0.113 | 0.887 | (0.774, 0.887] |
| 8 | 0.113 | 1.000 | (0.887, 1.000] |

**Resample 8 mẫu (ví dụ với random numbers):**

| Random | Selected ID | Sample | Count |
|:---:|:---:|:---:|:---:|
| 0.05 | 1 | (25, 30, 1) | 1 |
| 0.15 | 2 | (30, 50, 1) | 1 |
| 0.30 | 3 | (35, 40, 0) | 1 |
| 0.45 | 4 | (40, 60, 0) | 1 |
| 0.60 | 5 | (45, 70, 1) | 1 |
| 0.70 | 6 | (50, 80, 0) | 1 |
| 0.80 | 7 | (55, 90, 1) | 1 |
| 0.95 | 8 | (60, 100, 0) | 1 |

**Dataset sau resample cho Iteration 2:**
- Tất cả mẫu đều xuất hiện 1 lần (do trọng số chưa chênh lệch nhiều)

#### **Step 3b: Train Weak Learner G2(x) trên Resampled Data**

**Tìm threshold tốt nhất cho Income trên resampled data:**

| Threshold | Left (<=) | Right (>) | Error |
|:---:|:---:|:---:|:---:|
| 35 | [1] | [2,3,4,5,6,7,8] | 3/8 = 0.375 |
| 45 | [1,2,3] | [4,5,6,7,8] | 2/8 = 0.25 |
| 55 | [1,2,3,4] | [5,6,7,8] | 2/8 = 0.25 |
| 65 | [1,2,3,4,5] | [6,7,8] | 3/8 = 0.375 |
| 75 | [1,2,3,4,5,6] | [7,8] | 2/8 = 0.25 |
| 85 | [1,2,3,4,5,6,7] | [8] | 1/8 = 0.125 |
| 95 | [1,2,3,4,5,6,7,8] | [] | 0/8 = 0 |

**Best threshold: 85 (Error = 0.125)**

**Công thức G2(x):**
- Nếu Income <= 85 thì G2(x) = 1
- Nếu Income > 85 thì G2(x) = 0

#### **Step 3c: Calculate Error Rate trên Original Data**

$$err_2 = \frac{\sum_{i=1}^{8} w_i^{(2)} I(y_i \neq G_2(x_i))}{\sum_{i=1}^{8} w_i^{(2)}}$$

| ID | Income | Label | $G_2(x)$ | $I(y_i \neq G_2(x_i))$ | $w_i^{(2)}$ | $w_i^{(2)} \cdot I(\cdot)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 30 | 1 | 1 | 0 | 0.113 | 0 |
| 2 | 50 | 1 | 1 | 0 | 0.113 | 0 |
| 3 | 40 | 0 | 1 | 1 | 0.145 | 0.145 |
| 4 | 60 | 0 | 1 | 1 | 0.145 | 0.145 |
| 5 | 70 | 1 | 1 | 0 | 0.113 | 0 |
| 6 | 80 | 0 | 1 | 1 | 0.145 | 0.145 |
| 7 | 90 | 1 | 0 | 1 | 0.113 | 0.113 |
| 8 | 100 | 0 | 0 | 0 | 0.113 | 0 |

$$err_2 = \frac{0 + 0 + 0.145 + 0.145 + 0 + 0.145 + 0.113 + 0}{1} = 0.548$$

#### **Step 3d: Calculate Weight for G2(x)**

$$\text{weight}_2 = \frac{1}{2} \log\left(\frac{1 - err_2}{err_2}\right) = \frac{1}{2} \log\left(\frac{1 - 0.548}{0.548}\right)$$

$$\text{weight}_2 = \frac{1}{2} \log\left(\frac{0.452}{0.548}\right) = \frac{1}{2} \log(0.825) = \frac{1}{2} \times (-0.192) = -0.096$$

#### **Step 3e: Update Sample Weights**

$$w_i^{(3)} = \frac{w_i^{(2)} \exp(\text{weight}_2 \cdot I(y_i \neq G_2(x_i)))}{Z_2}$$

| ID | Income | Label | $G_2(x)$ | $I(y_i \neq G_2(x_i))$ | $\exp(\text{weight}_2 \cdot I(\cdot))$ | $w_i^{(2)} \cdot \exp(\cdot)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 30 | 1 | 1 | 0 | 1.000 | 0.113 |
| 2 | 50 | 1 | 1 | 0 | 1.000 | 0.113 |
| 3 | 40 | 0 | 1 | 1 | 0.908 | 0.132 |
| 4 | 60 | 0 | 1 | 1 | 0.908 | 0.132 |
| 5 | 70 | 1 | 1 | 0 | 1.000 | 0.113 |
| 6 | 80 | 0 | 1 | 1 | 0.908 | 0.132 |
| 7 | 90 | 1 | 0 | 1 | 0.908 | 0.103 |
| 8 | 100 | 0 | 0 | 0 | 1.000 | 0.113 |

$$Z_2 = 0.113 + 0.113 + 0.132 + 0.132 + 0.113 + 0.132 + 0.103 + 0.113 = 0.951$$

| ID | $w_i^{(3)}$ |
|:---:|:---:|
| 1 | 0.113/0.951 = 0.119 |
| 2 | 0.113/0.951 = 0.119 |
| 3 | 0.132/0.951 = 0.139 |
| 4 | 0.132/0.951 = 0.139 |
| 5 | 0.113/0.951 = 0.119 |
| 6 | 0.132/0.951 = 0.139 |
| 7 | 0.103/0.951 = 0.108 |
| 8 | 0.113/0.951 = 0.119 |

---

### **Step 4: Final Model**

**Kết hợp weak learners:**

$$G(x) = \text{sign}\left(\sum_{m=1}^{2} \text{weight}_m G_m(x)\right)$$

$$G(x) = \text{sign}(0.256 \cdot G_1(x) + (-0.096) \cdot G_2(x))$$

**Dự đoán cho từng mẫu:**

| ID | Age | Income | $G_1(x)$ | $G_2(x)$ | $\text{weight}_1 G_1(x)$ | $\text{weight}_2 G_2(x)$ | Sum | $G(x)$ | True Label |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25 | 30 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 1 ✓ |
| 2 | 30 | 50 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 1 ✓ |
| 3 | 35 | 40 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 0 ✗ |
| 4 | 40 | 60 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 0 ✗ |
| 5 | 45 | 70 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 1 ✓ |
| 6 | 50 | 80 | 1 | 1 | 0.256 | -0.096 | 0.160 | 1 | 0 ✗ |
| 7 | 55 | 90 | 1 | 0 | 0.256 | 0 | 0.256 | 1 | 1 ✓ |
| 8 | 60 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 ✓ |

**Accuracy: 5/8 = 62.5%**

---

## **Tóm tắt AdaBoost với Resampling**

AdaBoost hoạt động bằng cách:

1. **Khởi tạo** trọng số đều cho tất cả mẫu
2. **Lặp lại** cho mỗi weak learner:
   - **Resample** dataset dựa trên trọng số hiện tại
   - Huấn luyện weak learner trên dataset đã resample
   - Tính error rate trên dataset gốc
   - Tính trọng số $\text{weight}_m$ cho weak learner
   - Tăng trọng số cho các mẫu bị phân loại sai
3. **Kết hợp** tất cả weak learners với trọng số tương ứng

**Ưu điểm của Resampling:**
- Tập trung vào các mẫu khó phân loại
- Tạo đa dạng cho các weak learners
- Cải thiện hiệu suất tổng thể

**Nhược điểm:**
- Có thể mất thông tin từ các mẫu ít quan trọng
- Tăng thời gian tính toán
- Có thể gây overfitting nếu có quá nhiều iterations

