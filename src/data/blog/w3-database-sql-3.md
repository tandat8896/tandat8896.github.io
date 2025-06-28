---
title: "Hành Trình Chinh Phục SQL Nâng Cao: Từ Lý Thuyết Đến Thực Hành"
description: "Chia sẻ kinh nghiệm học SQL nâng cao với các kỹ thuật Subqueries, CTE, Stored Procedures, Triggers và ứng dụng thực tế trong phân tích dữ liệu."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - sql
  - database
  - week3
  - advanced-sql
draft: false
---

# Hành Trình Chinh Phục SQL Nâng Cao: Từ Lý Thuyết Đến Thực Hành

Hôm nay tôi muốn chia sẻ về hành trình của mình trong việc học SQL, đặc biệt là cách áp dụng nó vào phân tích dữ liệu. Hôm trước chúng ta đã học SQL về tạo bảng và truy vấn từ các câu lệnh cơ bản đến những kỹ thuật nâng cao. Đây là một cái nhìn tổng quan về những gì tôi đã học được và cách tôi đã thực hành để nắm vững chúng.

## 📊 Cơ Sở Dữ Liệu Mẫu

Trước khi đi vào chi tiết, hãy làm quen với cấu trúc cơ sở dữ liệu mà tôi sử dụng trong các ví dụ:

### Bảng Employees (Nhân viên)
| employee_id | first_name | last_name | email | hire_date | salary | manager_id | office_id |
|-------------|------------|-----------|-------|-----------|--------|------------|-----------|
| 1 | John | Smith | john.smith@company.com | 2020-01-15 | 75000 | NULL | 1 |
| 2 | Sarah | Johnson | sarah.j@company.com | 2020-03-20 | 65000 | 1 | 1 |
| 3 | Mike | Davis | mike.d@company.com | 2021-06-10 | 55000 | 1 | 2 |
| 4 | Lisa | Wilson | lisa.w@company.com | 2021-08-05 | 60000 | 2 | 1 |
| 5 | David | Brown | david.b@company.com | 2022-01-12 | 70000 | 1 | 3 |

### Bảng Orders (Đơn hàng)
| order_id | customer_id | order_date | total_amount | status |
|----------|-------------|------------|--------------|--------|
| 1 | 101 | 2024-01-15 | 1250.00 | delivered |
| 2 | 102 | 2024-01-20 | 890.50 | shipped |
| 3 | 101 | 2024-02-05 | 2100.00 | delivered |
| 4 | 103 | 2024-02-10 | 450.75 | pending |
| 5 | 102 | 2024-03-01 | 1750.25 | processing |

### Bảng Products (Sản phẩm)
| product_id | name | description | price | stock_quantity |
|------------|------|-------------|-------|----------------|
| 1 | Laptop Pro | High-performance laptop | 1200.00 | 50 |
| 2 | Wireless Mouse | Ergonomic wireless mouse | 45.00 | 200 |
| 3 | Monitor 24" | 24-inch HD monitor | 300.00 | 75 |
| 4 | Keyboard | Mechanical keyboard | 120.00 | 100 |
| 5 | Webcam | HD webcam with microphone | 80.00 | 150 |

## 1. 🎯 Nắm Vững Các Kỹ Thuật Truy Vấn Cơ Bản và Nâng Cao

### 1.1 SQL Subqueries (Truy vấn con)

**🔍 Lý thuyết:**
Truy vấn con là một câu lệnh SELECT được lồng bên trong một câu lệnh SQL khác. Chúng giúp chia nhỏ vấn đề phức tạp thành các phần nhỏ hơn và dễ quản lý. Có 3 loại subquery chính:

- **Scalar Subquery**: Trả về một giá trị duy nhất
- **Column Subquery**: Trả về một cột dữ liệu
- **Table Subquery**: Trả về một bảng dữ liệu

**💡 Ưu điểm:**
- Tách biệt logic phức tạp
- Dễ đọc và bảo trì
- Có thể tái sử dụng

**⚠️ Nhược điểm:**
- Có thể ảnh hưởng đến hiệu suất
- Khó debug khi phức tạp

**🚀 Use Case 1: Tìm nhân viên có lương cao hơn quản lý**

```sql
SELECT 
    e.first_name,
    e.last_name,
    e.salary,
    m.first_name AS manager_name,
    m.salary AS manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.employee_id
WHERE e.salary > (
    SELECT salary 
    FROM employees 
    WHERE employee_id = e.manager_id
);
```

**Kết quả:**
| first_name | last_name | salary | manager_name | manager_salary |
|------------|-----------|--------|--------------|----------------|
| David | Brown | 70000 | John | 75000 |
| Lisa | Wilson | 60000 | Sarah | 65000 |

**🚀 Use Case 2: Đơn hàng có giá trị cao hơn trung bình**

```sql
SELECT 
    order_id,
    customer_id,
    total_amount,
    (SELECT AVG(total_amount) FROM orders) as avg_order_value
FROM orders
WHERE total_amount > (
    SELECT AVG(total_amount) 
    FROM orders
);
```

### 1.2 Common Table Expressions (CTE)

**🔍 Lý thuyết:**
CTE là một bảng tạm thời được định nghĩa trong phạm vi của một câu lệnh SQL. CTE sử dụng từ khóa `WITH` và có thể được tham chiếu nhiều lần trong cùng một truy vấn.

**💡 Ưu điểm:**
- Mã sạch sẽ, dễ đọc
- Có thể tái sử dụng trong cùng truy vấn
- Hiệu suất tốt hơn subquery phức tạp

**🚀 Use Case 1: Phân tích sản phẩm bán chạy nhất**

```sql
WITH ProductOrderCount AS (
    SELECT 
        product_id,
        COUNT(*) as order_count,
        SUM(quantity) as total_quantity
    FROM order_items
    GROUP BY product_id
),
ProductRanking AS (
    SELECT 
        p.name,
        poc.order_count,
        poc.total_quantity,
        RANK() OVER (ORDER BY poc.order_count DESC) as rank
    FROM products p
    JOIN ProductOrderCount poc ON p.product_id = poc.product_id
)
SELECT * FROM ProductRanking WHERE rank <= 3;
```

**🚀 Use Case 2: Phân tích lương theo văn phòng**

```sql
WITH OfficeStats AS (
    SELECT 
        office_id,
        COUNT(*) as employee_count,
        AVG(salary) as avg_salary,
        MIN(salary) as min_salary,
        MAX(salary) as max_salary
    FROM employees
    GROUP BY office_id
)
SELECT 
    o.name as office_name,
    os.employee_count,
    ROUND(os.avg_salary, 2) as avg_salary,
    os.min_salary,
    os.max_salary
FROM offices o
JOIN OfficeStats os ON o.office_id = os.office_id
ORDER BY os.avg_salary DESC;
```

### 1.3 SQL Temp Table (Bảng tạm thời)

**🔍 Lý thuyết:**
Bảng tạm thời chỉ tồn tại trong phiên làm việc hiện tại và sẽ mất đi khi đóng phiên. Khác với CTE và Subquery, bảng tạm thời có thể được sử dụng lại trong cùng một phiên và có thể có index.

**💡 Ưu điểm:**
- Có thể tái sử dụng nhiều lần
- Có thể tạo index để tối ưu hiệu suất
- Phù hợp cho dữ liệu lớn

**⚠️ Nhược điểm:**
- Chiếm bộ nhớ tạm thời
- Chỉ tồn tại trong phiên hiện tại

**🚀 Use Case 1: Phân tích khách hàng VIP**

```sql
-- Tạo bảng tạm thời chứa thống kê khách hàng
CREATE TEMPORARY TABLE customer_analysis AS
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;

-- Phân tích khách hàng VIP
SELECT 
    first_name,
    last_name,
    total_orders,
    total_spent,
    CASE 
        WHEN total_spent > 2000 THEN 'VIP'
        WHEN total_spent > 1000 THEN 'Regular'
        ELSE 'New'
    END as customer_type
FROM customer_analysis
WHERE total_spent > 0
ORDER BY total_spent DESC;
```

## 2. ⚡ Tối Ưu Hóa và Tự Động Hóa với Stored Procedures và Triggers

### 2.1 SQL Stored Procedures (Thủ tục lưu trữ)

**🔍 Lý thuyết:**
Stored Procedures là các khối mã SQL được lưu trữ trong database và có thể được gọi lại nhiều lần. Chúng giúp tạo ra các khối mã có thể tái sử dụng và thực thi nhanh chóng.

**💡 Ưu điểm:**
- Tái sử dụng code
- Bảo mật cao
- Hiệu suất tốt
- Dễ bảo trì

**🚀 Use Case 1: Quy trình xử lý thanh toán**

```sql
DELIMITER //
CREATE PROCEDURE ProcessPayment(
    IN p_invoice_id INT,
    IN p_amount DECIMAL(10,2),
    IN p_payment_date DATE,
    OUT p_status VARCHAR(50)
)
BEGIN
    DECLARE current_balance DECIMAL(10,2);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET p_status = 'ERROR: Transaction failed';
    END;
    
    -- Bắt đầu transaction
    START TRANSACTION;
    
    -- Kiểm tra số dư hiện tại
    SELECT balance INTO current_balance 
    FROM invoices 
    WHERE invoice_id = p_invoice_id;
    
    -- Kiểm tra số tiền thanh toán
    IF p_amount > current_balance THEN
        SET p_status = 'ERROR: Payment amount exceeds balance';
        ROLLBACK;
    ELSE
        -- Chèn bản ghi thanh toán
        INSERT INTO payments (invoice_id, amount, payment_date)
        VALUES (p_invoice_id, p_amount, p_payment_date);
        
        -- Cập nhật số dư trong hóa đơn
        UPDATE invoices 
        SET balance = balance - p_amount
        WHERE invoice_id = p_invoice_id;
        
        -- Commit transaction
        COMMIT;
        SET p_status = 'SUCCESS: Payment processed';
    END IF;
END //
DELIMITER ;

-- Sử dụng stored procedure
CALL ProcessPayment(1, 500.00, '2025-06-28', @status);
SELECT @status;
```

**🚀 Use Case 2: Báo cáo doanh thu theo tháng**

```sql
DELIMITER //
CREATE PROCEDURE GetMonthlyRevenue(
    IN p_year INT,
    IN p_month INT
)
BEGIN
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') as month,
        COUNT(*) as total_orders,
        SUM(total_amount) as total_revenue,
        AVG(total_amount) as avg_order_value
    FROM orders
    WHERE YEAR(order_date) = p_year 
        AND (p_month IS NULL OR MONTH(order_date) = p_month)
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
    ORDER BY month;
END //
DELIMITER ;

-- Sử dụng stored procedure
CALL GetMonthlyRevenue(2024, NULL); -- Tất cả tháng trong năm 2024
CALL GetMonthlyRevenue(2024, 1);    -- Chỉ tháng 1 năm 2024
```

### 2.2 SQL Trigger (Bộ kích hoạt)

**🔍 Lý thuyết:**
Triggers là các tập hợp câu lệnh SQL chạy mỗi khi một hàng được chèn, cập nhật hoặc xóa trong một bảng. Chúng rất hữu ích cho việc tự động hóa các quy tắc nghiệp vụ và duy trì tính toàn vẹn của dữ liệu.

**💡 Loại Trigger:**
- **BEFORE INSERT/UPDATE/DELETE**: Chạy trước khi thực hiện thao tác
- **AFTER INSERT/UPDATE/DELETE**: Chạy sau khi thực hiện thao tác
- **INSTEAD OF**: Thay thế thao tác gốc

**🚀 Use Case 1: Tự động ghi log khi thêm nhân viên mới**

```sql
DELIMITER //
CREATE TRIGGER after_employee_insert
AFTER INSERT ON employees
FOR EACH ROW
BEGIN
    INSERT INTO employee_audit_log (
        employee_id,
        action_type,
        action_date,
        old_values,
        new_values,
        user_id
    ) VALUES (
        NEW.employee_id,
        'INSERT',
        NOW(),
        NULL,
        JSON_OBJECT(
            'first_name', NEW.first_name,
            'last_name', NEW.last_name,
            'email', NEW.email,
            'salary', NEW.salary
        ),
        USER()
    );
END //
DELIMITER ;

-- Test trigger
INSERT INTO employees (first_name, last_name, email, hire_date, salary)
VALUES ('Alice', 'Johnson', 'alice.j@company.com', '2025-06-28', 65000);
```

**🚀 Use Case 2: Tự động cập nhật tổng số đơn hàng của khách hàng**

```sql
DELIMITER //
CREATE TRIGGER after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE customers 
    SET total_orders = (
        SELECT COUNT(*) 
        FROM orders 
        WHERE customer_id = NEW.customer_id
    ),
    last_order_date = NEW.order_date
    WHERE customer_id = NEW.customer_id;
END //
DELIMITER ;

-- Test trigger
INSERT INTO orders (customer_id, order_date, total_amount, status)
VALUES (101, '2025-06-28', 850.00, 'pending');
```

## 3. 🔗 Khám Phá Các Khái Niệm Nâng Cao và Mô Hình Hóa Dữ Liệu

### 3.1 Self Join

**🔍 Lý thuyết:**
Self Join là việc join một bảng với chính nó để truy vấn thông tin liên quan trong cùng một bảng. Thường được sử dụng cho dữ liệu có cấu trúc phân cấp.

**🚀 Use Case: Phân tích cấu trúc quản lý**

```sql
SELECT 
    e.employee_id,
    e.first_name AS employee_name,
    e.last_name AS employee_lastname,
    e.salary AS employee_salary,
    m.first_name AS manager_name,
    m.last_name AS manager_lastname,
    m.salary AS manager_salary,
    CASE 
        WHEN e.salary > m.salary THEN 'Higher than manager'
        WHEN e.salary = m.salary THEN 'Equal to manager'
        ELSE 'Lower than manager'
    END as salary_comparison
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id
ORDER BY e.last_name, e.first_name;
```

### 3.2 Union và Union All

**🔍 Lý thuyết:**
- **UNION**: Kết hợp kết quả từ nhiều câu lệnh SELECT và loại bỏ duplicate
- **UNION ALL**: Kết hợp kết quả từ nhiều câu lệnh SELECT và giữ lại duplicate

**🚀 Use Case: Báo cáo tổng hợp khách hàng và nhà cung cấp**

```sql
SELECT 
    'CUSTOMER' as contact_type,
    first_name,
    last_name,
    email,
    'Active' as status
FROM customers
WHERE email IS NOT NULL

UNION

SELECT 
    'SUPPLIER' as contact_type,
    first_name,
    last_name,
    email,
    CASE 
        WHEN last_order_date > DATE_SUB(NOW(), INTERVAL 1 YEAR) THEN 'Active'
        ELSE 'Inactive'
    END as status
FROM suppliers
WHERE email IS NOT NULL

ORDER BY contact_type, last_name, first_name;
```

### 3.3 Recursive CTE (WITH RECURSIVE)

**🔍 Lý thuyết:**
CTE đệ quy cho phép xử lý các cấu trúc dữ liệu phân cấp. Sử dụng `WITH RECURSIVE` để định nghĩa base case và recursive case.

**🚀 Use Case: Phân tích cấu trúc tổ chức**

```sql
WITH RECURSIVE EmployeeHierarchy AS (
    -- Base case: CEO (không có manager)
    SELECT 
        employee_id,
        first_name,
        last_name,
        manager_id,
        0 as level,
        CAST(first_name AS CHAR(1000)) as hierarchy_path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: tìm tất cả nhân viên dưới quyền
    SELECT 
        e.employee_id,
        e.first_name,
        e.last_name,
        e.manager_id,
        eh.level + 1,
        CONCAT(eh.hierarchy_path, ' > ', e.first_name) as hierarchy_path
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
    WHERE eh.level < 5  -- Giới hạn độ sâu để tránh vòng lặp vô hạn
)
SELECT 
    level,
    first_name,
    last_name,
    hierarchy_path,
    salary
FROM EmployeeHierarchy
ORDER BY level, last_name, first_name;
```



## 🎯 Lời Kết

Hành trình học SQL này không chỉ trang bị cho tôi những kỹ năng kỹ thuật mà còn mở ra một cái nhìn sâu sắc về cách dữ liệu được tổ chức, truy vấn và quản lý. Với những kiến thức về Subqueries, CTEs, Temp Tables, Stored Procedures, Triggers và cả ERD, tôi cảm thấy tự tin hơn rất nhiều khi làm việc với dữ liệu.

### 📋 Tóm tắt những gì đã học:

1. **Subqueries**: Giải quyết vấn đề phức tạp bằng cách chia nhỏ
2. **CTE**: Viết mã sạch sẽ và dễ đọc hơn
3. **Temp Tables**: Xử lý dữ liệu lớn hiệu quả
4. **Stored Procedures**: Tự động hóa quy trình nghiệp vụ
5. **Triggers**: Duy trì tính toàn vẹn dữ liệu
6. **Self Join**: Xử lý dữ liệu phân cấp
7. **Recursive CTE**: Giải quyết bài toán đệ quy

Những kỹ thuật này không chỉ giúp tôi viết các truy vấn hiệu quả hơn mà còn mở ra cánh cửa để tôi có thể tự động hóa các quy trình nghiệp vụ và xây dựng các hệ thống quản lý dữ liệu mạnh mẽ.

Cảm ơn các bạn đã đọc! Hãy cùng nhau tiếp tục khám phá thế giới dữ liệu nhé! 🚀

---
