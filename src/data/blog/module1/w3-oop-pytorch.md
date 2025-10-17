---
title: "Hành Trình Khám Phá OOP và Cấu Trúc Dữ Liệu: Từ Stack, Queue đến Binary Search Tree"
description: "Bài viết chuyên sâu về Object-Oriented Programming và cấu trúc dữ liệu nâng cao: Stack, Queue, Tree, Binary Search Tree, K-D Tree, và ứng dụng trong AI/Machine Learning."
pubDatetime: 2025-06-28T10:00:00Z
tags:
  - python
  - oop
  - data-structure
  - week3
  - binary-tree
  - bst
  - k-d-tree
draft: false
---

# Week 3: Object-Oriented Programming & Data Structures - Hành Trình Khám Phá Cấu Trúc Dữ Liệu và OOP

##  Tổng Quan Tuần Học Tập

Tuần 3 đánh dấu một bước ngoặt quan trọng trong hành trình học tập của tôi khi tôi được tiếp cận với **Object-Oriented Programming (OOP)** và các **cấu trúc dữ liệu nâng cao**. Đây không chỉ là việc học lý thuyết mà còn là quá trình thực hành sâu sắc, từ những cấu trúc cơ bản như Stack, Queue đến những cấu trúc phức tạp như Binary Search Tree và K-D Tree.

##  Phần 1: Khởi Đầu với Stack và Queue

### Stack - Nguyên Lý LIFO

Tôi bắt đầu với **Stack** - một cấu trúc dữ liệu tuyến tính với nguyên lý **LIFO (Last-In, First-Out)**. Điều này có nghĩa là phần tử được thêm vào cuối cùng sẽ là phần tử đầu tiên được lấy ra.

```python
class MyStack:
    def __init__(self, capacity) -> None:
        self.__capacity = capacity
        self.__stack = []
    
    def is_full(self):
        return len(self.__stack) == self.__capacity
    
    def push(self, value):
        if not self.is_full():
            self.__stack.append(value)  # Thêm vào cuối
    
    def pop(self):
        if not self.is_empty():
            return self.__stack.pop(-1)  # Lấy từ cuối
```

**Những gì tôi học được:**
- **PUSH**: Thêm phần tử vào cuối stack
- **POP**: Lấy và xóa phần tử từ cuối stack
- **Triển khai bằng List**: Sử dụng `append()` và `pop(-1)`

### Queue - Nguyên Lý FIFO

Tiếp theo là **Queue** với nguyên tắc **FIFO (First-In, First-Out)**, nơi các phần tử được chèn và trích xuất tại hai đầu đối diện.

```python
class MyQueue:
    def __init__(self):
        self.__queue = []
    
    def enqueue(self, value):
        self.__queue.append(value)  # Thêm vào cuối
    
    def dequeue(self):
        if not self.is_empty():
            return self.__queue.pop(0)  # Lấy từ đầu
```

**Các thao tác cơ bản:**
- **enqueue**: Thêm phần tử vào cuối
- **dequeue**: Xóa phần tử từ đầu
- **is_empty**: Kiểm tra queue rỗng
- **peek**: Xem giá trị phần tử đầu mà không xóa

## 🌳 Phần 2: Khám Phá Cây (Tree) và Các Thuật Ngữ Cơ Bản

### Định Nghĩa và Thuật Ngữ

Cây là một cấu trúc dữ liệu **phi tuyến tính** nơi các node được tổ chức theo phân cấp. Tôi đã học được các thuật ngữ quan trọng:

- **Root Node**: Node gốc (không có node cha)
- **Parent Node**: Node cha
- **Child Node**: Node con
- **Leaf Node**: Node lá (không có node con)
- **Ancestor**: Tổ tiên của một node
- **Sibling**: Anh chị em (cùng cha)
- **Level/Depth**: Mức độ của node (số cạnh từ gốc đến node)
- **Height**: Chiều cao (đường đi dài nhất từ node đến node lá)

### Triển Khai Cây Cơ Bản

```python
class TreeNode:
    def __init__(self, data) -> None:
        self.data = data
        self.parent = None
        self.children = []
    
    def add_child(self, child):
        child.parent = self  # Liên kết đối tượng, không phải kế thừa!
        self.children.append(child)
    
    def get_level(self):
        level = 0 
        p = self.parent
        while p:
            level += 1 
            p = p.parent
        return level 
    
    def print_tree(self):
        space = ' ' * self.get_level() * 3 
        prefix = space + '|__' if self.parent else ''
        print(f"{prefix}{self.data}")
        if self.children:
            for child in self.children:
                child.print_tree()
```

**Kết quả khi chạy:**
```
A
   |__B
      |__D
      |__E
   |__C
      |__F
      |__G
```

## 🔍 Phần 3: Cây Nhị Phân (Binary Tree) và Thuật Toán

### Định Nghĩa và Phân Loại

Cây nhị phân là cây mà mỗi node có tối đa **hai node con** (con trái và con phải).

**Các loại cây nhị phân:**
- **Left Skew Tree**: Tất cả node con đều ở bên trái
- **Right Skew Tree**: Tất cả node con đều ở bên phải
- **Full Binary Tree**: Mỗi node có 0 hoặc 2 con
- **Balanced Binary Tree**: Cân bằng chiều cao
- **Unbalanced Binary Tree**: Không cân bằng

### Triển Khai Cây Nhị Phân

```python
class TreeNode:
    def __init__(self, key) -> None:
        self.left = None 
        self.right = None
        self.val = key

# Tạo cây nhị phân
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
```

### Duyệt Cây (Traversal)

Tôi đã học hai phương pháp duyệt chính:

#### 1. Depth First Search (DFS)
- **Inorder**: Trái → Gốc → Phải
- **Preorder**: Gốc → Trái → Phải  
- **Postorder**: Trái → Phải → Gốc

#### 2. Breadth First Search (BFS)
- Duyệt theo từng cấp độ
- Sử dụng Queue để lưu trữ các node

```python
from collections import deque

def bfs_traversal(root):
    if not root:
        return
    
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val, end=" ")
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

### Thao Tác Cơ Bản

#### 1. Chèn Node (Insert)
Sử dụng duyệt theo cấp độ (Level-Order Traversal) với Queue:

```python
def insert_node(root, key):
    if root is None:
        return TreeNode(key)
    
    queue = [root]
    while queue:
        temp = queue.pop(0)
        
        if temp.right is None:
            temp.right = TreeNode(key)
            return root
        else:
            queue.append(temp.right)
            
        if temp.left is None:
            temp.left = TreeNode(key)
            return root
        else:
            queue.append(temp.left)
    
    return root
```

#### 2. Xóa Node (Delete)
Thuật toán phức tạp hơn, đòi hỏi thay thế node cần xóa bằng node sâu nhất bên phải.

## 🎯 Phần 4: Binary Search Tree (BST) - Tối Ưu Hóa Tìm Kiếm

### Vấn Đề với Cây Nhị Phân Thông Thường

Tôi nhận ra rằng cây nhị phân tổng quát có hạn chế: **không đảm bảo tìm kiếm nhanh**. Không có quy tắc nào để quyết định nên duyệt nhánh trái hay phải.

### Giải Pháp: BST

BST khắc phục điều này bằng quy tắc nghiêm ngặt:
> "Với mỗi node x trong BST, tất cả các node trong cây con bên trái của x chứa giá trị nhỏ hơn x, và tất cả các node trong cây con bên phải của x chứa giá trị lớn hơn hoặc bằng x"

```python
def lookup(tree, key):
    if tree is None:
        return None
    if key == tree.key:
        return tree
    elif key < tree.key:
        return lookup(tree.left, key)  # Đi sang trái
    else:
        return lookup(tree.right, key)  # Đi sang phải
```

### Thuật Toán Xóa Node trong BST

Đây là phần phức tạp nhất mà tôi đã học:

**Trường hợp 1**: Node lá → Xóa trực tiếp
**Trường hợp 2**: Node có 1 con → Thay thế bằng con
**Trường hợp 3**: Node có 2 con → Thay thế bằng **successor** (node nhỏ nhất bên phải) hoặc **predecessor** (node lớn nhất bên trái)

> "Nếu xóa 30 thì chèn 40 lên" - Đây là nguyên tắc thay thế để duy trì trật tự BST.

## 🌐 Phần 5: K-D Tree - Giải Pháp Cho Bài Toán Phân Loại

### Vấn Đề với Phương Pháp Truyền Thống

Tôi nhận ra rằng các phương pháp tìm kiếm truyền thống như List, Array, Dict thường là **thuật toán quét cạn (brute force)** và kém hiệu quả cho bộ dữ liệu lớn.

### K-D Tree: Giải Pháp Hiệu Quả

K-D Tree là cây nhị phân mà mỗi node đại diện cho một điểm k chiều, được thiết kế đặc biệt cho:
- **Classification Problem**
- **K-nearest neighbors (KNN)**
- **Tìm kiếm không gian đa chiều**

**Quá trình xây dựng:**
1. Luân phiên qua các chiều
2. Chọn trung vị để chia tập dữ liệu
3. Tạo hai phân vùng xấp xỉ bằng nhau

## 🔧 Phần 6: Trực Quan Hóa với Graphviz

Tôi đã học cách tạo biểu diễn trực quan của cây bằng Graphviz:

```python
from graphviz import Graph

def add_edges(dot, node):
    if node is None:
        return
    if node.left:
        dot.edge(str(node.val), str(node.left.val))
        add_edges(dot, node.left)
    if node.right:
        dot.edge(str(node.val), str(node.right.val))
        add_edges(dot, node.right)

def draw_tree(root):
    dot = Graph(name="MyBinaryTree", filename="binary_tree.dot", format="png")
    dot.node(str(root.val))
    add_edges(dot, root)
    return dot
```

## 🧠 Phong Cách Học Tập và Tư Duy Phản Biện

### Đặc Điểm Nổi Bật

1. **Phân tích hiệu suất**: Tôi không chỉ học định nghĩa mà còn đánh giá hiệu quả của các cấu trúc dữ liệu
2. **Nghi vấn và mở rộng tư duy**: Đặt câu hỏi "Tại sao lại nối từ A sang E?"
3. **Đi sâu vào triển khai**: Chú trọng cách các cấu trúc được hiện thực hóa trong OOP
4. **Tự phản tư**: "minh nghi la ntn... neu co sai sot thi cac ban va TA chi giup a"
5. **Thực hành thử nghiệm**: "thêm dòng print self, print child trong code để dễ nhìn nha"

### Những Khám Phá Quan Trọng

- **Liên kết đối tượng vs Kế thừa**: `child.parent = self` không phải là kế thừa mà là liên kết đối tượng (association)
- **Truthy và Falsy trong Python**: Củng cố kiến thức lập trình cơ bản
- **Sự khác biệt Height vs Depth**: 
  - Depth: Từ node gốc đến node hiện tại
  - Height: Từ node hiện tại đến node lá

## 🎯 Lợi Ích và Ứng Dụng Thực Tế

### Lợi Ích Đạt Được

1. **Nâng cao hiệu suất xử lý dữ liệu**: BST và K-D Tree giúp tìm kiếm, thêm, xóa hiệu quả hơn brute force
2. **Kỹ năng tổ chức dữ liệu**: Trees cung cấp cách hiệu quả để tổ chức dữ liệu theo thứ bậc
3. **Hiểu biết sâu sắc về OOP**: Phân tích chi tiết về liên kết đối tượng
4. **Nền tảng cho AI/Machine Learning**: K-D Tree cho bài toán phân loại và KNN

### Ứng Dụng Thực Tế

- **File Explorer**: Cấu trúc thư mục
- **Database**: Indexing và query optimization
- **Machine Learning**: Decision Tree, K-D Tree
- **Game Development**: Pathfinding algorithms
- **Compilers**: Abstract Syntax Trees

## 🚀 Kết Luận và Hướng Tiếp Theo

Tuần 3 đã trang bị cho tôi một nền tảng vững chắc về:
- **Cấu trúc dữ liệu tuyến tính** (Stack, Queue)
- **Cấu trúc dữ liệu phi tuyến** (Tree, BST, K-D Tree)
- **Thuật toán duyệt và tìm kiếm**
- **Object-Oriented Programming** trong thực tế
- **Trực quan hóa dữ liệu** với Graphviz

Những kiến thức này không chỉ là lý thuyết mà còn là công cụ mạnh mẽ để giải quyết các bài toán thực tế, đặc biệt trong lĩnh vực AI và Machine Learning mà tôi đang theo đuổi.

**Hướng tiếp theo**: Tôi sẽ tiếp tục khám phá các cấu trúc dữ liệu nâng cao khác như Heap, Graph, và các thuật toán sắp xếp, tìm kiếm phức tạp hơn.

---

*"Học không chỉ là tiếp thu kiến thức, mà còn là quá trình đặt câu hỏi, thử nghiệm và khám phá những điều mới mẻ."* 