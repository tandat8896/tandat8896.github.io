---
title: "Computer Vision Classic: Từ Edge Detection đến Object Tracking"
pubDatetime: 2025-01-20T10:00:00Z
featured: false
description: "Khám phá các kỹ thuật kinh điển trong Computer Vision: Canny Edge Detection, SIFT, SURF, Haar Cascade, HOG, Template Matching, và Object Tracking với code Python và OpenCV cụ thể"
tags: ["Computer Vision", "OpenCV", "Image Processing", "Edge Detection", "Object Tracking"]
---

# Computer Vision Classic: Từ Edge Detection đến Object Tracking

Computer Vision là một lĩnh vực thú vị trong AI, nơi chúng ta dạy máy tính "nhìn" và hiểu thế giới xung quanh. Trong bài viết này, mình sẽ cùng các bạn khám phá các kỹ thuật "kinh điển" trong Computer Vision, từ edge detection cơ bản đến object tracking phức tạp. Mỗi kỹ thuật sẽ được minh họa bằng code Python và OpenCV cụ thể.

## Tại sao cần học Computer Vision cổ điển trong thời đại Deep Learning?

Khi deep learning đang thống trị với YOLO, ResNet, hay Transformer, nhiều người hỏi: "Sao vẫn cần học Canny, SIFT, Haar Cascade?" Thực ra, các kỹ thuật cổ điển này không chỉ bổ trợ mà còn giúp bạn **debug và troubleshoot** deep learning models hiệu quả hơn.

### Preprocessing thông minh

Deep learning models rất nhạy cảm với chất lượng dữ liệu. Background subtraction giúp loại bỏ noise, edge detection tạo thêm channel cho model học boundaries tốt hơn. Khi train object detection trên video, background subtraction giúp model tập trung vào moving objects thay vì học cả background tĩnh.

### Debugging Deep Learning Models

Đây là phần quan trọng nhất. Khi model không hoạt động, classical methods giúp bạn isolate vấn đề:

**1. Kiểm tra Data Quality:**
```python
# Khi model accuracy thấp, kiểm tra xem data có vấn đề không
def debug_data_quality(image_path, model_prediction):
    # Dùng Canny để kiểm tra edges
    edges = cv2.Canny(cv2.imread(image_path, 0), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Nếu edge density quá thấp → ảnh có thể bị blur hoặc noise
    if edge_density < 0.01:
        print("Warning: Image might be too blurry or noisy")
        return False
    
    # Dùng contour detection để kiểm tra objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Warning: No objects detected in image")
        return False
    
    return True
```

**2. So sánh với Baseline:**
```python
# So sánh deep model với classical baseline
def compare_with_baseline(image_path, deep_model_prediction):
    # HOG + SVM baseline
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(image_path)
    boxes_baseline, _ = hog.detectMultiScale(img)
    
    # Nếu baseline detect được mà deep model không → vấn đề ở model
    if len(boxes_baseline) > 0 and len(deep_model_prediction) == 0:
        print("Issue: Deep model failed but baseline succeeded")
        print("Possible causes: Model architecture, training data, or hyperparameters")
    
    # Nếu cả hai đều fail → vấn đề ở data
    elif len(boxes_baseline) == 0 and len(deep_model_prediction) == 0:
        print("Issue: Both methods failed - check data quality")
```

**3. Visualize Model Attention:**
```python
# Dùng edge detection để xem model có học đúng features không
def visualize_model_attention(image, model_activations):
    # Lấy edge map
    edges = cv2.Canny(image, 50, 150)
    
    # So sánh với model activations ở layer đầu
    # Nếu activations không tương quan với edges → model có thể không học đúng
    edge_correlation = np.corrcoef(edges.flatten(), 
                                  model_activations.flatten())[0, 1]
    
    if edge_correlation < 0.3:
        print("Warning: Model activations don't correlate with edges")
        print("Model might not be learning edge features properly")
```

**4. Debug False Positives/Negatives:**
```python
# Khi model miss detection, kiểm tra xem classical method có detect được không
def debug_missed_detection(image_path, ground_truth, model_prediction):
    img = cv2.imread(image_path)
    
    # Dùng Haar Cascade để kiểm tra
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')
    faces_haar = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # Nếu Haar detect được mà model không → model issue
    if len(faces_haar) > 0 and len(model_prediction) == 0:
        print("Model missed detection but Haar succeeded")
        print("Check: Model confidence threshold, NMS parameters, or training data")
    
    # Nếu cả hai đều detect được nhưng khác vị trí → alignment issue
    elif len(faces_haar) > 0 and len(model_prediction) > 0:
        iou = calculate_iou(faces_haar[0], model_prediction[0])
        if iou < 0.5:
            print("Detections don't align - check bounding box regression")
```

**5. Validate Preprocessing Pipeline:**
```python
# Kiểm tra xem preprocessing có làm hỏng data không
def validate_preprocessing(image, preprocessed_image):
    # So sánh edges trước và sau preprocessing
    edges_original = cv2.Canny(image, 50, 150)
    edges_processed = cv2.Canny(preprocessed_image, 50, 150)
    
    # Tính similarity
    similarity = np.sum(edges_original == edges_processed) / edges_original.size
    
    if similarity < 0.8:
        print("Warning: Preprocessing might be removing important features")
        print(f"Edge similarity: {similarity:.2%}")
```

### Hybrid Approaches

Kết hợp classical và deep learning thường cho kết quả tốt hơn. R-CNN ban đầu dùng selective search (classical) để propose regions, sau đó CNN classify. Hoặc dùng HOG features như auxiliary input cho deep model.

### Computational Efficiency

Trong production, không phải lúc nào cũng cần deep learning. Background subtraction chạy real-time, trong khi object recognition chỉ cần chạy mỗi vài frames. Trên mobile/IoT devices, classical methods vẫn là lựa chọn tốt.

### Kết luận

Deep learning không thay thế classical methods, mà **bổ sung** cho chúng. Hiểu cả hai giúp bạn debug tốt hơn, chọn đúng tool cho đúng job, và tạo ra các giải pháp hybrid hiệu quả. Trong bài viết này, chúng ta sẽ học các kỹ thuật cổ điển không chỉ để implement, mà còn để biết **khi nào** và **cách nào** dùng chúng để debug deep learning models.

---

## Mục Lục

1. [Background Subtraction](#1-background-subtraction)
2. [Edge Detection](#2-edge-detection)
3. [Line Detection và HOG Transform](#3-line-detection-và-hog-transform)
4. [Contour Detection](#4-contour-detection)
5. [Image Stitching](#5-image-stitching)
6. [Face Detection với Haar Cascade](#6-face-detection-với-haar-cascade)
7. [Object Tracking](#7-object-tracking)
8. [Object Tracking vs SIFT Feature](#8-object-tracking-vs-sift-feature)

---

## 1. Background Subtraction

Background Subtraction là kỹ thuật tách foreground (đối tượng chuyển động) khỏi background (nền tĩnh). Đây là bước đầu tiên quan trọng trong nhiều ứng dụng như video surveillance, traffic monitoring.

### Lý thuyết

**Nguyên lý cơ bản:**

Background Subtraction dựa trên giả định rằng background là tĩnh và foreground là các đối tượng chuyển động. Quá trình bao gồm:

1. **Xây dựng mô hình background**: Học phân phối màu sắc của background từ N frame đầu tiên
2. **So sánh pixel**: Với mỗi pixel trong frame hiện tại, so sánh với mô hình background
3. **Phân loại**: Pixel khác biệt đáng kể → foreground, pixel tương tự → background

**MOG2 (Mixture of Gaussians):**

MOG2 sử dụng Gaussian Mixture Model để mô tả background. Mỗi pixel được mô tả bởi K Gaussian distributions:

$$P(x_t) = \sum_{i=1}^{K} w_{i,t} \cdot \eta(x_t, \mu_{i,t}, \Sigma_{i,t})$$

Trong đó:
- $w_{i,t}$: Trọng số của Gaussian thứ i tại thời điểm t
- $\mu_{i,t}$: Mean của Gaussian thứ i
- $\Sigma_{i,t}$: Covariance matrix (thường là $\sigma^2 I$)

Pixel được phân loại là background nếu nó khớp với một trong K Gaussian có trọng số cao.

**KNN (K-Nearest Neighbors):**

KNN lưu trữ K giá trị gần nhất cho mỗi pixel. Pixel hiện tại được so sánh với K giá trị này:
- Nếu khoảng cách nhỏ hơn threshold → background
- Ngược lại → foreground

### Code Implementation

```python
import cv2
import numpy as np

# Phương pháp 1: BackgroundSubtractorMOG2 (Gaussian Mixture Model)
def background_subtraction_mog2(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Tạo background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,      # Số frame để học background
        varThreshold=50,  # Ngưỡng variance
        detectShadows=True # Phát hiện bóng
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Áp dụng background subtraction
        fgMask = backSub.apply(frame)
        
        # Xử lý fgMask ở đây...
    
    cap.release()

# Phương pháp 2: BackgroundSubtractorKNN
def background_subtraction_knn(video_path):
    cap = cv2.VideoCapture(video_path)
    
    backSub = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400.0,
        detectShadows=True
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgMask = backSub.apply(frame)
        # Xử lý fgMask ở đây...
    
    cap.release()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn video của bạn
    video_path = "video.mp4"
    background_subtraction_mog2(video_path)
```

---

## 2. Edge Detection

Edge Detection là kỹ thuật phát hiện các cạnh (edges) trong ảnh, đóng vai trò quan trọng trong nhiều ứng dụng Computer Vision. Edge là nơi có sự thay đổi đột ngột về cường độ sáng.

### Lý thuyết

**Gradient của ảnh:**

Gradient của ảnh $I(x,y)$ được tính bằng:
- Gradient theo X: $G_x = \frac{\partial I}{\partial x}$
- Gradient theo Y: $G_y = \frac{\partial I}{\partial y}$

Magnitude của gradient: $|G| = \sqrt{G_x^2 + G_y^2}$

Direction của gradient: $\theta = \arctan(\frac{G_y}{G_x})$

### 2.1. Sobel Operator

**Lý thuyết:**

Sobel operator sử dụng convolution với kernel để tính gradient. Kernel Sobel:

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Gradient được tính:
- $G_x = I * S_x$ (convolution)
- $G_y = I * S_y$

Sobel operator nhạy cảm với noise hơn so với các phương pháp khác nhưng tính toán nhanh.

```python
import cv2
import numpy as np

def sobel_edge_detection(image_path):
    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Sobel X (phát hiện cạnh dọc)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
    
    # Sobel Y (phát hiện cạnh ngang)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))
    
    # Kết hợp Sobel X và Y
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
    
    # Hoặc tính magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    
    return sobel_magnitude
```

### 2.2. Canny Edge Detection

**Lý thuyết:**

Canny là thuật toán edge detection phổ biến nhất, được thiết kế để tối ưu 3 tiêu chí:
1. **Low error rate**: Phát hiện đúng các edges thực sự
2. **Good localization**: Vị trí edge chính xác
3. **Single response**: Mỗi edge chỉ được phát hiện một lần

Canny gồm 5 bước:

1. **Gaussian Blur**: Làm mịn ảnh với Gaussian filter để giảm noise
   $$I_{smooth} = I * G_{\sigma}$$
   Trong đó $G_{\sigma}$ là Gaussian kernel với standard deviation $\sigma$

2. **Gradient Calculation**: Tính gradient bằng Sobel operator
   $$G_x = I_{smooth} * S_x, \quad G_y = I_{smooth} * S_y$$
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(\frac{G_y}{G_x})$$

3. **Non-Maximal Suppression**: Loại bỏ các pixel không phải local maximum trong hướng gradient
   - Chỉ giữ lại pixel có gradient magnitude lớn nhất so với 2 neighbors trong hướng gradient

4. **Double Thresholding**: Phân loại edge pixels
   - Strong edges: $|G| > T_{high}$ → chắc chắn là edge
   - Weak edges: $T_{low} < |G| \leq T_{high}$ → có thể là edge
   - Non-edges: $|G| \leq T_{low}$ → không phải edge

5. **Hysteresis**: Kết nối strong edges với weak edges lân cận
   - Weak edge được giữ lại nếu nó kết nối với strong edge

```python
def canny_edge_detection(image_path, low_threshold=50, high_threshold=150):
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Canny edge detection
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    return edges

# Canny với các tham số khác nhau
def canny_with_different_thresholds(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    edges1 = cv2.Canny(img, 50, 150)
    edges2 = cv2.Canny(img, 100, 200)
    edges3 = cv2.Canny(img, 50, 200)
    
    return edges1, edges2, edges3
```

### 2.3. Non-Maximal Suppression (Chi tiết)

**Lý thuyết:**

Non-Maximal Suppression (NMS) là bước quan trọng trong Canny để làm mỏng edges (edge thinning). 

**Nguyên lý:**
- Với mỗi pixel, xác định hướng gradient (0°, 45°, 90°, 135°)
- So sánh gradient magnitude của pixel với 2 neighbors trong hướng gradient
- Chỉ giữ lại pixel nếu nó có gradient magnitude lớn nhất

**Ví dụ:** Nếu gradient direction là 0° (ngang), so sánh với pixel bên trái và bên phải. Nếu gradient direction là 45°, so sánh với pixel ở góc trên-trái và góc dưới-phải.

```python
def non_maximal_suppression(gradient_magnitude, gradient_direction):
    """
    Non-Maximal Suppression cho edge detection
    
    Args:
        gradient_magnitude: Magnitude của gradient
        gradient_direction: Hướng của gradient (radians)
    
    Returns:
        Suppressed image
    """
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    # Chuyển đổi góc từ radians sang degrees và chuẩn hóa về [0, 180]
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Xác định hướng của edge
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbor1 = gradient_magnitude[i, j + 1]
                neighbor2 = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbor1 = gradient_magnitude[i + 1, j - 1]
                neighbor2 = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbor1 = gradient_magnitude[i + 1, j]
                neighbor2 = gradient_magnitude[i - 1, j]
            else:  # 112.5 <= angle < 157.5
                neighbor1 = gradient_magnitude[i + 1, j + 1]
                neighbor2 = gradient_magnitude[i - 1, j - 1]
            
            # Chỉ giữ lại pixel nếu nó là local maximum
            if gradient_magnitude[i, j] >= neighbor1 and gradient_magnitude[i, j] >= neighbor2:
                suppressed[i, j] = gradient_magnitude[i, j]
    
    return suppressed

# Sử dụng trong Canny
def canny_with_nms(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Tính gradient
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude và direction
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x)
    
    # Non-Maximal Suppression
    nms = non_maximal_suppression(magnitude, direction)
    
    # Double thresholding
    high_threshold = np.max(nms) * 0.3
    low_threshold = high_threshold * 0.5
    
    strong_edges = (nms >= high_threshold).astype(np.uint8) * 255
    weak_edges = ((nms >= low_threshold) & (nms < high_threshold)).astype(np.uint8) * 255
    
    # Hysteresis
    edges = hysteresis(strong_edges, weak_edges)
    
    return edges
```

### 2.4. Hysteresis Thresholding

**Lý thuyết:**

Hysteresis Thresholding giải quyết vấn đề của single threshold: nếu threshold quá cao sẽ mất nhiều edges, nếu quá thấp sẽ có nhiều noise.

**Nguyên lý:**
- Sử dụng 2 thresholds: $T_{low}$ và $T_{high}$ ($T_{low} < T_{high}$)
- Strong edges ($|G| > T_{high}$): Chắc chắn là edge, được giữ lại
- Weak edges ($T_{low} < |G| \leq T_{high}$): Chỉ giữ lại nếu kết nối với strong edge trong vùng lân cận 8-connected

**Lợi ích:**
- Giảm false positives (noise) nhờ $T_{high}$
- Giữ lại các edges yếu nhưng quan trọng nhờ $T_{low}$ và kết nối với strong edges

```python
def hysteresis(strong_edges, weak_edges):
    """
    Hysteresis thresholding: Kết nối strong edges với weak edges
    
    Args:
        strong_edges: Binary image của strong edges
        weak_edges: Binary image của weak edges
    
    Returns:
        Final edge map
    """
    rows, cols = strong_edges.shape
    final_edges = strong_edges.copy()
    
    # Tìm các weak edges kết nối với strong edges
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] > 0:
                # Kiểm tra xem có strong edge lân cận không
                neighborhood = strong_edges[i-1:i+2, j-1:j+2]
                if np.any(neighborhood > 0):
                    final_edges[i, j] = 255
    
    return final_edges

# Sử dụng
# edges = canny_with_nms("image.jpg")
```

---

## 3. Line Detection và HOG Transform

### 3.1. Line Detection với Hough Transform

**Lý thuyết:**

Hough Transform là kỹ thuật phát hiện các hình dạng hình học (đường thẳng, đường tròn) trong ảnh bằng cách chuyển đổi từ không gian ảnh sang không gian tham số.

**Hough Line Transform:**

Trong không gian ảnh, một đường thẳng có thể được biểu diễn bằng phương trình:
$$y = mx + c$$

Tuy nhiên, khi $m \to \infty$ (đường thẳng dọc), phương trình này không ổn định. Thay vào đó, ta dùng dạng chuẩn:
$$\rho = x\cos\theta + y\sin\theta$$

Trong đó:
- $\rho$: Khoảng cách từ gốc tọa độ đến đường thẳng
- $\theta$: Góc giữa đường thẳng và trục X

**Quá trình:**
1. Với mỗi edge pixel $(x, y)$, vẽ đường cong trong không gian $(\rho, \theta)$
2. Tìm các điểm giao nhau của các đường cong → tương ứng với đường thẳng trong ảnh
3. Sử dụng accumulator array để đếm số đường cong đi qua mỗi điểm $(\rho, \theta)$
4. Các điểm có giá trị cao trong accumulator → đường thẳng được phát hiện

```python
def line_detection_hough(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    # Vẽ các đường thẳng
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Probabilistic Hough Line Transform (nhanh hơn)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    img_p = img.copy()
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_p, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img, img_p

# Circle detection với Hough
def circle_detection_hough(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for i in circles[0, :]:
            cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        return img_color, circles
    
    return None, None
```

### 3.2. HOG (Histogram of Oriented Gradients)

**Lý thuyết:**

HOG là feature descriptor mạnh mẽ cho object detection, đặc biệt là pedestrian detection. HOG mô tả đối tượng bằng cách phân tích phân phối gradient directions trong các vùng cục bộ.

**Quá trình tính HOG:**

1. **Tính gradient**: Với mỗi pixel, tính gradient magnitude và direction
   $$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(\frac{G_y}{G_x})$$

2. **Chia ảnh thành cells**: Ảnh được chia thành các cells nhỏ (thường 8×8 pixels)

3. **Tính histogram cho mỗi cell**: 
   - Chia 360° thành N bins (thường 9 bins, mỗi bin 40°)
   - Với mỗi pixel trong cell, thêm gradient magnitude vào bin tương ứng với gradient direction

4. **Normalize blocks**: 
   - Nhóm các cells thành blocks (thường 2×2 cells)
   - Normalize histogram của block để robust với illumination changes
   - Có thể dùng L2-norm: $v = \frac{v}{\sqrt{||v||^2 + \epsilon^2}}$

5. **Feature vector**: Nối tất cả các normalized histograms thành một vector

**Ưu điểm:**
- Robust với illumination changes (nhờ normalization)
- Captures local shape information (gradient directions)
- Invariant với một số biến đổi hình học nhỏ

```python
from skimage.feature import hog

def hog_feature_extraction(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Tính HOG features
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, feature_vector=True)
    
    return fd, hog_image

# HOG cho pedestrian detection
def pedestrian_detection_hog(image_path):
    # Load pre-trained HOG descriptor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    # Detect people
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(32, 32), scale=1.05)
    
    # Vẽ bounding boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img, boxes
```

---

## 4. Contour Detection

**Lý thuyết:**

Contour Detection tìm các đường viền (contours) của objects trong ảnh nhị phân. Contour là đường cong nối các điểm có cùng cường độ sáng, tạo thành ranh giới của đối tượng.

**Nguyên lý:**

1. **Binary Image**: Contour detection yêu cầu ảnh nhị phân (binary image)
   - Có thể tạo bằng thresholding: $I_{binary}(x,y) = \begin{cases} 255 & \text{nếu } I(x,y) > T \\ 0 & \text{ngược lại} \end{cases}$

2. **Contour Tracing**: 
   - Thuật toán phổ biến: **Suzuki-Abe algorithm** (được implement trong OpenCV)
   - Bắt đầu từ một điểm trên boundary
   - Theo dõi boundary bằng cách tìm điểm kế tiếp trong vùng lân cận 8-connected
   - Dừng khi quay lại điểm xuất phát

3. **Contour Hierarchy**: 
   - **RETR_EXTERNAL**: Chỉ lấy contours ngoài cùng
   - **RETR_TREE**: Lấy tất cả contours và xây dựng cây phân cấp (parent-child)
   - **RETR_LIST**: Lấy tất cả contours không có hierarchy

4. **Contour Approximation**:
   - **CHAIN_APPROX_NONE**: Lưu tất cả các điểm
   - **CHAIN_APPROX_SIMPLE**: Chỉ lưu các điểm góc (tiết kiệm bộ nhớ)

**Các thuộc tính của Contour:**
- **Area**: Diện tích vùng được bao bởi contour
- **Perimeter**: Chu vi của contour
- **Bounding Rectangle**: Hình chữ nhật nhỏ nhất bao quanh contour
- **Minimum Area Rectangle**: Hình chữ nhật có diện tích nhỏ nhất (có thể xoay)
- **Minimum Enclosing Circle**: Đường tròn nhỏ nhất bao quanh contour

```python
def contour_detection(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vẽ contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    return contours, hierarchy, img_contours

# Contour properties
def contour_analysis(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_result = img.copy()
    
    for i, contour in enumerate(contours):
        # Area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_result, [box], 0, (0, 0, 255), 2)
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        center = (int(cx), int(cy))
        radius = int(radius)
        cv2.circle(img_result, center, radius, (255, 255, 0), 2)
    
    return img_result, contours
```

---

## 5. Image Stitching

Image Stitching là kỹ thuật ghép nhiều ảnh có phần chồng lấp thành một ảnh panorama lớn hơn. Quá trình này bao gồm:

1. **Feature Detection**: Phát hiện các điểm đặc trưng (keypoints) trong mỗi ảnh
2. **Feature Matching**: Khớp các điểm đặc trưng giữa các ảnh
3. **Homography Estimation**: Tính ma trận biến đổi (homography) để căn chỉnh ảnh
4. **Image Warping**: Biến đổi ảnh theo ma trận homography
5. **Blending**: Trộn các ảnh đã căn chỉnh thành panorama

### 5.1. Blob Detector

**Lý thuyết:**

Blob Detector phát hiện các vùng có tính chất đặc biệt (blobs) trong ảnh, thường là các vùng có độ sáng hoặc màu sắc khác biệt so với xung quanh.

**Laplacian of Gaussian (LoG):**

LoG được tính bằng cách:
1. Áp dụng Gaussian blur: $L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$
2. Tính Laplacian: $\nabla^2 L = \frac{\partial^2 L}{\partial x^2} + \frac{\partial^2 L}{\partial y^2}$

LoG có thể được tính trực tiếp:
$$\text{LoG}(x,y,\sigma) = \frac{x^2 + y^2 - 2\sigma^2}{\sigma^4} \cdot e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Blob được phát hiện tại các điểm cực trị của LoG trong scale-space.

**Difference of Gaussians (DoG):**

DoG là xấp xỉ của LoG, được tính nhanh hơn:
$$\text{DoG}(x,y,\sigma) = G(x,y,k\sigma) - G(x,y,\sigma)$$

Trong đó $k$ là hệ số scale (thường $k = \sqrt{2}$). DoG ≈ LoG khi $k \to 1$.

**Scale-space Detection:**
- Blob được phát hiện ở nhiều scale khác nhau
- Tìm cực trị trong không gian 3D: $(x, y, \sigma)$

```python
def blob_detection(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple Blob Detector
    params = cv2.SimpleBlobDetector_Params()
    
    # Lọc theo màu
    params.filterByColor = True
    params.blobColor = 255
    
    # Lọc theo diện tích
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 10000
    
    # Lọc theo hình dạng (circularity)
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    # Lọc theo độ lồi (convexity)
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    # Tạo detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Phát hiện blobs
    keypoints = detector.detect(gray)
    
    # Vẽ keypoints
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), 
                                          (0, 0, 255), 
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_with_keypoints, keypoints
```

### 5.2. SIFT Detector

**Lý thuyết:**

SIFT (Scale-Invariant Feature Transform) là một trong những feature detector mạnh mẽ nhất, có khả năng phát hiện các điểm đặc trưng bất biến với scale, rotation, và một phần với illumination.

**Quá trình 4 bước:**

1. **Scale-space Extrema Detection**: 
   - Xây dựng scale-space pyramid với nhiều octaves
   - Mỗi octave có nhiều scales (thường 5 scales)
   - Tính DoG giữa các scales liên tiếp
   - Tìm cực trị trong không gian 3D $(x, y, \sigma)$ bằng cách so sánh với 26 neighbors (8 trong cùng scale, 9 trong scale trên, 9 trong scale dưới)

2. **Keypoint Localization**:
   - Loại bỏ các điểm có độ tương phản thấp: $|D(x)| < 0.03$ (sau khi normalize)
   - Loại bỏ các điểm trên edge bằng cách kiểm tra ratio của eigenvalues của Hessian matrix:
     $$\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}$$
     Trong đó $r = 10$ (threshold ratio), $H$ là Hessian matrix của DoG

3. **Orientation Assignment**:
   - Tính gradient magnitude và direction trong vùng 16×16 xung quanh keypoint
   - Tạo histogram 36 bins (mỗi bin 10°) cho gradient directions
   - Gán orientation là bin có giá trị cao nhất
   - Nếu có bin khác có giá trị > 80% của max → tạo thêm keypoint với orientation đó

4. **Keypoint Descriptor**:
   - Chia vùng 16×16 thành 4×4 cells
   - Với mỗi cell, tính histogram 8 bins cho gradient directions
   - Kết quả: 4×4×8 = 128 chiều
   - Normalize descriptor để robust với illumination changes

```python
def sift_feature_detection(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tạo SIFT detector
    sift = cv2.SIFT_create(nfeatures=500)
    
    # Phát hiện keypoints và descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Vẽ keypoints
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_keypoints, keypoints, descriptors

# SIFT Feature Matching
def sift_feature_matching(img1_path, img2_path):
    # Đọc ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SIFT detector
    sift = cv2.SIFT_create()
    
    # Phát hiện keypoints và descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Feature matching với FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test để lọc matches tốt
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    # Vẽ matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, good_matches, kp1, kp2
```

### 5.3. Transform 2D-3D và Homography

**Lý thuyết:**

**Homography** là phép biến đổi projective giữa hai mặt phẳng, được biểu diễn bằng ma trận 3×3. Nó mô tả mối quan hệ giữa các điểm trên hai ảnh khác nhau của cùng một mặt phẳng.

**Công thức**: 
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Trong đó $H$ là ma trận homography 3×3:
$$H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}$$

**Tính toán Homography:**

Với 4 cặp điểm tương ứng $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, ta có hệ phương trình:
$$x'_i = \frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + h_{33}}$$
$$y'_i = \frac{h_{21}x_i + h_{22}y_i + h_{23}}{h_{31}x_i + h_{32}y_i + h_{33}}$$

Viết lại dưới dạng tuyến tính (sau khi nhân chéo):
$$\begin{bmatrix} x_i & y_i & 1 & 0 & 0 & 0 & -x'_i x_i & -x'_i y_i & -x'_i \\ 0 & 0 & 0 & x_i & y_i & 1 & -y'_i x_i & -y'_i y_i & -y'_i \end{bmatrix} \mathbf{h} = \mathbf{0}$$

Với $\mathbf{h} = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$

Cần tối thiểu 4 cặp điểm (8 phương trình cho 8 tham số độc lập, vì $h_{33}$ thường được set = 1).

**RANSAC cho Homography:**

1. Chọn ngẫu nhiên 4 cặp điểm
2. Tính homography từ 4 cặp điểm này
3. Đếm số inliers (điểm có error < threshold)
4. Lặp lại và chọn homography có nhiều inliers nhất

```python
def estimate_homography(kp1, kp2, matches):
    # Lấy tọa độ các điểm matched
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Ước lượng homography với RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, 
                                 cv2.RANSAC, 5.0)
    
    return H, mask

def warp_image(img, H):
    """Biến đổi ảnh theo ma trận homography"""
    h, w = img.shape[:2]
    
    # Tính kích thước ảnh sau khi warp
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    
    # Tìm bounding box
    x_min = int(np.min(warped_corners[:, 0, 0]))
    x_max = int(np.max(warped_corners[:, 0, 0]))
    y_min = int(np.min(warped_corners[:, 0, 1]))
    y_max = int(np.max(warped_corners[:, 0, 1]))
    
    # Điều chỉnh homography để ảnh không bị cắt
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_adjusted = translation @ H
    
    # Warp ảnh
    warped = cv2.warpPerspective(img, H_adjusted, 
                                (x_max - x_min, y_max - y_min))
    
    return warped, H_adjusted
```

### 5.4. Panorama Technique

Kỹ thuật tạo panorama từ nhiều ảnh:

```python
def create_panorama(image_paths):
    """
    Tạo panorama từ danh sách ảnh
    
    Args:
        image_paths: List đường dẫn đến các ảnh cần ghép
    """
    # Đọc ảnh đầu tiên làm base
    base_img = cv2.imread(image_paths[0])
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    for i in range(1, len(image_paths)):
        # Đọc ảnh tiếp theo
        next_img = cv2.imread(image_paths[i])
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện và match features
        kp1, des1 = sift.detectAndCompute(base_gray, None)
        kp2, des2 = sift.detectAndCompute(next_gray, None)
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            continue
        
        # Ước lượng homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Warp ảnh tiếp theo
        h1, w1 = base_img.shape[:2]
        h2, w2 = next_img.shape[:2]
        
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_warped = cv2.perspectiveTransform(corners2, H)
        
        all_corners = np.concatenate((corners1, corners2_warped), axis=0)
        
        # Tính kích thước panorama
        x_min = int(np.min(all_corners[:, 0, 0]))
        x_max = int(np.max(all_corners[:, 0, 0]))
        y_min = int(np.min(all_corners[:, 0, 1]))
        y_max = int(np.max(all_corners[:, 0, 1]))
        
        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H_translated = translation @ H
        
        # Warp cả hai ảnh
        warped_next = cv2.warpPerspective(next_img, H_translated, 
                                         (x_max - x_min, y_max - y_min))
        warped_base = cv2.warpPerspective(base_img, translation, 
                                         (x_max - x_min, y_max - y_min))
        
        # Blending: Simple average (có thể dùng advanced blending như multi-band)
        mask1 = (warped_base > 0).astype(np.uint8)
        mask2 = (warped_next > 0).astype(np.uint8)
        overlap = mask1 * mask2
        
        # Nơi overlap: average
        # Nơi không overlap: giữ nguyên
        result = warped_base.copy()
        result[overlap > 0] = (warped_base[overlap > 0] + warped_next[overlap > 0]) // 2
        result[mask2 > mask1] = warped_next[mask2 > mask1]
        
        # Cập nhật base image cho lần lặp tiếp theo
        base_img = result
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    
    return base_img
```

---

## 6. Face Detection với Haar Cascade

Face Detection là bài toán phát hiện khuôn mặt trong ảnh. Haar Cascade sử dụng các đặc trưng Haar-like và AdaBoost classifier để phát hiện khuôn mặt một cách hiệu quả.

### 6.1. Haar Features

Haar Features là các đặc trưng đơn giản dựa trên sự khác biệt về độ sáng giữa các vùng kề nhau. Có 4 loại cơ bản:

1. **Edge Features**: Phát hiện cạnh dọc/ngang
2. **Line Features**: Phát hiện đường thẳng
3. **Center-surround Features**: Phát hiện vùng trung tâm
4. **Diagonal Features**: Phát hiện đường chéo

**Nguyên lý**: Tính tổng pixel trong vùng sáng trừ tổng pixel trong vùng tối. Giá trị này phản ánh sự tương phản giữa các vùng.

```python
def compute_haar_feature(image, feature_type, x, y, width, height):
    """
    Tính giá trị Haar feature
    
    Args:
        image: Integral image
        feature_type: Loại feature ('edge_h', 'edge_v', 'line_h', 'line_v')
        x, y: Vị trí feature
        width, height: Kích thước feature
    """
    if feature_type == 'edge_h':
        # Horizontal edge: vùng trên trừ vùng dưới
        white = integral_sum(image, x, y, width, height // 2)
        black = integral_sum(image, x, y + height // 2, width, height // 2)
        return white - black
    elif feature_type == 'edge_v':
        # Vertical edge: vùng trái trừ vùng phải
        white = integral_sum(image, x, y, width // 2, height)
        black = integral_sum(image, x + width // 2, y, width // 2, height)
        return white - black
    # ... các loại feature khác
```

### 6.2. Integral Image

Integral Image (hay Summed Area Table) cho phép tính tổng pixel trong một vùng hình chữ nhật bất kỳ với độ phức tạp O(1) thay vì O(width × height).

**Công thức**:
- Integral Image: $I(x,y) = \sum_{i=0}^{x} \sum_{j=0}^{y} img(i,j)$
- Tổng trong vùng [x1, y1] đến [x2, y2]: 
  $Sum = I(x2,y2) - I(x1-1,y2) - I(x2,y1-1) + I(x1-1,y1-1)$

```python
def compute_integral_image(image):
    """
    Tính integral image
    
    Args:
        image: Ảnh grayscale
    
    Returns:
        Integral image
    """
    h, w = image.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int32)
    
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            integral[i, j] = (image[i-1, j-1] + 
                            integral[i-1, j] + 
                            integral[i, j-1] - 
                            integral[i-1, j-1])
    
    return integral

def integral_sum(integral, x, y, width, height):
    """
    Tính tổng pixel trong vùng [x, y] đến [x+width, y+height]
    sử dụng integral image
    """
    x2, y2 = x + width, y + height
    return (integral[y2, x2] - 
            integral[y, x2] - 
            integral[y2, x] + 
            integral[y, x])
```

### 6.3. AdaBoost

AdaBoost (Adaptive Boosting) là thuật toán ensemble learning kết hợp nhiều weak classifiers thành một strong classifier.

**Nguyên lý**:
1. Khởi tạo trọng số đồng đều cho tất cả training samples
2. Với mỗi iteration:
   - Tìm weak classifier tốt nhất (có error nhỏ nhất)
   - Cập nhật trọng số: tăng trọng số cho các samples bị phân loại sai
   - Tính trọng số cho weak classifier này
3. Kết hợp tất cả weak classifiers thành strong classifier

```python
def adaboost_training(features, labels, T=100):
    """
    Training AdaBoost classifier
    
    Args:
        features: Ma trận features (N samples × M features)
        labels: Labels (+1 hoặc -1)
        T: Số weak classifiers
    
    Returns:
        List các weak classifiers và trọng số của chúng
    """
    N, M = features.shape
    weights = np.ones(N) / N  # Khởi tạo trọng số đồng đều
    weak_classifiers = []
    alphas = []
    
    for t in range(T):
        # Tìm weak classifier tốt nhất
        best_error = float('inf')
        best_classifier = None
        best_threshold = None
        best_polarity = None
        
        for j in range(M):
            # Sắp xếp features
            sorted_indices = np.argsort(features[:, j])
            sorted_features = features[sorted_indices, j]
            sorted_labels = labels[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Thử các threshold
            for threshold in sorted_features:
                # Thử polarity = 1
                predictions = np.where(sorted_features >= threshold, 1, -1)
                error = np.sum(sorted_weights * (predictions != sorted_labels))
                
                if error < best_error:
                    best_error = error
                    best_classifier = j
                    best_threshold = threshold
                    best_polarity = 1
                
                # Thử polarity = -1
                predictions = np.where(sorted_features >= threshold, -1, 1)
                error = np.sum(sorted_weights * (predictions != sorted_labels))
                
                if error < best_error:
                    best_error = error
                    best_classifier = j
                    best_threshold = threshold
                    best_polarity = -1
        
        # Tính alpha (trọng số của weak classifier)
        alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        
        # Cập nhật trọng số samples
        predictions = np.where(features[:, best_classifier] >= best_threshold, 
                              best_polarity, -best_polarity)
        weights = weights * np.exp(-alpha * labels * predictions)
        weights = weights / np.sum(weights)  # Normalize
        
        weak_classifiers.append({
            'feature': best_classifier,
            'threshold': best_threshold,
            'polarity': best_polarity
        })
        alphas.append(alpha)
    
    return weak_classifiers, alphas

def adaboost_predict(features, weak_classifiers, alphas):
    """
    Dự đoán với AdaBoost classifier
    """
    predictions = np.zeros(features.shape[0])
    
    for classifier, alpha in zip(weak_classifiers, alphas):
        feature_idx = classifier['feature']
        threshold = classifier['threshold']
        polarity = classifier['polarity']
        
        pred = np.where(features[:, feature_idx] >= threshold, 
                       polarity, -polarity)
        predictions += alpha * pred
    
    return np.sign(predictions)
```

### 6.4. Face Detection với OpenCV

```python
def face_detection_haar(image_path):
    # Load pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                          'haarcascade_frontalface_default.xml')
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Tỷ lệ giảm kích thước mỗi lần scale
        minNeighbors=5,      # Số neighbors tối thiểu để giữ detection
        minSize=(30, 30),    # Kích thước tối thiểu
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Vẽ bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return img, faces
```

---

## 7. Object Tracking

Object Tracking là bài toán theo dõi đối tượng qua các frame trong video. Có nhiều phương pháp khác nhau:

### 7.1. Template Matching

**Lý thuyết:**

Template Matching tìm vị trí của template (mẫu) trong ảnh bằng cách so sánh template với từng vùng trong ảnh.

**Nguyên lý**: 

Template matching sử dụng correlation để đo độ tương đồng giữa template $T$ và vùng ảnh $I$:

**Cross-correlation:**
$$R(x,y) = \sum_{i,j} T(i,j) \cdot I(x+i, y+j)$$

**Normalized Cross-correlation (NCC):**
$$R(x,y) = \frac{\sum_{i,j} (T(i,j) - \bar{T}) \cdot (I(x+i, y+j) - \bar{I}_{x,y})}{\sqrt{\sum_{i,j} (T(i,j) - \bar{T})^2 \cdot \sum_{i,j} (I(x+i, y+j) - \bar{I}_{x,y})^2}}$$

Trong đó:
- $\bar{T}$: Mean của template
- $\bar{I}_{x,y}$: Mean của vùng ảnh tại $(x,y)$

NCC robust hơn với illumination changes vì đã normalize.

**Các phương pháp matching trong OpenCV:**
- `TM_CCOEFF`: Correlation coefficient
- `TM_CCOEFF_NORMED`: Normalized correlation coefficient (0-1)
- `TM_CCORR`: Cross-correlation
- `TM_CCORR_NORMED`: Normalized cross-correlation
- `TM_SQDIFF`: Squared difference (giá trị nhỏ = tốt)
- `TM_SQDIFF_NORMED`: Normalized squared difference

```python
def template_matching(image, template):
    """
    Template matching để tìm vị trí template trong ảnh
    
    Args:
        image: Ảnh lớn
        template: Template cần tìm
    
    Returns:
        Vị trí (x, y) của template
    """
    # Chuyển sang grayscale nếu cần
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Template matching với normalized correlation
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # Tìm vị trí có giá trị cao nhất
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Vẽ bounding box
    h, w = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    return top_left, bottom_right, max_val

def tracking_by_template_matching(video_path, template):
    """
    Tracking đối tượng trong video bằng template matching
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        return
    
    # Template matching cho frame đầu tiên
    top_left, bottom_right, confidence = template_matching(frame, template)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Template matching
        top_left, bottom_right, confidence = template_matching(frame, template)
        
        # Vẽ bounding box
        if confidence > 0.7:  # Ngưỡng confidence
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence:.2f}', 
                       (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Xử lý frame...
    
    cap.release()
```

### 7.2. Histogram Matching

**Lý thuyết:**

Histogram Matching sử dụng histogram màu để mô tả đối tượng và tìm đối tượng trong frame tiếp theo dựa trên sự tương đồng histogram.

**Nguyên lý:**

1. **Histogram của đối tượng**: Tính histogram màu của vùng ROI (Region of Interest) trong frame đầu tiên
   $$H_{obj}(c) = \frac{1}{N} \sum_{(x,y) \in ROI} \delta(I(x,y) - c)$$
   Trong đó $N$ là số pixel trong ROI, $c$ là giá trị màu

2. **Histogram Backprojection**: Với mỗi pixel trong frame mới, tính xác suất nó thuộc về đối tượng dựa trên histogram:
   $$BP(x,y) = H_{obj}(I(x,y))$$
   
   Backprojection tạo ra một probability map, trong đó giá trị cao = khả năng cao là đối tượng

3. **Tìm vị trí mới**: Sử dụng MeanShift hoặc CamShift để tìm vị trí có mật độ cao nhất trong backprojection map

**Ưu điểm:**
- Robust với một số biến đổi hình học (rotation, scale nhỏ)
- Hiệu quả với đối tượng có màu sắc đặc trưng

**Nhược điểm:**
- Không hiệu quả khi có nhiều đối tượng có màu tương tự
- Phụ thuộc vào chất lượng histogram ban đầu

```python
def calculate_histogram(roi):
    """
    Tính histogram cho vùng ROI
    
    Args:
        roi: Vùng quan tâm (Region of Interest)
    
    Returns:
        Histogram
    """
    # Chuyển sang HSV để tracking tốt hơn
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Tính histogram (chỉ dùng H và S, bỏ qua V để robust với illumination)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist

def histogram_backprojection(image, hist):
    """
    Histogram backprojection để tìm vùng có histogram tương tự
    
    Args:
        image: Ảnh cần tìm
        hist: Histogram của đối tượng
    
    Returns:
        Backprojection image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Backprojection
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    
    return dst
```

### 7.3. MeanShift

**Lý thuyết:**

MeanShift là thuật toán tracking dựa trên histogram, tìm mode (điểm có mật độ cao nhất) của phân phối trong không gian.

**Nguyên lý**: 

MeanShift là thuật toán non-parametric để tìm mode của probability density function:

1. **Khởi tạo**: Bắt đầu từ vị trí hiện tại $x_0$

2. **MeanShift iteration**: 
   $$x_{t+1} = \frac{\sum_{i=1}^{n} K(x_i - x_t) \cdot x_i}{\sum_{i=1}^{n} K(x_i - x_t)}$$
   
   Trong đó $K$ là kernel function (thường là Epanechnikov kernel hoặc Gaussian kernel)

3. **Trong tracking**: 
   - Tính histogram của đối tượng trong frame đầu
   - Với mỗi frame tiếp theo:
     - Tính histogram backprojection → tạo probability map
     - MeanShift tìm vị trí mới bằng cách di chuyển window về phía có mật độ cao hơn
     - Lặp lại cho đến khi hội tụ (vị trí không thay đổi đáng kể)

**Đặc điểm:**
- Kích thước window cố định
- Tự động tìm vị trí mới dựa trên density
- Hiệu quả với đối tượng có màu sắc đặc trưng

```python
def meanshift_tracking(video_path, initial_bbox):
    """
    Tracking bằng MeanShift
    
    Args:
        video_path: Đường dẫn video
        initial_bbox: Bounding box ban đầu (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        return
    
    # Setup tracking window
    x, y, w, h = initial_bbox
    track_window = (x, y, w, h)
    
    # Tính histogram của đối tượng
    roi = frame[y:y+h, x:x+w]
    roi_hist = calculate_histogram(roi)
    
    # Setup termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Histogram backprojection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        
        # MeanShift để tìm vị trí mới
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Vẽ bounding box
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Xử lý frame...
    
    cap.release()
```

### 7.4. CamShift

**Lý thuyết:**

CamShift (Continuously Adaptive MeanShift) là phiên bản cải tiến của MeanShift, tự động điều chỉnh kích thước và hướng của tracking window.

**Khác biệt với MeanShift**: 
- **MeanShift**: Kích thước window cố định
- **CamShift**: Kích thước và hướng window thay đổi theo đối tượng

**Quá trình CamShift:**

1. **MeanShift**: Tìm vị trí mới (giống MeanShift)

2. **Tính moment**: Sau khi tìm được vị trí, tính các moments của vùng trong window:
   $$M_{00} = \sum_{x,y} I(x,y)$$
   $$M_{10} = \sum_{x,y} x \cdot I(x,y), \quad M_{01} = \sum_{x,y} y \cdot I(x,y)$$
   $$M_{20} = \sum_{x,y} x^2 \cdot I(x,y), \quad M_{02} = \sum_{x,y} y^2 \cdot I(x,y), \quad M_{11} = \sum_{x,y} xy \cdot I(x,y)$$

3. **Tính kích thước và hướng mới**:
   - Centroid: $(c_x, c_y) = (\frac{M_{10}}{M_{00}}, \frac{M_{01}}{M_{00}})$
   - Tính eigenvalues và eigenvectors của covariance matrix để xác định kích thước và hướng
   - Điều chỉnh window dựa trên kết quả

4. **Lặp lại** cho đến khi hội tụ

**Ưu điểm:**
- Tự động điều chỉnh kích thước khi đối tượng thay đổi
- Robust hơn với scale changes
- Có thể track đối tượng xoay

```python
def camshift_tracking(video_path, initial_bbox):
    """
    Tracking bằng CamShift
    
    Args:
        video_path: Đường dẫn video
        initial_bbox: Bounding box ban đầu (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        return
    
    # Setup tracking window
    x, y, w, h = initial_bbox
    track_window = (x, y, w, h)
    
    # Tính histogram của đối tượng
    roi = frame[y:y+h, x:x+w]
    roi_hist = calculate_histogram(roi)
    
    # Setup termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Histogram backprojection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        
        # CamShift để tìm vị trí và kích thước mới
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
        # CamShift trả về rotated rectangle
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Xử lý frame...
    
    cap.release()
```

---

## 8. Object Tracking vs SIFT Feature

So sánh hai phương pháp tracking: tracking truyền thống (template/histogram) và tracking dựa trên SIFT features.

### 8.1. Tracking với SIFT Features

SIFT features có ưu điểm là robust với scale, rotation, và một phần với illumination changes.

```python
def sift_based_tracking(video_path, initial_bbox):
    """
    Tracking đối tượng bằng SIFT features
    
    Args:
        video_path: Đường dẫn video
        initial_bbox: Bounding box ban đầu (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        return
    
    # Extract template từ frame đầu
    x, y, w, h = initial_bbox
    template = frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # SIFT detector
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template_gray, None)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features trong frame hiện tại
        kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
        
        if des_frame is not None and len(des_template) > 0:
            # Match features
            matches = flann.knnMatch(des_template, des_frame, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 10:
                # Lấy tọa độ các điểm matched
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Điều chỉnh tọa độ template về tọa độ gốc
                src_pts[:, 0, 0] += x
                src_pts[:, 0, 1] += y
                
                # Ước lượng homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Tính bounding box mới từ homography
                    h_template, w_template = template_gray.shape
                    corners = np.float32([[0, 0], [w_template, 0], 
                                         [w_template, h_template], [0, h_template]]).reshape(-1, 1, 2)
                    corners_transformed = cv2.perspectiveTransform(corners, H)
                    
                    # Vẽ bounding box
                    corners_int = np.int32(corners_transformed)
                    cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)
        
        # Xử lý frame...
    
    cap.release()
```

### 8.2. So sánh các phương pháp

| Phương pháp | Ưu điểm | Nhược điểm | Ứng dụng |
|------------|---------|------------|----------|
| **Template Matching** | Đơn giản, nhanh | Không robust với scale/rotation | Đối tượng ít thay đổi |
| **Histogram Matching** | Robust với một số biến đổi | Không chính xác khi có đối tượng tương tự | Tracking màu sắc đặc trưng |
| **MeanShift** | Tự động tìm vị trí | Kích thước cố định | Đối tượng có màu sắc đặc trưng |
| **CamShift** | Tự điều chỉnh kích thước | Phụ thuộc vào histogram | Đối tượng thay đổi kích thước |
| **SIFT Features** | Robust với scale/rotation/illumination | Chậm, cần nhiều features | Đối tượng có texture rõ ràng |

---

## Tổng Kết

Trong bài viết này, chúng ta đã khám phá các kỹ thuật "kinh điển" trong Computer Vision:

1. **Background Subtraction**: Tách foreground khỏi background
2. **Edge Detection**: Phát hiện cạnh với Sobel, Canny (bao gồm NMS và Hysteresis)
3. **Line Detection & HOG**: Phát hiện đường thẳng và tính HOG features
4. **Contour Detection**: Tìm và phân tích contours
5. **Image Stitching**: Ghép ảnh với SIFT, Blob Detector, và Homography
6. **Face Detection**: Phát hiện khuôn mặt với Haar Features, Integral Image, và AdaBoost
7. **Object Tracking**: Template Matching, Histogram Matching, MeanShift, CamShift
8. **SIFT-based Tracking**: Tracking robust với SIFT features

Mỗi kỹ thuật đều có ưu nhược điểm riêng và phù hợp với các bài toán khác nhau. Việc hiểu rõ nguyên lý và cách implement sẽ giúp chúng ta lựa chọn phương pháp phù hợp cho từng ứng dụng cụ thể.

Hy vọng bài viết này giúp các bạn hiểu rõ hơn về các kỹ thuật Computer Vision cổ điển và cách áp dụng chúng trong thực tế!
