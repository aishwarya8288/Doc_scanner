# 📄 Automatic Document Scanner using Homography

> A classical computer vision-based document scanner that automatically detects document boundaries and generates a corrected, top-down scanned view using homography transformation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24.0-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- 📐 **Automatic Document Detection** – Detects document boundaries using contour approximation  
- 🔄 **Perspective Correction (Homography)** – Converts angled photos into flat scanned documents  
- 🧠 **RANSAC-Based Outlier Removal** – Robust homography estimation even with noise  
- 🖼️ **Edge & Corner Extraction** – Canny + contour analysis for accurate detection  
- ✨ **Image Enhancement** – Sharpening, thresholding, and contrast improvement  
- 📂 **Batch Processing Support** – Scan multiple images at once  
- ⚡ **Fast Processing** – Under 2–3 seconds per document  
- 🔍 **Noise & Shadow Reduction** – Improved detection under poor lighting  
- 🧪 **Synthetic Test Image Generator** – For algorithm benchmarking  

---

## 🎯 Objective

Design an intelligent computer vision system that:

1. Detects a document inside a natural scene image  
2. Extracts the boundary and corner points  
3. Computes the **homography matrix**  
4. Produces a clean, top-down scanned version of the document  

---

## 🧠 How It Works (Processing Pipeline)

```python
1. Convert image to grayscale  
2. Apply Gaussian blur  
3. Detect edges using Canny  
4. Find contours and identify largest quadrilateral  
5. Sort corner points (TL, TR, BR, BL)  
6. Compute Homography (cv2.findHomography)  
7. Warp image (cv2.warpPerspective)  
8. Enhance and save output  

Document Scanner/
│
├── batch_output/              # Batch processed results
├── scanned_outputs/           # Final flattened scans
├── test_images/               # Input sample images
│
├── batch_test.py              # Batch mode script
├── document_scanner.py        # Core homography + CV logic
├── test_image_generator.py    # Creates synthetic testing data
├── usage_examples.py          # Demonstration script
│
├── requirements.txt           # Dependencies
└── README.md                  # Documentation

## 🔍 Core Concepts Used

### 🟦 Projective Geometry
Mapping points between planes using a 3×3 homography matrix.

### 🟥 Contour Detection
Identifies the largest quadrilateral shape.

### 🟧 Canny Edge Detection
Extracts document edges.

### 🟩 Douglas–Peucker Algorithm
Simplifies contours to 4 points.

### 🟨 RANSAC Homography
Rejects outliers and computes stable transformation.

### 🟦 Perspective Warping
Creates the final corrected document scan.

---

## 🧠 Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Rotated or angled image | Homography correction |
| Shadow or uneven lighting | Preprocessing + thresholding |
| Background clutter | Contour filtering |
| Noisy edges | Gaussian blur + morphology |
| Wrong corner order | Custom sorting algorithm |

---

## 🛠️ Enhancements Included

- Automatic cropping
- Shadow reduction
- Sharpen + contrast boost
- Batch image scanning
- Corner-order validation
- Synthetic image testing tool

---

## 🔮 Future Enhancements

- OCR (Tesseract integration)
- Curved page flattening
- Mobile app version
- Web UI with Flask/React
- PDF output

