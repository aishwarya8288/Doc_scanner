📄 Automatic Document Scanner using Homography

A Computer Vision system that detects documents, extracts boundaries, and produces a corrected, top-down scanned view using homography transformation.

✨ Overview

This project implements a fully automatic document scanner using classical computer vision.
Given a normal camera photo—which may be rotated, skewed, shadowed, or captured at an angle—the system:

Detects the document boundary

Computes homography from the 4 corners

Applies perspective correction

Outputs a clean, top-down “scanned” document

This converts any phone-captured picture into a high-quality scanned copy without manual cropping.

🎯 Features
Feature	Description
🖼️ Document Boundary Detection	Detects the largest 4-point contour automatically
📐 Homography Estimation	Calculates perspective mapping using projective geometry
🔄 Perspective Warping	Produces a rectangular, flat, top-down transformed view
✨ Image Enhancement	Optional contrast + sharpening filters
🧠 Noise & Shadow Handling	Uses edge detection + morphological processing
🗂️ Batch Mode	Supports batch scanning of multiple images
🧪 Test Image Generator	Includes script to generate synthetic test images
🗂️ Project Structure
Document Scanner/
│
├── batch_output/               # Output of batch processing
├── scanned_outputs/            # Final scanned images
├── test_images/                # Testing dataset
│
├── batch_test.py               # Batch processing script
├── document_scanner.py         # Main scanner logic (CV pipeline)
├── test_image_generator.py     # Generator for synthetic test images
├── usage_examples.py           # Example usage script
│
├── requirements.txt            # Dependencies
└── README.md                   # Documentation (this file)

🧠 How It Works (Pipeline)

The scanner follows a well-structured computer vision workflow:

1️⃣ Preprocessing

Convert to grayscale

Gaussian blur

Canny edge detection

Morphological close to reduce noise

2️⃣ Document Detection

Find external contours

Select largest quadrilateral contour

Approximate using Douglas–Peucker algorithm

Extract 4 corner points

3️⃣ Homography Computation

Using the 4 corners, compute:

H = findHomography(src_points, dst_points, RANSAC)


This computes a mapping from skewed document → rectangle.

4️⃣ Perspective Transformation

Apply warp:

warped = warpPerspective(image, H, output_size)


Result:
A perfectly aligned, top-down scanned document.

📌 Input & Output Examples
Input Image	Scanned Output
Photo of document at angle	Clean flat document after warping

(Add screenshots if possible for maximum impact)

🚀 Quick Start
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run single image scan
python document_scanner.py

3️⃣ Run batch mode
python batch_test.py

4️⃣ Generate synthetic test images
python test_image_generator.py

🧪 Usage Example
from document_scanner import DocumentScanner

scanner = DocumentScanner()
output = scanner.scan("test_images/sample1.jpg")
output.save("scanned_outputs/output.jpg")

📚 Core Concepts Used

This project demonstrates strong knowledge of computer vision and geometry:

🔹 Projective Geometry

Understanding how 3D → 2D projections work.

🔹 Homography Estimation

Mapping 4 points from one plane to another using:

H ∈ R^(3×3)

🔹 RANSAC

Used to reject outliers while estimating homography.

🔹 Contour Detection

To locate edges and extract document boundaries.

🔹 Perspective Warping

Transforms camera photo into a top-down scanned view.

🧩 Challenges Solved
Challenge	How It's Solved
Document rotated / tilted	Homography corrects perspective
Shadows / uneven lighting	Preprocessing + adaptive thresholding
Background clutter	Largest contour selection
Noise	Gaussian blur + morphological ops
🔧 Enhancements Implemented

Shadow reduction

Automatic cropping

Contrast enhancement

Batch processing support

Synthetic test image generation

Error handling and logging

🔮 Future Improvements

OCR support (Tesseract integration)

Curved page flattening (deep learning)

Mobile app version

Automatic brightness correction

Edge refinement using deep CNN models

📝 Requirements
opencv-python
numpy
imutils


Install via:

pip install -r requirements.txt

📄 License

Licensed under the MIT License.

👩‍💻 Author

Aishwarya Khot
Final Year Computer Engineering Student
Passionate about Computer Vision, AI, and Full-Stack Development
