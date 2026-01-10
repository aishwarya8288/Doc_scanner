# Document Scanner - Setup Guide

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
# Using venv (Python 3.8+)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

---

## 📁 Project Structure

```
document-scanner/
├── document_scanner.py          # Main scanner implementation
├── test_image_generator.py      # Test image generator
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── test_images/               # Generated test cases
├── test_results/              # Scanner outputs
└── examples/                  # Usage examples
```

---

## 🔧 Installation Options

### Option A: Minimal Installation (Core Only)
```bash
pip install opencv-python numpy matplotlib Pillow
```

### Option B: Full Installation (Recommended)
```bash
pip install -r requirements.txt
```

### Option C: With Deep Learning Support
```bash
pip install -r requirements.txt
pip install tensorflow onnxruntime
```

---

## 🐍 Python Version Requirements

- **Minimum**: Python 3.8
- **Recommended**: Python 3.9 or 3.10
- **Tested on**: Python 3.9.13

---

## 💻 Platform-Specific Notes

### Windows
```bash
# If you encounter DLL errors, install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### macOS
```bash
# Install Homebrew first (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-dev
sudo apt-get install libopencv-dev python3-opencv
```

---

## 🧪 Testing Installation

### Run Basic Test
```bash
# Generate test images
python test_image_generator.py

# Test scanner
python batch_test.py
```

### Quick Validation
```python
from document_scanner import DocumentScanner
import cv2

# Test with generated image
scanner = DocumentScanner()
img = cv2.imread('test_images/01_perfect.jpg')
scanned, metadata = scanner.scan(img)

if scanned is not None:
    print("✓ Installation successful!")
    cv2.imwrite('test_output.jpg', scanned)
else:
    print("✗ Installation issue detected")
```

---

## 🔍 Troubleshooting

### Issue: "No module named 'cv2'"
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.1.78
```

### Issue: Import errors with NumPy
```bash
pip install --upgrade numpy
```

### Issue: matplotlib backend errors
```bash
# Add to your script
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
```

### Issue: Memory errors with large images
```python
# Reduce resize width in config
config = ScannerConfig(resize_width=600)
scanner = DocumentScanner(config)
```

---

## 🚢 Deployment Options

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Build & Run
```bash
docker build -t document-scanner .
docker run -p 8000:8000 document-scanner
```

---

## 📚 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Homography Explained](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

---

## 🤝 Support

If you encounter issues:
1. Check Python version: `python --version`
2. Verify pip version: `pip --version`
3. Update pip: `pip install --upgrade pip`
4. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

---
