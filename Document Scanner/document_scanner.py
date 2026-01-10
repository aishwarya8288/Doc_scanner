import cv2
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ScannerConfig:
    """Configuration for document scanner"""
    resize_width: int = 800
    blur_kernel: int = 5
    canny_low: int = 30
    canny_high: int = 100
    min_area_ratio: float = 0.1
    epsilon_factor: float = 0.02
    output_width: int = 850
    output_height: int = 1100


class DocumentScanner:
    """Industry-grade document scanner using homography"""
    
    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image: resize and denoise
        Returns preprocessed image and scale ratio
        """
        h, w = image.shape[:2]
        ratio = self.config.resize_width / w
        
        if w > self.config.resize_width:
            image = cv2.resize(image, (self.config.resize_width, 
                                      int(h * ratio)))
        else:
            ratio = 1.0
        
        # Denoise
        denoised = cv2.GaussianBlur(image, 
                                    (self.config.blur_kernel, 
                                     self.config.blur_kernel), 0)
        return denoised, ratio
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale edge detection with morphological operations
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Canny edge detection
        edges = cv2.Canny(filtered, 
                         self.config.canny_low, 
                         self.config.canny_high)
        
        # Dilate to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def find_document_contour(self, edges: np.ndarray, 
                             image_area: float) -> Optional[np.ndarray]:
        """
        Find the largest quadrilateral contour representing the document
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        min_area = image_area * self.config.min_area_ratio
        
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 
                                     self.config.epsilon_factor * peri, 
                                     True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                return approx
        
        return None
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in consistent manner: top-left, top-right, 
        bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        pts = pts.reshape(4, 2)
        
        # Sum: top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference: top-right has smallest diff, bottom-left largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def refine_corners(self, gray: np.ndarray, 
                       corners: np.ndarray) -> np.ndarray:
        """
        Refine corner locations to sub-pixel accuracy
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                   30, 0.001)
        
        corners = corners.reshape(-1, 1, 2).astype(np.float32)
        refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), 
                                   criteria)
        
        return refined.reshape(4, 2)
    
    def compute_homography(self, src_pts: np.ndarray, 
                          dst_pts: np.ndarray) -> np.ndarray:
        """
        Compute homography matrix using RANSAC
        """
        H, mask = cv2.findHomography(src_pts, dst_pts, 
                                     cv2.RANSAC, 5.0)
        return H
    
    def get_destination_points(self, ordered_pts: np.ndarray) -> np.ndarray:
        """
        Compute destination points for perspective transform
        """
        (tl, tr, br, bl) = ordered_pts
        
        # Compute width
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))
        
        # Compute height
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))
        
        # Use standard aspect ratio if needed
        if max_width == 0 or max_height == 0:
            max_width = self.config.output_width
            max_height = self.config.output_height
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        return dst, max_width, max_height
    
    def apply_perspective_transform(self, image: np.ndarray, 
                                   H: np.ndarray, 
                                   width: int, 
                                   height: int) -> np.ndarray:
        """
        Apply perspective transformation
        """
        warped = cv2.warpPerspective(image, H, (width, height), 
                                     flags=cv2.INTER_LINEAR)
        return warped
    
    def enhance_document(self, image: np.ndarray) -> np.ndarray:
        """
        Post-process scanned document: adaptive thresholding and enhancement
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return enhanced
    
    def scan(self, image: np.ndarray, 
            enhance: bool = True) -> Tuple[Optional[np.ndarray], dict]:
        """
        Complete document scanning pipeline
        
        Args:
            image: Input BGR image
            enhance: Apply post-processing enhancement
        
        Returns:
            Scanned document image and metadata
        """
        original = image.copy()
        h, w = image.shape[:2]
        image_area = h * w
        
        # Step 1: Preprocess
        processed, ratio = self.preprocess_image(image)
        
        # Step 2: Edge detection
        edges = self.detect_edges(processed)
        
        # Step 3: Find document contour
        contour = self.find_document_contour(edges, 
                                            processed.shape[0] * processed.shape[1])
        
        if contour is None:
            return None, {"error": "Document not found"}
        
        # Scale corners back to original size
        corners = contour.reshape(4, 2) / ratio
        
        # Step 4: Order corners
        ordered_corners = self.order_points(corners)
        
        # Step 5: Refine corners (optional but recommended)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        refined_corners = self.refine_corners(gray, ordered_corners)
        
        # Step 6: Compute destination points
        dst_points, out_w, out_h = self.get_destination_points(refined_corners)
        
        # Step 7: Compute homography
        H = self.compute_homography(refined_corners, dst_points)
        
        # Step 8: Apply perspective transform
        warped = self.apply_perspective_transform(original, H, out_w, out_h)
        
        # Step 9: Enhancement (optional)
        if enhance:
            enhanced = self.enhance_document(warped)
            final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            final = warped
        
        metadata = {
            "corners": refined_corners.tolist(),
            "homography": H.tolist(),
            "output_size": (out_w, out_h),
            "success": True
        }
        
        return final, metadata
    
    def visualize_detection(self, image: np.ndarray, 
                          corners: np.ndarray) -> np.ndarray:
        """
        Visualize detected document corners
        """
        vis = image.copy()
        corners = corners.astype(np.int32)
        
        # Draw contour
        cv2.drawContours(vis, [corners], -1, (0, 255, 0), 3)
        
        # Draw corners
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(vis, labels[i], (x + 15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis


def main():
    """
    Batch document scanning from input folder to output folder
    """

    input_dir = Path("./test_images")
    output_dir = Path("./scanned_outputs")

    output_dir.mkdir(parents=True, exist_ok=True)

    scanner = DocumentScanner()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("❌ No images found in input directory")
        return

    print(f"📂 Found {len(image_files)} images")
    success_count = 0
    failure_count = 0

    for idx, image_path in enumerate(sorted(image_files), start=1):
        print(f"\n🔍 [{idx}/{len(image_files)}] Processing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("⚠️  Could not read image — skipping")
            failure_count += 1
            continue

        scanned, metadata = scanner.scan(image, enhance=True)

        if scanned is None:
            print(f"❌ Scan failed: {metadata.get('error')}")
            failure_count += 1
            continue

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), scanned)

        print(f"✅ Saved: {output_path}")
        success_count += 1

    print("\n==================== SUMMARY ====================")
    print(f"✅ Successful scans : {success_count}")
    print(f"❌ Failed scans     : {failure_count}")
    print(f"📁 Output directory : {output_dir.resolve()}")
    print("=================================================")


if __name__ == "__main__":
    main()
