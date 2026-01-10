import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class DocumentTestGenerator:
    """Generate synthetic test images for document scanner"""
    
    def __init__(self, output_dir='test_images'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_document_content(self, width=850, height=1100):
        """Create a realistic document with text and content"""
        # Create white background
        doc = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(doc, "SAMPLE DOCUMENT", (50, 100),
                   cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
        
        # Add horizontal line under title
        cv2.line(doc, (50, 120), (width-50, 120), (0, 0, 0), 2)
        
        # Add paragraphs of text
        y_pos = 180
        line_height = 40
        
        paragraphs = [
            "This is a test document for evaluating",
            "the document scanner system.",
            "",
            "The scanner should be able to:",
            "1. Detect document boundaries accurately",
            "2. Compute correct homography matrix",
            "3. Handle various orientations",
            "4. Deal with lighting variations",
            "5. Remove background clutter",
            "",
            "Testing is crucial for robust systems.",
            "Multiple test cases ensure reliability.",
        ]
        
        for line in paragraphs:
            cv2.putText(doc, line, (80, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_pos += line_height
        
        # Add a box with information
        cv2.rectangle(doc, (50, y_pos + 20), (width-50, y_pos + 150),
                     (0, 0, 0), 2)
        cv2.putText(doc, "IMPORTANT NOTICE", (70, y_pos + 60),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(doc, "This document contains test data", (70, y_pos + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add footer
        cv2.putText(doc, "Page 1 of 1", (width//2 - 100, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        return doc
    
    def apply_perspective_transform(self, img, angle=0, scale=0.7, 
                                   shift_x=0, shift_y=0):
        """Apply perspective transformation to simulate camera view"""
        h, w = img.shape[:2]
        
        # Define source points (document corners)
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Apply rotation
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate around center
        center_x, center_y = w / 2, h / 2
        rotated_pts = []
        
        for x, y in src_pts:
            # Translate to origin
            x_temp = x - center_x
            y_temp = y - center_y
            
            # Rotate
            x_rot = x_temp * cos_a - y_temp * sin_a
            y_rot = x_temp * sin_a + y_temp * cos_a
            
            # Translate back and scale
            x_final = (x_rot * scale) + center_x + shift_x
            y_final = (y_rot * scale) + center_y + shift_y
            
            rotated_pts.append([x_final, y_final])
        
        dst_pts = np.float32(rotated_pts)
        
        # Calculate canvas size to fit rotated document
        canvas_w = int(w * 1.5)
        canvas_h = int(h * 1.5)
        offset_x = (canvas_w - w) // 2
        offset_y = (canvas_h - h) // 2
        
        # Adjust destination points
        dst_pts += [offset_x, offset_y]
        
        # Compute homography
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp image
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h),
                                     borderValue=(200, 200, 200))
        
        return warped
    
    def add_background(self, img, bg_type='wood'):
        """Add background texture"""
        h, w = img.shape[:2]
        
        if bg_type == 'wood':
            # Create wood-like texture
            background = np.random.randint(120, 180, (h, w, 3), dtype=np.uint8)
            # Add grain
            grain = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
            background = np.clip(background.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        elif bg_type == 'fabric':
            # Fabric-like texture
            background = np.ones((h, w, 3), dtype=np.uint8) * 180
            noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
            background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif bg_type == 'desk':
            # Desk with items
            background = np.ones((h, w, 3), dtype=np.uint8) * 160
            # Add some random rectangles (books, etc.)
            for _ in range(5):
                x1 = np.random.randint(0, w-100)
                y1 = np.random.randint(0, h-100)
                x2 = x1 + np.random.randint(50, 200)
                y2 = y1 + np.random.randint(50, 200)
                color = tuple(np.random.randint(100, 200, 3).tolist())
                cv2.rectangle(background, (x1, y1), (x2, y2), color, -1)
        
        else:  # plain
            background = np.ones((h, w, 3), dtype=np.uint8) * 200
        
        # Blend document with background
        # Find document region (non-gray areas in img)
        mask = np.any(img != [200, 200, 200], axis=-1).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = mask[:, :, np.newaxis] / 255.0
        
        result = (img * mask + background * (1 - mask)).astype(np.uint8)
        
        return result
    
    def add_shadow(self, img, intensity=0.4):
        """Add shadow to document"""
        h, w = img.shape[:2]
        
        # Create shadow mask (gradient from one corner)
        shadow = np.ones((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i/h)**2 + (j/w)**2)
                shadow[i, j] = 1 - (intensity * dist)
        
        shadow = np.clip(shadow, 0.6, 1.0)
        shadow = shadow[:, :, np.newaxis]
        
        result = (img * shadow).astype(np.uint8)
        return result
    
    def add_noise(self, img, noise_level=15):
        """Add gaussian noise"""
        noise = np.random.normal(0, noise_level, img.shape)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def adjust_brightness(self, img, factor=1.2):
        """Adjust brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def generate_test_images(self):
        """Generate complete test suite"""
        print("Generating test images...")
        
        # Create base document
        doc = self.create_document_content()
        
        # Test 1: Perfect front-facing document
        print("1. Creating perfect front-facing document...")
        cv2.imwrite(f"{self.output_dir}/01_perfect.jpg", doc)
        
        # Test 2: Slight rotation (15 degrees)
        print("2. Creating slightly rotated document...")
        rotated = self.apply_perspective_transform(doc, angle=15, scale=0.8)
        rotated_bg = self.add_background(rotated, 'wood')
        cv2.imwrite(f"{self.output_dir}/02_rotated_15deg.jpg", rotated_bg)
        
        # Test 3: Heavy rotation (45 degrees)
        print("3. Creating heavily rotated document...")
        rotated_45 = self.apply_perspective_transform(doc, angle=45, scale=0.7)
        rotated_45_bg = self.add_background(rotated_45, 'fabric')
        cv2.imwrite(f"{self.output_dir}/03_rotated_45deg.jpg", rotated_45_bg)
        
        # Test 4: Document with shadow
        print("4. Creating document with shadow...")
        shadow_doc = self.apply_perspective_transform(doc, angle=20, scale=0.75)
        shadow_doc = self.add_shadow(shadow_doc, intensity=0.5)
        shadow_doc_bg = self.add_background(shadow_doc, 'wood')
        cv2.imwrite(f"{self.output_dir}/04_with_shadow.jpg", shadow_doc_bg)
        
        # Test 5: Low light conditions
        print("5. Creating low light document...")
        low_light = self.apply_perspective_transform(doc, angle=-25, scale=0.7)
        low_light_bg = self.add_background(low_light, 'desk')
        low_light_bg = self.adjust_brightness(low_light_bg, factor=0.6)
        cv2.imwrite(f"{self.output_dir}/05_low_light.jpg", low_light_bg)
        
        # Test 6: Cluttered background
        print("6. Creating document with cluttered background...")
        cluttered = self.apply_perspective_transform(doc, angle=30, scale=0.65)
        cluttered_bg = self.add_background(cluttered, 'desk')
        cv2.imwrite(f"{self.output_dir}/06_cluttered_background.jpg", cluttered_bg)
        
        # Test 7: Noisy image
        print("7. Creating noisy document...")
        noisy = self.apply_perspective_transform(doc, angle=-15, scale=0.7)
        noisy_bg = self.add_background(noisy, 'wood')
        noisy_bg = self.add_noise(noisy_bg, noise_level=20)
        cv2.imwrite(f"{self.output_dir}/07_noisy.jpg", noisy_bg)
        
        # Test 8: High brightness
        print("8. Creating high brightness document...")
        bright = self.apply_perspective_transform(doc, angle=10, scale=0.75)
        bright_bg = self.add_background(bright, 'wood')
        bright_bg = self.adjust_brightness(bright_bg, factor=1.5)
        cv2.imwrite(f"{self.output_dir}/08_high_brightness.jpg", bright_bg)
        
        # Test 9: Perspective distortion
        print("9. Creating perspective distorted document...")
        h, w = doc.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[100, 50], [w-50, 100], [w-100, h-50], [50, h-100]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        perspective = cv2.warpPerspective(doc, M, (w+200, h+200),
                                         borderValue=(200, 200, 200))
        perspective_bg = self.add_background(perspective, 'fabric')
        cv2.imwrite(f"{self.output_dir}/09_perspective_distortion.jpg", perspective_bg)
        
        # Test 10: Combined challenges
        print("10. Creating document with multiple challenges...")
        combined = self.apply_perspective_transform(doc, angle=35, scale=0.65)
        combined = self.add_shadow(combined, intensity=0.6)
        combined_bg = self.add_background(combined, 'desk')
        combined_bg = self.add_noise(combined_bg, noise_level=15)
        combined_bg = self.adjust_brightness(combined_bg, factor=0.8)
        cv2.imwrite(f"{self.output_dir}/10_combined_challenges.jpg", combined_bg)
        
        print(f"\n✓ All test images generated in '{self.output_dir}/' directory")
        print("\nTest cases created:")
        print("  1. Perfect front-facing (baseline)")
        print("  2. 15° rotation")
        print("  3. 45° rotation")
        print("  4. With shadow")
        print("  5. Low light")
        print("  6. Cluttered background")
        print("  7. Noisy image")
        print("  8. High brightness")
        print("  9. Perspective distortion")
        print("  10. Multiple challenges combined")


def create_batch_test_script():
    """Create a script to test all generated images"""
    script = '''import cv2
import os
from document_scanner import DocumentScanner
import matplotlib.pyplot as plt

def test_all_images():
    """Test scanner on all generated images"""
    scanner = DocumentScanner()
    test_dir = 'test_images'
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    
    results = []
    
    for img_file in image_files:
        print(f"\\nProcessing: {img_file}")
        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ✗ Failed to load image")
            continue
        
        # Scan document
        scanned, metadata = scanner.scan(img, enhance=True)
        
        if scanned is None:
            print(f"  ✗ Scanning failed: {metadata.get('error')}")
            results.append((img_file, False))
            continue
        
        print(f"  ✓ Success! Output size: {metadata['output_size']}")
        results.append((img_file, True))
        
        # Save result
        output_path = os.path.join(output_dir, f"scanned_{img_file}")
        cv2.imwrite(output_path, scanned)
        
        # Visualize
        corners = metadata.get('corners')
        if corners:
            vis = scanner.visualize_detection(img, corners)
            vis_path = os.path.join(output_dir, f"detection_{img_file}")
            cv2.imwrite(vis_path, vis)
    
    # Summary
    print("\\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Success rate: {successful}/{total} ({100*successful/total:.1f}%)")
    print("\\nDetailed results:")
    for filename, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")

if __name__ == "__main__":
    test_all_images()
'''
    
    with open("batch_test.py", "w", encoding="utf-8") as f:

        f.write(script)
    
    print("\n✓ Created 'batch_test.py' for automated testing")


if __name__ == "__main__":
    # Generate test images
    generator = DocumentTestGenerator()
    generator.generate_test_images()
    
    # Create batch test script
    create_batch_test_script()
    
    print("\n" + "="*60)
    print("USAGE:")
    print("="*60)
    print("1. Run this script to generate test images")
    print("2. Run 'python batch_test.py' to test scanner on all images")
    print("3. Check 'test_results/' for output")