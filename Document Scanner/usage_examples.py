"""
Document Scanner - Usage Examples
Complete examples for different use cases
"""

import cv2
import numpy as np
from document_scanner import DocumentScanner, ScannerConfig
import os


# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================

def example_basic():
    """Simple document scanning"""
    print("="*60)
    print("EXAMPLE 1: Basic Document Scanning")
    print("="*60)
    
    # Initialize scanner with default settings
    scanner = DocumentScanner()
    
    # Load image
    image = cv2.imread('test_images/02_rotated_15deg.jpg')
    
    # Scan document
    scanned, metadata = scanner.scan(image, enhance=True)
    
    if scanned is not None:
        print("✓ Scan successful!")
        print(f"Output size: {metadata['output_size']}")
        cv2.imwrite('output_basic.jpg', scanned)
    else:
        print(f"✗ Scan failed: {metadata.get('error')}")


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

def example_custom_config():
    """Using custom scanner configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = ScannerConfig(
        resize_width=1000,      # Higher resolution processing
        blur_kernel=7,          # More aggressive blur
        canny_low=50,           # Adjusted edge detection
        canny_high=150,
        min_area_ratio=0.15,    # Require larger documents
        output_width=1200,      # Larger output
        output_height=1600
    )
    
    # Initialize with custom config
    scanner = DocumentScanner(config)
    
    image = cv2.imread('test_images/06_cluttered_background.jpg')
    scanned, metadata = scanner.scan(image, enhance=True)
    
    if scanned is not None:
        print("✓ Custom scan successful!")
        cv2.imwrite('output_custom.jpg', scanned)


# ============================================================================
# EXAMPLE 3: Batch Processing
# ============================================================================

def example_batch_processing():
    """Process multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    scanner = DocumentScanner()
    input_dir = 'test_images'
    output_dir = 'batch_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    success_count = 0
    
    for img_file in image_files:
        print(f"\nProcessing: {img_file}")
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        scanned, metadata = scanner.scan(img, enhance=True)
        
        if scanned is not None:
            output_path = os.path.join(output_dir, f"scanned_{img_file}")
            cv2.imwrite(output_path, scanned)
            print(f"  ✓ Saved to {output_path}")
            success_count += 1
        else:
            print(f"  ✗ Failed: {metadata.get('error')}")
    
    print(f"\n✓ Processed {success_count}/{len(image_files)} images")


# ============================================================================
# EXAMPLE 4: Video Stream Processing (Webcam)
# ============================================================================

def example_video_stream():
    """Real-time document scanning from webcam"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Video Stream Processing")
    print("="*60)
    print("Press 'c' to capture, 'q' to quit")
    
    scanner = DocumentScanner()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Try to detect document in real-time (without full scan)
        processed, _ = scanner.preprocess_image(frame)
        edges = scanner.detect_edges(processed)
        
        # Display edge detection
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, "Press 'c' to capture", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Document Scanner', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture and scan
            print("Capturing...")
            scanned, metadata = scanner.scan(frame, enhance=True)
            
            if scanned is not None:
                cv2.imwrite('captured_document.jpg', scanned)
                print("✓ Document captured and saved!")
                cv2.imshow('Scanned', scanned)
                cv2.waitKey(2000)
            else:
                print("✗ No document detected")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# EXAMPLE 5: Detection Visualization
# ============================================================================

def example_visualization():
    """Visualize the detection process"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Detection Visualization")
    print("="*60)
    
    scanner = DocumentScanner()
    image = cv2.imread('test_images/03_rotated_45deg.jpg')
    
    # Get intermediate results
    processed, ratio = scanner.preprocess_image(image)
    edges = scanner.detect_edges(processed)
    
    # Scan document
    scanned, metadata = scanner.scan(image, enhance=False)
    
    if scanned is not None:
        # Create visualization
        corners = np.array(metadata['corners'])
        vis = scanner.visualize_detection(image, corners)
        
        # Create side-by-side comparison
        h1, w1 = image.shape[:2]
        h2, w2 = scanned.shape[:2]
        
        # Resize for consistent display
        max_h = max(h1, h2)
        img_resized = cv2.resize(image, (int(w1 * max_h / h1), max_h))
        scanned_resized = cv2.resize(scanned, (int(w2 * max_h / h2), max_h))
        
        combined = np.hstack([img_resized, scanned_resized])
        
        cv2.imwrite('visualization_original.jpg', image)
        cv2.imwrite('visualization_edges.jpg', edges)
        cv2.imwrite('visualization_detection.jpg', vis)
        cv2.imwrite('visualization_comparison.jpg', combined)
        
        print("✓ Visualizations saved!")
        print("  - visualization_original.jpg")
        print("  - visualization_edges.jpg")
        print("  - visualization_detection.jpg")
        print("  - visualization_comparison.jpg")


# ============================================================================
# EXAMPLE 6: Error Handling
# ============================================================================

def example_error_handling():
    """Proper error handling"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Handling")
    print("="*60)
    
    scanner = DocumentScanner()
    
    test_cases = [
        'test_images/01_perfect.jpg',
        'nonexistent.jpg',
        'test_images/10_combined_challenges.jpg'
    ]
    
    for img_path in test_cases:
        print(f"\nTesting: {img_path}")
        
        try:
            # Try to load image
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"  ✗ Failed to load image")
                continue
            
            # Try to scan
            scanned, metadata = scanner.scan(image, enhance=True)
            
            if scanned is not None:
                print(f"  ✓ Success! Size: {metadata['output_size']}")
            else:
                print(f"  ✗ Scan failed: {metadata.get('error')}")
                
                # Try with different config
                print("  → Retrying with adjusted settings...")
                config = ScannerConfig(min_area_ratio=0.05)
                scanner2 = DocumentScanner(config)
                scanned, metadata = scanner2.scan(image, enhance=True)
                
                if scanned is not None:
                    print(f"  ✓ Success on retry!")
                else:
                    print(f"  ✗ Still failed")
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")


# ============================================================================
# EXAMPLE 7: Quality Metrics
# ============================================================================

def example_quality_metrics():
    """Compute quality metrics for scanned document"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Quality Metrics")
    print("="*60)
    
    scanner = DocumentScanner()
    image = cv2.imread('test_images/04_with_shadow.jpg')
    
    scanned, metadata = scanner.scan(image, enhance=True)
    
    if scanned is not None:
        # Compute metrics
        gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        contrast = gray.std()
        
        # Brightness (mean)
        brightness = gray.mean()
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (edges > 0).sum() / edges.size
        
        print(f"Quality Metrics:")
        print(f"  Sharpness:    {sharpness:.2f}")
        print(f"  Contrast:     {contrast:.2f}")
        print(f"  Brightness:   {brightness:.2f}")
        print(f"  Edge Density: {edge_density:.4f}")
        
        # Quality assessment
        if sharpness > 100 and contrast > 50:
            print("\n✓ Good quality scan!")
        else:
            print("\n⚠ Low quality scan - consider retaking")


# ============================================================================
# EXAMPLE 8: Saving Metadata
# ============================================================================

def example_save_metadata():
    """Save scan metadata for later use"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Saving Metadata")
    print("="*60)
    
    import json
    
    scanner = DocumentScanner()
    image = cv2.imread('test_images/02_rotated_15deg.jpg')
    
    scanned, metadata = scanner.scan(image, enhance=True)
    
    if scanned is not None:
        # Save image
        output_path = 'output_with_metadata.jpg'
        cv2.imwrite(output_path, scanned)
        
        # Save metadata
        metadata_path = 'output_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved image: {output_path}")
        print(f"✓ Saved metadata: {metadata_path}")
        print(f"\nMetadata:")
        print(json.dumps(metadata, indent=2))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run all examples
    examples = [
        ("Basic Usage", example_basic),
        ("Custom Configuration", example_custom_config),
        ("Batch Processing", example_batch_processing),
        ("Detection Visualization", example_visualization),
        ("Error Handling", example_error_handling),
        ("Quality Metrics", example_quality_metrics),
        ("Save Metadata", example_save_metadata),
    ]
    
    print("\n" + "="*60)
    print("DOCUMENT SCANNER - USAGE EXAMPLES")
    print("="*60)
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
    
    print("\n" + "="*60)
    choice = input("\nSelect example (1-8, or 0 for all): ").strip()
    
    if choice == '0':
        for name, func in examples:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice)-1][1]()
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)