import cv2
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
        print(f"\nProcessing: {img_file}")
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
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Success rate: {successful}/{total} ({100*successful/total:.1f}%)")
    print("\nDetailed results:")
    for filename, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")

if __name__ == "__main__":
    test_all_images()
