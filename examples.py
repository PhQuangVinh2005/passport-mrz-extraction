"""
Example script demonstrating different use cases of MRZ detector
"""

from pathlib import Path
from mrz_detector import MRZDetector


def example_1_basic_usage():
    """Example 1: Basic usage - process single image"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    detector = MRZDetector()
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        mrz_text = detector.process_image(image_path)
        print(f"Extracted MRZ text: {mrz_text}")
    else:
        print(f"Image not found: {image_path}")


def example_2_detailed_results():
    """Example 2: Get detailed detection results"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Detailed Results")
    print("="*60)
    
    detector = MRZDetector()
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        result = detector.extract_text_from_mrz(image_path)
        
        print(f"Number of detections: {result['num_detections']}")
        
        for mrz in result['mrz_regions']:
            print(f"\nMRZ Region {mrz['detection_id']}:")
            print(f"  Confidence: {mrz['detection_confidence']:.4f}")
            print(f"  Bounding box: {mrz['bbox']}")
            print(f"  Text: {mrz['full_text']}")
    else:
        print(f"Image not found: {image_path}")


def example_3_custom_confidence():
    """Example 3: Use custom confidence threshold"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Confidence Threshold")
    print("="*60)
    
    # Lower confidence = more detections (but possibly false positives)
    detector = MRZDetector(confidence=0.15)
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        result = detector.extract_text_from_mrz(image_path)
        print(f"Detections with 0.15 confidence: {result['num_detections']}")
    else:
        print(f"Image not found: {image_path}")


def example_4_save_crops():
    """Example 4: Save cropped MRZ regions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Save Cropped MRZ Regions")
    print("="*60)
    
    detector = MRZDetector()
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        mrz_text = detector.process_image(
            image_path, 
            save_crop=True, 
            output_dir="example_output"
        )
        print(f"MRZ text: {mrz_text}")
        print("Cropped regions saved to: example_output/")
    else:
        print(f"Image not found: {image_path}")


def example_5_batch_processing():
    """Example 5: Process multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)
    
    detector = MRZDetector()
    
    # List of images to process
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    results = []
    for img_path in image_paths:
        if Path(img_path).exists():
            mrz_text = detector.process_image(img_path)
            results.append({
                'image': img_path,
                'text': mrz_text
            })
            print(f"✓ {img_path}: {mrz_text}")
        else:
            print(f"✗ {img_path}: Not found")
    
    print(f"\nProcessed {len(results)} images")


def example_6_only_detection():
    """Example 6: Only detect MRZ without OCR"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Detection Only (No OCR)")
    print("="*60)
    
    detector = MRZDetector()
    # Don't initialize OCR
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        detections = detector.detect_mrz(image_path)
        
        print(f"Found {len(detections)} MRZ region(s)")
        for i, det in enumerate(detections, 1):
            print(f"  Detection {i}:")
            print(f"    BBox: {det['bbox']}")
            print(f"    Confidence: {det['confidence']:.4f}")
    else:
        print(f"Image not found: {image_path}")


def example_7_merge_boxes():
    """Example 7: Merge multiple detections for 2-line MRZ"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Merge Bounding Boxes (2-line MRZ)")
    print("="*60)
    
    detector = MRZDetector()
    
    image_path = "sample.jpg"
    if Path(image_path).exists():
        # Without merging
        print("\n🔹 Without merging (separate boxes):")
        result_separate = detector.extract_text_from_mrz(image_path, merge_boxes=False)
        print(f"  Number of detections: {result_separate['num_detections']}")
        for mrz in result_separate['mrz_regions']:
            print(f"  Region {mrz['detection_id']}: {mrz['full_text']}")
        
        # With merging
        print("\n🔸 With merging (combined box):")
        result_merged = detector.extract_text_from_mrz(image_path, merge_boxes=True)
        print(f"  Number of detections: {result_merged['num_detections']}")
        for mrz in result_merged['mrz_regions']:
            if 'original_bboxes' in mrz:
                print(f"  Merged from {len(mrz['original_bboxes'])} boxes")
            print(f"  Text: {mrz['full_text']}")
        
        print("\n💡 Benefits of merging:")
        print("  ✓ EasyOCR processes both MRZ lines together")
        print("  ✓ Better context for OCR accuracy")
        print("  ✓ No limitation by individual YOLO detections")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MRZ DETECTOR - USAGE EXAMPLES")
    print("="*60)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_detailed_results()
        example_3_custom_confidence()
        example_4_save_crops()
        example_5_batch_processing()
        example_6_only_detection()
        example_7_merge_boxes()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to run setup.py first and have sample images ready")
