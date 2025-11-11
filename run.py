"""
Simple script to run MRZ detection and OCR on a single image
Usage: python run.py <image_path>
"""

import sys
import argparse
from pathlib import Path
from mrz_detector import MRZDetector


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Detect and extract MRZ text from passport image"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to passport image"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to model weights (default: weights/best.pt)"
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped MRZ regions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_results",
        help="Directory to save outputs (default: inference_results)"
    )
    parser.add_argument(
        "--merge-boxes",
        action="store_true",
        help="Merge all detected bounding boxes into one (useful for 2-line MRZ)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model weights not found: {model_path}")
        print("   Please run setup.py first to download the model")
        sys.exit(1)
    
    print("="*60)
    print("MRZ DETECTION AND OCR")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence: {args.confidence}")
    print(f"Merge boxes: {args.merge_boxes}")
    print("="*60 + "\n")
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = MRZDetector(
            model_path=str(model_path),
            confidence=args.confidence
        )
        
        # Process image
        print(f"Processing image: {image_path.name}")
        result = detector.extract_text_from_mrz(str(image_path), merge_boxes=args.merge_boxes)
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Number of MRZ detections: {result['num_detections']}")
        
        if result['num_detections'] == 0:
            print("\n⚠️  No MRZ regions detected in the image")
            print("   Try lowering the confidence threshold with --confidence")
        else:
            for mrz in result['mrz_regions']:
                if 'original_bboxes' in mrz:
                    print(f"\n📍 Merged MRZ Region:")
                    print(f"   Original detections: {len(mrz['original_bboxes'])}")
                    print(f"   Original bboxes: {mrz['original_bboxes']}")
                    print(f"   Merged bbox: {mrz['bbox']}")
                    print(f"   Average confidence: {mrz['detection_confidence']:.4f} ({mrz['detection_confidence']*100:.2f}%)")
                else:
                    print(f"\n📍 MRZ Region {mrz['detection_id']}:")
                    print(f"   Detection confidence: {mrz['detection_confidence']:.4f} ({mrz['detection_confidence']*100:.2f}%)")
                    print(f"   Bounding box: {mrz['bbox']}")
                print(f"\n📝 Extracted Text:")
                print(f"   {mrz['full_text']}")
                
                if len(mrz['ocr_results']) > 1:
                    print(f"\n   Detailed OCR results:")
                    for i, ocr_item in enumerate(mrz['ocr_results'], 1):
                        print(f"      {i}. \"{ocr_item['text']}\" (conf: {ocr_item['confidence']:.3f})")
            
            # Save crops if requested
            if args.save_crop:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import cv2
                image = cv2.imread(str(image_path))
                
                for mrz in result['mrz_regions']:
                    # Use the appropriate bbox
                    x1, y1, x2, y2 = mrz.get('expanded_bbox', mrz['bbox'])
                    crop = image[y1:y2, x1:x2]
                    
                    crop_filename = f"{image_path.stem}_mrz_{mrz['detection_id']}.png"
                    crop_path = output_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    print(f"\n✓ Saved MRZ crop to: {crop_path}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
