"""
MRZ Detection and OCR Module
Detects Machine Readable Zone (MRZ) in passport images and extracts text using OCR
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
from ultralytics import YOLO
import easyocr


class MRZDetector:
    """Class for MRZ detection and OCR extraction"""
    
    def __init__(self, model_path: str = "weights/best.pt", confidence: float = 0.25):
        """
        Initialize MRZ Detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence: Confidence threshold for detection
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self.ocr_reader = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                f"Please run setup.py first to download the model."
            )
        
        print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✓ YOLO model loaded successfully")
        
    def initialize_ocr(self, gpu: bool = False):
        """
        Initialize EasyOCR reader
        
        Args:
            gpu: Whether to use GPU for OCR (requires CUDA)
        """
        if self.ocr_reader is None:
            print("Initializing EasyOCR reader...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=gpu)
            print("✓ EasyOCR reader initialized")
    
    def detect_mrz(self, image_path: Union[str, Path, np.ndarray]) -> List[Dict]:
        """
        Detect MRZ regions in image
        
        Args:
            image_path: Path to image or numpy array
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Load image if path provided
        if isinstance(image_path, (str, Path)):
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            input_img = str(image_path)
        elif isinstance(image_path, np.ndarray):
            input_img = image_path
        else:
            raise TypeError("Input must be image path or numpy array")
        
        # Run inference
        results = self.model(input_img, conf=self.confidence, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })
        
        return detections
    
    def expand_bbox(self, bbox: List[int], image_shape: Tuple[int, int], 
                    expand_ratio: float = 0.2) -> List[int]:
        """
        Expand bounding box by a ratio while keeping within image bounds
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: Image shape (height, width)
            expand_ratio: Expansion ratio (0.2 = 20% padding)
            
        Returns:
            Expanded bounding box [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape[:2]
        
        # Calculate expansion
        width = x2 - x1
        height = y2 - y1
        expand_w = int(width * expand_ratio / 2)
        expand_h = int(height * expand_ratio / 2)
        
        # Apply expansion with bounds checking
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(img_w, x2 + expand_w)
        new_y2 = min(img_h, y2 + expand_h)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def merge_bboxes(self, bboxes: List[List[int]], image_shape: Tuple[int, int],
                     expand_ratio: float = 0.2) -> List[int]:
        """
        Merge multiple bounding boxes into one large box that contains all of them.
        This is useful for MRZ with 2 lines detected separately by YOLO.
        Takes the top-left corner of the leftmost-topmost bbox and bottom-right corner 
        of the rightmost-bottommost bbox to create a larger bounding box.
        
        Args:
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            image_shape: Image shape (height, width, channels)
            expand_ratio: Additional expansion ratio after merging (0.2 = 20%)
            
        Returns:
            Merged and expanded bounding box [x1, y1, x2, y2]
        """
        if len(bboxes) == 0:
            return None
        
        if len(bboxes) == 1:
            # If only one bbox, just expand it
            return self.expand_bbox(bboxes[0], image_shape, expand_ratio)
        
        # Find the extreme coordinates
        min_x1 = min(bbox[0] for bbox in bboxes)  # Leftmost x1
        min_y1 = min(bbox[1] for bbox in bboxes)  # Topmost y1
        max_x2 = max(bbox[2] for bbox in bboxes)  # Rightmost x2
        max_y2 = max(bbox[3] for bbox in bboxes)  # Bottommost y2
        
        # Create merged bbox
        merged_bbox = [min_x1, min_y1, max_x2, max_y2]
        
        # Apply expansion
        expanded_merged = self.expand_bbox(merged_bbox, image_shape, expand_ratio)
        
        return expanded_merged
    
    def extract_text_from_mrz(self, image_path: Union[str, Path], 
                             expand_ratio: float = 0.2,
                             merge_boxes: bool = False) -> Dict:
        """
        Detect MRZ and extract text using OCR
        
        Args:
            image_path: Path to passport image
            expand_ratio: Ratio to expand bbox for better OCR
            merge_boxes: If True, merge all detected bboxes into one large box (useful for 2-line MRZ)
            
        Returns:
            Dictionary with MRZ detections and extracted text
        """
        # Initialize OCR if not already done
        if self.ocr_reader is None:
            self.initialize_ocr()
        
        # Load image
        if isinstance(image_path, (str, Path)):
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.imread(str(image_path))
        else:
            image = image_path
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect MRZ regions
        detections = self.detect_mrz(image_path)
        
        if len(detections) == 0:
            return {
                'image_path': str(image_path),
                'num_detections': 0,
                'mrz_regions': [],
                'merged': False
            }
        
        # Process MRZ regions
        mrz_results = []
        
        if merge_boxes and len(detections) > 1:
            # Merge all bboxes into one large box
            all_bboxes = [det['bbox'] for det in detections]
            merged_bbox = self.merge_bboxes(all_bboxes, image.shape, expand_ratio)
            
            # Calculate average confidence
            avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
            
            x1, y1, x2, y2 = merged_bbox
            mrz_crop = image[y1:y2, x1:x2]
            
            # Apply OCR on merged region
            ocr_results = self.ocr_reader.readtext(mrz_crop)
            
            # Extract text
            ocr_texts = []
            for (ocr_bbox, text, ocr_conf) in ocr_results:
                ocr_texts.append({
                    'text': text,
                    'confidence': ocr_conf
                })
            
            # Combine all text
            full_text = ' '.join([item['text'] for item in ocr_texts])
            
            mrz_results.append({
                'detection_id': 1,
                'bbox': merged_bbox,
                'original_bboxes': all_bboxes,  # Store original bboxes
                'detection_confidence': avg_confidence,
                'ocr_results': ocr_texts,
                'full_text': full_text
            })
        else:
            # Process each detection separately
            for idx, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                # Expand bbox for better OCR
                expanded_bbox = self.expand_bbox(bbox, image.shape, expand_ratio)
                x1, y1, x2, y2 = expanded_bbox
                
                # Crop MRZ region
                mrz_crop = image[y1:y2, x1:x2]
                
                # Apply OCR
                ocr_results = self.ocr_reader.readtext(mrz_crop)
                
                # Extract text
                ocr_texts = []
                for (ocr_bbox, text, ocr_conf) in ocr_results:
                    ocr_texts.append({
                        'text': text,
                        'confidence': ocr_conf
                    })
                
                # Combine all text
                full_text = ' '.join([item['text'] for item in ocr_texts])
                
                mrz_results.append({
                    'detection_id': idx + 1,
                    'bbox': bbox,
                    'expanded_bbox': expanded_bbox,
                    'detection_confidence': confidence,
                    'ocr_results': ocr_texts,
                    'full_text': full_text
                })
        
        return {
            'image_path': str(image_path),
            'num_detections': len(detections),
            'mrz_regions': mrz_results,
            'merged': merge_boxes and len(detections) > 1
        }
    
    def process_image(self, image_path: Union[str, Path], 
                     save_crop: bool = False, 
                     output_dir: str = "inference_results",
                     merge_boxes: bool = False) -> str:
        """
        Process single image and return MRZ text
        
        Args:
            image_path: Path to passport image
            save_crop: Whether to save cropped MRZ regions
            output_dir: Directory to save outputs
            merge_boxes: If True, merge all detected bboxes into one (useful for 2-line MRZ)
            
        Returns:
            Extracted MRZ text (or empty string if no detection)
        """
        result = self.extract_text_from_mrz(image_path, merge_boxes=merge_boxes)
        
        if result['num_detections'] == 0:
            return ""
        
        # Save crops if requested
        if save_crop:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image = cv2.imread(str(image_path))
            for mrz in result['mrz_regions']:
                # Use bbox directly (it's already expanded or merged)
                x1, y1, x2, y2 = mrz.get('expanded_bbox', mrz['bbox'])
                crop = image[y1:y2, x1:x2]
                
                crop_filename = f"{Path(image_path).stem}_mrz_{mrz['detection_id']}.png"
                crop_path = output_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)
        
        # Return text from first detection
        return result['mrz_regions'][0]['full_text']


def main():
    """Example usage"""
    # Initialize detector
    detector = MRZDetector(model_path="weights/best.pt", confidence=0.25)
    
    # Process image
    test_image = "sample.jpg"
    if Path(test_image).exists():
        mrz_text = detector.process_image(test_image, save_crop=True)
        print(f"Extracted MRZ text: {mrz_text}")
    else:
        print(f"Image not found: {test_image}")


if __name__ == "__main__":
    main()
