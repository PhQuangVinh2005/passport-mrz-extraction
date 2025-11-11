# MRZ Detection and OCR

Dự án Python đơn giản để phát hiện vùng Machine Readable Zone (MRZ) trên ảnh hộ chiếu và trích xuất text bằng OCR.

Simple Python project to detect Machine Readable Zone (MRZ) in passport images and extract text using OCR.

## Features

- ✅ Detect MRZ regions using YOLOv8
- ✅ Extract text from MRZ using EasyOCR
- ✅ Simple command-line interface
- ✅ No Docker or complex setup required
- ✅ Save cropped MRZ regions

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

## Installation

### Step 1: Clone repository

```bash
git clone https://github.com/PhQuangVinh2005/passport-mrz-extraction.git
cd passport-mrz-extraction
```

### Step 2: Run setup

```bash
python setup.py
```

This will:
- Install all required packages
- Download model weights from Google Drive (if configured)

**⚠️ Important:** Before running setup, copy `.env.example` to `.env` and set your Google Drive file ID:

1. Upload your `best.pt` model to Google Drive
2. Get shareable link: Right-click file → Share → Copy link
3. Extract file ID from link: `https://drive.google.com/file/d/FILE_ID/view`
4. Create `.env` in the project root (or copy `.env.example`) and set:

```
GDRIVE_FILE_ID=YOUR_GOOGLE_DRIVE_FILE_ID
```

Then run:

```powershell
python setup.py
```

**Alternative:** Manually place `best.pt` in `weights/` folder

## Usage

### Basic usage

```bash
python run.py <image_path>
```

### Examples

```bash
# Process single image
python run.py sample.jpg

# Save cropped MRZ regions
python run.py sample.jpg --save-crop

# Use custom confidence threshold
python run.py sample.jpg --confidence 0.3

# Specify output directory
python run.py sample.jpg --save-crop --output-dir my_results
```

### Advanced usage

```bash
python run.py <image_path> [options]

Options:
  --confidence FLOAT    Confidence threshold (default: 0.25)
  --model PATH          Path to model weights (default: weights/best.pt)
  --save-crop           Save cropped MRZ regions
  --output-dir PATH     Output directory (default: inference_results)
```

## Output

The script will display:
- Number of MRZ detections
- Detection confidence scores
- Bounding box coordinates
- Extracted text from MRZ

Example output:
```
============================================================
RESULTS
============================================================
Number of MRZ detections: 1

📍 MRZ Region 1:
   Detection confidence: 0.9456 (94.56%)
   Bounding box: [123, 456, 789, 567]

📝 Extracted Text:
   P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<
   
   Detailed OCR results:
      1. "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<" (conf: 0.892)
============================================================
```

## Project Structure

```
for_github_repo/
├── setup.py              # Setup script (install dependencies, download model)
├── run.py                # Main script to run inference
├── mrz_detector.py       # MRZ detection and OCR module
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── README.md            # This file
└── weights/             # Model weights (created after setup)
    └── best.pt          # YOLOv8 trained model
```

## Python API Usage

You can also use the module in your own Python code:

```python
from mrz_detector import MRZDetector

# Initialize detector
detector = MRZDetector(model_path="weights/best.pt", confidence=0.25)

# Process image
result = detector.extract_text_from_mrz("passport.jpg")

# Get MRZ text
if result['num_detections'] > 0:
    mrz_text = result['mrz_regions'][0]['full_text']
    print(f"MRZ Text: {mrz_text}")
```

## Troubleshooting

### Model not found
```bash
# Make sure you ran setup.py first
python setup.py

# Or manually place best.pt in weights/ folder
```

### No detections
Try lowering the confidence threshold:
```bash
python run.py image.jpg --confidence 0.15
```

### Slow OCR
Install with GPU support (requires CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Package installation errors
Try upgrading pip:
```bash
python -m pip install --upgrade pip
python setup.py
```

## Dependencies

- ultralytics - YOLOv8 for object detection
- opencv-python - Image processing
- easyocr - Optical Character Recognition
- torch - Deep learning framework
- numpy - Numerical computing
- gdown - Google Drive downloader

## License

MIT License

## Author

Your Name

## Acknowledgments

- YOLOv8 by Ultralytics
- EasyOCR by JaidedAI
