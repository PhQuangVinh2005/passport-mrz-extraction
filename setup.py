"""
Setup script for MRZ Detection and OCR
Downloads model weights from Google Drive and installs dependencies
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

def install_requirements():
    """Install required packages"""
    print("="*60)
    print("Installing required packages...")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing packages: {e}")
        sys.exit(1)


def download_model_weights():
    """Download model weights from Google Drive"""
    print("\n" + "="*60)
    print("Downloading model weights from Google Drive...")
    print("="*60)
    
    # Create weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Load .env (if exists) and get GDRIVE_FILE_ID from environment
    load_dotenv()
    GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")

    if not GDRIVE_FILE_ID:
        print("\n⚠️  WARNING: Google Drive file ID not configured in .env!")
        print("   Please create a .env file in the project root with the following:")
        print("     GDRIVE_FILE_ID=your_google_drive_file_id")
        print("   Or manually download the model weights to the folder below:")
        print(f"   {weights_dir.absolute() / 'best.pt'}")
        return
    
    output_path = weights_dir / "best.pt"
    
    if output_path.exists():
        print(f"✓ Model weights already exist at: {output_path}")
        overwrite = input("  Do you want to re-download? (y/n): ").lower()
        if overwrite != 'y':
            print("  Skipping download.")
            return
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(output_path), quiet=False)
        if output_path.exists():
            print(f"\n✓ Model weights downloaded to: {output_path}")
        else:
            print(f"\n❌ Download finished but file not found at: {output_path}")
            print("   Please verify the GDRIVE_FILE_ID and try again or download manually.")
    except Exception as e:
        print(f"\n❌ Error downloading weights: {e}")
        print("   Please manually download the model weights and place in:")
        print(f"   {output_path}")


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("MRZ DETECTION AND OCR - SETUP")
    print("="*60)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Download model weights
    download_model_weights()
    
    print("\n" + "="*60)
    print("✓ Setup completed!")
    print("="*60)
    print("\nNext steps:")
    print("  1. If model download failed, manually place 'best.pt' in 'weights/' folder")
    print("  2. Run inference: python run.py <image_path>")
    print("  3. Example: python run.py sample.jpg")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
