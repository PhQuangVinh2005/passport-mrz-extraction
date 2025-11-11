import pytest
from pathlib import Path
from mrz_detector import MRZDetector


def test_missing_model_raises():
    # Ensure that when model path doesn't exist, FileNotFoundError is raised
    fake_model = "weights/nonexistent_best.pt"
    # Make sure it's not accidentally present in workspace
    if Path(fake_model).exists():
        pytest.skip("Test skipped because fake model exists in workspace")

    with pytest.raises(FileNotFoundError):
        MRZDetector(model_path=fake_model)
