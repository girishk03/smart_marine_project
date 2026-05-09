import os
import pytest


def test_missing_weights_raises_file_not_found():
    from plastic_detector import PlasticDetector

    missing_path = os.path.join(os.path.dirname(__file__), "_does_not_exist.pt")
    with pytest.raises(FileNotFoundError):
        PlasticDetector(model_path=missing_path, device="cpu")


def test_detector_class_is_importable():
    from plastic_detector import PlasticDetector

    assert PlasticDetector is not None
