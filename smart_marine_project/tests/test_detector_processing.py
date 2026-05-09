import os
from pathlib import Path

import numpy as np
import pytest


def test_process_image_missing_file_raises():
    from plastic_detector import PlasticDetector

    missing_weights = os.path.join(os.path.dirname(__file__), "_no_weights.pt")
    with pytest.raises(FileNotFoundError):
        PlasticDetector(model_path=missing_weights, device="cpu")


def test_process_image_invalid_image_raises_value_error(tmp_path: Path):
    from plastic_detector import PlasticDetector

    # Skip test if weights are not present (we need a real detector instance)
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "ocean_waste_model_m2",
        "weights",
        "best.pt",
    )
    if not os.path.exists(model_path):
        pytest.skip("model weights not present")

    bad_file = tmp_path / "not_an_image.jpg"
    bad_file.write_text("this is not a real image")

    detector = PlasticDetector(model_path=model_path, device="cpu")

    with pytest.raises(ValueError):
        detector.process_image(str(bad_file), output_path=str(tmp_path / "out.jpg"))


def test_process_batch_empty_dir_returns_zero_summary(tmp_path: Path):
    from plastic_detector import PlasticDetector

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "ocean_waste_model_m2",
        "weights",
        "best.pt",
    )
    if not os.path.exists(model_path):
        pytest.skip("model weights not present")

    out_dir = tmp_path / "out"

    detector = PlasticDetector(model_path=model_path, device="cpu")
    result = detector.process_batch(str(tmp_path), str(out_dir))

    assert "summary" in result
    assert result["summary"]["total_images_processed"] == 0
    assert result["summary"]["detection_rate"] == "0%"
