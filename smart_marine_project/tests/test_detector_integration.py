import os
import time
import numpy as np
import pytest


MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "ocean_waste_model_m2",
    "weights",
    "best.pt",
)


def _integration_enabled() -> bool:
    return os.environ.get("SMART_MARINE_RUN_INTEGRATION", "0") == "1"


@pytest.mark.skipif(not _integration_enabled(), reason="set SMART_MARINE_RUN_INTEGRATION=1 to enable")
def test_smoke_inference_on_blank_frame():
    if not os.path.exists(MODEL_PATH_DEFAULT):
        pytest.skip("model weights not present")

    from plastic_detector import PlasticDetector

    detector = PlasticDetector(model_path=MODEL_PATH_DEFAULT, device="cpu", conf_threshold=0.3)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start = time.time()
    _, detection_info = detector.detect_objects(frame)
    elapsed = time.time() - start

    assert isinstance(detection_info, list)
    assert elapsed >= 0
