import os
import sys


# Ensure the Smart Marine project root and src/ are importable in tests.
# This must happen at import time so `import api_server` works even if
# a test imports modules before pytest hooks execute.
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure():
    # Path configuration is handled at import time above.
    return
