"""One-off: download the pi0.5 base checkpoint via openpi's own fsspec-based
downloader (no gsutil/gcloud needed). Run inside .venv_pi05."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "openpi" / "src"))

from openpi.shared import download

path = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print("DOWNLOADED_TO:", path)
