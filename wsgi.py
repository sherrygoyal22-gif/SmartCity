"""Production entry point:  gunicorn wsgi:app"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Hosted containers only get a slice of a big machine's CPU, but PyTorch / OpenCV start one
# thread per *machine* core. Dozens of threads fighting over a small CPU quota is a classic
# cause of very slow detection, so keep the thread count small (override with these variables).
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ.get("SMARTCITY_TORCH_THREADS", "2"))

from app.app import app  # noqa: E402  (creates / upgrades the database on import)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
