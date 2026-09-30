"""SmartCity - AI Garbage Detection: one-command launcher.

    python run.py

Installs / upgrades any missing dependencies, checks that the detection
model really loads, starts the server and opens the site in your browser.

    python run.py --reset-admin    set the admin login back to admin / admin123
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import webbrowser
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "5000"))

# import name -> (pip requirement, minimum version tuple or None)
REQUIRED = {
    "flask": ("Flask>=3.0", None),
    "werkzeug": ("Werkzeug>=3.0", None),
    "cv2": ("opencv-python>=4.8", None),
    "numpy": ("numpy>=1.26", None),
    # weights were trained with 8.4.x -> older engines can fail to load them
    "ultralytics": ("ultralytics>=8.4.0", (8, 4)),
}
DIST_NAME = {"cv2": "opencv-python", "flask": "Flask", "werkzeug": "Werkzeug"}


def _version(mod: str):
    for name in (DIST_NAME.get(mod, mod), "opencv-python-headless", "opencv-contrib-python"):
        try:
            parts = metadata.version(name).split(".")
            return tuple(int("".join(ch for ch in p if ch.isdigit()) or 0) for p in parts[:3])
        except metadata.PackageNotFoundError:
            if mod != "cv2":
                break
    return None


def ensure_dependencies() -> None:
    need = []
    for mod, (req, minimum) in REQUIRED.items():
        if importlib.util.find_spec(mod) is None:
            need.append(req)
        elif minimum and (_version(mod) or (0,)) < minimum:
            need.append(req)
    if not need:
        return
    print("Installing / updating packages (first run only, may take a few minutes):")
    print("   " + ", ".join(need))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *need])
    except subprocess.CalledProcessError:
        sys.exit("\nCould not install packages automatically. Check your internet "
                 "connection, then run:  python run.py  again.")


def check_model(app) -> None:
    """Load the model once now so problems show up here, not on first upload."""
    print("Checking detection model ...", end=" ", flush=True)
    try:
        used = app.extensions["detector"].self_check()
        print(f"OK ({Path(used).name})")
    except Exception as exc:  # noqa: BLE001
        cause = exc.__cause__ or exc
        print("FAILED")
        print(f"   Reason: {cause!r}")
        print("   The site will still start, but detection will show an error until this is fixed.")


def reset_admin(app) -> None:
    """Emergency fix for "wrong credentials": make sure admin / admin123 works."""
    from werkzeug.security import generate_password_hash

    from app.database import db

    database = app.config["DATABASE"]
    username = os.environ.get("SMARTCITY_ADMIN_USER", "admin").strip() or "admin"
    password = os.environ.get("SMARTCITY_ADMIN_PASSWORD") or "admin123"
    password_hash = generate_password_hash(password)
    existing = db.find_user_by_login(database, username)
    if existing is not None and existing["role"] != "admin":
        sys.exit(f"'{username}' is not an admin account. Set SMARTCITY_ADMIN_USER to another name.")
    if existing is None:
        db.create_user(
            database, username=username, full_name="City Administrator",
            email=None, role="admin", password_hash=password_hash,
        )
    else:
        db.update_user_password(database, existing["id"], password_hash)
        db.set_admin_active(database, existing["id"])
    print("\nAdmin login has been reset.")
    print(f"   Username : {username}")
    print(f"   Password : {password}")
    print("   Open http://127.0.0.1:5000/admin/login  (run: python run.py)\n")


def main() -> None:
    if sys.version_info < (3, 10):
        sys.exit("SmartCity needs Python 3.10 or newer.")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, str(ROOT))
    if "--reset-admin" not in sys.argv:
        ensure_dependencies()

    from app.app import app  # creates / upgrades the database on import

    if "--reset-admin" in sys.argv:
        reset_admin(app)
        return

    check_model(app)
    for note in app.extensions.get("seed_notes", []):
        print("First run:", note)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"\nSmartCity is running at {url}   (press Ctrl+C to stop)\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
