from __future__ import annotations

import logging
import time
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from app import auth
from app.auth import admin_required, login_required, staff_required, user_required

from app.database import db
from app.services.detector import DetectionError, GarbageDetector
from app.services import camera_gate
from app.services import geocode
from app.services.location import hotspot_clusters, optional_coordinates
from app.services.reports import detections_csv
from app.services.risk import assess_risk, validate_detection


BASE_DIR = Path(__file__).resolve().parent



# ------------------------------------------------------------------
# Where your data lives
# ------------------------------------------------------------------
# By default the database and photos sit inside this project folder. Set
# SMARTCITY_DATA_DIR to keep them somewhere else (a persistent disk when
# hosted, or a folder outside the project on your PC) so that replacing or
# redeploying the code can never overwrite your reports and cases.
_data_dir_env = os.environ.get("SMARTCITY_DATA_DIR", "").strip()
DATA_DIR = Path(_data_dir_env).expanduser().resolve() if _data_dir_env else None


def _prepare_data_dir() -> None:
    """Point static/uploads and static/results at DATA_DIR and adopt any existing data once."""
    if DATA_DIR is None:
        return
    import shutil

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_db = BASE_DIR / "database" / "smartcity.db"
    new_db = DATA_DIR / "smartcity.db"
    if old_db.exists() and not new_db.exists():
        shutil.copy2(old_db, new_db)  # first run with a data dir: keep what is already there
    for name in ("uploads", "results"):
        target = DATA_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        link = BASE_DIR / "static" / name
        if link.is_symlink():
            continue
        if link.is_dir():
            for item in link.iterdir():
                if not (target / item.name).exists():
                    shutil.move(str(item), str(target / item.name))
            shutil.rmtree(link, ignore_errors=True)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            logging.getLogger(__name__).warning("Could not link %s to %s; photos stay in the project folder.", link, target)


def _backup_database(database_path, keep: int = 10) -> None:
    """Copy the database to database/backups/ at start-up (newest 10 kept)."""
    import sqlite3
    import time

    database_path = Path(database_path)
    try:
        if not database_path.exists() or database_path.stat().st_size == 0:
            return
        folder = database_path.parent / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(folder.glob("smartcity-*.db"))
        if existing and time.time() - existing[-1].stat().st_mtime < 600:
            return  # restarted a moment ago: one backup is enough
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        source = sqlite3.connect(database_path)
        target = sqlite3.connect(folder / f"smartcity-{stamp}.db")
        with target:
            source.backup(target)
        source.close()
        target.close()
        for old in sorted(folder.glob("smartcity-*.db"))[:-keep]:
            old.unlink(missing_ok=True)
    except Exception as exc:  # a failed backup must never stop the app
        logging.getLogger(__name__).warning("Database backup skipped: %r", exc)


def _load_secret_key() -> str:
    """Signing key for login sessions.

    Uses SMARTCITY_SECRET_KEY when set; otherwise a random key is generated
    once and kept next to the database so sessions survive restarts and the
    key is never a value that is public in the source code.
    """
    configured = os.environ.get("SMARTCITY_SECRET_KEY")
    if configured:
        return configured
    key_file = (DATA_DIR or (BASE_DIR / "database")) / ".secret_key"
    try:
        return key_file.read_text().strip()
    except OSError:
        key = secrets.token_hex(32)
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(key)
        except OSError:
            pass
        return key

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
}


class DuplicateImageError(ValueError):
    def __init__(self, detection_id):
        self.detection_id = detection_id
        super().__init__(f"This exact image has already been processed as report #{detection_id}.")


def create_app(test_config=None):

    _prepare_data_dir()
    app = Flask(__name__)
    # Gunicorn/Render hide INFO logs by default; show the camera diagnostics.
    app.logger.setLevel(logging.INFO)

    app.config.from_mapping(
        SECRET_KEY=_load_secret_key(),

        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),

        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        SEND_FILE_MAX_AGE_DEFAULT=86400,

        # V4 is the current trained model. Set SMARTCITY_MODEL_PATH to use a
        # different compatible local model without editing application code.
        MODEL_PATH=Path(
            os.environ.get(
                "SMARTCITY_MODEL_PATH",
                BASE_DIR / "model" / "V4_best.pt",
            )
        ),

        # Keep the original Version 2 weights intact and use them only when
        # the configured primary model cannot be loaded.
        FALLBACK_MODEL_PATH=(
            BASE_DIR
            / "model"
            / "best_garbage.pt"
        ),

        UPLOAD_DIR=(
            BASE_DIR
            / "static"
            / "uploads"
        ),

        AFTER_UPLOAD_DIR=(
            BASE_DIR
            / "static"
            / "uploads"
        ),

        RESULT_DIR=(
            BASE_DIR
            / "static"
            / "results"
        ),

        DATABASE=(
            (DATA_DIR / "smartcity.db")
            if DATA_DIR is not None
            else (BASE_DIR / "database" / "smartcity.db")
        ),

        DETECTION_CONFIDENCE=0.30,

        # Lowest confidence an uploaded image is ever analysed at, whatever the
        # slider says. Tunable on Render with SMARTCITY_UPLOAD_MIN_CONF.
        UPLOAD_MIN_CONFIDENCE=float(os.environ.get("SMARTCITY_UPLOAD_MIN_CONF", "0.30")),

        # ---- Live camera confirmation (see app/services/camera_gate.py) ----
        # The old rule needed EVERY frame >= 55-60% confidence, which real
        # camera frames (blur, exposure, video compression) almost never reach,
        # so live detection always answered "clean". Frames are now judged
        # together. All values can be tuned on Render without a code change.
        #
        # Lowest confidence the model is asked to report for camera frames.
        CAMERA_DETECTION_FLOOR=float(os.environ.get("SMARTCITY_CAMERA_FLOOR", "0.35")),

        # A single frame counts as a "hit" at or above this confidence (%).
        # Raising the on-page confidence slider makes this stricter.
        CAMERA_HIT_CONFIDENCE=float(os.environ.get("SMARTCITY_CAMERA_HIT_CONF", "45")),

        # Frames captured per camera press, and how many must be hits.
        # Frames used per camera press: the browser sends 3, up to 4 are used.
        CAMERA_MIN_FRAMES=3,
        CAMERA_CONFIRMATION_FRAMES=4,
        CAMERA_MIN_HITS=int(os.environ.get("SMARTCITY_CAMERA_MIN_HITS", "3")),

        # The hit frames must also average at least this confidence (%).
        # This is the main guard that stops clean scenes being reported.
        CAMERA_MIN_MEAN_CONFIDENCE=float(os.environ.get("SMARTCITY_CAMERA_MEAN_CONF", "50")),

        # A box must cover at least 2% of the frame; stray blobs and slivers
        # of shadow are ignored.
        CAMERA_MIN_BOX_AREA_RATIO=float(os.environ.get("SMARTCITY_CAMERA_MIN_AREA", "0.02")),

        # Hit frames must show the garbage in the same part of the view
        # (overlap of their enclosing boxes, 0-1).
        CAMERA_MIN_FRAME_OVERLAP=float(os.environ.get("SMARTCITY_CAMERA_OVERLAP", "0.25")),

        # Only YOLO boxes whose class name matches one of these (case
        # insensitive) are accepted as garbage. This is a second, independent
        # guard on top of the confidence threshold: even a confident box for
        # an unrelated/background class is discarded. Both bundled models
        # (best_garbage.pt / V4_best.pt) were trained with a single
        # "garbage" class, so that is the default allow-list; add more
        # names here if a future model introduces additional litter classes.
        ALLOWED_GARBAGE_CLASSES=("garbage",),

        # Fixed admin/municipal-office coordinates. Every marker's popup on
        # the Hotspot Map shows its distance and compass direction from this
        # single point, so the map never depends on the visitor's own GPS.
        # Defaults to the Ludhiana Municipal Corporation office; override
        # via the SMARTCITY_ADMIN_LAT / SMARTCITY_ADMIN_LNG environment
        # variables if the admin office is in a different city.
        ADMIN_LATITUDE=(
            float(os.environ["SMARTCITY_ADMIN_LAT"])
            if os.environ.get("SMARTCITY_ADMIN_LAT")
            else 30.9010
        ),
        ADMIN_LONGITUDE=(
            float(os.environ["SMARTCITY_ADMIN_LNG"])
            if os.environ.get("SMARTCITY_ADMIN_LNG")
            else 75.8573
        ),

        REMINDER_AFTER_HOURS=24,

        ESCALATION_AFTER_HOURS=48,

        VALIDATION_MIN_CONFIDENCE=35.0,

        LOCATION_DUPLICATE_WINDOW_MINUTES=60,

        LOCATION_DUPLICATE_TOLERANCE_DEGREES=0.0005,

        HOTSPOT_TOLERANCE_DEGREES=0.0005,

        HOTSPOT_MIN_DETECTIONS=2,

        DISPLAY_TIMEZONE="Asia/Kolkata",

        DETECTION_IMAGE_SIZE=640,
    )

    if test_config:
        app.config.update(test_config)


    # ============================================
    # CREATE REQUIRED DIRECTORIES
    # ============================================

    for key in (
        "UPLOAD_DIR",
        "AFTER_UPLOAD_DIR",
        "RESULT_DIR",
    ):
        Path(
            app.config[key]
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


    # ============================================
    # INITIALIZE DATABASE
    # ============================================

    _backup_database(app.config["DATABASE"])
    db.init_db(
        app.config["DATABASE"]
    )
    db.backfill_missing_cases(app.config["DATABASE"])
    db.recompute_risk(app.config["DATABASE"], assess_risk)


    # ============================================
    # DEFAULT ACCOUNTS (first run only)
    # ============================================

    def ensure_default_accounts():
        """Create the first admin (and one demo citizen) when none exist yet."""
        database = app.config["DATABASE"]
        notes = []
        if db.count_users(database, "admin") == 0:
            username = os.environ.get("SMARTCITY_ADMIN_USER", "admin").strip() or "admin"
            password = os.environ.get("SMARTCITY_ADMIN_PASSWORD") or "admin123"
            db.create_user(
                database, username=username, full_name="City Administrator",
                email="admin@smartcity.local", role="admin",
                password_hash=generate_password_hash(password),
            )
            if os.environ.get("SMARTCITY_ADMIN_PASSWORD"):
                notes.append(f"Admin account created: {username} (password from SMARTCITY_ADMIN_PASSWORD)")
            else:
                notes.append(f"Admin login  ->  username: {username}   password: {password}   (change it in Profile)")
        if db.get_setting(database, "demo_user_seeded") is None:
            if db.count_users(database, "user") == 0:
                demo_id = db.create_user(
                    database, username="user", full_name="Demo Citizen",
                    email="user@smartcity.local", role="user",
                    password_hash=generate_password_hash("user123"),
                )
                # Reports saved before accounts existed belong to the demo citizen.
                db.assign_unowned_detections(database, demo_id)
                notes.append("User login   ->  username: user    password: user123   (demo citizen)")
            db.set_settings(database, {"demo_user_seeded": "1"})
        if db.get_setting(database, "demo_staff_seeded") is None:
            if db.count_users(database, "staff") == 0:
                db.create_staff(
                    database, username="staff", full_name="Demo Staff Worker",
                    email="staff@smartcity.local",
                    password_hash=generate_password_hash("staff123"),
                )
                notes.append("Staff login  ->  username: staff   password: staff123   (demo municipal worker)")
            db.set_settings(database, {"demo_staff_seeded": "1"})
        return notes

    app.extensions["seed_notes"] = ensure_default_accounts()


    # ============================================
       # LOAD YOLO MODEL
    # ============================================

    app.extensions["detector"] = GarbageDetector(
        app.config["MODEL_PATH"],
        app.config["FALLBACK_MODEL_PATH"],
        app.config["ALLOWED_GARBAGE_CLASSES"],
    )

    # Load the model in the background right after start-up so the first
    # camera detection does not also pay for loading it (slow on a hosted CPU).
    if os.environ.get(
        "SMARTCITY_WARMUP",
        "0" if os.environ.get("RENDER") else "1"
    ) != "0" and not test_config:
        import threading

        def _warm_up():
            try:
                app.extensions["detector"].self_check()
                app.logger.info("Detection model warmed up.")
            except Exception as exc:
                app.logger.warning("Model warm-up skipped: %r", exc)

        threading.Thread(
            target=_warm_up,
            daemon=True,
            name="model-warmup"
        ).start()

    # ============================================
    # SPEED: versioned static URLs, long caching, gzip
    # ============================================

    @app.url_defaults
    def add_static_version(endpoint, values):
        """/static/x.css?v=<file time>: changes only when the file changes, so it can be cached for a year."""
        if endpoint == "static" and "v" not in values:
            filename = values.get("filename")
            if filename:
                try:
                    values["v"] = int((Path(app.static_folder) / filename).stat().st_mtime)
                except OSError:
                    pass

    _COMPRESSIBLE = {
        "text/html", "text/css", "text/javascript", "application/javascript",
        "application/json", "application/manifest+json", "image/svg+xml",
    }

    @app.after_request
    def speed_headers(response):
        if request.endpoint == "static" and response.status_code in (200, 304):
            if request.args.get("v"):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "public, max-age=86400"
        wants_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        if (
            wants_gzip
            and response.status_code == 200
            and "Content-Encoding" not in response.headers
            and response.mimetype in _COMPRESSIBLE
            and (response.direct_passthrough or not response.is_streamed)
        ):
            import gzip

            response.direct_passthrough = False
            data = response.get_data()
            if len(data) > 600:
                response.set_data(gzip.compress(data, compresslevel=6))
                response.headers["Content-Encoding"] = "gzip"
            response.headers.add("Vary", "Accept-Encoding")
        return response

    # ============================================
    # TEMPLATE FILTER
    # ============================================

    @app.template_filter("percent")
    def percent(value):

        return (
            f"{value:.1f}%"
            if value is not None
            else "—"
        )


    @app.template_filter("india_datetime")
    def india_datetime(value):
        if not value:
            return "—"
        try:
            timestamp = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            try:
                display_timezone = ZoneInfo(app.config["DISPLAY_TIMEZONE"])
            except ZoneInfoNotFoundError:
                # Windows Python may not include the IANA timezone database.
                display_timezone = timezone(timedelta(hours=5, minutes=30), "IST")
            return timestamp.replace(tzinfo=timezone.utc).astimezone(
                display_timezone
            ).strftime("%d %b %Y, %I:%M %p IST")
        except (TypeError, ValueError):
            return value


    @app.template_filter("duration")
    def duration(minutes):
        if minutes is None:
            return "—"
        hours, remaining_minutes = divmod(max(int(minutes), 0), 60)
        days, remaining_hours = divmod(hours, 24)
        parts = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if remaining_hours:
            parts.append(f"{remaining_hours} hour{'s' if remaining_hours != 1 else ''}")
        if remaining_minutes or not parts:
            parts.append(f"{remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}")
        return " ".join(parts)


    # ============================================
    # GLOBAL TEMPLATE VARIABLES
    # ============================================

    @app.template_filter("status_label")
    def status_label(value):
        return {"PENDING": "Pending", "ASSIGNED": "Assigned", "CLEANING": "Cleaning",
                "COMPLETED": "Resolved"}.get(value, value or "No case")


    @app.template_filter("status_class")
    def status_class(value):
        return {"PENDING": "pending", "ASSIGNED": "progress", "CLEANING": "progress",
                "COMPLETED": "resolved"}.get(value, "none")


    @app.template_filter("status_group")
    def status_group(value):
        """Citizen-facing wording: three simple states instead of four."""
        return {"PENDING": "Pending", "ASSIGNED": "In Progress", "CLEANING": "In Progress",
                "COMPLETED": "Resolved"}.get(value, "Not a case")


    # ============================================
    # SESSION / CURRENT USER
    # ============================================

    STAFF_ENDPOINTS = {
        "static", "about", "profile", "change_password", "logout",
        "login", "admin_login", "staff_login", "register",
        "notifications", "notifications_read_all", "notification_open",
        "staff_dashboard", "staff_task", "staff_start_task", "staff_before_image",
        "staff_after_image", "staff_complete_task", "staff_report_issue",
        "pwa_manifest", "pwa_service_worker", "pwa_offline", "healthz",
    }


    @app.before_request
    def load_current_user():
        g.user = None
        user_id = session.get("user_id")
        if user_id is None or request.endpoint == "static":
            return
        user = db.get_user(app.config["DATABASE"], user_id)
        if user is None or not user["is_active"]:
            session.clear()
            return
        g.user = user
        if user["role"] == auth.ROLE_STAFF and request.endpoint not in STAFF_ENDPOINTS:
            # Staff only ever see their own dashboard/tasks, notifications and profile.
            if request.path.startswith("/api/"):
                return {"ok": False, "error": "Staff accounts cannot use this feature."}, 403
            if request.method == "GET" and request.endpoint is not None and not request.path.startswith("/admin"):
                return redirect(url_for("staff_dashboard"))
            abort(403)


    def is_admin():
        return g.user is not None and g.user["role"] == auth.ROLE_ADMIN


    def can_access_detection(record):
        """Admins see every report; citizens only the ones they submitted."""
        return record is not None and g.user is not None and (
            is_admin() or record["user_id"] == g.user["id"]
        )


    def admin_location():
        """Admin office coordinates: saved by the admin, else the configured default."""
        database = app.config["DATABASE"]
        try:
            latitude = float(db.get_setting(database, "admin_lat"))
            longitude = float(db.get_setting(database, "admin_lng"))
        except (TypeError, ValueError):
            latitude, longitude = app.config["ADMIN_LATITUDE"], app.config["ADMIN_LONGITUDE"]
        return {
            "lat": latitude,
            "lng": longitude,
            "label": db.get_setting(database, "admin_label", "Admin Office"),
            "state": db.get_setting(database, "admin_state", "") or "",
            "city": db.get_setting(database, "admin_city", "") or "",
        }


    # ============================================
    # GLOBAL TEMPLATE VARIABLES
    # ============================================

    @app.context_processor
    def globals_for_templates():
        user = g.get("user")
        admin = user is not None and user["role"] == auth.ROLE_ADMIN
        staff = user is not None and user["role"] == auth.ROLE_STAFF
        unread = 0
        if user is not None:
            unread = db.unread_notification_count(
                app.config["DATABASE"],
                "admin" if admin else ("staff" if staff else "user"), user["id"],
            )
        return {
            "active_path": request.path,
            "unread_notifications": unread,
            "current_user": user,
            "is_admin": admin,
            "is_staff": staff,
            "role_label": auth.ROLE_LABELS.get(user["role"], "") if user else "",
            "layout": "admin_base.html" if admin else ("staff_base.html" if staff else "base.html"),
            "user_initials": auth.initials(user["full_name"]) if user else "",
        }


    def refresh_notifications():
        db.refresh_case_notifications(
            app.config["DATABASE"],
            app.config["REMINDER_AFTER_HOURS"],
            app.config["ESCALATION_AFTER_HOURS"],
        )


    def serialize_map_records(records):
        """Turn real detection rows into the JSON payload the Leaflet map expects.

        Shared by the dashboards and the full map page so every surface stays
        backed by exactly the same real GPS data. Admins get links into the
        case system; citizens only get a link for reports they submitted.
        """
        map_data = []
        for row in records:
            if row["latitude"] is None or row["longitude"] is None:
                continue
            own_report = g.user is not None and row["user_id"] == g.user["id"]
            map_data.append({
                "detection_id": row["detection_id"],
                "lat": row["latitude"], "lng": row["longitude"],
                "created_at": row["created_at"], "count": row["detection_count"],
                "confidence": row["confidence"], "result_path": row["result_path"], "address": row["address"],
                "source": row["source"],
                "risk_level": row["risk_level"], "case_id": row["case_id"],
                "case_status": row["case_status"], "assignee_name": row["assignee_name"],
                "case_url": url_for("case_detail", case_id=row["case_id"])
                if is_admin() and row["case_id"] is not None else None,
                "report_url": url_for("report", detection_id=row["detection_id"])
                if is_admin() or own_report else None,
            })
        return map_data


    def build_hotspots(records):
        """Cluster real records into hotspots and attach real navigation URLs."""
        hotspots = hotspot_clusters(
            records,
            app.config["HOTSPOT_TOLERANCE_DEGREES"],
            app.config["HOTSPOT_MIN_DETECTIONS"],
        )
        owners = {row["detection_id"]: row["user_id"] for row in records}
        for hotspot in hotspots:
            latest = hotspot["latest_detection_id"]
            viewable = is_admin() or (g.user is not None and owners.get(latest) == g.user["id"])
            hotspot["report_url"] = url_for("report", detection_id=latest) if viewable else None
            hotspot["history_url"] = (
                url_for("admin_reports", q=hotspot["area_label"])
                if is_admin() and hotspot.get("area_label") else None
            )
        return hotspots


    def attach_hotspot_membership(map_data, hotspots):
        """Let an individual marker's popup say "N reports in this hotspot"."""
        membership = {}
        for hotspot in hotspots:
            for report_id in hotspot["report_ids"]:
                membership[report_id] = hotspot["detection_count"]
        for record in map_data:
            record["hotspot_count"] = membership.get(record["detection_id"])
        return map_data



    # ============================================
    # FILE VALIDATION
    # ============================================

    def allowed_file(filename):

        return (
            "."
            in filename
            and filename.rsplit(
                ".",
                1
            )[1].lower()
            in ALLOWED_EXTENSIONS
        )


    # ============================================
    # SAFE STATIC PATH
    # ============================================

    def safe_static_path(relative_path):

        target = (
            BASE_DIR
            / "static"
            / relative_path
        ).resolve()

        static_root = (
            BASE_DIR
            / "static"
        ).resolve()

        allowed = [static_root] + ([DATA_DIR] if DATA_DIR is not None else [])
        return (
            target
            if any(target.is_relative_to(root) for root in allowed)
            else None
        )


    def safe_after_evidence_path(relative_path):
        """Resolve only after-cleaning files stored flat in the uploads area."""
        normalized = (relative_path or "").replace("\\", "/")
        if not normalized.startswith("uploads/after-"):
            return None
        root = Path(app.config["AFTER_UPLOAD_DIR"]).resolve()
        target = (root / Path(relative_path).name).resolve()
        return target if target.is_relative_to(root) else None


    # ============================================
    # GET AUTOMATIC GPS
    #
    # Latitude and longitude are sent by
    # the browser automatically.
    #
    # The user does NOT type them.
    # ============================================

    def get_request_coordinates():

        latitude = request.form.get(
            "latitude",
            "",
        ).strip()

        longitude = request.form.get(
            "longitude",
            "",
        ).strip()

        latitude, longitude = optional_coordinates(
            latitude,
            longitude,
        )

        return latitude, longitude


    # ============================================
    # MAIN DETECTION FUNCTION
    # ============================================

    def detect_upload(
        upload,
        source,
    ):

        # ----------------------------------------
        # Validate image
        # ----------------------------------------

        if (
            upload is None
            or not upload.filename
        ):
            raise ValueError(
                "Choose an image before starting detection."
            )


        if not allowed_file(
            upload.filename
        ):
            raise ValueError(
                "Only JPG, JPEG, PNG, and WEBP "
                "image files are supported."
            )

        # Hash before saving so an exact repeat can reuse its original report.
        upload_bytes = upload.stream.read()
        upload.stream.seek(0)
        image_hash = sha256(upload_bytes).hexdigest()
        existing = db.find_detection_by_hash(
            app.config["DATABASE"],
            image_hash,
        )
        if existing is not None:
            raise DuplicateImageError(existing["id"])

        latitude, longitude = get_request_coordinates()
        address = (request.form.get("address") or "").strip()[:255] or None


        # ----------------------------------------
        # Generate unique filenames
        # ----------------------------------------

        extension = Path(
            secure_filename(
                upload.filename
            )
        ).suffix.lower()

        identifier = uuid.uuid4().hex


        upload_path = (
            Path(
                app.config["UPLOAD_DIR"]
            )
            / f"{identifier}{extension}"
        )


        result_path = (
            Path(
                app.config["RESULT_DIR"]
            )
            / f"{identifier}_detected.jpg"
        )


        try:

            # ------------------------------------
            # Save uploaded image
            # ------------------------------------

            upload.save(
                upload_path
            )


            # ------------------------------------
            # Get confidence
            # ------------------------------------

            try:

                confidence = float(
                    request.form.get(
                        "confidence",
                        app.config[
                            "DETECTION_CONFIDENCE"
                        ],
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = app.config[
                    "DETECTION_CONFIDENCE"
                ]


            # Never run below the upload floor: boxes under ~30% are mostly
            # textures (walls, grass, shadows), not garbage. A very low slider
            # value used to make clean photos look like garbage.
            confidence = min(
                max(
                    confidence,
                    app.config["UPLOAD_MIN_CONFIDENCE"],
                ),
                0.99,
            )


            # ------------------------------------
            # RUN YOLO
            # ------------------------------------

            detection = app.extensions["detector"].detect(
                upload_path,
                result_path,
                confidence,
                app.config["DETECTION_IMAGE_SIZE"],
            )

            is_valid, validation_reason = validate_detection(
                detection.detection_count,
                detection.average_confidence,
                app.config["VALIDATION_MIN_CONFIDENCE"],
            )

            location_duplicate = None
            if latitude is not None and longitude is not None and detection.detection_count:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    minutes=app.config["LOCATION_DUPLICATE_WINDOW_MINUTES"]
                )
                location_duplicate = db.find_recent_location_duplicate(
                    app.config["DATABASE"],
                    latitude=latitude,
                    longitude=longitude,
                    after_timestamp=cutoff.strftime("%Y-%m-%d %H:%M:%S"),
                    tolerance_degrees=app.config["LOCATION_DUPLICATE_TOLERANCE_DEGREES"],
                )

            is_duplicate = location_duplicate is not None
            duplicate_reason = (
                "A garbage detection was already recorded at this nearby location "
                f"within the last {app.config['LOCATION_DUPLICATE_WINDOW_MINUTES']} minutes "
                f"(report #{location_duplicate['id']})."
                if location_duplicate else None
            )
            risk_level, risk_score, risk_reason = assess_risk(
                detection.detection_count,
                detection.average_confidence,
                detection.box_area_ratio,
                is_duplicate,
            )


            # ------------------------------------
            # SAVE DETECTION
            # ------------------------------------

            record_id = db.create_detection(

                app.config[
                    "DATABASE"
                ],

                user_id=g.user["id"] if g.user is not None else None,

                detection_count=detection.detection_count,

                confidence=detection.average_confidence,

                image_path=(
                    f"uploads/"
                    f"{upload_path.name}"
                ),

                result_path=(
                    f"results/"
                    f"{result_path.name}"
                ),

                latitude=latitude,

                longitude=longitude,

                address=address,

                source=source,

                image_hash=image_hash,

                box_area_ratio=detection.box_area_ratio,

                is_valid=is_valid,

                validation_reason=validation_reason,

                is_duplicate=is_duplicate,

                duplicate_reason=duplicate_reason,

                risk_level=risk_level,

                risk_score=risk_score,

                risk_reason=risk_reason,
            )

            # Every valid garbage detection (upload or live camera) becomes
            # its own pending municipal case.
            if is_valid and detection.detection_count:
                db.create_alert_and_case(app.config["DATABASE"], record_id)


            return record_id


        except Exception:

            # ------------------------------------
            # Remove files if detection fails
            # ------------------------------------

            upload_path.unlink(
                missing_ok=True
            )

            result_path.unlink(
                missing_ok=True
            )

            raise


    # ============================================
    # SIGN IN / SIGN UP / SIGN OUT
    # ============================================

    def validate_account_fields(form, *, require_password=True, exclude_id=None):
        """Shared checks for registration and admin-created accounts."""
        full_name = (form.get("full_name") or "").strip()
        username = (form.get("username") or "").strip()
        email = (form.get("email") or "").strip()
        phone = (form.get("phone") or "").strip()
        password = form.get("password") or ""
        errors = []
        if len(full_name) < 2 or len(full_name) > 80:
            errors.append("Enter your full name (2-80 characters).")
        if not auth.USERNAME_RE.match(username):
            errors.append("Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
        if email and not auth.EMAIL_RE.match(email):
            errors.append("Enter a valid email address.")
        if phone and (len(phone) > 20 or not all(ch.isdigit() or ch in "+- ()" for ch in phone)):
            errors.append("Enter a valid phone number.")
        if require_password:
            if len(password) < auth.MIN_PASSWORD_LENGTH:
                errors.append(f"Password must be at least {auth.MIN_PASSWORD_LENGTH} characters.")
            if password != (form.get("confirm_password") if "confirm_password" in form else password):
                errors.append("The two passwords do not match.")
        if not errors:
            taken = db.username_or_email_taken(app.config["DATABASE"], username, email, exclude_id)
            if taken:
                errors.append(f"That {taken} is already registered.")
        return {"full_name": full_name, "username": username, "email": email,
                "phone": phone, "password": password}, errors


    def attempt_login(portal):
        """Shared sign-in flow. ``portal`` is the role this login page serves."""
        identifier = (request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""
        generic = "Incorrect username/email or password."

        if auth.login_blocked(identifier):
            return None, "Too many failed attempts. Please wait a few minutes and try again."
        user = db.find_user_by_login(app.config["DATABASE"], identifier) if identifier else None
        if user is None or not check_password_hash(user["password_hash"], password):
            auth.record_failure(identifier)
            return None, generic
        # Correct password, but the wrong door - say so without exposing anything else.
        if user["role"] != portal:
            page = {auth.ROLE_ADMIN: "Admin Login", auth.ROLE_STAFF: "Staff Login", auth.ROLE_USER: "User Login"}[user["role"]]
            kind = {auth.ROLE_ADMIN: "an administrator", auth.ROLE_STAFF: "a staff", auth.ROLE_USER: "a citizen"}[user["role"]]
            return None, f"This is {kind} account. Please use the {page} page."
        if not user["is_active"]:
            return None, "This account has been deactivated. Please contact the administrator."
        auth.clear_failures(identifier)
        auth.login_session(user, request.form.get("remember") == "on")
        db.touch_login(app.config["DATABASE"], user["id"])
        return user, None


    def post_login_redirect(user):
        target = request.args.get("next") or request.form.get("next")
        if auth.is_safe_next(target):
            # Never bounce someone into another role's area.
            if auth.area_for_path(target) == user["role"]:
                return redirect(target)
        return redirect(auth.home_for(user))


    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user is not None:
            return redirect(auth.home_for(g.user))
        error, identifier = None, ""
        if request.method == "POST":
            identifier = (request.form.get("identifier") or "").strip()
            user, error = attempt_login(auth.ROLE_USER)
            if user:
                flash(f"Welcome back, {user['full_name'].split()[0]}!", "success")
                return post_login_redirect(user)
        return render_template("login_user.html", error=error, identifier=identifier,
                               next_url=request.args.get("next", ""))


    @app.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if g.user is not None:
            return redirect(auth.home_for(g.user))
        error, identifier = None, ""
        if request.method == "POST":
            identifier = (request.form.get("identifier") or "").strip()
            user, error = attempt_login(auth.ROLE_ADMIN)
            if user:
                flash(f"Signed in as {user['full_name']}.", "success")
                return post_login_redirect(user)
        return render_template("login_admin.html", error=error, identifier=identifier,
                               next_url=request.args.get("next", ""))


    @app.route("/staff/login", methods=["GET", "POST"])
    def staff_login():
        if g.user is not None:
            return redirect(auth.home_for(g.user))
        error, identifier = None, ""
        if request.method == "POST":
            identifier = (request.form.get("identifier") or "").strip()
            user, error = attempt_login(auth.ROLE_STAFF)
            if user:
                flash(f"Signed in as {user['full_name']}.", "success")
                return post_login_redirect(user)
        return render_template("login_staff.html", error=error, identifier=identifier,
                               next_url=request.args.get("next", ""))


    @app.route("/register", methods=["GET", "POST"])
    def register():
        if g.user is not None:
            return redirect(auth.home_for(g.user))
        errors, values = [], {}
        if request.method == "POST":
            values, errors = validate_account_fields(request.form)
            if not errors:
                user_id = db.create_user(
                    app.config["DATABASE"], username=values["username"], full_name=values["full_name"],
                    email=values["email"], phone=values["phone"], role=auth.ROLE_USER,
                    password_hash=generate_password_hash(values["password"]),
                )
                auth.login_session(db.get_user(app.config["DATABASE"], user_id), False)
                flash("Your account is ready. You can report garbage now.", "success")
                return redirect(url_for("dashboard"))
            values.pop("password", None)
        return render_template("register.html", errors=errors, values=values)


    @app.post("/logout")
    def logout():
        role = g.user["role"] if g.user is not None else auth.ROLE_USER
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for(auth.LOGIN_ENDPOINTS[role]))


    # ============================================
    # PROFILE
    # ============================================

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        errors = []
        if request.method == "POST":
            full_name = (request.form.get("full_name") or "").strip()
            email = (request.form.get("email") or "").strip()
            phone = (request.form.get("phone") or "").strip()
            if len(full_name) < 2 or len(full_name) > 80:
                errors.append("Enter your full name (2-80 characters).")
            if email and not auth.EMAIL_RE.match(email):
                errors.append("Enter a valid email address.")
            if phone and (len(phone) > 20 or not all(ch.isdigit() or ch in "+- ()" for ch in phone)):
                errors.append("Enter a valid phone number.")
            if not errors and email and db.username_or_email_taken(
                app.config["DATABASE"], "\0", email, exclude_id=g.user["id"]
            ):
                errors.append("That email is already registered.")
            if not errors:
                db.update_user_profile(app.config["DATABASE"], g.user["id"], full_name, email, phone)
                flash("Profile updated.", "success")
                return redirect(url_for("profile"))
        summary = None if (is_admin() or g.user["role"] == auth.ROLE_STAFF) else db.user_summary(app.config["DATABASE"], g.user["id"])
        return render_template("profile.html", errors=errors, summary=summary,
                               total_users=db.count_users(app.config["DATABASE"], "user") if is_admin() else None)


    @app.post("/profile/password")
    @login_required
    def change_password():
        current = request.form.get("current_password") or ""
        new = request.form.get("new_password") or ""
        confirm = request.form.get("confirm_password") or ""
        if not check_password_hash(g.user["password_hash"], current):
            flash("Your current password is incorrect.", "error")
        elif len(new) < auth.MIN_PASSWORD_LENGTH:
            flash(f"New password must be at least {auth.MIN_PASSWORD_LENGTH} characters.", "error")
        elif new != confirm:
            flash("The new passwords do not match.", "error")
        else:
            db.update_user_password(app.config["DATABASE"], g.user["id"], generate_password_hash(new))
            flash("Password changed.", "success")
        return redirect(url_for("profile") + "#password")


    # ============================================
    # CITIZEN DASHBOARD
    # ============================================

    @app.get("/")
    @user_required
    def dashboard():
        database = app.config["DATABASE"]
        map_records = db.map_records(database)
        map_data = serialize_map_records(map_records)
        hotspots = build_hotspots(map_records)
        attach_hotspot_membership(map_data, hotspots)
        location = admin_location()

        return render_template(
            "dashboard.html",
            summary=db.user_summary(database, g.user["id"]),
            recent=db.recent_detections(database, limit=3, user_id=g.user["id"]),
            map_data=map_data,
            hotspots=hotspots,
            coordinate_count=len(map_data),
            admin_latitude=location["lat"],
            admin_longitude=location["lng"],
        )


    # ============================================
    # ADMIN DASHBOARD
    # ============================================

    @app.get("/admin")
    @admin_required
    def admin_dashboard():
        refresh_notifications()
        database = app.config["DATABASE"]
        summary, _recent, _trend, _status, _risk, _activity = db.dashboard_data(database)
        map_records = db.map_records(database)
        map_data = serialize_map_records(map_records)
        hotspots = build_hotspots(map_records)
        attach_hotspot_membership(map_data, hotspots)
        location = admin_location()

        return render_template(
            "admin_dashboard.html",
            summary=summary,
            recent=db.recent_detections(database, limit=5),
            total_users=db.count_users(database, "user"),
            map_data=map_data,
            hotspots=hotspots,
            coordinate_count=len(map_data),
            admin_latitude=location["lat"],
            admin_longitude=location["lng"],
        )


    # ============================================
    # IMAGE UPLOAD DETECTION
    # ============================================

    @app.get("/report")
    @login_required
    def report_garbage():
        return redirect(url_for("detect"))

    @app.route(
        "/detect",
        methods=["GET", "POST"],
    )
    @login_required
    def detect():

        if request.method == "POST":

            try:

                record_id = detect_upload(
                    request.files.get(
                        "image"
                    ),
                    "upload",
                )


                return redirect(
                    url_for(
                        "report",
                        detection_id=record_id,
                    )
                )


            except DuplicateImageError as exc:

                existing = db.get_detection(app.config["DATABASE"], exc.detection_id)
                if can_access_detection(existing):
                    flash(str(exc), "error")
                    return redirect(url_for("report", detection_id=exc.detection_id))
                flash("This exact image has already been reported to SmartCity.", "error")

            except (
                ValueError,
                DetectionError,
            ) as exc:

                flash(
                    str(exc),
                    "error",
                )


            except Exception:

                app.logger.exception(
                    "Unexpected detection error"
                )

                flash(
                    "An unexpected error occurred "
                    "while processing the image.",
                    "error",
                )


        return render_template(
            "detect.html",
            default_confidence=
                app.config[
                    "DETECTION_CONFIDENCE"
                ],
        )


    # ============================================
    # LIVE CAMERA DETECTION API
    # ============================================

    @app.post(
        "/api/camera-detect"
    )
    @login_required
    def camera_detect():

        try:

            slider = float(request.form.get("confidence", app.config["DETECTION_CONFIDENCE"]))
            # The model is asked for everything down to the camera floor; the
            # confirmation rules below decide what is accepted. Raising the
            # slider makes the camera stricter, it can never make it blind.
            confidence = min(max(slider, app.config["CAMERA_DETECTION_FLOOR"]), 0.99)
            hit_confidence = max(app.config["CAMERA_HIT_CONFIDENCE"], slider * 100.0)
            min_area = app.config["CAMERA_MIN_BOX_AREA_RATIO"]
            min_hits = app.config["CAMERA_MIN_HITS"]

            frames = request.files.getlist("validation_frames")
            if len(frames) < app.config["CAMERA_MIN_FRAMES"]:
                raise ValueError("Capture enough camera frames to confirm a garbage detection.")

            inspections = []
            checked_frames = frames[:app.config["CAMERA_CONFIRMATION_FRAMES"]]
            # Never demand more hits than frames that were actually sent.
            min_hits = min(min_hits, len(checked_frames))
            for position, frame in enumerate(checked_frames):
                inspection = app.extensions["detector"].inspect_bytes(
                    frame.read(), confidence, app.config["DETECTION_IMAGE_SIZE"]
                )
                inspections.append(inspection)
                app.logger.info(
                    "camera frame %s: boxes=%s highest_conf=%s largest_box=%s",
                    position + 1,
                    inspection.detection_count,
                    inspection.highest_confidence,
                    inspection.largest_box_ratio,
                )
                # Stop early once confirmation is impossible (saves CPU on a
                # hosted server): an obviously empty scene needs one model run.
                hits_so_far = sum(
                    1 for item in inspections
                    if camera_gate.frame_is_hit(item, hit_confidence, min_area)
                )
                remaining = len(checked_frames) - position - 1
                if hits_so_far + remaining < min_hits:
                    break

            verdict = camera_gate.evaluate_camera_frames(
                inspections,
                hit_confidence=hit_confidence,
                min_hits=min_hits,
                min_mean_confidence=app.config["CAMERA_MIN_MEAN_CONFIDENCE"],
                min_box_ratio=min_area,
                min_overlap=app.config["CAMERA_MIN_FRAME_OVERLAP"],
            )
            app.logger.info(
                "camera verdict: confirmed=%s hits=%s/%s best=%.1f mean_hit=%s reason=%s",
                verdict.confirmed, verdict.hits, verdict.frames,
                verdict.best_confidence, verdict.mean_hit_confidence, verdict.reason,
            )
            if not verdict.confirmed:
                raise ValueError(verdict.reason)

            # ------------------------------------
            # Run detection
            # ------------------------------------

            # The browser sends only validation frames; save the one the model
            # was most confident about (an explicit "image" field still wins).
            best_index = max(
                range(len(inspections)),
                key=lambda i: inspections[i].highest_confidence or 0.0,
            )
            image_file = request.files.get("image") or checked_frames[best_index]
            image_file.stream.seek(0)

            record_id = detect_upload(
                image_file,
                "camera",
            )


            # ------------------------------------
            # Retrieve saved record
            # ------------------------------------

            record = db.get_detection(
                app.config[
                    "DATABASE"
                ],
                record_id,
            )


            if record is None:

                return {
                    "ok": False,
                    "error":
                        "Detection record was not found.",
                }, 500

            case = db.get_case_by_detection(app.config["DATABASE"], record_id)


            # ------------------------------------
            # Return complete detection data
            # ------------------------------------

            return {

                "ok": True,

                "id": record_id,

                "count":
                    record[
                        "detection_count"
                    ],

                "confidence":
                    record[
                        "confidence"
                    ],

                "latitude":
                    record[
                        "latitude"
                    ],

                "longitude":
                    record[
                        "longitude"
                    ],

                "address":
                    record[
                        "address"
                    ],

                "created_at":
                    record[
                        "created_at"
                    ],

                "source":
                    record[
                        "source"
                    ],

                "is_valid":
                    bool(record["is_valid"]),

                "validation_reason":
                    record["validation_reason"],

                "is_duplicate":
                    bool(record["is_duplicate"]),

                "duplicate_reason":
                    record["duplicate_reason"],

                "risk_level":
                    record["risk_level"],

                "risk_score":
                    record["risk_score"],

                "case_id": case["id"] if case else None,

                "case_url":
                    url_for("case_detail", case_id=case["id"]) if case and is_admin() else None,

                "result_url":
                    url_for(
                        "static",
                        filename=
                            record[
                                "result_path"
                            ],
                    ),

                "report_url":
                    url_for(
                        "report",
                        detection_id=
                            record_id,
                    ),
            }


        except DuplicateImageError as exc:

            existing = db.get_detection(app.config["DATABASE"], exc.detection_id)
            if can_access_detection(existing):
                return {
                    "ok": False,
                    "error": str(exc),
                    "report_url": url_for("report", detection_id=exc.detection_id),
                }, 409
            return {
                "ok": False,
                "error": "This exact image has already been reported to SmartCity.",
            }, 409

        except (
            ValueError,
            DetectionError,
        ) as exc:

            return {

                "ok": False,

                "error":
                    str(exc),

            }, 400


        except HTTPException:
            raise  # e.g. 413 too large: let the error handler answer with the right status

        except Exception:

            app.logger.exception(
                "Camera detection error"
            )

            return {

                "ok": False,

                "error":
                    "Detection could not be completed.",

            }, 500


    # ============================================
    # REPORT LISTS  (citizen: "My Reports", admin: "All Garbage Reports")
    # ============================================

    def read_report_filters():
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        return {
            "search": request.args.get("q", "").strip(),
            "source": request.args.get("source", ""),
            "status": request.args.get("status", ""),
            "risk": request.args.get("risk", ""),
        }, page

    def report_list_page(scope, user_id=None, reporter=None):
        filters, page = read_report_filters()
        records, total = db.list_detections(
            app.config["DATABASE"], page=page, user_id=user_id, **filters
        )
        return render_template(
            "history.html",
            scope=scope,
            records=records,
            total=total,
            page=page,
            pages=max(1, ceil(total / 10)),
            reporter=reporter,
            search=filters["search"],
            source=filters["source"],
            status=filters["status"],
            risk=filters["risk"],
        )

    def csv_download(records, filename):
        return Response(
            detections_csv(records),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @app.get("/history")
    @login_required
    def history():
        """Old link kept working: send each role to its own report list."""
        target = "admin_reports" if is_admin() else "my_reports"
        return redirect(url_for(target, **request.args))

    @app.get("/my-reports")
    @user_required
    def my_reports():
        return report_list_page("mine", user_id=g.user["id"])

    @app.get("/my-reports/export.csv")
    @user_required
    def export_my_reports():
        filters, _page = read_report_filters()
        records = db.export_records(app.config["DATABASE"], user_id=g.user["id"], **filters)
        return csv_download(records, "my-garbage-reports.csv")

    @app.get("/admin/reports")
    @admin_required
    def admin_reports():
        reporter = None
        try:
            reporter_id = int(request.args.get("user", ""))
            reporter = db.get_user(app.config["DATABASE"], reporter_id)
        except ValueError:
            reporter_id = None
        return report_list_page(
            "all", user_id=reporter["id"] if reporter else None, reporter=reporter
        )

    @app.get("/admin/reports/export.csv")
    @admin_required
    def export_history():
        filters, _page = read_report_filters()
        try:
            user_id = int(request.args.get("user", ""))
        except ValueError:
            user_id = None
        records = db.export_records(app.config["DATABASE"], user_id=user_id, **filters)
        return csv_download(records, "smartcity-garbage-reports.csv")


    # ============================================
    # CASE MANAGEMENT (admin)
    # ============================================

    @app.get("/admin/cases")
    @admin_required
    def cases():
        refresh_notifications()
        status = request.args.get("status", "")
        risk = request.args.get("risk", "")
        return render_template(
            "cases.html",
            cases=db.list_cases(app.config["DATABASE"], status=status, risk=risk),
            issue_case_ids=db.open_issue_case_ids(app.config["DATABASE"]),
            status=status,
            risk=risk,
        )

    @app.get("/admin/case-detail")
    @admin_required
    def case_detail_index():
        return redirect(url_for("cases"))

    @app.get("/admin/cases/<int:case_id>")
    @admin_required
    def case_detail(case_id):
        case = db.get_case(app.config["DATABASE"], case_id)
        if case is None:
            abort(404)
        detection = db.get_detection(app.config["DATABASE"], case["detection_id"])
        return render_template(
            "case_detail.html",
            case=case,
            reporter=db.get_user(app.config["DATABASE"], detection["user_id"]) if detection else None,
            assignees=db.list_assignees(app.config["DATABASE"]),
            events=db.list_case_events(app.config["DATABASE"], case_id),
            issues=db.list_case_issues(app.config["DATABASE"], case_id),
        )

    @app.post("/admin/cases/<int:case_id>/assign")
    @admin_required
    def assign_case(case_id):
        try:
            assignee_id = int(request.form.get("assignee_id", ""))
        except ValueError:
            assignee_id = 0
        notes = (request.form.get("notes") or "").strip()[:1000]
        if db.assign_case(app.config["DATABASE"], case_id, assignee_id, notes):
            flash("Case assigned and moved to ASSIGNED.", "success")
        else:
            flash("This case cannot be assigned in its current state.", "error")
        return redirect(url_for("case_detail", case_id=case_id))

    @app.post("/admin/cases/<int:case_id>/status")
    @admin_required
    def update_case_status(case_id):
        new_status = request.form.get("status", "")
        notes = (request.form.get("notes") or "").strip()[:1000]
        if db.change_case_status(app.config["DATABASE"], case_id, new_status, notes):
            flash(f"Case moved to {new_status}.", "success")
        else:
            flash("That status change is not allowed.", "error")
        return redirect(url_for("case_detail", case_id=case_id))

    @app.post("/admin/cases/<int:case_id>/delete")
    @admin_required
    def delete_case(case_id):
        removable = db.delete_case(app.config["DATABASE"], case_id)
        if removable is None:
            flash("That case no longer exists.", "error")
        else:
            for relative in removable:
                target = safe_static_path(relative)
                if target:
                    target.unlink(missing_ok=True)
            flash(f"Case #{case_id} and its report were deleted.", "success")
        return redirect(request.form.get("back") if auth.is_safe_next(request.form.get("back"))
                        else url_for("cases"))

    @app.post("/admin/cases/<int:case_id>/after-image")
    @admin_required
    def upload_after_image(case_id):
        case = db.get_case(app.config["DATABASE"], case_id)
        upload = request.files.get("after_image")
        if case is None:
            abort(404)
        if case["status"] != "CLEANING":
            flash("After-cleaning evidence can only be added while a case is being cleaned.", "error")
            return redirect(url_for("case_detail", case_id=case_id))
        if upload is None or not upload.filename:
            flash("Choose an after-cleaning image to upload.", "error")
            return redirect(url_for("case_detail", case_id=case_id))
        if not allowed_file(upload.filename):
            flash("Only JPG, JPEG, PNG, and WEBP image files are supported.", "error")
            return redirect(url_for("case_detail", case_id=case_id))

        extension = Path(secure_filename(upload.filename)).suffix.lower()
        filename = f"after-case-{case_id}-{uuid.uuid4().hex}{extension}"
        target = Path(app.config["AFTER_UPLOAD_DIR"]) / filename
        upload.save(target)
        if cv2.imread(str(target)) is None:
            target.unlink(missing_ok=True)
            flash("The uploaded after-cleaning file is not a valid image.", "error")
            return redirect(url_for("case_detail", case_id=case_id))

        relative_path = f"uploads/{filename}"
        if not db.save_after_image(app.config["DATABASE"], case_id, relative_path):
            target.unlink(missing_ok=True)
            flash("After-cleaning evidence could not be saved for this case.", "error")
        else:
            flash("After-cleaning evidence uploaded.", "success")
        return redirect(url_for("case_detail", case_id=case_id))

    @app.post("/admin/cases/<int:case_id>/complete")
    @admin_required
    def complete_case(case_id):
        case = db.get_case(app.config["DATABASE"], case_id)
        if case is None:
            abort(404)
        after_image = safe_after_evidence_path(case["after_image_path"])
        if not after_image or not after_image.is_file():
            flash("Upload an after-cleaning image before completing this case.", "error")
            return redirect(url_for("case_detail", case_id=case_id))
        completion_notes = (request.form.get("completion_notes") or "").strip()[:1000]
        error = db.complete_case(app.config["DATABASE"], case_id, completion_notes)
        if error:
            flash(error, "error")
        else:
            flash("Case marked as resolved. Before/after evidence preserved.", "success")
        return redirect(url_for("case_detail", case_id=case_id))


    # ============================================
    # REPORT DETAILS (citizens: own reports only, admins: all)
    # ============================================

    @app.get("/reports/<int:detection_id>")
    @login_required
    def report(detection_id):
        record = db.get_detection(app.config["DATABASE"], detection_id)
        # A citizen asking for someone else's report gets the same 404 as a missing one.
        if not can_access_detection(record):
            abort(404)
        case = db.get_case_by_detection(app.config["DATABASE"], detection_id)
        return render_template(
            "report.html",
            record=record,
            case=case,
            reporter=db.get_user(app.config["DATABASE"], record["user_id"]) if is_admin() else None,
            events=db.list_case_events(app.config["DATABASE"], case["id"]) if case else [],
        )

    @app.post("/admin/reports/<int:detection_id>/delete")
    @admin_required
    def delete_history(detection_id):
        if db.detection_has_case(app.config["DATABASE"], detection_id):
            flash("This report is linked to a case. Delete the case instead.", "error")
            return redirect(url_for("admin_reports"))

        record = db.delete_detection(app.config["DATABASE"], detection_id)
        if record:
            for relative in (record["image_path"], record["result_path"]):
                target = safe_static_path(relative)
                if target:
                    target.unlink(missing_ok=True)
            flash("Report deleted.", "success")
        return redirect(url_for("admin_reports"))

    @app.post("/admin/reports/clear")
    @admin_required
    def clear_history():
        records = db.clear_detections(app.config["DATABASE"])
        for record in records:
            for relative in (record["image_path"], record["result_path"]):
                target = safe_static_path(relative)
                if target:
                    target.unlink(missing_ok=True)
        flash("Reports without a case were cleared. Case-linked evidence was preserved.", "success")
        return redirect(url_for("admin_reports"))


    # ============================================
    # NOTIFICATIONS (each role sees its own)
    # ============================================

    def notification_scope():
        if is_admin():
            return ("admin", None)
        if g.user["role"] == auth.ROLE_STAFF:
            return ("staff", g.user["id"])
        return ("user", g.user["id"])

    @app.get("/notifications")
    @login_required
    def notifications():
        if is_admin():
            refresh_notifications()
        audience, user_id = notification_scope()
        return render_template(
            "notifications.html",
            notifications=db.list_notifications(
                app.config["DATABASE"], limit=100, audience=audience, user_id=user_id
            ),
        )

    @app.post("/notifications/read-all")
    @login_required
    def notifications_read_all():
        audience, user_id = notification_scope()
        db.mark_notifications_read(app.config["DATABASE"], audience, user_id)
        flash("All notifications marked as read.", "success")
        return redirect(url_for("notifications"))

    @app.get("/notifications/<int:notification_id>/open")
    @login_required
    def notification_open(notification_id):
        audience, user_id = notification_scope()
        item = db.get_notification(app.config["DATABASE"], notification_id, audience, user_id)
        if item is None:
            abort(404)
        db.mark_notification_read(app.config["DATABASE"], notification_id)
        if is_admin():
            return redirect(url_for("case_detail", case_id=item["case_id"]))
        if g.user["role"] == auth.ROLE_STAFF:
            return redirect(url_for("staff_task", case_id=item["case_id"]))
        return redirect(url_for("report", detection_id=item["detection_id"]))


    # ============================================
    # MAP (citizens and admins)
    # ============================================

    @app.get("/map")
    @login_required
    def map_view():
        records = db.map_records(app.config["DATABASE"])

        # Same real-data helpers the dashboards use, so every map is always
        # backed by identical GPS/hotspot logic.
        map_data = serialize_map_records(records)
        hotspots = build_hotspots(records)
        attach_hotspot_membership(map_data, hotspots)
        location = admin_location()

        return render_template(
            "map.html",
            map_data=map_data,
            coordinate_count=len(map_data),
            hotspots=hotspots,
            summary=db.dashboard_data(app.config["DATABASE"])[0],
            admin_latitude=location["lat"],
            admin_longitude=location["lng"],
        )

    @app.get("/about")
    def about():
        return render_template("about.html")


    # ============================================
    # STAFF / MUNICIPAL WORKER DASHBOARD
    # ============================================

    def staff_case_or_404(case_id):
        """A case, only if it is currently assigned to the logged-in staff member."""
        case = db.get_staff_case(app.config["DATABASE"], case_id, g.user["id"])
        if case is None:
            abort(404)
        return case

    def save_staff_image(case_id, field, prefix):
        """Validate and store an uploaded evidence photo. Returns (relative_path, error)."""
        upload = request.files.get(field)
        if upload is None or not upload.filename:
            return None, "Choose an image to upload."
        if not allowed_file(upload.filename):
            return None, "Only JPG, JPEG, PNG, and WEBP image files are supported."
        extension = Path(secure_filename(upload.filename)).suffix.lower()
        filename = f"{prefix}-case-{case_id}-{uuid.uuid4().hex}{extension}"
        target = Path(app.config["AFTER_UPLOAD_DIR"]) / filename
        upload.save(target)
        if cv2.imread(str(target)) is None:
            target.unlink(missing_ok=True)
            return None, "The uploaded file is not a valid image."
        return f"uploads/{filename}", None

    @app.get("/staff")
    @staff_required
    def staff_dashboard():
        database = app.config["DATABASE"]
        tasks = db.list_staff_cases(database, g.user["id"])
        return render_template(
            "staff_dashboard.html",
            summary=db.staff_summary(database, g.user["id"]),
            active_tasks=[t for t in tasks if t["status"] in ("ASSIGNED", "CLEANING")],
            done_tasks=[t for t in tasks if t["status"] == "COMPLETED"],
            issues=db.list_staff_issues(database, g.user["id"]),
        )

    @app.get("/staff/tasks/<int:case_id>")
    @staff_required
    def staff_task(case_id):
        case = staff_case_or_404(case_id)
        return render_template(
            "staff_task.html",
            case=case,
            events=db.list_case_events(app.config["DATABASE"], case_id),
        )

    @app.post("/staff/tasks/<int:case_id>/start")
    @staff_required
    def staff_start_task(case_id):
        case = staff_case_or_404(case_id)
        if case["status"] != "ASSIGNED":
            flash("This task has already been started.", "error")
        else:
            notes = (request.form.get("notes") or "").strip()[:1000] or (case["notes"] or "")
            if db.change_case_status(app.config["DATABASE"], case_id, "CLEANING", notes):
                flash("Task accepted. Status is now In Progress - upload the before-cleaning photo next.", "success")
            else:
                flash("That status change is not allowed.", "error")
        return redirect(url_for("staff_task", case_id=case_id))

    @app.post("/staff/tasks/<int:case_id>/before-image")
    @staff_required
    def staff_before_image(case_id):
        case = staff_case_or_404(case_id)
        if case["status"] != "CLEANING":
            flash("Accept the task first, then upload the before-cleaning photo.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        relative, error = save_staff_image(case_id, "before_image", "before")
        if error:
            flash(error, "error")
            return redirect(url_for("staff_task", case_id=case_id))
        saved, previous = db.save_before_image(app.config["DATABASE"], case_id, relative)
        if not saved:
            (Path(app.config["AFTER_UPLOAD_DIR"]) / Path(relative).name).unlink(missing_ok=True)
            flash("Before-cleaning evidence could not be saved for this task.", "error")
        else:
            if previous:
                old = safe_static_path(previous)
                if old:
                    old.unlink(missing_ok=True)
            flash("Before-cleaning evidence uploaded.", "success")
        return redirect(url_for("staff_task", case_id=case_id))

    @app.post("/staff/tasks/<int:case_id>/after-image")
    @staff_required
    def staff_after_image(case_id):
        case = staff_case_or_404(case_id)
        if case["status"] != "CLEANING":
            flash("After-cleaning evidence can only be added while the task is In Progress.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        if not case["before_image_path"]:
            flash("Upload the before-cleaning photo first.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        if case["after_image_path"]:
            flash("After-cleaning evidence was already uploaded.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        relative, error = save_staff_image(case_id, "after_image", "after")
        if error:
            flash(error, "error")
        elif not db.save_after_image(app.config["DATABASE"], case_id, relative):
            (Path(app.config["AFTER_UPLOAD_DIR"]) / Path(relative).name).unlink(missing_ok=True)
            flash("After-cleaning evidence could not be saved for this task.", "error")
        else:
            flash("After-cleaning evidence uploaded. You can now mark the task as cleaned.", "success")
        return redirect(url_for("staff_task", case_id=case_id))

    @app.post("/staff/tasks/<int:case_id>/complete")
    @staff_required
    def staff_complete_task(case_id):
        case = staff_case_or_404(case_id)
        if not case["before_image_path"]:
            flash("Upload the before-cleaning photo before completing this task.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        after = safe_after_evidence_path(case["after_image_path"])
        if not after or not after.is_file():
            flash("Upload the after-cleaning photo before completing this task.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        notes = (request.form.get("completion_notes") or "").strip()[:1000]
        error = db.complete_case(app.config["DATABASE"], case_id, notes)
        if error:
            flash(error, "error")
        else:
            flash("Task marked as Cleaned. Thank you!", "success")
        return redirect(url_for("staff_task", case_id=case_id))

    STAFF_ISSUE_REASONS = (
        "Location not accessible", "Private property - entry refused", "Unsafe / hazardous waste",
        "Needs heavy machinery or a bigger team", "Garbage already removed", "Wrong location / not found",
        "Other",
    )

    @app.post("/staff/tasks/<int:case_id>/issue")
    @staff_required
    def staff_report_issue(case_id):
        case = staff_case_or_404(case_id)
        reason = (request.form.get("reason") or "").strip()
        details = (request.form.get("details") or "").strip()[:1000]
        if reason not in STAFF_ISSUE_REASONS:
            flash("Choose the reason the location cannot be cleaned.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        if reason == "Other" and not details:
            flash("Please describe the problem.", "error")
            return redirect(url_for("staff_task", case_id=case_id))
        if db.report_case_issue(app.config["DATABASE"], case["id"], g.user["id"], reason, details):
            flash("Issue reported. The admin was notified and will reassign this case.", "success")
            return redirect(url_for("staff_dashboard"))
        flash("An issue can only be reported on an active task.", "error")
        return redirect(url_for("staff_task", case_id=case_id))

    app.config["STAFF_ISSUE_REASONS"] = STAFF_ISSUE_REASONS


    # ============================================
    # USER MANAGEMENT (admin)
    # ============================================

    @app.get("/admin/users")
    @admin_required
    def users_admin():
        search = request.args.get("q", "").strip()
        return render_template(
            "users.html",
            users=db.list_users(app.config["DATABASE"], search=search),
            search=search,
            errors=[],
            form={},
        )

    @app.post("/admin/users/add")
    @admin_required
    def add_user():
        values, errors = validate_account_fields(request.form)
        if errors:
            values.pop("password", None)
            return render_template(
                "users.html", users=db.list_users(app.config["DATABASE"]),
                search="", errors=errors, form=values,
            ), 400
        if request.form.get("role") == auth.ROLE_STAFF:
            db.create_staff(
                app.config["DATABASE"], username=values["username"], full_name=values["full_name"],
                email=values["email"], phone=values["phone"],
                password_hash=generate_password_hash(values["password"]),
            )
            flash(f"Staff member {values['username']} was created. You can now assign cases to them.", "success")
            return redirect(url_for("users_admin"))
        db.create_user(
            app.config["DATABASE"], username=values["username"], full_name=values["full_name"],
            email=values["email"], phone=values["phone"], role=auth.ROLE_USER,
            password_hash=generate_password_hash(values["password"]),
        )
        flash(f"User {values['username']} was created.", "success")
        return redirect(url_for("users_admin"))

    @app.post("/admin/users/<int:user_id>/toggle")
    @admin_required
    def toggle_user(user_id):
        user = db.get_user(app.config["DATABASE"], user_id)
        if user is None or user["role"] not in (auth.ROLE_USER, auth.ROLE_STAFF):
            abort(404)
        db.set_user_active(app.config["DATABASE"], user_id, not user["is_active"])
        flash(
            f"{user['full_name']} was {'deactivated' if user['is_active'] else 'reactivated'}.",
            "success",
        )
        return redirect(url_for("users_admin"))

    @app.post("/admin/users/<int:user_id>/delete")
    @admin_required
    def delete_user(user_id):
        user = db.delete_user(app.config["DATABASE"], user_id)
        if user is None:
            abort(404)
        flash(f"Account {user['username']} was deleted. Their reports were kept.", "success")
        return redirect(url_for("users_admin"))

    @app.get("/admin/user-reports")
    @admin_required
    def user_reports():
        return render_template("user_reports.html", stats=db.user_report_stats(app.config["DATABASE"]))


    # ============================================
    # INSTALLABLE APP (PWA) - Android, iPhone/iPad and desktop
    # ============================================

    @app.get("/manifest.webmanifest")
    def pwa_manifest():
        icon = lambda name: url_for("static", filename=f"icons/{name}")
        manifest = {
            "name": "SmartCity AI Garbage Detection",
            "short_name": "SmartCity",
            "description": "Report garbage with your camera, track cleanup cases and view the city hotspot map.",
            "id": "/",
            "start_url": "/",
            "scope": "/",
            "display": "standalone",
            "orientation": "any",
            "background_color": "#0b1f4d",
            "theme_color": "#0b1f4d",
            "icons": [
                {"src": icon("icon-192.png"), "sizes": "192x192", "type": "image/png", "purpose": "any"},
                {"src": icon("icon-512.png"), "sizes": "512x512", "type": "image/png", "purpose": "any"},
                {"src": icon("icon-maskable-512.png"), "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
            ],
            "shortcuts": [
                {"name": "Report Garbage", "url": "/detect"},
                {"name": "Hotspot Map", "url": "/map"},
            ],
        }
        response = jsonify(manifest)
        response.mimetype = "application/manifest+json"
        return response

    @app.get("/sw.js")
    def pwa_service_worker():
        # Served from the site root so the worker can control every page.
        response = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Service-Worker-Allowed"] = "/"
        return response

    @app.get("/offline")
    def pwa_offline():
        return render_template("offline.html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}


    @app.get("/admin/diagnostics")
    @admin_required
    def admin_diagnostics():
        """Open /admin/diagnostics to see how fast detection is on this server."""
        detector = app.extensions["detector"]
        info = {"cpu_count": os.cpu_count()}
        try:
            started = time.perf_counter()
            detector.self_check()
            info["model_ready_ms_first_call"] = round((time.perf_counter() - started) * 1000)
            samples = []
            import numpy as _np
            blank = _np.full((640, 640, 3), 255, dtype=_np.uint8)
            for _ in range(3):
                started = time.perf_counter()
                detector._inspect_image(blank, 0.25, app.config["DETECTION_IMAGE_SIZE"])
                samples.append(round((time.perf_counter() - started) * 1000))
            info["one_detection_ms"] = samples
            info["model"] = str(detector.active_model_path)
            info["camera_capture_estimate_s"] = round(sum(samples) / len(samples) * 5 / 1000, 1)
        except Exception as exc:
            info["error"] = repr(exc)
        try:
            with open("/proc/meminfo") as handle:
                lines = dict(line.split(":", 1) for line in handle.read().splitlines() if ":" in line)
            info["memory_total_mb"] = int(lines["MemTotal"].split()[0]) // 1024
            info["memory_available_mb"] = int(lines["MemAvailable"].split()[0]) // 1024
        except Exception:
            pass
        return info

    # ============================================
    # ADMIN LOCATION
    # ============================================

    @app.route("/admin/location", methods=["GET", "POST"])
    @admin_required
    def admin_location_page():
        errors = []
        location = admin_location()
        if request.method == "POST":
            label = (request.form.get("label") or "").strip()[:80] or "Admin Office"
            state = (request.form.get("state") or "").strip()[:80]
            city = (request.form.get("city") or "").strip()[:80]
            state = geocode.canonical_state(state) or state
            raw_lat = (request.form.get("latitude") or "").strip()
            raw_lng = (request.form.get("longitude") or "").strip()
            latitude = longitude = None
            if raw_lat or raw_lng:
                try:
                    latitude, longitude = float(raw_lat), float(raw_lng)
                    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                        raise ValueError
                except ValueError:
                    errors.append("Enter a valid latitude (-90 to 90) and longitude (-180 to 180), or leave both empty.")
            else:
                # Coordinates are optional: work them out from the city / state.
                found = geocode.forward(city, state) if (city or state) else None
                if found:
                    latitude, longitude = found["lat"], found["lng"]
                else:
                    errors.append(
                        "Choose a state or city so the location can be found automatically, "
                        "or enter latitude and longitude / click the map."
                    )
            if not errors:
                db.set_settings(app.config["DATABASE"], {
                    "admin_lat": latitude, "admin_lng": longitude, "admin_label": label,
                    "admin_state": state, "admin_city": city,
                })
                flash("Admin location saved. Map distances now use this point.", "success")
                return redirect(url_for("admin_location_page"))
            location = {
                "lat": raw_lat or location["lat"], "lng": raw_lng or location["lng"],
                "label": label, "state": state, "city": city,
            }
        return render_template(
            "admin_location.html", location=location, errors=errors,
            states=sorted(geocode.STATE_CAPITALS),
        )

    @app.get("/admin/location/lookup")
    @admin_required
    def admin_location_lookup():
        """JSON helper for the location form: city/state -> coordinates, or coordinates -> city/state."""
        lat, lng = request.args.get("lat"), request.args.get("lng")
        if lat not in (None, "") and lng not in (None, ""):
            try:
                latitude, longitude = optional_coordinates(lat, lng)
            except ValueError:
                return {"ok": False, "error": "Invalid coordinates."}, 400
            found = geocode.reverse(latitude, longitude)
            return ({"ok": True, **found} if found else {"ok": False, "error": "No place name found for that point."})
        found = geocode.forward(request.args.get("city", ""), request.args.get("state", ""))
        if not found:
            return {"ok": False, "error": "Could not find that place. Check the spelling or click the map."}
        return {"ok": True, **found}


    # ============================================
    # DOWNLOAD RESULT
    # ============================================

    @app.get(
        "/download/<int:detection_id>"
    )
    @login_required
    def download_result(detection_id):

        record = db.get_detection(

            app.config[
                "DATABASE"
            ],

            detection_id,
        )

        if not can_access_detection(record):
            abort(404)

        target = (

            safe_static_path(
                record[
                    "result_path"
                ]
            )

            if record

            else None
        )


        if (
            not target
            or not target.is_file()
        ):

            abort(404)


        return send_file(

            target,

            as_attachment=True,

            download_name=(
                f"garbage-detection-"
                f"{detection_id}.jpg"
            ),
        )


    # ============================================
    # FILE TOO LARGE
    # ============================================

    @app.errorhandler(413)
    def too_large(_error):

        if request.path.startswith("/api/"):
            return {"ok": False, "error": "The photo is too large to upload. Try again."}, 413

        flash(

            "Image is too large. "
            "The maximum upload size is 32 MB.",

            "error",
        )


        return redirect(
            url_for("detect")
        )


    # ============================================
    # ERROR PAGES
    # ============================================

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template(
            "error.html", code=403, title="Access denied",
            message="Your account does not have permission to open this page.",
        ), 403

    @app.errorhandler(404)
    def not_found(_error):
        return render_template(
            "error.html", code=404, title="Page not found",
            message="The page you are looking for does not exist or is not available to you.",
        ), 404

    # Hosted behind an HTTPS proxy (Render, Railway, Fly, Nginx...): trust the
    # forwarded headers and mark the login cookie Secure. HTTPS is what lets
    # phones open the camera and install the app.
    if os.environ.get("SMARTCITY_BEHIND_PROXY") or os.environ.get("RENDER"):
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
        app.config["SESSION_COOKIE_SECURE"] = True

    return app


# ================================================
# APPLICATION INSTANCE
# ================================================

app = create_app()


# ================================================
# RUN APPLICATION
# ================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0", port=5000, debug=True
        
    )
