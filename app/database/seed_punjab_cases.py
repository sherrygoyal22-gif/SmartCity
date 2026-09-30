"""One-off script to seed demo garbage-report cases across several Punjab
cities, so the Hotspot Map has more than one cluster to show off.

Run once with:  python database/seed_punjab_cases.py
Safe to re-run: it skips seeding if any of these demo rows already exist
(checked via the SEED_TAG marker stored in duplicate_reason).
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "smartcity.db"
SEED_TAG = "demo-seed-punjab-v1"

# (city, address, lat, lng, risk_level, risk_score, detection_count,
#  confidence, box_area_ratio, image_stem, image_ext, source,
#  hours_ago, case_status, assignee_id)
CASES = [
    ("Ludhiana", "Sarabha Nagar Market, Ludhiana", 30.8952, 75.8459,
     "HIGH", 82, 6, 91.4, 0.27, "1343fbb67a44426d9c6fe7772aa92dd1", "jpg",
     "camera", 52, "COMPLETED", 1),
    ("Amritsar", "Ranjit Avenue Market, Amritsar", 31.6340, 74.8723,
     "MEDIUM", 48, 3, 72.0, 0.11, "b710576083ae43db9d6ed5da47162be8", "png",
     "upload", 26, "ASSIGNED", 3),
    ("Jalandhar", "Model Town Extension, Jalandhar", 31.3260, 75.5762,
     "LOW", 24, 1, 58.5, 0.05, "d46adc2d0c9f49b18ac9657f14d8d90a", "jpg",
     "upload", 6, "PENDING", None),
    ("Patiala", "Leela Bhawan Chowk, Patiala", 30.3398, 76.3869,
     "HIGH", 78, 5, 88.0, 0.24, "e2fe7022df424df39cfd554f48fda068", "jpg",
     "camera", 70, "CLEANING", 2),
    ("Bathinda", "Guru Kanshi Nagar, Bathinda", 30.2110, 74.9455,
     "MEDIUM", 52, 3, 74.5, 0.13, "e56243644304438699fb195ca64ed155", "jpg",
     "upload", 13, "PENDING", None),
    ("Mohali", "Phase 7 Community Park, Mohali", 30.7046, 76.7179,
     "LOW", 22, 1, 55.0, 0.04, "30865ad17e324b34b4bd48a739c370a3", "jpg",
     "upload", 96, "COMPLETED", 1),
    ("Hoshiarpur", "Adalat Bazaar, Hoshiarpur", 31.5310, 75.9145,
     "MEDIUM", 45, 2, 69.0, 0.10, "00887407ad2149d09352d20e217e3d0b", "jpg",
     "camera", 46, "ASSIGNED", 3),
    ("Moga", "Railway Road, Moga", 30.8158, 75.1712,
     "HIGH", 85, 7, 93.0, 0.29, "d4452be102ae4e3caa445f0f25b2225d", "jpg",
     "camera", 1, "PENDING", None),
    ("Ferozepur", "Zira Road, Ferozepur", 30.9331, 74.6225,
     "LOW", 27, 2, 61.0, 0.06, "ebe011ca969f47eb9b84d614afecc851", "jpg",
     "upload", 9, "PENDING", None),
    ("Pathankot", "Dhangu Road, Pathankot", 32.2746, 75.6520,
     "MEDIUM", 55, 3, 76.0, 0.14, "ce3b35599e214efab5bd9d92da33ad39", "jpg",
     "upload", 22, "CLEANING", 2),
]

RISK_REASON = {
    "HIGH": "High risk: high confidence, large detected area.",
    "MEDIUM": "Medium risk: moderate confidence, medium detected area.",
    "LOW": "Low risk: lower confidence, small detected area.",
}


def already_seeded(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM detections WHERE duplicate_reason = ? LIMIT 1", (SEED_TAG,)
    ).fetchone()
    return row is not None


def run() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    if already_seeded(connection):
        print("Demo Punjab cases already seeded — nothing to do.")
        connection.close()
        return

    now = datetime.now(timezone.utc)

    for (city, address, lat, lng, risk_level, risk_score, count, confidence,
         box_ratio, image_stem, ext, source, hours_ago, status,
         assignee_id) in CASES:
        created_at = (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
        image_path = f"uploads/{image_stem}.{ext}"
        result_path = f"results/{image_stem}_detected.jpg"
        image_hash = hashlib.md5(f"{SEED_TAG}:{city}".encode()).hexdigest()

        cursor = connection.execute(
            """INSERT INTO detections (
                created_at, detection_count, confidence, image_path, result_path,
                latitude, longitude, address, source, image_hash, box_area_ratio,
                is_valid, validation_reason, is_duplicate, duplicate_reason,
                risk_level, risk_score, risk_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1,
                'Detection passed the configured confidence validation.',
                0, ?, ?, ?, ?)""",
            (created_at, count, confidence, image_path, result_path, lat, lng,
             address, source, image_hash, box_ratio, SEED_TAG, risk_level,
             risk_score, RISK_REASON[risk_level]),
        )
        detection_id = cursor.lastrowid

        cursor = connection.execute(
            """INSERT INTO alerts (
                detection_id, created_at, status, risk_level, risk_score,
                confidence, latitude, longitude, address
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (detection_id, created_at, status, risk_level, risk_score,
             confidence, lat, lng, address),
        )
        alert_id = cursor.lastrowid

        case_notes = f"Assigned for cleanup in {city}." if assignee_id else None
        completion_notes = "Cleanup verified on-site; area cleared." if status == "COMPLETED" else None
        completed_at = created_at if status == "COMPLETED" else None
        after_image = (
            "uploads/after-case-5-b107bae6ad1442f9b229f464ab04596f.jpeg"
            if status == "COMPLETED" else None
        )
        resolution_minutes = 180 if status == "COMPLETED" else None

        cursor = connection.execute(
            """INSERT INTO cases (
                alert_id, detection_id, status, assignee_id, notes, created_at,
                updated_at, after_image_path, completed_at, completion_notes,
                resolution_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert_id, detection_id, status, assignee_id, case_notes,
             created_at, created_at, after_image, completed_at,
             completion_notes, resolution_minutes),
        )
        case_id = cursor.lastrowid

        events = [(case_id, "ALERT_CREATED",
                   f"Alert #{alert_id} created from validated detection #{detection_id}.")]
        if status in {"ASSIGNED", "CLEANING", "COMPLETED"} and assignee_id:
            assignee_name = connection.execute(
                "SELECT name FROM assignees WHERE id = ?", (assignee_id,)
            ).fetchone()["name"]
            events.append((case_id, "ASSIGNED", f"Case assigned to {assignee_name}."))
        if status in {"CLEANING", "COMPLETED"}:
            events.append((case_id, "STATUS_CHANGED", "Status changed from ASSIGNED to CLEANING."))
        if status == "COMPLETED":
            events.append((case_id, "CASE_COMPLETED", "Case completed after on-site cleanup."))
        connection.executemany(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, ?, ?)",
            events,
        )

        connection.execute(
            """INSERT INTO notifications (
                case_id, notification_type, message, dedupe_key, created_at, is_read
            ) VALUES (?, 'NEW_ALERT', ?, ?, ?, 1)""",
            (case_id, f"New {risk_level.title()}-risk garbage case #{case_id} was created in {city}.",
             f"case:{case_id}:new-alert", created_at),
        )

        print(f"Seeded case #{case_id} — {city} ({risk_level}, {status})")

    connection.commit()
    connection.close()
    print("Done. Refresh the Hotspot Map / Dashboard to see the new cities.")


if __name__ == "__main__":
    run()
