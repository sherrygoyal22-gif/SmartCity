from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    detection_count INTEGER NOT NULL,
    confidence REAL,
    image_path TEXT NOT NULL,
    result_path TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    address TEXT,
    source TEXT NOT NULL CHECK(source IN ('upload', 'camera'))
);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source);

CREATE TABLE IF NOT EXISTS assignees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN ('WORKER', 'TEAM')),
    is_demo INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL UNIQUE REFERENCES detections(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'PENDING',
    risk_level TEXT,
    risk_score INTEGER,
    confidence REAL,
    latitude REAL,
    longitude REAL,
    address TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL UNIQUE REFERENCES alerts(id),
    detection_id INTEGER NOT NULL UNIQUE REFERENCES detections(id),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'ASSIGNED', 'CLEANING', 'COMPLETED')),
    assignee_id INTEGER REFERENCES assignees(id),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS case_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email TEXT UNIQUE COLLATE NOCASE,
    full_name TEXT NOT NULL,
    phone TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user', 'staff')),
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS case_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    staff_id INTEGER REFERENCES users(id),
    reason TEXT NOT NULL,
    details TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'RESOLVED')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id),
    notification_type TEXT NOT NULL,
    message TEXT NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_read INTEGER NOT NULL DEFAULT 0
);
"""

MIGRATION_COLUMNS = {
    "image_hash": "TEXT",
    "box_area_ratio": "REAL",
    "is_valid": "INTEGER NOT NULL DEFAULT 1",
    "validation_reason": "TEXT",
    "is_duplicate": "INTEGER NOT NULL DEFAULT 0",
    "duplicate_reason": "TEXT",
    "risk_level": "TEXT",
    "risk_score": "INTEGER",
    "risk_reason": "TEXT",
    "user_id": "INTEGER",
}

NOTIFICATION_MIGRATION_COLUMNS = {
    # 'admin' notifications go to the municipal team, 'user' ones to the reporter.
    "audience": "TEXT NOT NULL DEFAULT 'admin'",
}

ASSIGNEE_MIGRATION_COLUMNS = {
    # Links a staff login to the assignee row used by the existing case workflow.
    "user_id": "INTEGER",
}

CASE_MIGRATION_COLUMNS = {
    "before_image_path": "TEXT",
    "started_at": "TEXT",
    "after_image_path": "TEXT",
    "completed_at": "TEXT",
    "completion_notes": "TEXT",
    "resolution_minutes": "INTEGER",
}


@contextmanager
def connect(database_path: str | Path):
    """Provide a transaction-scoped SQLite connection and always close it."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(database_path: str | Path) -> None:
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)
        # Existing Version 2 databases receive additive columns only.
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(detections)")
        }
        for name, definition in MIGRATION_COLUMNS.items():
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE detections ADD COLUMN {name} {definition}")
        existing_case_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(cases)")
        }
        for name, definition in CASE_MIGRATION_COLUMNS.items():
            if name not in existing_case_columns:
                connection.execute(f"ALTER TABLE cases ADD COLUMN {name} {definition}")
        existing_notification_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(notifications)")
        }
        for name, definition in NOTIFICATION_MIGRATION_COLUMNS.items():
            if name not in existing_notification_columns:
                connection.execute(f"ALTER TABLE notifications ADD COLUMN {name} {definition}")
        existing_assignee_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(assignees)")
        }
        for name, definition in ASSIGNEE_MIGRATION_COLUMNS.items():
            if name not in existing_assignee_columns:
                connection.execute(f"ALTER TABLE assignees ADD COLUMN {name} {definition}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_user ON detections(user_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_issues_case ON case_issues(case_id, status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_image_hash ON detections(image_hash)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_coordinates_created "
            "ON detections(latitude, longitude, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status, updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_case_events_case ON case_events(case_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(is_read, created_at DESC)"
        )
        connection.executemany(
            "INSERT OR IGNORE INTO assignees (name, kind, is_demo) VALUES (?, ?, 1)",
            [
                ("Demo Worker — Aman", "WORKER"),
                ("Demo Worker — Priya", "WORKER"),
                ("Demo Team — Ward 4", "TEAM"),
            ],
        )
    _upgrade_users_table(database_path)


def _upgrade_users_table(database_path) -> None:
    """Bring an existing ``users`` table up to the current shape, keeping every row.

    * adds the ``staff`` role to the role CHECK constraint (SQLite cannot alter a
      CHECK, so the table is rebuilt), and
    * converts the older column layout (name/status) to full_name/is_active.
    """
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        if row is None:
            return
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(users)")}
        if "'staff'" in (row["sql"] or "") and "full_name" in columns and "is_active" in columns:
            return
        old_rows = [dict(r) for r in connection.execute("SELECT * FROM users")]
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("ALTER TABLE users RENAME TO users_before_staff_upgrade")
        connection.execute(
            """CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT UNIQUE COLLATE NOCASE,
                full_name TEXT NOT NULL,
                phone TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user', 'staff')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )"""
        )
        for old in old_rows:
            active = old.get("is_active")
            if active is None:
                active = 1 if old.get("status", "active") == "active" else 0
            connection.execute(
                """INSERT INTO users (id, username, email, full_name, phone, password_hash, role,
                                      is_active, created_at, last_login_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)""",
                (
                    old["id"], old["username"], old.get("email") or None,
                    old.get("full_name") or old.get("name") or old["username"],
                    old.get("phone"), old["password_hash"],
                    old.get("role") if old.get("role") in ("admin", "user", "staff") else "user",
                    1 if active else 0,
                    old.get("created_at") or None, old.get("last_login_at"),
                ),
            )
        connection.execute("DROP TABLE users_before_staff_upgrade")
        connection.commit()
    finally:
        connection.close()


def create_detection(database_path, *, user_id=None, detection_count, confidence, image_path, result_path, latitude, longitude, address, source, image_hash, box_area_ratio, is_valid, validation_reason, is_duplicate, duplicate_reason, risk_level, risk_score, risk_reason):
    with connect(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO detections (
                detection_count, confidence, image_path, result_path, latitude, longitude,
                address, source, image_hash, box_area_ratio, is_valid, validation_reason,
                is_duplicate, duplicate_reason, risk_level, risk_score, risk_reason, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (detection_count, confidence, image_path, result_path, latitude, longitude,
             address, source, image_hash, box_area_ratio, is_valid, validation_reason,
             is_duplicate, duplicate_reason, risk_level, risk_score, risk_reason, user_id),
        )
        return cursor.lastrowid


def find_detection_by_hash(database_path, image_hash):
    with connect(database_path) as connection:
        return connection.execute(
            "SELECT * FROM detections WHERE image_hash = ? ORDER BY id ASC LIMIT 1",
            (image_hash,),
        ).fetchone()


def find_recent_location_duplicate(database_path, *, latitude, longitude, after_timestamp, tolerance_degrees):
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT * FROM detections
               WHERE detection_count > 0
                 AND latitude IS NOT NULL AND longitude IS NOT NULL
                 AND created_at >= ?
                 AND ABS(latitude - ?) <= ?
                 AND ABS(longitude - ?) <= ?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (after_timestamp, latitude, tolerance_degrees, longitude, tolerance_degrees),
        ).fetchone()


def get_detection(database_path, detection_id):
    with connect(database_path) as connection:
        return connection.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()


CASE_SELECT = """
SELECT c.id, c.alert_id, c.detection_id, c.status, c.notes, c.created_at,
       c.updated_at, c.after_image_path, c.completed_at, c.completion_notes,
       c.resolution_minutes, c.before_image_path, c.started_at,
       assignees.user_id AS assignee_user_id, a.risk_level, a.risk_score, a.confidence, a.latitude,
       a.longitude, a.address, a.created_at AS alert_created_at,
       d.image_path, d.result_path, d.detection_count, d.validation_reason,
       d.is_valid, d.is_duplicate, d.duplicate_reason, d.risk_reason,
       assignees.name AS assignee_name, assignees.kind AS assignee_kind
FROM cases c
JOIN alerts a ON a.id = c.alert_id
JOIN detections d ON d.id = c.detection_id
LEFT JOIN assignees ON assignees.id = c.assignee_id
"""


def create_alert_and_case(database_path, detection_id):
    """Create the single alert/case pair for an eligible detection."""
    with connect(database_path) as connection:
        detection = connection.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
        if detection is None or not detection["is_valid"] or not detection["detection_count"]:
            return None
        existing = connection.execute(
            "SELECT id FROM cases WHERE detection_id = ?", (detection_id,)
        ).fetchone()
        if existing:
            return existing["id"]
        cursor = connection.execute(
            """INSERT INTO alerts (
                detection_id, risk_level, risk_score, confidence, latitude, longitude, address
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (detection_id, detection["risk_level"], detection["risk_score"],
             detection["confidence"], detection["latitude"], detection["longitude"],
             detection["address"]),
        )
        alert_id = cursor.lastrowid
        cursor = connection.execute(
            "INSERT INTO cases (alert_id, detection_id) VALUES (?, ?)",
            (alert_id, detection_id),
        )
        case_id = cursor.lastrowid
        connection.executemany(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, ?, ?)",
            [
                (case_id, "ALERT_CREATED", f"Alert #{alert_id} created from validated detection #{detection_id}."),
                (case_id, "CASE_CREATED", "Case opened with PENDING status."),
            ],
        )
        create_notification(
            connection, case_id, "NEW_ALERT",
            f"New {detection['risk_level'] or 'unrated'}-risk garbage case #{case_id} was created.",
            f"case:{case_id}:new-alert",
        )
        if detection["risk_level"] == "HIGH":
            create_notification(
                connection, case_id, "HIGH_RISK",
                f"High-risk garbage case #{case_id} needs priority attention.",
                f"case:{case_id}:high-risk",
            )
        create_notification(
            connection, case_id, "REPORT_RECEIVED",
            f"Your report #{detection_id} was received and is waiting to be assigned.",
            f"case:{case_id}:user:received", audience="user",
        )
        return case_id


def create_notification(connection, case_id, notification_type, message, dedupe_key, audience="admin"):
    connection.execute(
        """INSERT OR IGNORE INTO notifications
           (case_id, notification_type, message, dedupe_key, audience) VALUES (?, ?, ?, ?, ?)""",
        (case_id, notification_type, message, dedupe_key, audience),
    )


def _notification_scope(audience, user_id):
    """SQL fragment limiting notifications to one audience (and one reporter)."""
    if audience == "user":
        return "n.audience = 'user' AND d.user_id = ?", [user_id]
    if audience == "staff":
        return (
            "n.audience = 'staff' AND c.assignee_id IN (SELECT id FROM assignees WHERE user_id = ?)",
            [user_id],
        )
    return "n.audience = 'admin'", []


def list_notifications(database_path, limit=20, audience="admin", user_id=None):
    where, params = _notification_scope(audience, user_id)
    with connect(database_path) as connection:
        return connection.execute(
            f"""SELECT n.*, c.status, c.detection_id, a.risk_level FROM notifications n
               JOIN cases c ON c.id = n.case_id
               JOIN alerts a ON a.id = c.alert_id
               JOIN detections d ON d.id = c.detection_id
               WHERE {where}
               ORDER BY n.created_at DESC, n.id DESC LIMIT ?""",
            [*params, limit],
        ).fetchall()


def unread_notification_count(database_path, audience="admin", user_id=None):
    where, params = _notification_scope(audience, user_id)
    with connect(database_path) as connection:
        return connection.execute(
            f"""SELECT COUNT(*) FROM notifications n
                JOIN cases c ON c.id = n.case_id
                JOIN detections d ON d.id = c.detection_id
                WHERE n.is_read = 0 AND {where}""",
            params,
        ).fetchone()[0]


def mark_notifications_read(database_path, audience="admin", user_id=None):
    where, params = _notification_scope(audience, user_id)
    with connect(database_path) as connection:
        connection.execute(
            f"""UPDATE notifications SET is_read = 1 WHERE is_read = 0 AND id IN (
                    SELECT n.id FROM notifications n
                    JOIN cases c ON c.id = n.case_id
                    JOIN detections d ON d.id = c.detection_id
                    WHERE {where})""",
            params,
        )


def get_notification(database_path, notification_id, audience="admin", user_id=None):
    """Fetch one notification the viewer is allowed to see, else None."""
    where, params = _notification_scope(audience, user_id)
    with connect(database_path) as connection:
        return connection.execute(
            f"""SELECT n.*, c.detection_id FROM notifications n
                JOIN cases c ON c.id = n.case_id
                JOIN detections d ON d.id = c.detection_id
                WHERE n.id = ? AND {where}""",
            [notification_id, *params],
        ).fetchone()


def mark_notification_read(database_path, notification_id):
    with connect(database_path) as connection:
        connection.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notification_id,))


def refresh_case_notifications(database_path, reminder_after_hours, escalation_after_hours):
    """Create at most one reminder and one escalation per unresolved case."""
    with connect(database_path) as connection:
        cases = connection.execute(
            """SELECT c.id, c.status,
                      (julianday(CURRENT_TIMESTAMP) - julianday(c.created_at)) * 24 AS age_hours
               FROM cases c WHERE c.status IN ('PENDING', 'ASSIGNED', 'CLEANING')"""
        ).fetchall()
        for case in cases:
            if case["age_hours"] >= escalation_after_hours:
                create_notification(
                    connection, case["id"], "ESCALATION",
                    f"Case #{case['id']} has exceeded the {escalation_after_hours}-hour resolution target.",
                    f"case:{case['id']}:escalation",
                )
            elif case["age_hours"] >= reminder_after_hours:
                create_notification(
                    connection, case["id"], "REMINDER",
                    f"Case #{case['id']} is still {case['status'].lower()} after {reminder_after_hours} hours.",
                    f"case:{case['id']}:reminder",
                )


def find_active_nearby_case(database_path, *, latitude, longitude, after_timestamp, tolerance_degrees):
    if latitude is None or longitude is None:
        return None
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT c.id FROM cases c
               JOIN alerts a ON a.id = c.alert_id
               WHERE c.status IN ('PENDING', 'ASSIGNED', 'CLEANING')
                 AND a.latitude IS NOT NULL AND a.longitude IS NOT NULL
                 AND c.created_at >= ?
                 AND ABS(a.latitude - ?) <= ?
                 AND ABS(a.longitude - ?) <= ?
               ORDER BY c.created_at DESC, c.id DESC LIMIT 1""",
            (after_timestamp, latitude, tolerance_degrees, longitude, tolerance_degrees),
        ).fetchone()


def list_cases(database_path, *, status="", risk=""):
    filters, params = [], []
    if status in {"PENDING", "ASSIGNED", "CLEANING", "COMPLETED"}:
        filters.append("c.status = ?")
        params.append(status)
    if risk in {"HIGH", "MEDIUM", "LOW"}:
        filters.append("a.risk_level = ?")
        params.append(risk)
    clause = " WHERE " + " AND ".join(filters) if filters else ""
    with connect(database_path) as connection:
        return connection.execute(
            CASE_SELECT + clause + " ORDER BY CASE a.risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END, c.created_at DESC",
            params,
        ).fetchall()


def get_case(database_path, case_id):
    with connect(database_path) as connection:
        return connection.execute(CASE_SELECT + " WHERE c.id = ?", (case_id,)).fetchone()


def get_case_by_detection(database_path, detection_id):
    with connect(database_path) as connection:
        return connection.execute(CASE_SELECT + " WHERE c.detection_id = ?", (detection_id,)).fetchone()


def list_assignees(database_path):
    with connect(database_path) as connection:
        return connection.execute("SELECT * FROM assignees ORDER BY kind, name").fetchall()


def list_case_events(database_path, case_id):
    with connect(database_path) as connection:
        return connection.execute(
            "SELECT * FROM case_events WHERE case_id = ? ORDER BY created_at DESC, id DESC",
            (case_id,),
        ).fetchall()


def assign_case(database_path, case_id, assignee_id, notes):
    with connect(database_path) as connection:
        case = connection.execute("SELECT status, detection_id FROM cases WHERE id = ?", (case_id,)).fetchone()
        assignee = connection.execute("SELECT name, user_id FROM assignees WHERE id = ?", (assignee_id,)).fetchone()
        if case is None or assignee is None or case["status"] not in {"PENDING", "ASSIGNED"}:
            return False
        connection.execute(
            """UPDATE cases SET assignee_id = ?, notes = ?, status = 'ASSIGNED',
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (assignee_id, notes, case_id),
        )
        connection.execute("UPDATE alerts SET status = 'ASSIGNED' WHERE id = (SELECT alert_id FROM cases WHERE id = ?)", (case_id,))
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'ASSIGNED', ?)",
            (case_id, f"Case assigned to {assignee['name']}."),
        )
        # A reassignment answers any issue staff reported on this case.
        connection.execute(
            "UPDATE case_issues SET status = 'RESOLVED', resolved_at = CURRENT_TIMESTAMP "
            "WHERE case_id = ? AND status = 'OPEN'",
            (case_id,),
        )
        if assignee["user_id"] is not None:
            event_total = connection.execute(
                "SELECT COUNT(*) FROM case_events WHERE case_id = ?", (case_id,)
            ).fetchone()[0]
            create_notification(
                connection, case_id, "TASK_ASSIGNED",
                f"New cleaning task: case #{case_id} was assigned to you.",
                f"case:{case_id}:staff:{assignee_id}:{event_total}", audience="staff",
            )
        create_notification(
            connection, case_id, "ASSIGNMENT",
            f"Case #{case_id} was assigned to {assignee['name']}.",
            f"case:{case_id}:assignment",
        )
        create_notification(
            connection, case_id, "REPORT_ASSIGNED",
            f"Your report #{case['detection_id']} was assigned to {assignee['name']}.",
            f"case:{case_id}:user:assigned", audience="user",
        )
        return True


def change_case_status(database_path, case_id, new_status, notes):
    transitions = {"ASSIGNED": {"CLEANING"}}
    with connect(database_path) as connection:
        case = connection.execute("SELECT status, detection_id FROM cases WHERE id = ?", (case_id,)).fetchone()
        if case is None or new_status not in transitions.get(case["status"], set()):
            return False
        connection.execute(
            "UPDATE cases SET status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP, "
            "started_at = CASE WHEN ? = 'CLEANING' THEN CURRENT_TIMESTAMP ELSE started_at END WHERE id = ?",
            (new_status, notes, new_status, case_id),
        )
        connection.execute("UPDATE alerts SET status = ? WHERE id = (SELECT alert_id FROM cases WHERE id = ?)", (new_status, case_id))
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'STATUS_CHANGED', ?)",
            (case_id, f"Status changed from {case['status']} to {new_status}."),
        )
        create_notification(
            connection, case_id, "STATUS_CHANGED",
            f"Case #{case_id} moved from {case['status']} to {new_status}.",
            f"case:{case_id}:status:{new_status.lower()}",
        )
        create_notification(
            connection, case_id, "REPORT_IN_PROGRESS",
            f"Cleaning has started for your report #{case['detection_id']}.",
            f"case:{case_id}:user:{new_status.lower()}", audience="user",
        )
        return True


def save_after_image(database_path, case_id, after_image_path):
    with connect(database_path) as connection:
        case = connection.execute(
            "SELECT status, after_image_path FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if case is None or case["status"] != "CLEANING" or case["after_image_path"]:
            return False
        connection.execute(
            "UPDATE cases SET after_image_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (after_image_path, case_id),
        )
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'AFTER_IMAGE_UPLOADED', ?)",
            (case_id, "After-cleaning evidence uploaded."),
        )
        return True


def complete_case(database_path, case_id, completion_notes):
    with connect(database_path) as connection:
        case = connection.execute(
            "SELECT c.status, c.detection_id, c.after_image_path, assignees.name AS assignee_name "
            "FROM cases c LEFT JOIN assignees ON assignees.id = c.assignee_id WHERE c.id = ?",
            (case_id,),
        ).fetchone()
        if case is None:
            return "Case was not found."
        if case["status"] != "CLEANING":
            return "Only a case in CLEANING status can be completed."
        if not case["after_image_path"]:
            return "Upload an after-cleaning image before completing this case."
        if not completion_notes:
            return "Add completion notes before completing this case."
        connection.execute(
            """UPDATE cases SET status = 'COMPLETED', completion_notes = ?,
               completed_at = CURRENT_TIMESTAMP,
               resolution_minutes = CAST((julianday(CURRENT_TIMESTAMP) - julianday(created_at)) * 1440 AS INTEGER),
               updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (completion_notes, case_id),
        )
        connection.execute(
            "UPDATE alerts SET status = 'COMPLETED' WHERE id = (SELECT alert_id FROM cases WHERE id = ?)",
            (case_id,),
        )
        completed_by = case["assignee_name"] or "the assigned cleaning team"
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'CASE_COMPLETED', ?)",
            (case_id, f"Case completed by {completed_by}."),
        )
        create_notification(
            connection, case_id, "COMPLETION",
            f"Case #{case_id} was completed by {completed_by}.",
            f"case:{case_id}:completion",
        )
        create_notification(
            connection, case_id, "REPORT_RESOLVED",
            f"Your report #{case['detection_id']} has been resolved. Thank you for helping keep the city clean!",
            f"case:{case_id}:user:resolved", audience="user",
        )
        return None


def delete_case(database_path, case_id):
    """Delete a case with its alert, activity, notifications and source report.

    Returns (detection_row, after_image_path, files_still_shared) info needed to
    clean up image files, or None when the case does not exist.
    """
    with connect(database_path) as connection:
        case = connection.execute(
            "SELECT id, alert_id, detection_id, after_image_path, before_image_path FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if case is None:
            return None
        detection = connection.execute(
            "SELECT * FROM detections WHERE id = ?", (case["detection_id"],)
        ).fetchone()
        connection.execute("DELETE FROM case_events WHERE case_id = ?", (case_id,))
        connection.execute("DELETE FROM case_issues WHERE case_id = ?", (case_id,))
        connection.execute("DELETE FROM notifications WHERE case_id = ?", (case_id,))
        connection.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        connection.execute("DELETE FROM alerts WHERE id = ?", (case["alert_id"],))
        connection.execute("DELETE FROM detections WHERE id = ?", (case["detection_id"],))
        # Only remove image files that no remaining record still points to.
        removable = []
        for relative in (detection["image_path"], detection["result_path"]):
            shared = connection.execute(
                "SELECT 1 FROM detections WHERE image_path = ? OR result_path = ? LIMIT 1", (relative, relative)
            ).fetchone()
            if not shared and relative:
                removable.append(relative)
        after = case["after_image_path"]
        if after and not connection.execute(
            "SELECT 1 FROM cases WHERE after_image_path = ? LIMIT 1", (after,)
        ).fetchone():
            removable.append(after)
        before = case["before_image_path"]
        if before and not connection.execute(
            "SELECT 1 FROM cases WHERE before_image_path = ? LIMIT 1", (before,)
        ).fetchone():
            removable.append(before)
    return removable


def backfill_missing_cases(database_path):
    """Open a case for every valid garbage detection that has none yet."""
    with connect(database_path) as connection:
        connection.execute(
            """UPDATE detections
               SET is_valid = 1,
                   validation_reason = 'Garbage detected.'
               WHERE is_valid = 0 AND COALESCE(detection_count, 0) > 0"""
        )
        ids = [row["id"] for row in connection.execute(
            """SELECT id FROM detections
               WHERE is_valid = 1 AND COALESCE(detection_count, 0) > 0
                 AND id NOT IN (SELECT detection_id FROM cases)
               ORDER BY id"""
        )]
    for detection_id in ids:
        create_alert_and_case(database_path, detection_id)
    return len(ids)


def recompute_risk(database_path, assess):
    """Re-apply the current risk rule to every saved garbage detection (idempotent)."""
    with connect(database_path) as connection:
        rows = connection.execute(
            """SELECT id, detection_count, confidence, box_area_ratio, is_duplicate
               FROM detections WHERE COALESCE(detection_count, 0) > 0"""
        ).fetchall()
        for row in rows:
            level, score, reason = assess(
                row["detection_count"], row["confidence"], row["box_area_ratio"], bool(row["is_duplicate"])
            )
            connection.execute(
                "UPDATE detections SET risk_level = ?, risk_score = ?, risk_reason = ? WHERE id = ?",
                (level, score, reason, row["id"]),
            )
            connection.execute(
                "UPDATE alerts SET risk_level = ?, risk_score = ? WHERE detection_id = ?",
                (level, score, row["id"]),
            )


def detection_has_case(database_path, detection_id):
    with connect(database_path) as connection:
        return connection.execute("SELECT 1 FROM cases WHERE detection_id = ?", (detection_id,)).fetchone() is not None


_DETECTION_JOINS = """ FROM detections
            LEFT JOIN cases c ON c.detection_id = detections.id
            LEFT JOIN alerts a ON a.id = c.alert_id
            LEFT JOIN assignees assignees_history ON assignees_history.id = c.assignee_id
            LEFT JOIN users reporter ON reporter.id = detections.user_id"""


def _detection_filters(search="", source="", status="", risk="", user_id=None):
    filters, params = [], []
    if user_id is not None:
        filters.append("detections.user_id = ?")
        params.append(user_id)
    if search:
        filters.append("(detections.address LIKE ? OR CAST(detections.id AS TEXT) LIKE ? OR reporter.full_name LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    if source in {"upload", "camera"}:
        filters.append("detections.source = ?")
        params.append(source)
    if status in {"PENDING", "ASSIGNED", "CLEANING", "COMPLETED"}:
        filters.append("c.status = ?")
        params.append(status)
    if risk in {"HIGH", "MEDIUM", "LOW"}:
        filters.append("COALESCE(a.risk_level, detections.risk_level) = ?")
        params.append(risk)
    clause = f" WHERE {' AND '.join(filters)}" if filters else ""
    return clause, params


def list_detections(database_path, *, search="", source="", status="", risk="", page=1, per_page=10, user_id=None):
    clause, params = _detection_filters(search, source, status, risk, user_id)
    with connect(database_path) as connection:
        total = connection.execute(f"SELECT COUNT(*){_DETECTION_JOINS}{clause}", params).fetchone()[0]
        rows = connection.execute(
            """SELECT detections.*, c.id AS case_id, c.status AS case_status,
                      c.completed_at, c.resolution_minutes, assignees_history.name AS assignee_name,
                      reporter.full_name AS reporter_name, reporter.username AS reporter_username,
                      COALESCE(a.risk_level, detections.risk_level) AS case_risk_level
               """ + _DETECTION_JOINS + clause + " ORDER BY detections.created_at DESC, detections.id DESC LIMIT ? OFFSET ?",
            [*params, per_page, (page - 1) * per_page],
        ).fetchall()
    return rows, total


def delete_detection(database_path, detection_id):
    record = get_detection(database_path, detection_id)
    if record:
        with connect(database_path) as connection:
            connection.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    return record


def clear_detections(database_path):
    with connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM detections WHERE id NOT IN (SELECT detection_id FROM cases)"
        ).fetchall()
        connection.execute("DELETE FROM detections WHERE id NOT IN (SELECT detection_id FROM cases)")
    return rows


def dashboard_data(database_path):
    with connect(database_path) as connection:
        summary = connection.execute(
            """SELECT COUNT(*) AS total_detections, COALESCE(SUM(detection_count), 0) AS total_objects,
                      AVG(confidence) AS average_confidence,
                      COALESCE(SUM(CASE WHEN date(created_at, 'localtime') = ? THEN 1 ELSE 0 END), 0) AS today_detections,
                      COALESCE(SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END), 0) AS valid_detections,
                      COALESCE(SUM(CASE WHEN is_valid = 0 THEN 1 ELSE 0 END), 0) AS invalid_detections,
                      COALESCE(SUM(CASE WHEN detection_count > 0 THEN 1 ELSE 0 END), 0) AS garbage_detections,
                      COALESCE(SUM(CASE WHEN detection_count = 0 THEN 1 ELSE 0 END), 0) AS no_garbage_detections,
                      (SELECT COUNT(*) FROM alerts) AS total_alerts,
                      (SELECT COUNT(*) FROM cases WHERE status = 'PENDING') AS pending_cases,
                      (SELECT COUNT(*) FROM cases WHERE status = 'ASSIGNED') AS assigned_cases,
                      (SELECT COUNT(*) FROM cases WHERE status = 'CLEANING') AS cleaning_cases,
                      (SELECT COUNT(*) FROM cases WHERE status = 'COMPLETED') AS completed_cases,
                      (SELECT AVG(resolution_minutes) FROM cases WHERE status = 'COMPLETED' AND resolution_minutes IS NOT NULL) AS average_resolution_minutes,
                      (SELECT COUNT(*) FROM alerts WHERE risk_level = 'HIGH') AS high_risk_cases,
                      (SELECT COUNT(*) FROM alerts WHERE risk_level = 'MEDIUM') AS medium_risk_cases,
                      (SELECT COUNT(*) FROM alerts WHERE risk_level = 'LOW') AS low_risk_cases
               FROM detections""",
            (date.today().isoformat(),),
        ).fetchone()
        recent = connection.execute("SELECT * FROM detections ORDER BY created_at DESC, id DESC LIMIT 5").fetchall()
        trend = connection.execute(
            """SELECT date(created_at, 'localtime') AS day, COUNT(*) AS detections, SUM(detection_count) AS objects
               FROM detections GROUP BY day ORDER BY day DESC LIMIT 14"""
        ).fetchall()
        status_distribution = connection.execute(
            "SELECT status, COUNT(*) AS total FROM cases GROUP BY status"
        ).fetchall()
        risk_distribution = connection.execute(
            "SELECT risk_level, COUNT(*) AS total FROM alerts WHERE risk_level IS NOT NULL GROUP BY risk_level"
        ).fetchall()
        activity = connection.execute(
            """SELECT * FROM (
                   SELECT d.created_at AS occurred_at, 'DETECTION' AS activity_type,
                          d.id AS detection_id, NULL AS case_id, d.address, d.risk_level,
                          NULL AS status, 'Detection saved' AS message
                   FROM detections d
                   UNION ALL
                   SELECT e.created_at AS occurred_at, e.event_type AS activity_type,
                          c.detection_id, c.id AS case_id, a.address, a.risk_level,
                          c.status, e.message
                   FROM case_events e
                   JOIN cases c ON c.id = e.case_id
                   JOIN alerts a ON a.id = c.alert_id
               ) ORDER BY occurred_at DESC LIMIT 8"""
        ).fetchall()
    return summary, recent, list(reversed(trend)), status_distribution, risk_distribution, activity


def map_records(database_path):
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT d.id AS detection_id, d.created_at, d.detection_count, d.confidence, d.result_path,
                      d.latitude, d.longitude, d.address, d.source, d.user_id,
                      COALESCE(a.risk_level, d.risk_level) AS risk_level,
                      c.id AS case_id, c.status AS case_status,
                      assignees.name AS assignee_name
               FROM detections d
               LEFT JOIN cases c ON c.detection_id = d.id
               LEFT JOIN alerts a ON a.id = c.alert_id
               LEFT JOIN assignees ON assignees.id = c.assignee_id
               WHERE d.latitude IS NOT NULL AND d.longitude IS NOT NULL
                 AND COALESCE(d.detection_count, 0) > 0
                 AND (c.status IS NULL OR c.status != 'COMPLETED')
               ORDER BY d.created_at DESC, d.id DESC"""
        ).fetchall()


def export_records(database_path, *, search="", source="", status="", risk="", user_id=None):
    clause, params = _detection_filters(search, source, status, risk, user_id)
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT detections.*, c.id AS case_id, c.status AS case_status,
                      assignees_history.name AS assignee_name, c.completed_at, c.resolution_minutes,
                      reporter.full_name AS reporter_name
               """ + _DETECTION_JOINS + clause + " ORDER BY detections.created_at DESC, detections.id DESC",
            params,
        ).fetchall()


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def create_user(database_path, *, username, full_name, password_hash, role="user", email=None, phone=None):
    with connect(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO users (username, email, full_name, phone, password_hash, role)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, email or None, full_name, phone or None, password_hash, role),
        )
        return cursor.lastrowid


def get_user(database_path, user_id):
    if user_id is None:
        return None
    with connect(database_path) as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def find_user_by_login(database_path, identifier):
    with connect(database_path) as connection:
        return connection.execute(
            "SELECT * FROM users WHERE username = ? OR email = ? LIMIT 1", (identifier, identifier)
        ).fetchone()


def username_or_email_taken(database_path, username, email, exclude_id=None):
    with connect(database_path) as connection:
        params = [username, email or "\0"]
        sql = "SELECT username, email FROM users WHERE (username = ? OR email = ?)"
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        row = connection.execute(sql, params).fetchone()
    if row is None:
        return None
    return "username" if row["username"].lower() == (username or "").lower() else "email"


def touch_login(database_path, user_id):
    with connect(database_path) as connection:
        connection.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))


def count_users(database_path, role=None):
    with connect(database_path) as connection:
        if role:
            return connection.execute("SELECT COUNT(*) FROM users WHERE role = ?", (role,)).fetchone()[0]
        return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def list_users(database_path, *, search="", role=None):
    filters, params = [], []
    if role:
        filters.append("u.role = ?")
        params.append(role)
    if search:
        filters.append("(u.full_name LIKE ? OR u.username LIKE ? OR u.email LIKE ?)")
        params.extend([f"%{search}%"] * 3)
    clause = " WHERE " + " AND ".join(filters) if filters else ""
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT u.*, (SELECT COUNT(*) FROM detections d WHERE d.user_id = u.id) AS report_count
               FROM users u""" + clause + " ORDER BY u.role, u.created_at DESC, u.id DESC",
            params,
        ).fetchall()


def update_user_profile(database_path, user_id, full_name, email, phone):
    with connect(database_path) as connection:
        connection.execute(
            "UPDATE users SET full_name = ?, email = ?, phone = ? WHERE id = ?",
            (full_name, email or None, phone or None, user_id),
        )


def update_user_password(database_path, user_id, password_hash):
    with connect(database_path) as connection:
        connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))


def set_user_active(database_path, user_id, active):
    with connect(database_path) as connection:
        connection.execute(
            "UPDATE users SET is_active = ? WHERE id = ? AND role IN ('user', 'staff')", (1 if active else 0, user_id)
        )


def delete_user(database_path, user_id):
    """Remove a citizen or staff account.

    Citizen reports stay in the city record, unowned. A staff member's name stays
    on the cases they handled; only the login link is removed.
    """
    with connect(database_path) as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE id = ? AND role IN ('user', 'staff')", (user_id,)
        ).fetchone()
        if user is None:
            return None
        connection.execute("UPDATE detections SET user_id = NULL WHERE user_id = ?", (user_id,))
        connection.execute("UPDATE assignees SET user_id = NULL WHERE user_id = ?", (user_id,))
        connection.execute("UPDATE case_issues SET staff_id = NULL WHERE staff_id = ?", (user_id,))
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return user


def assign_unowned_detections(database_path, user_id):
    """One-time demo helper: give pre-existing (ownerless) reports to a demo user."""
    with connect(database_path) as connection:
        connection.execute("UPDATE detections SET user_id = ? WHERE user_id IS NULL", (user_id,))


def user_report_stats(database_path):
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT u.id, u.full_name, u.username, u.email, u.is_active,
                      COUNT(d.id) AS total,
                      COALESCE(SUM(CASE WHEN c.status = 'PENDING' THEN 1 ELSE 0 END), 0) AS pending,
                      COALESCE(SUM(CASE WHEN c.status IN ('ASSIGNED', 'CLEANING') THEN 1 ELSE 0 END), 0) AS in_progress,
                      COALESCE(SUM(CASE WHEN c.status = 'COMPLETED' THEN 1 ELSE 0 END), 0) AS resolved,
                      MAX(d.created_at) AS last_report
               FROM users u
               LEFT JOIN detections d ON d.user_id = u.id
               LEFT JOIN cases c ON c.detection_id = d.id
               WHERE u.role = 'user'
               GROUP BY u.id
               ORDER BY total DESC, u.full_name""",
        ).fetchall()


def user_summary(database_path, user_id):
    """Dashboard counters for one citizen's own reports."""
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT COUNT(d.id) AS total,
                      COALESCE(SUM(CASE WHEN c.status = 'PENDING' THEN 1 ELSE 0 END), 0) AS pending,
                      COALESCE(SUM(CASE WHEN c.status IN ('ASSIGNED', 'CLEANING') THEN 1 ELSE 0 END), 0) AS in_progress,
                      COALESCE(SUM(CASE WHEN c.status = 'COMPLETED' THEN 1 ELSE 0 END), 0) AS resolved,
                      COUNT(DISTINCT CASE WHEN d.latitude IS NOT NULL
                            THEN ROUND(d.latitude, 3) || ',' || ROUND(d.longitude, 3) END) AS locations
               FROM detections d LEFT JOIN cases c ON c.detection_id = d.id
               WHERE d.user_id = ?""",
            (user_id,),
        ).fetchone()


def recent_detections(database_path, *, limit=5, user_id=None):
    """Latest reports (with case status and reporter) for dashboards."""
    clause, params = _detection_filters(user_id=user_id)
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT detections.*, c.id AS case_id, c.status AS case_status,
                      reporter.full_name AS reporter_name,
                      COALESCE(a.risk_level, detections.risk_level) AS case_risk_level
               """ + _DETECTION_JOINS + clause + " ORDER BY detections.created_at DESC, detections.id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()


# ---------------------------------------------------------------------------
# Settings (admin office location, etc.)
# ---------------------------------------------------------------------------

def get_setting(database_path, key, default=None):
    with connect(database_path) as connection:
        row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_settings(database_path, values):
    with connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(key, None if value is None else str(value)) for key, value in values.items()],
        )


# ---------------------------------------------------------------------------
# Staff / municipal workers
# ---------------------------------------------------------------------------

def create_staff(database_path, *, username, full_name, password_hash, email=None, phone=None):
    """Create a staff login and the assignee row that lets admins assign cases to it."""
    with connect(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO users (username, email, full_name, phone, password_hash, role)
               VALUES (?, ?, ?, ?, ?, 'staff')""",
            (username, email or None, full_name, phone or None, password_hash),
        )
        user_id = cursor.lastrowid
        name = full_name
        if connection.execute("SELECT 1 FROM assignees WHERE name = ?", (name,)).fetchone():
            name = f"{full_name} ({username})"
        connection.execute(
            "INSERT INTO assignees (name, kind, is_demo, user_id) VALUES (?, 'WORKER', 0, ?)",
            (name, user_id),
        )
        return user_id


def list_staff_cases(database_path, staff_id):
    with connect(database_path) as connection:
        return connection.execute(
            CASE_SELECT + " WHERE assignees.user_id = ? ORDER BY "
            "CASE c.status WHEN 'CLEANING' THEN 1 WHEN 'ASSIGNED' THEN 2 ELSE 3 END, "
            "CASE a.risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END, c.updated_at DESC",
            (staff_id,),
        ).fetchall()


def get_staff_case(database_path, case_id, staff_id):
    """The case only if it is currently assigned to this staff member."""
    with connect(database_path) as connection:
        return connection.execute(
            CASE_SELECT + " WHERE c.id = ? AND assignees.user_id = ?", (case_id, staff_id)
        ).fetchone()


def staff_summary(database_path, staff_id):
    with connect(database_path) as connection:
        row = connection.execute(
            """SELECT COALESCE(SUM(CASE WHEN c.status = 'ASSIGNED' THEN 1 ELSE 0 END), 0) AS assigned,
                      COALESCE(SUM(CASE WHEN c.status = 'CLEANING' THEN 1 ELSE 0 END), 0) AS in_progress,
                      COALESCE(SUM(CASE WHEN c.status = 'COMPLETED' THEN 1 ELSE 0 END), 0) AS cleaned
               FROM cases c JOIN assignees s ON s.id = c.assignee_id WHERE s.user_id = ?""",
            (staff_id,),
        ).fetchone()
        issues = connection.execute(
            "SELECT COUNT(*) FROM case_issues WHERE staff_id = ? AND status = 'OPEN'", (staff_id,)
        ).fetchone()[0]
    return {"assigned": row["assigned"], "in_progress": row["in_progress"],
            "cleaned": row["cleaned"], "open_issues": issues}


def save_before_image(database_path, case_id, before_image_path):
    """Store before-cleaning evidence. Returns (saved, previous_path_to_delete)."""
    with connect(database_path) as connection:
        case = connection.execute(
            "SELECT status, before_image_path FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        if case is None or case["status"] != "CLEANING":
            return False, None
        connection.execute(
            "UPDATE cases SET before_image_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (before_image_path, case_id),
        )
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'BEFORE_IMAGE_UPLOADED', ?)",
            (case_id, "Before-cleaning evidence uploaded by staff."),
        )
        return True, case["before_image_path"]


def report_case_issue(database_path, case_id, staff_id, reason, details):
    """Staff cannot clean this location: hand the case back to the admin.

    The case returns to PENDING and is unassigned so an admin can reassign it.
    Returns the new issue id, or None when the case is not this staff member's.
    """
    with connect(database_path) as connection:
        case = connection.execute(
            """SELECT c.status, c.detection_id, s.name AS staff_name
               FROM cases c JOIN assignees s ON s.id = c.assignee_id
               WHERE c.id = ? AND s.user_id = ?""",
            (case_id, staff_id),
        ).fetchone()
        if case is None or case["status"] not in ("ASSIGNED", "CLEANING"):
            return None
        cursor = connection.execute(
            "INSERT INTO case_issues (case_id, staff_id, reason, details) VALUES (?, ?, ?, ?)",
            (case_id, staff_id, reason, details or None),
        )
        issue_id = cursor.lastrowid
        connection.execute(
            "UPDATE cases SET status = 'PENDING', assignee_id = NULL, started_at = NULL, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (case_id,),
        )
        connection.execute(
            "UPDATE alerts SET status = 'PENDING' WHERE id = (SELECT alert_id FROM cases WHERE id = ?)",
            (case_id,),
        )
        detail_text = f" {details}" if details else ""
        connection.execute(
            "INSERT INTO case_events (case_id, event_type, message) VALUES (?, 'ISSUE_REPORTED', ?)",
            (case_id, f"{case['staff_name']} reported an issue: {reason}.{detail_text}"),
        )
        create_notification(
            connection, case_id, "ISSUE_REPORTED",
            f"Case #{case_id} could not be cleaned ({reason}). It needs to be reassigned.",
            f"case:{case_id}:issue:{issue_id}",
        )
        create_notification(
            connection, case_id, "REPORT_DELAYED",
            f"Your report #{case['detection_id']} needs another visit. The city team will reschedule it.",
            f"case:{case_id}:user:issue:{issue_id}", audience="user",
        )
        return issue_id


def list_case_issues(database_path, case_id):
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT i.*, u.full_name AS staff_name FROM case_issues i
               LEFT JOIN users u ON u.id = i.staff_id
               WHERE i.case_id = ? ORDER BY i.created_at DESC, i.id DESC""",
            (case_id,),
        ).fetchall()


def list_staff_issues(database_path, staff_id):
    with connect(database_path) as connection:
        return connection.execute(
            """SELECT i.*, a.address FROM case_issues i
               JOIN cases c ON c.id = i.case_id JOIN alerts a ON a.id = c.alert_id
               WHERE i.staff_id = ? ORDER BY i.created_at DESC, i.id DESC LIMIT 20""",
            (staff_id,),
        ).fetchall()


def open_issue_case_ids(database_path):
    with connect(database_path) as connection:
        return {r[0] for r in connection.execute("SELECT DISTINCT case_id FROM case_issues WHERE status = 'OPEN'")}


def set_admin_active(database_path, user_id):
    """Used by ``python run.py --reset-admin`` to make sure the admin can sign in."""
    with connect(database_path) as connection:
        connection.execute("UPDATE users SET is_active = 1 WHERE id = ? AND role = 'admin'", (user_id,))
