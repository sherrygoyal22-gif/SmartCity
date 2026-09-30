from __future__ import annotations

import csv
import io


CSV_FIELDS = ["id", "created_at", "reporter_name", "detection_count", "confidence", "box_area_ratio", "is_valid", "validation_reason", "is_duplicate", "duplicate_reason", "risk_level", "risk_score", "risk_reason", "case_id", "case_status", "assignee_name", "completed_at", "resolution_minutes", "image_path", "result_path", "latitude", "longitude", "address", "source"]


def _safe_cell(value):
    """Stop spreadsheet apps from running user-typed text as a formula."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def detections_csv(records) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow({field: _safe_cell(record[field]) for field in CSV_FIELDS})
    return stream.getvalue()
