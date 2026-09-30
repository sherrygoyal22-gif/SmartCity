from __future__ import annotations


def optional_coordinates(latitude, longitude):
    """Return validated optional browser coordinates; never invent a location."""
    if latitude in (None, "") or longitude in (None, ""):
        return None, None
    try:
        latitude, longitude = float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Location coordinates are invalid.") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Location coordinates are outside their valid range.")
    return latitude, longitude


def hotspot_clusters(records, tolerance_degrees, minimum_detections):
    """Group nearby real garbage detections using the Phase 2 coordinate approach.

    Every aggregate below (risk counts, report ids, case ids, sources, area
    label) is derived only from the real rows passed in — nothing here
    invents a report, a coordinate, or a case that does not exist in the
    database.
    """
    clusters = []
    risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    for record in records:
        if not record["detection_count"]:
            continue
        latitude, longitude = record["latitude"], record["longitude"]
        if latitude is None or longitude is None:
            continue
        cluster = next(
            (
                item for item in clusters
                if abs(latitude - item["latitude"]) <= tolerance_degrees
                and abs(longitude - item["longitude"]) <= tolerance_degrees
            ),
            None,
        )
        if cluster is None:
            cluster = {
                "latitude": latitude,
                "longitude": longitude,
                "records": [],
                "active_cases": 0,
                "completed_cases": 0,
                "highest_risk": None,
                "latest_detection": record["created_at"],
                "latest_detection_id": record["detection_id"],
                "latest_result_path": record["result_path"],
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "report_ids": [],
                "case_ids": [],
                "sources": [],
                "address_votes": {},
            }
            clusters.append(cluster)
        cluster["records"].append(record)
        count = len(cluster["records"])
        cluster["latitude"] += (latitude - cluster["latitude"]) / count
        cluster["longitude"] += (longitude - cluster["longitude"]) / count
        if record["case_status"] in {"PENDING", "ASSIGNED", "CLEANING"}:
            cluster["active_cases"] += 1
        elif record["case_status"] == "COMPLETED":
            cluster["completed_cases"] += 1
        if risk_order.get(record["risk_level"], 0) > risk_order.get(cluster["highest_risk"], 0):
            cluster["highest_risk"] = record["risk_level"]
        if record["created_at"] > cluster["latest_detection"]:
            cluster["latest_detection"] = record["created_at"]
            cluster["latest_detection_id"] = record["detection_id"]
            cluster["latest_result_path"] = record["result_path"]

        risk_key = {"HIGH": "high_count", "MEDIUM": "medium_count", "LOW": "low_count"}.get(record["risk_level"])
        if risk_key:
            cluster[risk_key] += 1

        cluster["report_ids"].append(record["detection_id"])
        if record["case_id"] is not None and record["case_id"] not in cluster["case_ids"]:
            cluster["case_ids"].append(record["case_id"])
        if record["source"] and record["source"] not in cluster["sources"]:
            cluster["sources"].append(record["source"])
        if record["address"]:
            cluster["address_votes"][record["address"]] = cluster["address_votes"].get(record["address"], 0) + 1

    hotspots = []
    for cluster in clusters:
        if len(cluster["records"]) >= minimum_detections:
            cluster["detection_count"] = len(cluster["records"])
            cluster.pop("records")
            # Most common real, user-supplied address in the cluster (if any)
            # is used as the human-readable hotspot name — never invented.
            address_votes = cluster.pop("address_votes")
            cluster["area_label"] = (
                max(address_votes, key=address_votes.get) if address_votes else None
            )
            # Most recent reports first, so popups show the freshest activity.
            cluster["report_ids"] = sorted(cluster["report_ids"], reverse=True)
            cluster["case_ids"] = sorted(cluster["case_ids"], reverse=True)
            hotspots.append(cluster)
    return sorted(hotspots, key=lambda item: item["detection_count"], reverse=True)
