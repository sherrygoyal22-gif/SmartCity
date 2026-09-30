"""Authentication and role-based access control for SmartCity.

Three roles exist:
  * ``admin`` - city administrators (dashboard, all reports, cases, users, ...)
  * ``staff`` - municipal workers who clean the cases assigned to them
  * ``user``  - citizens who report garbage and follow their own reports

Sessions are signed cookies (Flask ``session``); passwords are stored only as
salted hashes.
"""
from __future__ import annotations

import re
import time
from functools import wraps
from urllib.parse import urlparse

from flask import abort, g, redirect, request, session, url_for

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_STAFF = "staff"

ROLE_LABELS = {ROLE_ADMIN: "Administrator", ROLE_STAFF: "Staff", ROLE_USER: "Citizen"}
LOGIN_ENDPOINTS = {ROLE_ADMIN: "admin_login", ROLE_STAFF: "staff_login", ROLE_USER: "login"}

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 6

# --- simple in-memory login throttle (per client + account name) -------------
_FAILURES: dict[str, list[float]] = {}
MAX_ATTEMPTS = 5
WINDOW_SECONDS = 300


def _throttle_key(identifier: str) -> str:
    return f"{request.remote_addr}|{(identifier or '').strip().lower()}"


def login_blocked(identifier: str) -> bool:
    key = _throttle_key(identifier)
    now = time.time()
    recent = [t for t in _FAILURES.get(key, []) if now - t < WINDOW_SECONDS]
    _FAILURES[key] = recent
    return len(recent) >= MAX_ATTEMPTS


def record_failure(identifier: str) -> None:
    _FAILURES.setdefault(_throttle_key(identifier), []).append(time.time())


def clear_failures(identifier: str) -> None:
    _FAILURES.pop(_throttle_key(identifier), None)


# --- helpers -----------------------------------------------------------------

def is_safe_next(target: str | None) -> bool:
    """Only allow redirects back into this site."""
    if not target:
        return False
    parsed = urlparse(target)
    return not parsed.scheme and not parsed.netloc and target.startswith("/") and not target.startswith("//")


def home_for(user) -> str:
    role = user["role"] if user is not None else ROLE_USER
    if role == ROLE_ADMIN:
        return url_for("admin_dashboard")
    if role == ROLE_STAFF:
        return url_for("staff_dashboard")
    return url_for("dashboard")


def area_for_path(path: str) -> str:
    """Which role's area a URL belongs to (used to validate ?next= redirects)."""
    if path.startswith("/admin"):
        return ROLE_ADMIN
    if path.startswith("/staff"):
        return ROLE_STAFF
    return ROLE_USER


def initials(name: str | None) -> str:
    parts = [p for p in (name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def login_session(user, remember: bool) -> None:
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = bool(remember)


# --- decorators --------------------------------------------------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith("/api/"):
                return {"ok": False, "error": "Please sign in to continue."}, 401
            next_url = request.full_path.rstrip("?") if request.method == "GET" else None
            login_endpoint = LOGIN_ENDPOINTS[area_for_path(request.path)]
            return redirect(url_for(login_endpoint, next=next_url))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if g.user["role"] != ROLE_ADMIN:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def user_required(view):
    """Citizen-only pages; admins and staff are sent to their own dashboard instead."""
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if g.user["role"] != ROLE_USER:
            return redirect(home_for(g.user))
        return view(*args, **kwargs)
    return wrapped


def staff_required(view):
    """Staff-only pages (the municipal worker dashboard and its task actions)."""
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if g.user["role"] != ROLE_STAFF:
            if g.user["role"] == ROLE_ADMIN and request.method == "GET":
                return redirect(home_for(g.user))
            abort(403)
        return view(*args, **kwargs)
    return wrapped
