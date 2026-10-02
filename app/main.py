"""Compatibility alias for ``python run.py``.

The application now lives in ``app/app.py`` only. Render (``wsgi.py``) and the
local launcher (``run.py``) both load that one file, so a fix can never reach the
hosted site but miss your PC, or the other way round.
"""

from app.app import app, create_app  # noqa: F401
