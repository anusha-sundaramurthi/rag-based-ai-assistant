"""
Widget store — manages widget configs in data/widgets.json.
Each widget has its own Qdrant collection, API key, and config.
Master API key is hardcoded in .env for admin access.
"""

import json
import os
import uuid
from datetime import datetime

WIDGETS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "widgets.json"
)

MASTER_KEY = os.getenv("MASTER_API_KEY", "change-this-master-key")


def _load() -> dict:
    os.makedirs(os.path.dirname(WIDGETS_FILE), exist_ok=True)
    if not os.path.exists(WIDGETS_FILE):
        return {}
    with open(WIDGETS_FILE, "r") as f:
        return json.load(f)


def _save(data: dict):
    os.makedirs(os.path.dirname(WIDGETS_FILE), exist_ok=True)
    with open(WIDGETS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def verify_master_key(key: str) -> bool:
    return key == MASTER_KEY


def create_widget(name: str, config: dict) -> dict:
    widgets   = _load()
    widget_id = "wgt_" + uuid.uuid4().hex[:10]
    api_key   = "ak_"  + uuid.uuid4().hex[:16]
    widget    = {
        "widget_id":  widget_id,
        "name":       name,
        "api_key":    api_key,
        "collection": widget_id,   # own Qdrant collection
        "config":     config,
        "pdfs":       [],
        "created_at": datetime.utcnow().isoformat(),
    }
    widgets[widget_id] = widget
    _save(widgets)
    return widget


def get_widget(widget_id: str) -> dict | None:
    return _load().get(widget_id)


def get_all_widgets() -> list[dict]:
    return list(_load().values())


def get_widget_by_api_key(api_key: str) -> dict | None:
    for w in _load().values():
        if w.get("api_key") == api_key:
            return w
    return None


def verify_widget_key(widget_id: str, api_key: str) -> dict | None:
    """Verify widget exists AND api_key matches."""
    w = _load().get(widget_id)
    if w and w.get("api_key") == api_key:
        return w
    return None


def add_pdf(widget_id: str, filename: str):
    widgets = _load()
    if widget_id not in widgets:
        raise ValueError(f"Widget {widget_id} not found")
    if filename not in widgets[widget_id]["pdfs"]:
        widgets[widget_id]["pdfs"].append(filename)
    _save(widgets)


def delete_widget(widget_id: str):
    widgets = _load()
    if widget_id in widgets:
        del widgets[widget_id]
        _save(widgets)
