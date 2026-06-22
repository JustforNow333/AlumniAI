"""Shared Flask route helpers.

Every route blueprint duplicates the same try/except → jsonify({"error": ...})
pattern. This module provides a decorator and helper to reduce that boilerplate.
"""

import functools

from flask import jsonify, request


def handle_store_errors(error_base_class):
    """Decorator that catches *error_base_class* and returns a JSON error response.

    Usage::

        @bp.get("/<item_id>")
        @handle_store_errors(InsightStoreError)
        def get_item(item_id):
            return jsonify(get_insight(item_id))
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except error_base_class as exc:
                return jsonify({"error": str(exc)}), exc.status_code
        return wrapper
    return decorator


def get_json_payload():
    """Return the parsed JSON body, or an error tuple if invalid."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Request body must be a JSON object."}), 400)
    return payload, None
