#!/usr/bin/env python3
# web_ui.py
"""
Helper utilities for the Rain Predictor add-on UI.

This module no longer starts a Flask server or defines routes.
All HTTP endpoints and the web server are provided by Rain_Predictor.py.

You can optionally import HomeAssistantAPI or get_ha_state from here
if you want to reuse them in Rain_Predictor.py or other modules.
"""

import os
import logging
import requests

logging.basicConfig(level=logging.INFO)


class HomeAssistantAPI:
    """
    Minimal HA service client. Use from Rain_Predictor.py if desired.
    In most cases, Rain_Predictor.py already has its own HA client.
    """

    def __init__(self, base_url: str = "http://supervisor/core/api"):
        self.base_url = base_url
        self.token = os.environ.get("SUPERVISOR_TOKEN")

    def _headers(self):
        if not self.token:
            return None
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def call_service(self, service: str, entity_id: str, value):
        """
        Call a Home Assistant service, e.g. "input_number/set_value".
        Returns True on success, False on error.
        """
        headers = self._headers()
        if not headers:
            logging.warning("SUPERVISOR_TOKEN not found; cannot call HA service.")
            return False

        try:
            domain, service_name = service.split("/", 1)
        except ValueError:
            logging.error(f"Invalid service format: {service}")
            return False

        url = f"{self.base_url}/services/{domain}/{service_name}"
        data = {"entity_id": entity_id, "value": value}

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            if resp.status_code >= 400:
                logging.error(
                    "Error calling service %s for %s: %s %s",
                    service,
                    entity_id,
                    resp.status_code,
                    resp.text,
                )
                return False
            return True
        except requests.RequestException as e:
            logging.error("Exception calling service %s for %s: %s", service, entity_id, e)
            return False


def get_ha_state(entity_id: str, default=None):
    """
    Retrieve state of a Home Assistant entity. Returns 'default' on error.
    """
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return default

    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://supervisor/core/api/states/{entity_id}"

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.json().get("state", default)
    except requests.RequestException:
        return default