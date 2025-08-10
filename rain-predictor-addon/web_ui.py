#!/usr/bin/env python3
# /rain-predictor-addon/web_ui.py

import os
import json
from flask import Flask, render_template, jsonify, request
import requests
import logging
import io
import math
import numpy as np
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Radar sampling/analysis settings for manual selection
IMAGE_SIZE = 256
ZOOM = 8
COLOR_SCHEME = 3            # matches your UI default, adjust if needed
LAT_RANGE_DEG = 1.80        # must match predictor’s analysis_settings.lat_range_deg
LON_RANGE_DEG = 1.99        # must match predictor’s analysis_settings.lon_range_deg
RAIN_THRESHOLD = 60         # grayscale threshold (0–255) for "rain" pixels
MAX_FRAMES = 5              # how many past frames to use (recent frames are more relevant)

class HomeAssistantAPI:
    def __init__(self):
        self.base_url = "http://supervisor/core/api"
    
    def call_service(self, service, entity_id, value):
        token = os.environ.get('SUPERVISOR_TOKEN')
        if not token:
            logging.error("SUPERVISOR_TOKEN not found!")
            return False
        headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
        domain, service_name = service.split('/', 1)
        url = f"{self.base_url}/services/{domain}/{service_name}"
        data = { "entity_id": entity_id, "value": value }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code >= 400:
                logging.error(f"Error calling service {service} for {entity_id}: {response.status_code} {response.text}")
                return False
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Exception calling service {service} for {entity_id}: {e}")
            return False

ha_api = HomeAssistantAPI()

def get_ha_state(entity_id, default=None):
    token = os.environ.get('SUPERVISOR_TOKEN')
    if not token: 
        return default
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://supervisor/core/api/states/{entity_id}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json().get('state', default)
    except requests.exceptions.RequestException:
        return default
        
def get_all_data():
    return {
        'time_to_rain': get_ha_state('input_number.rain_arrival_minutes', '--'),
        'distance': get_ha_state('input_number.rain_prediction_distance', '--'),
        'speed': get_ha_state('input_number.rain_prediction_speed', '--'),
        'direction': get_ha_state('input_number.rain_cell_direction', 'N/A'),
        'bearing': get_ha_state('input_number.bearing_to_rain_cell', 'N/A'),
        'latitude': get_ha_state('input_number.rain_prediction_latitude', -24.98),
        'longitude': get_ha_state('input_number.rain_prediction_longitude', 151.86)
    }

@app.route('/')
def index():
    lat = get_ha_state('input_number.rain_prediction_latitude', -24.98)
    lon = get_ha_state('input_number.rain_prediction_longitude', 151.86)
    return render_template('index.html', latitude=lat, longitude=lon)

@app.route('/api/data')
def api_data():
    return jsonify(get_all_data())

@app.route('/api/set_location', methods=['POST'])
def set_location():
    logging.info("set_location endpoint called")
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        logging.error(f"Invalid data received: {data}")
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    lat = data['latitude']
    lon = data['longitude']

    logging.info(f"Received new location: Lat={lat}, Lon={lon}")

    lat_success = ha_api.call_service('input_number/set_value', 'input_number.rain_prediction_latitude', lat)
    lon_success = ha_api.call_service('input_number/set_value', 'input_number.rain_prediction_longitude', lon)

    if lat_success and lon_success:
        logging.info("Successfully updated Home Assistant entities")
        return jsonify({"status": "success", "message": "Location updated", "latitude": lat, "longitude": lon})
    else:
        logging.error("Failed to update Home Assistant entities")
        return jsonify({"status": "error", "message": "Failed to update Home Assistant entities"}), 500

@app.route('/api/update_config', methods=['POST'])
def update_config():
    logging.info("update_config endpoint called")
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    lat = data['latitude']
    lon = data['longitude']

    try:
        config_path = '/data/options.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        config['latitude'] = lat
        config['longitude'] = lon

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"Updated addon config with new location: Lat={lat}, Lon={lon}")
        return jsonify({"status": "success", "message": "Config updated"})

    except Exception as e:
        logging.error(f"Failed to update config: {e}")
        return jsonify({"status": "error", "message": "Failed to update config"}), 500

@app.route('/api/manual_selection', methods=['POST'])
def manual_selection():
    """Handle manual rain cell selection using prevailing motion estimation."""
    logging.info("manual_selection endpoint called")
    data = request.get_json()

    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    lat = float(data['latitude'])
    lng = float(data['longitude'])

    # Try to estimate prevailing motion from recent frames
    result = _estimate_prevailing_motion(lat, lng)

    if result is None:
        # Fallback if we couldn't estimate motion
        speed_kph = 15.0
        direction_deg = 90.0
        logging.warning("Prevailing motion estimation failed; using fallback values.")
    else:
        speed_kph, direction_deg = result

    return jsonify({
        "center": {"lat": lat, "lng": lng},
        "speed": round(speed_kph, 1),
        "direction": round(direction_deg, 1),
        "intensity": None
    })

@app.route('/health')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)