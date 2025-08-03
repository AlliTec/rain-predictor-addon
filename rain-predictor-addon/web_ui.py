# /rain-predictor-addon/web_ui.py

import os
from flask import Flask, render_template, jsonify, send_from_directory
import requests
import logging

app = Flask(__name__)

# Basic logging
logging.basicConfig(level=logging.INFO)

# --- Helper Function to Get HA State ---
def get_ha_state(entity_id):
    """Gets the state of a Home Assistant entity."""
    token = os.environ.get('SUPERVISOR_TOKEN')
    if not token:
        logging.error("SUPERVISOR_TOKEN not found!")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    url = f"http://supervisor/core/api/states/{entity_id}"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json().get('state', None)
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not get state for {entity_id}: {e}")
        return None

# --- Main Route to Serve the Web Page ---
@app.route('/')
def index():
    """Render the main web UI page."""
    # Get the unique Ingress path from the environment
    ingress_entry = os.environ.get('INGRESS_ENTRY', '')

    # Construct the full, absolute URL for the API endpoint
    api_url = f"{ingress_entry}/api/data"

    # For the map's initial view
    lat = get_ha_state('input_number.rain_prediction_latitude')
    lon = get_ha_state('input_number.rain_prediction_longitude')
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        lat = -24.98
        lon = 151.86

    # Pass all variables to the template
    return render_template('index.html', latitude=lat, longitude=lon, api_url=api_url)

# --- API Route to Provide Data to the Frontend ---
@app.route('/api/data')
def get_data():
    """Provide the latest prediction data as JSON."""
    data = {
        'time_to_rain': get_ha_state('input_number.rain_arrival_minutes'),
        'distance': get_ha_state('input_number.rain_prediction_distance'),
        'speed': get_ha_state('input_number.rain_prediction_speed'),
        'direction': get_ha_state('input_number.rain_cell_direction'),
        'bearing': get_ha_state('input_number.bearing_to_rain_cell')
    }
    return jsonify(data)

# --- Health Check Route (from your Dockerfile) ---
@app.route('/health')
def health_check():
    """Health check endpoint."""
    return "OK", 200
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)