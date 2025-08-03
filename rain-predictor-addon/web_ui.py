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
    # We will pass the lat/lon to the template for the map's initial view
    lat = get_ha_state('input_number.rain_prediction_latitude') # Assuming you have this entity
    lon = get_ha_state('input_number.rain_prediction_longitude') # Assuming you have this entity
    
    # Fallback to a default if entities don't exist yet
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        lat = -24.98
        lon = 151.86

    return render_template('index.html', latitude=lat, longitude=lon)

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