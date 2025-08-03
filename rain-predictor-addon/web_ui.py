# /rain-predictor-addon/web_ui.py

import os
import json
from flask import Flask, render_template, jsonify
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

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
    """Helper function to gather all data points from Home Assistant."""
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
# In web_ui.py
@app.route('/')
def index():
    """Render the main web UI page."""
    # Get the unique Ingress path from the environment
    ingress_entry = os.environ.get('INGRESS_ENTRY', '')

    # THIS IS THE KEY: Construct the full, absolute URL for the API endpoint
    api_url = f"{ingress_entry}/api/data"

    # Get initial values for the map so it loads correctly
    # Use defaults if the API call fails initially to prevent errors
    lat = get_ha_state('input_number.rain_prediction_latitude', -24.98)
    lon = get_ha_state('input_number.rain_prediction_longitude', 151.86)
    
    # Pass all variables to the template
    return render_template('index.html', latitude=lat, longitude=lon, api_url=api_url)

@app.route('/api/data')
def api_data():
    """API endpoint to provide subsequent data updates."""
    return jsonify(get_all_data())

@app.route('/health')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)