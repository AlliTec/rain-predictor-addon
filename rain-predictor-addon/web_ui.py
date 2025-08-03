# /rain-predictor-addon/web_ui.py

import os
from flask import Flask, render_template, jsonify
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def get_ha_state(entity_id):
    token = os.environ.get('SUPERVISOR_TOKEN')
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://supervisor/core/api/states/{entity_id}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json().get('state', None)
    except requests.exceptions.RequestException:
        return None

# --- NEW SIMPLIFIED INDEX ROUTE ---
@app.route('/')
def index():
    """Render the main web UI page without pre-fetching data."""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """Provide the latest prediction data as JSON."""
    data = {
        'time_to_rain': get_ha_state('input_number.rain_arrival_minutes'),
        'distance': get_ha_state('input_number.rain_prediction_distance'),
        'speed': get_ha_state('input_number.rain_prediction_speed'),
        'direction': get_ha_state('input_number.rain_cell_direction'),
        'bearing': get_ha_state('input_number.bearing_to_rain_cell'),
        # --- NEW: We now fetch lat/lon here for the frontend ---
        'latitude': get_ha_state('input_number.rain_prediction_latitude'),
        'longitude': get_ha_state('input_number.rain_prediction_longitude')
    }
    return jsonify(data)

@app.route('/health')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)