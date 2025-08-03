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
def index():
    """Render the main web UI page, embedding initial data."""
    ingress_entry = os.environ.get('INGRESS_ENTRY', '')
    initial_data = get_all_data()

    # Pass the Ingress path and all data (as a JSON string) to the template.
    return render_template('index.html', 
                           ingress_entry=ingress_entry, 
                           initial_data_json=json.dumps(initial_data))

@app.route('/api/data')
def api_data():
    """API endpoint to provide subsequent data updates."""
    return jsonify(get_all_data())

@app.route('/health')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099, debug=False)