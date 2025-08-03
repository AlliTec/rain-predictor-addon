#!/usr/bin/env python3
"""
Web UI for Rain Predictor Addon Configuration
"""

import json
import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class ConfigManager:
    """Manage addon configuration"""
    
    def __init__(self):
        self.config_path = '/data/options.json'
        self.config = self.load_config()
    
    def load_config(self):
        """Load current configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return self.get_default_config()
        except json.JSONDecodeError:
            logging.error("Invalid JSON in config file")
            return self.get_default_config()
    
    def save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.config = config
            return True
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            return False
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            "latitude": -24.981262201681634,
            "longitude": 151.86545522467864,
            "run_interval_minutes": 3,
            "api_url": "https://api.rainviewer.com/public/weather-maps.json",
            "entities": {
                "time": "input_number.rain_arrival_minutes",
                "distance": "input_number.rain_prediction_distance",
                "speed": "input_number.rain_prediction_speed",
                "direction": "input_number.rain_cell_direction",
                "bearing": "input_number.bearing_to_rain_cell"
            },
            "defaults": {
                "no_rain_value": 999,
                "no_direction_value": -1,
                "no_bearing_value": -1
            },
            "image_settings": {
                "size": 256,
                "zoom": 8,
                "color_scheme": 3,
                "options": "0_0"
            },
            "analysis_settings": {
                "rain_threshold": 75,
                "lat_range_deg": 1.80,
                "lon_range_deg": 1.99,
                "arrival_angle_threshold_deg": 45
            },
            "tracking_settings": {
                "max_tracking_distance_km": 30,
                "min_track_length": 2
            },
            "debug": {
                "log_level": "INFO",
                "save_images": False
            }
        }

config_manager = ConfigManager()

@app.route('/')
def index():
    """Main configuration page"""
    return render_template('index.html', config=config_manager.config)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify(config_manager.config)

@app.route('/api/config', methods=['POST'])
def save_config():
    """Save configuration"""
    try:
        new_config = request.json
        if config_manager.save_config(new_config):
            return jsonify({"success": True, "message": "Configuration saved successfully"})
        else:
            return jsonify({"success": False, "message": "Error saving configuration"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/validate', methods=['POST'])
def validate_config():
    """Validate configuration"""
    try:
        config = request.json
        errors = []
        
        # Validate latitude/longitude
        if not (-90 <= config.get('latitude', 0) <= 90):
            errors.append("Latitude must be between -90 and 90")
        if not (-180 <= config.get('longitude', 0) <= 180):
            errors.append("Longitude must be between -180 and 180")
        
        # Validate required entities
        if not config.get('entities', {}).get('time'):
            errors.append("Time entity is required")
        
        # Validate ranges
        if config.get('run_interval_minutes', 0) < 1:
            errors.append("Run interval must be at least 1 minute")
        
        if config.get('analysis_settings', {}).get('rain_threshold', 0) < 1:
            errors.append("Rain threshold must be at least 1")
        
        return jsonify({"valid": len(errors) == 0, "errors": errors})
    except Exception as e:
        return jsonify({"valid": False, "errors": [str(e)]})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the web server
    app.run(host='0.0.0.0', port=8099, debug=False)