#!/usr/bin/env python3
"""
Rain Predictor Home Assistant Addon
Converted from AppDaemon app to standalone addon with web interface
"""

import json
import time
import logging
import os
import sys
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from PIL import Image
import io
from scipy.ndimage import label
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import threading
import signal
from flask import Flask, jsonify, request, send_file, render_template_string
from flask_cors import CORS

class AddonConfig:
    """Load and manage addon configuration"""
    
    def __init__(self):
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from addon options"""
        try:
            # Try addon options first
            if os.path.exists('/data/options.json'):
                with open('/data/options.json', 'r') as f:
                    return json.load(f)
            # Fallback to config.json for development
            elif os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    'latitude': -27.4698,
                    'longitude': 153.0251,
                    'run_interval_minutes': 3,
                    'api_url': 'https://api.rainviewer.com/public/weather-maps.json',
                    'entities': {
                        'time': 'input_number.rain_time',
                        'distance': 'input_number.rain_distance',
                        'speed': 'input_number.rain_speed',
                        'direction': 'input_number.rain_direction',
                        'bearing': 'input_number.rain_bearing'
                    },
                    'defaults': {
                        'no_rain_value': 999,
                        'no_direction_value': -1,
                        'no_bearing_value': -1
                    },
                    'image_settings': {
                        'size': 256,
                        'zoom': 8,
                        'color_scheme': 3,
                        'options': '0_0'
                    },
                    'analysis_settings': {
                        'rain_threshold': 75,
                        'current_rain_threshold': 50,
                        'lat_range_deg': 1.80,
                        'lon_range_deg': 1.99,
                        'arrival_angle_threshold_deg': 45
                    },
                    'tracking_settings': {
                        'max_tracking_distance_km': 30,
                        'min_track_length': 2
                    },
                    'debug': {
                        'save_images': False,
                        'log_level': 'INFO'
                    },
                    'web_port': 8099
                }
        except json.JSONDecodeError as e:
            logging.error(f"Invalid configuration JSON: {e}")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate required configuration options"""
        required = ['latitude', 'longitude']
        for key in required:
            if key not in self.config:
                logging.error(f"Missing required configuration: {key}")
                sys.exit(1)
    
    def get(self, key, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            config_file = '/data/options.json' if os.path.exists('/data/options.json') else 'config.json'
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def update_location(self, lat, lng):
        """Update location coordinates"""
        self.config['latitude'] = lat
        self.config['longitude'] = lng
        self._save_config()
        logging.info(f"Location updated to ({lat}, {lng})")

class HomeAssistantAPI:
    """Handle Home Assistant API calls"""
    
    def __init__(self):
        self.base_url = "http://supervisor/core/api"
        self.token = os.environ.get('SUPERVISOR_TOKEN')
        
        # Check if running in Home Assistant
        if not self.token:
            logging.warning("SUPERVISOR_TOKEN not found - running in standalone mode")
            self.standalone = True
        else:
            self.standalone = False
    
    def call_service(self, service, entity_id, value):
        """Call a Home Assistant service"""
        if self.standalone:
            logging.debug(f"[Standalone] Would call {service} for {entity_id} with value {value}")
            return True
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        domain, service_name = service.split('/', 1)
        url = f"{self.base_url}/services/{domain}/{service_name}"
        data = {
            "entity_id": entity_id,
            "value": value
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            logging.debug(f"Successfully called {service} for {entity_id} with value {value}")
            return True
        except requests.exceptions.RequestException as e:
            error_details = e.response.text if hasattr(e, 'response') and e.response else str(e)
            logging.error(f"Error calling service {service} for {entity_id}: {error_details}")
            return False

class RainPredictor:
    """Main rain prediction logic"""
    
    def __init__(self, config, ha_api):
        self.config = config
        self.ha_api = ha_api
        self.running = False
        self._setup_logging()
        
        # Extract configuration values
        self.latitude = config.get('latitude')
        self.longitude = config.get('longitude')
        self.run_interval = config.get('run_interval_minutes', 3) * 60
        self.api_url = config.get('api_url', 'https://api.rainviewer.com/public/weather-maps.json')
        
        # Entity IDs
        self.entities = {
            'time': config.get('entities.time'),
            'distance': config.get('entities.distance'),
            'speed': config.get('entities.speed'),
            'direction': config.get('entities.direction'),
            'bearing': config.get('entities.bearing')
        }
        
        # Default values
        self.defaults = {
            'no_rain': config.get('defaults.no_rain_value', 999),
            'no_direction': config.get('defaults.no_direction_value', -1),
            'no_bearing': config.get('defaults.no_bearing_value', -1)
        }
        
        # Image settings
        self.image_size = config.get('image_settings.size', 256)
        self.image_zoom = config.get('image_settings.zoom', 8)
        self.image_color = config.get('image_settings.color_scheme', 3)
        self.image_opts = config.get('image_settings.options', '0_0')
        
        # Analysis settings
        self.threshold = config.get('analysis_settings.rain_threshold', 75)
        self.current_rain_threshold = config.get('analysis_settings.current_rain_threshold', 50)
        self.lat_range = config.get('analysis_settings.lat_range_deg', 1.80)
        self.lon_range = config.get('analysis_settings.lon_range_deg', 1.99)
        self.arrival_angle_threshold = config.get('analysis_settings.arrival_angle_threshold_deg', 45)
        
        # Tracking settings
        self.max_track_dist = config.get('tracking_settings.max_tracking_distance_km', 30)
        self.min_track_len = config.get('tracking_settings.min_track_length', 2)
        
        # Manual mode
        self.manual_mode = False
        self.manual_lat = None
        self.manual_lng = None
        
        # Debug settings
        self.save_images = config.get('debug.save_images', False)
        
        # Store latest prediction
        self.latest_prediction = {
            'time_to_rain': self.defaults['no_rain'],
            'speed': 0,
            'distance': self.defaults['no_rain'],
            'direction': self.defaults['no_direction'],
            'bearing': self.defaults['no_bearing'],
            'center': None
        }
        
        logging.info("Rain Predictor initialized successfully")
        self._log_config()
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level = self.config.get('debug.log_level', 'INFO')
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _log_config(self):
        """Log current configuration"""
        logging.info(f"Location: ({self.latitude}, {self.longitude})")
        logging.info(f"Run interval: {self.run_interval/60} minutes")
        logging.info(f"Image settings: Size={self.image_size}, Zoom={self.image_zoom}")
        logging.info(f"Analysis: Threshold={self.threshold}, Current Threshold={self.current_rain_threshold}")
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing from point 1 to point 2"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = atan2(x, y)
        bearing = degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def download_radar_image(self, image_url):
        """Download radar image from URL"""
        logging.debug(f"Downloading: {image_url}")
        try:
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            data = response.content
            
            if self.save_images:
                self._save_debug_image(data, image_url)
            
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image: {e}")
            return None
    
    def _save_debug_image(self, data, url):
        """Save downloaded image for debugging"""
        try:
            os.makedirs('/share/rain_predictor_debug', exist_ok=True)
            timestamp = int(time.time())
            filename = f"radar_{timestamp}.png"
            filepath = f"/share/rain_predictor_debug/{filename}"
            
            with open(filepath, 'wb') as f:
                f.write(data)
            logging.debug(f"Debug image saved: {filepath}")
        except Exception as e:
            logging.error(f"Error saving debug image: {e}")
    
    def analyze_radar_data(self, past_frames, api_data):
        """Analyze radar data - main prediction logic"""
        logging.info(f"Analyzing {len(past_frames)} frames")
        
        prediction = {
            'time_to_rain': self.defaults['no_rain'], 
            'speed': 0.0, 
            'distance': self.defaults['no_rain'],
            'direction': self.defaults['no_direction'], 
            'bearing': self.defaults['no_bearing'],
            'center': None
        }
        
        if not past_frames:
            logging.warning("No frames to analyze")
            self.latest_prediction = prediction
            return prediction
        
        # Check current location for rain first
        latest_frame = sorted(past_frames, key=lambda f: f.get('time', 0))[-1]
        if self._check_current_rain(latest_frame, api_data):
            logging.info("*** RAIN DETECTED AT CURRENT LOCATION! ***")
            prediction = {
                'time_to_rain': 0, 
                'speed': 0, 
                'distance': 0,
                'direction': self.defaults['no_direction'], 
                'bearing': self.defaults['no_bearing'],
                'center': {'lat': self.latitude, 'lng': self.longitude}
            }
            self.latest_prediction = prediction
            return prediction
        
        # Find closest approaching cell
        try:
            closest_cell = self._find_closest_approaching_cell(past_frames, api_data)
            if closest_cell:
                prediction.update(closest_cell)
        except Exception as e:
            logging.error(f"Error in radar analysis: {e}")
        
        self.latest_prediction = prediction
        return prediction
    
    def _check_current_rain(self, frame, api_data):
        """Check if rain is currently at location"""
        try:
            frame_path = frame.get('path')
            api_host = api_data.get('host')
            
            if not frame_path or not api_host:
                return False
            
            image_url = f"{api_host}{frame_path}/{self.image_size}/{self.image_zoom}/{self.latitude}/{self.longitude}/{self.image_color}/{self.image_opts}.png"
            image_data = self.download_radar_image(image_url)
            
            if not image_data:
                return False
            
            img = Image.open(io.BytesIO(image_data)).convert('L')
            img_array = np.array(img)
            img_height, img_width = img_array.shape
            center_y, center_x = (img_height - 1) / 2.0, (img_width - 1) / 2.0
            
            check_radius = 8
            y_start = max(0, int(center_y - check_radius))
            y_end = min(img_height, int(center_y + check_radius + 1))
            x_start = max(0, int(center_x - check_radius))
            x_end = min(img_width, int(center_x + check_radius + 1))
            
            center_area = img_array[y_start:y_end, x_start:x_end]
            return np.any(center_area > self.current_rain_threshold)
            
        except Exception as e:
            logging.error(f"Error checking current rain: {e}")
            return False
    
    def _find_closest_approaching_cell(self, past_frames, api_data):
        """Find closest approaching rain cell"""
        try:
            sorted_frames = sorted(past_frames, key=lambda f: f.get('time', 0))
            if len(sorted_frames) < self.min_track_len:
                return None
            
            # Extract cells from last few frames
            frame_cells = []
            for frame in sorted_frames[-self.min_track_len:]:
                cells = self._extract_cells_from_frame(frame, api_data)
                if cells:
                    frame_cells.append((frame['time'], cells))
            
            if not frame_cells:
                return None
            
            # Track cell movement across frames
            tracks = []
            for i in range(1, len(frame_cells)):
                prev_time, prev_cells = frame_cells[i-1]
                curr_time, curr_cells = frame_cells[i]
                time_diff = (curr_time - prev_time) / 3600.0  # hours
                
                for curr_cell in curr_cells:
                    # Find matching previous cell
                    if prev_cells:
                        matching_prev = min(prev_cells, key=lambda p: self.haversine(p['lat'], p['lon'], curr_cell['lat'], curr_cell['lon']))
                        track_dist = self.haversine(matching_prev['lat'], matching_prev['lon'], curr_cell['lat'], curr_cell['lon'])
                        if track_dist < self.max_track_dist:
                            speed = track_dist / time_diff if time_diff > 0 else 0
                            direction = self.calculate_bearing(matching_prev['lat'], matching_prev['lon'], curr_cell['lat'], curr_cell['lon'])
                            tracks.append({
                                'lat': curr_cell['lat'],
                                'lon': curr_cell['lon'],
                                'speed': speed,
                                'direction': direction
                            })
            
            if not tracks:
                return None
            
            # Find closest track
            closest = min(tracks, key=lambda t: self.haversine(self.latitude, self.longitude, t['lat'], t['lon']))
            distance = self.haversine(self.latitude, self.longitude, closest['lat'], closest['lon'])
            speed = closest['speed']
            direction = closest['direction']
            
            # Check if approaching
            bearing_to_cell = self.calculate_bearing(self.latitude, self.longitude, closest['lat'], closest['lon'])
            approach_angle = abs((bearing_to_cell - direction + 180) % 360 - 180)
            if approach_angle > self.arrival_angle_threshold:
                logging.debug(f"Cell not approaching (angle: {approach_angle}°)")
                return None
            
            time_to_rain = (distance / speed * 60) if speed > 0 else self.defaults['no_rain']
            time_to_rain = min(9999, max(0, round(time_to_rain)))
            
            return {
                'time_to_rain': time_to_rain,
                'speed': round(speed, 1),
                'distance': round(distance, 1),
                'direction': round(direction, 1) if direction else self.defaults['no_direction'],
                'bearing': round(bearing_to_cell, 1) if bearing_to_cell else self.defaults['no_bearing'],
                'center': {'lat': closest['lat'], 'lng': closest['lon']}
            }
        
        except Exception as e:
            logging.error(f"Error finding approaching cell: {e}")
            return None
    
    def _extract_cells_from_frame(self, frame, api_data):
        """Extract rain cells from a single frame"""
        try:
            frame_path = frame.get('path')
            api_host = api_data.get('host')
            
            if not frame_path or not api_host:
                return []
            
            image_url = f"{api_host}{frame_path}/{self.image_size}/{self.image_zoom}/{self.latitude}/{self.longitude}/{self.image_color}/{self.image_opts}.png"
            image_data = self.download_radar_image(image_url)
            
            if not image_data:
                return []
            
            img = Image.open(io.BytesIO(image_data)).convert('L')
            img_array = np.array(img)
            
            rain_pixels = img_array > self.threshold
            if not np.any(rain_pixels):
                return []
            
            labeled_image, num_labels = label(rain_pixels)
            
            cells = []
            img_height, img_width = img_array.shape
            lat_inc = self.lat_range / img_height
            lon_inc = self.lon_range / img_width
            center_y = (img_height - 1) / 2.0
            center_x = (img_width - 1) / 2.0
            
            for i in range(1, num_labels + 1):
                y_coords, x_coords = np.where(labeled_image == i)
                if len(y_coords) == 0:
                    continue
                
                centroid_x, centroid_y = np.mean(x_coords), np.mean(y_coords)
                lat_offset = (center_y - centroid_y) * lat_inc
                lon_offset = (centroid_x - center_x) * lon_inc
                est_lat = np.clip(self.latitude + lat_offset, -90, 90)
                est_lon = np.clip(self.longitude + lon_offset, -180, 180)
                
                cells.append({
                    'lat': est_lat,
                    'lon': est_lon,
                    'x': centroid_x,
                    'y': centroid_y
                })
            
            return cells
            
        except Exception as e:
            logging.error(f"Error extracting cells from frame: {e}")
            return []
    
    def run_prediction(self):
        """Run a single prediction cycle"""
        logging.info("Running prediction cycle")
        
        # Initialize with defaults
        values = {
            'time': self.defaults['no_rain'],
            'distance': self.defaults['no_rain'],
            'speed': 0.0,
            'direction': self.defaults['no_direction'],
            'bearing': self.defaults['no_bearing']
        }
        
        try:
            # Fetch API data
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            
            if not self._validate_api_response(api_data):
                logging.error("Invalid API response")
            else:
                past_frames = api_data['radar'].get('past', [])
                if past_frames:
                    prediction = self.analyze_radar_data(past_frames, api_data)
                    values.update({
                        'time': max(0, round(prediction['time_to_rain'])),
                        'distance': max(0, round(prediction['distance'], 1)),
                        'speed': max(0, round(prediction['speed'], 1)),
                        'direction': round(prediction['direction'], 1),
                        'bearing': round(prediction['bearing'], 1)
                    })
                else:
                    logging.warning("No past radar frames available")
        
        except Exception as e:
            logging.error(f"Error in prediction cycle: {e}")
        
        # Update Home Assistant entities
        self._update_entities(values)
    
    def _validate_api_response(self, api_data):
        """Validate API response structure"""
        required_keys = ['radar', 'host']
        for key in required_keys:
            if key not in api_data:
                logging.error(f"Missing API key: {key}")
                return False
        
        if 'past' not in api_data.get('radar', {}):
            logging.error("Missing 'past' in radar data")
            return False
        
        return True
    
    def _update_entities(self, values):
        """Update Home Assistant entities"""
        entity_map = {
            'time': 'time',
            'distance': 'distance', 
            'speed': 'speed',
            'direction': 'direction',
            'bearing': 'bearing'
        }
        
        for key, entity_key in entity_map.items():
            entity_id = self.entities.get(entity_key)
            if entity_id:
                success = self.ha_api.call_service(
                    'input_number/set_value',
                    entity_id,
                    values[key]
                )
                if success:
                    logging.debug(f"Updated {entity_id} = {values[key]}")
    
    def update_location(self, lat, lng):
        """Update location (called from web API)"""
        self.latitude = lat
        self.longitude = lng
        self.config.update_location(lat, lng)
        logging.info(f"Location updated to ({lat}, {lng})")
        # Run a prediction with new location
        self.run_prediction()
    
    def start(self):
        """Start the prediction service"""
        self.running = True
        logging.info(f"Starting rain predictor (interval: {self.run_interval/60} minutes)")
        
        def run_loop():
            while self.running:
                try:
                    self.run_prediction()
                    time.sleep(self.run_interval)
                except Exception as e:
                    logging.error(f"Unexpected error in main loop: {e}")
                    time.sleep(60)
        
        # Run in separate thread
        self.thread = threading.Thread(target=run_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the prediction service"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)


# Global references
config = None
ha_api = None
predictor = None

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Try to read index.html from file
        html_path = '/app/index.html' if os.path.exists('/app/index.html') else 'index.html'
        if os.path.exists(html_path):
            with open(html_path, 'r') as f:
                html_content = f.read()
            # Replace template variables
            html_content = html_content.replace('{{ latitude }}', str(config.get('latitude')))
            html_content = html_content.replace('{{ longitude }}', str(config.get('longitude')))
            return html_content
        else:
            return "index.html not found", 404
    except Exception as e:
        logging.error(f"Error serving index page: {e}")
        return f"Error: {e}", 500

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify({
        'latitude': config.get('latitude'),
        'longitude': config.get('longitude'),
        'run_interval_minutes': config.get('run_interval_minutes', 3)
    })

@app.route('/api/status')
def get_status():
    """Get current rain prediction status"""
    if predictor and predictor.latest_prediction:
        return jsonify(predictor.latest_prediction)
    else:
        return jsonify({
            'time_to_rain': 999,
            'speed': 0,
            'distance': 999,
            'direction': -1,
            'bearing': -1,
            'center': None
        })

@app.route('/api/data')
def get_data():
    """Get current rain data for visualization"""
    if predictor and predictor.latest_prediction:
        pred = predictor.latest_prediction
        return jsonify({
            'time_to_rain': pred.get('time_to_rain', 999),
            'speed': pred.get('speed', 0),
            'distance': pred.get('distance', 999),
            'direction': pred.get('direction', -1),
            'bearing': pred.get('bearing', -1),
            'center': pred.get('center')
        })
    else:
        return jsonify({})

@app.route('/api/set_location', methods=['POST'])
def set_location():
    """Update user location"""
    try:
        data = request.json
        lat = float(data.get('latitude'))
        lng = float(data.get('longitude'))
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Update location
        predictor.update_location(lat, lng)
        
        return jsonify({
            'success': True,
            'latitude': lat,
            'longitude': lng
        })
    except Exception as e:
        logging.error(f"Error setting location: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual_selection', methods=['POST'])
def manual_selection():
    """Handle manual rain cell selection"""
    try:
        data = request.json
        lat = float(data.get('latitude'))
        lng = float(data.get('longitude'))
        
        # For demo, return simulated cell data
        # In production, this would analyze the radar at that location
        return jsonify({
            'center': {'lat': lat, 'lng': lng},
            'speed': 25 + np.random.random() * 20,
            'direction': np.random.random() * 360,
            'intensity': np.random.random() * 100
        })
    except Exception as e:
        logging.error(f"Error in manual selection: {e}")
        return jsonify({'error': str(e)}), 400

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logging.info("Received shutdown signal")
    global predictor
    if predictor:
        predictor.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize components
        config = AddonConfig()
        ha_api = HomeAssistantAPI()
        predictor = RainPredictor(config, ha_api)
        
        # Start the prediction service
        predictor.start()
        
        # Start web server
        web_port = config.get('web_port', 8099)
        logging.info(f"Starting web server on port {web_port}")
        app.run(host='0.0.0.0', port=web_port, debug=False)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)