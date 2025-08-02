#!/usr/bin/env python3
"""
Rain Predictor Home Assistant Addon
Converted from AppDaemon app to standalone addon
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

class AddonConfig:
    """Load and manage addon configuration"""
    
    def __init__(self):
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from addon options"""
        try:
            with open('/data/options.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error("Configuration file not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid configuration JSON: {e}")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate required configuration options"""
        required = ['latitude', 'longitude', 'entities']
        for key in required:
            if key not in self.config:
                logging.error(f"Missing required configuration: {key}")
                sys.exit(1)
        
        if 'time' not in self.config['entities']:
            logging.error("Time entity must be configured")
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

class HomeAssistantAPI:
    """Handle Home Assistant API calls"""
    
    def __init__(self):
        self.base_url = "http://supervisor/core/api"
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('SUPERVISOR_TOKEN')}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def call_service(self, service, entity_id, value):
        """Call a Home Assistant service"""
        domain, service_name = service.split('/', 1)
        url = f"{self.base_url}/services/{domain}/{service_name}"
        data = {
            "entity_id": entity_id,
            "value": value
        }
        
        try:
            response = self.session.post(url, json=data, timeout=10)
            response.raise_for_status()
            logging.debug(f"Successfully called {service} for {entity_id} with value {value}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling service {service} for {entity_id}: {e}")
            return False

class RainPredictor:
    """Main rain prediction logic - adapted from AppDaemon version"""
    
    def __init__(self, config, ha_api):
        self.config = config
        self.ha_api = ha_api
        self.running = False
        self._setup_logging()
        
        # Extract configuration values
        self.latitude = config.get('latitude')
        self.longitude = config.get('longitude')
        self.run_interval = config.get('run_interval_minutes', 3) * 60
        self.api_url = config.get('api_url')
        
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
        self.lat_range = config.get('analysis_settings.lat_range_deg', 1.80)
        self.lon_range = config.get('analysis_settings.lon_range_deg', 1.99)
        self.arrival_angle_threshold = config.get('analysis_settings.arrival_angle_threshold_deg', 45)
        
        # Tracking settings
        self.max_track_dist = config.get('tracking_settings.max_tracking_distance_km', 30)
        self.min_track_len = config.get('tracking_settings.min_track_length', 2)
        
        # Debug settings
        self.save_images = config.get('debug.save_images', False)
        
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
        logging.info(f"Time entity: {self.entities['time']}")
        logging.info(f"Image settings: Size={self.image_size}, Zoom={self.image_zoom}")
        logging.info(f"Analysis: Threshold={self.threshold}, Angle={self.arrival_angle_threshold}°")
    
    # Helper methods (same as original)
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points"""
        R = 6371
        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
            c = 2 * asin(sqrt(a))
            distance = R * c
            return max(0, distance)
        except ValueError as e:
            logging.error(f"Haversine error: {e}")
            return float('inf')
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate initial bearing from point 1 to point 2"""
        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2_rad - lon1_rad
            y = sin(dlon) * cos(lat2_rad)
            x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
            initial_bearing = atan2(y, x)
            bearing = (degrees(initial_bearing) + 360) % 360
            return bearing
        except ValueError as e:
            logging.error(f"Bearing calculation error: {e}")
            return None
    
    def degrees_to_compass(self, degrees):
        """Convert degrees to compass direction"""
        if degrees is None:
            return "Unknown"
        
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def download_radar_image(self, image_url):
        """Download radar image from URL"""
        logging.debug(f"Downloading: {image_url}")
        try:
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            data = response.content
            
            # Optionally save images for debugging
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
        """Analyze radar data - main prediction logic (simplified from original)"""
        logging.info(f"Analyzing {len(past_frames)} frames")
        
        prediction = {
            'time_to_rain': None, 
            'speed_kph': None, 
            'distance_km': None,
            'direction_deg': None, 
            'bearing_to_cell_deg': None
        }
        
        if not past_frames:
            logging.warning("No frames to analyze")
            return prediction
        
        # Check current location for rain first
        latest_frame = sorted(past_frames, key=lambda f: f.get('time', 0))[-1]
        if self._check_current_rain(latest_frame, api_data):
            logging.info("*** RAIN DETECTED AT CURRENT LOCATION! ***")
            return {
                'time_to_rain': 0, 
                'speed_kph': 0, 
                'distance_km': 0,
                'direction_deg': self.defaults['no_direction'], 
                'bearing_to_cell_deg': self.defaults['no_bearing']
            }
        
        # Simplified analysis - find closest approaching cell
        try:
            closest_cell = self._find_closest_approaching_cell(past_frames, api_data)
            if closest_cell:
                prediction.update(closest_cell)
        except Exception as e:
            logging.error(f"Error in radar analysis: {e}")
        
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
            
            # Check small area around center
            check_radius = 5
            y_start = max(0, int(center_y - check_radius))
            y_end = min(img_height, int(center_y + check_radius + 1))
            x_start = max(0, int(center_x - check_radius))
            x_end = min(img_width, int(center_x + check_radius + 1))
            
            center_area = img_array[y_start:y_end, x_start:x_end]
            return np.any(center_area > self.threshold)
            
        except Exception as e:
            logging.error(f"Error checking current rain: {e}")
            return False
    
    def _find_closest_approaching_cell(self, past_frames, api_data):
        """Simplified method to find closest approaching rain cell"""
        # This is a simplified version - you would implement the full
        # cell tracking logic from your original code here
        
        try:
            # Get the last two frames for movement calculation
            sorted_frames = sorted(past_frames, key=lambda f: f.get('time', 0))
            if len(sorted_frames) < 2:
                return None
            
            frame1, frame2 = sorted_frames[-2], sorted_frames[-1]
            
            # Analyze both frames and find cells
            cells1 = self._extract_cells_from_frame(frame1, api_data)
            cells2 = self._extract_cells_from_frame(frame2, api_data)
            
            if not cells1 or not cells2:
                return None
            
            # Find closest cell in latest frame
            closest_cell = min(cells2, key=lambda c: self.haversine(
                self.latitude, self.longitude, c['lat'], c['lon']
            ))
            
            distance = self.haversine(self.latitude, self.longitude, 
                                    closest_cell['lat'], closest_cell['lon'])
            
            # Calculate basic prediction
            if distance < 100:  # Within 100km
                bearing = self.calculate_bearing(self.latitude, self.longitude,
                                               closest_cell['lat'], closest_cell['lon'])
                
                return {
                    'distance_km': round(distance, 2),
                    'bearing_to_cell_deg': round(bearing, 1) if bearing else self.defaults['no_bearing'],
                    'time_to_rain': min(999, round(distance * 2))  # Rough estimate
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
            
            # Find rain pixels
            rain_pixels = img_array > self.threshold
            if not np.any(rain_pixels):
                return []
            
            # Label connected components
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
                # Analyze radar data
                past_frames = api_data['radar'].get('past', [])
                if past_frames:
                    prediction = self.analyze_radar_data(past_frames, api_data)
                    
                    # Update values based on prediction
                    if prediction.get('time_to_rain') is not None:
                        values['time'] = max(0, round(prediction['time_to_rain']))
                    if prediction.get('distance_km') is not None:
                        values['distance'] = max(0, round(prediction['distance_km'], 1))
                    if prediction.get('speed_kph') is not None:
                        values['speed'] = max(0, round(prediction['speed_kph'], 1))
                    if prediction.get('direction_deg') is not None:
                        values['direction'] = round(prediction['direction_deg'], 1)
                    if prediction.get('bearing_to_cell_deg') is not None:
                        values['bearing'] = round(prediction['bearing_to_cell_deg'], 1)
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
    
    def start(self):
        """Start the prediction service"""
        self.running = True
        logging.info(f"Starting rain predictor (interval: {self.run_interval/60} minutes)")
        
        while self.running:
            try:
                self.run_prediction()
                time.sleep(self.run_interval)
            except KeyboardInterrupt:
                logging.info("Received interrupt signal")
                break
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        logging.info("Rain predictor stopped")
    
    def stop(self):
        """Stop the prediction service"""
        self.running = False

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
        
        # Start the service
        predictor.start()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)