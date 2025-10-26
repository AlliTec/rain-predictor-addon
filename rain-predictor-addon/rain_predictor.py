#!/usr/bin/env python3
"""
Rain Predictor Home Assistant Addon - Complete Implementation
Tracks rain cells and predicts arrival time at location
"""

import json
import time
import logging
import os
import sys
import requests
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import io
from scipy.ndimage import label
from math import radians, cos, sin, asin, sqrt, atan2, degrees
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

class RainCell:
    """Represents a tracked rain cell"""
    
    def __init__(self, cell_id, lat, lon, timestamp, intensity=0):
        self.id = cell_id
        self.positions = [(lat, lon, timestamp)]
        self.intensity = intensity
        self.last_seen = timestamp
    
    def add_position(self, lat, lon, timestamp, intensity=0):
        """Add a new position observation"""
        self.positions.append((lat, lon, timestamp))
        self.last_seen = timestamp
        self.intensity = max(self.intensity, intensity)
        
        # Keep only recent positions (last 10)
        if len(self.positions) > 10:
            self.positions = self.positions[-10:]
    
    def get_velocity(self):
        """Calculate velocity vector (km/h) from recent positions"""
        if len(self.positions) < 2:
            return None, None
        
        # Use last two positions for velocity
        (lat1, lon1, t1), (lat2, lon2, t2) = self.positions[-2:]
        
        time_diff = (t2 - t1).total_seconds() / 3600.0  # hours
        if time_diff <= 0:
            return None, None
        
        # Calculate distance
        distance_km = self._haversine(lat1, lon1, lat2, lon2)
        
        # Calculate bearing
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Calculate speed
        speed_kph = distance_km / time_diff
        
        return speed_kph, bearing
    
    def predict_position(self, minutes_ahead):
        """Predict future position based on current velocity"""
        speed_kph, bearing = self.get_velocity()
        
        if speed_kph is None or bearing is None:
            return None
        
        lat, lon, _ = self.positions[-1]
        
        # Calculate new position
        distance_km = speed_kph * (minutes_ahead / 60.0)
        new_lat, new_lon = self._point_at_distance_and_bearing(lat, lon, distance_km, bearing)
        
        return new_lat, new_lon
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance"""
        R = 6371
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    
    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing from point 1 to point 2"""
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2_rad - lon1_rad
        y = sin(dlon) * cos(lat2_rad)
        x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
        bearing = atan2(y, x)
        return (degrees(bearing) + 360) % 360
    
    def _point_at_distance_and_bearing(self, lat, lon, distance_km, bearing_deg):
        """Calculate point at given distance and bearing"""
        R = 6371
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        bearing_rad = radians(bearing_deg)
        
        new_lat_rad = asin(sin(lat_rad) * cos(distance_km / R) +
                          cos(lat_rad) * sin(distance_km / R) * cos(bearing_rad))
        
        new_lon_rad = lon_rad + atan2(sin(bearing_rad) * sin(distance_km / R) * cos(lat_rad),
                                      cos(distance_km / R) - sin(lat_rad) * sin(new_lat_rad))
        
        return degrees(new_lat_rad), degrees(new_lon_rad)

class RainPredictor:
    """Main rain prediction logic"""
    
    def __init__(self, config, ha_api):
        self.config = config
        self.ha_api = ha_api
        self.running = False
        self.tracked_cells = {}
        self.next_cell_id = 1
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
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    def _log_config(self):
        """Log current configuration"""
        logging.info(f"Location: ({self.latitude}, {self.longitude})")
        logging.info(f"Run interval: {self.run_interval/60} minutes")
        logging.info(f"Time entity: {self.entities['time']}")
        logging.info(f"Image settings: Size={self.image_size}, Zoom={self.image_zoom}")
        logging.info(f"Analysis: Threshold={self.threshold}, Angle={self.arrival_angle_threshold}°")
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate great-circle distance between two points"""
        R = 6371
        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
            c = 2 * asin(sqrt(a))
            return max(0, R * c)
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
            bearing = atan2(y, x)
            return (degrees(bearing) + 360) % 360
        except ValueError as e:
            logging.error(f"Bearing calculation error: {e}")
            return None
    
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
        """Analyze radar data and predict rain arrival"""
        logging.info(f"Analyzing {len(past_frames)} frames")
        
        prediction = {
            'time_to_rain': None,
            'speed_kph': None,
            'distance_km': None,
            'direction_deg': None,
            'bearing_to_cell_deg': None
        }
        
        if not past_frames or len(past_frames) < 2:
            logging.warning("Need at least 2 frames for tracking")
            return prediction
        
        # Check if rain is currently at location
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
        
        # Process all frames and track cells
        try:
            sorted_frames = sorted(past_frames, key=lambda f: f.get('time', 0))
            
            for frame in sorted_frames:
                frame_time = datetime.fromtimestamp(frame['time'])
                cells = self._extract_cells_from_frame(frame, api_data)
                
                if cells:
                    self._update_tracked_cells(cells, frame_time)
            
            # Find threatening cell
            threat = self._find_threatening_cell()
            
            if threat:
                prediction.update(threat)
                logging.info(f"Threat detected: {threat}")
            else:
                logging.info("No threatening cells detected")
        
        except Exception as e:
            logging.error(f"Error in radar analysis: {e}", exc_info=True)
        
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
            
            # Check area around center
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
                if len(y_coords) < 5:  # Minimum size filter
                    continue
                
                centroid_x, centroid_y = np.mean(x_coords), np.mean(y_coords)
                lat_offset = (center_y - centroid_y) * lat_inc
                lon_offset = (centroid_x - center_x) * lon_inc
                est_lat = np.clip(self.latitude + lat_offset, -90, 90)
                est_lon = np.clip(self.longitude + lon_offset, -180, 180)
                
                # Calculate average intensity
                intensity = np.mean(img_array[y_coords, x_coords])
                
                cells.append({
                    'lat': est_lat,
                    'lon': est_lon,
                    'intensity': intensity,
                    'size': len(y_coords)
                })
            
            logging.debug(f"Extracted {len(cells)} cells from frame")
            return cells
            
        except Exception as e:
            logging.error(f"Error extracting cells: {e}")
            return []
    
    def _update_tracked_cells(self, detected_cells, timestamp):
        """Update tracked cells with new detections"""
        # Match detected cells to existing tracks
        unmatched_cells = detected_cells.copy()
        
        for cell_id, tracked_cell in list(self.tracked_cells.items()):
            if not tracked_cell.positions:
                continue
            
            last_lat, last_lon, _ = tracked_cell.positions[-1]
            
            # Find closest detected cell
            closest_cell = None
            min_distance = float('inf')
            
            for cell in unmatched_cells:
                dist = self.haversine(last_lat, last_lon, cell['lat'], cell['lon'])
                if dist < min_distance and dist < self.max_track_dist:
                    min_distance = dist
                    closest_cell = cell
            
            # Update track if match found
            if closest_cell:
                tracked_cell.add_position(
                    closest_cell['lat'],
                    closest_cell['lon'],
                    timestamp,
                    closest_cell.get('intensity', 0)
                )
                unmatched_cells.remove(closest_cell)
        
        # Create new tracks for unmatched cells
        for cell in unmatched_cells:
            new_track = RainCell(
                self.next_cell_id,
                cell['lat'],
                cell['lon'],
                timestamp,
                cell.get('intensity', 0)
            )
            self.tracked_cells[self.next_cell_id] = new_track
            self.next_cell_id += 1
            logging.debug(f"Created new track ID {new_track.id}")
        
        # Clean up old tracks
        current_time = timestamp
        for cell_id in list(self.tracked_cells.keys()):
            track = self.tracked_cells[cell_id]
            age = (current_time - track.last_seen).total_seconds() / 60.0
            if age > 30:  # Remove tracks older than 30 minutes
                del self.tracked_cells[cell_id]
                logging.debug(f"Removed old track ID {cell_id}")
    
    def _find_threatening_cell(self):
        """Find cell most likely to reach location"""
        best_prediction = None
        min_arrival_time = float('inf')
        
        for cell_id, cell in self.tracked_cells.items():
            if len(cell.positions) < self.min_track_len:
                continue
            
            # Get current position and velocity
            current_lat, current_lon, _ = cell.positions[-1]
            speed_kph, direction_deg = cell.get_velocity()
            
            if speed_kph is None or speed_kph < 1:
                continue
            
            # Calculate current distance and bearing to location
            distance_km = self.haversine(current_lat, current_lon, self.latitude, self.longitude)
            bearing_to_location = self.calculate_bearing(current_lat, current_lon, self.latitude, self.longitude)
            
            if bearing_to_location is None:
                continue
            
            # Check if cell is moving toward location
            angle_diff = abs((direction_deg - bearing_to_location + 180) % 360 - 180)
            
            if angle_diff > self.arrival_angle_threshold:
                logging.debug(f"Cell {cell_id}: angle diff {angle_diff:.1f}° too large")
                continue
            
            # Calculate arrival time
            time_to_arrival_hours = distance_km / speed_kph
            time_to_arrival_minutes = time_to_arrival_hours * 60
            
            logging.info(f"Cell {cell_id}: dist={distance_km:.1f}km, speed={speed_kph:.1f}kph, "
                        f"dir={direction_deg:.1f}°, bearing={bearing_to_location:.1f}°, "
                        f"angle_diff={angle_diff:.1f}°, ETA={time_to_arrival_minutes:.0f}min")
            
            if time_to_arrival_minutes < min_arrival_time:
                min_arrival_time = time_to_arrival_minutes
                best_prediction = {
                    'time_to_rain': round(time_to_arrival_minutes),
                    'distance_km': round(distance_km, 1),
                    'speed_kph': round(speed_kph, 1),
                    'direction_deg': round(direction_deg, 1),
                    'bearing_to_cell_deg': round(bearing_to_location, 1)
                }
        
        return best_prediction
    
    def run_prediction(self):
        """Run a single prediction cycle"""
        logging.info("=" * 50)
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
            logging.error(f"Error in prediction cycle: {e}", exc_info=True)
        
        # Update Home Assistant entities
        self._update_entities(values)
        logging.info(f"Updated values: {values}")
    
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
                logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(60)
        
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
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    predictor = None
    try:
        config = AddonConfig()
        ha_api = HomeAssistantAPI()
        predictor = RainPredictor(config, ha_api)
        predictor.start()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)