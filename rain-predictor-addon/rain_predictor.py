#!/usr/bin/env python3
"""
Rain Predictor Home Assistant Addon - Debug Version
"""

import json
import time
import logging
import os
import sys
import requests
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
import io
from scipy.ndimage import label
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import signal

VERSION = "1.1.0-debug"

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
        
        if len(self.positions) > 10:
            self.positions = self.positions[-10:]
    
    def get_velocity(self):
        """Calculate velocity vector (km/h) from recent positions"""
        if len(self.positions) < 2:
            return None, None
        
        (lat1, lon1, t1), (lat2, lon2, t2) = self.positions[-2:]
        
        time_diff = (t2 - t1).total_seconds() / 3600.0
        if time_diff <= 0:
            return None, None
        
        distance_km = self._haversine(lat1, lon1, lat2, lon2)
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
        speed_kph = distance_km / time_diff
        
        return speed_kph, bearing
    
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
        
        logging.info(f"Rain Predictor {VERSION} initialized")
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
        logging.info("=" * 60)
        logging.info(f"Location: ({self.latitude}, {self.longitude})")
        logging.info(f"Run interval: {self.run_interval/60} minutes")
        logging.info(f"Time entity: {self.entities['time']}")
        logging.info(f"Image: Size={self.image_size}, Zoom={self.image_zoom}")
        logging.info(f"Threshold: {self.threshold}")
        logging.info(f"Lat/Lon Range: {self.lat_range}° x {self.lon_range}°")
        logging.info(f"Angle threshold: {self.arrival_angle_threshold}°")
        logging.info(f"Max track distance: {self.max_track_dist}km")
        logging.info(f"Min track length: {self.min_track_len}")
        logging.info(f"Save images: {self.save_images}")
        logging.info("=" * 60)
    
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
        """Save downloaded image for debugging with annotations"""
        try:
            os.makedirs('/share/rain_predictor_debug', exist_ok=True)
            timestamp = int(time.time())
            
            # Save original
            filename = f"radar_{timestamp}.png"
            filepath = f"/share/rain_predictor_debug/{filename}"
            with open(filepath, 'wb') as f:
                f.write(data)
            
            # Create annotated version
            img = Image.open(io.BytesIO(data)).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Draw center crosshair (your location)
            center_x = img.width // 2
            center_y = img.height // 2
            size = 10
            draw.line([(center_x - size, center_y), (center_x + size, center_y)], fill='red', width=3)
            draw.line([(center_x, center_y - size), (center_x, center_y + size)], fill='red', width=3)
            draw.ellipse([center_x-15, center_y-15, center_x+15, center_y+15], outline='red', width=2)
            
            annotated_path = f"/share/rain_predictor_debug/annotated_{timestamp}.png"
            img.save(annotated_path)
            
            logging.debug(f"Debug images saved: {filepath}, {annotated_path}")
        except Exception as e:
            logging.error(f"Error saving debug image: {e}")
    
    def analyze_radar_data(self, past_frames, api_data):
        """Analyze radar data and predict rain arrival"""
        logging.info("=" * 60)
        logging.info(f"ANALYZING {len(past_frames)} FRAMES")
        logging.info("=" * 60)
        
        prediction = {
            'time_to_rain': None,
            'speed_kph': None,
            'distance_km': None,
            'direction_deg': None,
            'bearing_to_cell_deg': None
        }
        
        if not past_frames:
            logging.warning("❌ No frames to analyze")
            return prediction
            
        if len(past_frames) < 2:
            logging.warning(f"❌ Only {len(past_frames)} frame(s) - need at least 2 for tracking")
            return prediction
        
        # Check if rain is currently at location
        latest_frame = sorted(past_frames, key=lambda f: f.get('time', 0))[-1]
        if self._check_current_rain(latest_frame, api_data):
            logging.info("🌧️  RAIN DETECTED AT CURRENT LOCATION!")
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
            logging.info(f"Processing frames from {datetime.fromtimestamp(sorted_frames[0]['time'])} to {datetime.fromtimestamp(sorted_frames[-1]['time'])}")
            
            total_cells_detected = 0
            for i, frame in enumerate(sorted_frames):
                frame_time = datetime.fromtimestamp(frame['time'])
                logging.info(f"\n--- Frame {i+1}/{len(sorted_frames)} at {frame_time} ---")
                
                cells = self._extract_cells_from_frame(frame, api_data)
                logging.info(f"Found {len(cells)} cell(s) in frame")
                
                if cells:
                    total_cells_detected += len(cells)
                    for j, cell in enumerate(cells):
                        dist = self.haversine(self.latitude, self.longitude, cell['lat'], cell['lon'])
                        bearing = self.calculate_bearing(self.latitude, self.longitude, cell['lat'], cell['lon'])
                        logging.info(f"  Cell {j+1}: lat={cell['lat']:.4f}, lon={cell['lon']:.4f}, "
                                   f"intensity={cell['intensity']:.1f}, size={cell['size']}, "
                                   f"dist={dist:.1f}km, bearing={bearing:.1f}°")
                    
                    self._update_tracked_cells(cells, frame_time)
            
            logging.info(f"\n📊 SUMMARY: Detected {total_cells_detected} total cells across all frames")
            logging.info(f"Currently tracking {len(self.tracked_cells)} cell(s)")
            
            # Show tracking details
            for cell_id, cell in self.tracked_cells.items():
                logging.info(f"\nTracked Cell #{cell_id}:")
                logging.info(f"  Positions: {len(cell.positions)}")
                logging.info(f"  Intensity: {cell.intensity:.1f}")
                speed, direction = cell.get_velocity()
                if speed and direction:
                    logging.info(f"  Velocity: {speed:.1f} km/h at {direction:.1f}°")
                else:
                    logging.info(f"  Velocity: Not enough data yet")
            
            # Find threatening cell
            threat = self._find_threatening_cell()
            
            if threat:
                logging.info(f"\n⚠️  THREAT DETECTED: {threat}")
                prediction.update(threat)
            else:
                logging.info("\n✅ No threatening cells detected")
        
        except Exception as e:
            logging.error(f"❌ Error in radar analysis: {e}", exc_info=True)
        
        return prediction
    
    def _check_current_rain(self, frame, api_data):
        """Check if rain is currently at location"""
        try:
            frame_path = frame.get('path')
            api_host = api_data.get('host')
            
            if not frame_path or not api_host:
                logging.warning("Missing frame path or API host")
                return False
            
            image_url = f"{api_host}{frame_path}/{self.image_size}/{self.image_zoom}/{self.latitude}/{self.longitude}/{self.image_color}/{self.image_opts}.png"
            logging.debug(f"Checking current rain at: {image_url}")
            
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
            max_intensity = np.max(center_area)
            
            logging.debug(f"Center area max intensity: {max_intensity} (threshold: {self.threshold})")
            
            return np.any(center_area > self.threshold)
            
        except Exception as e:
            logging.error(f"Error checking current rain: {e}", exc_info=True)
            return False
    
    def _extract_cells_from_frame(self, frame, api_data):
        """Extract rain cells from a single frame"""
        try:
            frame_path = frame.get('path')
            api_host = api_data.get('host')
            
            if not frame_path or not api_host:
                logging.warning("Missing frame path or API host")
                return []
            
            image_url = f"{api_host}{frame_path}/{self.image_size}/{self.image_zoom}/{self.latitude}/{self.longitude}/{self.image_color}/{self.image_opts}.png"
            image_data = self.download_radar_image(image_url)
            
            if not image_data:
                logging.warning("Failed to download radar image")
                return []
            
            img = Image.open(io.BytesIO(image_data)).convert('L')
            img_array = np.array(img)
            
            logging.debug(f"Image shape: {img_array.shape}, min: {np.min(img_array)}, max: {np.max(img_array)}")
            
            # Find rain pixels
            rain_pixels = img_array > self.threshold
            rain_pixel_count = np.sum(rain_pixels)
            
            logging.debug(f"Rain pixels above threshold {self.threshold}: {rain_pixel_count}")
            
            if not np.any(rain_pixels):
                logging.debug("No rain pixels found above threshold")
                return []
            
            # Label connected components
            labeled_image, num_labels = label(rain_pixels)
            
            logging.debug(f"Found {num_labels} connected components")
            
            cells = []
            img_height, img_width = img_array.shape
            lat_inc = self.lat_range / img_height
            lon_inc = self.lon_range / img_width
            center_y = (img_height - 1) / 2.0
            center_x = (img_width - 1) / 2.0
            
            for i in range(1, num_labels + 1):
                y_coords, x_coords = np.where(labeled_image == i)
                cell_size = len(y_coords)
                
                if cell_size < 5:
                    logging.debug(f"  Component {i}: size {cell_size} too small, skipping")
                    continue
                
                centroid_x, centroid_y = np.mean(x_coords), np.mean(y_coords)
                lat_offset = (center_y - centroid_y) * lat_inc
                lon_offset = (centroid_x - center_x) * lon_inc
                est_lat = np.clip(self.latitude + lat_offset, -90, 90)
                est_lon = np.clip(self.longitude + lon_offset, -180, 180)
                
                intensity = np.mean(img_array[y_coords, x_coords])
                
                cells.append({
                    'lat': est_lat,
                    'lon': est_lon,
                    'intensity': intensity,
                    'size': cell_size
                })
                
                logging.debug(f"  Component {i}: size={cell_size}, intensity={intensity:.1f}, "
                            f"centroid=({centroid_x:.1f},{centroid_y:.1f}), "
                            f"location=({est_lat:.4f},{est_lon:.4f})")
            
            return cells
            
        except Exception as e:
            logging.error(f"Error extracting cells: {e}", exc_info=True)
            return []
    
    def _update_tracked_cells(self, detected_cells, timestamp):
        """Update tracked cells with new detections"""
        unmatched_cells = detected_cells.copy()
        matched_count = 0
        
        for cell_id, tracked_cell in list(self.tracked_cells.items()):
            if not tracked_cell.positions:
                continue
            
            last_lat, last_lon, _ = tracked_cell.positions[-1]
            
            closest_cell = None
            min_distance = float('inf')
            
            for cell in unmatched_cells:
                dist = self.haversine(last_lat, last_lon, cell['lat'], cell['lon'])
                if dist < min_distance and dist < self.max_track_dist:
                    min_distance = dist
                    closest_cell = cell
            
            if closest_cell:
                tracked_cell.add_position(
                    closest_cell['lat'],
                    closest_cell['lon'],
                    timestamp,
                    closest_cell.get('intensity', 0)
                )
                unmatched_cells.remove(closest_cell)
                matched_count += 1
                logging.debug(f"Matched cell to track #{cell_id} (moved {min_distance:.1f}km)")
        
        # Create new tracks
        for cell in unmatched_cells:
            new_track = RainCell(
                self.next_cell_id,
                cell['lat'],
                cell['lon'],
                timestamp,
                cell.get('intensity', 0)
            )
            self.tracked_cells[self.next_cell_id] = new_track
            logging.info(f"➕ Created new track ID {self.next_cell_id}")
            self.next_cell_id += 1
        
        # Clean up old tracks
        current_time = timestamp
        removed_count = 0
        for cell_id in list(self.tracked_cells.keys()):
            track = self.tracked_cells[cell_id]
            age = (current_time - track.last_seen).total_seconds() / 60.0
            if age > 30:
                del self.tracked_cells[cell_id]
                logging.debug(f"Removed old track ID {cell_id} (age: {age:.1f}min)")
                removed_count += 1
        
        logging.debug(f"Track update: {matched_count} matched, {len(unmatched_cells)} new, {removed_count} removed")
    
    def _find_threatening_cell(self):
        """Find cell most likely to reach location"""
        best_prediction = None
        min_arrival_time = float('inf')
        
        logging.info(f"\n🔍 Evaluating {len(self.tracked_cells)} tracked cells for threats:")
        
        for cell_id, cell in self.tracked_cells.items():
            logging.info(f"\n  Cell #{cell_id}:")
            logging.info(f"    Track length: {len(cell.positions)} position(s)")
            
            if len(cell.positions) < self.min_track_len:
                logging.info(f"    ❌ Track too short (need {self.min_track_len})")
                continue
            
            current_lat, current_lon, _ = cell.positions[-1]
            speed_kph, direction_deg = cell.get_velocity()
            
            if speed_kph is None or direction_deg is None:
                logging.info(f"    ❌ Cannot calculate velocity")
                continue
                
            if speed_kph < 1:
                logging.info(f"    ❌ Moving too slowly ({speed_kph:.1f} km/h)")
                continue
            
            distance_km = self.haversine(current_lat, current_lon, self.latitude, self.longitude)
            bearing_to_location = self.calculate_bearing(current_lat, current_lon, self.latitude, self.longitude)
            
            if bearing_to_location is None:
                logging.info(f"    ❌ Cannot calculate bearing")
                continue
            
            angle_diff = abs((direction_deg - bearing_to_location + 180) % 360 - 180)
            
            logging.info(f"    Distance: {distance_km:.1f}km")
            logging.info(f"    Speed: {speed_kph:.1f}km/h")
            logging.info(f"    Moving: {direction_deg:.1f}°")
            logging.info(f"    Bearing to location: {bearing_to_location:.1f}°")
            logging.info(f"    Angle difference: {angle_diff:.1f}°")
            
            if angle_diff > self.arrival_angle_threshold:
                logging.info(f"    ❌ Not moving toward location (angle diff {angle_diff:.1f}° > threshold {self.arrival_angle_threshold}°)")
                continue
            
            time_to_arrival_hours = distance_km / speed_kph
            time_to_arrival_minutes = time_to_arrival_hours * 60
            
            logging.info(f"    ✅ THREAT! ETA: {time_to_arrival_minutes:.0f} minutes")
            
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
        logging.info("\n\n" + "=" * 60)
        logging.info(f"PREDICTION CYCLE STARTING - {datetime.now()}")
        logging.info("=" * 60)
        
        values = {
            'time': self.defaults['no_rain'],
            'distance': self.defaults['no_rain'],
            'speed': 0.0,
            'direction': self.defaults['no_direction'],
            'bearing': self.defaults['no_bearing']
        }
        
        try:
            logging.info(f"Fetching API data from: {self.api_url}")
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            api_data = response.json()
            
            logging.info(f"API response received. Host: {api_data.get('host', 'unknown')}")
            
            if not self._validate_api_response(api_data):
                logging.error("❌ Invalid API response")
            else:
                past_frames = api_data['radar'].get('past', [])
                logging.info(f"Found {len(past_frames)} past frames")
                
                if past_frames:
                    prediction = self.analyze_radar_data(past_frames, api_data)
                    
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
                    logging.warning("❌ No past radar frames available")
        
        except Exception as e:
            logging.error(f"❌ Error in prediction cycle: {e}", exc_info=True)
        
        self._update_entities(values)
        
        logging.info("\n" + "=" * 60)
        logging.info(f"FINAL VALUES TO BE SENT:")
        logging.info(f"  Time to rain: {values['time']} minutes")
        logging.info(f"  Distance: {values['distance']} km")
        logging.info(f"  Speed: {values['speed']} km/h")
        logging.info(f"  Direction: {values['direction']}°")
        logging.info(f"  Bearing: {values['bearing']}°")
        logging.info("=" * 60 + "\n")