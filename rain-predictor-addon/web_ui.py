#!/usr/bin/env python3
# /rain-predictor-addon/web_ui.py

import os
import io
import json
import math
import time
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import requests
from PIL import Image
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ========== Settings ==========
# Radar sampling/analysis settings for manual selection
RAIN_THRESHOLD = 65          # 60–75 works well; raise if motion looks noisy
MAX_TILES = 8                # safety cap on tile grid size for a click
META_TTL_SECONDS = 120       # cache RainViewer meta for 2 minutes
TILE_COLOR_SCHEME = 2        # grayscale is derived from this color scheme when downloading tiles

# Persistence
DATA_PATH = os.environ.get("DATA_PATH", "/data")
OPTIONS_PATH = os.path.join(DATA_PATH, "options.json")

# ========== Home Assistant helpers ==========
class HomeAssistantAPI:
    def __init__(self):
        self.base_url = "http://supervisor/core/api"

    def _headers(self):
        token = os.environ.get("SUPERVISOR_TOKEN")
        if not token:
            logging.error("SUPERVISOR_TOKEN not found!")
            return None
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def call_service(self, service: str, entity_id: str, value: Any) -> bool:
        headers = self._headers()
        if not headers:
            return False
        try:
            domain, service_name = service.split("/", 1)
            url = f"{self.base_url}/services/{domain}/{service_name}"
            data = {"entity_id": entity_id, "value": value}
            r = requests.post(url, headers=headers, json=data, timeout=10)
            if r.status_code >= 400:
                logging.error(f"Error calling service {service} for {entity_id}: {r.status_code} {r.text}")
                return False
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Exception calling service {service} for {entity_id}: {e}")
            return False

    def get_state(self, entity_id: str, default: Any = None) -> Any:
        token = os.environ.get("SUPERVISOR_TOKEN")
        if not token:
            return default
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://supervisor/core/api/states/{entity_id}"
        try:
            r = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            return r.json().get("state", default)
        except requests.exceptions.RequestException:
            return default

ha_api = HomeAssistantAPI()

def get_all_data():
    data = {
        "time_to_rain": ha_api.get_state("input_number.rain_arrival_minutes", "--"),
        "distance": ha_api.get_state("input_number.rain_prediction_distance", "--"),
        "speed": ha_api.get_state("input_number.rain_prediction_speed", "--"),
        "direction": ha_api.get_state("input_number.rain_cell_direction", "N/A"),
        "bearing": ha_api.get_state("input_number.bearing_to_rain_cell", "N/A"),
        "latitude": ha_api.get_state("input_number.rain_prediction_latitude", -24.98),
        "longitude": ha_api.get_state("input_number.rain_prediction_longitude", 151.86),
    }

    # Fallback for local development
    if data["time_to_rain"] == "--":
        try:
            data_path = os.environ.get("DATA_PATH", "/data")
            prediction_file = os.path.join(data_path, "prediction.json")
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            data["time_to_rain"] = prediction_data.get("time", "--")
            data["distance"] = prediction_data.get("distance", "--")
            data["speed"] = prediction_data.get("speed", "--")
            data["direction"] = prediction_data.get("direction", "N/A")
            data["bearing"] = prediction_data.get("bearing", "N/A")
            logging.debug("Loaded prediction data from file")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.debug(f"Could not load prediction data from file: {e}")
            
    return data

# ========== Options.json persistence ==========
def read_options_latlon(default_lat=-24.98, default_lng=151.86) -> Tuple[float, float]:
    try:
        if os.path.exists(OPTIONS_PATH):
            with open(OPTIONS_PATH, "r") as f:
                cfg = json.load(f)
                lat = float(cfg.get("latitude", default_lat))
                lng = float(cfg.get("longitude", default_lng))
                return lat, lng
    except Exception as e:
        logging.warning(f"read_options_latlon failed: {e}")
    return float(default_lat), float(default_lng)

def write_options_latlon(lat: float, lng: float) -> None:
    try:
        os.makedirs(DATA_PATH, exist_ok=True)
        cfg = {}
        if os.path.exists(OPTIONS_PATH):
            try:
                with open(OPTIONS_PATH, "r") as f:
                    cfg = json.load(f) or {}
            except Exception:
                cfg = {}
        cfg["latitude"] = float(lat)
        cfg["longitude"] = float(lng)
        with open(OPTIONS_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        logging.info(f"Wrote options.json with latitude={lat}, longitude={lng}")
    except Exception as e:
        logging.error(f"Failed to write options.json: {e}")

# ========== RainViewer helpers ==========
_meta_cache: Dict[str, Any] = {"data": None, "ts": 0.0}

def _fetch_rainviewer_meta() -> Dict[str, Any]:
    now = time.time()
    if _meta_cache["data"] and (now - _meta_cache["ts"] < META_TTL_SECONDS):
        return _meta_cache["data"]
    r = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=8)
    r.raise_for_status()
    meta = r.json()
    _meta_cache["data"] = meta
    _meta_cache["ts"] = now
    return meta

def _download_tile(ts: int, z: int, x: int, y: int, color: int = TILE_COLOR_SCHEME) -> Image.Image:
    url = f"https://tilecache.rainviewer.com/v2/radar/{ts}/256/{z}/{x}/{y}/{color}/1_1.png"
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        return Image.new("L", (256, 256), 0)
    # Convert to grayscale
    return Image.open(io.BytesIO(r.content)).convert("L")

def _lonlat_to_tilexy(lon: float, lat: float, z: int) -> Tuple[int, int]:
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def _tilexy_bounds(west: float, south: float, east: float, north: float, z: int) -> Tuple[int, int, int, int]:
    x1, y2 = _lonlat_to_tilexy(west, south, z)
    x2, y1 = _lonlat_to_tilexy(east, north, z)
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    return x_min, x_max, y_min, y_max

def _phase_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    # a, b: 2D arrays (float32) same shape
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.maximum(1e-6, np.abs(R))
    r = np.fft.irfft2(R, s=a.shape)
    maxpos = np.unravel_index(np.argmax(r), r.shape)
    shifts = np.array(maxpos, dtype=np.float32)
    h, w = a.shape
    if shifts[0] > h // 2:
        shifts[0] -= h
    if shifts[1] > w // 2:
        shifts[1] -= w
    dy, dx = shifts  # rows (y), cols (x)
    return float(dy), float(dx), float(r.max())

def _avg_motion_over_view(meta: Dict[str, Any], bounds: Dict[str, float], zoom_for_tiles: Optional[int] = None) -> Optional[Tuple[float, float]]:
    past = meta.get("radar", {}).get("past") or []
    if len(past) < 3:
        return None
    frames = past[-4:]  # last 3–4 frames are most relevant

    # Choose a moderate zoom to keep tile count small
    z = int(zoom_for_tiles or 6)
    z = max(3, min(8, z))

    x_min, x_max, y_min, y_max = _tilexy_bounds(bounds["west"], bounds["south"], bounds["east"], bounds["north"], z)
    # Cap the grid size
    if (x_max - x_min + 1) * (y_max - y_min + 1) > MAX_TILES * MAX_TILES:
        step = math.ceil((x_max - x_min + 1) / MAX_TILES)
        x_idxs = list(range(x_min, x_max + 1, step))
        y_idxs = list(range(y_min, y_max + 1, step))
    else:
        x_idxs = list(range(x_min, x_max + 1))
        y_idxs = list(range(y_min, y_max + 1))

    stacks: List[np.ndarray] = []
    for f in frames:
        tiles: List[np.ndarray] = []
        for y in y_idxs:
            row = []
            for x in x_idxs:
                im = _download_tile(f["time"], z, x, y)
                row.append(np.asarray(im, dtype=np.float32))
            tiles.append(np.concatenate(row, axis=1))
        canvas = np.concatenate(tiles, axis=0) if tiles else None
        if canvas is None:
            return None
        # Threshold to rain mask
        mask = (canvas >= RAIN_THRESHOLD).astype(np.float32)
        stacks.append(mask)

    shifts: List[Tuple[float, float]] = []
    peaks: List[float] = []
    for i in range(1, len(stacks)):
        dy, dx, peak = _phase_correlation(stacks[i - 1], stacks[i])
        shifts.append((dy, dx))
        peaks.append(peak)
    if not shifts:
        return None

    weights = np.maximum(1e-6, np.array(peaks, dtype=np.float32))
    dy = float(np.average([s[0] for s in shifts], weights=weights))
    dx = float(np.average([s[1] for s in shifts], weights=weights))

    # Pixels -> meters per pixel at this latitude and zoom (WebMercator)
    center_lat = (bounds["north"] + bounds["south"]) / 2.0
    m_per_pixel = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** z)

    # RainViewer "past" frames are ~5 minutes apart
    minutes_per_step = 5.0
    # dy positive = south, so invert to meters north
    meters_y = -dy * m_per_pixel
    meters_x = dx * m_per_pixel
    meters_per_hour_x = meters_x * (60.0 / minutes_per_step)
    meters_per_hour_y = meters_y * (60.0 / minutes_per_step)
    speed_kph = math.hypot(meters_per_hour_x, meters_per_hour_y) / 1000.0

    # Bearing: 0°=N, 90°=E
    bearing = (math.degrees(math.atan2(meters_per_hour_x, meters_per_hour_y)) + 360.0) % 360.0
    return speed_kph, bearing

# ========== Flask routes ==========
@app.route("/")
def index():
    # Prefer persisted options.json; fall back to HA entities; always cast to float
    lat_opt, lng_opt = read_options_latlon()
    # If HA has newer values, you can prefer them; here we keep options.json as the source of truth for persistence
    return render_template("index.html", latitude=float(lat_opt), longitude=float(lng_opt))

@app.route("/api/data")
def api_data():
    return jsonify(get_all_data())

@app.route("/api/set_location", methods=["POST"])
def set_location():
    logging.info("set_location endpoint called")
    data = request.get_json(force=True) or {}
    lat = data.get("latitude") or data.get("lat")
    lon = data.get("longitude") or data.get("lng")
    if lat is None or lon is None:
        logging.error(f"Invalid data received: {data}")
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    lat = float(lat)
    lon = float(lon)
    logging.info(f"Received new location: Lat={lat}, Lon={lon}")

    # Update HA (best-effort)
    lat_ok = ha_api.call_service("input_number/set_value", "input_number.rain_prediction_latitude", lat)
    lon_ok = ha_api.call_service("input_number/set_value", "input_number.rain_prediction_longitude", lon)
    if not (lat_ok and lon_ok):
        logging.warning("Updating HA entities failed or was partial; continuing to persist to options.json")

    # Persist so it survives restarts/reloads
    write_options_latlon(lat, lon)

    return jsonify({"status": "success", "message": "Location updated", "latitude": lat, "longitude": lon}), 200

@app.route("/api/update_config", methods=["POST"])
def update_config():
    logging.info("update_config endpoint called")
    data = request.get_json(force=True) or {}
    lat = data.get("latitude") or data.get("lat")
    lon = data.get("longitude") or data.get("lng")
    if lat is None or lon is None:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    lat = float(lat)
    lon = float(lon)
    try:
        write_options_latlon(lat, lon)
        logging.info(f"Updated addon config with new location: Lat={lat}, Lon={lon}")
        return jsonify({"status": "success", "message": "Config updated"})
    except Exception as e:
        logging.error(f"Failed to update config: {e}")
        return jsonify({"status": "error", "message": "Failed to update config"}), 500

@app.route("/api/manual_selection", methods=["POST"])
def manual_selection():
    """Estimate prevailing rain motion over the current map view and return a vector."""
    logging.info("manual_selection endpoint called")
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    lat = data.get("latitude") or data.get("lat")
    lng = data.get("longitude") or data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "Missing latitude/longitude"}), 400

    try:
        lat = float(lat)
        lng = float(lng)
    except Exception:
        return jsonify({"error": "Invalid latitude/longitude"}), 400

    bounds = data.get("bounds")
    zoom = data.get("zoom")
    # If bounds were not sent, create a small box around the click
    if not bounds:
        pad = 2.0
        bounds = {
            "north": lat + pad,
            "south": lat - pad,
            "east": lng + pad,
            "west": lng - pad,
        }

    # Compute average motion
    speed_kph = 0.0
    direction_deg = 0.0
    try:
        meta = _fetch_rainviewer_meta()
        result = _avg_motion_over_view(meta, bounds, zoom_for_tiles=zoom)
        if result:
            speed_kph, direction_deg = result
        else:
            logging.warning("Motion estimation returned None; falling back to stationary vector")
    except Exception as e:
        logging.exception(f"manual_selection: error computing motion: {e}")

    # Always return 200 so the UI marker appears; speed may be 0 if no rain in view
    return jsonify({
        "center": {"lat": float(lat), "lng": float(lng)},
        "speed": float(max(0.0, speed_kph)),
        "direction": float(direction_deg) % 360.0,
        "intensity": None,
    }), 200

@app.route("/health")
def health_check():
    return "OK", 200

if __name__ == "__main__":
    # Port 8099 to match your addon setup
    app.run(host="0.0.0.0", port=8099, debug=False)