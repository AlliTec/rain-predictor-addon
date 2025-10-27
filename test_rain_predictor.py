#!/usr/bin/env python3
"""
Diagnostic script to test Rain Predictor configuration
Run this locally to test your settings before deploying
"""

import requests
import json
from PIL import Image
import numpy as np
from scipy.ndimage import label
import io

# YOUR CONFIGURATION - Copy from config.yaml
CONFIG = {
    "latitude": -24.981262201681634,
    "longitude": 151.86545522467864,
    "api_url": "https://api.rainviewer.com/public/weather-maps.json",
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
    }
}

def test_api_connection():
    """Test RainViewer API connection"""
    print("\n" + "="*60)
    print("TEST 1: API Connection")
    print("="*60)
    
    try:
        response = requests.get(CONFIG['api_url'], timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("✅ API connection successful")
        print(f"   Host: {data.get('host', 'unknown')}")
        
        past_frames = data.get('radar', {}).get('past', [])
        print(f"   Past frames available: {len(past_frames)}")
        
        if past_frames:
            print(f"   Oldest frame: {past_frames[0].get('time', 'unknown')}")
            print(f"   Latest frame: {past_frames[-1].get('time', 'unknown')}")
            return data
        else:
            print("❌ No past frames available!")
            return None
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return None

def test_image_download(api_data):
    """Test downloading a radar image"""
    print("\n" + "="*60)
    print("TEST 2: Image Download")
    print("="*60)
    
    try:
        past_frames = api_data['radar']['past']
        latest_frame = past_frames[-1]
        
        frame_path = latest_frame['path']
        api_host = api_data['host']
        
        lat = CONFIG['latitude']
        lon = CONFIG['longitude']
        size = CONFIG['image_settings']['size']
        zoom = CONFIG['image_settings']['zoom']
        color = CONFIG['image_settings']['color_scheme']
        opts = CONFIG['image_settings']['options']
        
        url = f"{api_host}{frame_path}/{size}/{zoom}/{lat}/{lon}/{color}/{opts}.png"
        
        print(f"Downloading: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        print("✅ Image download successful")
        print(f"   Size: {len(response.content)} bytes")
        
        # Try to open image
        img = Image.open(io.BytesIO(response.content))
        print(f"   Image dimensions: {img.size}")
        print(f"   Image mode: {img.mode}")
        
        return response.content
        
    except Exception as e:
        print(f"❌ Image download failed: {e}")
        return None

def test_rain_detection(image_data):
    """Test rain detection in image"""
    print("\n" + "="*60)
    print("TEST 3: Rain Detection")
    print("="*60)
    
    try:
        img = Image.open(io.BytesIO(image_data)).convert('L')
        img_array = np.array(img)
        
        print(f"Image array shape: {img_array.shape}")
        print(f"Pixel value range: {np.min(img_array)} to {np.max(img_array)}")
        print(f"Mean pixel value: {np.mean(img_array):.2f}")
        
        threshold = CONFIG['analysis_settings']['rain_threshold']
        print(f"\nUsing threshold: {threshold}")
        
        rain_pixels = img_array > threshold
        rain_count = np.sum(rain_pixels)
        total_pixels = img_array.size
        rain_percentage = (rain_count / total_pixels) * 100
        
        print(f"Rain pixels: {rain_count} / {total_pixels} ({rain_percentage:.2f}%)")
        
        if rain_count == 0:
            print("❌ No rain detected above threshold!")
            print("   Consider lowering the threshold value")
            
            # Show distribution
            print("\n   Pixel value distribution:")
            for thresh in [30, 50, 75, 100, 150, 200]:
                count = np.sum(img_array > thresh)
                pct = (count / total_pixels) * 100
                print(f"     Above {thresh:3d}: {count:6d} pixels ({pct:5.2f}%)")
            
            return False
        
        # Label connected components
        labeled_image, num_labels = label(rain_pixels)
        print(f"\n✅ Found {num_labels} rain cell(s)")
        
        # Analyze each cell
        img_height, img_width = img_array.shape
        lat_inc = CONFIG['analysis_settings']['lat_range_deg'] / img_height
        lon_inc = CONFIG['analysis_settings']['lon_range_deg'] / img_width
        center_y = (img_height - 1) / 2.0
        center_x = (img_width - 1) / 2.0
        
        print("\n   Cell details:")
        for i in range(1, min(num_labels + 1, 11)):  # Show first 10
            y_coords, x_coords = np.where(labeled_image == i)
            size = len(y_coords)
            
            if size < 5:
                continue
            
            centroid_x, centroid_y = np.mean(x_coords), np.mean(y_coords)
            intensity = np.mean(img_array[y_coords, x_coords])
            
            # Calculate lat/lon offset
            lat_offset = (center_y - centroid_y) * lat_inc
            lon_offset = (centroid_x - center_x) * lon_inc
            est_lat = CONFIG['latitude'] + lat_offset
            est_lon = CONFIG['longitude'] + lon_offset
            
            print(f"   Cell {i}:")
            print(f"     Size: {size} pixels")
            print(f"     Avg intensity: {intensity:.1f}")
            print(f"     Centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
            print(f"     Est. location: ({est_lat:.4f}, {est_lon:.4f})")
            
            # Calculate distance from center
            dist_from_center = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
            print(f"     Distance from center: {dist_from_center:.1f} pixels")
        
        # Check center
        print(f"\n   Center location check:")
        check_radius = 5
        y_start = max(0, int(center_y - check_radius))
        y_end = min(img_height, int(center_y + check_radius + 1))
        x_start = max(0, int(center_x - check_radius))
        x_end = min(img_width, int(center_x + check_radius + 1))
        
        center_area = img_array[y_start:y_end, x_start:x_end]
        center_max = np.max(center_area)
        
        print(f"     Max value in center: {center_max}")
        if center_max > threshold:
            print("     ✅ RAIN AT YOUR LOCATION!")
        else:
            print("     ✅ No rain at your location")
        
        return True
        
    except Exception as e:
        print(f"❌ Rain detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration values"""
    print("\n" + "="*60)
    print("TEST 4: Configuration Check")
    print("="*60)
    
    print("Current configuration:")
    print(f"  Location: ({CONFIG['latitude']}, {CONFIG['longitude']})")
    print(f"  Image size: {CONFIG['image_settings']['size']}")
    print(f"  Zoom level: {CONFIG['image_settings']['zoom']}")
    print(f"  Rain threshold: {CONFIG['analysis_settings']['rain_threshold']}")
    print(f"  Lat range: {CONFIG['analysis_settings']['lat_range_deg']}°")
    print(f"  Lon range: {CONFIG['analysis_settings']['lon_range_deg']}°")
    
    # Calculate coverage
    size = CONFIG['image_settings']['size']
    lat_range = CONFIG['analysis_settings']['lat_range_deg']
    lon_range = CONFIG['analysis_settings']['lon_range_deg']
    
    # Approximate km per degree at this latitude
    lat = CONFIG['latitude']
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
    
    coverage_lat_km = lat_range * km_per_deg_lat
    coverage_lon_km = lon_range * km_per_deg_lon
    
    print(f"\n  Approximate coverage:")
    print(f"    Lat: {coverage_lat_km:.1f} km")
    print(f"    Lon: {coverage_lon_km:.1f} km")
    print(f"    Pixels per km (lat): {size / coverage_lat_km:.2f}")
    print(f"    Pixels per km (lon): {size / coverage_lon_km:.2f}")
    
    print("\n✅ Configuration loaded successfully")

def main():
    print("="*60)
    print("RAIN PREDICTOR DIAGNOSTIC TEST")
    print("="*60)
    
    # Test configuration
    test_configuration()
    
    # Test API
    api_data = test_api_connection()
    if not api_data:
        print("\n❌ Cannot proceed without API data")
        return
    
    # Test image download
    image_data = test_image_download(api_data)
    if not image_data:
        print("\n❌ Cannot proceed without image data")
        return
    
    # Test rain detection
    rain_detected = test_rain_detection(image_data)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if rain_detected:
        print("✅ All tests passed!")
        print("   The configuration should work in the addon")
    else:
        print("⚠️  Tests completed but no rain detected")
        print("   This could be normal if there's no rain currently")
        print("   Or you may need to lower the rain_threshold value")
    
    print("\nRecommendations:")
    if rain_detected:
        print("  - Deploy to Home Assistant")
        print("  - Enable debug logging initially")
        print("  - Monitor logs during rain events")
    else:
        print("  - Try lowering rain_threshold to 50 or lower")
        print("  - Run this test again during known rain")
        print("  - Check that lat_range_deg and lon_range_deg are correct")

if __name__ == "__main__":
    main()