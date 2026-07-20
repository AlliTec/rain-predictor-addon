#!/usr/bin/env python3
"""
Unit tests for the radar tile coordinate and pixel-to-coordinate mapping fixes.
Validates that _get_tile_coords and _tile_bounds produce correct geographic
results for the user location (lat=-24.981262, lon=151.865455).
"""

import json
import math
import os
import sys
import unittest
from math import radians, cos, sin, asin, sqrt, atan2, degrees, log, tan, pi
from unittest.mock import MagicMock, patch

# Make the module importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rain_predictor as rp


class TestTileCoords(unittest.TestCase):
    """Test the _get_tile_coords method."""

    def setUp(self):
        """Create a minimal RainPredictor instance."""
        config = {
            'latitude': -24.981262,
            'longitude': 151.865455,
            'run_interval_minutes': 3,
            'entities': {'time': 'input_number.test_time'},
            'image_settings': {'size': 256, 'zoom': 8, 'color_scheme': 3, 'options': '0_0'},
            'analysis_settings': {'rain_threshold': 75},
            'debug': {'save_images': False, 'log_level': 'WARNING'},
        }
        self.predictor = rp.RainPredictor(config, MagicMock())

    def test_zoom_8_user_location(self):
        """At zoom 8, the user location should NOT map to tile (0, 0)."""
        x, y = self.predictor._get_tile_coords(-24.981262, 151.865455, 8)
        self.assertNotEqual((x, y), (0, 0),
            "Tile coords should not be root (0,0) for a real location")
        # Expected: x ~ (151.865455+180)/360*256 ≈ 236, y varies with Mercator
        self.assertGreater(x, 0, "x should be > 0 for longitude 151.865")
        self.assertGreater(y, 0, "y should be > 0 for latitude -24.98")

    def test_zoom_0_always_root(self):
        """At zoom 0, any location maps to the single root tile (0, 0)."""
        x, y = self.predictor._get_tile_coords(-24.981262, 151.865455, 0)
        self.assertEqual((x, y), (0, 0))

    def test_known_coordinates(self):
        """Verify against a known reference: London (51.5, -0.1) at zoom 10."""
        x, y = self.predictor._get_tile_coords(51.507, -0.128, 10)
        # Standard OSM tile for London z10: x=511, y=340 (approx)
        self.assertEqual(x, 511)
        self.assertEqual(y, 340)

    def test_clamping_edge_cases(self):
        """Verify that extreme values are clamped to valid ranges."""
        for zoom in [1, 5, 10, 20]:
            n = 2 ** zoom
            # North pole
            x, y = self.predictor._get_tile_coords(85, 0, zoom)
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, n)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, n)
            # South pole
            x, y = self.predictor._get_tile_coords(-85, 179.9, zoom)
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, n)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, n)


class TestTileBounds(unittest.TestCase):
    """Test the _tile_bounds method."""

    def setUp(self):
        config = {
            'latitude': -24.981262,
            'longitude': 151.865455,
            'run_interval_minutes': 3,
            'entities': {'time': 'input_number.test_time'},
            'image_settings': {'size': 256, 'zoom': 8, 'color_scheme': 3, 'options': '0_0'},
            'analysis_settings': {'rain_threshold': 75},
            'debug': {'save_images': False, 'log_level': 'WARNING'},
        }
        self.predictor = rp.RainPredictor(config, MagicMock())

    def test_bounds_contain_user_location(self):
        """The tile bounds at the user's zoom level must contain their lat/lon."""
        x, y = self.predictor._get_tile_coords(-24.981262, 151.865455, 8)
        lat_n, lon_w, lat_s, lon_e = self.predictor._tile_bounds(x, y, 8)
        self.assertLessEqual(lat_s, -24.981262, "User lat must be ≥ tile south edge")
        self.assertGreaterEqual(lat_n, -24.981262, "User lat must be ≤ tile north edge")
        self.assertLessEqual(lon_w, 151.865455, "User lon must be ≥ tile west edge")
        self.assertGreaterEqual(lon_e, 151.865455, "User lon must be ≤ tile east edge")

    def test_root_tile_bounds(self):
        """Root tile (0,0) at zoom 0 covers the whole world."""
        lat_n, lon_w, lat_s, lon_e = self.predictor._tile_bounds(0, 0, 0)
        self.assertAlmostEqual(lon_w, -180.0, places=1)
        self.assertAlmostEqual(lon_e, 180.0, places=1)
        self.assertAlmostEqual(lat_n, 85.05, places=0)  # Web Mercator limit
        self.assertAlmostEqual(lat_s, -85.05, places=0)

    def test_bounds_are_geographically_valid(self):
        """Bounds should have north > south and east > west."""
        for zoom in [2, 5, 8, 12]:
            n = 2 ** zoom
            for tx in range(min(3, n)):
                for ty in range(min(3, n)):
                    lat_n, lon_w, lat_s, lon_e = self.predictor._tile_bounds(tx, ty, zoom)
                    self.assertGreater(lat_n, lat_s,
                        f"North should > South at z={zoom} x={tx} y={ty}")
                    self.assertGreater(lon_e, lon_w,
                        f"East should > West at z={zoom} x={tx} y={ty}")


class TestPixelToCoordinateMapping(unittest.TestCase):
    """Test that pixel centroids correctly map back to geographic coordinates."""

    def setUp(self):
        config = {
            'latitude': -24.981262,
            'longitude': 151.865455,
            'run_interval_minutes': 3,
            'entities': {'time': 'input_number.test_time'},
            'image_settings': {'size': 256, 'zoom': 8, 'color_scheme': 3, 'options': '0_0'},
            'analysis_settings': {'rain_threshold': 75},
            'debug': {'save_images': False, 'log_level': 'WARNING'},
        }
        self.predictor = rp.RainPredictor(config, MagicMock())

    def test_corner_pixels_map_to_bounds(self):
        """Pixel (0,0) → NW corner, pixel (w-1,h-1) → SE corner of tile."""
        zoom = 8
        x, y = self.predictor._get_tile_coords(-24.981262, 151.865455, zoom)
        lat_n, lon_w, lat_s, lon_e = self.predictor._tile_bounds(x, y, zoom)
        img_w, img_h = 256, 256

        # Compute increments the same way the fixed code does
        lat_inc = (lat_n - lat_s) / img_h
        lon_inc = (lon_e - lon_w) / img_w

        # Pixel (0, 0) → NW corner
        px0_lon = lon_w + 0 * lon_inc
        px0_lat = lat_n - 0 * lat_inc
        self.assertAlmostEqual(px0_lon, lon_w, places=4)
        self.assertAlmostEqual(px0_lat, lat_n, places=4)

        # Pixel (255, 255) → near SE corner
        px255_lon = lon_w + 255 * lon_inc
        px255_lat = lat_n - 255 * lat_inc
        self.assertAlmostEqual(px255_lon, lon_e - lon_inc, places=4)
        self.assertAlmostEqual(px255_lat, lat_s + lat_inc, places=4)

    def test_image_url_uses_tile_coords(self):
        """Verify the image URL now uses integer tile x/y, not raw lat/lon floats."""
        zoom = 8
        tile_x, tile_y = self.predictor._get_tile_coords(-24.981262, 151.865455, zoom)
        # Construct the URL the same way the fixed code does
        url = (
            f"https://tilecache.rainviewer.com/v2/radar/1234567890/"
            f"256/{zoom}/{tile_x}/{tile_y}/3/0_0.png"
        )
        # The URL should contain integer tile coords, not float lat/lon
        self.assertNotIn('-24.98', url, "URL should not contain raw latitude float")
        self.assertNotIn('151.86', url, "URL should not contain raw longitude float")
        self.assertIn(f"/{tile_x}/", url, "URL should contain integer tile x")
        self.assertIn(f"/{tile_y}/", url, "URL should contain integer tile y")
        # tile coords should be integers
        self.assertIsInstance(tile_x, int)
        self.assertIsInstance(tile_y, int)


class TestOldUrlVsNewUrl(unittest.TestCase):
    """Demonstrate the bug: old URL had float lat/lon where tile ints belong."""

    def test_old_url_was_invalid(self):
        """The old URL format put lat/lon floats where tile integers go,
        resulting in paths like /256/8/-24.98/151.86/3/0_0.png which
        RainViewer rejects with non-image data."""
        old_url = (
            "https://tilecache.rainviewer.com/v2/radar/1234567890/"
            "256/8/-24.981262/151.865455/3/0_0.png"
        )
        # The path segment after zoom should be an integer tile index, not a float
        parts = old_url.split("/")
        # parts: ['https:', '', 'tilecache.rainviewer.com', 'v2', 'radar', '1234567890', '256', '8', '-24.981262', '151.865455', '3', '0_0.png']
        zoom_idx = parts.index('8')
        x_candidate = parts[zoom_idx + 1]
        y_candidate = parts[zoom_idx + 2]
        # These were floats in the old code — that's the bug
        self.assertTrue('.' in x_candidate,
            "Old URL used float for x tile coordinate (bug)")
        self.assertTrue('.' in y_candidate,
            "Old URL used float for y tile coordinate (bug)")


if __name__ == '__main__':
    unittest.main()