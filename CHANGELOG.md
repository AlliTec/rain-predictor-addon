# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.65] - 2026-09-25
### Fixed
- Add-on build failure in Home Assistant: removed unneeded `gcc`, `musl-dev` and `python3-dev` from the Dockerfile, which conflicted with the `musl` version in the base image (`apk: unable to select packages`).
### Changed
- Synchronized version numbers across `config.yaml`, `Dockerfile` and `rain_predictor.py`.

## [1.1.64] - 2026-07-20
### Added
- Extensive logging to the `createTracker` function in the web UI to help diagnose the missing auto-track marker.
- Additional logging to help diagnose tracking issues.
- `input_number.rain_cell_latitude` and `input_number.rain_cell_longitude` entities to `config.yaml` for tracking rain cell location.
### Changed
- Auto track marker is now a circle around the tracked cell, with updated color and style for better visibility.
- Updated version number in `config.yaml`, `Dockerfile`, and `rain_predictor.py`.
### Fixed
- Radar tile request logic: correctly calculates (x,y) tiles for the user location.
- Coordinate mapping: uses actual tile bounds for pixel-to-lat/lon conversion.
- 'cannot identify image file' errors, by stopping root tile requests.
- Reverted threat detection logic to use bearing from cell to location, while displaying bearing from location to cell in the UI.
- Auto track marker not appearing due to a JavaScript error.
- Rain cell tracking accuracy improved by ensuring cell coordinates are always sent to the UI.
- Auto track marker not tracking the target cell by exposing rain cell coordinates (`rain_cell_latitude`, `rain_cell_longitude`) to Home Assistant entities from `rain_predictor.py`.
- Auto track marker not moving and sitting over the location marker; marker size reduced to half.
- Auto track prediction logic refined to start from the current rain cell location.
- Location marker appearing in incorrect location (Antarctica) by ensuring the user's configured latitude/longitude are passed to the UI.
- Add-on crashing due to `IndentationError` in `web_ui.py`.
- Web UI not displaying metrics (Time to Rain, Distance, Speed, Direction, Bearing) by correctly passing `all_data` to `index.html` and updating `updateDataDisplay` to use direct values.
