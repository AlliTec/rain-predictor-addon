# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.4-debug] - 2025-11-01
### Fixed
- Location marker appearing in incorrect location (Antarctica) by ensuring user's configured latitude/longitude are passed to the UI.

## [1.1.3-debug] - 2025-11-01
### Fixed
- Addon crashing due to `IndentationError` in `web_ui.py`.
- Auto track marker not moving and sitting over location marker.
- Auto track marker size unchanged (now reduced to half).
- Auto track prediction logic refined to start from the current rain cell location.

## [1.1.2-debug] - 2025-11-01
### Fixed
- Auto track marker not tracking the target cell by exposing rain cell coordinates (`rain_cell_latitude`, `rain_cell_longitude`) to Home Assistant entities from `rain_predictor.py`.
### Added
- `input_number.rain_cell_latitude` and `input_number.rain_cell_longitude` entities to `config.yaml` for tracking rain cell location.

## [1.1.1-debug] - 2025-11-01
### Fixed
- Web UI not displaying metrics (Time to Rain, Distance, Speed, Direction, Bearing) by correctly passing `all_data` to `index.html` and updating `updateDataDisplay` to use direct values.
### Changed
- Updated version number in `config.yaml`, `Dockerfile`, and `rain_predictor.py`.
