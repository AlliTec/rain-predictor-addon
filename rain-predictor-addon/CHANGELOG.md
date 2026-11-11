## [1.1.67] - 2026-11-11
- How much rain: each radar pixel's palette colour becomes a rain rate (Marshall-Palmer, Z = 200 R^1.6, capped at 100 mm/h); the mean rate of the cell pixels still to cross the property's line, times the hours it rains, is the expected depth. Shown as RAIN on the page; written to the optional `entities.rain_mm` input_number. The colour-to-dBZ steps are an assumption (RainViewer does not publish them): scale with `analysis_settings.rain_mm_calibration` against the rain gauge.

## [1.1.66] - 2026-11-11
- How long the rain lasts: the cell's own footprint is carried along its heading and speed, and the first and last pixels to cross the property's line give the leading-edge and trailing-edge arrival; the difference is the duration. Shown as LASTS on the page; written to the optional `entities.duration` (minutes) and `entities.rain_end` (minutes until the rain has passed) input_numbers when they exist in Home Assistant.

## [1.1.65] - 2026-09-25
- Fixed add-on build failure in Home Assistant: removed unneeded `gcc`, `musl-dev` and `python3-dev` from the Dockerfile, which conflicted with the `musl` version in the base image (`apk: unable to select packages`).
- Synchronized version numbers across `config.yaml`, `Dockerfile` and `rain_predictor.py`.

## [1.1.64] - 2026-07-20
- Fixed radar tile request logic: correctly calculates (x,y) tiles for user location.
- Fixed coordinate mapping: uses actual tile bounds for pixel-to-lat/lon conversion.
- Resolved 'cannot identify image file' errors by stopping root tile requests.

