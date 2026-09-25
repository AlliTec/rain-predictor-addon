## [1.1.65] - 2026-09-25
- Fixed add-on build failure in Home Assistant: removed unneeded `gcc`, `musl-dev` and `python3-dev` from the Dockerfile, which conflicted with the `musl` version in the base image (`apk: unable to select packages`).
- Synchronized version numbers across `config.yaml`, `Dockerfile` and `rain_predictor.py`.

## [1.1.64] - 2026-07-20
- Fixed radar tile request logic: correctly calculates (x,y) tiles for user location.
- Fixed coordinate mapping: uses actual tile bounds for pixel-to-lat/lon conversion.
- Resolved 'cannot identify image file' errors by stopping root tile requests.

