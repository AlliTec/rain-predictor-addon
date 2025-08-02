# Rain Predictor Home Assistant Addon

Advanced rain prediction using radar image analysis, converted from AppDaemon to a standalone Home Assistant addon.

## Features

- 🌧️ **Advanced Rain Prediction** - Analyzes radar images to predict rain arrival
- 📍 **Location-Based** - Tracks rain cells moving toward your specific location
- 🎯 **Directional Analysis** - Only predicts rain from cells actually approaching you
- 🏠 **Home Assistant Integration** - Updates input_number entities automatically
- 🌐 **Web UI** - Easy configuration through a beautiful web interface
- 🔧 **Highly Configurable** - Tune analysis parameters for your region
- 📊 **Multiple Data Points** - Time, distance, speed, direction, and bearing
- 🐛 **Debug Mode** - Save radar images and detailed logging

## Installation

### Method 1: Add Repository (Recommended)

1. **Add the addon repository** to Home Assistant:
   - Go to **Settings** → **Add-ons** → **Add-on Store**
   - Click the **⋮** menu → **Repositories**
   - Add: `https://github.com/yourusername/rain-predictor-addon`

2. **Install the addon**:
   - Find "Rain Predictor" in the addon store
   - Click **Install**

### Method 2: Local Installation

1. **Copy addon files** to your Home Assistant:
   ```
   /config/addons_local/rain_predictor/
   ├── config.yaml
   ├── Dockerfile
   ├── requirements.txt
   ├── run.sh
   ├── rain_predictor.py
   ├── web_ui.py
   └── templates/
       └── index.html
   ```

2. **Refresh** the addon store and install "Rain Predictor"

## Setup

### 1. Create Required Input Numbers

Add these to your `configuration.yaml`:

```yaml
input_number:
  rain_arrival_minutes:
    name: "Rain Arrival Time"
    min: 0
    max: 1440
    step: 1
    unit_of_measurement: "min"
    icon: mdi:timer-outline
    
  rain_prediction_distance:
    name: "Rain Cell Distance"
    min: 0
    max: 500
    step: 0.1
    unit_of_measurement: "km"
    icon: mdi:map-marker-distance
    
  rain_prediction_speed:
    name: "Rain Cell Speed"
    min: 0
    max: 200
    step: 0.1
    unit_of_measurement: "km/h"
    icon: mdi:speedometer
    
  rain_cell_direction:
    name: "Rain Cell Direction"
    min: -1
    max: 360
    step: 0.1
    unit_of_measurement: "°"
    icon: mdi:compass-outline
    
  bearing_to_rain_cell:
    name: "Bearing to Rain Cell"
    min: -1
    max: 360
    step: 0.1
    unit_of_measurement: "°"
    icon: mdi:compass
```

### 2. Configure the Addon

1. **Start the addon** (it will create default configuration)
2. **Open the Web UI** at `http://your-ha-ip:8099`
3. **Configure your settings**:
   - **Location**: Enter your exact latitude/longitude
   - **Entities**: Specify the input_number entity IDs
   - **Analysis**: Tune thresholds for your region
   - **Image**: Adjust radar parameters if needed

### 3. Key Configuration Parameters

#### Location Settings
- **Latitude/Longitude**: Your exact location in decimal degrees
- **Update Interval**: How often to check for rain (3-10 minutes recommended)

#### Analysis Settings
- **Rain Threshold**: Pixel intensity to consider as rain (75-100 typical)
- **Arrival Angle**: Degrees ± for approach detection (30-60° recommended)
- **Lat/Lon Range**: Degrees covered by radar image (tune based on zoom level)

#### Image Settings
- **Size**: 256x256 recommended for good detail vs performance
- **Zoom**: 8-10 for regional view, higher for local detail
- **Color Scheme**: 3 (default RainViewer scheme)

## Usage

### Dashboard Cards

Add these to your dashboard to display predictions:

```yaml
# Rain Arrival Card
- type: entities
  title: Rain Prediction
  entities:
    - entity: input_number.rain_arrival_minutes
      name: Time to Rain
      icon: mdi:weather-rainy
    - entity: input_number.rain_prediction_distance
      name: Distance
      icon: mdi:map-marker-distance
    - entity: input_number.rain_prediction_speed
      name: Speed
      icon: mdi:speedometer
    - entity: input_number.rain_cell_direction
      name: Direction
      icon: mdi:compass-outline
    - entity: input_number.bearing_to_rain_cell
      name: Bearing
      icon: mdi:compass

# Rain Status Card
- type: conditional
  conditions:
    - entity: input_number.rain_arrival_minutes
      state_not: "999"
  card:
    type: markdown
    content: >
      ## ⛈️ Rain Incoming!
      
      **{{ states('input_number.rain_arrival_minutes') | int }}** minutes away
      
      **{{ states('input_number.rain_prediction_distance') }}** km distance
      
      Moving at **{{ states('input_number.rain_prediction_speed') }}** km/h
```

### Automations

Create automations based on rain predictions:

```yaml
# Rain Alert Automation
- alias: "Rain Arriving Soon"
  trigger:
    - platform: numeric_state
      entity_id: input_number.rain_arrival_minutes
      below: 30
      above: 0
  condition:
    - condition: numeric_state
      entity_id: input_number.rain_arrival_minutes
      below: 999  # Not the "no rain" value
  action:
    - service: notify.mobile_app_your_phone
      data:
        title: "🌧️ Rain Alert"
        message: >
          Rain expected in {{ states('input_number.rain_arrival_minutes') | int }} minutes!
          Distance: {{ states('input_number.rain_prediction_distance') }}km
        data:
          tag: "rain_alert"
          
# Close Windows When Rain Detected
- alias: "Close Windows - Rain Detected"
  trigger:
    - platform: numeric_state
      entity_id: input_number.rain_arrival_minutes
      equals: 0
  action:
    - service: notify.family
      data:
        message: "Rain detected at location! Close windows and bring in laundry."
    # Add your specific window/device controls here
```

## Understanding the Data

### Entity Values

| Entity | Value | Meaning |
|--------|-------|---------|
| Time to Rain | 0 | Rain is currently at your location |
| Time to Rain | 1-999 | Minutes until rain arrives |
| Time to Rain | 999 | No rain predicted (default) |
| Direction | 0-360° | Direction the rain cell is moving |
| Direction | -1 | No direction data available |
| Bearing | 0-360° | Direction from you TO the rain cell |
| Bearing | -1 | No bearing data available |

### Compass Directions

- **N (0°)**: North
- **E (90°)**: East  
- **S (180°)**: South
- **W (270°)**: West

## Troubleshooting

### Common Issues

1. **No predictions showing**:
   - Check latitude/longitude are correct
   - Verify internet connection for radar data
   - Check logs for API errors

2. **Inaccurate predictions**:
   - Tune `lat_range_deg` and `lon_range_deg` for your zoom level
   - Adjust `rain_threshold` for your region's radar sensitivity
   - Modify `arrival_angle_threshold` (try 30-60°)

3. **Configuration not saving**:
   - Check Home Assistant supervisor token permissions
   - Verify input_number entities exist
   - Check addon logs for errors

### Debug Mode

Enable debug mode to troubleshoot:

1. Set **Log Level** to "Debug" in Web UI
2. Enable **Save Images** to see radar tiles
3. Check logs: **Settings** → **Add-ons** → **Rain Predictor** → **Log**
4. Debug images saved to `/config/share/rain_predictor_debug/`

### Log Analysis

Check addon logs for these indicators:

- `✓ RAIN DETECTED AT CURRENT LOCATION!` - Rain is present
- `Track X is approaching us` - Rain cell moving toward you
- `No rain cells are moving toward your location` - Safe for now
- `HTTP 4xx/5xx errors` - API connectivity issues

## Advanced Configuration

### Fine-Tuning for Your Region

Different regions may need different settings:

**Australia/Tropical**:
```yaml
rain_threshold: 85
arrival_angle_threshold_deg: 45
lat_range_deg: 1.8
lon_range_deg: 1.99
```

**Europe/Temperate**:
```yaml
rain_threshold: 70
arrival_angle_threshold_deg: 60
lat_range_deg: 1.5
lon_range_deg: 2.0
```

**US/Continental**:
```yaml
rain_threshold: 80
arrival_angle_threshold_deg: 45
lat_range_deg: 2.0
lon_range_deg: 2.5
```

### API Alternatives

While designed for RainViewer, you can use compatible APIs:
- `https://api.rainviewer.com/public/weather-maps.json` (Default)
- Other weather services with similar radar tile APIs

## Contributing

Found a bug or want to improve the addon? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Original AppDaemon version inspired the core prediction algorithm
- RainViewer API provides the radar data
- Home Assistant community for feedback and testing

## Support

- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussion**: Home Assistant Community Forum
- 📧 **Contact**: Open an issue for questions

---

**Note**: This addon provides weather predictions based on radar analysis. Actual weather conditions may vary. Always use official weather services for critical decisions.