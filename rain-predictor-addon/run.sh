#!/usr/bin/with-contenv bash

# Import Bashio
source /usr/lib/bashio/bashio.sh

# Set log level
LOG_LEVEL=$(bashio::config 'debug.log_level' 'INFO')
export LOG_LEVEL

bashio::log.info "Starting Rain Predictor Addon..."

# Check if Home Assistant is available
if bashio::supervisor.ping; then
    bashio::log.info "Home Assistant supervisor is available"
else
    bashio::log.warning "Home Assistant supervisor not available, some features may not work"
fi

# Validate required configuration
if ! bashio::config.exists 'latitude' || ! bashio::config.exists 'longitude'; then
    bashio::log.fatal "Latitude and longitude must be configured!"
    exit 1
fi

if ! bashio::config.exists 'entities.time'; then
    bashio::log.fatal "At least the time entity must be configured!"
    exit 1
fi

# Start the web UI in background
bashio::log.info "Starting web UI on port 8099..."
python3 /web_ui.py &
WEB_UI_PID=$!

# Start the main rain predictor
bashio::log.info "Starting rain predictor service..."
python3 /rain_predictor.py

# If main process exits, kill web UI

kill $WEB_UI_PID 2>/dev/null || true
