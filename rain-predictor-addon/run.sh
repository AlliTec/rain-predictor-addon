#!/usr/bin/with-contenv bashio

bashio::log.info "Starting Rain Predictor Addon..."

# Create necessary directories
mkdir -p /share/rain_predictor_debug

# Start web server in background
bashio::log.info "Starting web UI on port 8099..."
python3 /app/web_ui.py &
WEBSERVER_PID=$!

# Give web server time to start
sleep 2

# Start main rain predictor with error output
bashio::log.info "Starting rain prediction service..."
python3 /app/rain_predictor.py 2>&1 | while read line; do
    bashio::log.info "$line"
done &
MAIN_PID=$!

# Function to handle shutdown
cleanup() {
    bashio::log.info "Shutting down services..."
    kill $WEBSERVER_PID $MAIN_PID 2>/dev/null
    wait $WEBSERVER_PID $MAIN_PID 2>/dev/null
    bashio::log.info "Shutdown complete"
}

# Trap signals
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n

# If one exits, clean up the other
cleanup

exit $?