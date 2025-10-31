#!/bin/bash

echo "Starting Rain Predictor Addon (Local Development)..."

# Create necessary directories
mkdir -p ./debug_images
mkdir -p ./test_data

# Set environment variables for local testing
export DATA_PATH=./test_data
export SUPERVISOR_TOKEN="dummy_token_for_testing"

# Start web server in background
echo "Starting web UI on port 8099..."
python3 web_ui.py &
WEBSERVER_PID=$!

# Give web server time to start
sleep 2

# Start main rain predictor with error output
echo "Starting rain prediction service..."
python3 rain_predictor.py &
MAIN_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $WEBSERVER_PID $MAIN_PID 2>/dev/null
    wait $WEBSERVER_PID $MAIN_PID 2>/dev/null
    echo "Shutdown complete"
}

# Trap signals
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n

# If one exits, clean up the other
cleanup

exit $?
