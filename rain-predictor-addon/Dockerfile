ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.7
FROM $BUILD_FROM

# Install Python and required packages
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scipy \
    py3-pillow \
    py3-requests \
    py3-flask \
    py3-flask-cors \
    gcc \
    musl-dev \
    python3-dev

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Copy addon files
COPY run.sh /
COPY rain_predictor.py /
COPY web_ui.py /
COPY templates/ /templates/
COPY static/ /static/

# Make run script executable
RUN chmod a+x /run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8099/health || exit 1

# Labels
LABEL \
  io.hass.name="Rain Predictor" \
  io.hass.description="Advanced rain prediction using radar image analysis" \
  io.hass.arch="armhf|aarch64|amd64|armv7|i386" \
  io.hass.type="addon" \
  io.hass.version="1.0.0"


CMD [ "/run.sh" ]
