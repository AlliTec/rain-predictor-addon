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
    gcc \
    musl-dev \
    python3-dev

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Create app directory structure
RUN mkdir -p /app/templates /app/static

# Copy addon files
COPY run.sh /
COPY rain_predictor.py /app/
COPY web_ui.py /app/
COPY templates/index.html /app/templates/
COPY static/ /app/static/

# Make run script executable
RUN chmod a+x /run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8099/health || exit 1

# Labels
LABEL \
  io.hass.name="Rain Predictor" \
  io.hass.description="Advanced rain prediction using radar image analysis" \
  io.hass.arch="armhf|aarch64|amd64|armv7|i386" \
  io.hass.type="addon" \
  io.hass.version="1.1.7-debug"

CMD [ "/run.sh" ]