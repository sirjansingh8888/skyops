# SkyOps API + console, CPU inference.
# NOTE: written but not yet built on the development laptop (Docker is not installed there). Expect a ~3 GB image.
#   docker build -t skyops .
#   docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... skyops
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# OpenCV (pulled in by ultralytics) needs these shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-torch-cpu.txt ./
RUN pip install -r requirements-torch-cpu.txt && pip install -r requirements.txt

COPY . .
# Runtime data only (about 330 MB): the replay, C-MAPSS, a slice of the drone flight and the Dubai tiles.
# The trained weights are already in models/. Drive can throttle downloads: re-run the build if this step fails.
RUN python scripts/download_data.py --only opensky cmapss auair dubai --auair-frames 300

ENV SKYOPS_DEVICE=cpu
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "skyops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
