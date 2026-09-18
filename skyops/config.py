"""Central paths, physical constants and runtime settings for SkyOps."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
WEB_DIR = ROOT / "web"

OPENSKY_CSV = DATA_RAW / "opensky" / "opensky_trajectories.csv"
CMAPSS_DIR = DATA_RAW / "cmapss" / "CMAPSSData"
VISDRONE_DIR = DATA_RAW / "visdrone"
AUAIR_DIR = DATA_RAW / "auair"
DUBAI_DIR = DATA_RAW / "dubai"

# unit conversions
NM_TO_M = 1852.0
FT_TO_M = 0.3048
MS_TO_KT = 1.943844
MS_TO_FPM = 196.8504
R_EARTH_M = 6_371_008.8

# separation minima used by the conflict detector (ICAO en-route radar standard)
HORIZONTAL_SEP_NM = 5.0
VERTICAL_SEP_FT = 1000.0
# near an airport aircraft are legitimately sequenced closer together
TERMINAL_HORIZONTAL_SEP_NM = 3.0
TERMINAL_RADIUS_KM = 40.0
# look-ahead horizons for trajectory prediction (seconds)
PREDICTION_HORIZONS_S = (30, 60, 90, 120, 180, 240, 300)


class Settings(BaseSettings):
    """Runtime settings. Override with environment variables prefixed SKYOPS_ or a .env file."""

    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cpu"  # "cuda" on the GPU laptop
    yolo_weights: str = "yolov8n.pt"  # replaced by models/visdrone_yolo.pt after fine-tuning
    landuse_weights: str = "models/landuse_unet.pt"
    rul_model: str = "models/rul_lgbm.txt"
    assistant_model: str = "claude-opus-5"
    opensky_csv: str | None = None            # replay another recording (see scripts/record_opensky.py)
    opensky_client_id: str | None = None      # optional OpenSky API client for higher live-feed rate limits
    opensky_client_secret: str | None = None
    live_interval_s: float = 30.0

    model_config = SettingsConfigDict(env_prefix="SKYOPS_", env_file=str(ROOT / ".env"), extra="ignore")


settings = Settings()


def replay_csv() -> Path:
    """The capture used for replay: SKYOPS_OPENSKY_CSV when set, else the organisers' file."""
    if settings.opensky_csv:
        p = Path(settings.opensky_csv)
        return p if p.is_absolute() else ROOT / p
    return OPENSKY_CSV
