"""Central paths, physical constants and runtime settings for SkyOps."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
# The LLM SDKs read their keys (GEMINI_API_KEY / ANTHROPIC_API_KEY) from the process environment, so load .env into it.
load_dotenv(ROOT / ".env")
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
    device: str = "auto"  # "auto" = CUDA when PyTorch can see a GPU, else CPU; or force "cpu" / "cuda"
    yolo_weights: str = "yolov8n.pt"  # replaced by models/visdrone_yolo.pt after fine-tuning
    landuse_weights: str = "models/landuse_unet.pt"
    rul_model: str = "models/rul_lgbm.txt"
    assistant_provider: str = "gemini"        # "gemini" (free tier, default) or "claude"
    # Free-tier latency measured Sept 2026 on a two-round tool question: 3.6-flash ~10 s, 3.5-flash ~17 s,
    # 3.5-flash-lite ~4 s (but sloppier with units); 3.7-flash and 3.8-flash were overloaded (timeouts, 90-160 s).
    gemini_model: str = "gemini-3.6-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"  # finishes the question after a timeout, 429 or 5xx
    gemini_timeout_s: float = 25.0                         # hard cap per API call
    claude_model: str = "claude-opus-5"
    use_gru: bool = False                     # opt in to the GRU residual predictor (it did not beat dead reckoning)
    opensky_csv: str | None = None            # replay another recording (see scripts/record_opensky.py)
    opensky_client_id: str | None = None      # optional OpenSky API client for higher live-feed rate limits
    opensky_client_secret: str | None = None
    live_interval_s: float = 30.0

    model_config = SettingsConfigDict(env_prefix="SKYOPS_", env_file=str(ROOT / ".env"), extra="ignore")


settings = Settings()


def resolve_device() -> str:
    """Inference device: honours SKYOPS_DEVICE, with 'auto' picking CUDA when available."""
    if settings.device != "auto":
        return settings.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def replay_csv() -> Path:
    """The capture used for replay: SKYOPS_OPENSKY_CSV when set, else the organisers' file."""
    if settings.opensky_csv:
        p = Path(settings.opensky_csv)
        return p if p.is_absolute() else ROOT / p
    return OPENSKY_CSV
