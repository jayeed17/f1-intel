import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DATA_DIR = ROOT / "demo_data"

# Lap-time gain per lap from burning fuel (seconds). Rough public estimate; tune per track.
FUEL_EFFECT_PER_LAP = 0.035

# Default pit loss (seconds) when it can't be estimated from data.
DEFAULT_PIT_LOSS = 22.0

# Brake applications that scrub less speed than this (kph) are treated as dabs, not zones.
MIN_SPEED_DROP_KPH = 20.0

# Window around a corner apex for min-speed search (metres before, after).
CORNER_WINDOW = (150.0, 50.0)


def offline_mode() -> bool:
    """True when OFFLINE_MODE is set (env var, or st.secrets copied into the
    environment by the dashboard) — restricts the app to bundled demo_data/
    and never calls FastF1. Read dynamically (not cached) so tests and the
    dashboard can toggle it via environment variables at any time."""
    return os.environ.get("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on")
