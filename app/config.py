from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"

# Lap-time gain per lap from burning fuel (seconds). Rough public estimate; tune per track.
FUEL_EFFECT_PER_LAP = 0.035

# Default pit loss (seconds) when it can't be estimated from data.
DEFAULT_PIT_LOSS = 22.0

# Brake applications that scrub less speed than this (kph) are treated as dabs, not zones.
MIN_SPEED_DROP_KPH = 20.0

# Window around a corner apex for min-speed search (metres before, after).
CORNER_WINDOW = (150.0, 50.0)
