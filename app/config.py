from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
PREBUILT_DIR = ROOT / "data" / "prebuilt"
MODEL_DATA_DIR = ROOT / "data" / "model"
PREDICTIONS_DIR = ROOT / "predictions"

# Lap-time gain per lap from burning fuel (seconds). Rough public estimate; tune per track.
FUEL_EFFECT_PER_LAP = 0.035

# Default pit loss (seconds) when it can't be estimated from data.
DEFAULT_PIT_LOSS = 22.0

# Brake applications that scrub less speed than this (kph) are treated as dabs, not zones.
MIN_SPEED_DROP_KPH = 20.0

# Window around a corner apex for min-speed search (metres before, after).
CORNER_WINDOW = (150.0, 50.0)

# --------------------------------------------------------------------------
# Race predictor (app/models/race_predictor.py, scripts/build_race_dataset.py)
# --------------------------------------------------------------------------

# Canonical team identity, keyed by every raw name seen across data sources
# (Ergast/Jolpica constructorId, FastF1 TeamName) so a team's rolling features
# survive a rebrand instead of resetting to NaN. AlphaTauri -> RB -> Racing
# Bulls and Alfa Romeo -> Sauber -> Kick Sauber -> Audi are the same
# organisation continuing under new names; Cadillac (2026) is a genuinely new
# entrant with no prior lineage, so it correctly starts cold.
TEAM_ID = {
    "red_bull": "red_bull", "Red Bull Racing": "red_bull",
    "ferrari": "ferrari", "Ferrari": "ferrari",
    "mercedes": "mercedes", "Mercedes": "mercedes",
    "mclaren": "mclaren", "McLaren": "mclaren",
    "aston_martin": "aston_martin", "Aston Martin": "aston_martin",
    "alpine": "alpine", "Alpine F1 Team": "alpine", "Alpine": "alpine",
    "williams": "williams", "Williams": "williams",
    "haas": "haas", "Haas F1 Team": "haas",
    "alphatauri": "rb_alphatauri", "AlphaTauri": "rb_alphatauri",
    "rb": "rb_alphatauri", "RB F1 Team": "rb_alphatauri", "Racing Bulls": "rb_alphatauri",
    "alfa": "sauber_audi", "Alfa Romeo": "sauber_audi",
    "sauber": "sauber_audi", "Sauber": "sauber_audi",
    "Kick Sauber": "sauber_audi", "Audi": "sauber_audi",
    "cadillac": "cadillac", "Cadillac": "cadillac",
}

# Rough manual classification (street / high-speed / mixed), keyed by
# Ergast/Jolpica circuitId (stable across sponsor-name changes, unlike
# EventName). Judgment calls, not a physics model -- tune as needed.
CIRCUIT_TYPE = {
    "albert_park": "mixed", "americas": "mixed", "bahrain": "mixed",
    "baku": "street", "catalunya": "mixed", "hungaroring": "mixed",
    "imola": "mixed", "interlagos": "mixed", "jeddah": "street",
    "losail": "high-speed", "madring": "mixed", "marina_bay": "street",
    "miami": "street", "monaco": "street", "monza": "high-speed",
    "red_bull_ring": "mixed", "ricard": "mixed", "rodriguez": "mixed",
    "sepang": "mixed", "shanghai": "mixed", "silverstone": "high-speed",
    "spa": "high-speed", "suzuka": "high-speed", "vegas": "street",
    "villeneuve": "street", "yas_marina": "mixed", "zandvoort": "mixed",
}

# Seasons whose first races follow a major regulation reset -- prior-season
# form (rolling features) carries over less than usual.
REG_CHANGE_SEASONS = {2022, 2026}
REG_CHANGE_ROUNDS = 3  # first N rounds of a reset season get the flag
