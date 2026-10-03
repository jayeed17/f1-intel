from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
PREBUILT_DIR = ROOT / "data" / "prebuilt"
MODEL_DATA_DIR = ROOT / "data" / "model"
RACE_DATASET_PATH = MODEL_DATA_DIR / "race_dataset.parquet"
PREDICTIONS_DIR = ROOT / "predictions"

# Committed (not gitignored, unlike MODEL_DIR) frozen race-predictor snapshot
# -- see app/models/race_predictor.py's freeze_model()/load_frozen_model().
FROZEN_MODEL_DIR = MODEL_DATA_DIR / "frozen"

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

# circuit_id -> first season under a materially different layout. Circuit
# history (app/models/*_predictor.py's circuit-history features) treats the
# circuit as brand-new starting that season -- editions before it don't
# count as "prior history here", even though Ergast/Jolpica keeps the same
# circuitId across the layout change. 2026 Spanish GP moves from Catalunya's
# old full layout to a reconfigured one.
CIRCUIT_HISTORY_RESET = {"catalunya": 2026}

# Street-circuit classification for the dashboard's "Street circuits" view
# ONLY -- deliberately separate from CIRCUIT_TYPE above (street/mixed/
# high-speed), which is a trained model feature and must not change
# (retraining would be needed, invalidating the frozen models' meaning).
# "street": run entirely on everyday public roads, walls/barriers
# throughout, no permanent racing infrastructure. "hybrid_street": built on
# closed-off roads/a purpose-laid-out lot for the race but with wide,
# flowing, more permanent-track-like characteristics -- or (Montreal) a
# dedicated permanent facility that still *plays* like a street circuit
# (tight, wall-lined, unforgiving, bumpy). Anything not listed defaults to
# "permanent" (classify_circuit() in app/analysis/street_circuits.py).
STREET_CIRCUIT_CLASS = {
    "monaco": "street",      # Monte Carlo public roads, barriers the entire lap, no runoff anywhere
    "baku": "street",        # Baku City Circuit: public roads through the Old City plus a long public-road straight
    "marina_bay": "street",  # Marina Bay: public roads around the bay, closed and lit for a night race
    "vegas": "street",       # Las Vegas Strip Circuit: public roads (the Strip, Koval Lane), reopened to traffic after
    "jeddah": "street",      # Jeddah Corniche Circuit: public corniche roads, walls close by despite very high average speed
    "albert_park": "hybrid_street",  # Melbourne: public park roads, but 2021's reconfig widened/smoothed it into permanent-track-like flow
    "miami": "hybrid_street",        # Miami: temporary circuit on closed public roads, but purpose-laid-out wide/sweeping stadium-lot loop
    "villeneuve": "hybrid_street",   # Montreal: a dedicated permanent island facility, but tight/wall-lined/bumpy -- plays like a street track
    "madring": "hybrid_street",      # Madring (2026 Madrid): explicitly combines a street section (IFEMA) with a purpose-built permanent section
}
