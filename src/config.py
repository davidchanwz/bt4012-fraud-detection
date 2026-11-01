"""Shared paths and settings so every script uses the same split boundaries."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
SCORE_DIR = ROOT / "reports" / "scores"     # row-level model scores (txn_id, score)
MODEL_DIR = ROOT / "models"                 # fitted models (gitignored)

RAW_FILE = RAW_DIR / "bs140513_032310.csv"
CLEAN_FILE = PROCESSED_DIR / "banksim_clean.parquet"
GRAPH_FILE = PROCESSED_DIR / "features_graph.parquet"

# Days before a transaction's fraud label is known (used by the g5_ features).
# A label from day s can be used from day s + LABEL_DELAY onwards.
LABEL_DELAY = 7

# The original Kaggle dataset (ealaxi/banksim1) needs a login, so download.py
# pulls a public GitHub mirror. We checked the file against the expected
# size and git blob hash (two unrelated repos have the identical blob).
RAW_URL = (
    "https://raw.githubusercontent.com/niharika-1806/"
    "Financial-Fraud-Detection/HEAD/bs140513_032310.csv"
)
RAW_SIZE_BYTES = 48_391_391
RAW_GIT_BLOB_SHA1 = "a114ec2e6a69e148035e2bf5431c52f37f8473c8"

# Chronological split on `step` (day index 0-179).
# train = days 0-119 (~4 months), val = 120-149, test = 150-179 (~1 month each)
# Fixed in week 1 -- don't change these after looking at model results.
TRAIN_END = 119
VAL_END = 149

SEED = 4012
