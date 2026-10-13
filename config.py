"""
Every assumption the pipeline and the dashboard depend on, in one place.

train.py and app.py both import from here, so the numbers in REPORT.md, the
cost sweep saved by training, and the sliders in the dashboard can never drift
apart. Change a figure here and re-run `main.py train` (or just the app, for
the cost figures, which the app recomputes live).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"   # small JSON/CSV outputs the app reads (committed)
DATASETS_DIR = ROOT / "datasets"     # downloaded data (git-ignored)
BEST_WEIGHTS = WEIGHTS_DIR / "best.pt"

# --- Dataset (public, used as-is, not forked) -------------------------------
# https://universe.roboflow.com/joseph-nelson/hard-hat-workers
# v5 = "raw_HeadHelmetClasses": raw images, person class dropped,
# 70/20/10 train/valid/test split, classes: head, helmet.
RF_WORKSPACE = "joseph-nelson"
RF_PROJECT = "hard-hat-workers"
RF_VERSION = 5
RF_FORMAT = "yolov11"

# The class whose presence means "violation": a head with no helmet on it.
POSITIVE_CLASS = "head"

# --- Training -----------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"   # finetuned from COCO-pretrained weights
EPOCHS = 40
IMGSZ = 640
BATCH = 16
PATIENCE = 10               # early stop if val mAP stalls for 10 epochs
SEED = 0

# --- Business assumptions (euros, per camera snapshot) ------------------------
# Unit of decision: one snapshot from one site camera.
# Positive = the snapshot shows at least one worker without a helmet.
COST_FN = 60.0      # missed snapshot: 5 more min of unprotected exposure (injury + fine risk)
COST_FP = 8.0       # false alarm: supervisor walks over (~10 min at ~€45/h loaded)
COST_TP = 8.0       # caught violation: same walk-over, but it prevents COST_FN
COST_TN = 0.0       # correct "all clear": nothing happens
COST_REVIEW = 0.50  # a person looks at one flagged snapshot (~30 s at ~€45/h + overhead)

BASE_RATE = 0.02    # share of snapshots that really contain a violation on a typical site

# --- Volume and staffing ------------------------------------------------------
CAMERAS = 20
SNAPSHOTS_PER_CAMERA_PER_HOUR = 12   # one every 5 minutes
SHIFT_HOURS = 10
SECONDS_PER_REVIEW = 30
REVIEWER_PRODUCTIVE_HOURS = 6        # per reviewer per day, after breaks/other duties
REVIEWERS_AVAILABLE = 1              # the site's safety officer, part of their day

CASES_PER_DAY = CAMERAS * SNAPSHOTS_PER_CAMERA_PER_HOUR * SHIFT_HOURS
REVIEWS_PER_REVIEWER_PER_DAY = REVIEWER_PRODUCTIVE_HOURS * 3600 // SECONDS_PER_REVIEW
REVIEW_CAPACITY_PER_DAY = REVIEWERS_AVAILABLE * REVIEWS_PER_REVIEWER_PER_DAY


def cost_params() -> dict:
    return dict(
        c_fn=COST_FN, c_fp=COST_FP, c_tp=COST_TP, c_tn=COST_TN,
        c_review=COST_REVIEW, base_rate=BASE_RATE,
    )
