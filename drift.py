import numpy as np
from scipy.stats import ks_2samp

from database import SessionLocal
from models import Prediction


# ---------------- PSI ---------------- #
def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints)

    psi = 0.0

    for i in range(len(breakpoints) - 1):
        e = ((expected >= breakpoints[i]) & (expected < breakpoints[i + 1])).mean()
        a = ((actual >= breakpoints[i]) & (actual < breakpoints[i + 1])).mean()

        if e == 0:
            e = 0.0001
        if a == 0:
            a = 0.0001

        psi += (e - a) * np.log(e / a)

    return psi


# ---------------- KS ---------------- #
def calculate_ks(expected, actual):
    return ks_2samp(expected, actual).statistic


# ---------------- FETCH DATA ---------------- #
def get_prediction_windows():
    db = SessionLocal()

    preds = db.query(Prediction).order_by(Prediction.created_at.desc()).all()

    if len(preds) < 20:
        return [], []

    # ✅ FIXED FIELD NAME
    values = [p.risk_probability for p in preds]

    recent = values[:10]
    previous = values[10:20]

    return previous, recent


# ---------------- DRIFT SCORE ---------------- #
def compute_drift_score():
    previous, recent = get_prediction_windows()

    if len(previous) == 0 or len(recent) == 0:
        return 0

    psi = calculate_psi(previous, recent)
    ks = calculate_ks(previous, recent)

    drift_score = (0.6 * psi) + (0.4 * ks)

    return float(drift_score)