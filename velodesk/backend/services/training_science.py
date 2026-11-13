"""
Training science calculations:
- TSS (Training Stress Score) - power-based and HR-estimated
- PMC (Performance Management Chart): CTL, ATL, TSB
- Power Curve (Mean Maximal Power)
- 3-Parameter Critical Power model (Morton 1996)
  P(t) = W'/t + CP + Pmax * e^(-t/tau)
  FTP = CP
"""
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# PMC time constants (days)
CTL_TAU = 42   # chronic training load decay constant
ATL_TAU = 7    # acute training load decay constant

# TSS estimation constants
MAX_HR_DEFAULT = 190
REST_HR_DEFAULT = 50


# ---------------------------------------------------------------------------
# TSS Calculation
# ---------------------------------------------------------------------------

def calculate_normalized_power(power_stream: List[float]) -> Optional[float]:
    """Calculate Normalized Power from a per-second power stream (Allen & Coggan)."""
    if not power_stream or len(power_stream) < 30:
        return None
    try:
        stream = np.array([max(0.0, float(p or 0)) for p in power_stream])
        # 30-second rolling average
        kernel = np.ones(30) / 30
        rolling = np.convolve(stream, kernel, mode='valid')
        # Raise to 4th power, mean, 4th root
        return float(np.mean(rolling ** 4) ** 0.25)
    except Exception:
        return None


def calculate_tss_from_power(
    power_stream: List[float],
    ftp: float,
    duration_seconds: int
) -> Tuple[float, float, float]:
    """
    Calculate TSS, NP, and IF from a power stream.
    Returns (tss, normalized_power, intensity_factor)
    """
    if not power_stream or ftp <= 0:
        return 0.0, 0.0, 0.0

    arr = np.array(power_stream, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)

    # Normalized Power: 30s rolling average -> ^4 -> mean -> ^0.25
    window = 30
    if len(arr) >= window:
        rolling = np.convolve(arr, np.ones(window) / window, mode='valid')
        np_val = float(np.mean(rolling ** 4) ** 0.25)
    else:
        np_val = float(np.mean(arr))

    if_ = np_val / ftp
    duration_hours = duration_seconds / 3600.0
    tss = (duration_hours * np_val * if_) / ftp * 100.0

    return round(tss, 1), round(np_val, 1), round(if_, 3)


def estimate_tss_from_hr(
    duration_seconds: int,
    avg_hr: float,
    max_hr: float = MAX_HR_DEFAULT,
    sport_type: str = "Ride",
    rest_hr: float = REST_HR_DEFAULT,
    ftp_hr_fraction: float = 0.89  # approx HR at threshold
) -> float:
    """
    Estimate TSS from heart rate using hrTSS method.
    Based on the relationship between HR reserve and training stress.
    """
    if avg_hr <= 0 or duration_seconds <= 0:
        return 0.0

    hr_reserve = max_hr - rest_hr
    lthr = rest_hr + ftp_hr_fraction * hr_reserve

    # HR-based intensity factor
    hr_if = (avg_hr - rest_hr) / (lthr - rest_hr)
    hr_if = max(0.0, min(hr_if, 1.5))

    duration_hours = duration_seconds / 3600.0
    tss = duration_hours * (hr_if ** 2) * 100.0

    return round(tss, 1)


def estimate_tss_from_rpe(duration_seconds: int, rpe: float) -> float:
    """
    Estimate TSS from session RPE (Borg CR10 scale, 1-10).
    Based on Foster's session-RPE method, calibrated so RPE 5 (moderate/Z2)
    over 1 hour ≈ 50 TSS, matching standard TSS scaling.

    RPE 1-2: very easy / recovery   (IF ~0.45-0.55)
    RPE 3-4: easy / endurance        (IF ~0.60-0.70)
    RPE 5-6: moderate / tempo        (IF ~0.75-0.85)
    RPE 7-8: hard / threshold        (IF ~0.90-1.00)
    RPE 9-10: very hard / max effort (IF ~1.05-1.20)
    """
    if rpe <= 0 or duration_seconds <= 0:
        return 0.0

    rpe = max(1.0, min(rpe, 10.0))
    # Map RPE 1-10 to an approximate intensity factor 0.40-1.20
    # Roughly linear with slight curve at the top end for the hardest efforts
    if_equiv = 0.35 + (rpe / 10.0) * 0.85

    duration_hours = duration_seconds / 3600.0
    tss = duration_hours * (if_equiv ** 2) * 100.0
    return round(tss, 1)


def estimate_tss_no_data(duration_seconds: int, sport_type: str) -> float:
    """
    Rough TSS estimate when no power or HR data is available.
    Uses typical intensity assumptions per sport.
    """
    duration_hours = duration_seconds / 3600.0
    sport_if = {
        "Ride": 0.65,
        "VirtualRide": 0.70,
        "Run": 0.75,
        "TrailRun": 0.80,
        "Walk": 0.45,
        "Hike": 0.50,
        "RopeJumping": 0.85,
        "CoreStability": 0.40,
        "WeightTraining": 0.50,
    }
    if_ = sport_if.get(sport_type, 0.60)
    return round(duration_hours * (if_ ** 2) * 100.0, 1)


# ---------------------------------------------------------------------------
# Performance Management Chart (PMC)
# ---------------------------------------------------------------------------

def calculate_pmc(
    daily_tss: Dict[datetime, float],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_ctl: float = 0.0,
    initial_atl: float = 0.0,
) -> List[Dict]:
    """
    Calculate CTL (fitness), ATL (fatigue), TSB (form) using
    Banister impulse-response model with exponential decay.

    CTL(d) = CTL(d-1) * e^(-1/42) + TSS(d) * (1 - e^(-1/42))
    ATL(d) = ATL(d-1) * e^(-1/7)  + TSS(d) * (1 - e^(-1/7))
    TSB(d) = CTL(d-1) - ATL(d-1)   (yesterday's fitness minus fatigue)
    """
    if not daily_tss:
        return []

    if start_date is None:
        start_date = min(daily_tss.keys())
    if end_date is None:
        end_date = max(daily_tss.keys())

    ctl_decay = np.exp(-1 / CTL_TAU)
    atl_decay = np.exp(-1 / ATL_TAU)
    ctl_gain = 1 - ctl_decay
    atl_gain = 1 - atl_decay

    ctl = initial_ctl
    atl = initial_atl
    results = []

    current = start_date
    while current <= end_date:
        tss = daily_tss.get(current, 0.0)
        tsb = ctl - atl  # form = yesterday's fitness - fatigue
        ctl = ctl * ctl_decay + tss * ctl_gain
        atl = atl * atl_decay + tss * atl_gain

        results.append({
            "date": current.isoformat(),
            "tss": round(tss, 1),
            "ctl": round(ctl, 1),
            "atl": round(atl, 1),
            "tsb": round(tsb, 1),
        })
        current += timedelta(days=1)

    return results


# ---------------------------------------------------------------------------
# Power Curve (Mean Maximal Power)
# ---------------------------------------------------------------------------

POWER_CURVE_DURATIONS = [
    1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300,
    360, 420, 480, 600, 720, 900, 1200, 1800, 2700, 3600, 5400, 7200
]


def calculate_mmp(power_stream: List[float], duration_seconds: int) -> Optional[float]:
    """Calculate mean maximal power for a given duration from a power stream."""
    arr = np.array(power_stream, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)

    if len(arr) < duration_seconds:
        return None

    # Sliding window mean
    kernel = np.ones(duration_seconds) / duration_seconds
    rolling = np.convolve(arr, kernel, mode='valid')
    return float(np.max(rolling))


def build_power_curve(power_stream: List[float]) -> Dict[int, float]:
    """Build full power curve from a single activity's power stream."""
    results = {}
    for dur in POWER_CURVE_DURATIONS:
        mmp = calculate_mmp(power_stream, dur)
        if mmp is not None:
            results[dur] = round(mmp, 1)
    return results

DURABILITY_LEVELS_KJ: List[int] = [0, 1000, 2000]

def build_power_curve_after_work(power_stream: List[float], work_kj: int) -> Dict[int, float]:
    if work_kj <= 0:
        return build_power_curve(power_stream)
    arr = np.nan_to_num(np.array(power_stream, dtype=float), nan=0.0)
    cumulative_kj = np.cumsum(arr) / 1000.0
    start_idx = int(np.searchsorted(cumulative_kj, float(work_kj)))
    if start_idx >= len(arr):
        return {}
    return build_power_curve(arr[start_idx:].tolist())

def merge_power_curves(curves: List[Dict[int, float]]) -> Dict[int, float]:
    """Merge multiple power curves, keeping the best (highest) value per duration."""
    merged = {}
    for curve in curves:
        for dur, power in curve.items():
            if dur not in merged or power > merged[dur]:
                merged[dur] = power
    return merged


# ---------------------------------------------------------------------------
# 3-Parameter Critical Power Model (Morton 1996)
# P(t) = W'/t + CP + Pmax * exp(-t / tau)
# where tau is derived, and FTP = CP
# ---------------------------------------------------------------------------

def _cp3_model(t: np.ndarray, cp: float, w_prime: float, p_max: float) -> np.ndarray:
    """
    3-parameter CP model: P(t) = W'/t + CP + (Pmax - CP) * exp(-t/tau)
    Morton (1996): Ergonomics 39(4)
    tau is implicit: as t->inf P->CP, short durations dominated by Pmax and W'
    We use the simplified form fitting CP, W', and an effective Pmax component.
    """
    # Avoid division by zero
    t = np.maximum(t, 0.1)
    # Morton's form: P = CP + W'/t  (2-param) extended with sprint component
    # Full 3-param: P(t) = Pmax * (1 - exp(-t * CP / W')) + CP * exp(-t * CP / W')
    # Simplified tractable form used in practice:
    tau = w_prime / (p_max - cp) if (p_max - cp) > 0 else 1.0
    return cp + w_prime / t + (p_max - cp) * np.exp(-t / tau)


def estimate_cp_single_effort(
    duration_seconds: float,
    power_watts: float,
    w_prime: float = 20000.0,
    p_max: float = None,
) -> Dict:
    """
    Estimate CP from a single maximal effort using Morton's 3-parameter model,
    following intervals.icu's approach: rather than fitting all 3 parameters to
    a multi-point power curve (which requires several recent maximal efforts of
    varying duration), fix W' and Pmax at reasonable defaults and solve for CP
    algebraically from one known (duration, power) pair.

    This only requires ONE maximal effort between 180s (3min) and 1800s (30min),
    matching intervals.icu's stated requirement — much more practical for riders
    who rarely do repeated all-out efforts across different durations.

    P(t) = W'/t + CP + (Pmax - CP) * e^(-t/tau)

    For efforts of 3-30 min, the exponential term is negligible (tau=30s means
    e^(-t/30) is essentially 0 beyond ~3min), so this simplifies to:
    P(t) ≈ W'/t + CP  =>  CP ≈ P(t) - W'/t
    """
    if duration_seconds < 180 or duration_seconds > 1800:
        logger.warning(
            "Single-effort CP estimation works best for 180-1800s efforts; "
            "got %.0fs — result may be less reliable", duration_seconds
        )

    tau = 30.0
    if p_max is None:
        # Without a known Pmax, the exponential term is ~0 for efforts >=3min anyway
        p_max = power_watts * 1.5  # rough anaerobic ceiling, only matters for short end

    exp_term = (p_max - (power_watts)) * np.exp(-duration_seconds / tau)
    # Solve: power = w_prime/t + cp + (p_max - cp)*e^(-t/tau)
    # Rearranged for cp:
    decay = np.exp(-duration_seconds / tau)
    cp = (power_watts - w_prime / duration_seconds - p_max * decay) / (1 - decay) if decay < 0.999 else \
         power_watts - w_prime / duration_seconds

    return {
        "cp": round(float(cp), 1),
        "w_prime": w_prime,
        "p_max": p_max,
        "method": "single_effort",
        "source_duration_s": duration_seconds,
        "source_power_w": power_watts,
    }


def fit_critical_power(
    power_curve: Dict[int, float],
    min_duration: int = 120,    # start at 2 min — short sprints distort CP
    max_duration: int = 1200,   # up to 20 min
) -> Optional[Dict]:
    """
    Fit the 3-parameter critical power model to a power curve.
    Uses durations between min_duration and max_duration seconds.

    Returns dict with cp, w_prime, p_max, r_squared, or None if fitting fails.
    FTP = CP (Morton's interpretation)
    """
    # Filter to relevant durations
    durations = []
    powers = []
    for dur, pwr in sorted(power_curve.items()):
        if min_duration <= dur <= max_duration and pwr > 0:
            durations.append(float(dur))
            powers.append(float(pwr))

    if len(durations) < 6:
        logger.warning("Insufficient data points for CP model fitting (%d points)", len(durations))
        return None

    t = np.array(durations)
    p = np.array(powers)

    # Initial parameter estimates
    # CP should be close to 20-min power — use 80th percentile of longer efforts
    long_mask = t >= 600  # 10+ min efforts
    if np.sum(long_mask) >= 2:
        cp_init = float(np.mean(p[long_mask]) * 0.95)
    else:
        cp_init = float(np.percentile(p, 30))
    p_max_init = float(np.max(p))
    w_prime_init = 20000.0  # typical W' ~15-25 kJ

    try:
        popt, pcov = curve_fit(
            _cp3_model,
            t, p,
            p0=[cp_init, w_prime_init, p_max_init],
            bounds=(
                [50,  1000,  p_max_init * 0.5],   # lower bounds
                [500, 100000, 2000]                 # upper bounds
            ),
            maxfev=10000,
        )
        cp, w_prime, p_max = popt

        # Sanity check: CP should be plausible relative to 20-min power
        if len(p[long_mask]) > 0:
            p20_approx = float(np.min(p[t >= 1200])) if np.sum(t >= 1200) > 0 else float(np.min(p[long_mask]))
            if cp > p20_approx * 1.10:
                logger.warning("CP %.1fW seems too high vs 20-min power %.1fW, capping", cp, p20_approx)
                cp = p20_approx * 1.05  # CP is typically ~95-105% of 20-min power

        # Calculate R²
        p_pred = _cp3_model(t, *popt)
        ss_res = np.sum((p - p_pred) ** 2)
        ss_tot = np.sum((p - np.mean(p)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "cp": round(float(cp), 1),
            "w_prime": round(float(w_prime), 0),
            "p_max": round(float(p_max), 1),
            "r_squared": round(float(r_squared), 4),
            "ftp": round(float(cp), 1),  # FTP = CP per Morton
        }

    except (RuntimeError, ValueError) as e:
        logger.error("CP model fitting failed: %s", e)
        return None


def get_power_zones(ftp: float) -> List[Dict]:
    """
    Calculate Coggan power zones based on FTP.
    """
    zones = [
        {"zone": 1, "name": "Active Recovery", "min": 0,    "max": round(ftp * 0.55)},
        {"zone": 2, "name": "Endurance",        "min": round(ftp * 0.56), "max": round(ftp * 0.75)},
        {"zone": 3, "name": "Tempo",            "min": round(ftp * 0.76), "max": round(ftp * 0.90)},
        {"zone": 4, "name": "Threshold",        "min": round(ftp * 0.91), "max": round(ftp * 1.05)},
        {"zone": 5, "name": "VO2 Max",          "min": round(ftp * 1.06), "max": round(ftp * 1.20)},
        {"zone": 6, "name": "Anaerobic",        "min": round(ftp * 1.21), "max": round(ftp * 1.50)},
        {"zone": 7, "name": "Neuromuscular",    "min": round(ftp * 1.51), "max": 9999},
    ]
    return zones


def watts_to_pct(watts: Optional[float], ftp: float) -> Optional[float]:
    if not watts or ftp <= 0:
        return None
    return round(float(watts) / ftp, 4)


def resolve_intervals_to_watts(
    intervals: List[Dict[str, Any]], ftp: float
) -> List[Dict[str, Any]]:
    def watts(pct: Optional[float]) -> Optional[int]:
        return round(float(pct) * ftp) if pct else None

    result: List[Dict[str, Any]] = []
    for iv in intervals:
        item: Dict[str, Any] = dict(iv)
        if iv.get("ftp_pct_low"):
            item["power_low"] = watts(iv["ftp_pct_low"])
            item["power_high"] = watts(iv.get("ftp_pct_high") or iv["ftp_pct_low"])
        if iv.get("rest_pct_low"):
            item["rest_power_low"] = watts(iv["rest_pct_low"])
            item["rest_power_high"] = watts(iv.get("rest_pct_high") or iv["rest_pct_low"])
        steps: List[Dict[str, Any]] = []
        for step in iv.get("steps") or []:
            s: Dict[str, Any] = dict(step)
            if step.get("ftp_pct_low"):
                s["power_low"] = watts(step["ftp_pct_low"])
                s["power_high"] = watts(step.get("ftp_pct_high") or step["ftp_pct_low"])
            steps.append(s)
        item["steps"] = steps
        result.append(item)
    return result