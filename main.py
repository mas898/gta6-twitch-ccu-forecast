"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun gta6-twitch-ccu-forecast-01).

The program reads one case file per run (the launch's history up to the end of hour 1, that
hour's print included, plus past launches' hourly curves as `comparables`) and writes five
quantiles (levels 0.05/0.25/0.5/0.75/0.95) of the Twitch concurrent-viewer count for each of
hours 2..24 of the UTC release day.

Model, fitted at run time from the case file's own comparables (no baked-in parameters, no
network, standard library only):

  Past launches' release-day curves show one strong regularity: the log ratio of an hour's
  count to the hour-1 print falls with the size of the hour-1 print, and flattens out at the
  top end. A small hour-1 print (the category was quiet at 00:00 UTC, before most time zones
  unlocked) grows several-fold into the evening peak; a huge hour-1 print (the launch was
  already global at midnight UTC) stays nearly flat. So for each hour h we fit, over the
  comparables, a quadratic in the centered log hour-1 print,

      log(v_h / v_1) = a_h + b_h * lx + c_h * lx^2,   lx = log v_1 - mean log v_1,

  and predict the target's median as p1 * exp(a_h + b_h * lx + c_h * lx^2), where lx is the
  target's centered log hour-1 print, clamped to the comparables' observed range so a record
  launch never extrapolates the curve past the data.

  The spread is the local residual scale: the standard deviation of the regression residuals
  of the comparables whose log hour-1 print lies within 1.2 of the target's (falling back to
  the 8 nearest), shrunk by 0.8 and floored at 0.12 log units. Quantiles are the median times
  exp(z_tau * spread) with the standard normal quantiles. The 0.8 shrink was calibrated on
  the four practice launches (it brings the 90% band's coverage to about its nominal level).

  When an hour has too few usable comparables, or anything unexpected happens, the hour falls
  back to the persistence forecast (every quantile around the hour-1 print, spread widening
  like a random walk), so the program always writes a valid forecast.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, no network). Outside the
grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

HOURS = range(2, 25)
MAX_VIEWERS = 1_000_000_000

# Minimum usable comparables for an hour's regression; below this, fall back to persistence.
MIN_CURVES = 5
# Local-spread window: comparables within this log distance of the target's hour-1 print.
LOCAL_WINDOW = 1.2
# Fallback count for the local spread when the window holds too few comparables.
LOCAL_MIN_COUNT = 8
# Shrink applied to the local residual spread (calibrated on the practice launches).
SPREAD_SHRINK = 0.8
# Floor on the log spread, so no hour claims more precision than the data supports.
SPREAD_FLOOR = 0.12
# The persistence fallback's per-square-root-hour log spread.
PERSIST_SIGMA = 0.15


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp; return None when it does not parse."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def round2(x):
    """x rounded to 2 decimals, half up; a whole number comes back as an int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def as_number(value):
    """A finite number as a float, else None (guards against nulls and strings)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def mean(values):
    return sum(values) / len(values)


def pstdev(values, center):
    return math.sqrt(sum((v - center) ** 2 for v in values) / len(values))


def release_day_curve(comparable):
    """A comparable's release-day curve: (hour-1 viewers, {hour: viewers} for hours 1..24).

    Returns None when the comparable has no usable hour-1 print. Hours with no point, or a
    non-positive count, are simply absent from the map; each hour's regression uses the
    comparables that have that hour.
    """
    day_start = parse_time(comparable.get("day_start"))
    hourly = comparable.get("hourly")
    if day_start is None or not isinstance(hourly, list):
        return None
    points = {}
    for row in hourly:
        if not isinstance(row, dict):
            continue
        at = parse_time(row.get("from"))
        viewers = as_number(row.get("viewers"))
        if at is None or viewers is None or viewers <= 0:
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in points:
            points[hour] = viewers
    first = points.get(1)
    if first is None:
        return None
    return first, points


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination; None when singular."""
    n = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for cc in range(col, n + 1):
                rows[r][cc] -= factor * rows[col][cc]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))) / rows[i][i]
    return out


def polyfit2(xs, ys):
    """Least-squares quadratic y = a + b*x + c*x^2; returns (a, b, c).

    Falls back to the constant mean when the normal equations are singular.
    """
    basis = [[1.0, x, x * x] for x in xs]
    matrix = [[sum(r[i] * r[j] for r in basis) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i] * y for r, y in zip(basis, ys)) for i in range(3)]
    beta = solve_linear(matrix, rhs)
    if beta is None:
        return mean(ys), 0.0, 0.0
    return beta[0], beta[1], beta[2]


def fit_hour(curves, hour):
    """Fit one hour's quadratic over the curves that have it.

    curves: list of (log hour-1 print, {hour: viewers}). Returns (beta, xbar, lo, hi,
    residuals, log_firsts) with x centered at the mean log hour-1 print, or None when too
    few curves have this hour.
    """
    usable = [(lv, pts[hour]) for lv, pts in curves if hour in pts]
    if len(usable) < MIN_CURVES:
        return None
    log_firsts = [lv for lv, _ in usable]
    xbar = mean(log_firsts)
    xs = [lv - xbar for lv in log_firsts]
    ys = [math.log(v) - lv for lv, v in usable]
    a, b, c = polyfit2(xs, ys)
    residuals = [y - (a + b * x + c * x * x) for x, y in zip(xs, ys)]
    return (a, b, c), xbar, min(log_firsts), max(log_firsts), residuals, log_firsts


def local_spread(residuals, log_firsts, target_log_first):
    """Residual scale near the target's hour-1 size: the local window's SD, shrunk."""
    near = [r for r, lv in zip(residuals, log_firsts) if abs(lv - target_log_first) <= LOCAL_WINDOW]
    if len(near) < LOCAL_MIN_COUNT:
        order = sorted(range(len(log_firsts)), key=lambda i: abs(log_firsts[i] - target_log_first))
        near = [residuals[i] for i in order[:LOCAL_MIN_COUNT]]
    if not near:
        return SPREAD_FLOOR
    sd = pstdev(near, mean(near))
    return max(SPREAD_FLOOR, SPREAD_SHRINK * sd)


def tidy_quantiles(values):
    """Clamp to (0, 1e9], round to 2 decimals, and enforce non-decreasing order."""
    out = []
    prev = 0.0
    for value in values:
        q = min(MAX_VIEWERS, max(value, 0.01))
        q = round2(q)
        if q <= 0:
            q = 0.01
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def persistence_row(p1, hour):
    """The fallback row: every quantile around the hour-1 print, spread widening in h."""
    width = PERSIST_SIGMA * math.sqrt(hour - 1)
    return tidy_quantiles([p1 * math.exp(z * width) for z in Z])


def model_row(p1, hour, fit):
    """One hour's forecast row from its fitted quadratic, or the fallback when unfit."""
    if fit is None:
        return persistence_row(p1, hour)
    (a, b, c), xbar, lo, hi, residuals, log_firsts = fit
    target = math.log(p1)
    lx = min(hi, max(lo, target)) - xbar
    median = p1 * math.exp(a + b * lx + c * lx * lx)
    spread = local_spread(residuals, log_firsts, target)
    return tidy_quantiles([median * math.exp(z * spread) for z in Z])


def hour1_print(case):
    """Hour 1's print: the `known` row, else the last hourly history point, else 1."""
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = as_number(row.get("viewers"))
            if viewers is not None and viewers > 0:
                return viewers
    hourly = case.get("history_hourly") or []
    for row in reversed(hourly):
        if isinstance(row, dict):
            viewers = as_number(row.get("viewers"))
            if viewers is not None and viewers > 0:
                return viewers
    return 1.0


def forecast_case(case):
    """Forecast one case: 23 rows, hours 2..24, five quantiles each."""
    p1 = max(1.0, hour1_print(case))
    curves = []
    for comparable in case.get("comparables") or []:
        if not isinstance(comparable, dict):
            continue
        curve = release_day_curve(comparable)
        if curve is not None:
            first, points = curve
            curves.append((math.log(first), points))
    rows = []
    for hour in HOURS:
        try:
            fit = fit_hour(curves, hour)
            rows.append({"hour": hour, "q": model_row(p1, hour, fit)})
        except Exception as exc:  # never lose the case over one hour's fit
            print(f"hour {hour}: falling back ({exc})", file=sys.stderr)
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception as exc:  # a malformed case still gets a valid-shaped fallback
            print(f"{path.name}: falling back ({exc})", file=sys.stderr)
            forecast = {"hours": [{"hour": h, "q": persistence_row(1.0, h)} for h in HOURS]}
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
