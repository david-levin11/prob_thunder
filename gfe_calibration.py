import numpy as np
import xarray as xr
from pathlib import Path

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------

GRIB_FILE = Path(
    r"C:\Users\David.Levin\NBMLightningVer\blend.t12z.core.f036.ak.grib2"
)

FCST_PERIOD = "Night"

INPUT_IS_PERCENT = True
OUTPUT_AS_PERCENT = True
ZERO_STAYS_ZERO = True

# ---------------------------------------------------------------------
# HARD-CODED 12-HOUR NIGHT/DAY ISOTONIC REGRESSION
# Values are in 0-1 probability units.
# ---------------------------------------------------------------------

RAW_PROB_12HR_NIGHT = np.array([
    0.0000000000,
    0.0151058246,
    0.0639222402,
    0.1307769930,
    0.2330520926,
    0.3282108362,
    0.4304019468,
    0.5126648841
], dtype=float)

CAL_PROB_12HR_NIGHT = np.array([
    0.0018024029,
    0.0304905597,
    0.1783750854,
    0.3673135261,
    0.5696382198,
    0.7730003805,
    0.9172789015,
    0.9595959596
], dtype=float)

RAW_PROB_12HR_DAY = np.array([
    0.0000000000,
    0.0158176926,
    0.0644448641,
    0.1314523272,
    0.2342858449,
    0.3321290598,
    0.4253702126,
    0.5137450072
], dtype=float)

CAL_PROB_12HR_DAY = np.array([
    0.0015543500,
    0.0277707320,
    0.1583320615,
    0.3270040687,
    0.5200537463,
    0.6601998346,
    0.7731219265,
    0.7876657613
], dtype=float)

# ---------------------------------------------------------------------
# READ GRIB
# ---------------------------------------------------------------------

ds = xr.open_dataset(
    GRIB_FILE,
    engine="cfgrib",
    backend_kwargs={"indexpath": ""},
)

var_name = list(ds.data_vars)[0]
raw = ds[var_name]

# Convert NBM values to 0-1 probability
raw_prob = raw / 100.0 if INPUT_IS_PERCENT else raw
raw_prob = raw_prob.clip(0.0, 1.0)

# ---------------------------------------------------------------------
# APPLY CALIBRATION
# ---------------------------------------------------------------------

if FCST_PERIOD == "Night":
    raw_arr = RAW_PROB_12HR_NIGHT
    cal_arr = CAL_PROB_12HR_NIGHT
else:
    raw_arr = RAW_PROB_12HR_DAY
    cal_arr = CAL_PROB_12HR_DAY

calibrated = xr.apply_ufunc(
    lambda a: np.interp(
        np.clip(a, 0.0, 1.0),
        raw_arr,
        cal_arr,
        left=cal_arr[0],
        right=cal_arr[-1],
    ),
    raw_prob,
)

if ZERO_STAYS_ZERO:
    calibrated = calibrated.where(raw_prob != 0.0, 0.0)

if OUTPUT_AS_PERCENT:
    calibrated = calibrated * 100.0

calibrated.name = f"{var_name}_calibrated"
calibrated.attrs = raw.attrs.copy()
calibrated.attrs["long_name"] = "Isotonic calibrated 12-hour night thunder probability"
calibrated.attrs["units"] = "%" if OUTPUT_AS_PERCENT else "1"

print(calibrated)
print(f"Raw max:        {float(raw_prob.max()):.4f}")
print(f"Calibrated max: {float((calibrated / 100.0 if OUTPUT_AS_PERCENT else calibrated).max()):.4f}")