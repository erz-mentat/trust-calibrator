# trust-calibrator

A model says: confidence 0.8.
It's wrong 35% of the time.

That's not confidence — that's noise with a label.

This module maps raw scores to *calibrated* probabilities
via piecewise-linear interpolation [dependency-free, stdlib only].

---

## What it does

```
raw_score → calibrated_probability_of_correctness
```

Nothing magic. Just honest math.

## Quick Example

```python
from trust_calibrator import TrustCalibrator

cal = TrustCalibrator(points=((0.0, 0.05), (0.3, 0.60), (0.5, 0.80), (1.0, 0.95)))

print(cal.apply(0.42))  # calibrated score in [0, 1]
```

## Why explicit uncertainty?

Because pretending to be certain isn't safer — it's just *wrong*.

Calibrated scores make downstream decisions more reliable:
- thresholds behave predictably
- risk-aware routing becomes possible [manual review, fallbacks, etc.]
- you know what you don't know

## Calibration Format (JSON)

```json
{
  "version": 1,
  "points": [[0.0, 0.05], [0.3, 0.60], [0.5, 0.80], [1.0, 0.95]]
}
```

## CLI

```bash
pip install .
python trust_calibrator.py show --calibration calibration.json
python trust_calibrator.py build --report reliability.json --out calibration.json
```

---

> Built as part of [erz](https://x.com/erz_qwzhdrch) —
> explicit uncertainty is a feature, not a bug.
