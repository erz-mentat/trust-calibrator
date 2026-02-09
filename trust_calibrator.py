from __future__ import annotations

"""
trust_calibrator.py

A tiny, dependency-free trust/confidence calibrator.

It converts a model-produced score in [0, 1] into a calibrated probability of
correctness via piecewise-linear interpolation between calibration points.

Calibration file schema (JSON):
  {
    "version": 1,
    "points": [
      [0.0, 0.05],
      [0.3, 0.60],
      [0.5, 0.80],
      [1.0, 0.95]
    ]
  }

Each point is (input_score, calibrated_score). Inputs must be within [0, 1].
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

CalibrationPoint = Tuple[float, float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_float(value: Any, *, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {type(value).__name__}") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {f!r}")
    return f


def _normalize_points(points: Iterable[CalibrationPoint]) -> Tuple[CalibrationPoint, ...]:
    normalized: list[CalibrationPoint] = []
    for idx, pair in enumerate(points):
        if not isinstance(pair, tuple) and not isinstance(pair, list):
            raise ValueError(
                "points must be an iterable of (input, output) pairs; "
                f"item {idx} is {type(pair).__name__}"
            )
        if len(pair) != 2:
            raise ValueError(f"each point must have length 2; item {idx} has {len(pair)}")
        inp = _as_float(pair[0], name=f"points[{idx}][0]")
        out = _as_float(pair[1], name=f"points[{idx}][1]")
        if not (0.0 <= inp <= 1.0):
            raise ValueError(f"point input must be in [0, 1], got {inp!r} at index {idx}")
        if not (0.0 <= out <= 1.0):
            raise ValueError(f"point output must be in [0, 1], got {out!r} at index {idx}")
        normalized.append((inp, out))

    normalized.sort(key=lambda p: p[0])
    if not normalized:
        return tuple()

    # Enforce strictly increasing inputs; duplicates are ambiguous.
    deduped: list[CalibrationPoint] = [normalized[0]]
    for inp, out in normalized[1:]:
        prev_inp, prev_out = deduped[-1]
        if inp == prev_inp:
            if out != prev_out:
                raise ValueError(
                    "duplicate input points with different outputs are not allowed: "
                    f"{inp} maps to both {prev_out} and {out}"
                )
            continue
        deduped.append((inp, out))

    return tuple(deduped)


@dataclass(frozen=True)
class TrustCalibrator:
    """Piecewise-linear calibrator for scores in [0, 1]."""

    points: Tuple[CalibrationPoint, ...]
    clamp_input: bool = True

    def __post_init__(self) -> None:
        pts = _normalize_points(self.points)
        if not pts:
            raise ValueError("points must not be empty")
        object.__setattr__(self, "points", pts)

    def apply(self, value: float) -> float:
        """Calibrate a single value."""
        v = _as_float(value, name="value")
        if self.clamp_input:
            v = _clamp01(v)
        elif not (0.0 <= v <= 1.0):
            raise ValueError(f"value must be in [0, 1], got {v!r}")

        # Fast paths.
        if len(self.points) == 1:
            return self.points[0][1]

        # Piecewise-linear interpolation.
        for idx, (inp, out) in enumerate(self.points):
            if v == inp:
                return out
            if v < inp:
                if idx == 0:
                    return out
                prev_inp, prev_out = self.points[idx - 1]
                if inp == prev_inp:
                    return out
                ratio = (v - prev_inp) / (inp - prev_inp)
                return prev_out + ratio * (out - prev_out)
        return self.points[-1][1]

    def apply_many(self, values: Iterable[float]) -> list[float]:
        """Calibrate multiple values."""
        return [self.apply(v) for v in values]

    def to_json_dict(self) -> dict[str, Any]:
        return {"version": 1, "points": [[a, b] for a, b in self.points]}


def calibrate(value: float, points: Iterable[CalibrationPoint], *, clamp_input: bool = True) -> float:
    """Convenience wrapper for one-off calibration without keeping an object."""
    return TrustCalibrator(points=tuple(points), clamp_input=clamp_input).apply(value)


def load_calibrator(path: str | Path) -> TrustCalibrator:
    """Load a calibrator from a calibration JSON file."""
    p = Path(path)
    # "utf-8-sig" tolerates a UTF-8 BOM which is common in Windows tooling.
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    points = parse_points(data)
    return TrustCalibrator(points=tuple(points))


def try_load_calibrator(path: str | Path) -> Optional[TrustCalibrator]:
    """Like `load_calibrator`, but returns None when the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return load_calibrator(p)


def save_calibrator(calibrator: TrustCalibrator, path: str | Path) -> None:
    """Write a calibration JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calibrator.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")


def parse_points(obj: Any) -> list[CalibrationPoint]:
    """Parse points from the calibration JSON object."""
    if not isinstance(obj, Mapping):
        raise ValueError("calibration JSON must be an object/dict")
    raw = obj.get("points")
    if raw is None:
        raise ValueError('calibration JSON missing required key: "points"')
    if not isinstance(raw, list):
        raise ValueError('"points" must be a JSON array')

    points: list[CalibrationPoint] = []
    for idx, entry in enumerate(raw):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            points.append((_as_float(entry[0], name=f"points[{idx}][0]"), _as_float(entry[1], name=f"points[{idx}][1]")))
            continue
        if isinstance(entry, Mapping):
            if "input" not in entry or "output" not in entry:
                raise ValueError(f'points[{idx}] object must contain keys "input" and "output"')
            points.append((_as_float(entry["input"], name=f"points[{idx}].input"), _as_float(entry["output"], name=f"points[{idx}].output")))
            continue
        raise ValueError("each point must be a 2-item array or an object with input/output fields")
    return points


def _get_by_dotted_key(obj: Any, dotted_key: str) -> Any:
    cur = obj
    if dotted_key.strip() == "":
        return cur
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping):
            raise KeyError(f'cannot descend into "{part}": current value is not an object')
        if part not in cur:
            raise KeyError(f'missing key "{part}" while resolving "{dotted_key}"')
        cur = cur[part]
    return cur


def calibrator_from_buckets(
    buckets: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "confidence",
    accuracy_key: str = "accuracy",
    clamp_min: float = 0.05,
    clamp_max: float = 0.95,
    add_endpoints: bool = True,
) -> TrustCalibrator:
    """
    Build a calibrator from bucketed reliability data.

    Each bucket is expected to contain:
    - score_key: average predicted score/confidence for the bucket
    - accuracy_key: observed accuracy for the bucket
    """
    if clamp_min < 0.0 or clamp_max > 1.0 or clamp_min >= clamp_max:
        raise ValueError("clamp_min/clamp_max must satisfy 0 <= clamp_min < clamp_max <= 1")

    points: list[CalibrationPoint] = []
    for idx, bucket in enumerate(buckets):
        if not isinstance(bucket, Mapping):
            raise ValueError(f"bucket {idx} must be an object/dict")
        score = bucket.get(score_key)
        acc = bucket.get(accuracy_key)
        if score is None or acc is None:
            continue
        inp = _as_float(score, name=f"buckets[{idx}].{score_key}")
        out = _as_float(acc, name=f"buckets[{idx}].{accuracy_key}")
        out = max(clamp_min, min(clamp_max, out))
        points.append((inp, out))

    points = list(_normalize_points(points))
    if not points:
        raise ValueError("no usable buckets found (missing score/accuracy fields?)")

    if add_endpoints:
        if points[0][0] > 0.0:
            points.insert(0, (0.0, clamp_min))
        if points[-1][0] < 1.0:
            points.append((1.0, clamp_max))

    return TrustCalibrator(points=tuple(points))


def load_buckets_from_report(
    report_path: str | Path,
    *,
    buckets_key: str = "buckets",
) -> list[Mapping[str, Any]]:
    """
    Load a bucket list from a JSON report.

    Supported formats:
    - A list at the root: [ { ...bucket... }, ... ]
    - An object with a nested list at `buckets_key` (dotted path).
    """
    p = Path(report_path)
    # "utf-8-sig" tolerates a UTF-8 BOM which is common in Windows tooling.
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        buckets = data
    else:
        buckets = _get_by_dotted_key(data, buckets_key)

    if not isinstance(buckets, list):
        raise ValueError("buckets must be a JSON array")
    for idx, b in enumerate(buckets):
        if not isinstance(b, Mapping):
            raise ValueError(f"bucket {idx} must be an object/dict")
    return buckets  # type: ignore[return-value]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect a trust/confidence calibrator (piecewise-linear)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build a calibration JSON from a bucketed reliability report.")
    p_build.add_argument("--report", type=Path, required=True, help="Path to a JSON report.")
    p_build.add_argument("--out", type=Path, required=True, help="Where to write the calibration JSON.")
    p_build.add_argument(
        "--buckets-key",
        default="buckets",
        help='Dotted key path to the buckets array inside the report JSON (default: "buckets").',
    )
    p_build.add_argument(
        "--score-key",
        default="confidence",
        help='Bucket field for the average predicted score (default: "confidence").',
    )
    p_build.add_argument(
        "--accuracy-key",
        default="accuracy",
        help='Bucket field for observed accuracy (default: "accuracy").',
    )
    p_build.add_argument("--clamp-min", type=float, default=0.05, help="Lower clamp for accuracy.")
    p_build.add_argument("--clamp-max", type=float, default=0.95, help="Upper clamp for accuracy.")
    p_build.add_argument(
        "--no-endpoints",
        action="store_true",
        help="Do not automatically add endpoints at 0.0 and 1.0.",
    )

    p_show = sub.add_parser("show", help="Print the calibration mapping as JSON.")
    p_show.add_argument("--calibration", type=Path, required=True, help="Path to a calibration JSON file.")

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.cmd == "show":
        cal = load_calibrator(args.calibration)
        print(json.dumps(cal.to_json_dict(), indent=2, sort_keys=True))
        return

    if args.cmd == "build":
        buckets = load_buckets_from_report(args.report, buckets_key=args.buckets_key)
        cal = calibrator_from_buckets(
            buckets,
            score_key=args.score_key,
            accuracy_key=args.accuracy_key,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            add_endpoints=not args.no_endpoints,
        )
        save_calibrator(cal, args.out)
        return

    raise RuntimeError(f"unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()
