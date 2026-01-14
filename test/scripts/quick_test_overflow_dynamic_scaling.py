#!/usr/bin/env python3
"""
Quick synthetic test for overflow-driven "upgrade precision" events.

Goal
----
We want a *fast* and *isolated* test (no real flow equations / no JAX) to answer:

1) For different L, after how many micro-steps would FP16 storage overflow,
   forcing us to upgrade to a higher precision buffer (e.g. FP32)?
2) Does Dynamic Exponent Scaling (store T/mu in FP16 + store mu in FP32) delay
   or eliminate that overflow point?

This script uses a simple magnitude growth model for two tensors:
  - H2 (quadratic): size ~ n^2
  - H4 (quartic / "Hint"): size ~ n^4 (we only model its magnitude, not allocate it)

We define per-step max magnitude:
  mu2(step) = mu2_0(L) * growth(L)^step
  mu4(step) = mu4_0(L) * growth(L)^step

Overflow criterion (no scaling):
  mu > fp16_max (or a user-specified conservative threshold)

Overflow criterion (with exponent scaling):
  never overflows for FP16 storage (because max |T/mu| = 1),
  unless mu itself overflows float32 (practically unreachable in defaults).

Outputs
-------
- results.csv / results.json
- steps_to_overflow_vs_L.png (if matplotlib available)

Usage
-----
  python test/scripts/quick_test_overflow_dynamic_scaling.py
  python test/scripts/quick_test_overflow_dynamic_scaling.py --L 2 10 --max-steps 1000
  python test/scripts/quick_test_overflow_dynamic_scaling.py --threshold 1e4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


FP16_MAX = float(np.finfo(np.float16).max)  # ~ 65504
FP32_MAX = float(np.finfo(np.float32).max)  # ~ 3.4e38


@dataclass(frozen=True)
class ResultRow:
    L: int
    n: int
    growth: float
    mu0_h2: float
    mu0_h4: float
    # earliest step where "no scaling" would overflow fp16 (or threshold)
    step_overflow_no_scale: int | None
    # earliest step where "scaling" would overflow fp16 storage (normally None)
    step_overflow_scaled: int | None


def _mu0_model(L: int, kind: Literal["h2", "h4"], base: float, power: float) -> float:
    # simple scaling with system size to emulate "bigger systems have bigger magnitudes"
    if kind == "h2":
        return float(base * (L**power))
    # h4 tends to have larger dynamic range; give it a slightly stronger dependence by default
    return float(base * (L ** (power + 1.0)))


def _growth_model(L: int, alpha: float) -> float:
    # growth per micro-step; depends on n=L^2 (more terms -> faster magnitude growth)
    n = L * L
    return float(1.0 + alpha * n)


def _first_step_exceed(mu0: float, growth: float, threshold: float, max_steps: int) -> int | None:
    """
    Smallest integer step s in [0, max_steps] such that mu0 * growth^s > threshold.
    Returns None if never exceeds within max_steps.
    """
    if not (math.isfinite(mu0) and mu0 > 0.0 and math.isfinite(growth) and growth > 0.0):
        return 0
    if mu0 > threshold:
        return 0
    if growth <= 1.0:
        return None
    # Solve mu0 * growth^s > threshold  ->  s > log(threshold/mu0)/log(growth)
    rhs = math.log(threshold / mu0) / math.log(growth)
    s = int(math.floor(rhs) + 1)
    return s if s <= max_steps else None


def run(
    L_min: int,
    L_max: int,
    max_steps: int,
    alpha: float,
    mu0_base_h2: float,
    mu0_base_h4: float,
    mu0_power: float,
    threshold: float,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for L in range(L_min, L_max + 1):
        n = L * L
        g = _growth_model(L, alpha=alpha)
        mu0_h2 = _mu0_model(L, "h2", base=mu0_base_h2, power=mu0_power)
        mu0_h4 = _mu0_model(L, "h4", base=mu0_base_h4, power=mu0_power)

        # Without scaling: overflow when either tensor exceeds threshold
        s2 = _first_step_exceed(mu0_h2, g, threshold=threshold, max_steps=max_steps)
        s4 = _first_step_exceed(mu0_h4, g, threshold=threshold, max_steps=max_steps)
        if s2 is None and s4 is None:
            s_no = None
        elif s2 is None:
            s_no = s4
        elif s4 is None:
            s_no = s2
        else:
            s_no = min(s2, s4)

        # With exponent scaling: values stored are normalized to max=1 -> no FP16 overflow.
        # Only potential overflow is the scale itself (stored as fp32), which is extremely unlikely.
        # We still compute when mu would exceed fp32 max (if ever).
        s2_fp32 = _first_step_exceed(mu0_h2, g, threshold=FP32_MAX, max_steps=max_steps)
        s4_fp32 = _first_step_exceed(mu0_h4, g, threshold=FP32_MAX, max_steps=max_steps)
        if s2_fp32 is None and s4_fp32 is None:
            s_scaled = None
        elif s2_fp32 is None:
            s_scaled = s4_fp32
        elif s4_fp32 is None:
            s_scaled = s2_fp32
        else:
            s_scaled = min(s2_fp32, s4_fp32)

        rows.append(
            ResultRow(
                L=L,
                n=n,
                growth=g,
                mu0_h2=mu0_h2,
                mu0_h4=mu0_h4,
                step_overflow_no_scale=s_no,
                step_overflow_scaled=s_scaled,
            )
        )
    return rows


def write_outputs(out_dir: Path, rows: list[ResultRow], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {"meta": meta, "rows": [r.__dict__ for r in rows]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # CSV
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "L",
                "n",
                "growth",
                "mu0_h2",
                "mu0_h4",
                "step_overflow_no_scale",
                "step_overflow_scaled",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def plot(out_dir: Path, rows: list[ResultRow], max_steps: int, threshold: float) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print(f"[WARN] matplotlib not available; skipping plot. Data saved under: {out_dir}")
        return

    Ls = [r.L for r in rows]
    # For plotting, map "None" (no overflow within window) to max_steps+1.
    y_no = [r.step_overflow_no_scale if r.step_overflow_no_scale is not None else (max_steps + 1) for r in rows]
    y_sc = [r.step_overflow_scaled if r.step_overflow_scaled is not None else (max_steps + 1) for r in rows]

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(Ls, y_no, marker="o", label=f"No scaling (overflow>{threshold:g})")
    plt.plot(Ls, y_sc, marker="s", label="Exponent scaling (scale in FP32)")
    plt.xlabel("L")
    plt.ylabel(f"First step exceeding threshold (cap={max_steps+1} means none)")
    plt.title("Overflow point vs L (synthetic model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "steps_to_overflow_vs_L.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", nargs=2, type=int, default=[2, 10], help="L_min L_max (inclusive)")
    ap.add_argument("--max-steps", type=int, default=1000, help="Max micro-steps to scan")
    ap.add_argument(
        "--threshold",
        type=float,
        default=FP16_MAX,
        help=f"Overflow threshold for FP16 storage. Default is fp16_max={FP16_MAX:.0f}. "
        f"Use 1e4 to mirror the conservative heuristic in our code.",
    )
    ap.add_argument("--alpha", type=float, default=1e-3, help="Growth coefficient in growth(L)=1+alpha*(L^2)")
    ap.add_argument("--mu0-base-h2", type=float, default=1.0, help="Base magnitude for H2 at L=1")
    ap.add_argument("--mu0-base-h4", type=float, default=5.0, help="Base magnitude for H4 at L=1")
    ap.add_argument("--mu0-power", type=float, default=1.0, help="mu0(L) scales like L^power (H4 uses power+1)")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: test/overflow_bench/<timestamp>)")
    args = ap.parse_args()

    L_min, L_max = int(args.L[0]), int(args.L[1])
    max_steps = int(args.max_steps)
    threshold = float(args.threshold)

    repo_root = Path(__file__).resolve().parents[2]
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out_dir = repo_root / "test" / "overflow_bench" / run_id

    rows = run(
        L_min=L_min,
        L_max=L_max,
        max_steps=max_steps,
        alpha=float(args.alpha),
        mu0_base_h2=float(args.mu0_base_h2),
        mu0_base_h4=float(args.mu0_base_h4),
        mu0_power=float(args.mu0_power),
        threshold=threshold,
    )

    meta = {
        "L_min": L_min,
        "L_max": L_max,
        "max_steps": max_steps,
        "alpha": float(args.alpha),
        "threshold": threshold,
        "threshold_note": "fp16 overflow threshold (or conservative heuristic)",
        "fp16_max": FP16_MAX,
        "fp32_max": FP32_MAX,
        "mu0_base_h2": float(args.mu0_base_h2),
        "mu0_base_h4": float(args.mu0_base_h4),
        "mu0_power": float(args.mu0_power),
    }

    write_outputs(out_dir, rows, meta=meta)
    plot(out_dir, rows, max_steps=max_steps, threshold=threshold)

    print(f"Saved: {out_dir}")
    print("Tip: use --threshold 1e4 to emulate our FP16->FP32 upgrade heuristic.")


if __name__ == "__main__":
    main()

