#!/usr/bin/env python3
"""
Toy "flow-like" precision staircase test (H2/H4 coupled, no real physics).

Plot (as requested)
-------------------
- x-axis: micro-step index
- y-axis: precision level (low -> high)
- for each L: two staircase curves (dynamic exponent scaling ON vs OFF)

This is a *toy* model: it does NOT run the real flow equations. It only mimics the
"shape" of a coupled update and how peak tensor magnitudes might grow with L/steps.

Outputs
-------
Creates: test/precision_stairs/<timestamp>/
  - series.csv
  - results.json
  - precision_stairs.png (if matplotlib available)
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

# Dependency-free (no numpy). IEEE-754 maxima:
FP16_MAX = 65504.0
FP32_MAX = 3.4028235e38

Precision = Literal["fp16", "fp32", "fp64"]


def _precision_level(p: Precision) -> int:
    # low -> high
    return {"fp16": 0, "fp32": 1, "fp64": 2}[p]


@dataclass(frozen=True)
class StepPoint:
    step: int
    L: int
    mode: Literal["no_dynamic", "dynamic"]
    mu: float
    precision: Precision
    level: int


def toy_microstep_update(a2: float, a4: float, L: int, dt: float) -> tuple[float, float]:
    """
    A tiny coupled update that resembles a "flow-like" micro-step:
      a2 <- a2 + dt * F2(a2,a4)
      a4 <- a4 + dt * F4(a2,a4)

    Heuristic coefficients chosen so that:
    - larger L reaches higher mu sooner
    - staircases appear within O(10^2) steps for L~6..10 (with default params)
    """
    n = L * L

    # "Diagonalization" decay of quadratic off-diagonal part
    k_decay2 = 0.08 + 0.0008 * n

    # Coupling H4 -> H2 (can inject back into a2)
    k_42 = 0.02 + 0.0002 * n

    # Growth term for quartic sector (captures dynamic-range pain in toy form)
    k_grow4 = 0.015 + 0.0009 * n

    # Coupling H2 -> H4 (quadratic driving quartic)
    k_24 = 0.006 + 0.00015 * n

    # Mild saturation to avoid instant blow-up for small L
    sat = 1.0 + 0.02 * abs(a4)

    da2 = (-k_decay2 * a2 + k_42 * a4) * dt
    da4 = (k_grow4 * a4 + k_24 * (a2 * a2)) * dt / sat

    return (a2 + da2), (a4 + da4)


def mu_proxy(a2: float, a4: float, L: int) -> float:
    """
    Proxy for checkpoint peak magnitude. We avoid allocating real tensors:
      mu2 ~ |a2| * L^2   (quadratic scales like n)
      mu4 ~ |a4| * L^4   (quartic scales like n^2)
    """
    mu2 = abs(a2) * (L * L)
    mu4 = abs(a4) * (L**4)
    return float(max(mu2, mu4))


def choose_precision_no_dynamic(mu: float, fp16_threshold: float) -> Precision:
    """
    No exponent scaling: store raw tensors in the chosen dtype.
    Upgrade when mu exceeds dtype max/threshold.
    """
    if (not math.isfinite(mu)) or mu > FP32_MAX:
        return "fp64"
    if mu > fp16_threshold:
        return "fp32"
    return "fp16"


def choose_precision_dynamic(mu: float) -> Precision:
    """
    With exponent scaling: store normalized T/mu in FP16 (never overflows),
    store mu as FP32. Upgrade only when mu cannot fit FP32.
    """
    if (not math.isfinite(mu)) or mu > FP32_MAX:
        return "fp64"
    return "fp16"


def run_for_L(L: int, steps: int, dt: float, a2_0: float, a4_0: float, fp16_threshold: float) -> list[StepPoint]:
    pts: list[StepPoint] = []
    a2 = float(a2_0)
    a4 = float(a4_0)
    for s in range(steps + 1):
        mu = mu_proxy(a2, a4, L)
        p0 = choose_precision_no_dynamic(mu, fp16_threshold=fp16_threshold)
        p1 = choose_precision_dynamic(mu)
        pts.append(StepPoint(step=s, L=L, mode="no_dynamic", mu=mu, precision=p0, level=_precision_level(p0)))
        pts.append(StepPoint(step=s, L=L, mode="dynamic", mu=mu, precision=p1, level=_precision_level(p1)))
        a2, a4 = toy_microstep_update(a2, a4, L=L, dt=dt)
    return pts


def write_outputs(out_dir: Path, pts: list[StepPoint], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "points": [
                    {
                        "step": p.step,
                        "L": p.L,
                        "mode": p.mode,
                        "mu": p.mu,
                        "precision": p.precision,
                        "level": p.level,
                    }
                    for p in pts
                ],
            },
            indent=2,
        )
    )

    with (out_dir / "series.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "L", "mode", "mu", "precision", "level"])
        for p in pts:
            w.writerow([p.step, p.L, p.mode, f"{p.mu:.6g}", p.precision, p.level])


def plot(out_dir: Path, pts: list[StepPoint], L_min: int, L_max: int) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print(f"[WARN] matplotlib not available; skipping plot. Data saved under: {out_dir}")
        return

    series: dict[tuple[int, str], list[StepPoint]] = {}
    for p in pts:
        series.setdefault((p.L, p.mode), []).append(p)
    for k in series:
        series[k].sort(key=lambda x: x.step)

    plt.figure(figsize=(10.2, 5.4))
    for L in range(L_min, L_max + 1):
        for mode, ls in (("no_dynamic", "-"), ("dynamic", "--")):
            s = series.get((L, mode), [])
            if not s:
                continue
            xs = [p.step for p in s]
            ys = [p.level for p in s]
            label = f"L={L} {'dyn' if mode=='dynamic' else 'no-dyn'}"
            plt.step(xs, ys, where="post", linestyle=ls, linewidth=1.3, alpha=0.85, label=label)

    plt.yticks([0, 1, 2], ["fp16", "fp32", "fp64"])
    plt.xlabel("micro-step")
    plt.ylabel("required storage precision (low→high)")
    plt.title("Precision staircase vs step (toy H2/H4-coupled flow model)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_stairs.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", nargs=2, type=int, default=[2, 10], help="L_min L_max (inclusive)")
    ap.add_argument("--steps", type=int, default=250, help="Number of micro-steps to simulate")
    ap.add_argument("--dt", type=float, default=1.0, help="Toy micro-step size")
    ap.add_argument(
        "--fp16-threshold",
        type=float,
        default=1e4,
        help=f"FP16 upgrade threshold (default 1e4). Use {FP16_MAX:.0f} for true FP16 max.",
    )
    ap.add_argument("--a2-0", type=float, default=1.0, help="Initial a2 amplitude")
    ap.add_argument("--a4-0", type=float, default=2e-3, help="Initial a4 amplitude")
    ap.add_argument("--out", type=str, default=None, help="Output directory (optional)")
    args = ap.parse_args()

    L_min, L_max = int(args.L[0]), int(args.L[1])
    steps = int(args.steps)
    dt = float(args.dt)
    fp16_threshold = float(args.fp16_threshold)

    repo_root = Path(__file__).resolve().parents[2]
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out_dir = repo_root / "test" / "precision_stairs" / run_id

    pts: list[StepPoint] = []
    for L in range(L_min, L_max + 1):
        pts.extend(
            run_for_L(
                L=L,
                steps=steps,
                dt=dt,
                a2_0=float(args.a2_0),
                a4_0=float(args.a4_0),
                fp16_threshold=fp16_threshold,
            )
        )

    meta: dict[str, Any] = {
        "L_min": L_min,
        "L_max": L_max,
        "steps": steps,
        "dt": dt,
        "fp16_threshold": fp16_threshold,
        "fp16_max": FP16_MAX,
        "fp32_max": FP32_MAX,
        "a2_0": float(args.a2_0),
        "a4_0": float(args.a4_0),
        "note": "Toy coupled update; not the real flow equations.",
    }
    write_outputs(out_dir, pts, meta=meta)
    plot(out_dir, pts, L_min=L_min, L_max=L_max)

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

