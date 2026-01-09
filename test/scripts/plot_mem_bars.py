#!/usr/bin/env python3
"""
Plot memory usage vs L for each mode (original/ckpt/recursive/hybrid).

This script:
- Parses memlog JSONL files under test/<mode>/ produced by main_itc_cpu_d2.py + memlog().
- Uses the same metric as benchmark_ckpt.py: Net RSS = Peak(RSS after main:before_flow) - Baseline(RSS at main:before_flow).
- Treats L<=5 as "ground truth" (per user note) and fits a simple scaling model vs n=L^2, dominated by n^4.
- Predicts L=6..10 from the fitted model, then plots grouped bar charts for L=3..10.

Output:
- Prints a table to stdout
- Saves figure to test/memory_bars_L3-10.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "test"

MODES = ["original", "ckpt", "recursive", "hybrid"]
L_MIN, L_MAX = 3, 10

# Fair-comparison settings (requested):
# - float32 compute/storage for H2/H4 (except hybrid buffer which is fp16 by design)
# - base-case length aligned across recursive/hybrid
DIM = 2
T_STEPS = 1000  # qmax/len(dl_list) in main_itc_cpu_d2.py default
CKPT_STEP = 20
BASE_CASE_STEPS = 20
BYTES_F32 = 4
BYTES_F16 = 2


def _parse_L_from_name(name: str) -> Optional[int]:
    # Example: memlog-dim2-L6-d1.00-O4-x0.00-Jz0.10-p0-original.jsonl
    for part in name.split("-"):
        if part.startswith("L") and part[1:].isdigit():
            return int(part[1:])
    return None


def parse_memlog_net_mb(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return None

    rows: List[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None

    baseline = None
    flow_start_idx = 0
    for i, r in enumerate(rows):
        if r.get("tag") == "main:before_flow":
            baseline = r.get("rss_mb", None)
            flow_start_idx = i
            break
    if baseline is None:
        rss_all = [r.get("rss_mb") for r in rows if "rss_mb" in r]
        baseline = min(rss_all) if rss_all else None
        flow_start_idx = 0

    flow_rows = rows[flow_start_idx:]
    rss_vals = [r.get("rss_mb") for r in flow_rows if "rss_mb" in r]
    if not rss_vals or baseline is None:
        return None

    peak = float(max(rss_vals))
    net = float(max(0.0, peak - float(baseline)))
    peak_row = max((r for r in flow_rows if "rss_mb" in r), key=lambda r: r["rss_mb"])

    return {
        "baseline_mb": float(baseline),
        "peak_mb": peak,
        "net_mb": net,
        "entries": len(rows),
        "peak_tag": peak_row.get("tag"),
        "peak_step": peak_row.get("step"),
    }


def find_best_memlog(mode: str, L: int) -> Optional[Path]:
    """
    Prefer test/<mode>/memlog-...-<mode>.jsonl, but fall back to any matching file under test/.
    If multiple exist, pick the one with the most parseable entries.
    """
    candidates: List[Path] = []
    mode_dir = TEST_DIR / mode
    if mode_dir.exists():
        candidates.extend(sorted(mode_dir.glob(f"memlog-*-L{L}-*-{mode}.jsonl")))
    # fallback: sometimes files live directly under test/
    candidates.extend(sorted(TEST_DIR.glob(f"memlog-*-L{L}-*-{mode}.jsonl")))

    best = None
    best_entries = -1
    for p in candidates:
        info = parse_memlog_net_mb(p)
        if info is None:
            continue
        if info["entries"] > best_entries:
            best_entries = info["entries"]
            best = p
    return best


def fit_net_model(train_L: List[int], train_net_mb: List[float]) -> Tuple[np.ndarray, float]:
    """
    Fit net_mb ~ a*n^4 + b (simple 2-parameter model).
    We keep it minimal and stable with few points.
    """
    L_arr = np.array(train_L, dtype=np.float64)
    n = (L_arr**2).astype(np.float64)
    x = n**4
    y = np.array(train_net_mb, dtype=np.float64)

    # Design matrix [x, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeff
    return coeff, float(np.sqrt(np.mean((A @ coeff - y) ** 2)))  # rmse


def predict_net_mb(coeff: np.ndarray, L: int) -> float:
    n = float(L * L)
    x = n**4
    a, b = float(coeff[0]), float(coeff[1])
    return float(max(0.0, a * x + b))


def estimate_net_mb_formula(mode: str, L: int) -> float:
    """
    Fast/consistent estimator dominated by storage of n^4 tensors.
    dim=2 => n = L^2.
    Returns MB (decimal, 1e6 bytes).
    """
    n = (L**DIM)
    n4 = float(n**4)

    if mode == "original":
        # Full flow storage ~ T snapshots of H4 (dominant term)
        bytes_ = T_STEPS * n4 * BYTES_F32
    elif mode == "ckpt":
        # Linear checkpoint: store ~T/ckpt checkpoints (each holds H4) + segment buffer of length ckpt_step
        checkpoints = (T_STEPS / CKPT_STEP)
        bytes_ = (checkpoints + CKPT_STEP) * n4 * BYTES_F32
    elif mode == "recursive":
        # Recursive base-case trajectory buffer in float32
        bytes_ = BASE_CASE_STEPS * n4 * BYTES_F32
    elif mode == "hybrid":
        # Hybrid base-case contiguous buffer in float16
        bytes_ = BASE_CASE_STEPS * n4 * BYTES_F16
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(bytes_ / 1e6)


def main() -> None:
    # Collect measured values
    measured: Dict[str, Dict[int, float]] = {m: {} for m in MODES}
    meta: Dict[str, Dict[int, dict]] = {m: {} for m in MODES}

    for mode in MODES:
        for L in range(L_MIN, L_MAX + 1):
            p = find_best_memlog(mode, L)
            if not p:
                continue
            info = parse_memlog_net_mb(p)
            if not info:
                continue
            measured[mode][L] = info["net_mb"]
            meta[mode][L] = {"file": str(p), **info}

    # Build formula series (always available) and optionally overlay measured values
    values: Dict[str, List[float]] = {m: [] for m in MODES}
    tags: Dict[str, List[str]] = {m: [] for m in MODES}  # "formula"
    Ls = list(range(L_MIN, L_MAX + 1))
    for mode in MODES:
        for L in Ls:
            values[mode].append(estimate_net_mb_formula(mode, L))
            tags[mode].append("formula")

    # Print table
    print("\nFormula estimate (MB): dominant n^4 storage terms, with BASE_CASE_STEPS=20 and float32 (hybrid buffer float16).")
    print(f"Assumptions: dim={DIM}, T_STEPS={T_STEPS}, CKPT_STEP={CKPT_STEP}, BASE_CASE_STEPS={BASE_CASE_STEPS}, float32 bytes={BYTES_F32}")
    print("\n" + f"{'L':>2}  " + "  ".join([f"{m:>10s}" for m in MODES]) + "   (measured net MB if available)")
    for i, L in enumerate(Ls):
        row = [f"{L:>2}"]
        for m in MODES:
            v = values[m][i]
            row.append(f"{v:10.0f}")

        # attach measured if present
        meas_parts = []
        for m in MODES:
            if L in measured[m]:
                meas_parts.append(f"{m}={measured[m][L]:.0f}")
        meas_str = ("   meas: " + ", ".join(meas_parts)) if meas_parts else ""
        print("  ".join(row) + meas_str)
    print()

    # Plot grouped bars (log scale for readability)
    x = np.arange(len(Ls))
    width = 0.2
    offsets = {
        "original": -1.5 * width,
        "ckpt": -0.5 * width,
        "recursive": 0.5 * width,
        "hybrid": 1.5 * width,
    }
    colors = {
        "original": "#d62728",
        "ckpt": "#1f77b4",
        "recursive": "#2ca02c",
        "hybrid": "#9467bd",
    }

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    for mode in MODES:
        y = np.array(values[mode], dtype=np.float64)
        bars = ax.bar(x + offsets[mode], y, width=width, label=mode, color=colors[mode], alpha=0.85)
        # all bars are formula-based; no hatching needed

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("L (dim=2, n=L^2)")
    ax.set_ylabel("memory (MB) [log scale]")
    ax.set_yscale("log")
    ax.set_title("Peak memory by mode")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(ncols=4, fontsize=10)
    plt.tight_layout()

    out = TEST_DIR / "memory_bars_L3-10.png"
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


