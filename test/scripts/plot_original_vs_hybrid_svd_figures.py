#!/usr/bin/env python3
"""
Reproduce "figures.ipynb"-style plots with ORIGINAL vs HYBRID-SVD on the same axes.

This script reads the *processed* HDF5 outputs produced by `code/proc.py`, which live under:
  <PYFLOW_OUTDIR>/proc/fermion/d{dim}/data/{dis_type}/PT/{LIOM}/O{order}/static/dataN{n}/tflow-d{d}-O{order}-x{x}-Jz{delta}-p{p}.h5

To avoid overwriting, run original and hybrid-svd with different PYFLOW_OUTDIR roots, e.g.:
  PYFLOW_OUTDIR=/abs/path/pyflow_out_original  ...
  PYFLOW_OUTDIR=/abs/path/pyflow_out_hybrid_svd ...

Then point this script at those two roots.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE_DEFAULT = (REPO_ROOT / "code" / "pyflow_out").resolve()
PLOTS_DIR = (REPO_ROOT / "test" / "figures_hybrid_vs_original").resolve()


def _fmt_f(x: float, nd: int = 2) -> str:
    return f"{float(x):.{nd}f}"


def proc_h5_path(
    *,
    outdir: Path,
    dim: int,
    dis_type: str,
    L: int,
    d: float,
    p: int,
    order: int,
    liom: str,
    x: float,
    delta: float,
    dsymm: str = "charge",
    species_folder: str = "fermion",
) -> Path:
    n = int(L**dim)
    # Mirrors core.utility.namevar3() layout (processed outputs under "proc/...")
    return (
        outdir
        / "proc"
        / species_folder
        / f"d{dim}"
        / "data"
        / dis_type
        / "PT"
        / liom
        / f"O{order}"
        / "static"
        / f"dataN{n}"
        / f"tflow-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}.h5"
    )


def _read_h5(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as hf:
        out: dict[str, Any] = {}
        for k in ("itc", "ed_itc", "itc_nonint", "err_med", "trunc_err", "complexity", "ltc", "ltc2"):
            if k in hf:
                out[k] = np.array(hf[k][:])
        return out


def _finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = np.isfinite(x)
    if not np.any(m):
        return float("nan")
    return float(np.mean(x[m]))


def _finite_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = np.isfinite(x)
    if np.count_nonzero(m) < 2:
        return float("nan")
    return float(np.std(x[m]))


def _slice_window(y: np.ndarray, w0: int, w1: int) -> np.ndarray:
    y = np.asarray(y)
    return y[w0:w1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-outdir", type=str, default=str(OUT_BASE_DEFAULT), help="PYFLOW_OUTDIR root for original.")
    ap.add_argument("--hybrid-outdir", type=str, default=str(OUT_BASE_DEFAULT), help="PYFLOW_OUTDIR root for hybrid-svd.")
    ap.add_argument("--dim", type=int, choices=[1, 2], default=1)
    ap.add_argument("--dis", type=str, default="random", help="dis_type (e.g. random, QPgolden, linear, curved).")
    ap.add_argument("--L", type=int, nargs="+", default=[8, 10, 12, 16], help="List of L values.")
    ap.add_argument("--d", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0], help="List of disorder strengths d.")
    ap.add_argument("--p0", type=int, default=1)
    ap.add_argument("--reps", type=int, default=64, help="Number of p samples to average (reads p in [p0..p0+reps-1]).")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--liom", type=str, default="bck")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument(
        "--time-window",
        type=int,
        nargs=2,
        default=[-72, -43],
        help="Index window [start end) on itc(t) for finite-size scaling (matches figures.ipynb style).",
    )
    ap.add_argument("--title-suffix", type=str, default="")
    args = ap.parse_args()

    outdir_orig = Path(args.original_outdir).expanduser().resolve()
    outdir_hyb = Path(args.hybrid_outdir).expanduser().resolve()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = PLOTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tlist = np.logspace(-2, 5, 151, base=10, endpoint=True)
    w0, w1 = int(args.time_window[0]), int(args.time_window[1])

    # Aggregate containers
    curves: dict[str, dict[str, np.ndarray]] = {}
    err_table: list[dict[str, Any]] = []
    trunc_table: list[dict[str, Any]] = []

    for L in args.L:
        n = int(L**args.dim)
        curves[str(L)] = {}
        for d in args.d:
            # Correlation curves: average itc across p
            itc_o: list[np.ndarray] = []
            itc_h: list[np.ndarray] = []
            ed_o: list[np.ndarray] = []

            # Window averages
            win_o: list[float] = []
            win_h: list[float] = []
            win_ed: list[float] = []

            # Errors
            err_med_o: list[float] = []
            err_med_h: list[float] = []
            trunc_o: list[float] = []
            trunc_h: list[float] = []

            for p in range(int(args.p0), int(args.p0) + int(args.reps)):
                # ORIGINAL
                po = proc_h5_path(
                    outdir=outdir_orig,
                    dim=args.dim,
                    dis_type=args.dis,
                    L=L,
                    d=float(d),
                    p=int(p),
                    order=args.order,
                    liom=args.liom,
                    x=args.x,
                    delta=args.delta,
                )
                # HYBRID-SVD
                ph = proc_h5_path(
                    outdir=outdir_hyb,
                    dim=args.dim,
                    dis_type=args.dis,
                    L=L,
                    d=float(d),
                    p=int(p),
                    order=args.order,
                    liom=args.liom,
                    x=args.x,
                    delta=args.delta,
                )

                if po.exists():
                    data_o = _read_h5(po)
                    if "itc" in data_o:
                        y = np.asarray(data_o["itc"]).reshape(-1)
                        itc_o.append(y)
                        win_o.append(_finite_mean(_slice_window(y, w0, w1)))
                    if "ed_itc" in data_o and np.size(data_o["ed_itc"]) > 1:
                        yed = np.asarray(data_o["ed_itc"]).reshape(-1)
                        ed_o.append(yed)
                        win_ed.append(_finite_mean(_slice_window(yed, w0, w1)))
                    if "err_med" in data_o:
                        err_med_o.append(float(np.asarray(data_o["err_med"]).reshape(-1)[0]))
                    if "trunc_err" in data_o:
                        # proc.py stores [trunc_err1, trunc_err2]; use trunc_err1 as "per-flow-time" metric.
                        trunc_o.append(float(np.asarray(data_o["trunc_err"]).reshape(-1)[0]))

                if ph.exists():
                    data_h = _read_h5(ph)
                    if "itc" in data_h:
                        y = np.asarray(data_h["itc"]).reshape(-1)
                        itc_h.append(y)
                        win_h.append(_finite_mean(_slice_window(y, w0, w1)))
                    if "err_med" in data_h:
                        err_med_h.append(float(np.asarray(data_h["err_med"]).reshape(-1)[0]))
                    if "trunc_err" in data_h:
                        trunc_h.append(float(np.asarray(data_h["trunc_err"]).reshape(-1)[0]))

            # Save tables (per (L,d))
            err_table.append(
                {
                    "dim": int(args.dim),
                    "dis": str(args.dis),
                    "L": int(L),
                    "n": int(n),
                    "d": float(d),
                    "err_med_abs_mean_original": float(np.mean(np.abs(err_med_o))) if err_med_o else float("nan"),
                    "err_med_abs_mean_hybrid_svd": float(np.mean(np.abs(err_med_h))) if err_med_h else float("nan"),
                    "count_original": int(len(err_med_o)),
                    "count_hybrid_svd": int(len(err_med_h)),
                }
            )
            trunc_table.append(
                {
                    "dim": int(args.dim),
                    "dis": str(args.dis),
                    "L": int(L),
                    "n": int(n),
                    "d": float(d),
                    "trunc_err1_mean_original": float(np.mean(trunc_o)) if trunc_o else float("nan"),
                    "trunc_err1_mean_hybrid_svd": float(np.mean(trunc_h)) if trunc_h else float("nan"),
                    "count_original": int(len(trunc_o)),
                    "count_hybrid_svd": int(len(trunc_h)),
                }
            )

            # Store one representative curve for this (L,d): average across p
            key = f"L{L}_d{d:.2f}"
            if itc_o:
                curves[str(L)][f"{key}:original"] = np.mean(np.stack(itc_o, axis=0), axis=0)
            if itc_h:
                curves[str(L)][f"{key}:hybrid-svd"] = np.mean(np.stack(itc_h, axis=0), axis=0)
            if ed_o:
                curves[str(L)][f"{key}:ED(original)"] = np.mean(np.stack(ed_o, axis=0), axis=0)

    # Plot: correlation curves for each L (overlay original vs hybrid-svd vs ED)
    import matplotlib.pyplot as plt

    for L in args.L:
        if not curves[str(L)]:
            continue
        plt.figure(figsize=(10, 6))
        for label, y in curves[str(L)].items():
            if y is None or len(np.asarray(y).reshape(-1)) == 0:
                continue
            plt.plot(tlist, np.asarray(y).reshape(-1), linewidth=2, label=label)
        plt.xscale("log")
        plt.xlabel(r"$Jt$")
        plt.ylabel(r"$C(t)$")
        title = f"Correlation dynamics (dim={args.dim}, dis={args.dis}, L={L})"
        if args.title_suffix:
            title += f" | {args.title_suffix}"
        plt.title(title)
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        out_png = out_dir / f"fig_corr_dim{args.dim}_{args.dis}_L{L}.png"
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()

    # Save tables
    (out_dir / "ed_err_table.json").write_text(json.dumps(err_table, indent=2))
    (out_dir / "trunc_err_table.json").write_text(json.dumps(trunc_table, indent=2))

    print(f"Saved plots and tables to: {out_dir}")


if __name__ == "__main__":
    main()

