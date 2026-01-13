#!/usr/bin/env python3
"""
1D ITC vs ED comparison for Original vs Hybrid.

Why 1D:
- The published notebook (`figures.ipynb`) compares ED to FE mainly in 1D.
- 2D ITC can be noisier and the pipeline is less validated.

What this script does:
- Runs `code/main_itc_cpu.py` for original and hybrid modes
- Uses hybrid cutoff=1e-6
- Extracts `itc`, `itc_nonint`, and `ed_itc` from the output HDF5
- Applies the same preprocessing conventions used in `code/proc.py` / `figures.ipynb`
  (×4 scaling and optional non-interacting calibration + normalization)
- Averages over multiple repeats (different p values) to reduce Monte-Carlo noise
- Saves a plot and a JSON summary

Usage:
  python test/scripts/test_hybrid_itc_vs_ed_1d.py 12 12 --reps 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
OUT_DIR = REPO_ROOT / "test" / "accuracy_test"

DEFAULT_DIS_TYPE = "random"
DEFAULT_METHOD = "tensordot"
DEFAULT_D = 1.0
DEFAULT_P0 = 0

ODE_TOL = "1e-6"
ORIGINAL_CUTOFF = "1e-4"
HYBRID_CUTOFF = "1e-6"

PROCESS_TIMEOUT = 360000


@dataclass(frozen=True)
class ModeSpec:
    name: str
    use_ckpt: str
    cutoff: str


MODES = [
    ModeSpec("original", "0", ORIGINAL_CUTOFF),
    ModeSpec("hybrid", "hybrid", HYBRID_CUTOFF),
]


def _get_num_cores() -> int:
    try:
        from psutil import cpu_count as _psutil_cpu_count  # type: ignore

        n = _psutil_cpu_count(logical=False)
        if n is None:
            n = _psutil_cpu_count(logical=True)
        return int(n) if n else (os.cpu_count() or 1)
    except Exception:
        return int(os.cpu_count() or 1)


_NUM_CORES = str(_get_num_cores())


BASE_ENV = {
    "PYFLOW_MEMLOG": "0",
    "PYFLOW_TIMELOG": "0",
    "PYFLOW_SCRAMBLE": "0",
    "PYFLOW_OVERWRITE": "1",
    "PYFLOW_ODE_RTOL": ODE_TOL,
    "PYFLOW_ODE_ATOL": ODE_TOL,
    # Ensure ITC datasets are written (main_itc_cpu.py defaults ladder on, but keep explicit)
    "PYFLOW_LADDER": "1",
    "PYFLOW_ITC": "0",
    # For accuracy comparisons, prefer a numerically safe hybrid buffer dtype.
    # (Hybrid mode will still default to FP16 where safe, but forcing FP32 avoids overflow artifacts.)
    "PYFLOW_HYBRID_BUFFER_DTYPE": "float32",
    # Avoid oversubscription
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={_NUM_CORES} inter_op_parallelism_threads=1",
}


def h5_path_for_1d(L: int, dis_type: str, d: float, p: int) -> Path:
    dim = 1
    order = 4
    x = 0.0
    delta = 0.1
    n = L**dim
    pot_folder = dis_type
    return (
        CODE_DIR
        / "pyflow_out"
        / "fermion"
        / "d1"
        / "data"
        / pot_folder
        / "PT"
        / "bck"
        / f"O{order}"
        / "static"
        / f"dataN{n}"
        / f"tflow-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}.h5"
    )


def run_case(L: int, mode: ModeSpec, dis_type: str, method: str, d: float, p: int) -> None:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["USE_CKPT"] = mode.use_ckpt
    env["PYFLOW_CUTOFF"] = mode.cutoff

    cmd = [sys.executable, str(CODE_DIR / "main_itc_cpu.py"), str(L), dis_type, method, str(d), str(p)]
    print(f"    Run p={p:3d} mode={mode.name:8s} cutoff={mode.cutoff} ...", end="", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=CODE_DIR, env=env, capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f" [FAILED] ({dt:.1f}s)")
        tail = (res.stderr or "").splitlines()[-12:]
        for line in tail:
            print(f"      ! {line}")
        raise RuntimeError(f"Run failed for mode={mode.name}, p={p}")
    print(f" [OK] ({dt:.1f}s)")


def read_itc_bundle(h5_path: Path) -> dict:
    import h5py

    with h5py.File(h5_path, "r") as hf:
        if "itc" not in hf:
            raise KeyError("Missing dataset `itc` in H5.")
        if "ed_itc" not in hf:
            raise KeyError("Missing dataset `ed_itc` in H5 (ED may be disabled or n too large).")
        itc = np.array(hf["itc"][:])
        itc_nonint = np.array(hf["itc_nonint"][:]) if "itc_nonint" in hf else None
        ed_itc = np.array(hf["ed_itc"][:])
        return {"itc": itc, "itc_nonint": itc_nonint, "ed_itc": ed_itc}


def preprocess_like_proc(itc: np.ndarray, itc_nonint: np.ndarray | None, ed_itc: np.ndarray) -> dict:
    """
    Match the plotting conventions used in `code/proc.py` / `figures.ipynb`:
    - multiply by 4 so that ED starts at ~1 (raw ED starts at ~0.25)
    - optional non-interacting calibration using `itc_nonint`
    - normalize so C(0)=1 for ED and FE
    """
    itc = np.real(np.array(itc))
    ed_itc = np.real(np.array(ed_itc))

    itc = 4.0 * itc
    ed_itc = 4.0 * ed_itc

    if itc_nonint is not None:
        itc_nonint = 4.0 * np.real(np.array(itc_nonint))

    if itc.ndim > 1:
        itc = np.mean(itc, axis=0)
    if ed_itc.ndim > 1:
        ed_itc = np.mean(ed_itc, axis=0)
    if itc_nonint is not None and itc_nonint.ndim > 1:
        itc_nonint = np.mean(itc_nonint, axis=0)

    x_best = 0.0
    if itc_nonint is not None:
        nfit = min(75, itc.size, itc_nonint.size)
        if nfit >= 5 and np.isfinite(itc[:nfit]).all() and np.isfinite(itc_nonint[:nfit]).all():
            xs = np.linspace(0.0, 1.0, 101, endpoint=True)
            errs = np.empty_like(xs)
            base = np.array(itc, copy=True)
            for i, x in enumerate(xs):
                test = base.copy()
                test += x * (1.0 - test[0])
                test *= 1.0 / test[0]
                denom = np.maximum(np.abs(itc_nonint[:nfit]), 1e-12)
                errs[i] = float(np.mean(np.abs((test[:nfit] - itc_nonint[:nfit]) / denom)))
            x_best = float(xs[int(np.argmin(errs))])
            itc = itc + x_best * (1.0 - itc[0])
            itc = itc / itc[0]

    # normalize ED and nonint
    if ed_itc.size and np.isfinite(ed_itc[0]) and abs(ed_itc[0]) > 0:
        ed_itc = ed_itc / ed_itc[0]
    if itc_nonint is not None and itc_nonint.size and np.isfinite(itc_nonint[0]) and abs(itc_nonint[0]) > 0:
        itc_nonint = itc_nonint / itc_nonint[0]

    return {"itc": itc, "ed_itc": ed_itc, "itc_nonint": itc_nonint, "x_best": x_best}


def compute_errors(itc: np.ndarray, ed_itc: np.ndarray) -> dict:
    itc = np.real(np.array(itc)).reshape(-1)
    ed_itc = np.real(np.array(ed_itc)).reshape(-1)
    if itc.shape != ed_itc.shape:
        raise ValueError(f"Shape mismatch: itc={itc.shape}, ed_itc={ed_itc.shape}")
    mask = np.isfinite(itc) & np.isfinite(ed_itc)
    n_total = int(itc.size)
    n_finite = int(np.count_nonzero(mask))
    if n_finite == 0:
        return {"error": "All entries non-finite.", "n_total": n_total}
    diff = itc[mask] - ed_itc[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    denom = np.maximum(np.abs(ed_itc[mask]), 1e-12)
    rel = np.abs(diff) / denom
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_rel": float(np.mean(rel)),
        "max_rel": float(np.max(rel)),
    }


def plot_curves(t: np.ndarray, curves: dict, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    for label, y in curves.items():
        plt.plot(t, y, linewidth=2, label=label)
    plt.xscale("log")
    plt.xlabel(r"$Jt$")
    plt.ylabel(r"$C(t)$")
    plt.title(title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("L_min", type=int)
    ap.add_argument("L_max", type=int, nargs="?", default=None)
    ap.add_argument("--reps", type=int, default=8, help="Number of repeats (p values) to average.")
    ap.add_argument("--dis", type=str, default=DEFAULT_DIS_TYPE)
    ap.add_argument("--method", type=str, default=DEFAULT_METHOD)
    ap.add_argument("--d", type=float, default=DEFAULT_D)
    ap.add_argument("--p0", type=int, default=DEFAULT_P0)
    args = ap.parse_args()

    L_min = args.L_min
    L_max = args.L_max if args.L_max is not None else L_min

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tlist = np.logspace(-2, 5, 151, base=10, endpoint=True)

    results: dict[str, dict] = {}

    for L in range(L_min, L_max + 1):
        print("\n" + "=" * 90)
        print(f"1D ITC vs ED (L={L}, n={L})  dis={args.dis}  d={args.d}  reps={args.reps}")
        print("=" * 90)

        per_mode_curves = {}
        per_mode_meta = {}

        for mode in MODES:
            itc_acc = []
            ed_acc = []
            x_best_list = []
            for p in range(args.p0, args.p0 + args.reps):
                run_case(L, mode, args.dis, args.method, args.d, p)
                h5p = h5_path_for_1d(L, args.dis, args.d, p)
                if not h5p.exists():
                    raise FileNotFoundError(f"Expected output not found: {h5p}")
                data = read_itc_bundle(h5p)
                pre = preprocess_like_proc(data["itc"], data["itc_nonint"], data["ed_itc"])
                itc_acc.append(pre["itc"])
                ed_acc.append(pre["ed_itc"])
                x_best_list.append(pre["x_best"])

            itc_mean = np.mean(np.stack(itc_acc, axis=0), axis=0)
            ed_mean = np.mean(np.stack(ed_acc, axis=0), axis=0)
            per_mode_curves[mode.name] = {"itc": itc_mean, "ed": ed_mean}
            per_mode_meta[mode.name] = {"x_best_mean": float(np.mean(x_best_list)), "x_best_std": float(np.std(x_best_list))}

        # Use ED from original for plotting (ED should match closely across modes, just noise)
        curves = {
            "ED (avg)": per_mode_curves["original"]["ed"],
            "Original (FE, avg)": per_mode_curves["original"]["itc"],
            "Hybrid (FE, avg)": per_mode_curves["hybrid"]["itc"],
        }

        out_png = OUT_DIR / f"itc_vs_ed_1d_L{L}_d{args.d:.2f}_reps{args.reps}.png"
        plot_curves(tlist, curves, out_png, title=f"1D ITC vs ED (L={L}, n={L})")
        print(f"  Plot saved: {out_png}")

        # Errors vs ED (avg)
        err_orig = compute_errors(per_mode_curves["original"]["itc"], per_mode_curves["original"]["ed"])
        err_hyb = compute_errors(per_mode_curves["hybrid"]["itc"], per_mode_curves["original"]["ed"])

        print(f"  original: rmse={err_orig.get('rmse', float('nan')):.3e} max_abs={err_orig.get('max_abs', float('nan')):.3e}")
        print(f"  hybrid  : rmse={err_hyb.get('rmse', float('nan')):.3e} max_abs={err_hyb.get('max_abs', float('nan')):.3e}")

        results[str(L)] = {
            "L": L,
            "n": L,
            "params": {"dis": args.dis, "d": args.d, "reps": args.reps, "p0": args.p0, "method": args.method},
            "meta": per_mode_meta,
            "errors": {"original_vs_ed": err_orig, "hybrid_vs_ed": err_hyb},
            "plot": str(out_png),
        }

    out_json = OUT_DIR / "itc_vs_ed_1d_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results JSON: {out_json}")


if __name__ == "__main__":
    main()


