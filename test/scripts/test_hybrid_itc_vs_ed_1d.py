#!/usr/bin/env python3
"""
1D ITC vs ED comparison for Original vs Hybrid-SVD (full hybrid implementation).

Why 1D:
- The published notebook (`figures.ipynb`) compares ED to FE mainly in 1D.
- 2D ITC can be noisier and the pipeline is less validated.

What this script does:
- Runs `code/main_itc_cpu.py` for original and hybrid-svd modes
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
    extra_env: dict[str, str] | None = None


MODES = [
    ModeSpec(
        "original",
        "0",
        ORIGINAL_CUTOFF,
        extra_env={
            # Be explicit: avoid inheriting any hybrid flags from the shell.
            "PYFLOW_USE_COPY_ROUTINES": "0",
            "PYFLOW_HYBRID_SVD": "0",
            "PYFLOW_HYBRID_COMPRESS": "",
            # Ensure ED is enabled for the baseline run.
            "PYFLOW_SKIP_ED": "0",
        },
    ),
    ModeSpec(
        "hybrid-svd",
        "hybrid",
        HYBRID_CUTOFF,
        extra_env={
            # Use the "complete" hybrid implementation from spinless_fermion copy.py
            "PYFLOW_USE_COPY_ROUTINES": "1",
            # Enable projection-based compression path (Hybrid-SVD / rSVD)
            "PYFLOW_HYBRID_SVD": "1",
            "PYFLOW_HYBRID_COMPRESS": "hybrid-svd",
            # Keep explicit for reproducibility (defaults are the same).
            "PYFLOW_HYBRID_EXP_SCALE": "1",
            "PYFLOW_HYBRID_PRUNE": "0",
            # Critical speedup: ED is identical across modes for the same disorder (p),
            # so we compute ED only once in the original run and reuse it for hybrid preprocessing.
            "PYFLOW_SKIP_ED": "1",
        },
    ),
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
    if mode.extra_env:
        env.update(mode.extra_env)
    # Allow callers to "unset" a var by passing an empty string.
    if env.get("PYFLOW_FORCE_STEPS", "").strip() == "":
        env.pop("PYFLOW_FORCE_STEPS", None)

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
        itc = np.array(hf["itc"][:])
        itc_nonint = np.array(hf["itc_nonint"][:]) if "itc_nonint" in hf else None
        ed_itc = np.array(hf["ed_itc"][:]) if "ed_itc" in hf else None
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
    ap.add_argument("--tol", type=str, default=ODE_TOL, help="ODE tol (rtol=atol). Smaller => more accurate, slower.")
    ap.add_argument("--cutoff-original", type=str, default=ORIGINAL_CUTOFF, help="Truncation cutoff for original.")
    ap.add_argument("--cutoff-hybrid", type=str, default=HYBRID_CUTOFF, help="Truncation cutoff for hybrid-svd.")
    ap.add_argument(
        "--x64",
        action="store_true",
        help="Enable float64 in JAX via PYFLOW_ENABLE_X64=1 (more accurate, slower).",
    )
    ap.add_argument("--svd-rank-h2", type=int, default=None, help="Hybrid-SVD rank for H2 snapshots.")
    ap.add_argument("--svd-rank-h4", type=int, default=None, help="Hybrid-SVD rank for H4 snapshots (on n^2×n^2 reshape).")
    ap.add_argument(
        "--svd-store-dtype",
        type=str,
        default=None,
        choices=["float16", "float32"],
        help="Hybrid-SVD stored factor dtype (float32 improves accuracy, costs memory).",
    )
    ap.add_argument("--svd-niter", type=int, default=None, help="Hybrid-SVD power iterations (more accurate, slower).")
    ap.add_argument("--svd-oversample", type=int, default=None, help="Hybrid-SVD oversampling (more accurate, slower).")
    ap.add_argument(
        "--force-steps",
        type=int,
        default=None,
        help="Force exactly N flow steps via PYFLOW_FORCE_STEPS (speeds up runs; changes physics).",
    )
    args = ap.parse_args()

    # Apply runtime knobs to the base env used for ALL runs.
    BASE_ENV["PYFLOW_ODE_RTOL"] = str(args.tol)
    BASE_ENV["PYFLOW_ODE_ATOL"] = str(args.tol)
    if args.x64:
        BASE_ENV["PYFLOW_ENABLE_X64"] = "1"

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

        # Speed plan:
        # For each p, run original first (computes ED), then run hybrid-svd with ED skipped and reuse ED from original.
        itc_acc: dict[str, list[np.ndarray]] = {m.name: [] for m in MODES}
        ed_acc: list[np.ndarray] = []
        x_best_list: dict[str, list[float]] = {m.name: [] for m in MODES}

        # Local helper: per-run base env extras
        def _extra_for_run(mode: ModeSpec) -> dict[str, str]:
            extra = dict(mode.extra_env or {})
            if args.force_steps is not None and args.force_steps > 0:
                extra["PYFLOW_FORCE_STEPS"] = str(int(args.force_steps))
            else:
                # Ensure we don't inherit a forced-step run from the shell.
                # (run_case will remove the key when value is empty)
                extra["PYFLOW_FORCE_STEPS"] = ""
            return extra

        # Customize per-run mode configs based on CLI knobs
        mode_by_name = {m.name: m for m in MODES}
        _orig_base = mode_by_name["original"]
        _hyb_base = mode_by_name["hybrid-svd"]

        original_mode = ModeSpec(_orig_base.name, _orig_base.use_ckpt, str(args.cutoff_original), extra_env=_orig_base.extra_env)
        # For hybrid we keep the default speed behavior (skip ED) but allow SVD knob overrides.
        hyb_extra = dict(_hyb_base.extra_env or {})
        if args.svd_rank_h2 is not None:
            hyb_extra["PYFLOW_HYBRID_SVD_RANK_H2"] = str(int(args.svd_rank_h2))
        if args.svd_rank_h4 is not None:
            hyb_extra["PYFLOW_HYBRID_SVD_RANK_H4"] = str(int(args.svd_rank_h4))
        if args.svd_store_dtype is not None:
            hyb_extra["PYFLOW_HYBRID_SVD_STORE_DTYPE"] = str(args.svd_store_dtype)
        if args.svd_niter is not None:
            hyb_extra["PYFLOW_HYBRID_SVD_NITER"] = str(int(args.svd_niter))
        if args.svd_oversample is not None:
            hyb_extra["PYFLOW_HYBRID_SVD_OVERSAMPLE"] = str(int(args.svd_oversample))
        hybrid_mode = ModeSpec(_hyb_base.name, _hyb_base.use_ckpt, str(args.cutoff_hybrid), extra_env=hyb_extra)

        for p in range(args.p0, args.p0 + args.reps):
            # original (with ED)
            original_mode_p = ModeSpec(original_mode.name, original_mode.use_ckpt, original_mode.cutoff, extra_env=_extra_for_run(original_mode))
            run_case(L, original_mode_p, args.dis, args.method, args.d, p)
            h5p = h5_path_for_1d(L, args.dis, args.d, p)
            if not h5p.exists():
                raise FileNotFoundError(f"Expected output not found: {h5p}")
            data_o = read_itc_bundle(h5p)
            if data_o.get("ed_itc") is None:
                raise KeyError("Missing dataset `ed_itc` in H5 (ED may be disabled or n too large).")
            pre_o = preprocess_like_proc(data_o["itc"], data_o["itc_nonint"], data_o["ed_itc"])
            itc_acc["original"].append(pre_o["itc"])
            ed_acc.append(pre_o["ed_itc"])
            x_best_list["original"].append(float(pre_o["x_best"]))

            # hybrid-svd (skip ED, reuse ED from original)
            hybrid_mode_p = ModeSpec(hybrid_mode.name, hybrid_mode.use_ckpt, hybrid_mode.cutoff, extra_env=_extra_for_run(hybrid_mode))
            run_case(L, hybrid_mode_p, args.dis, args.method, args.d, p)
            data_h = read_itc_bundle(h5p)  # same output path; overwritten by latest run for this p
            pre_h = preprocess_like_proc(data_h["itc"], data_h["itc_nonint"], pre_o["ed_itc"])
            itc_acc["hybrid-svd"].append(pre_h["itc"])
            x_best_list["hybrid-svd"].append(float(pre_h["x_best"]))

        # Aggregate
        for mode in MODES:
            itc_mean = np.mean(np.stack(itc_acc[mode.name], axis=0), axis=0)
            per_mode_curves[mode.name] = {"itc": itc_mean}
            per_mode_meta[mode.name] = {
                "x_best_mean": float(np.mean(x_best_list[mode.name])),
                "x_best_std": float(np.std(x_best_list[mode.name])),
            }

        ed_mean = np.mean(np.stack(ed_acc, axis=0), axis=0)
        per_mode_curves["original"]["ed"] = ed_mean

        # Use ED from original for plotting (ED should match closely across modes, just noise)
        curves = {
            "ED (avg)": per_mode_curves["original"]["ed"],
            "Original (FE, avg)": per_mode_curves["original"]["itc"],
            "Hybrid-SVD (FE, avg)": per_mode_curves["hybrid-svd"]["itc"],
        }

        out_png = OUT_DIR / f"itc_vs_ed_1d_L{L}_d{args.d:.2f}_reps{args.reps}.png"
        plot_curves(tlist, curves, out_png, title=f"1D ITC vs ED (L={L}, n={L})")
        print(f"  Plot saved: {out_png}")

        # Errors vs ED (avg)
        err_orig = compute_errors(per_mode_curves["original"]["itc"], per_mode_curves["original"]["ed"])
        err_hyb = compute_errors(per_mode_curves["hybrid-svd"]["itc"], per_mode_curves["original"]["ed"])

        print(f"  original: rmse={err_orig.get('rmse', float('nan')):.3e} max_abs={err_orig.get('max_abs', float('nan')):.3e}")
        print(f"  hyb-svd : rmse={err_hyb.get('rmse', float('nan')):.3e} max_abs={err_hyb.get('max_abs', float('nan')):.3e}")

        results[str(L)] = {
            "L": L,
            "n": L,
            "params": {"dis": args.dis, "d": args.d, "reps": args.reps, "p0": args.p0, "method": args.method},
            "meta": per_mode_meta,
            "errors": {"original_vs_ed": err_orig, "hybrid_svd_vs_ed": err_hyb},
            "plot": str(out_png),
        }

    out_json = OUT_DIR / "itc_vs_ed_1d_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results JSON: {out_json}")


if __name__ == "__main__":
    main()


