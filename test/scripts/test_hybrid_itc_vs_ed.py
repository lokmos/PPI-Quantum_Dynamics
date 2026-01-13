#!/usr/bin/env python3
"""
Compare FE-ITC curves against ED, for Original vs Hybrid modes.

This mimics the comparison style used in `figures.ipynb`:
- read FE curve: `itc`
- read ED curve: `ed_itc`
- plot them together and report numerical errors.

Hybrid uses cutoff=1e-6 (as requested).

Usage:
  python test/scripts/test_hybrid_itc_vs_ed.py 2 2

Notes:
- Requires output to include `itc` and `ed_itc`. For `main_itc_cpu_d2.py` this
  is enabled via env: PYFLOW_LADDER=1 (and optionally PYFLOW_ITC=1).
"""

from __future__ import annotations

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

DEFAULT_DIS_TYPE = "--step=1000"
DEFAULT_METHOD = "tensordot"
DEFAULT_D = 1.0
DEFAULT_P = 0

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
    # Enable ITC/ED comparison datasets in H5
    "PYFLOW_LADDER": "1",
    "PYFLOW_ITC": "0",
    # Avoid oversubscription
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={_NUM_CORES} inter_op_parallelism_threads=1",
}


def h5_path_for(L: int, dis_type: str, d: float, p: int) -> Path:
    dim = 2
    order = 4
    x = 0.0
    delta = 0.1
    n = L**dim
    pot_folder = dis_type
    return (
        CODE_DIR
        / "pyflow_out"
        / "fermion"
        / "d2"
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

    cmd = [sys.executable, str(CODE_DIR / "main_itc_cpu_d2.py"), str(L), dis_type, method, str(d), str(p)]
    print(f"  Running L={L}, mode={mode.name:8s}, cutoff={mode.cutoff} ...", end="", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=CODE_DIR, env=env, capture_output=True, text=True, timeout=PROCESS_TIMEOUT)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f" [FAILED] ({dt:.1f}s)")
        tail = (res.stderr or "").splitlines()[-10:]
        for line in tail:
            print(f"    ! {line}")
        raise RuntimeError(f"Run failed for mode={mode.name}")
    print(f" [OK] ({dt:.1f}s)")


def read_itc_and_ed(h5_path: Path) -> dict:
    import h5py  # local import to keep the script import-light

    with h5py.File(h5_path, "r") as hf:
        if "itc" not in hf:
            raise KeyError("Missing dataset `itc` in H5 (enable via PYFLOW_LADDER=1).")
        if "ed_itc" not in hf:
            raise KeyError("Missing dataset `ed_itc` in H5 (ED may be disabled or n too large).")
        itc = np.array(hf["itc"][:])
        itc_nonint = np.array(hf["itc_nonint"][:]) if "itc_nonint" in hf else None
        ed_itc = np.array(hf["ed_itc"][:])
        return {"itc": itc, "itc_nonint": itc_nonint, "ed_itc": ed_itc}


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def compute_errors(itc: np.ndarray, ed_itc: np.ndarray) -> dict:
    itc = np.real(np.array(itc)).reshape(-1)
    ed_itc = np.real(np.array(ed_itc)).reshape(-1)
    if itc.shape != ed_itc.shape:
        raise ValueError(f"Shape mismatch: itc={itc.shape}, ed_itc={ed_itc.shape}")

    mask = _finite_mask(itc, ed_itc)
    n_total = int(itc.size)
    n_finite = int(np.count_nonzero(mask))
    if n_finite == 0:
        return {
            "n_total": n_total,
            "n_finite": 0,
            "error": "All entries are non-finite (nan/inf).",
            "n_nonfinite_itc": int(np.count_nonzero(~np.isfinite(itc))),
            "n_nonfinite_ed_itc": int(np.count_nonzero(~np.isfinite(ed_itc))),
        }

    itc_f = itc[mask]
    ed_f = ed_itc[mask]
    diff = itc_f - ed_f
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))

    denom = np.maximum(np.abs(ed_f), 1e-12)
    rel = np.abs(diff) / denom
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_rel": float(np.mean(rel)),
        "max_rel": float(np.max(rel)),
    }


def preprocess_like_proc(itc: np.ndarray, itc_nonint: np.ndarray | None, ed_itc: np.ndarray) -> dict:
    """
    Match the plotting conventions used in `code/proc.py` / `figures.ipynb`:
    - multiply by 4 so that ED starts at ~1 (since raw ED ITC starts at ~0.25)
    - if `itc_nonint` exists, apply the same shift+normalize calibration used in proc.py
    """
    itc = np.real(np.array(itc))
    ed_itc = np.real(np.array(ed_itc))

    # proc.py uses a factor 4
    itc = 4.0 * itc
    ed_itc = 4.0 * ed_itc

    if itc_nonint is not None:
        itc_nonint = np.real(np.array(itc_nonint))
        # itc_nonint in H5 is already mean over states in dyn_itc; proc.py uses mean(...) again defensively
        itc_nonint = 4.0 * itc_nonint

    # Flatten/average if extra leading dims exist
    if itc.ndim > 1:
        itc = np.mean(itc, axis=0)
    if ed_itc.ndim > 1:
        ed_itc = np.mean(ed_itc, axis=0)
    if itc_nonint is not None and itc_nonint.ndim > 1:
        itc_nonint = np.mean(itc_nonint, axis=0)

    # Apply proc.py calibration only if we have a usable non-interacting reference
    x_best = 0.0
    if itc_nonint is not None:
        # Use the same objective as proc.py (first ~75 points)
        nfit = min(75, itc.size, itc_nonint.size)
        if nfit >= 5 and np.isfinite(itc[:nfit]).all() and np.isfinite(itc_nonint[:nfit]).all():
            points = 101
            xs = np.linspace(0.0, 1.0, points, endpoint=True)
            errs = np.empty(points, dtype=float)
            base = np.array(itc, copy=True)
            for i, x in enumerate(xs):
                test = base.copy()
                # shift so that after normalization we can better match the non-interacting curve
                test += x * (1.0 - test[0])
                test *= 1.0 / test[0]
                denom = np.maximum(np.abs(itc_nonint[:nfit]), 1e-12)
                errs[i] = float(np.mean(np.abs((test[:nfit] - itc_nonint[:nfit]) / denom)))
            x_best = float(xs[int(np.argmin(errs))])
            itc = itc + x_best * (1.0 - itc[0])
            itc = itc / itc[0]

    # Always normalize ED so it starts at 1 (consistent with figure styling)
    if ed_itc.size and np.isfinite(ed_itc[0]) and abs(ed_itc[0]) > 0:
        ed_itc = ed_itc / ed_itc[0]
    if itc_nonint is not None and itc_nonint.size and np.isfinite(itc_nonint[0]) and abs(itc_nonint[0]) > 0:
        itc_nonint = itc_nonint / itc_nonint[0]

    return {"itc": itc, "ed_itc": ed_itc, "itc_nonint": itc_nonint, "x_best": x_best}


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
    L_min = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    L_max = int(sys.argv[2]) if len(sys.argv) > 2 else L_min

    dis_type = DEFAULT_DIS_TYPE
    method = DEFAULT_METHOD
    d = DEFAULT_D
    p = DEFAULT_P

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same t-grid as the code uses.
    tlist = np.logspace(-2, 5, 151, base=10, endpoint=True)

    all_results: dict[str, dict] = {}

    for L in range(L_min, L_max + 1):
        print("\n" + "=" * 90)
        print(f"ITC vs ED comparison (L={L}, n={L*L})")
        print("=" * 90)

        per_mode_data = {}
        for mode in MODES:
            run_case(L, mode, dis_type, method, d, p)
            h5p = h5_path_for(L, dis_type, d, p)
            if not h5p.exists():
                raise FileNotFoundError(f"Expected output not found: {h5p}")
            data = read_itc_and_ed(h5p)
            # Copy arrays immediately: the next mode run overwrites the same H5 path.
            per_mode_data[mode.name] = {
                "itc": data["itc"].copy(),
                "itc_nonint": None if data["itc_nonint"] is None else data["itc_nonint"].copy(),
                "ed_itc": data["ed_itc"].copy(),
            }

        # Compute errors vs ED for each mode
        res = {"L": L, "n": L * L, "modes": {}}
        for mode_name, data in per_mode_data.items():
            pre = preprocess_like_proc(data["itc"], data["itc_nonint"], data["ed_itc"])
            res["modes"][mode_name] = compute_errors(pre["itc"], pre["ed_itc"])
            res["modes"][mode_name]["x_best"] = pre["x_best"]
            # keep tiny previews for debugging/traceability
            res["modes"][mode_name]["preview_itc"] = pre["itc"][:5].tolist() if pre["itc"].size else []
            res["modes"][mode_name]["preview_ed"] = pre["ed_itc"][:5].tolist() if pre["ed_itc"].size else []

        # Print a compact table
        for mode_name in ("original", "hybrid"):
            r = res["modes"][mode_name]
            if "error" in r:
                print(f"  {mode_name:8s}: ERROR: {r['error']} | nonfinite(itc,ed)=({r.get('n_nonfinite_itc')},{r.get('n_nonfinite_ed_itc')})")
            else:
                print(
                    f"  {mode_name:8s}: rmse={r['rmse']:.3e}  max_abs={r['max_abs']:.3e}  "
                    f"mean_rel={r['mean_rel']:.3e}  max_rel={r['max_rel']:.3e}  "
                    f"(finite {r['n_finite']}/{r['n_total']})  x_best={r.get('x_best',0.0):.2f}"
                )

        # Plot: original vs ED and hybrid vs ED (two subplots)
        out_png = OUT_DIR / f"itc_vs_ed_L{L}_d{d:.2f}_p{p}.png"
        pre_orig = preprocess_like_proc(
            per_mode_data["original"]["itc"],
            per_mode_data["original"]["itc_nonint"],
            per_mode_data["original"]["ed_itc"],
        )
        pre_hyb = preprocess_like_proc(
            per_mode_data["hybrid"]["itc"],
            per_mode_data["hybrid"]["itc_nonint"],
            per_mode_data["hybrid"]["ed_itc"],
        )
        curves = {
            "ED": np.array(pre_orig["ed_itc"]).reshape(-1),
            "Original (FE)": np.array(pre_orig["itc"]).reshape(-1),
            "Hybrid (FE)": np.array(pre_hyb["itc"]).reshape(-1),
        }
        plot_curves(tlist, curves, out_png, title=f"ITC vs ED (L={L}, n={L*L})")
        print(f"  Plot saved: {out_png}")

        all_results[str(L)] = res

    out_json = OUT_DIR / "itc_vs_ed_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results JSON: {out_json}")


if __name__ == "__main__":
    main()


