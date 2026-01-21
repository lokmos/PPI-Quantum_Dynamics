#!/usr/bin/env python3
"""
One-click correctness validation: ORIGINAL vs HYBRID-SVD (the "ultimate" method).

Runs two small cases:
- 1D: L=9
- 2D: L=3

For each case it runs:
- ORIGINAL   (USE_CKPT=0)
- HYBRID-SVD (USE_CKPT=hybrid + PYFLOW_HYBRID_SVD=1 + PYFLOW_HYBRID_COMPRESS=hybrid-svd)

Then compares key datasets written to the output HDF5 (H2_diag, Hint, liom2/liom4, trunc_err, dl_list)
and writes:
- summary.json
- overlay plots (optional)

Notes:
- This validation uses the *non-ladder* LIOM branch (PYFLOW_LADDER=0), because the ladder/ITC branch
  `flow_int_fl` does not use checkpoint/hybrid-svd.
- To keep it fast, ED is disabled (PYFLOW_SKIP_ED=1).
- Outputs are isolated under a unique run directory via PYFLOW_OUTDIR, so nothing gets overwritten.
- HYBRID does NOT enable adaptive grid here; instead we can increase lmax (flow-time horizon) to help convergence.
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
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
OUT_ROOT = REPO_ROOT / "test" / "oneclick_validate"


@dataclass(frozen=True)
class Case:
    dim: int
    L: int
    entrypoint: Path


CASES = [
    Case(dim=1, L=9, entrypoint=CODE_DIR / "main_itc_cpu.py"),
    # NOTE: user requested 1D-only correctness validation by default.
    # (2D can be added back if needed.)
    # Case(dim=2, L=3, entrypoint=CODE_DIR / "main_itc_cpu_d2 copy.py"),
]


def out_h5_path(
    *,
    outdir: Path,
    dim: int,
    dis_type: str,
    L: int,
    d: float,
    p: int,
    order: int = 4,
    liom: str = "bck",
    x: float = 0.0,
    delta: float = 0.1,
    species_folder: str = "fermion",
) -> Path:
    """
    Mirrors core.utility.namevar() layout (non-proc).
    """
    n = int(L**dim)
    return (
        outdir
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


def read_h5(path: Path) -> dict[str, Any]:
    import h5py

    keys = ("H2_diag", "Hint", "liom2", "liom4", "trunc_err", "dl_list")
    with h5py.File(path, "r") as hf:
        out: dict[str, Any] = {}
        for k in keys:
            if k in hf:
                out[k] = np.array(hf[k][:])
        return out


def diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0, "a_len": float(a.size), "b_len": float(b.size)}
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return {"all_nonfinite": 1.0}
    d = a[m] - b[m]
    rmse = float(np.sqrt(np.mean(d**2)))
    max_abs = float(np.max(np.abs(d)))
    denom = np.maximum(np.abs(a[m]), 1e-12)
    rel = np.abs(d) / denom
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_rel": float(np.mean(rel)),
        "max_rel": float(np.max(rel)),
        "n": float(np.count_nonzero(m)),
    }


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _save_case_plots(case_dir: Path, *, data_o: dict[str, Any], data_h: dict[str, Any], title_prefix: str) -> None:
    """
    Default plotting output:
    - scatter original vs hybrid-svd for H2_diag
    - histograms of log10(|diff|) for Hint/liom2/liom4
    - line plots for dl_list and trunc_err
    """
    plt = _safe_import_matplotlib()
    plots_dir = case_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _dump_npz(path: Path, **arrays: Any) -> None:
        """
        Save plot inputs/derived arrays for reproducibility.
        """
        safe: dict[str, Any] = {}
        for k, v in arrays.items():
            try:
                safe[k] = np.asarray(v)
            except Exception:
                # best-effort: keep raw object
                safe[k] = v
        np.savez_compressed(path, **safe)

    if plt is None:
        # Fallback: dump arrays for manual plotting
        _dump_npz(
            plots_dir / "arrays.npz",
            **{
                "H2_diag_original": np.asarray(data_o.get("H2_diag", np.array([]))),
                "H2_diag_hybrid_svd": np.asarray(data_h.get("H2_diag", np.array([]))),
                "Hint_original": np.asarray(data_o.get("Hint", np.array([]))),
                "Hint_hybrid_svd": np.asarray(data_h.get("Hint", np.array([]))),
                "liom2_original": np.asarray(data_o.get("liom2", np.array([]))),
                "liom2_hybrid_svd": np.asarray(data_h.get("liom2", np.array([]))),
                "liom4_original": np.asarray(data_o.get("liom4", np.array([]))),
                "liom4_hybrid_svd": np.asarray(data_h.get("liom4", np.array([]))),
                "dl_list_original": np.asarray(data_o.get("dl_list", np.array([]))),
                "dl_list_hybrid_svd": np.asarray(data_h.get("dl_list", np.array([]))),
                "trunc_err_original": np.asarray(data_o.get("trunc_err", np.array([]))),
                "trunc_err_hybrid_svd": np.asarray(data_h.get("trunc_err", np.array([]))),
            },
        )
        print(f"[WARN] matplotlib not available; saved arrays to: {plots_dir / 'arrays.npz'}")
        return

    def _scatter_xy(a: np.ndarray, b: np.ndarray, out: Path, title: str) -> None:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]
        if a.size == 0:
            return
        # Dump the actual data used for the scatter plot.
        _dump_npz(
            out.with_name(out.stem + "_data.npz"),
            a=a,
            b=b,
            lo=float(np.min([a.min(), b.min()])),
            hi=float(np.max([a.max(), b.max()])),
        )
        if a.size > 20000:
            idx = np.random.default_rng(0).choice(a.size, size=20000, replace=False)
            a = a[idx]
            b = b[idx]
        lo = float(np.min([a.min(), b.min()]))
        hi = float(np.max([a.max(), b.max()]))
        plt.figure(figsize=(6, 6))
        plt.scatter(a, b, s=6, alpha=0.35)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlabel("original")
        plt.ylabel("hybrid-svd")
        plt.title(title)
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

    def _hist_log_abs_diff(a: np.ndarray, b: np.ndarray, out: Path, title: str) -> None:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        m = np.isfinite(a) & np.isfinite(b)
        d = (a[m] - b[m]).astype(np.float64, copy=False)
        if d.size == 0:
            return
        x = np.log10(np.abs(d) + 1e-18)
        # Dump the exact data used for the histogram.
        _dump_npz(
            out.with_name(out.stem + "_data.npz"),
            a=a[m],
            b=b[m],
            diff=d,
            log10_absdiff=x,
        )
        plt.figure(figsize=(7, 4.5))
        plt.hist(x, bins=80, alpha=0.85)
        plt.xlabel(r"$\log_{10}(|\Delta| + 10^{-18})$")
        plt.ylabel("count")
        plt.title(title)
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

    def _as_square_matrix(x: np.ndarray) -> np.ndarray | None:
        """
        Try to view `x` as a square matrix for gauge-invariant comparisons.
        - If already (n,n), return as-is.
        - If total size is a perfect square, reshape to (n,n).
        Otherwise return None.
        """
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == x.shape[1]:
            return x
        s = int(np.sqrt(x.size))
        if s * s == x.size and s > 0:
            return x.reshape((s, s))
        return None

    def _liom2_svals(x: np.ndarray) -> np.ndarray:
        """
        Gauge/rotation/sign robust fingerprint for LIOM2:
        use singular values of its matrix representation.

        For LIOM2 stored as a dense operator in some basis, singular values are
        invariant under orthogonal/unitary similarity transforms and also
        invariant under a global sign flip.
        """
        m = _as_square_matrix(x)
        if m is None:
            # Fallback: treat as vector; singular value = norm
            v = np.asarray(x, dtype=np.float64).reshape(-1)
            v = v[np.isfinite(v)]
            return np.array([float(np.linalg.norm(v))], dtype=np.float64)
        a = np.asarray(m, dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.linalg.svd(a, compute_uv=False)
        s = np.sort(s)[::-1]
        return s

    def _plot_svals(a: np.ndarray, b: np.ndarray, out: Path, title: str) -> None:
        sa = _liom2_svals(a)
        sb = _liom2_svals(b)
        n = min(sa.size, sb.size)
        if n == 0:
            return
        _dump_npz(
            out.with_name(out.stem + "_data.npz"),
            svals_original=sa,
            svals_hybrid_svd=sb,
        )
        plt.figure(figsize=(9, 4.5))
        plt.plot(np.arange(n), sa[:n], label="original", linewidth=2)
        plt.plot(np.arange(n), sb[:n], label="hybrid-svd", linewidth=2, alpha=0.85)
        plt.yscale("log")
        plt.xlabel("index (sorted)")
        plt.ylabel("singular value")
        plt.title(title)
        plt.grid(alpha=0.25, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

    def _plot_lines(a: np.ndarray, b: np.ndarray, out: Path, title: str) -> None:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        n = min(a.size, b.size)
        if n == 0:
            return
        _dump_npz(
            out.with_name(out.stem + "_data.npz"),
            a=a,
            b=b,
            n_common=int(n),
            a_common=a[:n],
            b_common=b[:n],
        )
        plt.figure(figsize=(9, 4.5))
        plt.plot(np.arange(n), a[:n], label="original", linewidth=2)
        plt.plot(np.arange(n), b[:n], label="hybrid-svd", linewidth=2, alpha=0.85)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.title(title)
        plt.grid(alpha=0.25, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

    if "H2_diag" in data_o and "H2_diag" in data_h:
        _scatter_xy(
            np.asarray(data_o["H2_diag"]),
            np.asarray(data_h["H2_diag"]),
            plots_dir / "H2_diag_scatter.png",
            f"{title_prefix} | H2_diag (original vs hybrid-svd)",
        )

    # Hint / liom4: element-wise diffs are meaningful.
    for k in ("Hint", "liom4"):
        if k in data_o and k in data_h:
            _hist_log_abs_diff(
                np.asarray(data_o[k]),
                np.asarray(data_h[k]),
                plots_dir / f"{k}_log10_absdiff_hist.png",
                f"{title_prefix} | {k} log10(|diff|)",
            )

    # liom2: element-wise diffs are *not* gauge-invariant (sign/rotations in (near-)degenerate subspaces).
    # Use singular values as a physical, basis-agnostic fingerprint.
    if "liom2" in data_o and "liom2" in data_h:
        a2 = np.asarray(data_o["liom2"])
        b2 = np.asarray(data_h["liom2"])
        _plot_svals(
            a2,
            b2,
            plots_dir / "liom2_svals_compare.png",
            f"{title_prefix} | liom2 singular values (gauge-invariant)",
        )
        sa = _liom2_svals(a2)
        sb = _liom2_svals(b2)
        _hist_log_abs_diff(
            sa,
            sb,
            plots_dir / "liom2_log10_absdiff_hist.png",
            f"{title_prefix} | liom2 log10(|Δ singular values|) (gauge-invariant)",
        )

    if "dl_list" in data_o and "dl_list" in data_h:
        _plot_lines(
            np.asarray(data_o["dl_list"]),
            np.asarray(data_h["dl_list"]),
            plots_dir / "dl_list_compare.png",
            f"{title_prefix} | dl_list compare",
        )
    if "trunc_err" in data_o and "trunc_err" in data_h:
        _plot_lines(
            np.asarray(data_o["trunc_err"]),
            np.asarray(data_h["trunc_err"]),
            plots_dir / "trunc_err_compare.png",
            f"{title_prefix} | trunc_err compare",
        )


def run_one(
    *,
    case: Case,
    mode: str,
    dis_type: str,
    method: str,
    d: float,
    p: int,
    outdir: Path,
    force_steps: int | None,
    ode_tol: str,
    cutoff: str,
    svd_rank_h2: int,
    svd_rank_h4: int,
    svd_store_dtype: str,
    svd_niter: int,
    svd_oversample: int,
    lmax: float | None,
) -> None:
    assert mode in ("original", "hybrid-svd")

    env = os.environ.copy()
    # Critical for correctness: make ORIGINAL and HYBRID run the *same* disorder realisation.
    # core.init.Hinit() respects PYFLOW_SEED when set.
    env["PYFLOW_SEED"] = str(int(p))
    env["PYFLOW_OUTDIR"] = str(outdir)
    env["PYFLOW_OVERWRITE"] = "1"
    env["PYFLOW_SKIP_ED"] = "1"
    env["PYFLOW_LADDER"] = "0"
    env["PYFLOW_ITC"] = "0"
    # Ensure adaptive grid is OFF (we use larger lmax instead).
    env["PYFLOW_ADAPTIVE_GRID"] = "0"
    # Reduce numeric drift across modes (slower but small-L is fine).
    env["PYFLOW_ENABLE_X64"] = "1"
    env["PYFLOW_ODE_RTOL"] = str(ode_tol)
    env["PYFLOW_ODE_ATOL"] = str(ode_tol)
    env["PYFLOW_CUTOFF"] = str(cutoff)
    env["USE_JIT_FLOW"] = "0"

    if lmax is not None and lmax > 0:
        env["PYFLOW_LMAX"] = str(float(lmax))
    else:
        env.pop("PYFLOW_LMAX", None)

    if force_steps is not None and force_steps > 0:
        env["PYFLOW_FORCE_STEPS"] = str(int(force_steps))
    else:
        env.pop("PYFLOW_FORCE_STEPS", None)

    if mode == "original":
        env["USE_CKPT"] = "0"
        env["PYFLOW_USE_COPY_ROUTINES"] = "0"
        env["PYFLOW_HYBRID_SVD"] = "0"
        env["PYFLOW_HYBRID_COMPRESS"] = ""
    else:
        env["USE_CKPT"] = "hybrid"
        # Use copy routines so hybrid-svd compression is available.
        env["PYFLOW_USE_COPY_ROUTINES"] = "1"
        # Make rSVD deterministic across runs (for debugging/tuning).
        env["PYFLOW_HYBRID_SVD_SEED"] = str(int(p))
        # Keep buffers in float32 for stability (hybrid stores trajectories).
        env["PYFLOW_HYBRID_BUFFER_DTYPE"] = "float32"
        # Make exponent scaling explicit (default is on in copy routines).
        env["PYFLOW_HYBRID_EXP_SCALE"] = "1"
        env["PYFLOW_HYBRID_SVD"] = "1"
        env["PYFLOW_HYBRID_COMPRESS"] = "hybrid-svd"
        env["PYFLOW_HYBRID_SVD_RANK_H2"] = str(int(svd_rank_h2))
        env["PYFLOW_HYBRID_SVD_RANK_H4"] = str(int(svd_rank_h4))
        env["PYFLOW_HYBRID_SVD_STORE_DTYPE"] = str(svd_store_dtype)
        env["PYFLOW_HYBRID_SVD_NITER"] = str(int(svd_niter))
        env["PYFLOW_HYBRID_SVD_OVERSAMPLE"] = str(int(svd_oversample))

    cmd = [sys.executable, str(case.entrypoint), str(case.L), dis_type, method, str(d), str(p)]
    print(f"[RUN] dim={case.dim} L={case.L} mode={mode} outdir={outdir}", flush=True)
    subprocess.run(cmd, cwd=str(CODE_DIR), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dis", type=str, default="random")
    ap.add_argument("--method", type=str, default="tensordot")
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--cutoff", type=str, default="1e-4")
    ap.add_argument("--tol", type=str, default="1e-8")
    # Default: compare near-converged endpoints (do not force a fixed step count).
    ap.add_argument("--force-steps", type=int, default=-1, help="Set >0 to force exactly N steps.")
    # Make HYBRID converge by increasing the flow-time horizon (lmax).
    # (original keeps its built-in lmax; hybrid uses this override)
    ap.add_argument("--lmax-hybrid", type=float, default=3000.0)
    # More accurate Hybrid-SVD defaults (small-L correctness run).
    ap.add_argument("--svd-rank-h2", type=int, default=64)
    # For n=9, H4 reshape is (n^2, n^2)=(81,81); using rank=81 is (near-)exact.
    ap.add_argument("--svd-rank-h4", type=int, default=81)
    ap.add_argument("--svd-store-dtype", type=str, default="float32", choices=["float16", "float32"])
    ap.add_argument("--svd-niter", type=int, default=2)
    ap.add_argument("--svd-oversample", type=int, default=16)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "params": {
            "dis": args.dis,
            "method": args.method,
            "d": float(args.d),
            "p": int(args.p),
            "cutoff": str(args.cutoff),
            "tol": str(args.tol),
            "force_steps": int(args.force_steps),
            "lmax_hybrid": float(args.lmax_hybrid),
            "svd": {
                "rank_h2": int(args.svd_rank_h2),
                "rank_h4": int(args.svd_rank_h4),
                "store_dtype": str(args.svd_store_dtype),
                "niter": int(args.svd_niter),
                "oversample": int(args.svd_oversample),
            },
        },
        "cases": [],
    }

    for case in CASES:
        case_dir = run_dir / f"dim{case.dim}_L{case.L}"
        out_o = case_dir / "out_original"
        out_h = case_dir / "out_hybrid_svd"
        out_o.mkdir(parents=True, exist_ok=True)
        out_h.mkdir(parents=True, exist_ok=True)

        # Run both modes
        run_one(
            case=case,
            mode="original",
            dis_type=args.dis,
            method=args.method,
            d=float(args.d),
            p=int(args.p),
            outdir=out_o,
            force_steps=int(args.force_steps) if int(args.force_steps) > 0 else None,
            ode_tol=str(args.tol),
            cutoff=str(args.cutoff),
            svd_rank_h2=int(args.svd_rank_h2),
            svd_rank_h4=int(args.svd_rank_h4),
            svd_store_dtype=str(args.svd_store_dtype),
            svd_niter=int(args.svd_niter),
            svd_oversample=int(args.svd_oversample),
            lmax=None,
        )
        run_one(
            case=case,
            mode="hybrid-svd",
            dis_type=args.dis,
            method=args.method,
            d=float(args.d),
            p=int(args.p),
            outdir=out_h,
            force_steps=int(args.force_steps) if int(args.force_steps) > 0 else None,
            ode_tol=str(args.tol),
            cutoff=str(args.cutoff),
            svd_rank_h2=int(args.svd_rank_h2),
            svd_rank_h4=int(args.svd_rank_h4),
            svd_store_dtype=str(args.svd_store_dtype),
            svd_niter=int(args.svd_niter),
            svd_oversample=int(args.svd_oversample),
            lmax=float(args.lmax_hybrid),
        )

        # Compare outputs
        x = 0.0 if args.dis != "curved" else 1.0
        h5_o = out_h5_path(outdir=out_o, dim=case.dim, dis_type=args.dis, L=case.L, d=float(args.d), p=int(args.p), x=x)
        h5_h = out_h5_path(outdir=out_h, dim=case.dim, dis_type=args.dis, L=case.L, d=float(args.d), p=int(args.p), x=x)
        if not h5_o.exists() or not h5_h.exists():
            raise FileNotFoundError(f"Missing output: original={h5_o.exists()} hybrid={h5_h.exists()} | {h5_o} | {h5_h}")

        data_o = read_h5(h5_o)
        data_h = read_h5(h5_h)

        # Default: always output plots per case
        _save_case_plots(
            case_dir,
            data_o=data_o,
            data_h=data_h,
            title_prefix=f"dim={case.dim} L={case.L} dis={args.dis} d={args.d:.2f} p={args.p}",
        )

        metrics: dict[str, Any] = {}
        for k in ("H2_diag", "Hint", "liom2", "liom4", "trunc_err", "dl_list"):
            if k in data_o and k in data_h:
                metrics[k] = diff_metrics(np.asarray(data_o[k]), np.asarray(data_h[k]))
            else:
                metrics[k] = {"missing": True, "has_original": bool(k in data_o), "has_hybrid": bool(k in data_h)}

        case_rec = {
            "dim": case.dim,
            "L": case.L,
            "original_h5": str(h5_o),
            "hybrid_svd_h5": str(h5_h),
            "metrics_hybrid_minus_original": metrics,
        }
        summary["cases"].append(case_rec)

    out_json = run_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()

