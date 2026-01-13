#!/usr/bin/env python3
"""
Validate that checkpoint algorithms reproduce the same *forward* flow as the original method.

Goal:
- Ensure ckpt/recursive/hybrid are numerically consistent with original when they use
  the same dl_list and ODE tolerances, by comparing final H2/H4 snapshots after a fixed number of steps.

Why forward-only?
- Forward flow is the shared core. Backward LIOM reconstruction is expected to match too, but is more expensive
  and may depend on additional operator ODE details. Start by validating the forward core.

Usage:
  python test/scripts/validate_ckpt_correctness_d2.py --L 3 --steps 200 --cutoff 1e-2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"


def run_mode(L: int, mode: str, steps: int, cutoff: str, dis_type: str, method: str, d: str, p: str) -> str:
    env = os.environ.copy()
    env["PYFLOW_FORCE_STEPS"] = str(steps)
    env["PYFLOW_CUTOFF"] = str(cutoff)
    env["PYFLOW_OVERWRITE"] = "1"
    # keep tolerances explicit
    env.setdefault("PYFLOW_ODE_RTOL", "1e-6")
    env.setdefault("PYFLOW_ODE_ATOL", "1e-6")

    if mode == "original":
        env["USE_CKPT"] = "0"
    elif mode == "ckpt":
        env["USE_CKPT"] = "1"
    elif mode == "recursive":
        env["USE_CKPT"] = "recursive"
    elif mode == "hybrid":
        env["USE_CKPT"] = "hybrid"
        # for correctness testing, use safe dtype
        env.setdefault("PYFLOW_HYBRID_BUFFER_DTYPE", "float32")
    else:
        raise ValueError(mode)

    cmd = [sys.executable, str(CODE_DIR / "main_itc_cpu_d2.py"), str(L), dis_type, method, d, p]
    r = subprocess.run(cmd, cwd=str(CODE_DIR), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError(f"run failed for mode={mode}")
    return r.stdout + "\n" + r.stderr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cutoff", type=str, default="1e-2")
    ap.add_argument("--dis_type", type=str, default="--step=1000")
    ap.add_argument("--method", type=str, default="tensordot")
    ap.add_argument("--d", type=str, default="1.0")
    ap.add_argument("--p", type=str, default="0")
    args = ap.parse_args()

    print(f"Validating forward-flow consistency: L={args.L} forced_steps={args.steps} cutoff={args.cutoff}")
    for mode in ("original", "ckpt", "recursive", "hybrid"):
        print(f"\n--- running {mode} ---")
        out = run_mode(args.L, mode, args.steps, args.cutoff, args.dis_type, args.method, args.d, args.p)
        # Print just the convergence / mode switch / errors summary lines
        lines = [ln for ln in out.splitlines() if ("MODE SWITCH" in ln or "FORCE_STEPS" in ln or "ERROR" in ln or "Converged" in ln)]
        for ln in lines[-20:]:
            print(ln)

    print("\nNext step: compare the produced HDF5 snapshots (H2_diag / Hint) numerically in a Python env with h5py/numpy.")


if __name__ == "__main__":
    main()

