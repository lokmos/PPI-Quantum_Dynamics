#!/usr/bin/env python3
"""
Compare recursive checkpoint baseline (uniform split) vs tau-based split strategies.

Metrics reported (from `[RECURSIVE_STATS]` JSON line):
- recomputed_steps: total forward steps recomputed during backward pass (proxy for extra work)
- base_cases: number of base-case blocks executed
- traj_snapshots_total: total dense per-step snapshots materialized in base-cases (proxy for transient storage work)
- stack_checkpoints_max: max recursion depth + 1 (proxy for persistent stored checkpoints)

Usage examples:
  python test/scripts/compare_recursive_tau.py --L 3 --steps 200
  python test/scripts/compare_recursive_tau.py --L 3 --steps 200 --tau-h4 --w4 0.5 1.0 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
MAIN = CODE_DIR / "main_itc_cpu_d2.py"


STATS_RE = re.compile(r"^\[RECURSIVE_STATS\]\s+(?P<json>\{.*\})\s*$")


@dataclass(frozen=True)
class Case:
    name: str
    split: str
    tau_include_hint: bool
    tau_w4: float
    tau_mode: str
    extra_env: dict


def run_case(args, case: Case) -> dict:
    env = os.environ.copy()
    env["USE_CKPT"] = "recursive"
    env["PYFLOW_OVERWRITE"] = "1"
    env["PYFLOW_RECURSIVE_STATS"] = "1"
    env["PYFLOW_RECURSIVE_SPLIT"] = case.split
    env["PYFLOW_TAU_MODE"] = case.tau_mode
    env["PYFLOW_TAU_INCLUDE_HINT"] = "1" if case.tau_include_hint else "0"
    env["PYFLOW_TAU_W4"] = str(case.tau_w4)
    # extra weights/modes
    for k, v in (case.extra_env or {}).items():
        env[str(k)] = str(v)
    env["PYFLOW_FORCE_STEPS"] = str(args.steps)
    env["PYFLOW_CUTOFF"] = str(args.cutoff)
    env.setdefault("PYFLOW_ODE_RTOL", args.rtol)
    env.setdefault("PYFLOW_ODE_ATOL", args.atol)

    cmd = [
        sys.executable,
        str(MAIN),
        str(args.L),
        args.dis_type,
        args.method,
        str(args.d),
        str(args.p),
    ]
    r = subprocess.run(cmd, cwd=str(CODE_DIR), env=env, capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        # still try to extract stats, but usually won't exist
        tail = "\n".join(out.splitlines()[-60:])
        raise RuntimeError(f"Case failed: {case.name}\n{tail}")

    stats = None
    for line in out.splitlines():
        m = STATS_RE.match(line.strip())
        if m:
            stats = json.loads(m.group("json"))
            break
    if stats is None:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"Did not find [RECURSIVE_STATS] in output for case={case.name}\n{tail}")
    stats["case"] = case.name
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200, help="Force steps (ensures comparable dl_list length).")
    ap.add_argument("--cutoff", type=str, default="1e-2")
    ap.add_argument("--rtol", type=str, default="1e-6")
    ap.add_argument("--atol", type=str, default="1e-6")
    ap.add_argument("--dis-type", type=str, default="--step=1000")
    ap.add_argument("--method", type=str, default="tensordot")
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--tau-h4", action="store_true", help="Also run tau split with H4 included (expensive).")
    ap.add_argument("--w4", type=float, nargs="*", default=[1.0], help="Weights for H4 in tau metric.")
    ap.add_argument("--include-expensive", action="store_true", help="Include RHS-norm and step-doubling error tau modes.")
    ap.add_argument("--only", type=str, default="", help="Run only cases whose name contains this substring (case-insensitive).")
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "test" / "accuracy_test" / "recursive_tau_report.json"))
    args = ap.parse_args()

    cases = [
        Case("uniform", "uniform", False, 1.0, "delta_h2", {}),
        # H2-only tau metrics
        Case("tau_delta_h2", "tau", False, 1.0, "delta_h2", {}),
        Case("tau_offdiag", "tau", False, 1.0, "offdiag", {}),
        Case("tau_inv_h2", "tau", False, 1.0, "inv", {}),
        Case("tau_combo_h2", "tau", False, 1.0, "combo", {"PYFLOW_TAU_W_OFF": 1.0, "PYFLOW_TAU_W_INV": 1.0, "PYFLOW_TAU_W_RHS": 0.0}),
        # H2+H4 tau metrics (these are the ones that can differ when H4 evolves strongly)
        Case("tau_delta_h2_h4_w4=1", "tau", True, 1.0, "delta_h2_h4", {}),
        Case("tau_inv_h2_h4_w4=1", "tau", True, 1.0, "inv", {}),
        Case("tau_combo_h2_h4_w4=1", "tau", True, 1.0, "combo", {"PYFLOW_TAU_W_OFF": 1.0, "PYFLOW_TAU_W_INV": 1.0, "PYFLOW_TAU_W_RHS": 0.0}),
    ]
    if args.tau_h4:
        for w in args.w4:
            cases.append(Case(f"tau_delta_h2_h4_w4={w:g}", "tau", True, float(w), "delta_h2_h4", {}))
            cases.append(Case(f"tau_combo_h4_w4={w:g}", "tau", True, float(w), "combo", {"PYFLOW_TAU_W_OFF": 1.0, "PYFLOW_TAU_W_INV": 1.0, "PYFLOW_TAU_W_RHS": 0.0}))

    if args.include_expensive:
        cases.append(Case("tau_rhs", "tau", False, 1.0, "rhs", {"PYFLOW_TAU_W_RHS": 1.0}))
        # step-doubling error needs explicit enable
        cases.append(Case("tau_err", "tau", False, 1.0, "err", {"PYFLOW_TAU_ERR_ENABLE": 1, "PYFLOW_TAU_W_ERR": 1.0}))

    if args.only:
        key = args.only.lower()
        cases = [c for c in cases if key in c.name.lower()]
        # Always include baseline if not explicitly filtered in
        if not any(c.name == "uniform" for c in cases):
            cases = [Case("uniform", "uniform", False, 1.0, "delta_h2", {})] + cases

    print(f"Comparing recursive split strategies (L={args.L}, forced_steps={args.steps})")
    results = []
    for c in cases:
        print(f"  - running {c.name} ...", flush=True)
        results.append(run_case(args, c))

    baseline = next(r for r in results if r["case"] == "uniform")
    # Print summary table
    print("\nRESULTS:")
    print("case                 tau_mode        recomputed_steps  saved_vs_uniform  base_cases  traj_snaps_total  stack_ckpts  tau_mid_diff  tau_fallback  tau_dt_nonzero  tau_dt_nonfinite")
    print("-" * 95)
    for r in results:
        saved = int(baseline["recomputed_steps"]) - int(r["recomputed_steps"])
        print(
            f"{r['case']:<20} {str(r.get('tau_mode','-')):<13} {r['recomputed_steps']:>15} {saved:>16} "
            f"{r['base_cases']:>10} {r['traj_snapshots_total']:>16} {r['stack_checkpoints_max']:>11} "
            f"{r.get('tau_mid_diff',0):>12} {r.get('tau_mid_fallback',0):>12} {r.get('tau_dt_nonzero',0):>14} {r.get('tau_dt_nonfinite',0):>15}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()

