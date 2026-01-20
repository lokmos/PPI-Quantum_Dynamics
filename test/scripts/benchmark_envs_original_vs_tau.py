#!/usr/bin/env python3
"""
Benchmark (paper-oriented): Hybrid-Original vs Hybrid-SVD across multiple physical environments (2D only).

Design goals:
- Flow-equation / numerical settings are FIXED inside this script (paper reproducibility).
- Physical environments (disorder/potential type + strength) are enumerated here.
- Runs HYBRID-ORIGINAL and HYBRID-SVD for each (env, L) pair with identical forced steps.
- Produces net-memory numbers from memlog (peak RSS - baseline RSS) during the *flow phase only*.

Usage (one command):
  python test/scripts/benchmark_envs_original_vs_tau.py

Optional:
  python test/scripts/benchmark_envs_original_vs_tau.py --L 3 10
  python test/scripts/benchmark_envs_original_vs_tau.py --only-env random_d1 qp_golden_d1
  python test/scripts/benchmark_envs_original_vs_tau.py --no-jit
  python test/scripts/benchmark_envs_original_vs_tau.py --L 5 8 --cutoff 1e-2 --tol 1e-2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
MAIN = CODE_DIR / "main_itc_cpu_d2 copy.py"

# -----------------------------------------------------------------------------
# Fixed L ranges (paper-oriented)
# -----------------------------------------------------------------------------
# Keep conservative defaults; override via --L if desired.
L_MIN_DEFAULT, L_MAX_DEFAULT = 6, 8


# -----------------------------------------------------------------------------
# Fixed "flow-equation" / numerical settings (paper baseline)
# -----------------------------------------------------------------------------
FIXED_METHOD = "tensordot"
FIXED_P = 0
FIXED_CUTOFF = "1e-2"
FIXED_ODE_TOL = "1e-2"        # rtol=atol
FIXED_FORCE_STEPS = "1000"    # exactly 1000 steps for comparability across modes

# Net memory: sample RSS every N flow steps
FIXED_MEMLOG_EVERY = "5"

# Speed / stability knobs (can be overridden by environment variables before running the script)
BASE_ENV: dict[str, str] = {
    "PYFLOW_MEMLOG": "1",
    "PYFLOW_MEMLOG_EVERY": FIXED_MEMLOG_EVERY,
    "PYFLOW_TIMELOG": "1",
    "PYFLOW_OVERWRITE": "1",
    "PYFLOW_SKIP_ED": "1",  # skip QuSpin ED for speed and to avoid polluting mem peaks
    "PYFLOW_CUTOFF": FIXED_CUTOFF,
    "PYFLOW_ODE_RTOL": FIXED_ODE_TOL,
    "PYFLOW_ODE_ATOL": FIXED_ODE_TOL,
    "PYFLOW_FORCE_STEPS": FIXED_FORCE_STEPS,
    # Keep CPU libraries from oversubscribing; let XLA handle parallelism.
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


# -----------------------------------------------------------------------------
# Physical environments (2D, spinless fermion) — add/remove cases here
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvCase:
    key: str
    dis_type: str
    d: float
    extra_env: dict[str, str] | None = None


ENV_CASES: list[EnvCase] = [
    # Benchmark default environment
    EnvCase("random_d1", "random", 1.0),
    EnvCase("random_d4", "random", 4.0),
    EnvCase("qp_golden_d1", "QPgolden", 1.0),
    EnvCase("qp_golden_d4", "QPgolden", 4.0),
    EnvCase("linear_d1", "linear", 1.0),
    # curved uses xlist=[1.] inside main_itc_cpu_d2.py; keep d moderate
    EnvCase("curved_d1", "curved", 1.0),
]


# -----------------------------------------------------------------------------
# Hybrid configurations
# -----------------------------------------------------------------------------
HYBRID_ORIGINAL_ENV: dict[str, str] = {
    # Implemented in copy routines (explicit dispatch in main_itc_cpu_d2 copy.py)
    "USE_CKPT": "hybrid-original",
    # Keep explicit for reproducibility (defaults are the same).
    "PYFLOW_HYBRID_EXP_SCALE": "1",
    "PYFLOW_HYBRID_PRUNE": "0",
}

HYBRID_SVD_ENV: dict[str, str] = {
    "USE_CKPT": "hybrid",
    # Enable Hybrid-SVD / rSVD compression path inside hybrid checkpointing.
    "PYFLOW_HYBRID_SVD": "1",
    "PYFLOW_HYBRID_COMPRESS": "hybrid-svd",
    # Keep explicit for reproducibility (defaults are the same).
    "PYFLOW_HYBRID_EXP_SCALE": "1",
    "PYFLOW_HYBRID_PRUNE": "0",
}


RE_RECURSIVE_STATS = re.compile(r"^\[RECURSIVE_STATS\]\s+(?P<json>\{.*\})\s*$")


def parse_memlog(memlog_path: Path) -> dict[str, Any]:
    """Compute net RSS during flow: peak - baseline between main:before_flow and main:after_flow."""
    if not memlog_path.exists():
        return {"error": f"missing memlog: {memlog_path}"}
    rows: list[dict[str, Any]] = []
    for line in memlog_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return {"error": "empty memlog"}

    baseline = None
    start_idx = 0
    for i, r in enumerate(rows):
        if r.get("tag") == "main:before_flow":
            baseline = float(r.get("rss_mb", 0.0))
            start_idx = i
            break
    if baseline is None:
        rss_all = [float(r.get("rss_mb", 0.0)) for r in rows if "rss_mb" in r]
        baseline = min(rss_all) if rss_all else 0.0

    end_idx = None
    for i in range(start_idx, len(rows)):
        if rows[i].get("tag") == "main:after_flow":
            end_idx = i
            break
    flow_rows = rows[start_idx : (end_idx + 1 if end_idx is not None else None)]
    rss_vals = [float(r.get("rss_mb")) for r in flow_rows if "rss_mb" in r]
    if not rss_vals:
        return {"error": "no rss during flow"}
    peak = max(rss_vals)
    net = max(0.0, peak - baseline)
    return {"rss_baseline_mb": baseline, "rss_peak_mb": peak, "rss_net_mb": net, "entries": len(rows)}


def run_case(
    L: int,
    env_case: EnvCase,
    mode: str,
    use_jit_flow: bool,
    run_dir: Path,
    base_env: dict[str, str],
) -> dict[str, Any]:
    """Run a single (env, L, mode) benchmark and return stats dict."""
    assert mode in ("hybrid-original", "hybrid-svd")
    n = L * L
    out_dir = run_dir / env_case.key / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    memlog_path = out_dir / f"memlog-L{L}-n{n}-{env_case.key}-{mode}.jsonl"

    env = os.environ.copy()
    env.update(base_env)
    env["USE_JIT_FLOW"] = "1" if use_jit_flow else "0"
    env["PYFLOW_MEMLOG_FILE"] = str(memlog_path)
    if env_case.extra_env:
        env.update(env_case.extra_env)

    if mode == "hybrid-original":
        env.update(HYBRID_ORIGINAL_ENV)
    else:
        env.update(HYBRID_SVD_ENV)

    cmd = [
        sys.executable,
        str(MAIN),
        str(L),
        env_case.dis_type,
        FIXED_METHOD,
        str(env_case.d),
        str(FIXED_P),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(CODE_DIR), env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    stats: dict[str, Any] = {
        "L": L,
        "n": n,
        "env": env_case.key,
        "mode": mode,
        "status": "ok" if proc.returncode == 0 else "error",
        "elapsed_s": elapsed,
        "memlog": str(memlog_path),
        "returncode": proc.returncode,
    }

    # Parse recursive stats if present
    rec_stats = None
    for line in proc.stdout.splitlines():
        m = RE_RECURSIVE_STATS.match(line.strip())
        if m:
            try:
                rec_stats = json.loads(m.group("json"))
            except Exception:
                rec_stats = None
    if rec_stats is not None:
        stats["recursive_stats"] = rec_stats

    if proc.returncode != 0:
        # keep last stderr lines for debugging
        stats["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-40:])
        return stats

    mem = parse_memlog(memlog_path)
    stats.update(mem)
    return stats


def print_table(
    results: list[dict[str, Any]],
    env_keys: list[str],
    L_min: int,
    L_max: int,
    *,
    cutoff: str,
    ode_tol: str,
    force_steps: str,
    memlog_every: str,
) -> None:
    # Group by env then by L
    by_env: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for r in results:
        by_env.setdefault(r["env"], {}).setdefault(int(r["L"]), {})[r["mode"]] = r

    print("\n" + "=" * 120)
    print("BENCHMARK (2D spinless fermion): Hybrid-Original vs Hybrid-SVD")
    print(
        f"Settings: method={FIXED_METHOD} cutoff={cutoff} ode_tol={ode_tol} steps={force_steps} memlog_every={memlog_every}"
    )
    print("=" * 120)

    for env_key in env_keys:
        print(f"\n### 环境: {env_key}")
        hdr = f"{'L':<4} {'n':<5} │ {'HybOrig(NetMB)':>14} {'HybSVD(NetMB)':>14} │ {'内存下降':>8}"
        print(hdr)
        print("─" * 78)
        for L in range(L_min, L_max + 1):
            row = by_env.get(env_key, {}).get(L, {})
            hyb_orig = row.get("hybrid-original")
            hyb_svd = row.get("hybrid-svd")

            def _fmt_mb(x):
                if not x or x.get("status") != "ok" or "rss_net_mb" not in x:
                    return "-"
                return f"{float(x['rss_net_mb']):.1f}"

            o_mb = _fmt_mb(hyb_orig)
            s_mb = _fmt_mb(hyb_svd)

            reduct = "-"
            if hyb_orig and hyb_svd and hyb_orig.get("status") == "ok" and hyb_svd.get("status") == "ok":
                o = float(hyb_orig.get("rss_net_mb", 0.0))
                s = float(hyb_svd.get("rss_net_mb", 0.0))
                if o > 1e-6:
                    reduct = f"{(1.0 - s / o) * 100.0:+.0f}%"

            print(f"{L:<4} {L*L:<5} │ {o_mb:>14} {s_mb:>14} │ {reduct:>8}")
        print("─" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--L",
        nargs=2,
        type=int,
        default=None,
        help=f"Override L range (inclusive): L_min L_max. If omitted, use defaults {L_MIN_DEFAULT}..{L_MAX_DEFAULT}.",
    )
    ap.add_argument("--only-env", nargs="*", default=None, help="Subset of env keys to run (space-separated)")
    ap.add_argument("--no-jit", action="store_true", help="Disable USE_JIT_FLOW (avoid compile overhead)")
    ap.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help=f"Override truncation cutoff (maps to PYFLOW_CUTOFF). Default is {FIXED_CUTOFF}.",
    )
    ap.add_argument(
        "--tol",
        type=str,
        default=None,
        help=f"Override ODE tolerance (rtol=atol; maps to PYFLOW_ODE_RTOL/ATOL). Default is {FIXED_ODE_TOL}.",
    )
    args = ap.parse_args()

    cutoff = str(args.cutoff) if args.cutoff is not None else FIXED_CUTOFF
    ode_tol = str(args.tol) if args.tol is not None else FIXED_ODE_TOL
    base_env = dict(BASE_ENV)
    base_env["PYFLOW_CUTOFF"] = cutoff
    base_env["PYFLOW_ODE_RTOL"] = ode_tol
    base_env["PYFLOW_ODE_ATOL"] = ode_tol

    # For printing tables (cover full union of L across modes)
    if args.L is None:
        L_min, L_max = L_MIN_DEFAULT, L_MAX_DEFAULT
    else:
        L_min, L_max = int(args.L[0]), int(args.L[1])
    L_min_all, L_max_all = L_min, L_max
    use_jit = not bool(args.no_jit)

    env_cases = ENV_CASES
    if args.only_env:
        want = set(args.only_env)
        env_cases = [e for e in ENV_CASES if e.key in want]
        missing = want.difference({e.key for e in env_cases})
        if missing:
            print(f"[WARN] Unknown env keys ignored: {sorted(missing)}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = REPO_ROOT / "test" / "bench_envs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("BENCHMARK PLAN")
    print(f"- L range: {L_min}..{L_max}")
    print(f"- Schedule: for each L={L_min}..{L_max}, run ALL envs: hybrid-original then hybrid-svd")
    print(f"- Envs: {[e.key for e in env_cases]}")
    print(f"- Modes: ['hybrid-original', 'hybrid-svd']")
    print(f"- Run dir: {run_dir}")
    print(f"- USE_JIT_FLOW: {'ON' if use_jit else 'OFF'}")
    print(f"- cutoff: {cutoff} | ode_tol: {ode_tol} | force_steps: {FIXED_FORCE_STEPS} | memlog_every: {FIXED_MEMLOG_EVERY}")
    print("=" * 120)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for L in range(L_min, L_max + 1):
        for env_case in env_cases:
            # hybrid-original
            print(f"\n[RUN] env={env_case.key} L={L} mode=hybrid-original ...", flush=True)
            r = run_case(L, env_case, "hybrid-original", use_jit, run_dir, base_env)
            results.append(r)
            if r.get("status") != "ok":
                print("  [FAILED]")
                print(r.get("stderr_tail", "(no stderr)"))
            else:
                print(
                    f"  [OK] net={r.get('rss_net_mb', float('nan')):.1f}MB peak={r.get('rss_peak_mb', float('nan')):.1f}MB elapsed={r.get('elapsed_s', 0.0):.1f}s"
                )

            # hybrid-svd
            print(f"\n[RUN] env={env_case.key} L={L} mode=hybrid-svd ...", flush=True)
            r = run_case(L, env_case, "hybrid-svd", use_jit, run_dir, base_env)
            results.append(r)
            if r.get("status") != "ok":
                print("  [FAILED]")
                print(r.get("stderr_tail", "(no stderr)"))
            else:
                print(
                    f"  [OK] net={r.get('rss_net_mb', float('nan')):.1f}MB peak={r.get('rss_peak_mb', float('nan')):.1f}MB elapsed={r.get('elapsed_s', 0.0):.1f}s"
                )

    # Save JSON
    out_json = run_dir / "results.json"
    out_json.write_text(json.dumps({"run_dir": str(run_dir), "results": results}, indent=2))
    print(f"\nSaved: {out_json}")

    print_table(
        results,
        [e.key for e in env_cases],
        L_min_all,
        L_max_all,
        cutoff=cutoff,
        ode_tol=ode_tol,
        force_steps=FIXED_FORCE_STEPS,
        memlog_every=FIXED_MEMLOG_EVERY,
    )
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

