#!/usr/bin/env python3
"""
Benchmark (paper-oriented): Original vs Adaptive (on top of recursive checkpointing) across multiple physical
environments (2D only).

Design goals:
- Flow-equation / numerical settings are FIXED inside this script (paper reproducibility).
- Physical environments (disorder/potential type + strength) are enumerated here.
- Runs ORIGINAL and ADAPTIVE for each (env, L) pair with identical forced steps.
- Produces net-memory numbers from memlog (peak RSS - baseline RSS) during the *flow phase only*.
- Also captures [RECURSIVE_STATS] JSON from stdout for extra diagnostics (recomputed steps, etc.).

Usage (one command):
  python test/scripts/benchmark_envs_original_vs_tau.py

Optional:
  python test/scripts/benchmark_envs_original_vs_tau.py --L 3 10
  python test/scripts/benchmark_envs_original_vs_tau.py --only-env random_d1 qp_golden_d1
  python test/scripts/benchmark_envs_original_vs_tau.py --no-jit
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
MAIN = CODE_DIR / "main_itc_cpu_d2.py"

# -----------------------------------------------------------------------------
# Fixed L ranges (paper-oriented)
# -----------------------------------------------------------------------------
# Original is more memory-hungry; cap at smaller L.
ORIG_L_MIN, ORIG_L_MAX = 2, 8
# Adaptive (on recursive) can scale further; extend L if desired.
ADAPT_L_MIN, ADAPT_L_MAX = 2, 8


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
    EnvCase("random_d4", "random", 4.0),
    EnvCase("qp_golden_d1", "QPgolden", 1.0),
    EnvCase("qp_golden_d4", "QPgolden", 4.0),
    EnvCase("linear_d1", "linear", 1.0),
    # curved uses xlist=[1.] inside main_itc_cpu_d2.py; keep d moderate
    EnvCase("curved_d1", "curved", 1.0),
]


# -----------------------------------------------------------------------------
# Adaptive configuration (runs on top of recursive checkpointing)
# -----------------------------------------------------------------------------
ADAPTIVE_ENV: dict[str, str] = {
    "USE_CKPT": "recursive",
    # Split strategy is irrelevant to adaptive grid itself; keep it explicit to avoid inheriting
    # shell/user env vars (e.g. a leftover "tau" split).
    "PYFLOW_RECURSIVE_SPLIT": "uniform",
    # Enable adaptive grid (controller). Since this script forces a fixed number of steps for
    # reproducibility, explicitly allow adaptive grid even when PYFLOW_FORCE_STEPS is set.
    "PYFLOW_ADAPTIVE_GRID": "1",
    "PYFLOW_ADAPTIVE_METHOD": "controller",
    "PYFLOW_ADAPTIVE_ALLOW_FORCE": "1",
    # Defaults tuned for benchmarking; override by exporting these env vars if needed.
    "PYFLOW_ADAPTIVE_TARGET": "1e-2",
    "PYFLOW_ADAPTIVE_INCLUDE_H4": "0",
    "PYFLOW_ADAPTIVE_W4": "1.0",
    "PYFLOW_ADAPTIVE_MIN_DL": "1e-8",
    "PYFLOW_ADAPTIVE_MAX_DL": "1e2",
    "PYFLOW_ADAPTIVE_MAX_STEPS": "20000",
    "PYFLOW_ADAPTIVE_LOG_EVERY": "200",
    # Helpful: print stats JSON
    "PYFLOW_RECURSIVE_STATS": "1",
}

ORIG_ENV: dict[str, str] = {
    "USE_CKPT": "0",
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
) -> dict[str, Any]:
    """Run a single (env, L, mode) benchmark and return stats dict."""
    assert mode in ("original", "adaptive")
    n = L * L
    out_dir = run_dir / env_case.key / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    memlog_path = out_dir / f"memlog-L{L}-n{n}-{env_case.key}-{mode}.jsonl"

    env = os.environ.copy()
    env.update(BASE_ENV)
    env["USE_JIT_FLOW"] = "1" if use_jit_flow else "0"
    env["PYFLOW_MEMLOG_FILE"] = str(memlog_path)
    if env_case.extra_env:
        env.update(env_case.extra_env)

    if mode == "original":
        env.update(ORIG_ENV)
    else:
        env.update(ADAPTIVE_ENV)

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


def print_table(results: list[dict[str, Any]], env_keys: list[str], L_min: int, L_max: int) -> None:
    # Group by env then by L
    by_env: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for r in results:
        by_env.setdefault(r["env"], {}).setdefault(int(r["L"]), {})[r["mode"]] = r

    print("\n" + "=" * 120)
    print("BENCHMARK (2D spinless fermion): Original vs Adaptive (on recursive)")
    print(f"Fixed: method={FIXED_METHOD} cutoff={FIXED_CUTOFF} ode_tol={FIXED_ODE_TOL} steps={FIXED_FORCE_STEPS} memlog_every={FIXED_MEMLOG_EVERY}")
    print("=" * 120)

    for env_key in env_keys:
        print(f"\n### 环境: {env_key}")
        hdr = f"{'L':<4} {'n':<5} │ {'Original(NetMB)':>14} {'Adaptive(NetMB)':>16} │ {'内存下降':>8} {'重算步数':>10}"
        print(hdr)
        print("─" * 90)
        for L in range(L_min, L_max + 1):
            row = by_env.get(env_key, {}).get(L, {})
            orig = row.get("original")
            adp = row.get("adaptive")

            def _fmt_mb(x):
                if not x or x.get("status") != "ok" or "rss_net_mb" not in x:
                    return "-"
                return f"{float(x['rss_net_mb']):.1f}"

            o_mb = _fmt_mb(orig)
            a_mb = _fmt_mb(adp)

            reduct = "-"
            if orig and adp and orig.get("status") == "ok" and adp.get("status") == "ok":
                o = float(orig.get("rss_net_mb", 0.0))
                a = float(adp.get("rss_net_mb", 0.0))
                if o > 1e-6:
                    reduct = f"{(1.0 - a / o) * 100.0:+.0f}%"

            recompute = "-"
            if adp and adp.get("recursive_stats"):
                recompute = str(adp["recursive_stats"].get("recomputed_steps", "-"))

            print(f"{L:<4} {L*L:<5} │ {o_mb:>14} {a_mb:>16} │ {reduct:>8} {recompute:>10}")
        print("─" * 90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--L",
        nargs=2,
        type=int,
        default=None,
        help="Override L range for BOTH modes: L_min L_max (inclusive). If omitted, use paper defaults (orig 2..8, recursive_tau 2..10).",
    )
    ap.add_argument("--only-env", nargs="*", default=None, help="Subset of env keys to run (space-separated)")
    ap.add_argument("--no-jit", action="store_true", help="Disable USE_JIT_FLOW (avoid compile overhead)")
    args = ap.parse_args()

    # For printing tables (cover full union of L across modes)
    if args.L is None:
        orig_L_min, orig_L_max = ORIG_L_MIN, ORIG_L_MAX
        adapt_L_min, adapt_L_max = ADAPT_L_MIN, ADAPT_L_MAX
    else:
        orig_L_min, orig_L_max = int(args.L[0]), int(args.L[1])
        adapt_L_min, adapt_L_max = orig_L_min, orig_L_max

    L_min_all = min(orig_L_min, adapt_L_min)
    L_max_all = max(orig_L_max, adapt_L_max)
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
    print(f"- L range (original): {orig_L_min}..{orig_L_max}")
    print(f"- L range (adaptive): {adapt_L_min}..{adapt_L_max}")
    if args.L is None:
        print(
            f"- Schedule: for each L={orig_L_min}..{orig_L_max}, run ALL envs: original then adaptive; "
            f"after ALL L done, run adaptive-only for L={max(adapt_L_min, orig_L_max + 1)}..{adapt_L_max}"
        )
    else:
        print(f"- Schedule: for each L={orig_L_min}..{orig_L_max}, run ALL envs: original then adaptive")
    print(f"- Envs: {[e.key for e in env_cases]}")
    print(f"- Modes: ['original', 'adaptive']")
    print(f"- Run dir: {run_dir}")
    print(f"- USE_JIT_FLOW: {'ON' if use_jit else 'OFF'}")
    print("=" * 120)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    # Phase 1: for each L within original-range; run ALL envs: original then adaptive
    for L in range(orig_L_min, orig_L_max + 1):
        for env_case in env_cases:
            # original
            print(f"\n[RUN] env={env_case.key} L={L} mode=original ...", flush=True)
            r = run_case(L, env_case, "original", use_jit, run_dir)
            results.append(r)
            if r.get("status") != "ok":
                print("  [FAILED]")
                print(r.get("stderr_tail", "(no stderr)"))
            else:
                print(
                    f"  [OK] net={r.get('rss_net_mb', float('nan')):.1f}MB peak={r.get('rss_peak_mb', float('nan')):.1f}MB elapsed={r.get('elapsed_s', 0.0):.1f}s"
                )

            # adaptive for the same L (only if within adaptive range)
            if adapt_L_min <= L <= adapt_L_max:
                print(f"\n[RUN] env={env_case.key} L={L} mode=adaptive ...", flush=True)
                r = run_case(L, env_case, "adaptive", use_jit, run_dir)
                results.append(r)
                if r.get("status") != "ok":
                    print("  [FAILED]")
                    print(r.get("stderr_tail", "(no stderr)"))
                else:
                    print(
                        f"  [OK] net={r.get('rss_net_mb', float('nan')):.1f}MB peak={r.get('rss_peak_mb', float('nan')):.1f}MB elapsed={r.get('elapsed_s', 0.0):.1f}s"
                    )

    # Phase 2: after ALL envs finish phase 1, run adaptive-only for L beyond original max (e.g., 9..10)
    adapt_only_start = max(adapt_L_min, orig_L_max + 1)
    if adapt_only_start <= adapt_L_max:
        for L in range(adapt_only_start, adapt_L_max + 1):
            for env_case in env_cases:
                print(f"\n[RUN] env={env_case.key} L={L} mode=adaptive ...", flush=True)
                r = run_case(L, env_case, "adaptive", use_jit, run_dir)
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

    print_table(results, [e.key for e in env_cases], L_min_all, L_max_all)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

