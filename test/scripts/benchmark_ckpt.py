#!/usr/bin/env python3
"""
Benchmark script with flexible planning for Mixed-Precision & Recursive Checkpointing.
Updated to report NET MEMORY usage (Flow Phase Peak - Baseline).

Features:
- Calculates "Net Memory" to isolate algorithm cost from Python/JAX overhead.
- Define custom ranges per mode.
- Strict serial execution.
- Auto-aggregated reporting.

Usage Examples:
    python benchmark_ckpt.py --plan "original:3-4 recursive:3-6 hybrid:3-8"
"""

import os
import sys
import json
import subprocess
import time
import shutil
import math
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
CODE_DIR = REPO_ROOT / "code"
TEST_DIR = REPO_ROOT / "test"

# Default parameters
DEFAULT_DIS_TYPE = "random"
DEFAULT_METHOD = "tensordot"
DEFAULT_D = 1.0
DEFAULT_P = 0
DEFAULT_ODE_TOL = "1e-4"
DEFAULT_CUTOFF = None  # pass via --cutoff (maps to PYFLOW_CUTOFF)

# Process Execution Settings
PROCESS_TIMEOUT = 360000 

# Get CPU count (psutil optional; base env may not have it)
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
    "PYFLOW_MEMLOG": "1",
    "PYFLOW_MEMLOG_EVERY": "5",
    "PYFLOW_TIMELOG": "1",
    "PYFLOW_SCRAMBLE": "0",
    "PYFLOW_OVERWRITE": "1",
    "OMP_NUM_THREADS": _NUM_CORES,
    "MKL_NUM_THREADS": _NUM_CORES,
    "OPENBLAS_NUM_THREADS": _NUM_CORES,
    "NUMBA_NUM_THREADS": _NUM_CORES,
    "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={_NUM_CORES}",
    # Optional: fast memory-profiling mode (skip expensive ODEs but keep allocations realistic).
    # Enable by running benchmark_ckpt.py with env var PYFLOW_FASTMEM=1.
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_memlog(filepath: Path) -> dict:
    """Parse a memlog JSONL file and extract NET memory statistics."""
    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}
    
    rows = []
    try:
        for line in filepath.read_text().splitlines():
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception as e:
        return {"error": str(e)}
    
    if not rows:
        return {"error": "No valid entries in memlog"}
    
    # 1. Find Baseline (RSS just before flow starts)
    baseline_rss = 0.0
    flow_start_idx = 0
    
    # Look for the specific tag emitted by main_itc_cpu_d2.py
    for i, r in enumerate(rows):
        if r.get("tag") == "main:before_flow":
            baseline_rss = r.get("rss_mb", 0.0)
            flow_start_idx = i
            break
    
    # Fallback: if tag not found, use min RSS as baseline (rough approximation)
    if baseline_rss == 0.0:
        rss_all = [r.get("rss_mb", 0) for r in rows if "rss_mb" in r]
        baseline_rss = min(rss_all) if rss_all else 0.0

    # 2. Find Peak during Flow
    # We only look at entries AFTER the flow started
    flow_rows = rows[flow_start_idx:]
    rss_values = [r.get("rss_mb") for r in flow_rows if "rss_mb" in r]
    
    if not rss_values:
        return {"error": "No RSS values found during flow phase"}
    
    peak_rss = max(rss_values)
    net_rss = peak_rss - baseline_rss
    
    # Clamp negative net_rss to 0 (shouldn't happen, but just in case)
    net_rss = max(0.0, net_rss)

    return {
        "entries": len(rows),
        "rss_peak_mb": peak_rss,       # Absolute Peak
        "rss_baseline_mb": baseline_rss, # Baseline
        "rss_net_mb": net_rss          # Net Increase (Algorithm Cost)
    }

def _extract_tag_rows(memlog_path: Path) -> dict:
    """Parse memlog JSONL and return a dict tag->row (last occurrence wins)."""
    tags = {}
    try:
        lines = memlog_path.read_text().splitlines()
    except Exception:
        return tags
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tag = obj.get("tag")
        if tag:
            tags[tag] = obj
    return tags

def append_predicted_peak(memlog_path: Path) -> None:
    """
    Append a JSONL record (tag=fast:predicted_peak) to the memlog itself.

    Rules (as agreed):
    - original/ckpt: extrapolate storage by (T_total / store_steps)
    - ckpt/recursive/hybrid: add one bck segment/base-case buffer peak (measured via tags)
    - recursive/hybrid: bck peak upper bound also includes binary recursion depth stack
    - always include transient peak above (storage + bck buffer) from measured peak
    """
    if not memlog_path.exists():
        return

    # Avoid duplicates within the same file
    try:
        tail = memlog_path.read_text().splitlines()[-8:]
        for line in tail:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("tag") == "fast:predicted_peak":
                return
    except Exception:
        pass

    tags = _extract_tag_rows(memlog_path)
    main_before = tags.get("main:before_flow", {})
    baseline = main_before.get("rss_mb")
    L = main_before.get("L")
    n = main_before.get("n")

    stats = parse_memlog(memlog_path)
    peak = stats.get("rss_peak_mb")

    meta = tags.get("fast:before_storage", {})
    mode = meta.get("mode") or main_before.get("mode")
    T_total = meta.get("T_total")
    store_steps = meta.get("store_steps")
    ckpt_step = meta.get("ckpt_step")
    bck_steps = meta.get("bck_steps")

    def f(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    def i(x, default=None):
        try:
            return int(x)
        except Exception:
            return default

    baseline = f(baseline)
    peak = f(peak)
    L = i(L)
    n = i(n)
    T_total = i(T_total)
    store_steps = i(store_steps)
    ckpt_step = i(ckpt_step)
    bck_steps = i(bck_steps)

    rss_before_storage = f(tags.get("fast:before_storage", {}).get("rss_mb"), baseline)
    rss_after_storage = f(tags.get("fast:after_storage_k", {}).get("rss_mb"), rss_before_storage)
    rss_after_bck_alloc = f(tags.get("fast:after_bck_alloc", {}).get("rss_mb"), rss_after_storage)

    storage_delta_k = None
    if rss_before_storage is not None and rss_after_storage is not None:
        storage_delta_k = max(0.0, rss_after_storage - rss_before_storage)

    bck_delta = 0.0
    if rss_after_bck_alloc is not None and rss_after_storage is not None:
        bck_delta = max(0.0, rss_after_bck_alloc - rss_after_storage)

    transient_delta = 0.0
    if peak is not None:
        ref = rss_after_storage or 0.0
        if rss_after_bck_alloc is not None:
            ref = max(ref, rss_after_bck_alloc)
        transient_delta = max(0.0, peak - float(ref))

    storage_full = 0.0
    if mode in ("original", "ckpt") and storage_delta_k is not None and T_total and store_steps and store_steps > 0:
        storage_full = float(storage_delta_k) * (float(T_total) / float(store_steps))

    depth = None
    stack_mb = 0.0
    if mode in ("recursive", "hybrid") and T_total and bck_steps and bck_steps > 0 and n:
        ratio = max(1.0, float(T_total) / float(bck_steps))
        depth = int(math.ceil(math.log(ratio, 2.0)))
        n2 = n * n
        n4 = n2 * n2
        state_bytes = (n2 + n4) * 4  # float32
        stack_mb = (state_bytes / 1e6) * float(depth)

    predicted_full_peak = None
    if baseline is not None:
        if mode == "original":
            predicted_full_peak = baseline + storage_full + transient_delta
        elif mode == "ckpt":
            predicted_full_peak = baseline + storage_full + bck_delta + transient_delta
        elif mode in ("recursive", "hybrid"):
            predicted_full_peak = baseline + bck_delta + stack_mb + transient_delta

    record = {
        "tag": "fast:predicted_peak",
        "L": L,
        "n": n,
        "mode": mode,
        "T_total": T_total,
        "store_steps": store_steps,
        "ckpt_step": ckpt_step,
        "bck_steps": bck_steps,
        "rss_baseline_mb": baseline,
        "rss_peak_mb_measured": peak,
        "rss_peak_mb_full_est": predicted_full_peak,
        "storage_delta_k_mb": storage_delta_k,
        "storage_full_est_mb": storage_full,
        "bck_delta_mb": bck_delta,
        "transient_delta_mb": transient_delta,
        "recursive_depth": depth,
        "recursive_stack_est_mb": stack_mb,
        "rule": "storage_extrapolate + bck_buffer + transient; recursive/hybrid add recursion_depth*state_size",
    }

    try:
        with memlog_path.open("a") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return

def run_single_benchmark(L: int, mode: str, dis_type: str, method: str, d: float, p: int,
                         verbose: bool, ode_tol: str, use_jit: bool, cutoff: str | None = None) -> dict:
    """Run benchmark for a specific mode."""
    
    if mode == "original":
        use_ckpt_val = "0"
    elif mode == "ckpt":
        use_ckpt_val = "1"
    elif mode == "recursive":
        use_ckpt_val = "recursive"
    elif mode == "hybrid":
        use_ckpt_val = "hybrid"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fastmem = os.environ.get("PYFLOW_FASTMEM", "0") in ("1", "true", "True")
    mode_dir = (TEST_DIR / "fast" / mode) if fastmem else (TEST_DIR / mode)
    mode_dir.mkdir(parents=True, exist_ok=True)

    dim = 2; order = 4; x = 0.0; delta = 0.1
    log_suffix = "-fastmem" if fastmem else ""
    log_filename = f"memlog-dim{dim}-L{L}-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}-{mode}{log_suffix}.jsonl"
    memlog_path = mode_dir / log_filename
    
    if memlog_path.exists():
        memlog_path.unlink()

    env = os.environ.copy()
    env.update(BASE_ENV)
    env["USE_CKPT"] = use_ckpt_val
    env["USE_JIT_FLOW"] = "1" if use_jit else "0"
    env["PYFLOW_ODE_RTOL"] = ode_tol
    env["PYFLOW_ODE_ATOL"] = ode_tol
    if cutoff is not None:
        env["PYFLOW_CUTOFF"] = str(cutoff)

    cmd = [
        sys.executable,
        str(CODE_DIR / "main_itc_cpu_d2.py"),
        str(L), dis_type, method, str(d), str(p)
    ]

    jit_tag = "[JIT]" if use_jit else ""
    print(f"  Running L={L}, mode={mode:<10} {jit_tag} ...", end="", flush=True)
    start_time = time.time()
    
    try:
        if verbose:
            print() 
            result = subprocess.run(cmd, cwd=CODE_DIR, env=env, timeout=PROCESS_TIMEOUT)
        else:
            result = subprocess.run(cmd, cwd=CODE_DIR, env=env, capture_output=True, text=True, timeout=PROCESS_TIMEOUT)

        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f" [FAILED] ({elapsed:.1f}s)")
            if not verbose:
                err_lines = result.stderr.strip().splitlines()[-3:] if result.stderr else ['Unknown Error']
                for line in err_lines:
                    print(f"    ! {line}")
            return {"L": L, "mode": mode, "status": "error", "error": "Process failed"}
        
        stats = parse_memlog(memlog_path)
        if "error" in stats:
            print(f" [NO LOG] ({elapsed:.1f}s)")
            return {"L": L, "mode": mode, "status": "error", "error": stats["error"]}
        
        # Display NET memory in the live progress
        print(f" [OK] Net: {stats['rss_net_mb']:.1f} MB (Abs: {stats['rss_peak_mb']:.0f}) ({elapsed:.1f}s)")

        # Fastmem: write predicted full-flow peak back into the memlog itself
        if fastmem:
            append_predicted_peak(memlog_path)

        return {
            "L": L, "mode": mode, "status": "ok", 
            "elapsed_s": elapsed, "memlog": str(memlog_path), **stats
        }

    except subprocess.TimeoutExpired:
        print(" [TIMEOUT]")
        return {"L": L, "mode": mode, "status": "timeout"}
    except KeyboardInterrupt:
        print("\n[Aborted by user]")
        sys.exit(1)
    except Exception as e:
        print(f" [ERROR] {e}")
        return {"L": L, "mode": mode, "status": "error", "error": str(e)}

def collect_all_results(L_min, L_max, d, p):
    modes = ["original", "ckpt", "recursive", "hybrid"]
    data = {} 
    dim = 2; order = 4; x = 0.0; delta = 0.1
    fastmem = os.environ.get("PYFLOW_FASTMEM", "0") in ("1", "true", "True")
    base_dir = (TEST_DIR / "fast") if fastmem else TEST_DIR
    for L in range(L_min, L_max + 1):
        data[L] = {}
        for mode in modes:
            log_suffix = "-fastmem" if fastmem else ""
            filename = f"memlog-dim{dim}-L{L}-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}-{mode}{log_suffix}.jsonl"
            path = base_dir / mode / filename
            if path.exists():
                stats = parse_memlog(path)
                if "error" not in stats:
                    data[L][mode] = stats
                    data[L][mode]["path"] = str(path)
    return data

def print_report(data, L_min, L_max):
    """Print consolidated report table focusing on NET MEMORY."""
    print("\n" + "=" * 110)
    print("BENCHMARK REPORT: Net Memory Usage (Peak - Baseline) in MB")
    print("This metric isolates the algorithm's memory cost from Python/JAX overhead.")
    print("=" * 110)
    
    # Headers
    headers = f"{'L':<4} {'n':<4} │ {'Original':>10} {'Linear':>10} {'Recursive':>10} {'Hybrid':>10} │ {'Reduct(Rec)':>12} {'Reduct(Hyb)':>12}"
    print(headers)
    print("─" * 110)

    for L in range(L_min, L_max + 1):
        if L not in data or not data[L]:
            continue
            
        row = data[L]
        n = L*L
        
        # Helper to get NET memory
        def get_val(m):
            return row.get(m, {}).get("rss_net_mb", None)
        
        orig = get_val("original")
        lin = get_val("ckpt")
        rec = get_val("recursive")
        hyb = get_val("hybrid")
        
        s_orig = f"{orig:.0f}" if orig is not None else "-"
        s_lin = f"{lin:.0f}" if lin is not None else "-"
        s_rec = f"{rec:.0f}" if rec is not None else "-"
        s_hyb = f"{hyb:.0f}" if hyb is not None else "-"
        
        # Calculate reductions relative to Original NET memory
        # Note: If Original Net is very small (e.g. < 1MB), reduction might be noisy
        def calc_reduct(val):
            if orig is not None and val is not None and orig > 1.0:
                return f"{(1 - val/orig)*100:+.0f}%"
            return "-"

        r_rec = calc_reduct(rec)
        r_hyb = calc_reduct(hyb)
        
        print(f"{L:<4} {n:<4} │ {s_orig:>10} {s_lin:>10} {s_rec:>10} {s_hyb:>10} │ {r_rec:>12} {r_hyb:>12}")
    print("─" * 110)

def parse_plan(plan_str):
    tasks = [] 
    parts = plan_str.split()
    for part in parts:
        try:
            mode_part, range_part = part.split(':')
            start, end = map(int, range_part.split('-'))
            tasks.append({
                "mode": mode_part,
                "range": range(start, end + 1)
            })
        except ValueError:
            print(f"Error: Invalid plan format '{part}'. Expected 'mode:start-end'.")
            sys.exit(1)
    return tasks

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    plan_str = None
    mode_arg = "all"
    verbose = False
    use_jit = False
    ode_tol = DEFAULT_ODE_TOL
    cutoff = DEFAULT_CUTOFF
    pos_args = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--plan="):
            plan_str = arg.split("=", 1)[1]
        elif arg == "--plan":
            i += 1
            if i < len(args): plan_str = args[i]
        elif arg.startswith("--mode="):
            mode_arg = arg.split("=", 1)[1]
        elif arg in ("--verbose", "-v"):
            verbose = True
        elif arg == "--jit":
            use_jit = True
        elif arg.startswith("--tol="):
            ode_tol = arg.split("=", 1)[1]
        elif arg.startswith("--cutoff="):
            cutoff = arg.split("=", 1)[1]
        elif arg == "--cutoff":
            i += 1
            if i < len(args):
                cutoff = args[i]
        else:
            pos_args.append(arg)
        i += 1
            
    dis_type = pos_args[2] if len(pos_args) > 2 else DEFAULT_DIS_TYPE
    method = pos_args[3] if len(pos_args) > 3 else DEFAULT_METHOD
    d = float(pos_args[4]) if len(pos_args) > 4 else DEFAULT_D
    p = int(pos_args[5]) if len(pos_args) > 5 else DEFAULT_P

    raw_tasks = []
    if plan_str:
        raw_tasks = parse_plan(plan_str)
        all_L = [L for t in raw_tasks for L in t["range"]]
        L_min_global = min(all_L) if all_L else 3
        L_max_global = max(all_L) if all_L else 3
    else:
        L_min_arg = int(pos_args[0]) if len(pos_args) > 0 else 3
        L_max_arg = int(pos_args[1]) if len(pos_args) > 1 else 6
        valid_modes = ["original", "ckpt", "recursive", "hybrid"]
        modes_to_run = valid_modes if mode_arg == "all" else [mode_arg]
        for m in modes_to_run:
            raw_tasks.append({"mode": m, "range": range(L_min_arg, L_max_arg + 1)})
        L_min_global = L_min_arg
        L_max_global = L_max_arg

    # L-first Execution Sort
    flat_tasks = []
    for t in raw_tasks:
        mode = t["mode"]
        for L in t["range"]:
            flat_tasks.append((L, mode))
    mode_priority = {"original": 0, "ckpt": 1, "recursive": 2, "hybrid": 3}
    flat_tasks.sort(key=lambda x: (x[0], mode_priority.get(x[1], 99)))

    # Fastmem: treat test/fast as a fresh run directory (clear once per benchmark job)
    if os.environ.get("PYFLOW_FASTMEM", "0") in ("1", "true", "True"):
        fast_dir = TEST_DIR / "fast"
        shutil.rmtree(fast_dir, ignore_errors=True)
        fast_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print(f"BENCHMARK EXECUTION PLAN (L-First, Net Memory Reporting)")
    print(f"ODE tol: {ode_tol} | cutoff: {cutoff if cutoff is not None else 'DEFAULT'} | JIT: {'ON' if use_jit else 'OFF'}")
    print("=" * 110)
    current_L = -1
    for L, mode in flat_tasks:
        if L != current_L:
            print(f"  L={L}: ", end="")
            current_L = L
        print(f"[{mode}] ", end="")
    print("\n" + "=" * 110)
    
    for L, mode in flat_tasks:
        run_single_benchmark(L, mode, dis_type, method, d, p, verbose, ode_tol, use_jit, cutoff=cutoff)

    print("\nGenerating Consolidated Report (Net Memory)...")
    all_data = collect_all_results(L_min_global, L_max_global, d, p)
    print_report(all_data, L_min_global, L_max_global)

    # (fastmem summary is appended per-run)

if __name__ == "__main__":
    main()