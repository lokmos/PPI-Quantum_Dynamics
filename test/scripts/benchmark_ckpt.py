#!/usr/bin/env python3
"""
Benchmark script to compare memory usage between original and checkpoint modes.

Usage:
    python benchmark_ckpt.py [L_min] [L_max] [dis_type] [method] [d] [p]

Examples:
    python benchmark_ckpt.py 3 6           # L from 3 to 6 with defaults
    python benchmark_ckpt.py 4 8 random tensordot 1.0 0
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()  # test/scripts/
REPO_ROOT = SCRIPT_DIR.parent.parent          # Quantum Dynamics/
CODE_DIR = REPO_ROOT / "code"                 # code/ (where main_itc_cpu_d2.py lives)
TEST_DIR = REPO_ROOT / "test"                 # test/ (memlog output)

# Default parameters
DEFAULT_DIS_TYPE = "random"
DEFAULT_METHOD = "tensordot"
DEFAULT_D = 1.0
DEFAULT_P = 0

# Environment variables for consistent runs
BASE_ENV = {
    "PYFLOW_MEMLOG": "1",
    "PYFLOW_MEMLOG_EVERY": "50",
    "PYFLOW_TIMELOG": "1",
    "PYFLOW_SCRAMBLE": "0",
    "PYFLOW_ODE_RTOL": "1e-6",
    "PYFLOW_ODE_ATOL": "1e-6",
    "PYFLOW_OVERWRITE": "1",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def parse_memlog(filepath: Path) -> dict:
    """Parse a memlog JSONL file and extract statistics."""
    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}
    
    rows = []
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    if not rows:
        return {"error": "No valid entries in memlog"}
    
    rss_values = [r.get("rss_mb") for r in rows if "rss_mb" in r]
    if not rss_values:
        return {"error": "No RSS values found"}
    
    peak_entry = max((r for r in rows if "rss_mb" in r), key=lambda r: r["rss_mb"])
    
    return {
        "entries": len(rows),
        "rss_min_mb": min(rss_values),
        "rss_max_mb": max(rss_values),
        "rss_peak_mb": peak_entry["rss_mb"],
        "peak_tag": peak_entry.get("tag", ""),
        "peak_step": peak_entry.get("step", ""),
    }


def run_benchmark(L: int, dis_type: str, method: str, d: float, p: int, use_ckpt: bool) -> dict:
    """Run a single benchmark and return results."""
    mode = "ckpt" if use_ckpt else "original"
    dim = 2
    order = 4
    x = 0.0
    delta = 0.1  # Jz
    
    # Expected memlog filename (matches main_itc_cpu_d2.py naming)
    log_filename = f"memlog-dim{dim}-L{L}-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}-{mode}.jsonl"
    memlog_path = TEST_DIR / log_filename
    
    # Remove existing memlog to ensure fresh run
    if memlog_path.exists():
        memlog_path.unlink()
    
    # Build environment
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["USE_CKPT"] = "1" if use_ckpt else "0"
    
    # Build command - run from CODE_DIR where main_itc_cpu_d2.py lives
    cmd = [
        sys.executable,
        str(CODE_DIR / "main_itc_cpu_d2.py"),
        str(L),
        dis_type,
        method,
        str(d),
        str(p),
    ]
    
    print(f"  Running L={L}, mode={mode}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=CODE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"    ⚠ Process exited with code {result.returncode}")
            # Print last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")[-5:]
            for line in stderr_lines:
                print(f"      {line}")
            return {
                "L": L,
                "mode": mode,
                "status": "error",
                "returncode": result.returncode,
                "elapsed_s": elapsed,
            }
        
        # Parse memlog
        stats = parse_memlog(memlog_path)
        
        return {
            "L": L,
            "n": L ** dim,
            "mode": mode,
            "status": "ok",
            "elapsed_s": elapsed,
            "memlog_file": str(memlog_path),
            **stats,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "L": L,
            "mode": mode,
            "status": "timeout",
        }
    except Exception as e:
        return {
            "L": L,
            "mode": mode,
            "status": "exception",
            "error": str(e),
        }


def print_results_table(results: list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Memory Usage Comparison (Original vs Checkpoint)")
    print("=" * 80)
    
    # Group by L
    by_L = {}
    for r in results:
        L = r["L"]
        if L not in by_L:
            by_L[L] = {}
        by_L[L][r["mode"]] = r
    
    # Header
    print(f"{'L':>4} {'n':>5} │ {'Original (MB)':>14} {'Ckpt (MB)':>14} │ {'Reduction':>10} {'Time Orig':>10} {'Time Ckpt':>10}")
    print("─" * 80)
    
    for L in sorted(by_L.keys()):
        modes = by_L[L]
        n = L ** 2
        
        orig = modes.get("original", {})
        ckpt = modes.get("ckpt", {})
        
        orig_mem = orig.get("rss_peak_mb", "N/A")
        ckpt_mem = ckpt.get("rss_peak_mb", "N/A")
        orig_time = orig.get("elapsed_s", "N/A")
        ckpt_time = ckpt.get("elapsed_s", "N/A")
        
        if isinstance(orig_mem, (int, float)) and isinstance(ckpt_mem, (int, float)):
            if orig_mem > 0:
                reduction = (1 - ckpt_mem / orig_mem) * 100
                reduction_str = f"{reduction:+.1f}%"
            else:
                reduction_str = "N/A"
            orig_mem_str = f"{orig_mem:.1f}"
            ckpt_mem_str = f"{ckpt_mem:.1f}"
        else:
            reduction_str = "N/A"
            orig_mem_str = str(orig_mem) if orig.get("status") == "ok" else orig.get("status", "?")
            ckpt_mem_str = str(ckpt_mem) if ckpt.get("status") == "ok" else ckpt.get("status", "?")
        
        if isinstance(orig_time, (int, float)):
            orig_time_str = f"{orig_time:.1f}s"
        else:
            orig_time_str = str(orig_time)
        
        if isinstance(ckpt_time, (int, float)):
            ckpt_time_str = f"{ckpt_time:.1f}s"
        else:
            ckpt_time_str = str(ckpt_time)
        
        print(f"{L:>4} {n:>5} │ {orig_mem_str:>14} {ckpt_mem_str:>14} │ {reduction_str:>10} {orig_time_str:>10} {ckpt_time_str:>10}")
    
    print("─" * 80)
    print("Note: Reduction > 0 means checkpoint mode uses LESS memory")
    print("      Reduction < 0 means checkpoint mode uses MORE memory")


def save_results_json(results: list, output_path: Path):
    """Save detailed results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nDetailed results saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Parse arguments
    args = sys.argv[1:]
    
    L_min = int(args[0]) if len(args) > 0 else 3
    L_max = int(args[1]) if len(args) > 1 else 6
    dis_type = args[2] if len(args) > 2 else DEFAULT_DIS_TYPE
    method = args[3] if len(args) > 3 else DEFAULT_METHOD
    d = float(args[4]) if len(args) > 4 else DEFAULT_D
    p = int(args[5]) if len(args) > 5 else DEFAULT_P
    
    print("=" * 80)
    print("Flow Equation Checkpoint Benchmark")
    print("=" * 80)
    print(f"L range:   {L_min} to {L_max}")
    print(f"dis_type:  {dis_type}")
    print(f"method:    {method}")
    print(f"d:         {d}")
    print(f"p:         {p}")
    print(f"Output:    {TEST_DIR}")
    print("=" * 80)
    
    # Ensure test directory exists
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for L in range(L_min, L_max + 1):
        print(f"\n[L={L}, n={L**2}]")
        
        # Run original mode first
        result_orig = run_benchmark(L, dis_type, method, d, p, use_ckpt=False)
        results.append(result_orig)
        
        if result_orig.get("status") == "ok":
            print(f"    Original: {result_orig.get('rss_peak_mb', 'N/A'):.1f} MB peak, {result_orig.get('elapsed_s', 0):.1f}s")
        else:
            print(f"    Original: {result_orig.get('status', 'unknown')} - {result_orig.get('error', '')}")
        
        # Run checkpoint mode
        result_ckpt = run_benchmark(L, dis_type, method, d, p, use_ckpt=True)
        results.append(result_ckpt)
        
        if result_ckpt.get("status") == "ok":
            print(f"    Ckpt:     {result_ckpt.get('rss_peak_mb', 'N/A'):.1f} MB peak, {result_ckpt.get('elapsed_s', 0):.1f}s")
        else:
            print(f"    Ckpt:     {result_ckpt.get('status', 'unknown')} - {result_ckpt.get('error', '')}")
    
    # Print summary table
    print_results_table(results)
    
    # Save detailed results
    json_output = TEST_DIR / f"benchmark-L{L_min}-{L_max}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    save_results_json(results, json_output)


if __name__ == "__main__":
    main()

