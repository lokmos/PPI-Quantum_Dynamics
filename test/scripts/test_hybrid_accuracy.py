#!/usr/bin/env python3
"""
Test script to compare accuracy between hybrid mode and original mode.
Hybrid mode uses cutoff=1e-6 for high precision.

This script:
1. Runs both original and hybrid modes with specified parameters
2. Compares eigenvalues and calculates errors
3. Generates a comprehensive accuracy report

Usage:
    python test_hybrid_accuracy.py [L_min] [L_max]
    
Example:
    python test_hybrid_accuracy.py 3 6
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
CODE_DIR = REPO_ROOT / "code"
TEST_DIR = REPO_ROOT / "test"

# Parameters for accuracy test
HYBRID_CUTOFF = "1e-6"  # High precision cutoff for hybrid mode
ORIGINAL_CUTOFF = "1e-4"  # Default cutoff for original mode
ODE_TOL = "1e-6"  # High precision ODE tolerance
USE_JIT = True

# Test parameters
DEFAULT_DIS_TYPE = "--step=1000"  # Deterministic disorder for reproducibility
DEFAULT_METHOD = "tensordot"
DEFAULT_D = 1.0
DEFAULT_P = 0

PROCESS_TIMEOUT = 360000

# Get CPU count
def _get_num_cores() -> int:
    try:
        from psutil import cpu_count as _psutil_cpu_count
        n = _psutil_cpu_count(logical=False)
        if n is None:
            n = _psutil_cpu_count(logical=True)
        return int(n) if n else (os.cpu_count() or 1)
    except Exception:
        return int(os.cpu_count() or 1)

_NUM_CORES = str(_get_num_cores())

BASE_ENV = {
    "PYFLOW_MEMLOG": "0",  # Disable memlog for accuracy test
    "PYFLOW_TIMELOG": "0",
    "PYFLOW_SCRAMBLE": "0",
    "PYFLOW_OVERWRITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "XLA_FLAGS": f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={_NUM_CORES} inter_op_parallelism_threads=1",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def run_single_test(L: int, mode: str, cutoff: str, dis_type: str, method: str, 
                    d: float, p: int, test_dir: Path) -> dict:
    """Run a single test for specified mode."""
    
    if mode == "original":
        use_ckpt_val = "0"
    elif mode == "hybrid":
        use_ckpt_val = "hybrid"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["USE_CKPT"] = use_ckpt_val
    env["USE_JIT_FLOW"] = "1" if USE_JIT else "0"
    env["PYFLOW_ODE_RTOL"] = ODE_TOL
    env["PYFLOW_ODE_ATOL"] = ODE_TOL
    env["PYFLOW_CUTOFF"] = str(cutoff)
    # Force overwrite to ensure fresh results
    env["PYFLOW_OVERWRITE"] = "1"
    
    cmd = [
        sys.executable,
        str(CODE_DIR / "main_itc_cpu_d2.py"),
        str(L), dis_type, method, str(d), str(p)
    ]
    
    print(f"  Running L={L}, mode={mode:10s}, cutoff={cutoff} ...", end="", flush=True)
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=CODE_DIR, env=env, 
                              capture_output=True, text=True, 
                              timeout=PROCESS_TIMEOUT)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f" [FAILED] ({elapsed:.1f}s)")
            err_lines = result.stderr.strip().splitlines()[-5:] if result.stderr else ['Unknown Error']
            for line in err_lines:
                print(f"    ! {line}")
            return {"L": L, "mode": mode, "status": "error", "error": "Process failed"}
        
        print(f" [OK] ({elapsed:.1f}s)")
        return {
            "L": L, "mode": mode, "status": "ok", 
            "elapsed_s": elapsed,
        }
    
    except subprocess.TimeoutExpired:
        print(" [TIMEOUT]")
        return {"L": L, "mode": mode, "status": "timeout"}
    except Exception as e:
        print(f" [ERROR] {e}")
        return {"L": L, "mode": mode, "status": "error", "error": str(e)}


def read_eigenvalues_from_h5(L: int, d: float, p: int, dis_type: str) -> dict:
    """Read flow eigenvalues and ED eigenvalues from HDF5 file."""
    try:
        import h5py
    except ImportError:
        print("Error: h5py not installed. Install with: pip install h5py")
        return {"error": "h5py not available"}
    
    dim = 2
    order = 4
    x = 0.0
    delta = 0.1
    n = L ** dim
    
    # Determine potential type folder
    if dis_type.startswith("--"):
        pot_folder = dis_type
    else:
        pot_folder = dis_type
    
    # Construct file path
    h5_path = (CODE_DIR / "pyflow_out" / "fermion" / "d2" / "data" / 
               pot_folder / "PT" / "bck" / f"O{order}" / "static" / 
               f"dataN{n}" / f"tflow-d{d:.2f}-O{order}-x{x:.2f}-Jz{delta:.2f}-p{p}.h5")
    
    if not h5_path.exists():
        return {"error": f"File not found: {h5_path}"}
    
    try:
        with h5py.File(h5_path, 'r') as hf:
            # Read exact diagonalization eigenvalues (many-body spectrum at half-filling)
            if 'ed' not in hf:
                return {"error": "ED results not found in HDF5 (dataset 'ed')"}
            ed_eigs = np.array(hf['ed'][:])

            # Read flow spectrum.
            # In this codebase, the comparable many-body levels are stored as 'flevels' when ED ran.
            if 'flevels' in hf:
                flow_eigs = np.array(hf['flevels'][:])
                flow_source = "flevels"
            elif 'H2_diag' in hf:
                # Fallback: some runs may only store a diagonal object. Try to coerce to 1D levels.
                h2 = np.array(hf['H2_diag'][:])
                if h2.ndim == 1:
                    flow_eigs = h2
                    flow_source = "H2_diag(1d)"
                elif h2.ndim == 2 and h2.shape[0] == h2.shape[1]:
                    # NOTE: This is generally NOT the same object as the many-body ED spectrum,
                    # but if dimensions happen to match, we can still compare.
                    flow_eigs = np.diag(h2)
                    flow_source = "diag(H2_diag)"
                else:
                    return {"error": f"Unsupported H2_diag shape {h2.shape}; expected 1D or square 2D."}
            else:
                return {"error": "No flow spectrum found in HDF5 (expected 'flevels' or 'H2_diag')"}
            
            # Read error list if available
            errlist = None
            if 'err' in hf:
                errlist = np.array(hf['err'][:])
            
            return {
                "flow_eigenvalues": flow_eigs,
                "ed_eigenvalues": ed_eigs,
                "error_list": errlist,
                "file_path": str(h5_path),
                "flow_source": flow_source,
            }
    
    except Exception as e:
        return {"error": f"Failed to read HDF5: {e}"}


def compare_eigenvalues(orig_data: dict, hybrid_data: dict) -> dict:
    """Compare eigenvalues between original and hybrid modes."""
    
    if "error" in orig_data or "error" in hybrid_data:
        return {"error": "Missing data"}
    
    flow_orig = np.array(orig_data["flow_eigenvalues"])
    flow_hybrid = np.array(hybrid_data["flow_eigenvalues"])
    ed_eigs = np.array(orig_data["ed_eigenvalues"])  # should be the same for both runs

    # Coerce to real if needed (numerical noise / complex dtype)
    flow_orig = np.real(flow_orig)
    flow_hybrid = np.real(flow_hybrid)
    ed_eigs = np.real(ed_eigs)

    # Sort spectra to make level-by-level comparison meaningful
    flow_orig = np.sort(flow_orig.reshape(-1))
    flow_hybrid = np.sort(flow_hybrid.reshape(-1))
    ed_eigs = np.sort(ed_eigs.reshape(-1))

    if flow_orig.shape != ed_eigs.shape:
        return {"error": f"Spectrum size mismatch: flow(original)={flow_orig.shape}, ed={ed_eigs.shape}. "
                         f"Hint: use dataset 'flevels' for many-body levels (written when ED runs)."}
    if flow_hybrid.shape != ed_eigs.shape:
        return {"error": f"Spectrum size mismatch: flow(hybrid)={flow_hybrid.shape}, ed={ed_eigs.shape}. "
                         f"Hint: use dataset 'flevels' for many-body levels (written when ED runs)."}
    
    # Handle NaN/Inf robustly: compute errors only on indices where all spectra are finite
    finite_mask = np.isfinite(flow_orig) & np.isfinite(flow_hybrid) & np.isfinite(ed_eigs)
    n_total = int(ed_eigs.size)
    n_finite = int(np.count_nonzero(finite_mask))
    if n_finite == 0:
        return {
            "error": "All compared spectral entries are non-finite (nan/inf).",
            "n_total": n_total,
            "n_finite": n_finite,
            "n_nonfinite_flow_original": int(np.count_nonzero(~np.isfinite(flow_orig))),
            "n_nonfinite_flow_hybrid": int(np.count_nonzero(~np.isfinite(flow_hybrid))),
            "n_nonfinite_ed": int(np.count_nonzero(~np.isfinite(ed_eigs))),
            "flow_source_original": orig_data.get("flow_source", "unknown"),
            "flow_source_hybrid": hybrid_data.get("flow_source", "unknown"),
            # small previews for debugging
            "preview_flow_original": flow_orig[: min(5, n_total)].tolist(),
            "preview_flow_hybrid": flow_hybrid[: min(5, n_total)].tolist(),
            "preview_ed": ed_eigs[: min(5, n_total)].tolist(),
        }

    if n_finite != n_total:
        # Keep going, but surface that we had to drop non-finite entries.
        dropped = n_total - n_finite
    else:
        dropped = 0

    flow_orig_f = flow_orig[finite_mask]
    flow_hybrid_f = flow_hybrid[finite_mask]
    ed_f = ed_eigs[finite_mask]

    # Calculate errors
    # 1. Flow eigenvalues vs ED (absolute errors)
    err_orig_vs_ed = np.abs(flow_orig_f - ed_f)
    err_hybrid_vs_ed = np.abs(flow_hybrid_f - ed_f)
    
    # 2. Original vs Hybrid (consistency check)
    err_orig_vs_hybrid = np.abs(flow_orig_f - flow_hybrid_f)
    
    # 3. Relative errors (avoid division by very small numbers)
    ed_abs = np.abs(ed_f)
    mask = ed_abs > 1e-10
    rel_err_orig = np.zeros_like(err_orig_vs_ed)
    rel_err_hybrid = np.zeros_like(err_hybrid_vs_ed)
    rel_err_orig[mask] = err_orig_vs_ed[mask] / ed_abs[mask]
    rel_err_hybrid[mask] = err_hybrid_vs_ed[mask] / ed_abs[mask]
    
    max_orig = float(np.max(err_orig_vs_ed)) if err_orig_vs_ed.size else float("nan")
    max_hyb = float(np.max(err_hybrid_vs_ed)) if err_hybrid_vs_ed.size else float("nan")
    mean_orig = float(np.mean(err_orig_vs_ed)) if err_orig_vs_ed.size else float("nan")
    mean_hyb = float(np.mean(err_hybrid_vs_ed)) if err_hybrid_vs_ed.size else float("nan")

    return {
        "abs_error_original_vs_ed": err_orig_vs_ed,
        "abs_error_hybrid_vs_ed": err_hybrid_vs_ed,
        "abs_error_original_vs_hybrid": err_orig_vs_hybrid,
        "rel_error_original_vs_ed": rel_err_orig,
        "rel_error_hybrid_vs_ed": rel_err_hybrid,
        "n_total_levels": n_total,
        "n_finite_levels_used": n_finite,
        "n_levels_dropped_nonfinite": dropped,
        "max_abs_error_original": max_orig,
        "max_abs_error_hybrid": max_hyb,
        "mean_abs_error_original": mean_orig,
        "mean_abs_error_hybrid": mean_hyb,
        "max_rel_error_original": float(np.max(rel_err_orig)) if rel_err_orig.size else float("nan"),
        "max_rel_error_hybrid": float(np.max(rel_err_hybrid)) if rel_err_hybrid.size else float("nan"),
        "mean_rel_error_original": float(np.mean(rel_err_orig)) if rel_err_orig.size else float("nan"),
        "mean_rel_error_hybrid": float(np.mean(rel_err_hybrid)) if rel_err_hybrid.size else float("nan"),
        "improvement_factor_max": (max_orig / max_hyb) if (np.isfinite(max_orig) and np.isfinite(max_hyb) and max_hyb > 0) else float("nan"),
        "improvement_factor_mean": (mean_orig / mean_hyb) if (np.isfinite(mean_orig) and np.isfinite(mean_hyb) and mean_hyb > 0) else float("nan"),
    }


def print_accuracy_report(results: dict, L_values: list):
    """Print comprehensive accuracy report."""
    
    print("\n" + "="*100)
    print("ACCURACY TEST REPORT: Hybrid vs Original Mode")
    print(f"Hybrid cutoff: {HYBRID_CUTOFF}, Original cutoff: {ORIGINAL_CUTOFF}, ODE tolerance: {ODE_TOL}")
    print("="*100)
    
    # Table header
    header = (f"{'L':<4} {'n':<5} │ {'Original':^20} │ {'Hybrid':^20} │ {'Improvement':^15}")
    subheader = (f"{'':4} {'':5} │ {'Max Err':>10} {'Mean Err':>10} │ {'Max Err':>10} {'Mean Err':>10} │ {'Max':>7} {'Mean':>7}")
    print(header)
    print(subheader)
    print("─"*100)
    
    for L in L_values:
        if L not in results:
            continue
        
        res = results[L]
        if "error" in res:
            # Print compact but actionable diagnostics if present.
            extra = ""
            if any(k in res for k in ("n_nonfinite_flow_original", "n_nonfinite_flow_hybrid", "n_nonfinite_ed")):
                extra = (f" | nonfinite(orig,hyb,ed)="
                         f"({res.get('n_nonfinite_flow_original','?')},"
                         f"{res.get('n_nonfinite_flow_hybrid','?')},"
                         f"{res.get('n_nonfinite_ed','?')})"
                         f" | src=({res.get('flow_source_original','?')},{res.get('flow_source_hybrid','?')})")
            print(f"{L:<4} {L*L:<5} │ ERROR: {res['error']}{extra}")
            continue
        
        n = L * L
        orig_max = res['max_abs_error_original']
        orig_mean = res['mean_abs_error_original']
        hyb_max = res['max_abs_error_hybrid']
        hyb_mean = res['mean_abs_error_hybrid']
        imp_max = res['improvement_factor_max']
        imp_mean = res['improvement_factor_mean']
        
        print(f"{L:<4} {n:<5} │ {orig_max:>10.2e} {orig_mean:>10.2e} │ "
              f"{hyb_max:>10.2e} {hyb_mean:>10.2e} │ {imp_max:>7.2f}× {imp_mean:>7.2f}×")
    
    print("─"*100)
    
    # Summary statistics
    print("\nSummary Statistics:")
    all_orig_max = [results[L]['max_abs_error_original'] for L in L_values if L in results and 'error' not in results[L]]
    all_hyb_max = [results[L]['max_abs_error_hybrid'] for L in L_values if L in results and 'error' not in results[L]]
    all_imp_max = [results[L]['improvement_factor_max'] for L in L_values if L in results and 'error' not in results[L]]
    
    if all_orig_max and all_hyb_max:
        print(f"  Original mode - Average max error: {np.mean(all_orig_max):.2e}")
        print(f"  Hybrid mode   - Average max error: {np.mean(all_hyb_max):.2e}")
        print(f"  Average improvement factor: {np.mean(all_imp_max):.2f}×")
    
    print("="*100)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Parse command line arguments
    L_min = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    L_max = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    
    dis_type = DEFAULT_DIS_TYPE
    method = DEFAULT_METHOD
    d = DEFAULT_D
    p = DEFAULT_P
    
    # Create test directory
    test_dir = TEST_DIR / "accuracy_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("HYBRID MODE ACCURACY TEST")
    print(f"Testing system sizes: L ∈ [{L_min}, {L_max}]")
    print(f"Hybrid cutoff: {HYBRID_CUTOFF}, Original cutoff: {ORIGINAL_CUTOFF}")
    print(f"ODE tolerance: {ODE_TOL}, JIT: {'ON' if USE_JIT else 'OFF'}")
    print("="*100)
    
    L_values = list(range(L_min, L_max + 1))
    
    # Run tests and collect data immediately
    start_time = time.time()
    results = {}
    
    for L in L_values:
        print(f"\nTesting L={L} (n={L*L}):")
        
        # Run original mode
        result_orig = run_single_test(L, "original", ORIGINAL_CUTOFF, dis_type, method, d, p, test_dir)
        
        if result_orig['status'] != 'ok':
            results[L] = {"error": f"Original mode failed: {result_orig.get('error', 'Unknown')}"}
            continue
        
        # Read original mode results immediately
        print(f"  Reading original mode results...", end="", flush=True)
        orig_data = read_eigenvalues_from_h5(L, d, p, dis_type)
        
        if "error" in orig_data:
            print(f" [ERROR: {orig_data['error']}]")
            results[L] = {"error": orig_data['error']}
            continue
        
        # Save original data temporarily
        orig_flow_eigs = orig_data["flow_eigenvalues"].copy()
        orig_ed_eigs = orig_data["ed_eigenvalues"].copy()
        orig_flow_source = orig_data.get("flow_source", "unknown")
        print(" [OK]")
        
        # Run hybrid mode (will overwrite the HDF5 file)
        result_hybrid = run_single_test(L, "hybrid", HYBRID_CUTOFF, dis_type, method, d, p, test_dir)
        
        if result_hybrid['status'] != 'ok':
            results[L] = {"error": f"Hybrid mode failed: {result_hybrid.get('error', 'Unknown')}"}
            continue
        
        # Read hybrid mode results immediately
        print(f"  Reading hybrid mode results...", end="", flush=True)
        hybrid_data = read_eigenvalues_from_h5(L, d, p, dis_type)
        
        if "error" in hybrid_data:
            print(f" [ERROR: {hybrid_data['error']}]")
            results[L] = {"error": hybrid_data['error']}
            continue
        
        print(" [OK]")
        
        # Reconstruct orig_data with saved eigenvalues
        orig_data_reconstructed = {
            "flow_eigenvalues": orig_flow_eigs,
            "ed_eigenvalues": orig_ed_eigs,
            "flow_source": orig_flow_source,
        }
        
        # Compare eigenvalues
        print(f"  Comparing results...", end="", flush=True)
        comparison = compare_eigenvalues(orig_data_reconstructed, hybrid_data)
        
        if "error" in comparison:
            print(f" [ERROR: {comparison['error']}]")
            results[L] = comparison
        else:
            print(" [OK]")
            results[L] = comparison
    
    total_time = time.time() - start_time
    print(f"\nAll tests completed in {total_time:.1f}s")
    
    # Print report
    print_accuracy_report(results, L_values)
    
    # Save detailed results to JSON
    output_file = test_dir / "accuracy_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for L, res in results.items():
            json_results[L] = {}
            for key, val in res.items():
                if isinstance(val, np.ndarray):
                    json_results[L][key] = val.tolist()
                elif isinstance(val, (np.float64, np.float32)):
                    json_results[L][key] = float(val)
                else:
                    json_results[L][key] = val
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

