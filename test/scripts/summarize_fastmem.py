#!/usr/bin/env python3
"""
Summarize fastmem runs into test/fast/summary.csv (+ summary.jsonl).

Reads memlog JSONL under test/fast/<mode>/memlog-*-fastmem.jsonl and produces
per-(L,mode) measured peak RSS and an extrapolated "full-flow" peak estimate:

- original: scale storage linearly from allocated store_steps -> T_total
- ckpt: same for checkpoint storage; add one segment buffer + transient
- recursive/hybrid: bck peak determined by recursion depth (binary split)

This is intentionally conservative: we add components rather than assuming
peaks do not overlap.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
FAST_DIR = REPO_ROOT / "test" / "fast"
MODES = ["original", "ckpt", "recursive", "hybrid"]


def parse_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def extract_points(rows: list[dict]) -> Dict[str, Any]:
    baseline = None
    baseline_idx = 0
    for i, r in enumerate(rows):
        if r.get("tag") == "main:before_flow":
            baseline = r.get("rss_mb")
            baseline_idx = i
            break
    flow_rows = rows[baseline_idx:]
    rss_vals = [r.get("rss_mb") for r in flow_rows if "rss_mb" in r]
    peak = max(rss_vals) if rss_vals else None

    pts = {}
    for r in flow_rows:
        tag = r.get("tag")
        if tag in ("fast:before_storage", "fast:after_storage_k", "fast:after_transient_step", "fast:after_bck_alloc", "fast:after_bck_compute", "fast:after_flow"):
            pts[tag] = r

    # Meta parameters:
    # - Prefer fast:before_storage fields if present
    # - Fall back to main:before_flow (it always has L/n/dim)
    meta = dict(pts.get("fast:before_storage", {}) or {})
    if baseline_idx < len(rows):
        meta.setdefault("L", rows[baseline_idx].get("L"))
        meta.setdefault("n", rows[baseline_idx].get("n"))
        meta.setdefault("dim", rows[baseline_idx].get("dim"))
    # Fallbacks (rare)
    L = meta.get("L") or (rows[0].get("L") if rows else None)
    mode = meta.get("mode") or (rows[0].get("mode") if rows else None)

    return {
        "baseline_mb": baseline,
        "peak_mb": peak,
        "net_mb": (peak - baseline) if (peak is not None and baseline is not None) else None,
        "L": L,
        "mode": mode,
        "pts": pts,
        "meta": meta,
    }


def safe_get_rss(pts: Dict[str, dict], tag: str) -> Optional[float]:
    r = pts.get(tag)
    if not r:
        return None
    return r.get("rss_mb")


def est_full_peak(info: Dict[str, Any]) -> Dict[str, Any]:
    pts = info["pts"]
    meta = info["meta"] or {}

    baseline = info["baseline_mb"]
    peak = info["peak_mb"]
    mode = meta.get("mode") or info.get("mode")
    L = meta.get("L") or info.get("L")
    n = meta.get("n")

    T_total = int(meta.get("T_total") or 0) or None
    store_steps = int(meta.get("store_steps") or 0) or None
    ckpt_step = int(meta.get("ckpt_step") or 0) or None
    bck_steps = int(meta.get("bck_steps") or meta.get("base_case_steps") or 0) or None

    rss_before_storage = safe_get_rss(pts, "fast:before_storage") or baseline
    rss_after_storage = safe_get_rss(pts, "fast:after_storage_k") or rss_before_storage
    rss_after_bck = safe_get_rss(pts, "fast:after_bck_alloc") or rss_after_storage

    # Prefer theoretical allocated bytes for storage/bck buffers (RSS deltas can be 0 for small L).
    # This keeps the extrapolation stable even at L=3/4.
    storage_delta_k = None
    bck_delta = None

    if n is not None:
        n = int(n)
        n2 = n * n
        n4 = n2 * n2

        if mode == "original" and store_steps:
            storage_delta_k = ((n2 + n4) * 4.0 * float(store_steps)) / 1e6  # MB
        elif mode == "ckpt" and store_steps and ckpt_step:
            num_ckpts_k = int(math.ceil(float(store_steps) / float(ckpt_step))) + 1
            storage_delta_k = ((n2 + n4) * 4.0 * float(num_ckpts_k)) / 1e6

        if mode in ("ckpt", "recursive") and bck_steps:
            bck_delta = ((n2 + n4) * 4.0 * float(bck_steps)) / 1e6
        elif mode == "hybrid" and bck_steps:
            # hybrid buffer stores fp16 copies
            bck_delta = ((n2 + n4) * 2.0 * float(bck_steps)) / 1e6

    # Fallback to RSS deltas if theory couldn't be computed
    if storage_delta_k is None:
        storage_delta_k = max(0.0, (rss_after_storage or 0.0) - (rss_before_storage or 0.0))
    if bck_delta is None:
        bck_delta = max(0.0, (rss_after_bck or 0.0) - (rss_after_storage or 0.0))

    # transient peak beyond baseline + (short-run storage + bck buffer)
    transient_delta = 0.0
    if peak is not None and baseline is not None:
        ref = baseline + float(storage_delta_k) + float(bck_delta)
        transient_delta = max(0.0, float(peak) - ref)

    # Extrapolate storage to full T_total
    storage_full = 0.0
    if mode == "original" and T_total and store_steps and storage_delta_k is not None:
        storage_full = float(storage_delta_k) * (float(T_total) / float(store_steps))
    elif mode == "ckpt" and T_total and store_steps and ckpt_step and storage_delta_k is not None:
        num_ckpts_full = int(math.ceil(float(T_total) / float(ckpt_step))) + 1
        num_ckpts_k = int(math.ceil(float(store_steps) / float(ckpt_step))) + 1
        storage_full = float(storage_delta_k) * (float(num_ckpts_full) / float(num_ckpts_k))

    # Recursive/hybrid bck peak via binary recursion depth
    depth = None
    stack_mb = 0.0
    if mode in ("recursive", "hybrid") and T_total and bck_steps and n is not None:
        # depth = ceil(log2(T/B))
        depth = int(math.ceil(math.log(max(1.0, float(T_total) / float(bck_steps)), 2.0)))
        # One state ~ H2 + H4 in float32 (compute dtype)
        n = int(n)
        n2 = n * n
        n4 = n2 * n2
        state_bytes = (n2 + n4) * 4  # float32
        stack_mb = (state_bytes / 1e6) * float(depth)

    full_peak_est = None
    if baseline is not None:
        full_peak_est = float(baseline) + float(storage_full) + float(bck_delta) + float(stack_mb) + float(transient_delta)

    return {
        "T_total": T_total,
        "store_steps": store_steps,
        "ckpt_step": ckpt_step,
        "bck_steps": bck_steps,
        "storage_delta_k_mb": storage_delta_k,
        "storage_full_est_mb": storage_full,
        "bck_delta_mb": bck_delta,
        "recursive_depth": depth,
        "recursive_stack_est_mb": stack_mb,
        "transient_delta_mb": transient_delta,
        "rss_peak_full_est_mb": full_peak_est,
        "L": L,
        "mode": mode,
        "n": n,
    }


def main() -> None:
    out_csv = FAST_DIR / "summary.csv"
    out_jsonl = FAST_DIR / "summary.jsonl"
    FAST_DIR.mkdir(parents=True, exist_ok=True)

    # CLI:
    # - default: rebuild summary from all memlogs found
    # - --append --memlog <path>: append one record (no de-dupe; caller controls fresh runs)
    args = sys.argv[1:]
    append = ("--append" in args)
    memlog_path = None
    if "--memlog" in args:
        i = args.index("--memlog")
        if i + 1 < len(args):
            memlog_path = Path(args[i + 1])
            if not memlog_path.is_absolute():
                memlog_path = (REPO_ROOT / memlog_path).resolve()

    rows_out = []
    if append and memlog_path is not None:
        p = memlog_path
        if not p.exists():
            return
        rel = str(p.relative_to(REPO_ROOT)) if str(p).startswith(str(REPO_ROOT)) else str(p)
        rows = parse_jsonl(p)
        if not rows:
            return
        info = extract_points(rows)
        est = est_full_peak(info)
        rows_out.append({
            "path": rel,
            "L": est.get("L"),
            "mode": est.get("mode"),
            "n": est.get("n"),
            "baseline_mb": info.get("baseline_mb"),
            "peak_mb": info.get("peak_mb"),
            "net_mb": info.get("net_mb"),
            **est,
        })
    else:
        for mode in MODES:
            mode_dir = FAST_DIR / mode
            if not mode_dir.exists():
                continue
            for p in sorted(mode_dir.glob("memlog-*-fastmem.jsonl")):
                rows = parse_jsonl(p)
                if not rows:
                    continue
                info = extract_points(rows)
                est = est_full_peak(info)
                rec = {
                    "path": str(p.relative_to(REPO_ROOT)),
                    "L": est.get("L"),
                    "mode": est.get("mode") or mode,
                    "n": est.get("n"),
                    "baseline_mb": info.get("baseline_mb"),
                    "peak_mb": info.get("peak_mb"),
                    "net_mb": info.get("net_mb"),
                    **est,
                }
                rows_out.append(rec)

    # Write CSV
    fieldnames = [
        "L", "mode", "n",
        "baseline_mb", "peak_mb", "net_mb",
        "T_total", "store_steps", "ckpt_step", "bck_steps",
        "storage_delta_k_mb", "storage_full_est_mb",
        "bck_delta_mb", "recursive_depth", "recursive_stack_est_mb",
        "transient_delta_mb",
        "rss_peak_full_est_mb",
        "path",
    ]
    if append:
        # append mode: create header if needed
        file_exists = out_csv.exists()
        with out_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            for r in rows_out:
                w.writerow({k: r.get(k) for k in fieldnames})
    else:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in sorted(rows_out, key=lambda x: (int(x.get("L") or 999), str(x.get("mode") or ""))):
                w.writerow({k: r.get(k) for k in fieldnames})

    # Write JSONL
    if append:
        with out_jsonl.open("a") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with out_jsonl.open("w") as f:
            for r in sorted(rows_out, key=lambda x: (int(x.get("L") or 999), str(x.get("mode") or ""))):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not append:
        print(f"Saved: {out_csv}")
        print(f"Saved: {out_jsonl}")


if __name__ == "__main__":
    main()


