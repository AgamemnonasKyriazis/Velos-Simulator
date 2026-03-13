import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path


_CLI_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "0.33", "0.33", "0.33"]

import pygame_simulator as sim


DUMP_TOP_CANDIDATE_QUEUES = True
DUMP_BASE_DIR = Path.home() / "tmp" / "ratio_candidates"


def frange(start, stop, step):
    vals = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 4))
        x += step
    return vals


def eval_triplet(f_ratio, e_ratio, b_ratio, n_jobs=None):
    arch = sim.build_default_architecture()
    use_n_jobs = n_jobs if n_jobs is not None and n_jobs > 0 else sim.N_JOBS
    workload = sim.Workload.from_realistic(
        use_n_jobs,
        sim.WORKLOAD_SEED,
        f_ratio,
        e_ratio,
        b_ratio,
        sim.MIN_JOB_CYCLES,
        sim.MAX_JOB_CYCLES,
    )

    _, mono = sim.run_to_completion("monolithic", architecture=arch, workload=workload)
    _, tiled = sim.run_to_completion("tiled", architecture=arch, workload=workload)
    _, stateful = sim.run_to_completion("stateful", architecture=arch, workload=workload)

    if mono is None or tiled is None or stateful is None:
        raise RuntimeError("Simulation did not produce metrics for one of the modes")

    mono_mk     = mono["makespan"]
    tiled_mk    = tiled["makespan"]
    stateful_mk = stateful["makespan"]

    mono_p95     = mono["p95_turnaround"]
    tiled_p95    = tiled["p95_turnaround"]
    stateful_p95 = stateful["p95_turnaround"]

    tiled_impr    = (mono_p95 - tiled_p95) / mono_p95    # (mono_mk - tiled_mk) / mono_mk
    stateful_impr = (mono_p95 - stateful_p95) / mono_p95 # (mono_mk - stateful_mk) / mono_mk

    objective     = (stateful_impr - tiled_impr)

    return {
        "f": f_ratio,
        "e": e_ratio,
        "b": b_ratio,
        "mono": mono_mk,
        "tiled": tiled_mk,
        "stateful": stateful_mk,
        "tiled_impr": tiled_impr,
        "stateful_impr": stateful_impr,
        "objective": objective,
        "direct_gain_vs_tiled": (tiled_mk - stateful_mk) / tiled_mk,
        "defrag_successes": stateful["defrag_successes"],
        "task_queue": workload.task_queue,
    }


def eval_triplet_worker(payload):
    f_ratio, e_ratio, b_ratio, n_jobs = payload
    sim.EXPORT_ON_FINISH = False
    return eval_triplet(f_ratio, e_ratio, b_ratio, n_jobs=n_jobs)


def _run_payloads(payloads, workers):
    if not payloads:
        return []

    if workers <= 1:
        return [eval_triplet_worker(p) for p in payloads]

    chunk = max(1, len(payloads) // max(1, workers * 4))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        return pool.map(eval_triplet_worker, payloads, chunksize=chunk)


def grid_search(step, f_min, f_max, e_min, e_max, b_min, workers, n_jobs):
    payloads = []
    for f in frange(f_min, f_max, step):
        for e in frange(e_min, e_max, step):
            b = round(1.0 - f - e, 4)
            if b < b_min or b > 1.0:
                continue
            payloads.append((f, e, b, n_jobs))

    results = _run_payloads(payloads, workers)
    results.sort(key=lambda x: x["objective"], reverse=True)
    return results


def refine(best, step, radius, b_min, workers, n_jobs):
    payloads = []
    f0, e0 = best["f"], best["e"]
    f_vals = frange(max(0.0, f0 - radius), min(1.0, f0 + radius), step)
    e_vals = frange(max(0.0, e0 - radius), min(1.0, e0 + radius), step)
    for f in f_vals:
        for e in e_vals:
            b = round(1.0 - f - e, 4)
            if b < b_min or b > 1.0:
                continue
            payloads.append((f, e, b, n_jobs))

    results = _run_payloads(payloads, workers)
    results.sort(key=lambda x: x["objective"], reverse=True)
    return results


def pct(x):
    return f"{100.0 * x:.2f}%"


def _jsonable_queue(queue):
    return [[list(shape), int(runtime)] for shape, runtime in queue]


def dump_top_candidates(candidates, n_jobs):
    if not DUMP_TOP_CANDIDATE_QUEUES or not candidates:
        return None

    run_dir = DUMP_BASE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for idx, row in enumerate(candidates, start=1):
        use_n_jobs = n_jobs if n_jobs is not None and n_jobs > 0 else sim.N_JOBS
        queue = list(row.get("task_queue", []))
        if not queue:
            print("ERROR: missing evaluated task_queue for top candidate dump. Aborting run.")
            sys.exit(1)

        arch = sim.build_default_architecture()

        meta = {
            "rank": idx,
            "ratios": {"fragment": row["f"], "elephant": row["e"], "background": row["b"]},
            "objective": row["objective"],
            "direct_gain_vs_tiled": row["direct_gain_vs_tiled"],
            "makespan": {
                "monolithic": row["mono"],
                "tiled": row["tiled"],
                "stateful": row["stateful"],
            },
            "improvement_vs_monolithic": {
                "tiled": row["tiled_impr"],
                "stateful": row["stateful_impr"],
            },
            "sim_config": {
                "workload_seed": sim.WORKLOAD_SEED,
                "workload_profile": "realistic",
                "n_jobs": use_n_jobs,
                "min_job_cycles": sim.MIN_JOB_CYCLES,
                "max_job_cycles": sim.MAX_JOB_CYCLES,
                "grid_h": arch.grid_h,
                "grid_w": arch.grid_w,
                "host_enqueue_latency": arch.host_enqueue_latency,
                "config_cost_per_area": arch.config_cost_per_area,
                "stateless_cost_per_area": arch.stateless_cost_per_area,
                "stateful_cost_per_area": arch.stateful_cost_per_area,
                "trigger_multiplier": arch.defrag_trigger_free_area_multiplier,
                "smart_threshold": arch.smart_stateless_completion_threshold,
                "non_monolithic_bw_factor": arch.non_monolithic_bw_factor,
            },
        }

        queue_payload = {
            "rank": idx,
            "ratios": meta["ratios"],
            "task_queue": _jsonable_queue(queue),
        }

        meta_path = run_dir / f"rank_{idx:02d}_meta.json"
        queue_path = run_dir / f"rank_{idx:02d}_task_queue.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        queue_path.write_text(json.dumps(queue_payload, indent=2))

        summary_rows.append({
            "rank": idx,
            "f": row["f"],
            "e": row["e"],
            "b": row["b"],
            "objective": row["objective"],
            "direct_gain_vs_tiled": row["direct_gain_vs_tiled"],
            "mono": row["mono"],
            "tiled": row["tiled"],
            "stateful": row["stateful"],
            "meta_file": str(meta_path),
            "task_queue_file": str(queue_path),
        })

    summary_path = run_dir / "summary_top_candidates.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Optimize workload ratios for maximum (stateful% - tiled%) improvement.")
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--f-min", type=float, default=0.35)
    parser.add_argument("--f-max", type=float, default=0.70)
    parser.add_argument("--e-min", type=float, default=0.10)
    parser.add_argument("--e-max", type=float, default=0.40)
    parser.add_argument("--b-min", type=float, default=0.10)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(_CLI_ARGS)

    sim.EXPORT_ON_FINISH = False

    coarse = grid_search(
        args.coarse_step,
        args.f_min,
        args.f_max,
        args.e_min,
        args.e_max,
        args.b_min,
        args.workers,
        args.n_jobs,
    )
    if not coarse:
        print("No valid coarse candidates.")
        return

    best_coarse = coarse[0]
    fine = refine(best_coarse, args.fine_step, args.radius, args.b_min, args.workers, args.n_jobs)
    best = fine[0] if fine else best_coarse

    print("=== BEST RATIOS ===")
    print(f"F={best['f']:.4f}, E={best['e']:.4f}, B={best['b']:.4f}")
    print(f"Monolithic makespan: {best['mono']}")
    print(f"Tiled makespan:      {best['tiled']}")
    print(f"Stateful makespan:   {best['stateful']}")
    print(f"Tiled improvement:   {pct(best['tiled_impr'])}")
    print(f"Stateful improvement:{pct(best['stateful_impr'])}")
    print(f"Objective (S-T):     {pct(best['objective'])}")
    print(f"Direct gain vs tiled:{pct(best['direct_gain_vs_tiled'])}")

    print("\n=== TOP CANDIDATES (fine) ===")
    top_rows = fine[: args.top] if fine else coarse[: args.top]
    for row in top_rows:
        print(
            f"F={row['f']:.3f} E={row['e']:.3f} B={row['b']:.3f} | "
            f"obj={pct(row['objective'])} | "
            f"gain_vs_tiled={pct(row['direct_gain_vs_tiled'])} | "
            f"mk(m/t/s)=({row['mono']}/{row['tiled']}/{row['stateful']})"
        )

    dumped_dir = dump_top_candidates(top_rows, args.n_jobs)
    if dumped_dir is not None:
        print(f"\nDumped top candidate queues to: {dumped_dir}")


if __name__ == "__main__":
    main()
