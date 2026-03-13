import sys
from collections import deque
from dataclasses import dataclass, field
import math
import json
import os
import random
from typing import Optional
import pygame


GRID_H = 4
GRID_W = 4
CELL = 44
MARGIN = 24
SIDEBAR_W = 480
FPS = 60

MIN_WINDOW_W = 1200
MIN_WINDOW_H = 700

MODE = "monolithic"  # monolithic | tiled | stateless | smart_stateless | stateful
AUTO_RUN = True
RUN_NAME = "sim_run"
RESULTS_CSV_PATH = "/home/akyriazis/work/PolyBench-CGRA/sim_metrics.csv"
EXPORT_ON_FINISH = True
CLEAN_RESULTS_ON_BOOT = True
HOST_ENQUEUE_LATENCY = 100

CONFIG_COST_PER_AREA = 3000
STATELESS_COST_PER_AREA = 0
STATEFUL_COST_PER_AREA = int(CONFIG_COST_PER_AREA + 30)
DEFRAG_TRIGGER_FREE_AREA_MULTIPLIER = 2
SMART_STATELESS_COMPLETION_THRESHOLD = 0.50
NON_MONOLITHIC_BW_FACTOR = 3.0


N_JOBS = 32
MIN_JOB_CYCLES = 465
MAX_JOB_CYCLES = 1924740
MIN_JOB_H      = 1
MAX_JOB_H      = 2
MIN_JOB_W      = 1
MAX_JOB_W      = 2

WORKLOAD_PROFILE = "load_from_file"  # stateful_vs_tiled | uniform_random | realistic | load_from_file
WORKLOAD_SEED = 2337
TASK_QUEUE_FILE = "/home/akyriazis/tmp/ratio_candidates/20260311_202124/rank_01_task_queue.json"

DEFAULT_FRAGMENT_RATIO = 0.45
DEFAULT_ELEPHANT_RATIO = 0.20
DEFAULT_BACKGROUND_RATIO = 0.35


def _read_ratio_arg(index, default):
    try:
        return float(sys.argv[index])
    except (IndexError, ValueError):
        return default


FRAGMENT_RATIO = _read_ratio_arg(1, DEFAULT_FRAGMENT_RATIO)
ELEPHANT_RATIO = _read_ratio_arg(2, DEFAULT_ELEPHANT_RATIO)
BACKGROUND_RATIO = _read_ratio_arg(3, DEFAULT_BACKGROUND_RATIO)

_ratio_sum = FRAGMENT_RATIO + ELEPHANT_RATIO + BACKGROUND_RATIO
if _ratio_sum <= 0:
    FRAGMENT_RATIO = DEFAULT_FRAGMENT_RATIO
    ELEPHANT_RATIO = DEFAULT_ELEPHANT_RATIO
    BACKGROUND_RATIO = DEFAULT_BACKGROUND_RATIO
else:
    FRAGMENT_RATIO /= _ratio_sum
    ELEPHANT_RATIO /= _ratio_sum
    BACKGROUND_RATIO /= _ratio_sum


@dataclass(frozen=True)
class ArchitectureModel:
    grid_h: int
    grid_w: int
    host_enqueue_latency: int
    config_cost_per_area: int
    stateless_cost_per_area: int
    stateful_cost_per_area: int
    defrag_trigger_free_area_multiplier: float
    smart_stateless_completion_threshold: float
    non_monolithic_bw_factor: float

    def validate(self):
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.host_enqueue_latency < 0:
            raise ValueError("Host enqueue latency must be >= 0")
        if self.config_cost_per_area < 0 or self.stateless_cost_per_area < 0 or self.stateful_cost_per_area < 0:
            raise ValueError("Cost parameters must be >= 0")
        if self.defrag_trigger_free_area_multiplier < 0:
            raise ValueError("Defrag trigger multiplier must be >= 0")
        if not (0.0 <= self.smart_stateless_completion_threshold <= 1.0):
            raise ValueError("Smart stateless threshold must be in [0,1]")
        if self.non_monolithic_bw_factor <= 0:
            raise ValueError("Non-monolithic BW factor must be > 0")


@dataclass(frozen=True)
class Workload:
    task_queue: tuple[tuple[tuple[int, int], int], ...]
    profile: str
    seed: Optional[int] = None
    fragment_ratio: Optional[float] = None
    elephant_ratio: Optional[float] = None
    background_ratio: Optional[float] = None
    source_file: Optional[str] = None

    @staticmethod
    def from_stateful_vs_tiled(n_jobs, seed_value, fragment_ratio, elephant_ratio, background_ratio, min_job_cycles, max_job_cycles, min_job_h, max_job_h, min_job_w, max_job_w):
        queue = _generate_stateful_vs_tiled_queue(
            n_jobs,
            seed_value,
            fragment_ratio,
            elephant_ratio,
            background_ratio,
            min_job_cycles,
            max_job_cycles,
            min_job_h,
            max_job_h,
            min_job_w,
            max_job_w,
        )
        return Workload(task_queue=tuple(queue), profile="stateful_vs_tiled", seed=seed_value, fragment_ratio=fragment_ratio, elephant_ratio=elephant_ratio, background_ratio=background_ratio)

    @staticmethod
    def from_uniform_random(n_jobs, seed_value, min_job_cycles, max_job_cycles, min_job_h, max_job_h, min_job_w, max_job_w):
        queue = _generate_uniform_random_queue(
            n_jobs,
            seed_value,
            min_job_cycles,
            max_job_cycles,
            min_job_h,
            max_job_h,
            min_job_w,
            max_job_w,
        )
        return Workload(task_queue=tuple(queue), profile="uniform_random", seed=seed_value)

    @staticmethod
    def from_realistic(n_jobs, seed_value, fragment_ratio, elephant_ratio, background_ratio, min_job_cycles, max_job_cycles):
        queue = _generate_realistic_queue(n_jobs, seed_value, fragment_ratio, elephant_ratio, background_ratio, min_job_cycles, max_job_cycles)
        return Workload(task_queue=tuple(queue), profile="realistic", seed=seed_value, fragment_ratio=fragment_ratio, elephant_ratio=elephant_ratio, background_ratio=background_ratio)

    @staticmethod
    def from_file(path):
        queue = _generate_load_from_file_workload(path)
        return Workload(task_queue=tuple(queue), profile="load_from_file", source_file=path)

    def validate_for_architecture(self, arch: ArchitectureModel):
        for idx, (shape, runtime) in enumerate(self.task_queue):
            h, w = shape
            if h <= 0 or w <= 0 or runtime <= 0:
                raise ValueError(f"Invalid task at index {idx}: non-positive shape/runtime")
            if h > arch.grid_h or w > arch.grid_w:
                raise ValueError(f"Invalid task at index {idx}: shape ({h},{w}) exceeds architecture ({arch.grid_h},{arch.grid_w})")


def _choose_shape(rng, candidates):
    return candidates[rng.randrange(len(candidates))]


def _in_bounds(shape, min_job_h, max_job_h, min_job_w, max_job_w):
    return (
        min_job_h <= shape[0] <= max_job_h
        and min_job_w <= shape[1] <= max_job_w
    )


def _generate_stateful_vs_tiled_queue(n_jobs, seed_value, fragment_ratio, elephant_ratio, background_ratio, min_job_cycles, max_job_cycles, min_job_h, max_job_h, min_job_w, max_job_w):
    rng = random.Random(seed_value)

    fragment_shapes = [s for s in [(1, 2), (2, 1), (1, 3), (3, 1)] if _in_bounds(s, min_job_h, max_job_h, min_job_w, max_job_w)]
    if not fragment_shapes:
        fragment_shapes = [
            (min_job_h, max_job_w),
            (max_job_h, min_job_w),
        ]

    background_shapes = [
        (h, w)
        for h in range(min_job_h, max_job_h + 1)
        for w in range(min_job_w, max_job_w + 1)
    ]

    elephant_h = max(2, max_job_h)
    elephant_w = max(2, max_job_w)
    elephant_candidates = [(elephant_h, elephant_w)]
    if elephant_h > 2 and _in_bounds((elephant_h - 1, elephant_w), min_job_h, max_job_h, min_job_w, max_job_w):
        elephant_candidates.append((elephant_h - 1, elephant_w))
    if elephant_w > 2 and _in_bounds((elephant_h, elephant_w - 1), min_job_h, max_job_h, min_job_w, max_job_w):
        elephant_candidates.append((elephant_h, elephant_w - 1))

    n_frag = int(round(n_jobs * fragment_ratio))
    n_ele = int(round(n_jobs * elephant_ratio))
    if n_ele == 0 and n_jobs >= 8:
        n_ele = 1
    n_bg = max(0, n_jobs - n_frag - n_ele)
    while n_frag + n_ele + n_bg > n_jobs:
        n_bg = max(0, n_bg - 1)

    frag_rt_lo = max(min_job_cycles, int(0.45 * max_job_cycles))
    frag_rt_hi = max(frag_rt_lo, int(0.75 * max_job_cycles))
    bg_rt_lo = min_job_cycles
    bg_rt_hi = max(bg_rt_lo, int(0.40 * max_job_cycles))
    ele_rt_lo = max(min_job_cycles, int(0.85 * max_job_cycles))
    ele_rt_hi = max(ele_rt_lo, int(1.10 * max_job_cycles))

    frag = [(_choose_shape(rng, fragment_shapes), rng.randint(frag_rt_lo, frag_rt_hi)) for _ in range(n_frag)]
    bg = [(_choose_shape(rng, background_shapes), rng.randint(bg_rt_lo, bg_rt_hi)) for _ in range(n_bg)]
    ele = [(_choose_shape(rng, elephant_candidates), rng.randint(ele_rt_lo, ele_rt_hi)) for _ in range(n_ele)]

    rng.shuffle(frag)
    rng.shuffle(bg)
    rng.shuffle(ele)

    queue = []
    front_frag = int(0.65 * len(frag))
    queue.extend(frag[:front_frag])
    frag_rem = frag[front_frag:]

    if ele:
        queue.append(ele.pop(0))

    mixed = frag_rem + bg
    rng.shuffle(mixed)

    inject_stride = 4
    since_last_ele = 0
    for item in mixed:
        queue.append(item)
        since_last_ele += 1
        if ele and since_last_ele >= inject_stride:
            queue.append(ele.pop(0))
            since_last_ele = 0

    queue.extend(ele)
    return queue[:n_jobs]


def _generate_uniform_random_queue(n_jobs, seed_value, min_job_cycles, max_job_cycles, min_job_h, max_job_h, min_job_w, max_job_w):
    rng = random.Random(seed_value)
    queue = []
    for _ in range(n_jobs):
        shape = (
            rng.randint(min_job_h, max_job_h),
            rng.randint(min_job_w, max_job_w),
        )
        runtime = rng.randint(min_job_cycles, max_job_cycles)
        queue.append((shape, runtime))
    return queue


def _generate_realistic_queue(n_jobs, seed_value, fragment_ratio, elephant_ratio, background_ratio, min_job_cycles, max_job_cycles):
    rng = random.Random(seed_value)
    queue = rng.choices(
        [
            ((1,1), min_job_cycles),
            ((2,1), max_job_cycles),
            ((1,2), max_job_cycles),
        ], 
        weights=(fragment_ratio, elephant_ratio, background_ratio),
        k=n_jobs
    )

    return queue


def _generate_load_from_file_workload(file_path):
    if not file_path:
        raise ValueError("TASK_QUEUE_FILE is empty while WORKLOAD_PROFILE='load_from_file'")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TASK_QUEUE_FILE not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    queue_raw = payload.get("task_queue") if isinstance(payload, dict) else payload
    if not isinstance(queue_raw, list):
        raise ValueError("Invalid task queue file format: expected a list or an object with 'task_queue'")

    queue = []
    for idx, item in enumerate(queue_raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid task entry at index {idx}: expected [shape, runtime]")

        shape_raw, runtime_raw = item
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 2:
            raise ValueError(f"Invalid shape at index {idx}: expected [h, w]")

        h = int(shape_raw[0])
        w = int(shape_raw[1])
        runtime = int(runtime_raw)

        if h <= 0 or w <= 0 or runtime <= 0:
            raise ValueError(f"Non-positive shape/runtime at index {idx}")
        queue.append(((h, w), runtime))

    return queue

_RESULTS_CLEANED = False


def maybe_clean_results_on_boot():
    global _RESULTS_CLEANED
    if _RESULTS_CLEANED or not CLEAN_RESULTS_ON_BOOT:
        return

    if RESULTS_CSV_PATH and os.path.exists(RESULTS_CSV_PATH):
        os.remove(RESULTS_CSV_PATH)

    _RESULTS_CLEANED = True


def build_default_architecture() -> ArchitectureModel:
    arch = ArchitectureModel(
        grid_h=GRID_H,
        grid_w=GRID_W,
        host_enqueue_latency=HOST_ENQUEUE_LATENCY,
        config_cost_per_area=CONFIG_COST_PER_AREA,
        stateless_cost_per_area=STATELESS_COST_PER_AREA,
        stateful_cost_per_area=STATEFUL_COST_PER_AREA,
        defrag_trigger_free_area_multiplier=DEFRAG_TRIGGER_FREE_AREA_MULTIPLIER,
        smart_stateless_completion_threshold=SMART_STATELESS_COMPLETION_THRESHOLD,
        non_monolithic_bw_factor=NON_MONOLITHIC_BW_FACTOR,
    )
    arch.validate()
    return arch


def build_default_workload() -> Workload:
    if WORKLOAD_PROFILE == "stateful_vs_tiled":
        return Workload.from_stateful_vs_tiled(
            N_JOBS,
            WORKLOAD_SEED,
            FRAGMENT_RATIO,
            ELEPHANT_RATIO,
            BACKGROUND_RATIO,
            MIN_JOB_CYCLES,
            MAX_JOB_CYCLES,
            MIN_JOB_H,
            MAX_JOB_H,
            MIN_JOB_W,
            MAX_JOB_W,
        )
    if WORKLOAD_PROFILE == "uniform_random":
        return Workload.from_uniform_random(
            N_JOBS,
            WORKLOAD_SEED,
            MIN_JOB_CYCLES,
            MAX_JOB_CYCLES,
            MIN_JOB_H,
            MAX_JOB_H,
            MIN_JOB_W,
            MAX_JOB_W,
        )
    if WORKLOAD_PROFILE == "realistic":
        return Workload.from_realistic(
            N_JOBS,
            WORKLOAD_SEED,
            FRAGMENT_RATIO,
            ELEPHANT_RATIO,
            BACKGROUND_RATIO,
            MIN_JOB_CYCLES,
            MAX_JOB_CYCLES,
        )
    if WORKLOAD_PROFILE == "load_from_file":
        return Workload.from_file(TASK_QUEUE_FILE)
    raise ValueError(f"Unsupported WORKLOAD_PROFILE: {WORKLOAD_PROFILE}")


@dataclass
class Task:
    tid: int
    h: int
    w: int
    exec_total: int
    arrival_tick: int
    kname: str = ""
    status: str = "UNKNOWN"
    remaining: int = 0
    owned_tiles: list[tuple[int, int]] = field(default_factory=list)
    t_arrival: int = 0
    t_scheduled: int = 0
    t_configured: int = 0
    t_completed: int = 0

    @property
    def area(self) -> int:
        return self.h * self.w

    @property
    def shape(self) -> tuple[int, int]:
        return (self.h, self.w)


class Simulator:
    def __init__(
        self,
        mode: str,
        architecture: Optional[ArchitectureModel] = None,
        workload: Optional[Workload] = None,
        run_name: str = RUN_NAME,
        results_csv_path: str = RESULTS_CSV_PATH,
    ):
        self.mode = mode
        self.architecture = architecture if architecture is not None else build_default_architecture()

        self.workload = workload if workload is not None else build_default_workload()
        self.workload.validate_for_architecture(self.architecture)

        self.run_name = run_name
        self.results_csv_path = results_csv_path
        self.reset()

    def reset(self):
        self.grid = [[-1 for _ in range(self.architecture.grid_w)] for _ in range(self.architecture.grid_h)]
        self.tasks = []
        for i, ((h, w), runtime) in enumerate(self.workload.task_queue):
            if self.mode == "monolithic":
                effective_runtime = runtime
            else:
                effective_runtime = math.ceil(runtime * self.architecture.non_monolithic_bw_factor)

            self.tasks.append(
                Task(tid=i, h=h, w=w, exec_total=effective_runtime, arrival_tick=i, kname=f"kernel_{i}")
            )

        self.tick = 0
        self.arrival_cursor = 0
        self.pending = deque()
        self.configuring: dict[int, Task] = {}
        self.running: dict[int, Task] = {}
        self.completed: dict[int, Task] = {}

        self.defrag_attempts = 0
        self.defrag_successes = 0
        self.events = []
        self.done = False
        self.exported = False

    def free_area(self) -> int:
        return sum(1 for r in range(self.architecture.grid_h) for c in range(self.architecture.grid_w) if self.grid[r][c] == -1)

    def can_place(self, task: Task, top: int, left: int) -> bool:
        if top < 0 or left < 0 or top + task.h > self.architecture.grid_h or left + task.w > self.architecture.grid_w:
            return False
        for rr in range(top, top + task.h):
            for cc in range(left, left + task.w):
                if self.grid[rr][cc] != -1:
                    return False
        return True

    def place_at(self, task: Task, top: int, left: int):
        owned = []
        for rr in range(top, top + task.h):
            for cc in range(left, left + task.w):
                self.grid[rr][cc] = task.tid
                owned.append((rr, cc))
        task.owned_tiles = owned
        if task.remaining <= 0:
            task.remaining = task.exec_total

    def find_first_fit(self, task: Task):
        for top in range(self.architecture.grid_h - task.h + 1):
            for left in range(self.architecture.grid_w - task.w + 1):
                if self.can_place(task, top, left):
                    return (top, left)
        return None

    def release(self, task: Task):
        for rr, cc in task.owned_tiles:
            self.grid[rr][cc] = -1
        task.owned_tiles = []
        task.status = "COMPLETED"

    def mark_scheduled_and_configured(self, task: Task):
        if task.status == "ISSUED":
            task.t_scheduled = self.tick
            task.t_configured = self.tick + (self.architecture.config_cost_per_area * task.area)
        elif task.t_configured == 0:
            task.t_configured = self.tick
        task.status = "CONFIGURING"

    def task_is_ready(self, task: Task) -> bool:
        return self.tick >= (task.t_arrival + self.architecture.host_enqueue_latency)

    def export_kernel_csv(self, path: str | None = None):
        out_path = path if path is not None else self.results_csv_path
        if out_path is None or out_path == "":
            return

        file_empty = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
        with open(out_path, "a", encoding="utf-8") as fp:
            if file_empty:
                fp.write("kernel_id,run_name,mode,kname,status,t_arrival,t_scheduled,t_configured,t_completed\n")

            for task in self.tasks:
                fp.write(
                    f"{task.tid},{self.run_name},{self.mode},{task.kname},{task.status},"
                    f"{task.t_arrival},{task.t_scheduled},{task.t_configured},{task.t_completed}\n"
                )

    def _snapshot(self):
        grid_cp = [row[:] for row in self.grid]
        running_tiles = {t.tid: list(t.owned_tiles) for t in self.running.values()}
        configuring_tiles = {t.tid: list(t.owned_tiles) for t in self.configuring.values()}
        return (grid_cp, running_tiles, configuring_tiles)

    def _restore(self, snap):
        grid_cp, running_tiles, configuring_tiles = snap
        self.grid = [row[:] for row in grid_cp]
        for tid, tiles in running_tiles.items():
            if tid in self.running:
                self.running[tid].owned_tiles = list(tiles)
        for tid, tiles in configuring_tiles.items():
            if tid in self.configuring:
                self.configuring[tid].owned_tiles = list(tiles)

    def _clear_running_tiles(self):
        for task in self.running.values():
            for rr, cc in task.owned_tiles:
                self.grid[rr][cc] = -1
            task.owned_tiles = []
        for task in self.configuring.values():
            for rr, cc in task.owned_tiles:
                self.grid[rr][cc] = -1
            task.owned_tiles = []

    def _find_bottom_left(self, task: Task):
        for top in range(self.architecture.grid_h - task.h, -1, -1):
            for left in range(0, self.architecture.grid_w - task.w + 1):
                if self.can_place(task, top, left):
                    return (top, left)
        return None

    def attempt_defrag_for(self, blocked: Task) -> bool:
        if self.mode in ("monolithic", "tiled"):
            return False
        if not self.running and not self.configuring:
            return False
        if self.free_area() < self.architecture.defrag_trigger_free_area_multiplier * blocked.area:
            return False

        self.defrag_attempts += 1
        snapshot = self._snapshot()

        active = sorted(
            list(self.running.values()) + list(self.configuring.values()),
            key=lambda t: (t.area, t.arrival_tick),
        )
        self._clear_running_tiles()

        compaction_ok = True
        for task in active:
            dest = self._find_bottom_left(task)
            if dest is None:
                compaction_ok = False
                break
            self.place_at(task, dest[0], dest[1])

        loaded_after_compaction = False
        if compaction_ok:
            pos = self.find_first_fit(blocked)
            if pos is not None:
                self.place_at(blocked, pos[0], pos[1])
                loaded_after_compaction = True

        success = compaction_ok and loaded_after_compaction
        self.events.append(
            {
                "tick": self.tick,
                "kind": "defrag",
                "blocked": blocked.tid,
                "shape": blocked.shape,
                "free_area": self.free_area(),
                "running": len(self.running),
                "success": success,
            }
        )

        if not success:
            self._restore(snapshot)
            return False

        self.pending.popleft()
        self.mark_scheduled_and_configured(blocked)
        self.configuring[blocked.tid] = blocked
        self.defrag_successes += 1

        for task in self.running.values():
            if task.tid == blocked.tid:
                continue
            if self.mode == "stateless":
                migration_overhead = (self.architecture.config_cost_per_area + self.architecture.stateless_cost_per_area) * task.area
                task.remaining = task.exec_total + migration_overhead
            elif self.mode == "smart_stateless":
                progress = 1.0 - (task.remaining / task.exec_total)
                if progress >= self.architecture.smart_stateless_completion_threshold:
                    continue
                migration_overhead = (self.architecture.config_cost_per_area + self.architecture.stateless_cost_per_area) * task.area
                task.remaining = task.exec_total + migration_overhead
            else:
                migration_overhead = (self.architecture.config_cost_per_area + self.architecture.stateful_cost_per_area) * task.area
                task.remaining += migration_overhead
        return True

    def _place_pending_fifo(self):
        while self.pending:
            blocked = self.pending[0]
            if not self.task_is_ready(blocked):
                break
            pos = self.find_first_fit(blocked)
            if pos is not None:
                self.place_at(blocked, pos[0], pos[1])
                self.mark_scheduled_and_configured(blocked)
                self.configuring[blocked.tid] = blocked
                self.pending.popleft()
                continue

            did_defrag = self.attempt_defrag_for(blocked)
            if not did_defrag:
                break

    def _arrive_one(self):
        if self.arrival_cursor >= len(self.tasks):
            return
        task = self.tasks[self.arrival_cursor]
        task.status = "ISSUED"
        task.remaining = task.exec_total
        task.t_arrival = self.tick
        self.pending.append(task)
        self.arrival_cursor += 1

    def _schedule_monolithic(self):
        if self.running or self.configuring or not self.pending:
            return

        if not self.task_is_ready(self.pending[0]):
            return

        task = self.pending.popleft()
        pos = self.find_first_fit(task)
        if pos is None:
            self.pending.appendleft(task)
            return

        self.place_at(task, pos[0], pos[1])
        self.mark_scheduled_and_configured(task)
        self.configuring[task.tid] = task

    def _activate_configured_tasks(self):
        ready = []
        for tid, task in self.configuring.items():
            if self.tick >= task.t_configured:
                ready.append(tid)

        for tid in ready:
            task = self.configuring.pop(tid)
            task.status = "SCHEDULED"
            self.running[tid] = task

    def _execute_one_tick(self):
        finished = []
        for task in self.running.values():
            task.remaining -= 1
            if task.remaining <= 0:
                finished.append(task.tid)

        for tid in finished:
            task = self.running[tid]
            self.release(task)
            self.completed[tid] = task
            completion = max(self.tick + 1, task.t_configured + 1)
            task.t_completed = completion
            del self.running[tid]

    def step(self):
        if self.done:
            return

        self._arrive_one()
        if self.mode == "monolithic":
            self._schedule_monolithic()
        else:
            self._place_pending_fifo()
        self._activate_configured_tasks()
        self._execute_one_tick()
        self.tick += 1

        if self.arrival_cursor == len(self.tasks) and not self.pending and not self.configuring and not self.running:
            self.done = True
            self.events.append(
                {
                    "tick": self.tick,
                    "kind": "finished",
                    "makespan": self.tick,
                    "mode": self.mode,
                }
            )
            if EXPORT_ON_FINISH and not self.exported:
                self.export_kernel_csv()
                self.exported = True


def color_for_task(tid: int):
    if tid < 0:
        return (247, 247, 247)
    return (
        80 + (tid * 57) % 155,
        80 + (tid * 97) % 155,
        80 + (tid * 133) % 155,
    )


def percentile(values, p: float):
    if not values:
        return None
    if p <= 0:
        return sorted(values)[0]
    if p >= 100:
        return sorted(values)[-1]
    arr = sorted(values)
    rank = math.ceil((p / 100.0) * len(arr)) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


def compute_metrics(sim: Simulator):
    if not sim.done:
        return None

    turnaround = []
    for task in sim.tasks:
        if task.t_completed == 0 and task.status != "COMPLETED":
            continue
        turnaround.append(task.t_completed - task.t_arrival)

    if not turnaround:
        return None

    mean_turnaround = sum(turnaround) / len(turnaround)
    return {
        "mode": sim.mode,
        "makespan": sim.tick,
        "mean_turnaround": mean_turnaround,
        "p95_turnaround": percentile(turnaround, 95),
        "p99_turnaround": percentile(turnaround, 99),
        "defrag_attempts": sim.defrag_attempts,
        "defrag_successes": sim.defrag_successes,
    }


def run_to_completion(mode: str, architecture: Optional[ArchitectureModel] = None, workload: Optional[Workload] = None):
    sim = Simulator(mode, architecture=architecture, workload=workload)
    while not sim.done:
        sim.step()
    return sim, compute_metrics(sim)


def print_mode_comparison(architecture: Optional[ArchitectureModel] = None, workload: Optional[Workload] = None):
    modes = ["monolithic", "tiled", "stateless", "smart_stateless", "stateful"]
    sims = {}
    metrics = {}
    for mode in modes:
        sim, m = run_to_completion(mode, architecture=architecture, workload=workload)
        sims[mode] = sim
        metrics[mode] = m

    baseline_makespan = metrics["monolithic"]["makespan"]
    print("--- Mode Comparison ---")
    print(
        "mode       makespan  speedup  improve%  mean_tt  p95_tt  p99_tt  defrag(s/a)"
    )
    for mode in modes:
        m = metrics[mode]
        speedup = baseline_makespan / m["makespan"]
        improve = ((baseline_makespan - m["makespan"]) / baseline_makespan) * 100.0
        print(
            f"{mode:<10} {m['makespan']:>8}  {speedup:>7.3f}  {improve:>8.2f}  "
            f"{m['mean_turnaround']:>7.2f}  {m['p95_turnaround']:>6}  {m['p99_turnaround']:>6}  "
            f"{m['defrag_successes']:>3}/{m['defrag_attempts']:<3}"
        )

    return sims, metrics


def draw(sim: Simulator, surface, font, big_font, running_auto: bool):
    surface.fill((22, 24, 29))

    gx = MARGIN
    gy = MARGIN
    grid_w_px = sim.architecture.grid_w * CELL
    grid_h_px = sim.architecture.grid_h * CELL

    pygame.draw.rect(surface, (45, 48, 54), (gx - 2, gy - 2, grid_w_px + 4, grid_h_px + 4), border_radius=8)

    for r in range(sim.architecture.grid_h):
        for c in range(sim.architecture.grid_w):
            tid = sim.grid[r][c]
            col = color_for_task(tid)
            rect = pygame.Rect(gx + c * CELL, gy + r * CELL, CELL - 2, CELL - 2)
            pygame.draw.rect(surface, col, rect, border_radius=6)

    sx = gx + grid_w_px + 30
    lines = [
        f"Mode: {sim.mode}",
        f"Tick: {sim.tick}",
        f"Arrival cursor: {sim.arrival_cursor}/{len(sim.tasks)}",
        f"Pending: {len(sim.pending)}",
        f"Configuring: {len(sim.configuring)}",
        f"Running: {len(sim.running)}",
        f"Completed: {len(sim.completed)}",
        f"Free area: {sim.free_area()}",
        f"Defrag attempts: {sim.defrag_attempts}",
        f"Defrag successes: {sim.defrag_successes}",
        f"Done: {sim.done}",
    ]

    title = big_font.render("Reactive Defrag Simulator", True, (238, 240, 245))
    surface.blit(title, (sx, gy))

    y = gy + 48
    for line in lines:
        txt = font.render(line, True, (210, 215, 225))
        surface.blit(txt, (sx, y))
        y += 28

    controls = [
        "Controls:",
        "Space: pause/resume",
        "N: single tick",
        "R: reset",
        "1/2/3/4/5: monolithic/tiled/stateless/smart/stateful",
        "S: run to completion",
        "C: compare all modes",
        f"Auto run: {running_auto}",
    ]
    y += 14
    for line in controls:
        txt = font.render(line, True, (174, 183, 201))
        surface.blit(txt, (sx, y))
        y += 24

    y += 12
    recent = sim.events[-10:]
    header = font.render("Recent events:", True, (238, 240, 245))
    surface.blit(header, (sx, y))
    y += 26
    if not recent:
        surface.blit(font.render("(none)", True, (130, 140, 160)), (sx, y))
    else:
        for e in reversed(recent):
            if e["kind"] == "defrag":
                text = f"t={e['tick']} blocked#{e['blocked']} {e['shape']} success={e['success']}"
            else:
                text = f"t={e['tick']} FINISHED makespan={e['makespan']}"
            surface.blit(font.render(text, True, (130, 220, 180) if "FINISHED" in text else (160, 172, 195)), (sx, y))
            y += 22


def print_summary(sim: Simulator):
    print("--- Simulation Summary ---")
    print("mode", sim.mode)
    print("tick", sim.tick)
    print("completed", len(sim.completed), "/", len(sim.tasks))
    print("defrag_attempts", sim.defrag_attempts)
    print("defrag_successes", sim.defrag_successes)
    print("done", sim.done)
    metrics = compute_metrics(sim)
    if metrics is not None:
        print("mean_turnaround", f"{metrics['mean_turnaround']:.2f}")
        print("p95_turnaround", metrics["p95_turnaround"])
        print("p99_turnaround", metrics["p99_turnaround"])


def main():
    maybe_clean_results_on_boot()
    architecture = build_default_architecture()
    workload = build_default_workload()
    workload.validate_for_architecture(architecture)
    pygame.init()
    pygame.display.set_caption("FPGA Defragmentation Simulator")
    width = max(MIN_WINDOW_W, MARGIN * 2 + architecture.grid_w * CELL + SIDEBAR_W)
    height = max(MIN_WINDOW_H, MARGIN * 2 + architecture.grid_h * CELL)
    surface = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 28, bold=True)

    sim = Simulator(MODE, architecture=architecture, workload=workload)
    auto_run = AUTO_RUN

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print_summary(sim)
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_run = not auto_run
                elif event.key == pygame.K_n:
                    sim.step()
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_s:
                    while not sim.done:
                        sim.step()
                    print_summary(sim)
                elif event.key == pygame.K_c:
                    print_mode_comparison(architecture=architecture, workload=workload)
                elif event.key == pygame.K_1:
                    sim = Simulator("monolithic", architecture=architecture, workload=workload)
                elif event.key == pygame.K_2:
                    sim = Simulator("tiled", architecture=architecture, workload=workload)
                elif event.key == pygame.K_3:
                    sim = Simulator("stateless", architecture=architecture, workload=workload)
                elif event.key == pygame.K_4:
                    sim = Simulator("smart_stateless", architecture=architecture, workload=workload)
                elif event.key == pygame.K_5:
                    sim = Simulator("stateful", architecture=architecture, workload=workload)

        if auto_run and not sim.done:
            sim.step()

        draw(sim, surface, font, big_font, auto_run)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
