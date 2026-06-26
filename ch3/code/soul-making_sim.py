"""Chapter 3 — Soul-Making / Problem of Evil simulation.

A population of agents whose *resources*, *happiness*, and *virtue* evolve under
two forces:

1) **Nature** ("natural evil"): stochastic resource shocks plus occasional
   "horrendous" events that halve an agent's resources.
2) **Interpersonal help**: happier agents may transfer resources to sadder ones,
   and the helper's virtue shifts in response to the outcome.

A parameter sweep (driven from ``__main__``) runs many independent simulations
per configuration in parallel and saves aggregated summary tables.

Paper symbol -> code name:
    mu_N (natural evil level)        -> p["event_bias"]
    sigma_N (natural-event noise)    -> p["event_noise"]
    r (rate of horrendous events)    -> p["hor_rate"]
    gamma (happiness concavity)      -> p["concavity"]
    the six mu_V dynamics            -> p["equation"] in {1..6} (happinessChangeToVirtueChange)
"""
from __future__ import annotations

# Limit each worker process to a single math-library thread (see ch1/ch2): stops
# a multiprocessing Pool from oversubscribing CPUs on a big SLURM node. Must run
# before numpy is imported, so it stays at the very top of the module.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import itertools
import math
import multiprocessing as mp
import random
import statistics as stats
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Agents and the single-run simulation
# ============================================================================

class Agent:
    """
    Represents an individual in the Soul-Making simulation.
    
    Attributes
    ----------
    n : int
        Unique identifier for the agent.
    resources : float
        Current accumulated resources (material well-being), bounded [0, 100].
    concavity : float
        Parameter shaping how resources translate into happiness. 
        A value < 1.0 represents diminishing marginal utility of resources.
    virtue : float
        Current moral virtue, representing how 'good' the agent's soul is, bounded [0, 100].
    happiness : float
        Calculated subjective well-being derived from resources via concavity.
    dR, dH, dV : float
        Deltas tracking changes in resources, happiness, and virtue over the current interaction.
    hor_flag : bool
        Boolean flag marking if the agent recently suffered a 'horrendous' evil (resource halving event).
    """
    def __init__(self, n: int = 0, resources: float = 50.0, concavity: float = 1.0, virtue: float = 50.0):
        self.n = n
        self.resources = resources
        self.concavity = concavity
        self.virtue = virtue

        self.happiness = 100*(self.resources/100)**self.concavity

        self.dR = np.nan
        self.dH = np.nan
        self.dV = np.nan

        self.hor_flag = False

    def __repr__(self):
        """
        Return a string representation of the Agent.
        """
        return f"Agent(n={self.n}, r={round(self.resources,1)}, h={round(self.happiness,1)}, v={round(self.virtue,1)})"
    
    def adjust_resources(self, amount: float):
        """
        Adjust the agent's resources by `amount`, strictly maintaining bounds [0, 100].
        Also recalibrates happiness dynamically in response.

        :param amount: A float representing the amount to adjust resources by (positive or negative).
        """
        old_resources = self.resources
        self.resources = max(0, min(self.resources+amount,100))
        self.dR = self.resources - old_resources
        self.adjust_happiness()

    def adjust_happiness(self):
        """
        Recalculate the agent's absolute happiness derived from the current resources
        and the concavity attribute. Logs the difference `dH` relative to previous state.
        """
        old_happiness = self.happiness
        self.happiness = 100*(self.resources/100)**self.concavity
        self.dH = self.happiness - old_happiness

    def adjust_virtue(self, amount: float):
        """
        Adjust the agent's moral virtue by `amount`, strictly maintaining bounds [0, 100].

        :param amount: A float modifying the agent's soul/virtue.
        """
        old_virtue = self.virtue
        self.virtue = max(0, min(self.virtue+amount, 100))
        self.dV = self.virtue - old_virtue

__all__ = [
    "createAgents",
    "happinessChangeToVirtueChange",
    "naturalInteraction",
    "random_pairs",
    "personalInteraction",
    "getStats",
    "soul_making_sim",
]

# ---------- Core helpers ----------

def createAgents(p: Dict[str, Any]) -> List[Agent]:
    """Create a list of agents with random initial resources and virtue.

    Parameters
    ----------
    p : dict
        Simulation parameters. Expected keys include:
        - 'nagents': int, number of agents to create
        - 'concavity': float, passed to Agent constructor
        - 'min_initial_virtue', 'max_initial_virtue': floats in [0, 100]

    Returns
    -------
    list[Agent]
        Newly created agents with randomized starting states.
    """
    nagents = int(p["nagents"])  # make intent explicit

    agents = [
        Agent(
            n=i,
            resources=float(np.random.uniform(0, 100)),
            concavity=p["concavity"],
            virtue=float(
                np.random.uniform(p["min_initial_virtue"], p["max_initial_virtue"])  # type: ignore[arg-type]
            ),
        )
        for i in range(nagents)
    ]
    return agents


def happinessChangeToVirtueChange(p: Dict[str, Any], V: float, dH: float) -> float:
    """Map an agent's change in happiness to a (noisy) change in virtue.

    The mapping uses one of six equation families, selected by `p['equation']`:
        1:  bias =  (V/50 - 1) * (dH/5)
        2:  bias = -(V/50 - 1) * (dH/5)
        3:  bias =  |V/50 - 1| * (dH/5)
        4:  bias = -|V/50 - 1| * (dH/5)
        5:  bias =  (V/50 - 1) * |dH/5|
        6:  bias = -(V/50 - 1) * |dH/5|

    A Gaussian perturbation with stdev = sqrt(|dH|) is added.

    Parameters
    ----------
    p : dict
        Must include an integer 'equation' in {1,2,3,4,5,6}.
    V : float
        Current virtue (0..100 scale assumed).
    dH : float
        Recent change in happiness.

    Returns
    -------
    float
        The proposed change in virtue dV.
    """
    # Division by 5 keeps world-type effects modest.
    mag = dH / 5.0
    vterm = (V / 50.0) - 1.0

    eq = int(p.get("equation", 1))
    if eq == 1:
        virtue_bias = vterm * mag
    elif eq == 2:
        virtue_bias = -vterm * mag
    elif eq == 3:
        virtue_bias = abs(vterm) * mag
    elif eq == 4:
        virtue_bias = -abs(vterm) * mag
    elif eq == 5:
        virtue_bias = vterm * abs(mag)
    elif eq == 6:
        virtue_bias = -vterm * abs(mag)
    else:
        raise ValueError(f"Invalid equation number: {eq}")

    # Noise scale ties volatility to the magnitude of happiness change.
    stdev = math.sqrt(abs(dH))
    dV = random.gauss(virtue_bias, stdev)
    return dV


def naturalInteraction(agents: Sequence[Agent], p: Dict[str, Any], metrics: Dict[str, Any] | None = None) -> None:
    """Apply *nature* shocks to each agent's resources and virtue.

    For each agent:
    - With probability `p['hor_rate']`, a *harmful event* occurs: resources are
      halved (i.e., resource change `dR = -0.5 * resources`). When `p['dist']`
      is 'fair', we restrict harmful events to above-average-happiness agents.
      When `p['dist']` is 'lowVirtue', we restrict harmful events to agents
      whose virtue is at or below the current average.
    - Otherwise a Gaussian event with mean `p['event_bias']` and stdev
      `p['event_noise']` shifts resources.
    - In all cases, virtue adjusts in response to the agent's resulting
      happiness change via `happinessChangeToVirtueChange`.
    """
    # Capture pre-step baselines so we can measure *nature-only* effects.
    res_before = [a.resources for a in agents]
    v_before = [a.virtue for a in agents]

    dist = p.get("dist", "random")
    mean_happiness = float(np.mean([a.happiness for a in agents]))
    mean_virtue = float(np.mean([a.virtue for a in agents]))

    if dist == "fair":
        eligible = [a.happiness >= mean_happiness for a in agents]
    elif dist == "lowVirtue":
        eligible = [a.virtue <= mean_virtue for a in agents]
    else:
        eligible = None

    n_eligible = int(np.sum(eligible)) if eligible is not None else len(agents)

    for i, agent in enumerate(agents):
        # Determine whether a harmful event happens to this agent.
        if eligible is not None:
            hor_probability = p["hor_rate"] * len(agents) / n_eligible if n_eligible else 0
            hor_applies = eligible[i] and (random.random() < hor_probability)
        else:
            hor_applies = random.random() < p["hor_rate"]

        if hor_applies:
            # Harmful event: lose half of current resources.
            dR = -0.5 * agent.resources
            agent.hor_flag = True
        else:
            # Regular stochastic event.
            dR = float(np.random.normal(p["event_bias"], p["event_noise"]))
            agent.hor_flag = False

        agent.adjust_resources(dR)

        # Map the happiness change (set by adjust_resources) to a virtue change.
        dV = happinessChangeToVirtueChange(p, agent.virtue, agent.dH)
        agent.adjust_virtue(dV)

        if metrics is not None:
            # virtue delta due to nature this step
            metrics.setdefault("nat_virtue_dV", []).append(float(agent.virtue - v_before[i]))
            # resource loss only when a horror happened
            if getattr(agent, "hor_flag", False):
                metrics["hor_count"] = metrics.get("hor_count", 0) + 1
                loss = float(res_before[i] - agent.resources)
                metrics["hor_losses"] = metrics.get("hor_losses", 0) + loss


def random_pairs(agents: Sequence[Agent]) -> List[Tuple[Agent, Agent]]:
    """Return random disjoint pairs of agents while preserving input order.

    The original implementation shuffled indices rather than the agents list
    (to avoid side-effects). If the population size is odd, the last agent is
    intentionally dropped (unchanged vs. original behavior).
    """
    indices = list(range(len(agents)))
    random.shuffle(indices)
    # Pair consecutive indices: (0,1), (2,3), ...  Any last odd index is dropped.
    return [(agents[indices[i]], agents[indices[i + 1]]) for i in range(0, len(indices) - 1, 2)]


def personalInteraction(agents: Sequence[Agent], p: Dict[str, Any], metrics: Dict[str, Any] | None = None) -> None:
    """Pair agents and allow happier ones to (probabilistically) help sadder ones.

    For each pair, the happier agent *may* transfer resources to the sadder one
    based on a `propensity_to_help` that increases with the helper's virtue and
    the recipient's sadness. Helping is gated by three conditions:
      - helper must *not* have just experienced a harmful natural event,
      - helper must have resources > 20,
      - a random draw falls below `propensity_to_help`.

    If a transfer occurs, the helper's virtue updates in proportion to the
    average of (recipient's + helper's) happiness changes, damped by
    `exp(p['event_bias'])` to reflect background conditions.
    """
    for a, b in random_pairs(agents):
        # Count this pairing
        if metrics is not None:
            metrics["pair_total"] = metrics.get("pair_total", 0) + 1

        # Identify happy/sad actors for this pair.
        helper = a if a.happiness > b.happiness else b
        helped = b if a.happiness > b.happiness else a

        # Agents can help if they haven't experienced a horror and aren't in poverty
        can_help = (not getattr(helper, "hor_flag", False)) and (helper.resources > 20)

        if metrics is not None and can_help:
            metrics["pair_can_help_true"] = metrics.get("pair_can_help_true", 0) + 1

        if not can_help:
            continue

        # Propensity combines helper virtue and recipient unhappiness.
        propensity_to_help = ((helper.virtue / 100.0) + (1.0 - (helped.happiness / 100.0)))/2

        # Agent choses to help with probability propensity_to_help
        chooses_to_help = propensity_to_help > random.random()

        if metrics is not None and chooses_to_help:
            metrics["pair_chooses_true"] = metrics.get("pair_chooses_true", 0) + 1

        if not chooses_to_help:
            continue

        # Transfer half the resource gap scaled by willingness to help.
        resources_given = 0.5 * (helper.resources - helped.resources) * propensity_to_help
        helper.adjust_resources(-resources_given)
        helped.adjust_resources(+resources_given)

        # Virtue change for helper reflects perceived good done under conditions.
        dV = 5.0 * (helped.dH - helper.dH) / 2.0 / p['damping']
        helper.adjust_virtue(dV)

        if metrics is not None:
            metrics['helps'] = metrics['helps'] + 1
            metrics['help_resources_given'] = metrics['help_resources_given'] + resources_given
            metrics['help_virtue_dV'] = metrics['help_virtue_dV'] + dV


def getStats(agents: Iterable[Agent]) -> Dict[str, float]:
    """Compute population-level summary statistics.

    Returns a dict with:
    - 'meanHappiness': average current happiness
    - 'meanVirtue': average current virtue
    - 'meanDifference': average |happiness - virtue|
    """
    # Convert iterator once to avoid multiple passes.
    agents_list = list(agents)

    mean_h = float(np.mean([a.happiness for a in agents_list])) if agents_list else 0.0
    mean_v = float(np.mean([a.virtue for a in agents_list])) if agents_list else 0.0
    mean_diff = float(np.mean([abs(a.happiness - a.virtue) for a in agents_list])) if agents_list else 0.0

    return {
        "meanHappiness": mean_h,
        "meanVirtue": mean_v,
        "meanDifference": mean_diff,
    }


def soul_making_sim(p: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Soul-Making simulation for `p['steps']` time steps.

    Parameters
    ----------
    p : dict
        Simulation parameters. Expected keys include:
        - 'steps': int, number of time steps to simulate
        - Keys required by `createAgents`, `naturalInteraction`, `personalInteraction`.

    Returns
    -------
    dict
        A merged dict of the input parameters and summary statistics at the
        start and end of the run. Keys include:
        'Initial/Final Happiness', 'Initial/Final Virtue', 'Initial/Final Difference'.
    """
    p = dict(p)
    agents = createAgents(p)
    initial_stats = getStats(agents)

    # --- metrics accumulator ---
    metrics: Dict[str, Any] = {
        "hor_count": 0,
        "hor_losses": 0,
        "nat_virtue_dV": [],
        "pair_total": 0,
        "pair_can_help_true": 0,
        "pair_chooses_true": 0,
        "helps": 0,
        "help_resources_given": 0,
        "help_virtue_dV": 0,
    }

    p['damping'] = 1.0 + math.exp(float(p["event_bias"]))

    for _ in range(int(p["steps"])):
        naturalInteraction(agents, p, metrics=metrics)
        personalInteraction(agents, p, metrics=metrics)

    final_stats = getStats(agents)

    pair_total = metrics["pair_total"] or 0
    results = {
        # Agent States
        "Initial Happiness": initial_stats["meanHappiness"],
        "Initial Virtue": initial_stats["meanVirtue"],
        "Initial Difference": initial_stats["meanDifference"],
        "Final Happiness": final_stats["meanHappiness"],
        "Final Virtue": final_stats["meanVirtue"],
        "Final Difference": final_stats["meanDifference"],
        # Natural Interactions
        "Natural Virtue": 
            stats.mean(metrics['nat_virtue_dV']) if metrics['nat_virtue_dV'] else math.nan,
        "Horrors": 
            metrics["hor_count"],
        "Horror Losses":  
            metrics['hor_losses'] / metrics['hor_count'] if metrics["hor_count"] else 0.0,
        # Personal Interactions
        "Can Help Rate":
            (100.0 * metrics["pair_can_help_true"] / pair_total if pair_total else 0.0),
        "Helps": 
            metrics['helps'],
        "Help Rate":
            (100.0 * metrics["pair_chooses_true"] / pair_total if pair_total else 0.0),
        "Help Resources": 
            metrics['help_resources_given'] / metrics['helps'] if metrics['helps'] else 0.0,
        "Help Virtue": 
            metrics["help_virtue_dV"] / metrics['helps'] if metrics['helps'] else 0.0,
    }

    # Return original parameters alongside summary outputs for convenient logging.
    return {**p, **results}

# ============================================================================
# Parameter sweep, parallel execution, and aggregation
# ============================================================================

# DEBUG flips to a tiny, serial sweep that is easy to step through; the normal
# run sweeps event_bias over a wide range in parallel.
DEBUG = False

if DEBUG:
    USE_PARALLEL = False
    EVENT_BIAS_MIN, EVENT_BIAS_MAX, EVENT_BIAS_STEP = 0.0, 0.0, 0.5
else:
    USE_PARALLEL = True
    EVENT_BIAS_MIN, EVENT_BIAS_MAX, EVENT_BIAS_STEP = -3.0, 3.0, 0.05

def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _empty_group(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "params": dict(params),
        "count": 0,
        "sums": {},
        "counts": {},
        "metric_order": [],
        "max": {},
        "min": {},
    }


def _add_metric(group: Dict[str, Any], col: str, value: Any) -> None:
    if not _is_numeric(value):
        return

    if col not in group["metric_order"]:
        group["metric_order"].append(col)

    numeric = float(value)
    if math.isnan(numeric):
        return

    group["sums"][col] = group["sums"].get(col, 0.0) + numeric
    group["counts"][col] = group["counts"].get(col, 0) + 1


def _update_max(group: Dict[str, Any], col: str, value: Any) -> None:
    if not _is_numeric(value):
        return

    numeric = float(value)
    if math.isnan(numeric):
        return

    if col not in group["max"] or numeric > group["max"][col]:
        group["max"][col] = numeric


def _update_min(group: Dict[str, Any], col: str, value: Any) -> None:
    if not _is_numeric(value):
        return

    numeric = float(value)
    if math.isnan(numeric):
        return

    if col not in group["min"] or numeric < group["min"][col]:
        group["min"][col] = numeric


def _add_result_to_group(group: Dict[str, Any], result: Dict[str, Any], keys: Sequence[str]) -> None:
    group["count"] += 1

    for col, value in result.items():
        if col in keys:
            continue
        _add_metric(group, col, value)

    _update_max(group, "Final Happiness", result.get("Final Happiness"))
    _update_max(group, "Final Virtue", result.get("Final Virtue"))
    _update_min(group, "Final Difference", result.get("Final Difference"))


def _merge_group(accumulators: Dict[Tuple[Any, ...], Dict[str, Any]], partial: Dict[str, Any], keys: Sequence[str]) -> None:
    key = tuple(partial["params"][k] for k in keys)
    target = accumulators.get(key)

    if target is None:
        accumulators[key] = partial
        return

    target["count"] += partial["count"]

    for col in partial["metric_order"]:
        if col not in target["metric_order"]:
            target["metric_order"].append(col)

    for col, value in partial["sums"].items():
        target["sums"][col] = target["sums"].get(col, 0.0) + value

    for col, value in partial["counts"].items():
        target["counts"][col] = target["counts"].get(col, 0) + value

    for col, value in partial["max"].items():
        if col not in target["max"] or value > target["max"][col]:
            target["max"][col] = value

    for col, value in partial["min"].items():
        if col not in target["min"] or value < target["min"][col]:
            target["min"][col] = value


def _run_batch(task: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    params, repeats = task
    keys = tuple(params.keys())
    group = _empty_group({k: params[k] for k in keys})

    for _ in range(repeats):
        _add_result_to_group(group, soul_making_sim(params), keys)

    return group


def _iter_batch_tasks(param_combinations: Sequence[Dict[str, Any]], repeats: int, batch_size: int):
    for params in param_combinations:
        remaining = repeats
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            yield params, current_batch
            remaining -= current_batch


def _seed_worker() -> None:
    seed = (os.getpid() ^ time.time_ns()) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)


def resolve_workers(requested=None):
    """Number of worker processes to use.

    Honours an explicit request, then SLURM's allocation (so a shared node is
    never oversubscribed), then falls back to the local CPU count.
    """
    if requested:
        return max(1, int(requested))

    ntasks = os.environ.get("SLURM_NTASKS")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if ntasks and cpus_per_task:
        try:
            total = int(ntasks) * int(cpus_per_task)
            if total > 0:
                return total
        except ValueError:
            pass

    on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    if on_node:
        try:
            if int(on_node) > 0:
                return int(on_node)
        except ValueError:
            pass

    return max(1, mp.cpu_count())


def progress(iterable, total=None, desc=None):
    """tqdm bar that still renders inside a non-interactive SLURM log file.

    SLURM redirects output to a file (not a TTY); a fixed width plus a 1 s
    refresh, together with ``python -u`` in the batch script, make the bar update
    live in the .txt log instead of appearing only at the very end.
    """
    return tqdm(iterable, total=total, desc=desc,
                mininterval=1.0, dynamic_ncols=False, ncols=80)


def run_serial(batch_tasks, total_batches):
    """Run simulation batches sequentially, yielding partial aggregates."""
    for task in progress(batch_tasks, total=total_batches, desc="Simulations"):
        yield _run_batch(task)


def run_parallel(batch_tasks, total_batches, workers: int):
    """Run simulation batches in parallel, yielding partial aggregates."""
    with mp.Pool(processes=workers, initializer=_seed_worker) as pool:
        for partial in progress(pool.imap_unordered(_run_batch, batch_tasks, chunksize=1),
                                total=total_batches, desc="Simulations"):
            yield partial


def build_summary_frames(accumulators: Dict[Tuple[Any, ...], Dict[str, Any]], keys: Sequence[str]):
    metric_order = []
    for group in accumulators.values():
        for col in group["metric_order"]:
            if col not in metric_order:
                metric_order.append(col)

    mean_rows = []
    grouped_rows = []
    for group in accumulators.values():
        mean_row = dict(group["params"])
        for col in metric_order:
            count = group["counts"].get(col, 0)
            mean_row[col] = group["sums"].get(col, 0.0) / count if count else math.nan

        grouped_row = dict(mean_row)
        grouped_row["Best Final Happiness"] = group["max"].get("Final Happiness", math.nan)
        grouped_row["Best Final Virtue"] = group["max"].get("Final Virtue", math.nan)
        grouped_row["Best Final Difference"] = group["min"].get("Final Difference", math.nan)

        mean_rows.append(mean_row)
        grouped_rows.append(grouped_row)

    T_mean = pd.DataFrame(mean_rows).sort_values(list(keys)).reset_index(drop=True)
    T_grouped = pd.DataFrame(grouped_rows).sort_values(list(keys)).reset_index(drop=True)
    return T_mean, T_grouped

if __name__ == '__main__':
    start_time = time.time()

    mp.freeze_support()

    p = argparse.ArgumentParser(description="Run Soul-Making simulations")
    p.add_argument('--N', type=int, nargs='+', default=[100_000])
    p.add_argument('--steps', type=int, nargs='+', default=[100])
    p.add_argument('--nagents', type=int, nargs='+', default=[4])
    p.add_argument('--min_initial_virtue', type=int, nargs='+', default=[0])
    p.add_argument('--max_initial_virtue', type=int, nargs='+', default=[100])
    p.add_argument('--concavity', type=float, nargs='+', default=[0.5])
    p.add_argument('--equation', type=int, nargs='+', default=[1,2,3,4,5,6])
    p.add_argument('--dist', nargs='+', default=["random","fair","lowVirtue"])
    p.add_argument('--event_bias', type=float, nargs='+',
                   default=np.arange(EVENT_BIAS_MIN, EVENT_BIAS_MAX + EVENT_BIAS_STEP, EVENT_BIAS_STEP))
    p.add_argument('--event_noise', type=float, nargs='+', default=[1])
    p.add_argument('--lambda', type=float, nargs='+', default = [0]) # For skew distributions
    p.add_argument('--hor_rate', type=float, nargs='+', default=[0,0.01,0.5])
    p.add_argument('--virtue_spread', type=float, nargs='+', default=[0])
    p.add_argument('--output', type=str, default="soul-making.pkl")
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=5000)
    args = p.parse_args()
    base_parameters = vars(args)
    requested_workers = base_parameters.pop("workers")
    batch_size = int(base_parameters.pop("batch_size"))
    output_name = base_parameters.pop("output")
    
    N = int(base_parameters.get("N",[10000])[0])
    if N <= 0:
        raise ValueError("--N must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    param_combinations = [
        dict(zip(base_parameters.keys(), combo))
        for combo in itertools.product(*base_parameters.values())
    ]

    total_batches = len(param_combinations) * math.ceil(N / batch_size)
    total_runs = len(param_combinations) * N
    batch_tasks = _iter_batch_tasks(param_combinations, N, batch_size)
    worker_count = resolve_workers(requested_workers)
    print(
        f"Running {total_runs:,} simulations "
        f"({len(param_combinations):,} parameter combinations, "
        f"{total_batches:,} batches, {worker_count} workers)."
    )

    if USE_PARALLEL:
        partial_results = run_parallel(batch_tasks, total_batches, worker_count)
    else:
        partial_results = run_serial(batch_tasks, total_batches)

    # Process partial aggregates without retaining one row per simulation.
    keys = list(base_parameters.keys())
    accumulators = {}
    for partial in partial_results:
        _merge_group(accumulators, partial, keys)

    keys_wo_bias = [k for k in keys if k != "event_bias"]

    T_mean, T_grouped = build_summary_frames(accumulators, keys)


    # 1) Best *mean* Final Virtue (pick the event_bias giving the highest mean)
    idx_v = T_mean.groupby(keys_wo_bias)['Final Virtue'].idxmax()
    best_v = (
        T_mean.loc[idx_v, keys_wo_bias + ['event_bias', 'Final Virtue']]
            .rename(columns={
                'event_bias': 'Best Event Bias',
                'Final Virtue': 'Best Mean Final Virtue'
            })
    )

    # 2) Best *mean* Final Difference
    #    Assuming “best” = closest to 0 (change to .idxmin() on the raw column if you truly want the minimum)
    T_mean['abs_Final_Difference'] = T_mean['Final Difference'].abs()
    idx_d = T_mean.groupby(keys_wo_bias)['abs_Final_Difference'].idxmin()
    best_d = (
        T_mean.loc[idx_d, keys_wo_bias + ['event_bias', 'Final Difference']]
            .rename(columns={
                'event_bias': 'Best Event Bias',
                'Final Difference': 'Best Mean Final Difference'
            })
    )

    # 3) Merge the two summaries so you get one row per (equation, hor_rate, ...), plus the winning biases
    best_by_bias = best_v.merge(best_d, on=keys_wo_bias)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full simulation results to a single dictionary
    results_dict = {
        'T_grouped': T_grouped,
        'best_by_bias': best_by_bias
    }
    
    pkl_out = results_dir / output_name
    pd.to_pickle(results_dict, pkl_out)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Plotting disabled for slurm compatibility
    print("Plots not generated.")
