"""Chapter 2 — Zollman-effect simulation.

A bandit-style model of a divided scientific community. Two groups of agents
test two arms (A and B, with B truly superior by ``delta``), update Beta
posteriors on their own and their network's evidence, and trust out-group
testimony only partially (weight ``w``, possibly after a ``delay``). A parameter
sweep runs every configuration in parallel and records when the community
reaches correct consensus and how accurate its credences become.

Paper symbol -> code name:
    w (out-group trust weight)                    -> MISTRUST_LEVELS / the `w` argument
    delta (success-rate gap between arms)         -> DELTAS / `delta`

Metric orientation note: the 'credence' metric below is stored as a Brier
*accuracy* (1 - Brier), so higher is better; the thesis discusses CA as the
Brier *error* (lower is better). The two are equivalent up to this 1 - x flip.
"""

# Limit each worker process to a single math-library thread (see ch1/ch3): stops
# a multiprocessing Pool from oversubscribing CPUs on a big SLURM node. Must run
# before numpy is imported, so it stays at the very top of the module.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import multiprocessing as mp
import pickle

import numpy as np
from tqdm import tqdm


# ============================================================================
# Simulation parameters
# ============================================================================

# ── Simulation scale ────────────────────────────────────────────────────────
NUM_RUNS  = 100_000   # Independent parallel simulations per configuration
NUM_MC    = 100_000   # Monte Carlo samples used to estimate each agent's credence
N_AGENTS  = 4         # Must be even; first half = group 1, second half = group 2

# ── Parameter sweep axes ────────────────────────────────────────────────────
TIME_STEPS     = [1, 10, 100, 1000, 10000]
MAX_STEPS      = max(TIME_STEPS)
DELTAS         = [0.0001, 0.001, 0.01, 0.1, 0.5]   # True advantage of arm B over arm A
TRIALS         = [1, 10, 100, 1000]                 # Trials performed per agent per step
MISTRUST_LEVELS = [0.01, 0.1, 0.5, 1.0]            # Out-group trust weight w ∈ (0,1]
DELAYS         = [0, 1, 5]                          # Time steps before mistrust level w takes effect

# ── Brier chunking ───────────────────────────────────────────────────────────
# The Brier calculation requires (chunk × N_AGENTS × NUM_MC) arrays.
# BRIER_CHUNK=100 keeps each beta-sample array ≈ 160 MB (float32), avoiding OOM
# while cutting Python-loop overhead ~1000× vs. the original one-run-at-a-time approach.
BRIER_CHUNK = 100


# ============================================================================
# Single-configuration simulation
# ============================================================================

def simulate_config(params):
    """
    Run one full sweep configuration and return outcome metrics.

    Parameters
    ----------
    params : tuple
        (delta, n_trials, w, max_steps, delay)
        delta    – true probability gap between arms A and B
        n_trials – number of trials each agent runs per time step
        w        – out-group trust weight; in-group weight is always 1.0
        max_steps – number of update steps to simulate
        delay    - number of steps before w takes effect (initial w=1.0)

    Returns
    -------
    (params, results) where results is a dict with keys:
        'consensus'          – list[float], final-step fraction of runs where all
                               agents correctly prefer arm B
        'consensus_all_steps'– list[float], same metric recorded at every step
                               (length max_steps + 1, step 0 = prior state)
        'credence'           – list[float], mean Brier accuracy of posterior credences
        'mse'                – list[float], mean squared error of expected-value estimates
    """
    delta, n_trials, w, max_steps, delay = params

    # ── Trust matrices ────────────────────────────────────────────────────────
    # T_initial is the full-trust matrix used before the delay
    T_initial = np.ones((N_AGENTS, N_AGENTS), dtype=np.float32)

    # T_final is the trust matrix with weight w
    T_final = np.full((N_AGENTS, N_AGENTS), w, dtype=np.float32)
    n1 = N_AGENTS // 2
    T_final[:n1, :n1] = 1.0   # group 1 trusts group 1
    T_final[n1:, n1:] = 1.0   # group 2 trusts group 2
    np.fill_diagonal(T_final, 1.0)

    # Precompute transposes used in every update step as S @ T_T.
    T_T_initial = np.ascontiguousarray(T_initial.T)
    T_T_final   = np.ascontiguousarray(T_final.T)


    # ── Prior beliefs (Beta distribution parameters) ─────────────────────────
    # Shape: (NUM_RUNS, N_AGENTS, 2)  – last axis indexes arms [A, B]
    # Initialised from Uniform(1,4) so agents start with mild, varied uncertainty.
    alpha = np.random.uniform(1, 4, size=(NUM_RUNS, N_AGENTS, 2)).astype(np.float32)
    beta  = np.random.uniform(1, 4, size=(NUM_RUNS, N_AGENTS, 2)).astype(np.float32)

    # ── True success probabilities ────────────────────────────────────────────
    # p_A ∈ [0, 1-delta]  →  p_B = p_A + delta  (arm B is always superior)
    p_A = np.random.uniform(0, 1 - delta, size=(NUM_RUNS, 1)).astype(np.float32)
    p_B = p_A + delta   # shape (NUM_RUNS, 1), broadcasts over agents

    results = {
        'consensus':           [],
        'consensus_all_steps': [],
        'credence':            [],
        'mse':                 [],
        'mse_per_run':         [],
        'mse_non_consensus':   [],
        'mse_consensus':       [],
        'min_evidence':        [],
        'divergence_trajectory': [],  # mean absolute between-group choice difference at each step
    }

    # ── Step-0 baseline (prior only, no evidence yet) ─────────────────────────
    # E[arm] = α/(α+β) is the posterior mean under a Beta prior.
    E = alpha / (alpha + beta)   # shape (NUM_RUNS, N_AGENTS, 2)
    consensus_initial = np.mean(np.all(E[:, :, 1] > E[:, :, 0], axis=1))
    results['consensus_all_steps'].append(float(consensus_initial))

    evidence_count = np.zeros((NUM_RUNS, N_AGENTS, 2), dtype=np.float32)

    # ── Main simulation loop ──────────────────────────────────────────────────
    for _step in range(1, max_steps + 1):
        # Choose trust matrix based on delay
        T_T = T_T_initial if _step <= delay else T_T_final


        # 1. Choose the arm with the higher posterior mean (greedy / exploit)
        #    choice[r,i] ∈ {0,1}; shape (NUM_RUNS, N_AGENTS)
        choice = np.argmax(E, axis=2)

        # Record mean absolute within-run divergence between groups before the belief update.
        # Absolute difference is taken per-run before averaging so that opposite-direction
        # divergences across runs don't cancel out.
        divergence = np.abs(
            np.mean(choice[:, :n1] == 1, axis=1) -   # group 1 fraction per run
            np.mean(choice[:, n1:] == 1, axis=1)      # group 2 fraction per run
        )   # shape (NUM_RUNS,)
        results['divergence_trajectory'].append(float(np.mean(divergence)))

        # 2. Look up each agent's true success probability for its chosen arm
        p_chosen = np.where(choice == 0, p_A, p_B)   # (NUM_RUNS, N_AGENTS)

        # 3. Draw binomial outcomes
        S = np.random.binomial(n_trials, p_chosen).astype(np.float32)  # successes
        F = (n_trials - S).astype(np.float32)                          # failures

        # 4. Separate evidence by chosen arm
        #    mask_B[r,i] = True  ↔  agent i chose arm B in run r
        mask_B = (choice == 1)
        mask_A = ~mask_B

        # 5. Update beliefs: each agent pools weighted evidence from the network.
        #    S_A[r,i] is the successes agent i contributed to arm A's update;
        #    the matrix multiply S_A @ T_T distributes that evidence to all agents
        #    according to their trust weights.
        #    (Two vectorised matmuls replace the original Python for-loop over arms.)
        S_A = S * mask_A;  F_A = F * mask_A
        S_B = S * mask_B;  F_B = F * mask_B

        alpha[:, :, 0] += S_A @ T_T
        beta[ :, :, 0] += F_A @ T_T
        alpha[:, :, 1] += S_B @ T_T
        beta[ :, :, 1] += F_B @ T_T

        trials_A = (mask_A * n_trials).astype(np.float32)
        trials_B = (mask_B * n_trials).astype(np.float32)
        evidence_count[:, :, 0] += trials_A @ T_T
        evidence_count[:, :, 1] += trials_B @ T_T

        # 6. Recompute posterior means in-place (reused as E at top of next step,
        #    eliminating the duplicate E_updated calculation from the original code).
        E = alpha / (alpha + beta)

        # 7. Consensus: fraction of runs where every agent correctly prefers arm B
        consensus = float(np.mean(np.all(E[:, :, 1] > E[:, :, 0], axis=1)))
        results['consensus_all_steps'].append(consensus)

    # Record the final-step consensus for heatmap use
    results['consensus'] = [consensus]

    # ── Brier accuracy of posterior credences ─────────────────────────────────
    # For each agent, estimate P(arm B > arm A) via Monte Carlo sampling from the
    # posterior Beta distributions, then compute the Brier score against the true
    # outcome (arm B is always better, so the "true" probability is 1 for B, 0 for A).
    #
    # Memory strategy: process BRIER_CHUNK runs at a time so each beta-sample array
    # stays ≈ BRIER_CHUNK × N_AGENTS × NUM_MC × 4 bytes in RAM.
    # This is ~1000× fewer Python iterations than the original per-run loop.
    brier_totals = np.zeros(N_AGENTS, dtype=np.float64)  # accumulate per-agent sums

    for start in range(0, NUM_RUNS, BRIER_CHUNK):
        end   = min(start + BRIER_CHUNK, NUM_RUNS)
        chunk = end - start   # actual number of runs in this chunk

        # Draw MC samples; shape: (chunk, N_AGENTS, NUM_MC)
        sA = np.random.beta(
            alpha[start:end, :, 0:1].astype(np.float64),   # broadcast over MC axis
            beta[ start:end, :, 0:1].astype(np.float64),
            size=(chunk, N_AGENTS, NUM_MC)
        )
        sB = np.random.beta(
            alpha[start:end, :, 1:2].astype(np.float64),
            beta[ start:end, :, 1:2].astype(np.float64),
            size=(chunk, N_AGENTS, NUM_MC)
        )

        # Credences: estimated probability that each arm is superior
        cred_B = np.mean(sB > sA, axis=-1)   # (chunk, N_AGENTS)
        cred_A = 1.0 - cred_B                # complementary (exact, no extra MC cost)

        # Brier score per agent: average squared error against true outcomes
        # (truth = 1 for B being better, 0 for A being worse)
        # Stored as accuracy = 1 - Brier (higher is better); see note at top of file.
        brier_chunk = 1.0 - ((cred_B - 1.0)**2 + cred_A**2) / 2.0   # (chunk, N_AGENTS)

        # Accumulate weighted sum; we'll divide by NUM_RUNS at the end
        brier_totals += brier_chunk.sum(axis=0)

    mean_brier_per_agent = brier_totals / NUM_RUNS
    results['credence'] = [float(np.mean(mean_brier_per_agent))]

    # ── Mean Squared Error of posterior mean estimates ────────────────────────
    # How close are agents' final expected-value estimates to the true probabilities?
    mse_A = (E[:, :, 0] - p_A)**2   # p_A broadcasts over agent axis
    mse_B = (E[:, :, 1] - p_B)**2
    mse_run = np.mean((mse_A + mse_B) / 2.0, axis=1)

    results['mse'] = [float(np.mean((mse_A + mse_B) / 2.0))]
    results['mse_per_run'] = mse_run.tolist()
    
    is_B_chosen = (E[:, :, 1] > E[:, :, 0])
    mse_consensus_arr = np.where(is_B_chosen, mse_B, mse_A)
    mse_non_consensus_arr = np.where(is_B_chosen, mse_A, mse_B)
    
    results['mse_consensus'] = [float(np.mean(mse_consensus_arr))]
    results['mse_non_consensus'] = [float(np.mean(mse_non_consensus_arr))]
    
    mse_non_consensus_run = np.mean(mse_non_consensus_arr, axis=1)
    results['mse_non_consensus_per_run'] = mse_non_consensus_run.tolist()
    
    # Worst MSE that an agent has in a theory in a given run
    max_mse_per_run = np.max(np.maximum(mse_A, mse_B), axis=1)
    results['max_mse_per_run'] = max_mse_per_run.tolist()

    # ── Minimum evidence ────────────────────────
    # Minimum trust-weighted evidence that any agent obtained in the run about any theory
    run_min_evidence = np.min(evidence_count, axis=(1, 2))
    results['min_evidence'] = [float(np.mean(run_min_evidence))]
    results['min_evidence_per_run'] = run_min_evidence.tolist()
    
    # Fraction of agents across all runs that received less than 1 effective piece of evidence about a theory
    results['fraction_agents_less_than_1'] = [float(np.mean(np.any(evidence_count < 1.0, axis=2)))]

    return params, results


# ============================================================================
# Parallel execution and output
# ============================================================================

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


def main():
    # Cartesian product of all sweep parameters
    configs = list(itertools.product(DELTAS, TRIALS, MISTRUST_LEVELS, TIME_STEPS, DELAYS))

    workers = resolve_workers()
    print(f"Running {len(configs)} configurations across {workers} workers...")
    with mp.Pool(processes=workers) as pool:
        raw_results = list(progress(pool.imap(simulate_config, configs),
                                    total=len(configs), desc="Simulations"))

    # Unpack flat results list into a nested dict keyed by (delta, n_trials, w, t, delay)
    data = {
        d: {n: {w: {t: {dy: {} for dy in DELAYS} for t in TIME_STEPS}
                    for w in MISTRUST_LEVELS}
                for n in TRIALS}
            for d in DELTAS
    }
    for (delta, n, w, t, dy), res in raw_results:
        data[delta][n][w][t][dy] = res

    output = {
        'data':            data,
        'TIME_STEPS':      TIME_STEPS,
        'MAX_STEPS':       MAX_STEPS,
        'DELTAS':          DELTAS,
        'TRIALS':          TRIALS,
        'MISTRUST_LEVELS': MISTRUST_LEVELS,
        'DELAYS':          DELAYS,
    }

    # Write next to this script's chapter results, regardless of working directory.
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "zollman-effect.pkl")
    # Ensure the directory exists or map to local directory
    target_dir = os.path.dirname(output_path)
    if target_dir and not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            # Fallback to current working directory if there's any permission/creation issue
            output_path = "zollman-effect_sweep.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Simulation complete. Results saved to '{output_path}'.")


if __name__ == '__main__':
    main()