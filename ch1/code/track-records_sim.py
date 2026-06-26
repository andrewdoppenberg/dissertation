"""Chapter 1 — Track-Records simulation.

A community of Bayesian "scientists" repeatedly forecasts a biased coin. Each
agent blends two sources of belief: its own (noisy) evidence and the credences
of the neighbours it trusts. The experiments differ in *how* agents choose those
neighbours and *how* their track records are scored, letting us ask when
track-record-based deference helps or hurts collective accuracy.

Paper symbol -> code name:
    b (inferential noise / expertise)        -> TRScientist.noise
    m (openness, fraction of informants)     -> TRScientist.m
    c (anti-sociality, self vs. social weight) -> TRScientist.c
    I (social inertia)                       -> InertialScientist.skepticism_k

Requires the Mesa 2.x API (mesa.time, positional Agent(unique_id, model));
Mesa 3.0 removed both. See requirements.txt.
"""

# Limit each worker process to a single math-library thread. Without this, every
# worker spawns one BLAS/OpenMP thread per core, so a multiprocessing Pool on a
# big SLURM node oversubscribes the CPUs badly. Must run before numpy is imported
# (directly or indirectly via mesa), so it stays at the very top of the file.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from tqdm import tqdm

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector


# ============================================================================
# Shared constants and math helpers
# ============================================================================

# The six candidate coin biases agents reason about (the hypothesis space).
HYP_SPACE = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)

# Bounds for the truncated-normal perception noise.
TRUNCNORM_LOWER = 0.0001
TRUNCNORM_UPPER = 0.9999


def fast_truncnorm(loc, scale, lower=TRUNCNORM_LOWER, upper=TRUNCNORM_UPPER):
    """Sample a truncated normal via inverse-transform sampling.

    Faster than scipy's truncnorm and hardened against the overflow/underflow
    that occurs when ``scale`` is tiny or ``loc`` sits in the distribution tails.
    """
    scale = max(scale, 1e-9)  # avoid division by zero / infinite z-scores

    p_alpha = ndtr((lower - loc) / scale)
    p_beta = ndtr((upper - loc) / scale)

    # Guard against precision loss at the tails: keep a strictly positive gap and
    # stay clear of 0/1 (ndtri(0) = -inf, ndtri(1) = +inf).
    p_beta = np.maximum(p_beta, p_alpha + 1e-16)
    p_alpha = np.clip(p_alpha, 1e-15, 1.0 - 1e-15)
    p_beta = np.clip(p_beta, 1e-15, 1.0 - 1e-15)

    z = ndtri(np.random.uniform(p_alpha, p_beta))
    return np.clip(z * scale + loc, lower, upper)


def perceived_toss(actual_toss, bias_prob):
    """Return the toss as perceived by an agent.

    With probability ``bias_prob`` the agent misreads the toss (0<->1); otherwise
    it perceives it correctly. The random draw always happens so the RNG stream
    is identical whether or not bias is active.
    """
    return 1 - actual_toss if random.random() < bias_prob else actual_toss


def bayesian_update(cred, hyp, toss):
    """One Bayesian update of a credence distribution on a single coin toss."""
    likelihood = np.abs((1 - toss) - hyp)
    denom = np.dot(cred, likelihood)
    # Fall back to the prior if the evidence is incompatible with every hypothesis.
    return (likelihood * cred / denom) if denom > 1e-10 else cred


def add_perception_noise(cred, noise):
    """Perturb a credence distribution with truncated-normal noise and renormalise."""
    if noise < 1e-6:
        return cred
    noisy = np.maximum(fast_truncnorm(cred, noise), 0)
    total = np.sum(noisy)
    return noisy / total if total > 0 else noisy


def update_on_toss(cred, hyp, toss, noise):
    """Bayesian update on a toss, then add perception noise. The core belief step."""
    return add_perception_noise(bayesian_update(cred, hyp, toss), noise)


# ============================================================================
# Model-level metric reporters (collected once per step)
# ============================================================================

def opinion_diversity(model):
    """Spread of opinion: std-dev of agents' point probability estimates.

    Each agent's point estimate is the expected coin bias under its credence.
    0 means perfect consensus; larger values mean more disagreement.
    """
    point_ests = np.array([np.dot(a.cred, a.hyp) for a in model.schedule.agents],
                          dtype=np.float64)
    return round(float(np.std(point_ests)), 6)


def neighbor_diversity(model):
    """Cumulative diversity of influence, as normalised Shannon entropy.

    Rather than a per-step snapshot, this keeps a running tally of every
    neighbour link ever formed and reports how evenly influence is spread across
    agents. Low entropy means a few agents have held a long-term monopoly on
    being trusted.
    """
    agents = model.schedule.agents
    n = len(agents)
    if n == 0:
        return 0.0

    # Persistent per-agent tally of how often each agent has been someone's neighbour.
    if not hasattr(model, 'cumulative_counts'):
        model.cumulative_counts = np.zeros(n, dtype=np.float64)

    id_to_idx = {a.unique_id: i for i, a in enumerate(agents)}
    for agent in agents:
        for neighbor in agent.neighbors:
            model.cumulative_counts[id_to_idx[neighbor.unique_id]] += 1

    total = model.cumulative_counts.sum()
    if total == 0:
        return 0.0

    probs = model.cumulative_counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n)
    normalised = round(float(entropy / max_entropy) if max_entropy > 0 else 0.0, 6)

    # NOTE: the `30 **` factor rescales normalised entropy from [0, 1] to ~[1, 30].
    # This looks unintentional; left unchanged so earlier results stay reproducible.
    return 30 ** normalised


def avg_neighbor_noise(model):
    """Mean ``noise`` of every agent that currently appears in anyone's neighbour list.

    Returns 0.0 if no agent has neighbours yet.
    """
    referenced = {
        neighbor.unique_id: neighbor
        for agent in model.schedule.agents
        for neighbor in agent.neighbors
    }
    if not referenced:
        return 0.0
    return round(float(np.mean([a.noise for a in referenced.values()])), 6)


# ============================================================================
# Agents
# ============================================================================

class TRScientist(Agent):
    """Baseline scientist: forecasts a coin and defers to the best track records.

    Each step the agent (1) observes its own coin toss and updates a private
    "evidential" credence, (2) picks the neighbours with the best track records,
    and (3) averages their credences into a "social" credence. In ``advance`` the
    two are blended by ``c`` (own weight) vs ``1 - c`` (social weight).
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.record = []                                          # Brier scores feeding this agent's track record
        self.m = round(random.uniform(0.05, 0.5), 2)             # openness: fraction of the community consulted
        self.hyp = HYP_SPACE                                     # hypothesis space (shared, read-only)
        self.cred = np.random.dirichlet(np.ones(6)).astype(np.float32)  # current credence over HYP_SPACE
        self.noise = random.uniform(0, 0.2)                      # perception noise / inverse expertise
        self.c = round(random.random(), 2)                       # weight on own evidence vs. neighbours
        self.bias_prob = 0                                       # toss-misperception rate (0 = perceive correctly)

        self.neighbors = []          # trusted informants chosen this step
        self.social = None           # mean credence of neighbours
        self.evidential = None        # credence from own evidence
        self.Brier = []              # per-step Brier scores vs. objective tosses
        self.BrierT = None           # Brier score vs. the true bias (reported metric)

        # r_avg() cache, invalidated whenever the record grows.
        self._r_avg_cache = None
        self._record_len_cache = 0

    # -- track record -------------------------------------------------------
    def r_avg(self):
        """Average recorded Brier score (lower is better), cached between updates."""
        if len(self.record) != self._record_len_cache:
            self._r_avg_cache = sum(self.record) / len(self.record) if self.record else 1.0
            self._record_len_cache = len(self.record)
        return self._r_avg_cache

    # -- belief updates -----------------------------------------------------
    def update_evidence(self):
        """Observe one private toss and form the evidential credence."""
        actual_toss = np.random.binomial(1, self.model.truth)
        # Brier score of the current forecast against objective reality.
        self.Brier.append(round((actual_toss - np.dot(self.cred, self.hyp)) ** 2, 4))
        observed = perceived_toss(actual_toss, self.bias_prob)
        self.evidential = update_on_toss(self.cred, self.hyp, observed, self.noise)

    def update_neighbors(self):
        """Select the ``m``-fraction of agents with the best (lowest) track records."""
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)

        if n < 1:
            self.neighbors = [self]
        elif not self.record:
            # No track records exist yet, so choose at random.
            self.neighbors = random.sample(agents, n)
        else:
            r_avgs = np.array([a.r_avg() for a in agents])
            k = min(n, len(r_avgs) - 1)
            indices = np.argpartition(r_avgs, k)[:n]
            self.neighbors = [agents[i] for i in indices] or [self]

    def update_social(self):
        """Social credence = mean credence of the chosen neighbours."""
        neighbor_creds = np.array([a.cred for a in self.neighbors], dtype=np.float32)
        self.social = np.mean(neighbor_creds, axis=0)

    def warm_up(self, steps):
        """Run ``steps`` private Bayesian updates before the sim, seeding a prior."""
        for _ in range(steps):
            toss = np.random.binomial(1, self.model.truth)
            observed = perceived_toss(toss, self.bias_prob)
            self.cred = update_on_toss(self.cred, self.hyp, observed, self.noise)

    # -- scheduler hooks (SimultaneousActivation) ---------------------------
    def step(self):
        self.update_evidence()
        self.update_neighbors()
        self.update_social()

    def advance(self):
        self.cred = (1 - self.c) * self.social + self.c * self.evidential
        truth_idx = int(self.model.truth * 5)
        self.BrierT = round(np.sum(self.cred ** 2) + 1 - 2 * self.cred[truth_idx], 4)


class RandomScientist(TRScientist):
    """Chooses neighbours uniformly at random, ignoring track records entirely."""

    def update_neighbors(self):
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)
        self.neighbors = random.sample(agents, n) if n >= 1 else [self]


class PatientScientist(TRScientist):
    """Starts fully self-reliant (c=1); the model later opens it to social influence."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.c = 1


class InertialScientist(TRScientist):
    """Trusts others only once their track record clearly beats a fixed prior.

    ``skepticism_k`` smooths a short record toward ``prior_score`` (Bayesian
    shrinkage), and the social weight opens gradually as the best neighbour's
    smoothed record improves, preventing early herding on lucky testimony.
    """

    def __init__(self, unique_id, model, skepticism_k):
        super().__init__(unique_id, model)
        self.skepticism_k = skepticism_k
        self.prior_score = 0.25
        self.base_c = self.c

    def r_avg(self):
        if not self.record:
            return self.prior_score
        n = len(self.record)
        return (sum(self.record) + self.skepticism_k * self.prior_score) / (n + self.skepticism_k)

    def advance(self):
        best_neighbor_score = min(neigh.r_avg() for neigh in self.neighbors)
        if best_neighbor_score >= self.prior_score:
            trust_factor = 0.0
        else:
            trust_factor = 1.0 - (best_neighbor_score / self.prior_score)

        # Open the social channel in proportion to earned trust.
        self.c = 1 - (1 - self.base_c) * trust_factor
        super().advance()


class DataSharingScientist(TRScientist):
    """Shares its raw coin observation, so peers can update on data, not just beliefs."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_observation = None

    def update_evidence(self):
        toss = np.random.binomial(1, self.model.truth)
        self.last_observation = toss
        self.Brier.append(round((toss - np.dot(self.cred, self.hyp)) ** 2, 4))
        self.evidential = update_on_toss(self.cred, self.hyp, toss, self.noise)

    def incorporate_shared_data(self):
        """Update the evidential credence on every other agent's shared toss."""
        shared = [a.last_observation for a in self.model.schedule.agents
                  if a.last_observation is not None and a is not self]
        if not shared:
            return
        cred = self.evidential.copy()
        for toss in shared:
            cred = update_on_toss(cred, self.hyp, toss, self.noise)
        self.evidential = cred

    def update_social(self):
        # On the first step there are no peer credences yet, so fall back to evidence.
        if self.model.schedule.steps == 0:
            self.social = self.evidential.copy()
        else:
            super().update_social()

    def step(self):
        self.update_evidence()
        self.update_neighbors()
        self.incorporate_shared_data()
        self.update_social()


class PublicExamScientist(TRScientist):
    """Scored on a single shared public toss rather than its own private toss."""

    def update_evidence(self):
        toss = self.model.public_toss
        self.Brier.append(round((toss - np.dot(self.cred, self.hyp)) ** 2, 4))
        self.evidential = update_on_toss(self.cred, self.hyp, toss, self.noise)


class NoisyTRScientist(TRScientist):
    """Perceives peers' track records with additive Gaussian noise."""

    def update_neighbors(self):
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)

        if n < 1:
            self.neighbors = [self]
        elif not self.record:
            self.neighbors = random.sample(agents, n)
        else:
            true_r_avgs = np.array([a.r_avg() for a in agents], dtype=np.float32)
            perceived = true_r_avgs + np.random.normal(0, self.model.track_record_noise, size=len(agents))
            k = min(n, len(perceived) - 1)
            indices = np.argpartition(perceived, k)[:n]
            self.neighbors = [agents[i] for i in indices] or [self]


class FuzzyTRScientist(TRScientist):
    """Perceives track records only in coarse score bands.

    Agents can tell good records from bad but cannot finely rank close ones, so
    they prefer better bands and break ties at random within a band.
    """

    def update_neighbors(self):
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)

        if n < 1:
            self.neighbors = [self]
        elif not self.record:
            self.neighbors = random.sample(agents, n)
        else:
            true_r_avgs = np.array([a.r_avg() for a in agents], dtype=np.float32)
            granularity = max(float(self.model.track_record_granularity), 1e-9)
            bands = np.floor(true_r_avgs / granularity).astype(int)

            selected = []
            for band in sorted(np.unique(bands)):
                if len(selected) >= n:
                    break
                band_indices = np.flatnonzero(bands == band).tolist()
                random.shuffle(band_indices)
                selected.extend(agents[i] for i in band_indices[:n - len(selected)])
            self.neighbors = selected or [self]


class DelayedTRScientist(TRScientist):
    """Sees peers' track records only after a publication delay of ``model.delay`` steps."""

    def delayed_r_avg(self):
        visible_count = len(self.record) - self.model.delay
        if visible_count <= 0:
            return 1.0  # nothing published yet
        return sum(self.record[:visible_count]) / visible_count

    def update_neighbors(self):
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)
        if n < 1:
            self.neighbors = [self]
            return

        r_avgs = np.array([a.delayed_r_avg() for a in agents])
        if np.all(r_avgs == 1.0):
            # Nothing published yet for anyone -> choose at random.
            self.neighbors = random.sample(agents, n)
        else:
            k = min(n, len(r_avgs) - 1)
            indices = np.argpartition(r_avgs, k)[:n]
            self.neighbors = [agents[i] for i in indices] or [self]


class PeerPredictionScientist(TRScientist):
    """Flips its own coin like the baseline, but is scored on predicting *all* tosses.

    The pre-update point forecast is cached before the belief update so the score
    reflects the belief that generated the forecast. ``PeerPredictionModel`` later
    averages each agent's forecast error across every agent's toss.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_flip = None        # this agent's toss this step
        self.last_prediction = None  # forecast made *before* this step's update

    def update_evidence(self):
        self.last_prediction = float(np.dot(self.cred, self.hyp))
        actual_toss = np.random.binomial(1, self.model.truth)
        self.last_flip = actual_toss
        self.Brier.append(round((actual_toss - self.last_prediction) ** 2, 4))
        observed = perceived_toss(actual_toss, self.bias_prob)
        self.evidential = update_on_toss(self.cred, self.hyp, observed, self.noise)

    def peer_brier_score(self):
        """Mean Brier score of this agent's forecast across every agent's toss."""
        if self.last_prediction is None:
            return 1.0
        scores = [(a.last_flip - self.last_prediction) ** 2
                  for a in self.model.schedule.agents if a.last_flip is not None]
        return round(sum(scores) / len(scores), 4) if scores else 1.0


# -- Warm-up (informed-prior) variants ---------------------------------------
# These agents run a private Bayesian warm-up before the sim. Lower-noise agents
# observe more pre-game tosses, so they enter with sharper, better-informed priors.

class PriorScientist(TRScientist):
    """Warms up for a number of steps that grows with the agent's noise advantage."""

    TRAINING_STEP_MULTIPLIER = 50

    def __init__(self, unique_id, model, training_steps=None):
        super().__init__(unique_id, model)
        if training_steps is None:
            training_steps = round(np.abs(self.noise - 0.2) * self.TRAINING_STEP_MULTIPLIER)
        self.warm_up(training_steps)


class Prior500Scientist(PriorScientist):
    """As ``PriorScientist`` but with a 10x longer warm-up."""

    TRAINING_STEP_MULTIPLIER = 500


class PriorInertialScientist(InertialScientist):
    """Combines informed warm-up priors with inertial, track-record-gated trust."""

    def __init__(self, unique_id, model, skepticism_k):
        super().__init__(unique_id, model, skepticism_k)
        self.warm_up(round(np.abs(self.noise - 0.2) * 50))


class PriorPeerPredictionScientist(PeerPredictionScientist):
    """Combines informed warm-up priors with peer-prediction scoring."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.warm_up(round(np.abs(self.noise - 0.2) * 50))


class RandomPriorPeerPredictionScientist(PriorPeerPredictionScientist):
    """Warm-up priors + peer-prediction scoring + random neighbour selection.

    Isolates the value of prior quality and peer-scored records from any benefit
    of strategic (track-record-based) network formation.
    """

    def update_neighbors(self):
        agents = self.model.schedule.agents
        n = round(len(agents) * self.m)
        self.neighbors = random.sample(agents, n) if n >= 1 else [self]


# ============================================================================
# Models
# ============================================================================

def make_datacollector():
    """The data collector shared by every track-record model."""
    return DataCollector(
        agent_reporters={"BrierT": "BrierT"},
        model_reporters={
            "truth": "truth",
            "OpinionDiversity": opinion_diversity,
            "NeighborDiversity": neighbor_diversity,
            "AvgNeighborNoise": avg_neighbor_noise,
        },
    )


class TRModel(Model):
    """Standard track-record model: agents accumulate Brier-scored records and
    preferentially adopt the best-scoring neighbours.

    Subclasses customise behaviour by setting ``agent_class``, overriding
    ``_make_agent`` (for a mixed population or extra constructor args), or
    overriding ``_record_value`` (what gets appended to a track record).
    """

    agent_class = TRScientist

    def __init__(self, truth, feedback_rate, n_agents=30):
        super().__init__()
        self.truth = truth
        self.fbr = feedback_rate            # probability a step yields track-record feedback
        self.n_agents = n_agents
        self.schedule = SimultaneousActivation(self)
        for i in range(n_agents):
            self.schedule.add(self._make_agent(i))
        self.datacollector = make_datacollector()

    def _make_agent(self, i):
        return self.agent_class(i, self)

    def _record_value(self, agent):
        """Value appended to an agent's track record when feedback fires."""
        return agent.Brier[-1]

    def _collect_feedback(self):
        """With probability ``fbr``, append this step's score to every record."""
        if np.random.binomial(1, self.fbr):
            for a in self.schedule.agents:
                a.record.append(self._record_value(a))
                a._record_len_cache = 0  # invalidate r_avg cache

    def step(self):
        self.schedule.step()
        self._collect_feedback()
        self.datacollector.collect(self)


class RandomModel(TRModel):
    """All agents pick neighbours at random (no track-record selection)."""

    agent_class = RandomScientist


class LessMonopolyModel(TRModel):
    """Half track-record agents, half random agents — dilutes epistemic monopolies."""

    def _make_agent(self, i):
        if i < self.n_agents // 2:
            return TRScientist(i, self)
        return RandomScientist(i, self)


class PatientModel(TRModel):
    """Agents gather evidence alone first; ``activate`` later opens them to influence."""

    agent_class = PatientScientist

    def activate(self):
        for a in self.schedule.agents:
            a.c = round(random.random(), 2)


class InertialModel(TRModel):
    """Social influence must be earned by a clear track-record advantage."""

    def _make_agent(self, i):
        return InertialScientist(i, self, skepticism_k=25)


class PriorInertialModel(TRModel):
    """Informed warm-up priors combined with inertial, track-record-gated trust."""

    def _make_agent(self, i):
        return PriorInertialScientist(i, self, skepticism_k=25)


class DataSharingModel(TRModel):
    """Agents inform one another with raw coin observations, not just credences."""

    agent_class = DataSharingScientist

    def step(self):
        agents = list(self.schedule.agents)
        # Phase the update so everyone shares data from the same time slice.
        for a in agents:
            a.update_evidence()
        for a in agents:
            a.update_neighbors()
        for a in agents:
            a.incorporate_shared_data()
        for a in agents:
            a.update_social()
        for a in agents:
            a.advance()

        self.schedule.steps += 1
        self.schedule.time += 1
        self._advance_time()

        self._collect_feedback()
        self.datacollector.collect(self)


class PublicExamModel(TRModel):
    """Every step all agents are scored on the same shared public toss."""

    agent_class = PublicExamScientist

    def __init__(self, truth, feedback_rate, n_agents=30):
        self.public_toss = None
        super().__init__(truth, feedback_rate, n_agents)

    def step(self):
        self.public_toss = np.random.binomial(1, self.truth)
        super().step()


class NoisyTrackRecordModel(TRModel):
    """Agents misperceive peers' track records, blocking perfectly meritocratic networks."""

    agent_class = NoisyTRScientist

    def __init__(self, truth, feedback_rate, n_agents=30, track_record_noise=0.2):
        self.track_record_noise = track_record_noise
        super().__init__(truth, feedback_rate, n_agents)


class FuzzyTrackRecordModel(TRModel):
    """Agents see coarse track-record bands and randomise within a band."""

    agent_class = FuzzyTRScientist

    def __init__(self, truth, feedback_rate, n_agents=30, track_record_granularity=0.05):
        self.track_record_granularity = track_record_granularity
        super().__init__(truth, feedback_rate, n_agents)


class DelayedTrackRecordModel(TRModel):
    """Track records become visible only after a publication delay."""

    agent_class = DelayedTRScientist

    def __init__(self, truth, feedback_rate, n_agents=30, delay=10):
        self.delay = delay
        super().__init__(truth, feedback_rate, n_agents)


class PeerPredictionModel(TRModel):
    """Records score each agent's forecast against *all* agents' tosses, not just its own.

    The wider scoring window accumulates information faster and rewards genuinely
    calibrated beliefs over lucky private streaks.
    """

    agent_class = PeerPredictionScientist

    def _record_value(self, agent):
        return agent.peer_brier_score()


class PriorModel(TRModel):
    """Agents enter with informed warm-up priors."""

    agent_class = PriorScientist


class Prior500Model(TRModel):
    """As ``PriorModel`` but with a 10x longer warm-up."""

    agent_class = Prior500Scientist


class PriorPeerPredictionModel(PeerPredictionModel):
    """Informed warm-up priors combined with peer-prediction scoring."""

    agent_class = PriorPeerPredictionScientist


class RandomPriorPeerPredictionModel(PeerPredictionModel):
    """Warm-up priors + peer-prediction scoring + random neighbour selection."""

    agent_class = RandomPriorPeerPredictionScientist


# ============================================================================
# Experiment configuration and parallel execution
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


def _package_results(model):
    """Join the per-agent ``BrierT`` series with the model-level diversity metrics."""
    agent_df = model.datacollector.get_agent_vars_dataframe()
    model_df = model.datacollector.get_model_vars_dataframe()[
        ["OpinionDiversity", "NeighborDiversity", "AvgNeighborNoise"]
    ]
    # agent_df has a (Step, AgentID) MultiIndex; align model metrics on the Step level.
    return agent_df.join(model_df, on="Step")


# -- Per-config setup and stepping hooks --------------------------------------

def hook_setup_baseline(model):
    """Force every agent to ignore social evidence (c=1): the asocial baseline."""
    for agent in model.schedule.agents:
        agent.c = 1


def hook_step_standard(model, num_steps):
    for _ in range(num_steps):
        model.step()


def hook_step_patience(model, num_steps):
    """Run 50 self-reliant steps, open agents to influence, then run the rest."""
    for _ in range(50):
        model.step()
    model.activate()
    for _ in range(num_steps - 50):
        model.step()


# Each experiment = a model class + an optional setup hook + a stepping hook.
CONFIGS = {
    "baseline":                     {"class": TRModel,                       "setup": hook_setup_baseline, "step": hook_step_standard},
    "testimony":                    {"class": TRModel,                       "setup": None,                "step": hook_step_standard},
    "patience":                     {"class": PatientModel,                  "setup": None,                "step": hook_step_patience},
    "less_monopoly":                {"class": LessMonopolyModel,             "setup": None,                "step": hook_step_standard},
    "random":                       {"class": RandomModel,                   "setup": None,                "step": hook_step_standard},
    "prior_inertial":               {"class": PriorInertialModel,            "setup": None,                "step": hook_step_standard},
    "inertial":                     {"class": InertialModel,                 "setup": None,                "step": hook_step_standard},
    "sharing":                      {"class": DataSharingModel,              "setup": None,                "step": hook_step_standard},
    "noisy_tr":                     {"class": NoisyTrackRecordModel,         "setup": None,                "step": hook_step_standard},
    "fuzzy_tr":                     {"class": FuzzyTrackRecordModel,         "setup": None,                "step": hook_step_standard},
    "communal":                     {"class": PublicExamModel,               "setup": None,                "step": hook_step_standard},
    "prior":                        {"class": PriorModel,                    "setup": None,                "step": hook_step_standard},
    "prior_500":                    {"class": Prior500Model,                 "setup": None,                "step": hook_step_standard},
    "delayed":                      {"class": DelayedTrackRecordModel,       "setup": None,                "step": hook_step_standard},
    "peer_prediction":              {"class": PeerPredictionModel,           "setup": None,                "step": hook_step_standard},
    "prior_peer_prediction":        {"class": PriorPeerPredictionModel,      "setup": None,                "step": hook_step_standard},
    "prior_peer_prediction_random": {"class": RandomPriorPeerPredictionModel, "setup": None,               "step": hook_step_standard},
    "prior_baseline":               {"class": PriorModel,                    "setup": hook_setup_baseline, "step": hook_step_standard},
    "prior_500_baseline":           {"class": Prior500Model,                 "setup": hook_setup_baseline, "step": hook_step_standard},
    "sharing_baseline":             {"class": DataSharingModel,              "setup": hook_setup_baseline, "step": hook_step_standard},
}


def _generic_worker(payload):
    """Run one simulation (one config, one random truth, one seed)."""
    config_key, run_id, num_steps, seed, n_agents = payload
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)

    conf = CONFIGS[config_key]
    truth = random.choice(HYP_SPACE)
    model = conf["class"](truth, 1, n_agents=n_agents)

    if conf["setup"]:
        conf["setup"](model)
    conf["step"](model, num_steps)

    return _package_results(model)


def run_simulation(config_name, num_runs=1, num_steps=100, n_agents=30, parallel=True):
    """Run ``num_runs`` independent simulations of ``config_name`` and stack the results."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")

    desc = f"Running {config_name}"
    seeds = np.random.default_rng().integers(0, 2 ** 32 - 1, size=num_runs)
    tasks = [(config_name, r, num_steps, s, n_agents) for r, s in zip(range(num_runs), seeds)]

    if parallel:
        with Pool(processes=resolve_workers()) as pool:
            results_iter = pool.imap_unordered(_generic_worker, tasks, chunksize=1)
            results = list(progress(results_iter, total=num_runs, desc=desc))
    else:
        results = [_generic_worker(t) for t in progress(tasks, total=len(tasks), desc=desc)]

    return pd.concat(results, ignore_index=False, copy=False, sort=False)


if __name__ == "__main__":
    import pickle

    # ── Simulation scale ────────────────────────────────────────────────────
    N_RUNS = 100_000
    N_STEPS = 200
    N_AGENTS = 30
    PARALLEL = True

    # (config name, agent count) pairs to run, in order.
    simulations_to_run = [
        ("baseline", N_AGENTS),
        ("testimony", N_AGENTS),
        ("less_monopoly", N_AGENTS),
        ("patience", N_AGENTS),
        ("prior_baseline", N_AGENTS),
        ("prior", N_AGENTS),
        ("prior_500_baseline", N_AGENTS),
        ("prior_500", N_AGENTS),
        ("peer_prediction", N_AGENTS),
        ("prior_peer_prediction", N_AGENTS),
        ("random", N_AGENTS),
        ("prior_peer_prediction_random", N_AGENTS),
        ("inertial", N_AGENTS),
        ("prior_inertial", N_AGENTS),
        ("delayed", N_AGENTS),
        ("sharing_baseline", N_AGENTS),
        ("sharing", N_AGENTS),
        ("communal", N_AGENTS),
        ("noisy_tr", N_AGENTS),
        ("fuzzy_tr", N_AGENTS),
        ("sharing_baseline", 5),
        ("sharing", 5),
    ]

    # Resolve output path relative to this script (not the working directory).
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    pkl_path = os.path.join(results_dir, "track-records.pkl")

    # Load any earlier results so runs accumulate across invocations.
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as fh:
            results_store = pickle.load(fh)
        print(f"Loaded existing results from {pkl_path} ({len(results_store)} entries)")
    else:
        results_store = {}

    COLS = ["BrierT", "OpinionDiversity", "NeighborDiversity", "AvgNeighborNoise"]

    for sim_name, n_agents in simulations_to_run:
        df = run_simulation(sim_name, num_runs=N_RUNS, num_steps=N_STEPS,
                            n_agents=n_agents, parallel=PARALLEL)

        # Average each metric across runs, per step.
        raw = df.reset_index(level="Step")[["Step"] + COLS]
        aggregated = raw.groupby("Step")[COLS].mean()

        key = f"{sim_name}_{n_agents}"
        results_store[key] = aggregated
        print(f"Stored '{key}' in results store")

    with open(pkl_path, "wb") as fh:
        pickle.dump(results_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved all results to {pkl_path} ({len(results_store)} entries total)")
