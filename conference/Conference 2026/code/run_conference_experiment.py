"""ICBTA 2026 conference experiment runner.

This script implements the empirical core of the trust-guided SDFL conference paper.
It deliberately evaluates the off-chain learning/governance core only; rewards, stake
slashing, and blockchain execution are NOT claimed or simulated here.

Methods
-------
C0: FedAvg (mean aggregation)
C1: Coordinate-wise trimmed mean (robust aggregation; equal weights)
C2: Static reputation weighting using current normalized accuracy gain A_i,t
C3: Dynamic trust weighting using T_i,t (no screening)
C4: Proposed dynamic trust + screening + trust-weighted trimmed mean

Scenarios
---------
S0: benign clients
S1: label-flip attackers
S2: sign-flip Byzantine attackers
S3: free-riders (near-zero update)

Design choices are deterministic across methods for a fixed seed:
- same train/validation split
- same client partition
- same adversarial client identities
- same participant schedule
- same local mini-batch shuffle schedule
This enables paired comparisons across methods.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TensorState = Dict[str, torch.Tensor]


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deterministic_int_seed(*values: int) -> int:
    """Stable integer seed mixer (independent of Python's randomized hash())."""
    x = 0x9E3779B9
    for v in values:
        x ^= (int(v) + 0x9E3779B9 + ((x << 6) & 0xFFFFFFFF) + (x >> 2)) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
    return int(x)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MNISTSmallCNN(nn.Module):
    """Compact CNN chosen to make multi-seed FL simulation computationally feasible."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_model() -> nn.Module:
    return MNISTSmallCNN()


# -----------------------------------------------------------------------------
# Dataset handling
# -----------------------------------------------------------------------------
def load_mnist(root: str, val_size: int, seed: int, download: bool = True):
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    if not (0 < val_size < len(train_full)):
        raise ValueError(f"val_size must be between 1 and {len(train_full)-1}")

    rng = np.random.default_rng(deterministic_int_seed(seed, 101))
    perm = rng.permutation(len(train_full))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_full, train_idx, val_idx, test


def iid_partition(indices: np.ndarray, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(deterministic_int_seed(seed, 201))
    shuffled = np.array(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, n_clients)
    return [np.asarray(x, dtype=np.int64) for x in splits]


def dirichlet_partition(
    labels_full: np.ndarray,
    train_indices: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 20,
    max_attempts: int = 100,
) -> List[np.ndarray]:
    """Label-skew Dirichlet partition with a minimum client sample count.

    The partitioning is repeated until every client contains at least `min_size`
    samples or `max_attempts` is reached.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")

    labels = labels_full[train_indices]
    n_classes = int(labels_full.max()) + 1

    for attempt in range(max_attempts):
        rng = np.random.default_rng(deterministic_int_seed(seed, 301, attempt))
        client_bins: List[List[int]] = [[] for _ in range(n_clients)]

        for cls in range(n_classes):
            local_positions = np.where(labels == cls)[0]
            class_indices = train_indices[local_positions].copy()
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=np.float64))
            cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            chunks = np.split(class_indices, cuts)
            for cid, chunk in enumerate(chunks):
                client_bins[cid].extend(chunk.tolist())

        sizes = [len(x) for x in client_bins]
        if min(sizes) >= min_size:
            for x in client_bins:
                rng.shuffle(x)
            return [np.asarray(x, dtype=np.int64) for x in client_bins]

    raise RuntimeError(
        f"Could not create Dirichlet partition with min_size={min_size} after {max_attempts} attempts. "
        f"Try reducing min_size or increasing alpha."
    )


# -----------------------------------------------------------------------------
# State helpers
# -----------------------------------------------------------------------------
def clone_state_cpu(model: nn.Module) -> TensorState:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: TensorState) -> None:
    model.load_state_dict(state, strict=True)


def state_to_vector(state: TensorState) -> torch.Tensor:
    return torch.cat([state[k].reshape(-1) for k in state.keys()], dim=0)


def vector_to_state(vec: torch.Tensor, template: TensorState) -> TensorState:
    out: TensorState = {}
    offset = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vec[offset : offset + n].reshape(v.shape).clone()
        offset += n
    if offset != vec.numel():
        raise ValueError("Vector length does not match template state")
    return out


def subtract_state(local_state: TensorState, global_state: TensorState) -> TensorState:
    return {k: local_state[k] - global_state[k] for k in global_state.keys()}


def add_delta(global_state: TensorState, delta: TensorState) -> TensorState:
    return {k: global_state[k] + delta[k] for k in global_state.keys()}


def delta_norm(delta: TensorState) -> float:
    return float(torch.linalg.vector_norm(state_to_vector(delta)).item())


# -----------------------------------------------------------------------------
# Evaluation / training
# -----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


class Client:
    def __init__(self, cid: int, dataset: Subset, device: torch.device, experiment_seed: int) -> None:
        self.cid = int(cid)
        self.dataset = dataset
        self.device = device
        self.experiment_seed = int(experiment_seed)

    def train_local(
        self,
        global_state: TensorState,
        round_idx: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        attack: Optional[str],
    ) -> TensorState:
        model = make_model().to(self.device)
        load_state(model, global_state)

        loader_gen = torch.Generator()
        loader_gen.manual_seed(deterministic_int_seed(self.experiment_seed, round_idx, self.cid, 401))
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            generator=loader_gen,
        )

        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        model.train()
        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if attack == "label_flip":
                    y = (y + 1) % 10
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()

        local_state = clone_state_cpu(model)
        delta = subtract_state(local_state, global_state)

        if attack == "sign_flip":
            delta = {k: -v for k, v in delta.items()}
        elif attack == "free_zero":
            delta = {k: torch.zeros_like(v) for k, v in delta.items()}

        return delta


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def mean_aggregate(deltas: Sequence[TensorState], weights: Optional[Sequence[float]] = None) -> TensorState:
    if len(deltas) == 0:
        raise ValueError("No deltas to aggregate")

    if weights is None:
        w = torch.full((len(deltas),), 1.0 / len(deltas), dtype=torch.float32)
    else:
        w_np = np.asarray(weights, dtype=np.float64)
        if np.any(~np.isfinite(w_np)) or np.any(w_np < 0):
            raise ValueError("Aggregation weights must be finite and non-negative")
        if w_np.sum() <= 1e-12:
            w_np = np.ones_like(w_np)
        w_np = w_np / w_np.sum()
        w = torch.tensor(w_np, dtype=torch.float32)

    out: TensorState = {}
    for key in deltas[0].keys():
        stack = torch.stack([d[key] for d in deltas], dim=0)
        shape = [len(deltas)] + [1] * (stack.ndim - 1)
        out[key] = (stack * w.reshape(shape)).sum(dim=0)
    return out


def weighted_trimmed_mean(
    deltas: Sequence[TensorState],
    trim_ratio: float,
    weights: Optional[Sequence[float]] = None,
) -> TensorState:
    """Coordinate-wise trimmed mean; optionally trust-weighted after trimming.

    For each parameter coordinate, the smallest/largest k values are removed, where
    k=floor(trim_ratio*m). The remaining values are averaged. If weights are given,
    the corresponding client weights are carried through the coordinate-wise sort and
    renormalized among the untrimmed values. This preserves both robust trimming and
    trust weighting, unlike an implementation that would overwrite one with the other.
    """
    if len(deltas) == 0:
        raise ValueError("No deltas to aggregate")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must satisfy 0 <= trim_ratio < 0.5")

    template = deltas[0]
    vecs = torch.stack([state_to_vector(d) for d in deltas], dim=0)  # [m, p]
    m = vecs.shape[0]
    k = int(math.floor(trim_ratio * m))

    sorted_vals, sorted_idx = torch.sort(vecs, dim=0)
    lo, hi = k, m - k
    if hi <= lo:
        lo, hi = 0, m
    kept_vals = sorted_vals[lo:hi, :]

    if weights is None:
        agg_vec = kept_vals.mean(dim=0)
    else:
        w_np = np.asarray(weights, dtype=np.float64)
        if len(w_np) != m:
            raise ValueError("weights length must equal number of deltas")
        if np.any(~np.isfinite(w_np)) or np.any(w_np < 0):
            raise ValueError("weights must be finite and non-negative")
        if w_np.sum() <= 1e-12:
            w_np = np.ones_like(w_np)
        w = torch.tensor(w_np, dtype=vecs.dtype)
        sorted_w = w[sorted_idx]
        kept_w = sorted_w[lo:hi, :]
        denom = kept_w.sum(dim=0).clamp_min(1e-12)
        agg_vec = (kept_vals * kept_w).sum(dim=0) / denom

    return vector_to_state(agg_vec, template)


def coordinate_median(deltas: Sequence[TensorState]) -> TensorState:
    template = deltas[0]
    vecs = torch.stack([state_to_vector(d) for d in deltas], dim=0)
    med = torch.median(vecs, dim=0).values
    return vector_to_state(med, template)


# -----------------------------------------------------------------------------
# Trust model
# -----------------------------------------------------------------------------
@dataclass
class TrustParams:
    alpha: float = 0.40
    beta: float = 0.20
    gamma: float = 0.30
    delta: float = 0.10
    rho: float = 0.20
    window_L: int = 10
    decay_lambda: float = 0.02
    recovery_eta: float = 0.10
    similarity_threshold: float = 0.55
    norm_kappa: float = 1.5
    accuracy_outlier_z: float = 3.5
    initial_trust: float = 0.60
    initial_consistency: float = 0.50

    def validate(self) -> None:
        if not math.isclose(self.alpha + self.beta + self.gamma + self.delta, 1.0, abs_tol=1e-9):
            raise ValueError("Trust weights must sum to 1")
        if self.accuracy_outlier_z <= 0:
            raise ValueError("accuracy_outlier_z must be > 0")
        if self.window_L <= 0:
            raise ValueError("window_L must be > 0")


def robust_accuracy_scores(delta_acc_by_id: Dict[int, float]) -> Tuple[Dict[int, float], Dict[int, float], float, float]:
    """
    Convert raw candidate-model validation effects into a heterogeneity-aware utility score.

    In non-IID FL, an honest local update can temporarily reduce shared-proxy accuracy even
    when the aggregate update remains useful. An absolute rule such as delta_acc >= 0 can
    therefore reject all honest updates near convergence. We instead compare each update
    with its round peers using the median and median absolute deviation (MAD).

    Returns
    -------
    A_by_id : dict
        Relative validation-utility score in (0,1), obtained by applying a logistic map to
        the robust z-score. A value of 0.5 corresponds to the round median.
    z_by_id : dict
        Robust z-score for each submitted update.
    center : float
        Round median of raw validation effects.
    scale : float
        Robust scale estimate (1.4826*MAD), with a conservative fallback when MAD=0.
    """
    ids = list(delta_acc_by_id.keys())
    vals = np.asarray([delta_acc_by_id[cid] for cid in ids], dtype=np.float64)
    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        # A zero MAD can occur with a small number of clients or near convergence.
        # Fall back to sample dispersion, then to a small fixed floor in accuracy units.
        scale = max(float(np.std(vals, ddof=0)), 1e-3)

    A_by_id: Dict[int, float] = {}
    z_by_id: Dict[int, float] = {}
    for cid in ids:
        z = float((delta_acc_by_id[cid] - center) / scale)
        z_by_id[cid] = z
        z_clip = float(np.clip(z, -6.0, 6.0))
        A_by_id[cid] = float(1.0 / (1.0 + math.exp(-z_clip)))
    return A_by_id, z_by_id, center, scale


def cosine_similarity_state(a: TensorState, b: TensorState) -> float:
    av = state_to_vector(a)
    bv = state_to_vector(b)
    denom = float(torch.linalg.vector_norm(av).item() * torch.linalg.vector_norm(bv).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(av, bv).item() / denom)


def similarity_quality(delta: TensorState, ref: TensorState, median_norm: float, kappa: float) -> float:
    cos = float(np.clip(cosine_similarity_state(delta, ref), -1.0, 1.0))
    directional = (1.0 + cos) / 2.0
    ratio = delta_norm(delta) / max(median_norm, 1e-12)
    norm_penalty = math.exp(-max(0.0, ratio - kappa))
    return float(np.clip(directional * norm_penalty, 0.0, 1.0))


class TrustTracker:
    def __init__(self, n_clients: int, params: TrustParams) -> None:
        params.validate()
        self.params = params
        self.trust = np.full(n_clients, params.initial_trust, dtype=np.float64)
        self.consistency = np.full(n_clients, params.initial_consistency, dtype=np.float64)
        self.participation_hist = np.zeros((n_clients, params.window_L), dtype=np.int8)
        self.a_hist: List[deque] = [deque(maxlen=params.window_L) for _ in range(n_clients)]

    def begin_round(self, round_idx: int, admitted: Sequence[int]) -> None:
        """Correct one-step inactivity decay and update participation window."""
        col = (round_idx - 1) % self.params.window_L
        self.participation_hist[:, col] = 0
        self.participation_hist[list(admitted), col] = 1

        admitted_set = set(int(x) for x in admitted)
        decay = math.exp(-self.params.decay_lambda)
        for cid in range(len(self.trust)):
            if cid not in admitted_set:
                self.trust[cid] *= decay
        np.clip(self.trust, 0.0, 1.0, out=self.trust)

    def update_active(
        self,
        cid: int,
        A: float,
        D: float,
        accepted: bool,
    ) -> float:
        p = self.params
        history_mean = float(np.mean(self.a_hist[cid])) if len(self.a_hist[cid]) else A
        improvement = max(0.0, A - history_mean)

        consistency_observation = A if accepted else 0.0
        self.consistency[cid] = (1.0 - p.rho) * self.consistency[cid] + p.rho * consistency_observation
        U = float(self.participation_hist[cid].mean())

        base = p.alpha * A + p.beta * self.consistency[cid] + p.gamma * D + p.delta * U
        recovered = base + p.recovery_eta * (1.0 - base) * improvement
        self.trust[cid] = float(np.clip(recovered, 0.0, 1.0))
        self.a_hist[cid].append(float(A))
        return self.trust[cid]


# -----------------------------------------------------------------------------
# Experiment config
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    method: str = "C4"
    scenario: str = "S1"
    seed: int = 0
    dir_alpha: float = 0.5
    iid: bool = False
    attack_frac: float = 0.20
    n_clients: int = 50
    clients_per_round: int = 10
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    trim_ratio: float = 0.10
    val_size: int = 1000
    eval_every: int = 5
    data_root: str = "./data"
    device: str = "cuda"
    out_dir: str = "outputs_conference/run"
    download: bool = True

    def validate(self) -> None:
        if self.method not in {"C0", "C1", "C2", "C3", "C4"}:
            raise ValueError("method must be one of C0..C4")
        if self.scenario not in {"S0", "S1", "S2", "S3"}:
            raise ValueError("scenario must be one of S0..S3")
        if not (0 <= self.attack_frac < 1):
            raise ValueError("attack_frac must be in [0,1)")
        if self.clients_per_round > self.n_clients:
            raise ValueError("clients_per_round cannot exceed n_clients")
        if self.rounds <= 0:
            raise ValueError("rounds must be > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be > 0")


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


def scenario_attack(scenario: str, cid: int, adversarial_ids: set[int]) -> Optional[str]:
    if cid not in adversarial_ids:
        return None
    if scenario == "S1":
        return "label_flip"
    if scenario == "S2":
        return "sign_flip"
    if scenario == "S3":
        return "free_zero"
    return None


def make_participant_schedule(n_clients: int, k: int, rounds: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(deterministic_int_seed(seed, 501))
    return [rng.choice(n_clients, size=k, replace=False).astype(int).tolist() for _ in range(rounds)]


def choose_adversaries(n_clients: int, attack_frac: float, seed: int, scenario: str) -> set[int]:
    if scenario == "S0" or attack_frac <= 0:
        return set()
    n_adv = max(1, int(round(n_clients * attack_frac)))
    rng = np.random.default_rng(deterministic_int_seed(seed, 601))
    return set(rng.choice(n_clients, size=n_adv, replace=False).astype(int).tolist())


def run_experiment(cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg.validate()
    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    trust_params = TrustParams()

    train_full, train_idx, val_idx, test_ds = load_mnist(
        root=cfg.data_root,
        val_size=cfg.val_size,
        seed=cfg.seed,
        download=cfg.download,
    )
    labels_full = np.asarray(train_full.targets)

    if cfg.iid:
        partitions = iid_partition(train_idx, cfg.n_clients, cfg.seed)
        partition_name = "iid"
    else:
        partitions = dirichlet_partition(labels_full, train_idx, cfg.n_clients, cfg.dir_alpha, cfg.seed)
        partition_name = f"dir{cfg.dir_alpha:g}"

    clients = [
        Client(cid, Subset(train_full, partitions[cid].tolist()), device=device, experiment_seed=cfg.seed)
        for cid in range(cfg.n_clients)
    ]

    val_loader = DataLoader(
        Subset(train_full, val_idx.tolist()),
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    adversarial_ids = choose_adversaries(cfg.n_clients, cfg.attack_frac, cfg.seed, cfg.scenario)
    participant_schedule = make_participant_schedule(cfg.n_clients, cfg.clients_per_round, cfg.rounds, cfg.seed)

    global_model = make_model().to(device)
    global_state = clone_state_cpu(global_model)
    tracker = TrustTracker(cfg.n_clients, trust_params)

    round_rows: List[dict] = []
    client_rows: List[dict] = []
    start_time = time.perf_counter()

    for round_idx in range(1, cfg.rounds + 1):
        round_start = time.perf_counter()
        admitted = participant_schedule[round_idx - 1]
        tracker.begin_round(round_idx, admitted)

        deltas_by_id: Dict[int, TensorState] = {}
        for cid in admitted:
            deltas_by_id[cid] = clients[cid].train_local(
                global_state=global_state,
                round_idx=round_idx,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                momentum=cfg.momentum,
                attack=scenario_attack(cfg.scenario, cid, adversarial_ids),
            )

        submitted_deltas = [deltas_by_id[cid] for cid in admitted]
        ref = coordinate_median(submitted_deltas)
        norms = [delta_norm(d) for d in submitted_deltas]
        median_norm = float(np.median(norms)) if norms else 1.0

        load_state(global_model, global_state)
        base_val_acc = evaluate_accuracy(global_model, val_loader, device)

        # First compute the raw validation effect and similarity for all submitted updates.
        raw_delta_acc: Dict[int, float] = {}
        raw_D: Dict[int, float] = {}
        for cid in admitted:
            cand_state = add_delta(global_state, deltas_by_id[cid])
            load_state(global_model, cand_state)
            cand_acc = evaluate_accuracy(global_model, val_loader, device)
            raw_delta_acc[cid] = float(cand_acc - base_val_acc)
            raw_D[cid] = similarity_quality(
                deltas_by_id[cid], ref, median_norm, trust_params.norm_kappa
            )

        # Heterogeneity-aware relative utility. The median/MAD remain robust while fewer than
        # half of the admitted clients are adversarial. This avoids the false-rejection failure
        # of the original absolute delta_acc >= 0 screening rule under non-IID data.
        A_by_id, utility_z_by_id, utility_center, utility_scale = robust_accuracy_scores(raw_delta_acc)

        metrics: Dict[int, dict] = {}
        for cid in admitted:
            metrics[cid] = {
                "delta_acc": raw_delta_acc[cid],
                "A": A_by_id[cid],
                "utility_z": utility_z_by_id[cid],
                "D": raw_D[cid],
            }

        # Screening is intentionally exclusive to C4 for clean component isolation.
        # An update is rejected only when its validation effect is an extreme lower-tail
        # outlier relative to the current round, or when its direction/norm quality is too low.
        accepted: List[int] = []
        for cid in admitted:
            if cfg.method == "C4":
                ok = (
                    metrics[cid]["utility_z"] >= -trust_params.accuracy_outlier_z
                    and metrics[cid]["D"] >= trust_params.similarity_threshold
                )
            else:
                ok = True
            if ok:
                accepted.append(cid)

        # Trust/reputation update. C2 is static per-round A only; C3/C4 use dynamic tracker.
        current_scores: Dict[int, float] = {}
        if cfg.method == "C2":
            for cid in admitted:
                current_scores[cid] = metrics[cid]["A"]
        elif cfg.method in {"C3", "C4"}:
            accepted_set = set(accepted)
            for cid in admitted:
                current_scores[cid] = tracker.update_active(
                    cid=cid,
                    A=metrics[cid]["A"],
                    D=metrics[cid]["D"],
                    accepted=(cid in accepted_set),
                )

        # Aggregation
        if len(accepted) > 0:
            accepted_deltas = [deltas_by_id[cid] for cid in accepted]
            if cfg.method == "C0":
                agg_delta = mean_aggregate(accepted_deltas)
            elif cfg.method == "C1":
                agg_delta = weighted_trimmed_mean(accepted_deltas, cfg.trim_ratio)
            elif cfg.method == "C2":
                agg_delta = mean_aggregate(accepted_deltas, [current_scores[cid] for cid in accepted])
            elif cfg.method == "C3":
                agg_delta = mean_aggregate(accepted_deltas, [current_scores[cid] for cid in accepted])
            elif cfg.method == "C4":
                agg_delta = weighted_trimmed_mean(
                    accepted_deltas,
                    cfg.trim_ratio,
                    [current_scores[cid] for cid in accepted],
                )
            else:  # defensive
                raise RuntimeError("Unknown method")
            global_state = add_delta(global_state, agg_delta)

        # Global evaluation
        load_state(global_model, global_state)
        global_val_acc = evaluate_accuracy(global_model, val_loader, device)
        do_test = (round_idx == 1) or (round_idx % cfg.eval_every == 0) or (round_idx == cfg.rounds)
        test_acc = evaluate_accuracy(global_model, test_loader, device) if do_test else np.nan

        admitted_adv = [cid for cid in admitted if cid in adversarial_ids]
        admitted_ben = [cid for cid in admitted if cid not in adversarial_ids]
        accepted_adv = [cid for cid in accepted if cid in adversarial_ids]
        accepted_ben = [cid for cid in accepted if cid not in adversarial_ids]

        dynamic_trust_enabled = cfg.method in {"C3", "C4"}
        benign_ids = [cid for cid in range(cfg.n_clients) if cid not in adversarial_ids]
        if dynamic_trust_enabled:
            trust_adv = tracker.trust[list(adversarial_ids)] if adversarial_ids else np.array([], dtype=float)
            trust_ben = tracker.trust[benign_ids]
            mean_trust_adv = float(np.mean(trust_adv)) if trust_adv.size else np.nan
            mean_trust_benign = float(np.mean(trust_ben)) if trust_ben.size else np.nan
            mean_trust_all = float(np.mean(tracker.trust))
        else:
            mean_trust_adv = np.nan
            mean_trust_benign = np.nan
            mean_trust_all = np.nan

        round_rows.append(
            {
                "seed": cfg.seed,
                "method": cfg.method,
                "scenario": cfg.scenario,
                "partition": partition_name,
                "dir_alpha": np.nan if cfg.iid else cfg.dir_alpha,
                "attack_frac": cfg.attack_frac if cfg.scenario != "S0" else 0.0,
                "round": round_idx,
                "base_val_acc": base_val_acc,
                "utility_center": utility_center,
                "utility_scale": utility_scale,
                "global_val_acc": global_val_acc,
                "test_acc": test_acc,
                "admitted": len(admitted),
                "accepted": len(accepted),
                "adv_admitted": len(admitted_adv),
                "adv_accepted": len(accepted_adv),
                "benign_admitted": len(admitted_ben),
                "benign_accepted": len(accepted_ben),
                "adv_accept_rate": len(accepted_adv) / len(admitted_adv) if admitted_adv else np.nan,
                "benign_accept_rate": len(accepted_ben) / len(admitted_ben) if admitted_ben else np.nan,
                "mean_trust_adv": mean_trust_adv,
                "mean_trust_benign": mean_trust_benign,
                "mean_trust_all": mean_trust_all,
                "round_runtime_sec": time.perf_counter() - round_start,
            }
        )

        # Log a full trust snapshot every round. Metrics that require a submitted update
        # are NaN for non-selected clients. This avoids selection bias in trust-distribution
        # analyses while retaining one compact client_metrics.csv file per run.
        accepted_set = set(accepted)
        admitted_set = set(admitted)
        norm_by_id = {cid: norms[pos] for pos, cid in enumerate(admitted)}
        for cid in range(cfg.n_clients):
            selected = cid in admitted_set
            if selected:
                m = metrics[cid]
                accepted_value = int(cid in accepted_set)
                static_score = current_scores.get(cid, np.nan)
            else:
                m = {"delta_acc": np.nan, "A": np.nan, "utility_z": np.nan, "D": np.nan}
                accepted_value = np.nan
                static_score = np.nan

            if cfg.method in {"C3", "C4"}:
                trust_value = tracker.trust[cid]
                c_value = tracker.consistency[cid]
                u_value = tracker.participation_hist[cid].mean()
            elif cfg.method == "C2" and selected:
                trust_value = static_score
                c_value = np.nan
                u_value = np.nan
            else:
                trust_value = np.nan
                c_value = np.nan
                u_value = np.nan

            client_rows.append(
                {
                    "seed": cfg.seed,
                    "method": cfg.method,
                    "scenario": cfg.scenario,
                    "partition": partition_name,
                    "round": round_idx,
                    "client_id": cid,
                    "is_adversarial": int(cid in adversarial_ids),
                    "attack": scenario_attack(cfg.scenario, cid, adversarial_ids) or "none",
                    "selected": int(selected),
                    "accepted": accepted_value,
                    "delta_acc": m["delta_acc"],
                    "A": m["A"],
                    "utility_z": m.get("utility_z", np.nan),
                    "D": m["D"],
                    "C": c_value,
                    "U": u_value,
                    "trust": trust_value,
                    "update_norm": norm_by_id.get(cid, np.nan),
                }
            )

        if round_idx == 1 or round_idx % 10 == 0 or round_idx == cfg.rounds:
            elapsed = time.perf_counter() - start_time
            latest_test = "NA" if np.isnan(test_acc) else f"{test_acc:.4f}"
            print(
                f"[{cfg.method} {cfg.scenario} {partition_name} seed={cfg.seed}] "
                f"round {round_idx:3d}/{cfg.rounds} val={global_val_acc:.4f} test={latest_test} "
                f"accepted={len(accepted)}/{len(admitted)} elapsed={elapsed/60:.1f} min"
            )

    round_df = pd.DataFrame(round_rows)
    client_df = pd.DataFrame(client_rows)

    metadata = {
        "config": asdict(cfg),
        "trust_params": asdict(trust_params),
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": "MNISTSmallCNN(Conv16-Conv32-FC64-FC10)",
        "model_parameters": sum(p.numel() for p in make_model().parameters()),
        "adversarial_client_ids": sorted(adversarial_ids),
        "partition_sizes": [int(len(x)) for x in partitions],
        "participant_schedule_seed_rule": "deterministic_int_seed(seed,501)",
        "total_runtime_sec": time.perf_counter() - start_time,
        "notes": (
            "C4 evaluates the off-chain trust-guided screening/aggregation core only. "
            "Rewards, slashing, and blockchain execution are outside this conference experiment."
        ),
    }
    return round_df, client_df, metadata


def save_outputs(round_df: pd.DataFrame, client_df: pd.DataFrame, metadata: dict, out_dir: str) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    round_df.to_csv(path / "round_metrics.csv", index=False)
    client_df.to_csv(path / "client_metrics.csv", index=False)
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved results to: {path.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one ICBTA 2026 SDFL experiment")
    p.add_argument("--method", default="C4", choices=["C0", "C1", "C2", "C3", "C4"])
    p.add_argument("--scenario", default="S1", choices=["S0", "S1", "S2", "S3"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir_alpha", type=float, default=0.5)
    p.add_argument("--iid", action="store_true")
    p.add_argument("--attack_frac", type=float, default=0.20)
    p.add_argument("--n_clients", type=int, default=50)
    p.add_argument("--clients_per_round", type=int, default=10)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--trim_ratio", type=float, default=0.10)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="outputs_conference/run")
    p.add_argument("--no_download", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ExperimentConfig(
        method=args.method,
        scenario=args.scenario,
        seed=args.seed,
        dir_alpha=args.dir_alpha,
        iid=args.iid,
        attack_frac=args.attack_frac,
        n_clients=args.n_clients,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        trim_ratio=args.trim_ratio,
        val_size=args.val_size,
        eval_every=args.eval_every,
        data_root=args.data_root,
        device=args.device,
        out_dir=args.out_dir,
        download=not args.no_download,
    )
    round_df, client_df, metadata = run_experiment(cfg)
    save_outputs(round_df, client_df, metadata, cfg.out_dir)


if __name__ == "__main__":
    main()
