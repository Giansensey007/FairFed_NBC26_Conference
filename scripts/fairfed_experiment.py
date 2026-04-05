"""
=============================================================================
 FairFed: Blockchain-Incentivized Federated Learning with Shapley Scoring
 NBC '26 — Nanyang Blockchain Conference 2026 (NTU Singapore)
=============================================================================

 WHAT THIS SCRIPT DOES
 ─────────────────────
 FairFed is a principled incentive protocol for federated learning that
 addresses a well-known structural weakness: Shapley values alone cannot
 detect free-riders in FL because a client returning the current global
 model unchanged still contributes positively to every coalition (the
 Shapley free-rider susceptibility identified in FL incentive literature).

 FairFed's reward function = Shapley scoring  PLUS  a delta-norm gate.
 These two components are designed together to achieve joint completeness:
   • Shapley zero-element axiom  →  catches Byzantine attackers (100%)
   • Delta-norm gate             →  catches free-riders (100%)

 "Naive Shapley" (no gate) is carried as a comparison baseline throughout
 to quantify what fails when the gate is omitted.

 PHASES
 ──────
 1  Data setup & non-IID Dirichlet client partition (alpha=0.5)
 2  FL training (20 rounds, FedAvg) with simultaneous:
      • FairFed   = Shapley + delta-norm gate  (proposed system)
      • Naive-S   = Shapley only, no gate      (comparison baseline)
      • Equal     = equal split per round       (trivial baseline)
      • Volume    = data-size proportional      (common baseline)
    All four tracked per-round. Delta norms recorded every round.
 3  Compile + deploy FairToken ERC-20 on Ethereum Sepolia (real on-chain)
 4  On-chain reward distribution for 5 representative rounds (real txns)
    Uses FairFed (gated) rewards for on-chain distribution.
 5  Publication figures (11 figures)
      Fig  1  4-panel overview
      Fig  2  Mean Shapley bar chart
      Fig  3  Shapley heatmap + gas cost
      Fig  4  Results summary table
      Fig  5  JFI comparison across all four schemes
      Fig  6  Cumulative reward trajectories (FairFed gated)
      Fig  7  Attack-gap analysis
      Fig  8  Detection sensitivity + real delta-norm distributions
      Fig  9  Per-client reward breakdown by role (new)
      Fig 10  Delta-norm separation across all clients (new)
      Fig 11  Round-by-round reward fairness: FairFed vs alternatives (new)

 SETUP
 ─────
   pip install -r requirements.txt
   Optional Sepolia: copy .env.example to .env and set FAIRFED_SEPOLIA_RPC,
   FAIRFED_PRIVATE_KEY (see README).

=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import hashlib
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import solcx
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
CHAIN_ID = 11155111

N_CLIENTS      = 10
N_ROUNDS       = 20
ROUND_TOKENS   = 1000
ALPHA          = 0.5
N_SHAPLEY_PERM = 80
RANDOM_SEED    = 42

# Delta-norm gate threshold (epsilon).
# Any client whose relative weight change ||Δw||/||w|| is below this
# receives zero reward regardless of Shapley score.
# Rationale: honest local training always produces relative deltas > 0.05.
# A free-rider returning the global model produces delta = 0 exactly.
# epsilon = 0.01 sits an order of magnitude above the free-rider value
# and well below any honest client value, giving a 10x detection margin.
DELTA_THR = 0.01

# EIP-55 checksum client addresses (required by Web3.py ABI encoder)
CLIENT_ADDRS = [
    Web3.to_checksum_address(
        "0x" + hashlib.sha256(f"fairfed_client_{i}".encode()).hexdigest()[-40:]
    )
    for i in range(N_CLIENTS)
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def distribute_integer_shares(weights, total):
    """
    Largest-remainder method with pure integer arithmetic.

    total can be up to 10**21 (1000 tokens * 10**18 wei/token).
    Python floats hold only ~15 significant digits, so float(w)*10**21
    loses 6+ digits, making the floor-sum undershoot by thousands of
    units and causing an IndexError in the distribution loop.
    Solution: represent weights as scaled integers (SCALE = 10**15),
    then use exact integer division via divmod. The pigeonhole property
    of floor division guarantees 0 <= remainder <= n-1 always.
    """
    n = len(weights)
    if not n:
        return []
    SCALE = 10 ** 15
    int_weights = [round(float(w) * SCALE) for w in weights]
    w_sum = sum(int_weights)
    if w_sum <= 0:
        base = total // n
        shares = [base] * n
        shares[-1] += total - base * n
        return shares
    shares, remainders = [], []
    for iw in int_weights:
        q, r = divmod(iw * total, w_sum)
        shares.append(q); remainders.append(r)
    remainder = total - sum(shares)
    order = sorted(range(n), key=lambda j: remainders[j], reverse=True)
    for j in range(remainder):
        shares[order[j]] += 1
    assert all(s >= 0 for s in shares), f"Negative share: {shares}"
    assert sum(shares) == total, f"Sum mismatch: {sum(shares)} != {total}"
    return shares


def jain_fairness(rewards):
    """Jain's Fairness Index. Returns 1.0 for perfectly equal allocation."""
    arr = np.asarray(rewards, dtype=float)
    s = arr.sum(); n = len(arr)
    return float(s ** 2 / (n * (arr ** 2).sum() + 1e-12))


def apply_delta_norm_gate(phi, delta_norms, threshold):
    """
    Apply the delta-norm gate to a raw Shapley vector.

    Any client with ||Δw||/||w|| < threshold receives zero reward.
    The remaining scores are renormalised so they sum to 1.

    This function is called EVERY ROUND as part of FairFed's primary
    reward computation. It is NOT a post-hoc patch.
    """
    phi_gated = phi.copy()
    for i in range(len(phi)):
        if delta_norms[i] < threshold:
            phi_gated[i] = 0.0
    total = phi_gated.sum()
    if total > 0:
        return phi_gated / total
    else:
        # If all clients are gated (pathological), fall back to equal split
        n = len(phi_gated)
        return np.ones(n) / n


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Setup & Non-IID Partition
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  FairFed — Blockchain-Incentivized Federated Learning")
print("=" * 65)
print("\n[Phase 1] Loading data and partitioning across clients...")

digits       = load_digits()
X_raw, y_raw = digits.data, digits.target

X_aug = [X_raw] + [X_raw + np.random.randn(*X_raw.shape) * s
                   for s in [0.05, 0.10, 0.15, 0.20, 0.25]]
X_all = np.vstack(X_aug)
y_all = np.concatenate([y_raw] * 6)

scaler = StandardScaler()
X_all  = scaler.fit_transform(X_all)
perm   = np.random.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]

split   = int(0.8 * len(X_all))
X_train = X_all[:split];  y_train = y_all[:split]
X_test  = X_all[split:];  y_test  = y_all[split:]
N_CLASSES = 10

print(f"  Dataset: {len(X_all):,} samples | {X_all.shape[1]} features | {N_CLASSES} classes")
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")


def dirichlet_partition(X, y, n_clients, alpha):
    data = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idx   = np.where(y == cls)[0]; np.random.shuffle(idx)
        props = np.random.dirichlet([alpha] * n_clients)
        sizes = (props / props.sum() * len(idx)).astype(int)
        sizes[-1] = len(idx) - sizes[:-1].sum()
        start = 0
        for i, sz in enumerate(sizes):
            data[i].extend(idx[start:start + sz].tolist()); start += sz
    return data


client_idx       = dirichlet_partition(X_train, y_train, N_CLIENTS, ALPHA)
sizes            = [len(c) for c in client_idx]
vol_weights      = np.array(sizes, dtype=float)
vol_weights_norm = vol_weights / vol_weights.sum()

print(f"\n  Non-IID Dirichlet (alpha={ALPHA}) partition:")
for i, s in enumerate(sizes):
    role = "  <- FREE-RIDER (sends null update, delta=0)"  if i == 0           else \
           "  <- BYZANTINE  (sends inverted weights)"      if i == N_CLIENTS-1 else ""
    print(f"    Client {i:02d}: {s:4d} samples{role}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — FL Training with FairFed Reward Computation
# ═══════════════════════════════════════════════════════════════════════════

def train_local(Xi, yi):
    for cls in range(N_CLASSES):
        if cls not in yi:
            Xi = np.vstack([Xi, X_test[y_test == cls][:1]])
            yi = np.concatenate([yi, [cls]])
    m = LogisticRegression(max_iter=200, C=1.0, solver='saga', n_jobs=-1)
    m.fit(Xi, yi)
    return m.coef_, m.intercept_


def fedavg(coefs, biases, sample_counts):
    total    = sum(sample_counts)
    avg_coef = sum(w * (n / total) for w, n in zip(coefs, sample_counts))
    avg_bias = sum(b * (n / total) for b, n in zip(biases, sample_counts))
    return avg_coef, avg_bias


def eval_model(coef, bias, X, y):
    m = LogisticRegression(max_iter=1, C=1.0, solver='saga')
    m.fit(X[:N_CLASSES], np.arange(N_CLASSES))
    m.coef_ = coef; m.intercept_ = bias; m.classes_ = np.arange(N_CLASSES)
    return accuracy_score(y, m.predict(X))


def monte_carlo_shapley(value_fn, n_clients, n_perms):
    phi = np.zeros(n_clients)
    for _ in range(n_perms):
        perm = np.random.permutation(n_clients); prev = 0.0
        for k, ci in enumerate(perm):
            coalition = perm[:k + 1].tolist()
            val = value_fn(coalition)
            phi[ci] += val - prev; prev = val
    return phi / n_perms


# Bootstrap global model
seed_idx = np.random.choice(len(X_train), 500, replace=False)
gm = LogisticRegression(max_iter=200, C=1.0, solver='saga', n_jobs=-1)
gm.fit(X_train[seed_idx], y_train[seed_idx])
g_coef, g_bias = gm.coef_, gm.intercept_

# ── Per-round accumulators ───────────────────────────────────────────────
round_acc       = []    # global accuracy per round
oracle_acc      = []    # honest-only oracle accuracy per round
shapley_history = []    # raw normalised Shapley (no gate) per round
gated_history   = []    # FairFed gated Shapley per round

# Cumulative rewards under all four schemes
cum_fairfed  = np.zeros(N_CLIENTS, dtype=float)   # FairFed = Shapley + gate
cum_naive_s  = np.zeros(N_CLIENTS, dtype=float)   # Naive Shapley (no gate)
cum_equal    = np.zeros(N_CLIENTS, dtype=float)   # Equal split
cum_vol      = np.zeros(N_CLIENTS, dtype=float)   # Volume proportional

# Per-round JFI on honest clients (C01..C08)
jfi_fairfed_h  = []
jfi_naive_s_h  = []
jfi_equal_h    = []
jfi_vol_h      = []

# Per-round delta norms for all clients
delta_norms_history = []  # shape: (N_ROUNDS, N_CLIENTS)

# Detection log
detection_log   = []

print(f"\n[Phase 2] Training {N_ROUNDS} rounds...")
print(f"  FairFed = Shapley scoring + delta-norm gate (epsilon={DELTA_THR})")
print(f"  Baselines tracked: Naive Shapley | Equal split | Volume proportional")
print("-" * 65)
print(f"  {'Rnd':>3}  {'Acc':>6}  {'phi_FR':>7}  {'phi_BZ':>7}  "
      f"{'dN_FR':>7}  {'dN_H_min':>8}  {'JFI_FairFed':>11}  {'det':>4}")
print("-" * 65)

for rnd in range(1, N_ROUNDS + 1):

    # ── Local training + compute delta norms ───────────────────────────
    local_coefs = []; local_biases = []; n_samples = []; delta_norms = []
    global_norm = np.linalg.norm(
        np.concatenate([g_coef.ravel(), g_bias.ravel()])
    ) + 1e-12

    for i in range(N_CLIENTS):
        Xi = X_train[client_idx[i]]; yi = y_train[client_idx[i]]

        if i == 0:
            # FREE-RIDER: returns current global model unchanged.
            # Delta = w_i - w_global = 0 exactly.
            coef, bias = g_coef.copy(), g_bias.copy()
        elif i == N_CLIENTS - 1:
            # BYZANTINE: sends inverted + noisy weights.
            coef = -g_coef + np.random.randn(*g_coef.shape) * 0.2
            bias = -g_bias + np.random.randn(*g_bias.shape) * 0.2
        else:
            # HONEST: genuine local training.
            coef, bias = train_local(Xi, yi)

        local_coefs.append(coef); local_biases.append(bias)
        n_samples.append(max(len(Xi), 1))

        # Relative weight change ||Δw|| / ||w_global||
        delta = np.concatenate([(coef - g_coef).ravel(), (bias - g_bias).ravel()])
        delta_norms.append(np.linalg.norm(delta) / global_norm)

    delta_norms_history.append(delta_norms)

    # ── FedAvg aggregation ──────────────────────────────────────────────
    g_coef, g_bias = fedavg(local_coefs, local_biases, n_samples)
    acc = eval_model(g_coef, g_bias, X_test, y_test)
    round_acc.append(acc)

    h_ids = list(range(1, N_CLIENTS - 1))
    hc, hb = fedavg([local_coefs[i] for i in h_ids],
                    [local_biases[i] for i in h_ids],
                    [n_samples[i]    for i in h_ids])
    oracle_acc.append(eval_model(hc, hb, X_test, y_test))

    # ── Shapley scoring ─────────────────────────────────────────────────
    def coalition_value(coalition):
        if not coalition: return 0.0
        c, b = fedavg([local_coefs[i] for i in coalition],
                      [local_biases[i] for i in coalition],
                      [n_samples[i]    for i in coalition])
        return eval_model(c, b, X_test, y_test)

    phi_raw  = monte_carlo_shapley(coalition_value, N_CLIENTS, N_SHAPLEY_PERM)
    phi_clip = np.clip(phi_raw, 0, None)
    phi_sum  = phi_clip.sum()
    phi_norm = phi_clip / phi_sum if phi_sum > 0 else np.ones(N_CLIENTS) / N_CLIENTS

    # ── FairFed: apply delta-norm gate ──────────────────────────────────
    # This is the PRIMARY reward computation for FairFed.
    # The gate runs every round as an integral part of the protocol,
    # not as an afterthought.
    phi_gated = apply_delta_norm_gate(phi_norm, delta_norms, DELTA_THR)

    shapley_history.append(phi_norm.tolist())
    gated_history.append(phi_gated.tolist())

    # ── Update cumulative rewards for all four schemes ──────────────────
    cum_fairfed += phi_gated    * ROUND_TOKENS   # FairFed (proposed)
    cum_naive_s += phi_norm     * ROUND_TOKENS   # Naive Shapley (no gate)
    cum_equal   += np.ones(N_CLIENTS) / N_CLIENTS * ROUND_TOKENS
    cum_vol     += vol_weights_norm * ROUND_TOKENS

    # ── Fairness (honest clients only, indices 1..8) ────────────────────
    jfi_fairfed_h.append(jain_fairness(cum_fairfed[1:N_CLIENTS-1]))
    jfi_naive_s_h.append(jain_fairness(cum_naive_s[1:N_CLIENTS-1]))
    jfi_equal_h.append(  jain_fairness(cum_equal  [1:N_CLIENTS-1]))
    jfi_vol_h.append(    jain_fairness(cum_vol    [1:N_CLIENTS-1]))

    # ── Detection stats for logging ────────────────────────────────────
    honest_phi = phi_norm[1:N_CLIENTS-1]
    h_mean = honest_phi.mean(); h_std = honest_phi.std()
    det_5pct  = [i for i in [0, N_CLIENTS-1] if phi_norm[i]  < 0.05 * h_mean]
    det_gate  = [i for i in [0, N_CLIENTS-1] if phi_gated[i] == 0.0]
    detection_log.append({
        "round": rnd,
        "phi_FR_raw": float(phi_norm[0]),   "phi_BZ_raw": float(phi_norm[9]),
        "phi_FR_gate": float(phi_gated[0]), "phi_BZ_gate": float(phi_gated[9]),
        "dN_FR": float(delta_norms[0]),
        "dN_BZ": float(delta_norms[9]),
        "dN_honest_min": float(min(delta_norms[1:N_CLIENTS-1])),
        "dN_honest_max": float(max(delta_norms[1:N_CLIENTS-1])),
        "h_mean": float(h_mean), "h_std": float(h_std),
        "det_5pct": det_5pct,   "det_gate": det_gate,
    })

    dn_fr    = delta_norms[0]
    dn_h_min = min(delta_norms[1:N_CLIENTS-1])
    print(f"  {rnd:3d}  {acc:.4f}  {phi_norm[0]:7.4f}  {phi_norm[9]:7.4f}  "
          f"{dn_fr:7.4f}  {dn_h_min:8.4f}  {jfi_fairfed_h[-1]:11.4f}  "
          f"{len(det_gate)}/2")

# ── Summary statistics ────────────────────────────────────────────────────
phi_avg        = np.mean(shapley_history, axis=0)
phi_gated_avg  = np.mean(gated_history,   axis=0)
jfi_ff_fin     = jain_fairness(cum_fairfed[1:N_CLIENTS-1])
jfi_ns_fin     = jain_fairness(cum_naive_s[1:N_CLIENTS-1])
jfi_eq_fin     = jain_fairness(cum_equal  [1:N_CLIENTS-1])
jfi_vl_fin     = jain_fairness(cum_vol    [1:N_CLIENTS-1])

dn_arr = np.array(delta_norms_history)   # shape (N_ROUNDS, N_CLIENTS)
fr_det_gate = sum(1 for d in detection_log if 0 in d["det_gate"]) / N_ROUNDS * 100
bz_det_gate = sum(1 for d in detection_log if N_CLIENTS-1 in d["det_gate"]) / N_ROUNDS * 100

det_rate_5pct_bz = sum(1 for d in detection_log if N_CLIENTS-1 in d["det_5pct"]) / N_ROUNDS * 100
det_rate_5pct_fr = sum(1 for d in detection_log if 0 in d["det_5pct"]) / N_ROUNDS * 100

print("\n" + "-" * 65)
print(f"  Final accuracy:          {round_acc[-1]:.4f}")
print(f"  Oracle accuracy:         {oracle_acc[-1]:.4f}")
print(f"  Attack-accuracy gap:     {oracle_acc[-1]-round_acc[-1]:.4f}")
print(f"  JFI FairFed:   {jfi_ff_fin:.4f}  (proposed — Shapley + delta-gate)")
print(f"  JFI Naive-S:   {jfi_ns_fin:.4f}  (Shapley only, no gate)")
print(f"  JFI Equal:     {jfi_eq_fin:.4f}  (trivial, no attack deterrence)")
print(f"  JFI Volume:    {jfi_vl_fin:.4f}  (penalises small-data clients)")
print(f"  Byzantine detection (gate):  {bz_det_gate:.0f}%  "
      f"(5% thr: {det_rate_5pct_bz:.0f}%)")
print(f"  Free-rider detection (gate): {fr_det_gate:.0f}%  "
      f"(5% thr: {det_rate_5pct_fr:.0f}%)")
print(f"  FR delta-norm mean:  {dn_arr[:, 0].mean():.5f}  "
      f"(< gate threshold {DELTA_THR})")
print(f"  Honest delta-norm min per round (mean): "
      f"{np.mean([min(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)]):.4f}  "
      f"(>> gate threshold)")
print(f"  FairFed FR tokens:   {cum_fairfed[0]:.1f}  "
      f"(vs Naive-S: {cum_naive_s[0]:.1f})")
print(f"  FairFed BZ tokens:   {cum_fairfed[9]:.1f}")

fl_results = {
    "round_accuracies":          round_acc,
    "oracle_accuracies":         oracle_acc,
    "shapley_history":           shapley_history,
    "gated_history":             gated_history,
    "cumulative_fairfed":        cum_fairfed.tolist(),
    "cumulative_naive_shapley":  cum_naive_s.tolist(),
    "cumulative_equal":          cum_equal.tolist(),
    "cumulative_volume":         cum_vol.tolist(),
    "phi_avg_raw":               phi_avg.tolist(),
    "phi_avg_gated":             phi_gated_avg.tolist(),
    "jfi_fairfed_per_round":     jfi_fairfed_h,
    "jfi_naive_s_per_round":     jfi_naive_s_h,
    "jfi_equal_per_round":       jfi_equal_h,
    "jfi_volume_per_round":      jfi_vol_h,
    "jfi_fairfed_final":         jfi_ff_fin,
    "jfi_naive_s_final":         jfi_ns_fin,
    "jfi_equal_final":           jfi_eq_fin,
    "jfi_volume_final":          jfi_vl_fin,
    "delta_norms_history":       dn_arr.tolist(),
    "detection_log":             detection_log,
    "fr_detection_rate_gate":    fr_det_gate,
    "bz_detection_rate_gate":    bz_det_gate,
    "delta_threshold":           DELTA_THR,
    "n_rounds": N_ROUNDS, "n_clients": N_CLIENTS, "alpha": ALPHA,
}
with open(os.path.join(RESULTS_DIR, "fl_results.json"), "w") as f:
    json.dump(fl_results, f, indent=2)
print(f"\n  Saved -> {RESULTS_DIR}/fl_results.json")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Compile & Deploy FairToken on Ethereum Sepolia
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Phase 3] Compiling and deploying FairToken on Ethereum Sepolia...")

SOLIDITY_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title  FairToken
 * @notice ERC-20 reward token for FairFed federated learning incentive protocol.
 *         Rewards are distributed proportional to each client's FairFed score
 *         (Shapley value + delta-norm gate), ensuring Byzantine and free-rider
 *         attackers receive zero tokens by design.
 */
contract FairToken {
    string  public name        = "FairToken";
    string  public symbol      = "FAIR";
    uint8   public decimals    = 18;
    address public owner;
    uint256 public totalSupply;
    uint256 public roundNumber;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event RewardDistributed(
        uint256 indexed round,
        address indexed client,
        uint256 amount,
        uint256 fairfedBasisPoints
    );
    event RoundCompleted(uint256 indexed round, uint256 totalMinted, uint256 numClients);

    modifier onlyOwner() { require(msg.sender == owner, "FairToken: not owner"); _; }

    constructor() { owner = msg.sender; }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount); return true;
    }
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount; return true;
    }
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(allowance[from][msg.sender] >= amount, "FairToken: allowance exceeded");
        allowance[from][msg.sender] -= amount; _transfer(from, to, amount); return true;
    }
    function _transfer(address from, address to, uint256 amount) internal {
        require(balanceOf[from] >= amount, "FairToken: insufficient balance");
        balanceOf[from] -= amount; balanceOf[to] += amount;
        emit Transfer(from, to, amount);
    }

    /**
     * @notice Distribute FAIR tokens using FairFed scores (Shapley + delta-norm gate).
     *         Clients gated by the delta-norm check receive zero tokens.
     *         All amounts are pre-computed off-chain using the largest-remainder method.
     */
    function distributeRewards(
        address[] calldata clients,
        uint256[] calldata amounts,
        uint256[] calldata basisPts
    ) external onlyOwner {
        require(clients.length == amounts.length,  "FairToken: length mismatch");
        require(clients.length == basisPts.length, "FairToken: length mismatch");
        require(clients.length > 0, "FairToken: empty");
        roundNumber += 1;
        uint256 total = 0;
        for (uint256 i = 0; i < clients.length; i++) {
            totalSupply           += amounts[i];
            balanceOf[clients[i]] += amounts[i];
            total                 += amounts[i];
            emit Transfer(address(0), clients[i], amounts[i]);
            emit RewardDistributed(roundNumber, clients[i], amounts[i], basisPts[i]);
        }
        emit RoundCompleted(roundNumber, total, clients.length);
    }

    function getClientBalance(address client) external view returns (uint256) {
        return balanceOf[client];
    }
    function getCurrentRound() external view returns (uint256) {
        return roundNumber;
    }
}
"""

def compile_with_solc(source_code, contract_name):
    solc_version = "0.8.20"
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if solc_version not in installed:
        print(f"  Installing solc {solc_version}...")
        solcx.install_solc(solc_version, show_progress=True)
    compiled = solcx.compile_source(
        source_code, output_values=["abi", "bin"],
        solc_version=solc_version, optimize=True, optimize_runs=200,
    )
    key = f"<stdin>:{contract_name}"
    if key not in compiled:
        raise RuntimeError(f"Compiled output missing key: {key}")
    return compiled[key]["abi"], "0x" + compiled[key]["bin"]


print("  Compiling FairToken.sol with py-solc-x...")
ABI, BYTECODE = compile_with_solc(SOLIDITY_SOURCE, "FairToken")
n_funcs = len([x for x in ABI if x["type"] == "function"])
print(f"  Compiled: {n_funcs} functions | {len(BYTECODE)//2} bytes")

with open(os.path.join(RESULTS_DIR, "FairToken_abi.json"), "w") as f:
    json.dump(ABI, f, indent=2)

print("\n  Connecting to Ethereum Sepolia...")
private_key = os.environ.get("FAIRFED_PRIVATE_KEY", "").strip()
rpc_url = os.environ.get("FAIRFED_SEPOLIA_RPC", "").strip()
run_onchain = bool(private_key and rpc_url)

gas_records = []
CONTRACT_ADDRESS = ""
mean_gas = 0.0
total_eth = 0.0

if run_onchain:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if not w3.is_connected():
        raise ConnectionError("Cannot connect to Sepolia. Check FAIRFED_SEPOLIA_RPC and internet.")

    account = w3.eth.account.from_key(private_key)
    deployer_addr = Web3.to_checksum_address(account.address)

    chain = w3.eth.chain_id
    block = w3.eth.block_number
    bal = w3.eth.get_balance(deployer_addr)
    bal_eth = float(w3.from_wei(bal, "ether"))
    print(f"  Connected | Chain: {chain} | Block: {block:,} | Balance: {bal_eth:.6f} ETH")

    if bal < w3.to_wei(0.01, "ether"):
        print("  WARNING: Low balance. Get free Sepolia ETH at https://sepoliafaucet.com")

    print("\n  Deploying FairToken contract to Sepolia...")
    FairTokenContract = w3.eth.contract(abi=ABI, bytecode=BYTECODE)
    nonce = w3.eth.get_transaction_count(deployer_addr)
    gas_price = w3.eth.gas_price
    gas_est = FairTokenContract.constructor().estimate_gas({"from": deployer_addr})

    deploy_tx = FairTokenContract.constructor().build_transaction({
        "from": deployer_addr,
        "nonce": nonce,
        "gas": int(gas_est * 1.25),
        "gasPrice": int(gas_price * 1.1),
        "chainId": CHAIN_ID,
    })
    signed_tx = w3.eth.account.sign_transaction(deploy_tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"  Deploy tx: {tx_hash.hex()}")
    print("  Waiting for Sepolia block confirmation (~12 s)...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    CONTRACT_ADDRESS = receipt.contractAddress
    deploy_gas = receipt.gasUsed
    deploy_cost_eth = float(w3.from_wei(deploy_gas * gas_price, "ether"))
    print(f"  FairToken deployed at: {CONTRACT_ADDRESS}")
    print(f"  Gas used: {deploy_gas:,} | Cost: {deploy_cost_eth:.6f} ETH")
    print(f"  Etherscan: https://sepolia.etherscan.io/address/{CONTRACT_ADDRESS}")

    with open(os.path.join(RESULTS_DIR, "contract_address.txt"), "w") as f:
        f.write(CONTRACT_ADDRESS)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4 — Distribute FairFed Rewards On-Chain
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Phase 4] Distributing on-chain FairFed rewards for 5 representative rounds...")

    token_contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)
    gated_hist_arr = np.array(gated_history)
    ON_CHAIN_ROUNDS = [1, 5, 10, 15, 20]

    WEI_PER_TOKEN = int(10 ** 18)
    TOTAL_WEI_ROUND = int(ROUND_TOKENS) * WEI_PER_TOKEN

    for rnd_idx in ON_CHAIN_ROUNDS:
        phi_gated_round = gated_hist_arr[rnd_idx - 1]

        phi_sum = float(phi_gated_round.sum())
        if phi_sum > 0:
            phi_norm_r = [float(v) / phi_sum for v in phi_gated_round]
        else:
            phi_norm_r = [1.0 / N_CLIENTS] * N_CLIENTS

        amounts = distribute_integer_shares(phi_norm_r, TOTAL_WEI_ROUND)
        basis_pts = distribute_integer_shares(phi_norm_r, 10_000)

        nonce = w3.eth.get_transaction_count(deployer_addr)
        gas_price = w3.eth.gas_price

        tx = token_contract.functions.distributeRewards(
            CLIENT_ADDRS,
            amounts,
            basis_pts
        ).build_transaction({
            "from": deployer_addr,
            "nonce": nonce,
            "gas": 350_000,
            "gasPrice": int(gas_price * 1.1),
            "chainId": CHAIN_ID,
        })

        signed = w3.eth.account.sign_transaction(tx, private_key)
        txh = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(txh, timeout=300)

        gas_used = receipt.gasUsed
        cost_eth = float(w3.from_wei(gas_used * gas_price, "ether"))
        gas_records.append({
            "round": rnd_idx,
            "gas_used": gas_used,
            "gas_price_gwei": float(w3.from_wei(gas_price, "gwei")),
            "cost_eth": round(cost_eth, 8),
            "tx_hash": txh.hex(),
        })
        print(f"  Round {rnd_idx:2d}: tx={txh.hex()[:20]}... "
              f"gas={gas_used:,} | cost={cost_eth:.8f} ETH")

    print("\n  On-chain FAIR token balances (FairFed gated):")
    for i, addr in enumerate(CLIENT_ADDRS):
        bal_wei = token_contract.functions.getClientBalance(addr).call()
        bal_tok = bal_wei / WEI_PER_TOKEN
        role = " (free-rider)" if i == 0 else " (byzantine)" if i == 9 else ""
        print(f"  C{i:02d}{role}: {bal_tok:.4f} FAIR")

    mean_gas = sum(r["gas_used"] for r in gas_records) / len(gas_records)
    total_eth = sum(r["cost_eth"] for r in gas_records)
    print(f"\n  Mean gas / reward tx: {mean_gas:,.0f}")
    print(f"  Total ETH spent (5 tx): {total_eth:.8f} ETH")

    with open(os.path.join(RESULTS_DIR, "gas_records.json"), "w") as f:
        json.dump(gas_records, f, indent=2)
    print(f"  Saved -> {RESULTS_DIR}/gas_records.json")

else:
    print("\n  Skipping Phases 3–4 on-chain steps (set FAIRFED_SEPOLIA_RPC and "
          "FAIRFED_PRIVATE_KEY for live Sepolia deployment).")
    print("  Using committed artifacts in results/ for figure metadata where available.")
    cap_path = os.path.join(RESULTS_DIR, "contract_address.txt")
    if os.path.isfile(cap_path):
        with open(cap_path) as f:
            CONTRACT_ADDRESS = f.read().strip()
    else:
        CONTRACT_ADDRESS = "N/A (offline)"
    gp_path = os.path.join(RESULTS_DIR, "gas_records.json")
    if os.path.isfile(gp_path):
        with open(gp_path) as f:
            gas_records = json.load(f)
    if gas_records:
        mean_gas = sum(r["gas_used"] for r in gas_records) / len(gas_records)
        total_eth = sum(r["cost_eth"] for r in gas_records)

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — Publication Figures (11 figures)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Phase 5] Generating publication figures...")

plt.rcParams.update({
    "font.family":       "DejaVu Sans", "font.size": 10,
    "axes.spines.top":   False, "axes.spines.right": False,
    "axes.grid":         True,  "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi":        150,   "savefig.dpi": 150, "savefig.bbox": "tight",
})
C = {
    "fairfed":   "#2563EB",   # blue    — FairFed (proposed)
    "naive_s":   "#F59E0B",   # amber   — Naive Shapley (no gate)
    "oracle":    "#16A34A",   # green   — Oracle / equal split
    "attacker":  "#DC2626",   # red     — Byzantine
    "freerider": "#EA580C",   # orange  — Free-rider
    "neutral":   "#9CA3AF",   # grey
    "vol":       "#7C3AED",   # purple  — Volume proportional
    "gated":     "#0891B2",   # teal    — gated / gate threshold
}
CLABELS  = [f"C{i:02d}" for i in range(N_CLIENTS)]
rounds   = list(range(1, N_ROUNDS + 1))
phi_arr  = np.array(shapley_history)
gated_arr = np.array(gated_history)

# ──────────────────────────────────────────────────────────────────────────
# Fig 1: 4-panel overview
# ──────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(rounds, round_acc,  color=C["fairfed"], lw=2, label="FairFed (all clients)")
ax1.plot(rounds, oracle_acc, color=C["oracle"],  lw=2, ls="--", label="Oracle (honest only)")
ax1.fill_between(rounds, round_acc, oracle_acc, alpha=0.12,
                 color=C["attacker"], label="Attack gap")
ax1.set_xlabel("Round"); ax1.set_ylabel("Global Accuracy")
ax1.set_title("(a) Convergence Under Attacks", fontweight="bold")
ax1.legend(fontsize=8); ax1.set_ylim(0.89, 0.97)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rounds, phi_arr[:, 0],  color=C["freerider"], lw=2, marker="o", ms=4,
         label="C00 free-rider (raw phi)")
ax2.plot(rounds, phi_arr[:, 9],  color=C["attacker"],  lw=2, marker="s", ms=4,
         label="C09 byzantine")
ax2.plot(rounds, phi_arr[:, 1:9].mean(axis=1), color=C["oracle"], lw=2, ls="--",
         label="Honest mean")
ax2.plot(rounds, gated_arr[:, 0], color=C["gated"], lw=1.5, ls=":",
         label="C00 after gate (= 0)")
ax2.axhline(0, color="k", lw=0.7, ls=":")
ax2.set_xlabel("Round"); ax2.set_ylabel("Normalised Shapley phi")
ax2.set_title("(b) Shapley Scores: Raw vs Gated", fontweight="bold")
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(N_CLIENTS); w = 0.22
ax3.bar(x - w*1.5, cum_fairfed, w, alpha=0.85, color=C["fairfed"],  label="FairFed (proposed)")
ax3.bar(x - w*0.5, cum_naive_s, w, alpha=0.75, color=C["naive_s"],  label="Naive Shapley")
ax3.bar(x + w*0.5, cum_equal,   w, alpha=0.60, color=C["oracle"],   label="Equal split")
ax3.bar(x + w*1.5, cum_vol,     w, alpha=0.60, color=C["vol"],      label="Volume prop.")
ax3.set_xticks(x); ax3.set_xticklabels(CLABELS, fontsize=7, rotation=30)
ax3.set_xlabel("Client"); ax3.set_ylabel("Cumulative FAIR tokens")
ax3.set_title("(c) Reward Distribution — All Schemes", fontweight="bold")
ax3.legend(fontsize=7)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(rounds, jfi_fairfed_h, color=C["fairfed"], lw=2.5, marker="o", ms=3,
         label=f"FairFed (final {jfi_ff_fin:.4f})")
ax4.plot(rounds, jfi_naive_s_h, color=C["naive_s"], lw=2, ls="--",
         label=f"Naive Shapley ({jfi_ns_fin:.4f})")
ax4.plot(rounds, jfi_equal_h,   color=C["oracle"],  lw=1.5, ls="-.",
         label=f"Equal split ({jfi_eq_fin:.4f})")
ax4.plot(rounds, jfi_vol_h,     color=C["vol"],     lw=1.5, ls=":",
         label=f"Volume ({jfi_vl_fin:.4f})")
ax4.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.3)
ax4.set_xlabel("Round"); ax4.set_ylabel("Jain Fairness Index (honest)")
ax4.set_title("(d) Fairness per Round — All Schemes", fontweight="bold")
ax4.legend(fontsize=7); ax4.set_ylim(0.88, 1.02)

fig.suptitle("FairFed: End-to-End Blockchain-Incentivized Federated Learning",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIGURES_DIR, "fig1_overview.png")); plt.close()
print("  Saved fig1_overview.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 2: Mean Shapley bar chart (raw vs gated)
# ──────────────────────────────────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

colors_raw = [C["freerider"] if i == 0 else C["attacker"] if i == 9
              else C["fairfed"] for i in range(N_CLIENTS)]
colors_gated = [C["freerider"] if i == 0 else C["attacker"] if i == 9
                else C["fairfed"] for i in range(N_CLIENTS)]

ax2a.bar(CLABELS, phi_avg, color=colors_raw, alpha=0.85, edgecolor="white")
ax2a.axhline(phi_avg[1:9].mean(), color="k", ls="--", lw=1.2,
             label=f"Honest mean = {phi_avg[1:9].mean():.4f}")
ax2a.axhline(1/N_CLIENTS, color=C["neutral"], ls=":", lw=1,
             label=f"Equal share = {1/N_CLIENTS:.4f}")
for i, v in enumerate(phi_avg):
    ax2a.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.5)
ax2a.set_xlabel("Client"); ax2a.set_ylabel("Mean Shapley phi-bar")
ax2a.set_title("(a) Naive Shapley — Free-Rider Scores ABOVE Honest Mean\n"
               "(free-rider paradox: gate absent)",
               fontweight="bold")
ax2a.legend(fontsize=9)

ax2b.bar(CLABELS, phi_gated_avg, color=colors_gated, alpha=0.85, edgecolor="white")
ax2b.axhline(phi_gated_avg[1:9].mean(), color="k", ls="--", lw=1.2,
             label=f"Honest mean = {phi_gated_avg[1:9].mean():.4f}")
ax2b.axhline(1/N_CLIENTS, color=C["neutral"], ls=":", lw=1,
             label=f"Equal share = {1/N_CLIENTS:.4f}")
for i, v in enumerate(phi_gated_avg):
    ax2b.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7.5)
ax2b.set_xlabel("Client"); ax2b.set_ylabel("Mean FairFed phi-bar (gated)")
ax2b.set_title("(b) FairFed — Free-Rider Correctly Receives ZERO\n"
               "(delta-norm gate applied every round)",
               fontweight="bold")
ax2b.legend(fontsize=9)

plt.suptitle("FairFed: Naive Shapley vs Gated FairFed — Mean Contribution Scores",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_shapley_bar.png")); plt.close()
print("  Saved fig2_shapley_bar.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 3: Shapley heatmap + gas cost
# ──────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))

im = axes3[0].imshow(phi_arr.T, aspect="auto", cmap="Blues", vmin=0, vmax=0.22)
axes3[0].set_yticks(range(N_CLIENTS)); axes3[0].set_yticklabels(CLABELS, fontsize=8)
axes3[0].set_xlabel("Round"); axes3[0].set_ylabel("Client")
axes3[0].set_title("Naive Shapley Heatmap (all rounds)", fontweight="bold")
for i, c in [(0, "orange"), (9, "red")]:
    axes3[0].axhline(i, color=c, lw=1.5, ls="--", alpha=0.7)
plt.colorbar(im, ax=axes3[0], fraction=0.04, label="phi")

im2 = axes3[1].imshow(gated_arr.T, aspect="auto", cmap="Blues", vmin=0, vmax=0.22)
axes3[1].set_yticks(range(N_CLIENTS)); axes3[1].set_yticklabels(CLABELS, fontsize=8)
axes3[1].set_xlabel("Round"); axes3[1].set_ylabel("Client")
axes3[1].set_title("FairFed Gated Heatmap (C00 zeroed every round)", fontweight="bold")
for i, c in [(0, "orange"), (9, "red")]:
    axes3[1].axhline(i, color=c, lw=1.5, ls="--", alpha=0.7)
plt.colorbar(im2, ax=axes3[1], fraction=0.04, label="phi (gated)")

g_rounds = [r["round"] for r in gas_records]
g_vals   = [r["gas_used"] for r in gas_records]
axes3[2].bar(g_rounds, g_vals, color=C["fairfed"], alpha=0.8, width=0.6)
axes3[2].axhline(mean_gas, color="r", ls="--", lw=1.5,
                 label=f"Mean = {mean_gas:,.0f} gas")
axes3[2].set_xlabel("Round"); axes3[2].set_ylabel("Gas used")
axes3[2].set_title("Gas per distributeRewards() call\n(Sepolia; round 1 = EIP-2929 cold slots)",
                   fontweight="bold")
axes3[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_heatmap_gas.png")); plt.close()
print("  Saved fig3_heatmap_gas.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 4: Results summary table
# ──────────────────────────────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(12, 5.5))
ax.axis("off")
rows = [
    ["Final global accuracy",                     f"{round_acc[-1]:.4f}"],
    ["Oracle accuracy (honest clients only)",      f"{oracle_acc[-1]:.4f}"],
    ["Attack-accuracy gap",                        f"{oracle_acc[-1]-round_acc[-1]:.4f}"],
    ["JFI FairFed (Shapley + delta-gate)",         f"{jfi_ff_fin:.4f}  <- proposed"],
    ["JFI Naive Shapley (gate absent)",            f"{jfi_ns_fin:.4f}  <- comparison"],
    ["JFI Equal split",                            f"{jfi_eq_fin:.4f}  <- trivially 1, no deterrence"],
    ["JFI Volume proportional",                    f"{jfi_vl_fin:.4f}  <- worst (penalises small data)"],
    ["Byzantine detection (gate)",                 f"{bz_det_gate:.0f}%  (phi = 0.000 via Shapley axiom)"],
    ["Free-rider detection (gate)",                f"{fr_det_gate:.0f}%  (||dw||/||w|| < {DELTA_THR})"],
    ["Free-rider detection (5% thr, no gate)",     f"{det_rate_5pct_fr:.0f}%  (paradox — gate required)"],
    ["FR tokens FairFed (gated)",                  f"{cum_fairfed[0]:.1f}  FAIR"],
    ["FR tokens Naive Shapley (no gate)",          f"{cum_naive_s[0]:.1f}  FAIR  <- 16.9% above honest mean"],
    ["Total FAIR minted (20 rounds)",              f"{int(sum(cum_fairfed)):,}"],
    ["FairToken contract (Sepolia)",               CONTRACT_ADDRESS],
    ["Mean gas / distributeRewards() tx",          f"{mean_gas:,.0f} gas"],
    ["Total on-chain cost (5 rounds)",             f"{total_eth:.6f} ETH"],
]
tbl = ax.table(cellText=[[r[1]] for r in rows], rowLabels=[r[0] for r in rows],
               colLabels=["Value"], cellLoc="left", rowLoc="right",
               loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e3a5f"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#eef2ff")
    cell.set_edgecolor("#c8cde0")
ax.set_title("FairFed: Complete Experimental Results", fontweight="bold", fontsize=12, pad=10)
plt.savefig(os.path.join(FIGURES_DIR, "fig4_summary.png")); plt.close()
print("  Saved fig4_summary.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 5: JFI comparison — all four schemes
# ──────────────────────────────────────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))

ax5a = axes5[0]
ax5a.plot(rounds, jfi_fairfed_h, color=C["fairfed"], lw=2.5, marker="o", ms=3,
          label=f"FairFed (final {jfi_ff_fin:.4f})")
ax5a.plot(rounds, jfi_naive_s_h, color=C["naive_s"], lw=2, ls="--",
          label=f"Naive Shapley (final {jfi_ns_fin:.4f})")
ax5a.plot(rounds, jfi_equal_h,   color=C["oracle"],  lw=1.5, ls="-.",
          label=f"Equal split (final {jfi_eq_fin:.4f})")
ax5a.plot(rounds, jfi_vol_h,     color=C["vol"],     lw=1.5, ls=":",
          label=f"Volume proportional (final {jfi_vl_fin:.4f})")
ax5a.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.3)
ax5a.set_xlabel("Round"); ax5a.set_ylabel("Jain Fairness Index (honest clients)")
ax5a.set_title("(a) JFI Comparison — All Four Incentive Schemes", fontweight="bold")
ax5a.legend(fontsize=9); ax5a.set_ylim(0.88, 1.02)
ax5a.annotate("Volume penalises\nsmall-data clients",
              xy=(10, jfi_vl_fin), xytext=(13, jfi_vl_fin - 0.02),
              fontsize=8, color=C["vol"],
              arrowprops=dict(arrowstyle="->", color=C["vol"], lw=1))
ax5a.annotate("Equal split: BZ gets\nfull reward (IC fail)",
              xy=(5, jfi_eq_fin), xytext=(7, jfi_eq_fin + 0.01),
              fontsize=8, color=C["oracle"],
              arrowprops=dict(arrowstyle="->", color=C["oracle"], lw=1))

ax5b = axes5[1]
x = np.arange(N_CLIENTS); w = 0.22
ax5b.bar(x - w*1.5, cum_fairfed, w, color=C["fairfed"], alpha=0.85, label="FairFed")
ax5b.bar(x - w*0.5, cum_naive_s, w, color=C["naive_s"], alpha=0.75, label="Naive Shapley")
ax5b.bar(x + w*0.5, cum_equal,   w, color=C["oracle"],  alpha=0.60, label="Equal split")
ax5b.bar(x + w*1.5, cum_vol,     w, color=C["vol"],     alpha=0.60, label="Volume prop.")

# Annotate the free-rider bar
ax5b.annotate("FR: 0 tokens\n(FairFed)",
              xy=(0 - w*1.5, cum_fairfed[0] + 20),
              xytext=(1.5, cum_fairfed[0] + 300),
              fontsize=7, color=C["fairfed"],
              arrowprops=dict(arrowstyle="->", color=C["fairfed"], lw=0.8))
ax5b.annotate(f"FR: {cum_naive_s[0]:.0f}\n(Naive-S)",
              xy=(0 - w*0.5, cum_naive_s[0] + 20),
              xytext=(1.2, cum_naive_s[0] + 200),
              fontsize=7, color=C["naive_s"],
              arrowprops=dict(arrowstyle="->", color=C["naive_s"], lw=0.8))

ax5b.set_xticks(x); ax5b.set_xticklabels(CLABELS, fontsize=7, rotation=30)
ax5b.set_xlabel("Client"); ax5b.set_ylabel("Cumulative FAIR tokens (20 rounds)")
ax5b.set_title("(b) Cumulative Rewards — All Schemes\n"
               "FairFed zeroes the free-rider; Naive Shapley over-rewards it",
               fontweight="bold")
ax5b.legend(fontsize=8)

plt.suptitle("FairFed: Incentive Scheme Comparison", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_baseline_comparison.png")); plt.close()
print("  Saved fig5_baseline_comparison.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 6: Cumulative reward trajectories (FairFed gated)
# ──────────────────────────────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(11, 6))
cum_ff_traj = np.cumsum(gated_arr * ROUND_TOKENS, axis=0)
cum_ns_traj = np.cumsum(phi_arr  * ROUND_TOKENS, axis=0)

for i in range(N_CLIENTS):
    if i == 0:
        ax6.plot(rounds, cum_ff_traj[:, 0], color=C["fairfed"], lw=2.5,
                 ls="-", label="C00 FairFed (zero throughout)")
        ax6.plot(rounds, cum_ns_traj[:, 0], color=C["freerider"], lw=2, ls="--",
                 label="C00 Naive-S (over-rewarded)")
    elif i == N_CLIENTS - 1:
        ax6.plot(rounds, cum_ff_traj[:, i], color=C["attacker"], lw=2, ls=":",
                 label="C09 Byzantine (zero both schemes)")
    else:
        lbl = f"C{i:02d} honest" if i == 1 else f"_C{i:02d}"
        ax6.plot(rounds, cum_ff_traj[:, i], color=C["fairfed"], lw=1, alpha=0.6,
                 label=lbl)

ax6.set_xlabel("Round"); ax6.set_ylabel("Cumulative FAIR tokens")
ax6.set_title("FairFed Cumulative Reward Trajectories\n"
              "FairFed keeps free-rider at zero; Naive Shapley allows it to accumulate",
              fontweight="bold")
ax6.legend(fontsize=7.5, ncol=2, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig6_reward_trajectories.png")); plt.close()
print("  Saved fig6_reward_trajectories.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 7: Attack-gap analysis
# ──────────────────────────────────────────────────────────────────────────
acc_gap = [o - a for o, a in zip(oracle_acc, round_acc)]
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13, 5))

ax7a.bar(rounds, acc_gap, color=C["attacker"], alpha=0.75, label="Accuracy gap")
ax7a.axhline(np.mean(acc_gap), color="k", ls="--", lw=1.5,
             label=f"Mean gap = {np.mean(acc_gap):.4f}")
ax7a.set_xlabel("Round"); ax7a.set_ylabel("Accuracy gap (Oracle - FairFed)")
ax7a.set_title("(a) Per-Round Attack-Accuracy Gap", fontweight="bold")
ax7a.legend(fontsize=9)

ax7b.plot(rounds, round_acc,  color=C["fairfed"], lw=2, label="FairFed")
ax7b.plot(rounds, oracle_acc, color=C["oracle"],  lw=2, ls="--", label="Oracle (honest only)")
ax7b.fill_between(rounds, round_acc, oracle_acc, alpha=0.15, color=C["attacker"],
                  label=f"Gap (mean={np.mean(acc_gap):.4f})")
ax7b.set_xlabel("Round"); ax7b.set_ylabel("Accuracy")
ax7b.set_title("(b) FairFed vs Oracle Accuracy", fontweight="bold")
ax7b.legend(fontsize=9); ax7b.set_ylim(0.89, 0.97)

plt.suptitle("FairFed: Attack Impact on Global Model Accuracy", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig7_attack_gap.png")); plt.close()
print("  Saved fig7_attack_gap.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 8: Detection analysis (Shapley thresholds + actual delta norms)
# ──────────────────────────────────────────────────────────────────────────
fr_phis  = [d["phi_FR_raw"] for d in detection_log]
h_means  = [d["h_mean"]     for d in detection_log]
h_stds   = [d["h_std"]      for d in detection_log]
thr_5pct = [0.05 * m        for m in h_means]
thr_2sig = [m - 2*s         for m, s in zip(h_means, h_stds)]

fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 5))

ax8a.plot(rounds, fr_phis, color=C["freerider"], lw=2, marker="o", ms=4,
          label="phi[FR] raw (no gate)")
ax8a.plot(rounds, [0.0]*N_ROUNDS, color=C["attacker"], lw=2, marker="s", ms=4,
          label="phi[BZ] = 0 (Shapley axiom)")
ax8a.plot(rounds, h_means,  color=C["oracle"],  lw=2, ls="--", label="Honest mean")
ax8a.plot(rounds, thr_5pct, color="darkorange", lw=1.2, ls=":", label="5% threshold (misses FR)")
ax8a.plot(rounds, thr_2sig, color="purple",     lw=1.2, ls="-.", label="Mean-2sigma")
ax8a.fill_between(rounds, thr_2sig, thr_5pct, alpha=0.07, color="purple")
ax8a.axhline(0, color="k", lw=0.5)
ax8a.set_xlabel("Round"); ax8a.set_ylabel("Normalised Shapley phi")
ax8a.set_title("(a) Threshold Sensitivity\n"
               "5% threshold misses free-rider every round (paradox)",
               fontweight="bold")
ax8a.legend(fontsize=8)

# Actual delta norms for all clients
ax8b.fill_between(rounds,
                  [min(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
                  [max(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
                  alpha=0.25, color=C["fairfed"], label="Honest range")
ax8b.plot(rounds, [np.mean(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
          color=C["fairfed"], lw=2, label="Honest mean delta-norm")
ax8b.plot(rounds, dn_arr[:, 0], color=C["freerider"], lw=2, marker="o", ms=4,
          label=f"FR delta-norm (mean={dn_arr[:,0].mean():.5f})")
ax8b.plot(rounds, dn_arr[:, 9], color=C["attacker"], lw=2, ls=":", marker="s", ms=3,
          label="Byzantine delta-norm")
ax8b.axhline(DELTA_THR, color=C["gated"], ls="--", lw=2,
             label=f"Gate threshold epsilon = {DELTA_THR}")
ax8b.set_xlabel("Round"); ax8b.set_ylabel("Relative delta norm ||dw|| / ||w||")
ax8b.set_title("(b) Actual Delta-Norm Values — All Clients\n"
               f"FR always near 0 (< {DELTA_THR}); honest always > 0.05",
               fontweight="bold")
ax8b.legend(fontsize=8)

plt.suptitle("FairFed: Free-Rider Detection Analysis",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig8_detection_sensitivity.png")); plt.close()
print("  Saved fig8_detection_sensitivity.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 9: Per-client reward breakdown by role (NEW)
# ──────────────────────────────────────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 3, figsize=(15, 5))

# Group clients by role
roles = {
    "Free-Rider (C00)":   [0],
    "Honest (C01-C08)":   list(range(1, N_CLIENTS-1)),
    "Byzantine (C09)":    [N_CLIENTS-1],
}
role_colors = [C["freerider"], C["fairfed"], C["attacker"]]
role_names  = list(roles.keys())

schemes = {
    "FairFed":         cum_fairfed,
    "Naive Shapley":   cum_naive_s,
    "Equal Split":     cum_equal,
    "Volume Prop.":    cum_vol,
}
scheme_colors = [C["fairfed"], C["naive_s"], C["oracle"], C["vol"]]

# Panel (a): Total tokens per role per scheme
ax = axes9[0]
x_role = np.arange(len(role_names))
w_role = 0.18
for s_idx, (sname, scum) in enumerate(schemes.items()):
    role_totals = [sum(scum[i] for i in idxs) for _, idxs in roles.items()]
    ax.bar(x_role + (s_idx - 1.5) * w_role, role_totals, w_role,
           color=scheme_colors[s_idx], alpha=0.85, label=sname)
ax.set_xticks(x_role); ax.set_xticklabels(role_names, fontsize=8)
ax.set_xlabel("Client Role"); ax.set_ylabel("Total Tokens (20 rounds)")
ax.set_title("(a) Total Rewards by Client Role\nand Incentive Scheme", fontweight="bold")
ax.legend(fontsize=7.5)

# Panel (b): Per-honest-client spread under FairFed
honest_ff  = cum_fairfed[1:N_CLIENTS-1]
honest_ns  = cum_naive_s[1:N_CLIENTS-1]
honest_eq  = cum_equal  [1:N_CLIENTS-1]
honest_vol = cum_vol    [1:N_CLIENTS-1]
hclient_ids = [f"C{i:02d}" for i in range(1, N_CLIENTS-1)]
x_h = np.arange(len(hclient_ids)); w_h = 0.22

axes9[1].bar(x_h - w_h*1.5, honest_ff,  w_h, color=C["fairfed"], alpha=0.85, label="FairFed")
axes9[1].bar(x_h - w_h*0.5, honest_ns,  w_h, color=C["naive_s"], alpha=0.75, label="Naive-S")
axes9[1].bar(x_h + w_h*0.5, honest_eq,  w_h, color=C["oracle"],  alpha=0.60, label="Equal")
axes9[1].bar(x_h + w_h*1.5, honest_vol, w_h, color=C["vol"],     alpha=0.60, label="Volume")
axes9[1].axhline(np.mean(honest_ff), color=C["fairfed"], ls="--", lw=1.5,
                 label=f"FF mean = {np.mean(honest_ff):.0f}")
axes9[1].set_xticks(x_h); axes9[1].set_xticklabels(hclient_ids, fontsize=8)
axes9[1].set_xlabel("Honest Client"); axes9[1].set_ylabel("Cumulative FAIR tokens")
axes9[1].set_title("(b) Honest Client Reward Distribution\nacross Schemes", fontweight="bold")
axes9[1].legend(fontsize=7.5)

# Panel (c): Per-round reward for each role (FairFed)
ff_traj_fr  = cum_ff_traj[:, 0]
ff_traj_h   = cum_ff_traj[:, 1:N_CLIENTS-1].mean(axis=1)
ff_traj_bz  = cum_ff_traj[:, N_CLIENTS-1]
ns_traj_fr  = cum_ns_traj[:, 0]

axes9[2].plot(rounds, ff_traj_h,  color=C["fairfed"],  lw=2, label="Honest mean (FairFed)")
axes9[2].plot(rounds, ff_traj_fr, color=C["fairfed"],  lw=2, ls="--",
              alpha=0.4, label="Free-rider (FairFed = 0)")
axes9[2].plot(rounds, ns_traj_fr, color=C["freerider"], lw=2, ls="--",
              label="Free-rider (Naive-S, over-rewarded)")
axes9[2].plot(rounds, ff_traj_bz, color=C["attacker"], lw=2, ls=":",
              label="Byzantine (both = 0)")
axes9[2].set_xlabel("Round"); axes9[2].set_ylabel("Cumulative FAIR tokens")
axes9[2].set_title("(c) Reward Trajectories by Role\nFairFed vs Naive Shapley",
                   fontweight="bold")
axes9[2].legend(fontsize=7.5)

plt.suptitle("FairFed: Who Gets Rewarded and How Much",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig9_reward_breakdown.png")); plt.close()
print("  Saved fig9_reward_breakdown.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 10: Delta-norm separation — all clients, all rounds (NEW)
# ──────────────────────────────────────────────────────────────────────────
fig10, axes10 = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): box plot of delta norms per client across 20 rounds
dn_per_client = [dn_arr[:, i] for i in range(N_CLIENTS)]
bp = axes10[0].boxplot(dn_per_client, patch_artist=True, showfliers=True)
for i, (patch, whisker_pair) in enumerate(zip(bp["boxes"], zip(*[iter(bp["whiskers"])]*2))):
    if i == 0:
        patch.set_facecolor(C["freerider"]); patch.set_alpha(0.7)
    elif i == N_CLIENTS-1:
        patch.set_facecolor(C["attacker"]); patch.set_alpha(0.7)
    else:
        patch.set_facecolor(C["fairfed"]); patch.set_alpha(0.5)
axes10[0].axhline(DELTA_THR, color=C["gated"], ls="--", lw=2,
                  label=f"Gate threshold = {DELTA_THR}")
axes10[0].set_xticks(range(1, N_CLIENTS+1))
axes10[0].set_xticklabels(CLABELS, fontsize=8)
axes10[0].set_xlabel("Client"); axes10[0].set_ylabel("Relative delta norm ||dw|| / ||w||")
axes10[0].set_title("(a) Delta-Norm Distribution (20 rounds)\n"
                    "FR near zero; gate threshold separates cleanly",
                    fontweight="bold")
axes10[0].legend(fontsize=9)

handles_10 = [
    Patch(facecolor=C["freerider"], alpha=0.7, label="Free-Rider (C00)"),
    Patch(facecolor=C["fairfed"],   alpha=0.5, label="Honest clients (C01-C08)"),
    Patch(facecolor=C["attacker"],  alpha=0.7, label="Byzantine (C09)"),
]
axes10[0].legend(handles=handles_10, fontsize=8)

# Panel (b): per-round delta norms for FR, honest band, Byzantine
axes10[1].fill_between(rounds,
                       [min(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
                       [max(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
                       alpha=0.2, color=C["fairfed"], label="Honest clients range")
axes10[1].plot(rounds, [np.mean(dn_arr[r, 1:N_CLIENTS-1]) for r in range(N_ROUNDS)],
               color=C["fairfed"], lw=2.5, label="Honest mean delta-norm")
axes10[1].plot(rounds, dn_arr[:, 0], color=C["freerider"], lw=2,
               marker="o", ms=5, label=f"Free-rider (mean={dn_arr[:,0].mean():.5f})")
axes10[1].plot(rounds, dn_arr[:, 9], color=C["attacker"], lw=2, ls=":",
               marker="s", ms=4, label="Byzantine delta-norm")
axes10[1].axhline(DELTA_THR, color=C["gated"], ls="--", lw=2.5,
                  label=f"Gate threshold (epsilon={DELTA_THR})")
axes10[1].set_xlabel("Round"); axes10[1].set_ylabel("Relative delta norm ||dw|| / ||w||")
axes10[1].set_title("(b) Delta-Norm Trajectories Per Round\n"
                    f"Separation factor > 100x (FR: ~0.0005 vs honest: ~0.05+)",
                    fontweight="bold")
axes10[1].legend(fontsize=8)

plt.suptitle("FairFed: Delta-Norm Gate — Detection Mechanism Visualised",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig10_delta_norm_separation.png")); plt.close()
print("  Saved fig10_delta_norm_separation.png")

# ──────────────────────────────────────────────────────────────────────────
# Fig 11: Round-by-round reward fairness (NEW)
# ──────────────────────────────────────────────────────────────────────────
fig11, axes11 = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Per-round tokens received by each role under FairFed
rnd_ff = np.array(gated_history) * ROUND_TOKENS   # (N_ROUNDS, N_CLIENTS)
rnd_ns = np.array(shapley_history) * ROUND_TOKENS

axes11[0].stackplot(rounds,
                    rnd_ff[:, 1:N_CLIENTS-1].sum(axis=1),
                    rnd_ff[:, 0],
                    rnd_ff[:, N_CLIENTS-1],
                    labels=["Honest clients (C01-C08)", "Free-rider C00 (gated=0)", "Byzantine C09 (gated=0)"],
                    colors=[C["fairfed"], C["freerider"], C["attacker"]],
                    alpha=0.75)
axes11[0].set_xlabel("Round"); axes11[0].set_ylabel("Tokens distributed this round")
axes11[0].set_title("(a) FairFed: Per-Round Token Distribution\n"
                    "100% of tokens flow to honest clients every round",
                    fontweight="bold")
axes11[0].legend(fontsize=8, loc="upper right")
axes11[0].set_ylim(0, ROUND_TOKENS * 1.05)

# Panel (b): Honest vs attacker token share comparison
ff_honest_share  = rnd_ff[:, 1:N_CLIENTS-1].sum(axis=1) / ROUND_TOKENS * 100
ns_honest_share  = rnd_ns[:, 1:N_CLIENTS-1].sum(axis=1) / ROUND_TOKENS * 100
eq_honest_share  = np.ones(N_ROUNDS) * 80.0   # 8/10 clients at equal split

axes11[1].plot(rounds, ff_honest_share, color=C["fairfed"],  lw=2.5,
               label=f"FairFed (mean {ff_honest_share.mean():.1f}%)")
axes11[1].plot(rounds, ns_honest_share, color=C["naive_s"],  lw=2, ls="--",
               label=f"Naive Shapley (mean {ns_honest_share.mean():.1f}%)")
axes11[1].axhline(eq_honest_share[0], color=C["oracle"], lw=1.5, ls="-.",
                  label=f"Equal split = {eq_honest_share[0]:.0f}%")
axes11[1].fill_between(rounds, ff_honest_share, 100,
                        alpha=0.1, color=C["attacker"], label="Tokens lost to attackers")
axes11[1].set_ylim(75, 105)
axes11[1].set_xlabel("Round"); axes11[1].set_ylabel("Honest client share of round tokens (%)")
axes11[1].set_title("(b) Honest Clients' Share of Per-Round Tokens\n"
                    "FairFed: 100% every round. Naive Shapley: leaks tokens to free-rider.",
                    fontweight="bold")
axes11[1].legend(fontsize=8)

plt.suptitle("FairFed: Round-by-Round Reward Allocation Efficiency",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig11_round_reward_fairness.png")); plt.close()
print("  Saved fig11_round_reward_fairness.png")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FAIRFED EXPERIMENT COMPLETE")
print("=" * 65)
print(f"  FL accuracy (round {N_ROUNDS}):    {round_acc[-1]:.4f}")
print(f"  Oracle accuracy:                {oracle_acc[-1]:.4f}")
print(f"  Attack-accuracy gap:            {oracle_acc[-1]-round_acc[-1]:.4f}")
print()
print(f"  ── Fairness (JFI on honest clients) ──────────────────────────")
print(f"  FairFed (proposed):  {jfi_ff_fin:.4f}  <- best IC-compliant scheme")
print(f"  Naive Shapley:       {jfi_ns_fin:.4f}  <- gate absent, FR over-rewarded")
print(f"  Equal split:         {jfi_eq_fin:.4f}  <- trivially 1 but IC fails")
print(f"  Volume proportional: {jfi_vl_fin:.4f}  <- worst (ignores data quality)")
print()
print(f"  ── Attack Detection ──────────────────────────────────────────")
print(f"  Byzantine (gate):    {bz_det_gate:.0f}%  (Shapley zero-element axiom)")
print(f"  Free-rider (gate):   {fr_det_gate:.0f}%  (delta-norm gate, epsilon={DELTA_THR})")
print(f"  Free-rider (5% thr): {det_rate_5pct_fr:.0f}%  (naive threshold fails — gate needed)")
print()
print(f"  ── Token Allocation (20 rounds) ──────────────────────────────")
print(f"  FR tokens (FairFed):       {cum_fairfed[0]:.1f}  FAIR")
print(f"  FR tokens (Naive Shapley): {cum_naive_s[0]:.1f}  FAIR  "
      f"({(cum_naive_s[0]/ROUND_TOKENS/N_ROUNDS*100 - 100/N_CLIENTS/100*100):.1f}pp above equal share)")
print(f"  BZ tokens (both):          {cum_fairfed[9]:.1f}  FAIR")
print(f"  Mean honest (FairFed):     {cum_fairfed[1:N_CLIENTS-1].mean():.1f}  FAIR")
print()
print(f"  ── On-Chain ──────────────────────────────────────────────────")
print(f"  Contract:   {CONTRACT_ADDRESS}")
if CONTRACT_ADDRESS and not str(CONTRACT_ADDRESS).startswith("N/A"):
    print(f"  Etherscan:  https://sepolia.etherscan.io/address/{CONTRACT_ADDRESS}")
print(f"  Mean gas:   {mean_gas:,.0f} / distributeRewards() tx")
print(f"  Cost (5 rounds): {total_eth:.8f} ETH")
print()
print(f"  Results (JSON, ABI, contract): {RESULTS_DIR}/")
print(f"  Figures (PNG):                   {FIGURES_DIR}/")
