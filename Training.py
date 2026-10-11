"""
Training.py  —  GPU-optimised PPO for Flatland MARL
=====================================================

Architecture:
  - 11 CPU worker processes  (mp.Process, spawn)  — each owns one Flatland env
  - 1  main process          (GPU)                 — batched inference + PPO update
  - Communication via mp.Pipe (worker→main: obs/rew/done, main→worker: actions)

Why spawn (not fork):
  - Main process initialises CUDA before workers start.
  - fork() copies the CUDA context into child processes → corrupts it.
  - spawn() starts a fresh Python interpreter per worker → always safe.

Why workers build their own env (not passed from main):
  - Flatland's StateMaskingObs uses copy.deepcopy on attr-slotted objects.
  - These cannot be pickled/sent across process boundaries.
  - This was the original crash: KeyboardInterrupt trace ending in
    slots_getstate inside DistToNextJunction → deepcopy → mp pipe send.
  - Fix: each worker calls make_env(seed_offset=rank) itself.

GPU optimisations:
  - 11 envs step in parallel on CPU while main does PPO backward.
  - All agents × all workers batched into one tensor → one GPU forward pass.
  - Rollout buffer pre-allocated on GPU — no repeated CPU→GPU copies.
"""

import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

from flatland.envs.rail_env import RailEnv
from flatland.envs.persistence import RailEnvPersister

from config import Config
from NewAgentInitObs import StateMaskingObs
from agents.models import ACNet, CostCritic
from expert_mask import state_classifier, get_valid_actions, valid_action_mask   # noqa: F401 (re-exported)
from competition.sampling_env_generator import sampling_env_generator
from flatland.envs.step_utils.states import TrainState

# ECML 2026 challenge reward function (Flatland >= 4.2.6). Older versions fall
# back to Flatland's default rewards, which lack the collision / intermediate-stop
# penalties and use cancellation factor 1 instead of 5.
try:
    from flatland.envs.rewards import ECML2026Rewards
except ImportError:
    ECML2026Rewards = None

from importlib.metadata import version as _pkg_version
_FLATLAND_VERSION = tuple(int(x) for x in _pkg_version("flatland-rl").split(".")[:2])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WORKERS    = max(1, min(11, (os.cpu_count() or 2) - 1))   # 11 on a 12-core Colab L4; fewer on smaller machines
ROLLOUT_LEN  = 128      # steps collected per worker before PPO update
PPO_EPOCHS   = 4
PPO_CLIP     = 0.2
ENTROPY_COEF = 0.001    # small: with ~97% Forward/Stop choices a larger bonus pushes Stop up
GAMMA        = 0.99
GAE_LAMBDA   = 0.95     # GAE: 1 = Monte-Carlo returns, 0 = one-step TD
LR           = 3e-4
MAX_EPISODES = 11000
SAVE_EVERY   = 50
A_SIZE       = 4        # internal: 0=Left, 1=Forward, 2=Right, 3=Stop
A_MAX        = Config.max_agents   # trains per env are padded to this many rows

# Reward shaping: Flatland 4.x only gives reward at episode end, so the agents
# get almost no feedback during long episodes. Each step, every agent
# additionally receives SHAPING_COEF × (cells it moved closer to its current
# target = next intermediate stop, or the final target once all stops are served).
# Moving away is penalised by the same amount; waiting / malfunctions give 0.
# Serving a stop gives SHAPING_STOP_BONUS (the challenge charges 50 per missed stop).
# Set both to 0.0 to train on Flatland's raw reward only.
SHAPING_COEF       = 0.1
SHAPING_STOP_BONUS = 5.0

# Soft version of the score's per-train cap at -T (max_episode_steps): the part
# of a train's reward beyond -T is multiplied by this factor (0 = hard cap like
# the score, 1 = no cap). Small but non-zero so trains past the cap still learn.
SOFT_CAP_FACTOR = 0.5

# Constrained RL (PPO-Lagrangian). Two costs, each kept below its limit by a
# multiplier lambda_k that grows while the measured rate is above the limit
# and shrinks (down to 0) once it is below:
#   0 collision     : rate = moving->stopped transitions / on-map train-steps.
#                     Every such transition is what ECML2026Rewards charges
#                     250 x speed for (blocked or deliberate stop alike);
#                     malfunctions are excluded (they go to MALFUNCTION, not STOPPED).
#   1 never departed: rate = trains still off the map at episode end / trains.
#                     The per-step learning signal is "held at departure":
#                     READY_TO_DEPART before and after the step.
# Policy advantage (per train) = (A_reward - sum_k lambda_k * A_cost_k) / (1 + sum_k lambda_k).
N_COSTS        = 2
COST_NAMES     = ("collision", "never_departed")
COST_LIMITS    = (0.002, 0.01)   # collision: <= 1 stop per 500 on-map train-steps
LAMBDA_INIT    = (0.0, 0.0)
LAMBDA_LR      = 0.02     # step per rollout, scaled by the relative violation (rate-limit)/limit
LAMBDA_MAX     = 100.0
NEVER_DEP_WINDOW = 20     # episodes used to estimate the never-departed rate

_SHUTDOWN = "SHUTDOWN"
_RESET    = "RESET"


# ---------------------------------------------------------------------------
# Environment factory  (called inside worker processes only)
# ---------------------------------------------------------------------------

def make_env(rank: int = 0) -> RailEnv:
    """Load the ECML 2026 competition map (same rail as every evaluation level).
    Schedules are sampled per episode in env_worker via sampling_env_generator."""
    if ECML2026Rewards is None:
        raise RuntimeError("Competition training needs flatland-rl>=4.2.6 (ECML2026Rewards).")
    env, _ = RailEnvPersister.load_new(Config.scenario_path,
                                       obs_builder=StateMaskingObs(),
                                       rewards=ECML2026Rewards())
    # sampling_env_generator wraps env.rail_generator; keep the original so we
    # can re-wrap it for every new schedule instead of nesting wrappers.
    env._base_rail_generator = env.rail_generator
    return env


# ---------------------------------------------------------------------------
# Expert heuristics: state_classifier / get_valid_actions now live in
# expert_mask.py (shared with the submission observation builder).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Action masking helper  (used for sampling AND in the PPO update so the
# importance ratio is computed on the same distribution actions came from)
# ---------------------------------------------------------------------------

def masked_policy(policies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renormalise policies over valid actions. Falls back to uniform-over-valid
    if the network puts ~zero mass on every valid action."""
    p = policies.clamp(min=0) * mask
    s = p.sum(-1, keepdim=True)
    uniform = mask / mask.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.where(s > 1e-8, p / s.clamp(min=1e-8), uniform)


# ---------------------------------------------------------------------------
# Worker process  (pure CPU — no CUDA, no network)
# ---------------------------------------------------------------------------

def env_worker(rank: int, obs_pipe, act_pipe, start_episode: int = 0):
    """
    Owns one Flatland env. All arrays are padded to A_MAX rows; padding rows
    have agent_mask 0, a Stop-only action mask and done=True. Protocol:
      Startup  : sends (obs [A_MAX, obs_size], valid_tgts [A_MAX, A_SIZE], agent_mask [A_MAX])
      Loop     : recvs action dict {agent_id: flatland_action (int)}
                 sends (obs, shaped_rewards, done [A_MAX] bool, done_all, valid_tgts, raw_rewards, ep_stats, costs)
                 ep_stats is None mid-episode, else {normalized_reward, pct_arrived, pct_stops, n_agents,
                 collision_rate, never_departed}
                 costs [A_MAX, 3] = (moving->stopped, held at departure, was on map) per train, 0/1
      _RESET   : starts a new episode, sends (obs, valid_tgts, agent_mask)
      _SHUTDOWN: exits
    """
    env    = make_env(rank)
    obs_sz = Config.obs_size
    # The competition map never changes, so this (pos,dir) -> transitions cache
    # (and the one in shortestpath.py) stays valid for the whole run.
    _trans_cache: dict = {}
    # Per-worker seed stream: reproducible, but different schedules/malfunctions
    # every episode and different across workers. start_episode is mixed in so a
    # resumed run does not replay the schedules from episode 0.
    seed_rng = np.random.default_rng([Config.seed + rank, start_episode])
    n = 0                       # trains in the current episode (<= A_MAX)
    episodes_on_schedule = 0

    def next_seed() -> int:
        return int(seed_rng.integers(0, 2**31 - 1))

    def new_episode():
        """New schedule every Config.schedule_reuse episodes, otherwise replay the
        current one (building a schedule on this map is slow)."""
        nonlocal n, episodes_on_schedule
        if episodes_on_schedule % Config.schedule_reuse == 0:
            env.rail_generator = env._base_rail_generator      # avoid nesting wrappers
            sampling_env_generator(env,
                                   line_length=int(seed_rng.choice(Config.line_lengths)),
                                   scene=str(seed_rng.choice(Config.scenes)))
            env.number_of_agents = int(seed_rng.integers(Config.min_agents, A_MAX + 1))
            obs_dict, _ = env.reset(regenerate_rail=True, regenerate_schedule=True,
                                    random_seed=next_seed())
        else:
            obs_dict, _ = env.reset(regenerate_rail=False, regenerate_schedule=False,
                                    random_seed=next_seed())
        episodes_on_schedule += 1
        n = len(env.agents)
        assert n <= A_MAX, f"{n} trains > Config.max_agents={A_MAX}"
        return obs_dict

    def pack_obs(obs_dict):
        arr = np.zeros((A_MAX, obs_sz), dtype=np.float32)
        for i in range(n):
            if obs_dict.get(i) is not None:
                arr[i] = obs_dict[i]
        return arr

    def compute_valid_tgts(obs_dict):
        """Multi-hot [A_MAX, A_SIZE] mask of expert-valid actions (padding: Stop)."""
        vtgts = np.zeros((A_MAX, A_SIZE), dtype=np.float32)
        vtgts[n:, 3] = 1.0
        for i in range(n):
            vtgts[i] = valid_action_mask(env, obs_dict.get(i), i, A_SIZE, _trans_cache)
        return vtgts

    def agent_mask() -> np.ndarray:
        m = np.zeros(A_MAX, dtype=np.float32)
        m[:n] = 1.0
        return m

    wp = env.obs_builder.wp     # WaypointTracker: next intermediate stop per train

    def distances(targets) -> np.ndarray:
        """Shortest-path distance (cells, inf if unreachable) from each train to targets[i]."""
        return np.array([wp.distance(i, targets[i]) for i in range(n)], dtype=np.float64)

    def episode_metrics(cum_rew: np.ndarray) -> dict:
        """ECML 2026 challenge metrics for the episode that just ended."""
        m = float(env._max_episode_steps)
        norm = float(np.maximum(cum_rew, -m).sum() / (m * n) + 1.0)
        arrived = sum(a.state == TrainState.DONE for a in env.agents) / n
        stops = float(np.mean([wp.stops_served_fraction(i) for i in range(n)]))
        never_dep = sum(a.state.is_off_map_state() for a in env.agents) / n
        return {"normalized_reward": norm, "pct_arrived": float(arrived),
                "pct_stops": stops, "n_agents": n,
                "collision_rate": ep_coll / max(ep_onmap, 1), "never_departed": float(never_dep),
                "avg_reward": float(cum_rew.mean()),           # raw Flatland reward per train
                "avg_train_reward": float(cum_train.mean())}   # soft-capped + shaped, per train

    def start():
        nonlocal targets, prev_dist, cum_rew, cum_train, ep_coll, ep_onmap
        obs_dict = new_episode()
        targets   = [wp.target(i) for i in range(n)]
        prev_dist = distances(targets)
        cum_rew   = np.zeros(n, dtype=np.float64)   # per-agent Flatland return this episode
        cum_train = np.zeros(n, dtype=np.float64)   # per-agent training reward (what PPO sees)
        ep_coll, ep_onmap = 0, 0                     # moving->stopped events / on-map train-steps
        obs_pipe.send((pack_obs(obs_dict), compute_valid_tgts(obs_dict), agent_mask()))

    targets, prev_dist, cum_rew, cum_train = [], None, None, None
    ep_coll, ep_onmap = 0, 0
    start()

    while True:
        msg = act_pipe.recv()

        if msg == _SHUTDOWN:
            break

        if msg == _RESET:
            start()
            continue

        # Normal env step. Flatland returns (obs, rewards, dones, info); there is no
        # separate truncation flag — on timeout Flatland marks every agent done itself.
        pre_state = [a.state for a in env.agents]
        obs_dict, rew_dict, term, _info = env.step({i: a for i, a in msg.items() if i < n})

        obs  = pack_obs(obs_dict)
        rews = np.zeros(A_MAX, dtype=np.float32)
        rews[:n] = [rew_dict.get(i, 0.0) for i in range(n)]
        # The score caps each train's total at -T, so penalties beyond that cost
        # nothing there. Soft cap: the part of this step's reward that lies beyond
        # -T keeps only SOFT_CAP_FACTOR of its weight. cum_rew itself stays raw (for logs).
        cap = float(env._max_episode_steps)
        over = np.minimum(cum_rew + rews[:n], -cap) - np.minimum(cum_rew, -cap)
        capped = np.zeros(A_MAX, dtype=np.float32)
        capped[:n] = rews[:n] - (1.0 - SOFT_CAP_FACTOR) * over
        cum_rew += rews[:n]

        # Constraint costs (see COST_NAMES). Flatland action 4 = STOP_MOVING.
        costs = np.zeros((A_MAX, 3), dtype=np.float32)
        for i, ag in enumerate(env.agents):
            pre, post = pre_state[i], ag.state
            if pre.is_on_map_state():
                costs[i, 2] = 1.0
                if pre == TrainState.MOVING and post == TrainState.STOPPED:
                    costs[i, 0] = 1.0
            elif pre == TrainState.READY_TO_DEPART and post == TrainState.READY_TO_DEPART:
                costs[i, 1] = 1.0
        ep_coll  += int(costs[:n, 0].sum())
        ep_onmap += int(costs[:n, 2].sum())

        # Progress towards the target each train had BEFORE this step. If the
        # train just served a stop, it gets the bonus and its new target from now on.
        cur_dist = distances(targets)
        delta    = np.where(np.isfinite(prev_dist) & np.isfinite(cur_dist), prev_dist - cur_dist, 0.0)
        shaped   = capped.copy()
        shaped[:n] += SHAPING_COEF * delta
        new_targets = [wp.target(i) for i in range(n)]
        for i in range(n):
            if new_targets[i] != targets[i]:
                shaped[i] += SHAPING_STOP_BONUS
                cur_dist[i] = wp.distance(i, new_targets[i])
        targets, prev_dist = new_targets, cur_dist
        cum_train += shaped[:n]

        done = np.ones(A_MAX, dtype=bool)
        done[:n] = [bool(term.get(i, False)) for i in range(n)]
        done_all = bool(term.get('__all__', False)) or bool(done[:n].all())

        ep_stats = episode_metrics(cum_rew) if done_all else None
        obs_pipe.send((obs, shaped, done, done_all, compute_valid_tgts(obs_dict), rews, ep_stats, costs))


# ---------------------------------------------------------------------------
# Rollout buffer  (pre-allocated on GPU)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Shape: [T, W, A, ...] — T=timesteps, W=workers, A=agents."""

    def __init__(self, T, W, A, obs_sz, act_sz, device):
        self.T      = T
        self.device = device
        kw = dict(device=device)
        self.obs        = torch.zeros(T, W, A, obs_sz, **kw)
        self.actions    = torch.zeros(T, W, A,          dtype=torch.long, **kw)
        self.rewards    = torch.zeros(T, W, A,          **kw)
        self.dones      = torch.zeros(T, W,             **kw)   # episode ended after this step
        self.agent_done = torch.zeros(T, W, A,          **kw)   # train already done BEFORE this step (padding = 1)
        self.valid_tgts = torch.zeros(T, W, A, act_sz,  **kw)
        self.log_probs  = torch.zeros(T, W, A,          **kw)
        self.values     = torch.zeros(T, W, A,          **kw)
        self.agent_mask = torch.zeros(T, W, A,          **kw)   # 1 = real train, 0 = padding
        self.costs      = torch.zeros(T, W, A, 3,       **kw)   # moving->stopped, held at departure, on map
        self.c_values   = torch.zeros(T, W, A, N_COSTS, **kw)
        self.ptr = 0

    def store(self, obs, actions, rewards, dones, agent_done, valid_tgts, log_probs, values,
              agent_mask, costs, c_values):
        t = self.ptr
        self.obs[t]        = obs
        self.actions[t]    = actions
        self.rewards[t]    = rewards
        self.dones[t]      = dones
        self.agent_done[t] = agent_done
        self.valid_tgts[t] = valid_tgts
        self.log_probs[t]  = log_probs
        self.values[t]     = values
        self.agent_mask[t] = agent_mask
        self.costs[t]      = costs
        self.c_values[t]   = c_values
        self.ptr += 1

    def reset_ptr(self):
        self.ptr = 0

    def active(self) -> torch.Tensor:
        """[T, W, A] 1 = real train that was still running at this step."""
        return self.agent_mask * (1.0 - self.agent_done)

    def _gae(self, rewards, values, boot_v, boot_alive):
        """Per-train GAE. rewards/values: [T, W, A, ...]; boot_v: [W, A, ...];
        boot_alive: [W, A] (1 = train still running after the last step).
        Returns (advantages, returns), same shape as rewards."""
        extra = [1] * (rewards.dim() - 3)
        ep_nd = (1.0 - self.dones).view(self.T, -1, 1, *extra)          # episode continues
        alive = (1.0 - self.agent_done).view(*self.agent_done.shape, *extra)
        b_alive = boot_alive.view(*boot_alive.shape, *extra)
        adv  = torch.zeros_like(rewards)
        last = torch.zeros_like(boot_v)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_v, next_alive = boot_v, b_alive
            else:
                next_v, next_alive = values[t + 1], alive[t + 1]
            cont  = ep_nd[t] * next_alive       # 0 if the episode ended or the train finished
            delta = rewards[t] + GAMMA * next_v * cont - values[t]
            last  = delta + GAMMA * GAE_LAMBDA * cont * last
            adv[t] = last
        return adv, adv + values

    def compute_gae(self, boot_v, boot_c, boot_alive):
        """Reward and cost advantages/returns, per train. boot_v: [W, A], boot_c: [W, A, N_COSTS]."""
        adv_r, ret_r = self._gae(self.rewards, self.values, boot_v, boot_alive)
        adv_c, ret_c = self._gae(self.costs[..., :N_COSTS], self.c_values, boot_c, boot_alive)
        return adv_r, ret_r, adv_c, ret_c

    def collision_rate(self) -> float:
        """Moving->stopped transitions per on-map train-step in this rollout."""
        return float(self.costs[..., 0].sum() / self.costs[..., 2].sum().clamp(min=1.0))


# ---------------------------------------------------------------------------
# PPO update  (everything on GPU)
# ---------------------------------------------------------------------------

def _masked_norm(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Normalise x to zero mean / unit std over entries where m == 1.
    x: [N, A] or [N, A, K] (normalised per K); m: [N, A]."""
    mm = m.unsqueeze(-1) if x.dim() == 3 else m
    cnt  = mm.sum(dim=(0, 1)).clamp(min=1.0)
    mean = (x * mm).sum(dim=(0, 1)) / cnt
    var  = (((x - mean) ** 2) * mm).sum(dim=(0, 1)) / cnt
    std  = var.sqrt()
    return torch.where(std > 1e-8, (x - mean) / (std + 1e-8), x - mean) * mm


def ppo_update(network, optimizer, buffer, adv_r, ret_r,
               cost_critic, c_optimizer, adv_c, ret_c, lambdas):
    T, W, A, obs_sz = buffer.obs.shape

    flat_obs      = buffer.obs.view(T * W, A, obs_sz)
    flat_acts     = buffer.actions.view(T * W, A)
    flat_tgts     = buffer.valid_tgts.view(T * W, A, -1)
    flat_lp_old   = buffer.log_probs.view(T * W, A)
    flat_amask    = buffer.agent_mask.view(T * W, A)
    flat_active   = buffer.active().view(T * W, A)        # real trains still running
    flat_ret      = ret_r.view(T * W, A)
    flat_c_ret    = ret_c.view(T * W, A, N_COSTS)
    n_active      = flat_active.sum().clamp(min=1.0)

    # Only agents that actually had a choice (>1 valid action) contribute to the
    # actor/entropy loss. Done agents, padding rows and forced moves carry no learning signal.
    has_choice = (flat_tgts.sum(-1) > 1).float() * flat_active   # [T*W, A]
    n_choice   = has_choice.sum().clamp(min=1.0)

    # Per-train Lagrangian advantage: reward GAE minus lambda-weighted cost GAEs,
    # each normalised separately over running trains; dividing by (1 + sum lambda)
    # keeps the scale stable.
    a_r = _masked_norm(adv_r.view(T * W, A), flat_active)              # [T*W, A]
    a_c = _masked_norm(adv_c.view(T * W, A, N_COSTS), flat_active)     # [T*W, A, N_COSTS]
    lam = torch.tensor(lambdas, device=a_r.device, dtype=a_r.dtype)
    adv = (a_r - (a_c * lam).sum(-1)) / (1.0 + lam.sum())

    # Return statistics over running trains, for scale-free value losses
    r_mean = (flat_ret * flat_active).sum() / n_active
    r_std  = ((((flat_ret - r_mean) ** 2) * flat_active).sum() / n_active).sqrt().clamp(min=1e-8)
    am3    = flat_active.unsqueeze(-1)
    c_mean = (flat_c_ret * am3).sum(dim=(0, 1)) / n_active
    c_std  = ((((flat_c_ret - c_mean) ** 2) * am3).sum(dim=(0, 1)) / n_active).sqrt().clamp(min=1e-8)

    total_loss = 0.0
    for _ in range(PPO_EPOCHS):
        policies, estimations, cur_v = network(flat_obs, flat_amask)   # cur_v: [T*W, A]

        # Same masked distribution the actions were sampled from
        m_pol  = masked_policy(policies, flat_tgts)

        # Actor loss (PPO clip), per-train advantage
        act_p  = m_pol.gather(2, flat_acts.unsqueeze(-1)).squeeze(-1)
        log_p  = torch.log(act_p + 1e-8)
        ratio  = torch.exp(log_p - flat_lp_old)
        surr1  = ratio * adv
        surr2  = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv
        a_loss = -(torch.min(surr1, surr2) * has_choice).sum() / n_choice

        # Critic loss: per-train MSE on normalised returns, running trains only
        c_err  = ((cur_v - flat_ret) / r_std) ** 2
        c_loss = (c_err * flat_active).sum() / n_active

        # Entropy over valid actions only (unmasked entropy would push mass onto invalid moves)
        ent    = (-(m_pol * torch.log(m_pol + 1e-8)).sum(-1) * has_choice).sum() / n_choice

        # Estimator (valid-action imitation), real trains only
        e_all  = F.binary_cross_entropy_with_logits(estimations, flat_tgts, reduction="none").mean(-1)
        e_loss = (e_all * flat_amask).sum() / flat_amask.sum().clamp(min=1.0)

        loss = a_loss + 0.5 * c_loss - ENTROPY_COEF * ent + 0.5 * e_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

        # Cost critics: per-train MSE on per-cost normalised returns
        cv      = cost_critic(flat_obs, flat_amask)               # [T*W, A, N_COSTS]
        cv_err  = (((cv - flat_c_ret) / c_std) ** 2).mean(-1)
        cv_loss = (cv_err * flat_active).sum() / n_active
        c_optimizer.zero_grad()
        cv_loss.backward()
        nn.utils.clip_grad_norm_(cost_critic.parameters(), 0.5)
        c_optimizer.step()

    return total_loss / PPO_EPOCHS


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Workers: {N_WORKERS}  (cores used: {N_WORKERS + 1} / {os.cpu_count()})")
    print(f"Flatland {_pkg_version('flatland-rl')} | rewards: "
          + ("ECML2026Rewards (challenge)" if ECML2026Rewards is not None
             else "DefaultRewards — install flatland-rl>=4.2.6 for the ECML 2026 challenge rewards"))
    print()

    # ---- Network + optimiser ----
    network = ACNet(
        state_size=Config.obs_size,
        action_size=A_SIZE,
        n_agents=A_MAX,
    ).to(device)
    optimizer = optim.AdamW(network.parameters(), lr=LR,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    cost_critic = CostCritic(Config.obs_size, N_COSTS).to(device)
    c_optimizer = optim.Adam(cost_critic.parameters(), lr=LR)
    lambdas     = list(LAMBDA_INIT)

    # ---- Resume from checkpoint ----
    start_episode = 0
    ckpt_dir = os.path.abspath(Config.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = [f for f in os.listdir(ckpt_dir)
             if f.startswith("checkpoint_") and f.endswith(".pth")]
    if ckpts:
        latest = max(ckpts, key=lambda f: int(f.split("_")[1].split(".")[0]))
        ckpt   = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
        network.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = ckpt.get("episode", 0)
        if "cost_critic_state_dict" in ckpt:
            cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
            c_optimizer.load_state_dict(ckpt["c_optimizer_state_dict"])
            lambdas = list(ckpt.get("lambdas", LAMBDA_INIT))
        print(f"Resumed: {latest}  (episode {start_episode}, lambdas {lambdas})")

    # ---- Spawn worker processes ----
    # spawn is set in __main__ block before train() is called.
    # Workers are daemon=True so they die automatically if main crashes.
    workers, obs_pipes, act_pipes = [], [], []
    for rank in range(N_WORKERS):
        p_obs, c_obs = mp.Pipe(duplex=False)   # worker → main  (p_obs=read, c_obs=write)
        p_act, c_act = mp.Pipe(duplex=False)   # main → worker  (p_act=read, c_act=write)
        # Worker needs: c_obs (write end, to send obs) + p_act (read end, to recv actions)
        # Main  needs: p_obs (read end, to recv obs)  + c_act (write end, to send actions)
        p = mp.Process(target=env_worker,
                       args=(rank, c_obs, p_act, start_episode),
                       daemon=True)
        p.start()
        workers.append(p)
        obs_pipes.append(p_obs)
        act_pipes.append(c_act)
    print(f"Spawned {N_WORKERS} env workers.")

    # Collect initial obs + valid-action masks from each worker
    # (the first schedule on the competition map can take a minute to build)
    init_data     = [obs_pipes[w].recv() for w in range(N_WORKERS)]
    obs_np        = np.stack([d[0] for d in init_data])   # [W, A, obs]
    valid_tgts_np = np.stack([d[1] for d in init_data])   # [W, A, A_SIZE]
    amask_np      = np.stack([d[2] for d in init_data])   # [W, A] 1 = real train
    dones_np      = amask_np < 0.5                         # [W, A] padding counts as done

    # ---- Rollout buffer (GPU) ----
    buffer = RolloutBuffer(
        T=ROLLOUT_LEN, W=N_WORKERS, A=A_MAX,
        obs_sz=Config.obs_size, act_sz=A_SIZE, device=device,
    )

    episode_count = start_episode
    last_saved_ep = start_episode
    recent_stats  = deque(maxlen=100)   # challenge metrics of the last 100 episodes
    never_dep_buf = deque(maxlen=NEVER_DEP_WINDOW)   # never-departed fraction per recent episode
    ep_reward_buf = [0.0] * N_WORKERS   # running reward sum per worker, reset each episode
    t_start       = time.time()
    print(f"Training: episodes {start_episode} → {MAX_EPISODES}\n")

    # ==========================================================================
    # Main loop
    # ==========================================================================
    while episode_count < MAX_EPISODES:
        buffer.reset_ptr()
        rollout_ep_rewards: list[float] = []   # EpRewards of episodes completed this rollout
        rollout_ep_stats:   list[dict]  = []   # challenge metrics of those episodes
        network.eval()

        # ----------------------------------------------------------------------
        # ROLLOUT COLLECTION
        # Key timing:
        #   1. GPU forward pass  (~ms)
        #   2. Send actions to all workers simultaneously
        #   3. Workers step envs IN PARALLEL on their cores  (~seconds, the bottleneck)
        #   4. Collect results
        # Step 3 overlaps with any CPU work we do between send and recv.
        # ----------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(ROLLOUT_LEN):

                # 1. Batched GPU inference — all workers × all agents in one call
                obs_t              = torch.from_numpy(obs_np).to(device)      # [W, A, obs]
                amask_t            = torch.from_numpy(amask_np).to(device)    # [W, A]
                policies, _, values_t = network(obs_t, amask_t)                # [W, A, act], [W, A]
                c_values_t         = cost_critic(obs_t, amask_t)               # [W, A, N_COSTS]

                # 2. Build the current-state action mask. Done agents, padding rows
                #    (and any row with no valid action) are forced to Stop, so they
                #    contribute log-prob 0 and no gradient.
                forced = dones_np | (valid_tgts_np.sum(-1) <= 0)
                valid_tgts_np[forced] = 0.0
                valid_tgts_np[forced, 3] = 1.0
                valid_t = torch.from_numpy(valid_tgts_np).to(device)          # [W, A, A_SIZE]

                # 3. Sample from the masked policy and record ITS log-prob
                m_pol     = masked_policy(torch.nan_to_num(policies), valid_t)
                actions_t = Categorical(probs=m_pol).sample()                 # [W, A]
                log_probs = torch.log(m_pol.gather(2, actions_t.unsqueeze(-1)).squeeze(-1) + 1e-8)
                actions_np = actions_t.cpu().numpy()

                # internal 0=L,1=F,2=R,3=Stop → Flatland 1=L,2=F,3=R,4=Stop
                # (the worker drops padding rows before env.step)
                actions_dicts = [
                    {i: int(a) + 1 if a < 3 else 4 for i, a in enumerate(actions_np[w])}
                    for w in range(N_WORKERS)
                ]

                # 4. Send actions to ALL workers — they start stepping in parallel NOW
                for w in range(N_WORKERS):
                    act_pipes[w].send(actions_dicts[w])

                # 5. Collect results — blocks until all workers finish
                results          = [obs_pipes[w].recv() for w in range(N_WORKERS)]
                next_obs_np      = np.stack([r[0] for r in results])          # [W, A, obs]
                rewards_np       = np.stack([r[1] for r in results])          # [W, A] shaped (for learning)
                next_dones_np    = np.stack([r[2] for r in results])          # [W, A] bool
                done_all         = [r[3]            for r in results]         # [W] episode over
                valid_tgts_np    = np.stack([r[4] for r in results])          # [W, A, A_SIZE] next-state masks
                raw_rewards_np   = np.stack([r[5] for r in results])          # [W, A] Flatland reward (for logs)
                costs_np         = np.stack([r[7] for r in results])          # [W, A, 3] constraint costs

                rewards_t = torch.from_numpy(rewards_np).to(device)
                dones_t   = torch.tensor([float(d) for d in done_all], device=device)

                buffer.store(obs_t, actions_t, rewards_t, dones_t,
                             torch.from_numpy(dones_np.astype(np.float32)).to(device),
                             valid_t, log_probs, values_t, amask_t,
                             torch.from_numpy(costs_np).to(device), c_values_t)

                # Accumulate rewards BEFORE the done check so each episode's final
                # step (which carries Flatland's end-of-episode penalty) is counted
                # in that episode, not the next one.
                for w in range(N_WORKERS):
                    ep_reward_buf[w] += float(raw_rewards_np[w].sum())

                # ---- Inline reset + per-episode logging ----
                finished = [w for w in range(N_WORKERS) if done_all[w]]
                for w in finished:
                    episode_count += 1
                    rollout_ep_rewards.append(ep_reward_buf[w])
                    stats = results[w][6] or {"normalized_reward": 0.0, "pct_arrived": 0.0,
                                              "pct_stops": 0.0, "n_agents": 0,
                                              "collision_rate": 0.0, "never_departed": 0.0,
                                              "avg_reward": 0.0, "avg_train_reward": 0.0}
                    rollout_ep_stats.append(stats)
                    recent_stats.append(stats)
                    never_dep_buf.append(stats["never_departed"])
                    elapsed = time.time() - t_start
                    print(f"  [W{w:02d}] Ep {episode_count:4d}/{MAX_EPISODES} | "
                          f"trains {stats['n_agents']:3d} | "
                          f"EpReward {ep_reward_buf[w]:9.1f} | "
                          f"AvgRew/train {stats['avg_reward']:8.1f} | "
                          f"TrainRew/train {stats['avg_train_reward']:8.1f} | "
                          f"Score {stats['normalized_reward']:.3f} | "
                          f"Arrived {stats['pct_arrived']*100:5.1f}% | "
                          f"Stops {stats['pct_stops']*100:5.1f}% | "
                          f"Coll {stats['collision_rate']:.4f} | "
                          f"NeverDep {stats['never_departed']*100:5.1f}% | "
                          f"{elapsed/60:.1f}min")
                    ep_reward_buf[w] = 0.0   # reset accumulator
                    act_pipes[w].send(_RESET)   # all finished workers reset in parallel
                for w in finished:
                    reset_data        = obs_pipes[w].recv()
                    next_obs_np[w]    = reset_data[0]
                    valid_tgts_np[w]  = reset_data[1]
                    amask_np[w]       = reset_data[2]
                    next_dones_np[w]  = reset_data[2] < 0.5

                obs_np   = next_obs_np
                dones_np = next_dones_np

        # ----------------------------------------------------------------------
        # PPO UPDATE  (GPU — workers idle during this, waiting for next actions)
        # ----------------------------------------------------------------------
        network.train()
        with torch.no_grad():
            obs_t   = torch.from_numpy(obs_np).to(device)
            amask_t = torch.from_numpy(amask_np).to(device)
            _, _, boot_v = network(obs_t, amask_t)                       # [W, A]
            boot_c       = cost_critic(obs_t, amask_t)                   # [W, A, N_COSTS]
            boot_alive   = torch.from_numpy((~dones_np).astype(np.float32)).to(device)

        adv_r, ret_r, adv_c, ret_c = buffer.compute_gae(boot_v, boot_c, boot_alive)
        avg_loss  = ppo_update(network, optimizer, buffer, adv_r, ret_r,
                               cost_critic, c_optimizer, adv_c, ret_c, lambdas)

        # ----------------------------------------------------------------------
        # LAGRANGE MULTIPLIERS  (dual ascent on the measured constraint rates)
        # ----------------------------------------------------------------------
        rates = [buffer.collision_rate(),
                 float(np.mean(never_dep_buf)) if never_dep_buf else None]
        for k in range(N_COSTS):
            if rates[k] is None or (k == 1 and not rollout_ep_stats):
                continue   # never-departed is only re-measured when an episode ends
            violation  = (rates[k] - COST_LIMITS[k]) / COST_LIMITS[k]
            lambdas[k] = float(np.clip(lambdas[k] + LAMBDA_LR * violation, 0.0, LAMBDA_MAX))

        # ----------------------------------------------------------------------
        # LOGGING
        # ----------------------------------------------------------------------
        elapsed = time.time() - t_start
        if rollout_ep_rewards:
            n_ep  = len(rollout_ep_rewards)
            score = sum(s["normalized_reward"] for s in rollout_ep_stats) / n_ep
            arr   = sum(s["pct_arrived"]       for s in rollout_ep_stats) / n_ep
            avg_r = sum(s["avg_reward"]        for s in rollout_ep_stats) / n_ep
            ep_reward_str = (f"EpReward {sum(rollout_ep_rewards)/n_ep:8.2f} | "
                             f"AvgRew/train {avg_r:8.1f} | "
                             f"Score {score:.3f} | Arrived {arr*100:5.1f}% (n={n_ep})")
        else:
            # No episode finished this rollout — fall back to per-step mean (per real train)
            per_train = buffer.rewards.sum(dim=-1) / buffer.agent_mask.sum(dim=-1).clamp(min=1.0)
            ep_reward_str = f"ShapedReward/step/train {per_train.mean().item():.3f} (no ep done)"
        if recent_stats:
            avg_score = sum(s["normalized_reward"] for s in recent_stats) / len(recent_stats)
            avg_arr   = sum(s["pct_arrived"]       for s in recent_stats) / len(recent_stats)
            avg_rew   = sum(s["avg_reward"]        for s in recent_stats) / len(recent_stats)
            ep_reward_str += (f" | Last{len(recent_stats)}: Score {avg_score:.3f}, "
                              f"Arrived {avg_arr*100:5.1f}%, AvgRew/train {avg_rew:.1f}")
        nd_str = f"{rates[1]*100:.1f}%" if rates[1] is not None else "n/a"
        print(f"Ep {episode_count:4d}/{MAX_EPISODES} | "
              f"Loss {avg_loss:.4f} | "
              f"{ep_reward_str} | "
              f"Coll {rates[0]:.4f} (lim {COST_LIMITS[0]}) | NeverDep {nd_str} (lim {COST_LIMITS[1]*100:.0f}%) | "
              f"lambda {lambdas[0]:.2f}/{lambdas[1]:.2f} | "
              f"{elapsed/60:.1f}min elapsed")

        # ----------------------------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------------------------
        # Episode count can jump by several per rollout (many workers), so a
        # `% SAVE_EVERY == 0` test would often be skipped. Track the gap instead.
        if episode_count - last_saved_ep >= SAVE_EVERY:
            path = os.path.join(ckpt_dir, f"checkpoint_{episode_count}.pth")
            torch.save({
                "model_state_dict":     network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode":              episode_count,
                "cost_critic_state_dict": cost_critic.state_dict(),
                "c_optimizer_state_dict": c_optimizer.state_dict(),
                "lambdas":              lambdas,
            }, path)
            last_saved_ep = episode_count
            print(f"  >> Saved {path}")

        # (Workers are reset inline during the rollout — nothing to do here.)

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    for w in range(N_WORKERS):
        act_pipes[w].send(_SHUTDOWN)
    for p in workers:
        p.join(timeout=10)

    final = os.path.join(ckpt_dir, "appo_final.pth")
    torch.save({
        "model_state_dict":     network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode":              episode_count,
        "cost_critic_state_dict": cost_critic.state_dict(),
        "c_optimizer_state_dict": c_optimizer.state_dict(),
        "lambdas":              lambdas,
    }, final)
    print(f"\nDone. Saved → {final}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Must be set before ANY CUDA call and before spawning.
    # force=True in case something imported above already set it.
    mp.set_start_method("spawn", force=True)
    train()
