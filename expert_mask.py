"""
Expert valid-action mask (port of _NextValidActions from the NeurIPS reference code).

Shared by Training.py (action masking + estimator targets) and the submission
observation builder (the evaluator's policy never sees the env, so the mask has
to travel inside the observation).

Internal actions: 0=Left, 1=Forward, 2=Right, 3=Stop.
"""
import numpy as np

from shortestpath import rail_transitions    # Flatland-version-independent get_transitions
from flatland.envs.step_utils.states import TrainState

# Loosened mask: on plain, free track (state 0) a train that can actually act
# (moving, stopped or ready to depart) may choose Stop as well as Forward, so the
# policy can learn to hold back early. The safety rules (red signal, occupied
# cell ahead, junction exits) stay hard. False = original NeurIPS expert mask.
ALLOW_STOP_ON_TRACK = False
_CAN_ACT = (TrainState.MOVING, TrainState.STOPPED, TrainState.READY_TO_DEPART)


def state_classifier(env, pos, direction, _cache=None):
    """
    Classify current cell:
      0: Normal  1: Stopping Point  2: Junction  3: Must Stop  4: Stopping+Must Stop
    Uses an optional dict cache to avoid repeated rail.get_transitions() calls.
    """
    key = (pos, direction)
    if _cache is not None and key in _cache:
        avb_moves = _cache[key]
    else:
        avb_moves = rail_transitions(env.rail, pos, direction)
        if _cache is not None:
            _cache[key] = avb_moves
    if sum(avb_moves) > 1:
        return 2
    avb_idx     = avb_moves.index(1)
    dir_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    next_pos    = (pos[0] + dir_offsets[avb_idx][0],
                   pos[1] + dir_offsets[avb_idx][1])
    is_next_junction = False
    for d in range(4):
        nkey = (next_pos, d)
        if _cache is not None and nkey in _cache:
            trans = _cache[nkey]
        else:
            trans = rail_transitions(env.rail, next_pos, d)
            if _cache is not None:
                _cache[nkey] = trans
        if sum(trans) > 2:
            is_next_junction = True
            break
    occupied = any(a.position == next_pos for a in env.agents)
    if is_next_junction:
        return 4 if occupied else 1
    return 3 if occupied else 0


def get_valid_actions(env, obs, agent_id, _cache=None):
    """
    Returns a list of valid internal actions: 0=Left, 1=Forward, 2=Right, 3=Stop.
    _cache: optional dict for caching rail.get_transitions results.
    """
    # ECML 2026: a stop only counts if the train actually stands still on it.
    # Force Stop for one step when the train is on its next unserved stop.
    wp = getattr(env.obs_builder, "wp", None)
    if wp is not None and wp.env is env and wp.at_unserved_stop(agent_id):
        return [3]

    traffic_signal = obs[24]
    homo_junctions = np.asarray(obs[27:30])
    if traffic_signal == -1:
        return [3]
    currentobs  = np.reshape(obs[0:24], (3, -1))
    agent       = env.agents[agent_id]
    actual_pos  = agent.position  if agent.position  is not None else agent.initial_position
    actual_dir  = agent.direction if agent.direction is not None else agent.initial_direction
    state = state_classifier(env, actual_pos, actual_dir, _cache)
    if state in [3, 4]:
        return [3]
    elif state == 0:
        if ALLOW_STOP_ON_TRACK and agent.state in _CAN_ACT:
            return [1, 3]
        return [1]
    elif state == 1:
        SolExist         = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
        agentsblocking   = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
        agentsblockjunct = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
        for i in range(3):
            if SolExist[i] == 1 and agentsblocking[i] == 0:
                return [1, 3]
        if int(np.sum(homo_junctions == 1)) >= 2:
            for i in range(3):
                if SolExist[i] == 1 and agentsblockjunct[i] == 0 and homo_junctions[i] == 1:
                    return [1, 3]
        return [3]
    else:  # Junction
        SolExist         = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
        agentsblocking   = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
        agentsblockjunct = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
        validactions = []
        for i in range(3):
            if SolExist[i] == 1 and agentsblocking[i] == 0:
                validactions.append(i)
        if validactions:
            return validactions
        if int(np.sum(homo_junctions == 1)) >= 2:
            for i in range(3):
                if SolExist[i] == 1 and agentsblockjunct[i] == 0 and homo_junctions[i] == 1:
                    validactions.append(i)
                    break
            if validactions:
                return validactions
        return [3]


def valid_action_mask(env, obs, agent_id, n_actions=4, _cache=None):
    """Multi-hot mask of expert-valid actions. Falls back to Stop on errors / missing obs."""
    mask = np.zeros(n_actions, dtype=np.float32)
    if obs is None:
        mask[3] = 1.0
        return mask
    try:
        mask[get_valid_actions(env, obs, agent_id, _cache)] = 1.0
    except Exception:
        mask[3] = 1.0
    return mask
