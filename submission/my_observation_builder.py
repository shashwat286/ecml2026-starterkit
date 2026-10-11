"""
ECML 2026 observation builder: this project's 30-value observation plus the
expert valid-action mask (4 values: Left, Forward, Right, Stop).

The evaluator's policy never sees the env, so the mask has to travel inside the
observation. Set in the Docker image as
    OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder
"""
import os
import sys

import numpy as np

# Project modules live one folder up (NewAgentInitObs.py, expert_mask.py, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewAgentInitObs import StateMaskingObs      # noqa: E402
from expert_mask import valid_action_mask        # noqa: E402

OBS_SIZE = StateMaskingObs.OBS_SIZE   # 30
N_ACTIONS = 4


class MyObservationBuilder(StateMaskingObs):
    def __init__(self):
        super().__init__()
        self._mask_cache = {}   # (pos, dir) -> rail transitions; the map is fixed

    def get_many(self, handles=None):
        base = super().get_many(handles)
        out = {}
        for h, o in base.items():
            mask = valid_action_mask(self.env, o, h, N_ACTIONS, self._mask_cache)
            o = np.zeros(OBS_SIZE, dtype=np.float32) if o is None else np.asarray(o, dtype=np.float32)
            out[h] = np.concatenate([o, mask])
        return out
