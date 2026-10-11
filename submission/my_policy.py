"""
ECML 2026 policy: the shared actor from agents/models.py ACNet, greedy over the
expert-valid actions. Set in the Docker image as
    POLICY=submission.my_policy.MyPolicy

Checkpoint: $CHECKPOINT, else submission/checkpoint.pth (any checkpoint_*.pth
written by Training.py works; only model_state_dict is used).
"""
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from flatland.envs.RailEnvPolicy import RailEnvPolicy    # noqa: E402

from agents.models import ACNet                          # noqa: E402
from submission.my_observation_builder import OBS_SIZE, N_ACTIONS   # noqa: E402

torch.set_num_threads(1)   # evaluation runs on 4 CPUs; one thread is plenty for an MLP


class MyPolicy(RailEnvPolicy):
    def __init__(self):
        super().__init__()
        self.net = ACNet(OBS_SIZE, N_ACTIONS, n_agents=1)
        path = os.environ.get("CHECKPOINT", os.path.join(_HERE, "checkpoint.pth"))
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            sd = state.get("model_state_dict", state)
            # Only the actor is used at evaluation; loading just its weights keeps
            # checkpoints from older critic versions usable.
            self.net.actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items()
                                            if k.startswith("actor.")})
            print(f"[MyPolicy] loaded {path}")
        else:
            print(f"[MyPolicy] WARNING: no checkpoint at {path}, using an untrained network")
        self.net.eval()

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, int]:
        if len(handles) == 0:
            return {}
        obs = np.stack([np.asarray(o, dtype=np.float32) for o in observations])
        x = torch.from_numpy(obs[:, :OBS_SIZE])
        mask = torch.from_numpy(obs[:, OBS_SIZE:OBS_SIZE + N_ACTIONS])
        with torch.no_grad():
            logits = self.net.actor(x).masked_fill(mask < 0.5, float("-inf"))
        acts = logits.argmax(-1).tolist()
        # internal 0=L,1=F,2=R,3=Stop -> Flatland 1=L,2=F,3=R,4=Stop
        return {h: (a + 1 if a < 3 else 4) for h, a in zip(handles, acts)}

    def act(self, observation: Any, **kwargs) -> int:
        return self.act_many([0], [observation])[0]
