import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    """
    Standard 7-layer MLP structure consistent with NeurIPS 2020 architecture.
    """
    def __init__(self, in_features, out_features):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1024), nn.ReLU(),
            nn.Linear(1024, out_features)
        )

    def forward(self, x):
        return self.net(x)

class ACNet(nn.Module):
    def __init__(self, state_size, action_size, n_agents):
        """
        Multi-Agent Actor-Critic Network
        - Centralized critic with one value per train (own encoding + pooled context)
        - Multiple Actors using the DeepMLP structure (Decentralized)
        - Multiple Action Estimators using the same DeepMLP structure (Decentralized)
        """
        super(ACNet, self).__init__()
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size

        # --- CENTRALIZED CRITIC ---
        # Works for any number of trains (the competition uses 8 to 532): each
        # train's observation is encoded separately, then pooled (masked mean + max)
        # so padding rows are ignored. n_agents is kept for API compatibility only.
        self.critic_enc = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.critic_net = nn.Sequential(
            nn.Linear(2 * 256 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Per-train value: [own encoding, pooled context] -> V_i, so each train
        # gets its own advantage (credit assignment) instead of one shared value.
        self.value_head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # --- SHARED ACTOR ---
        # Single actor network shared across all homogeneous agents
        self.actor = DeepMLP(state_size, action_size)

        # --- SHARED ESTIMATOR ---
        self.estimator = DeepMLP(state_size, action_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Adjust head gains for stability
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        
        nn.init.orthogonal_(self.actor.net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.estimator.net[-1].weight, gain=0.01)
        # Start out preferring to drive: Stop logit -5 (~0.7% Stop when Forward/Stop
        # are both allowed). With the loosened mask almost every train-step has a
        # Forward/Stop choice; an unbiased net would stop half the time on plain
        # track, and every stop while moving costs 250. RL can raise it where useful.
        with torch.no_grad():
            self.actor.net[-1].bias[3] = -5.0

    def forward(self, x, agent_mask=None):
        """
        Args:
            x (torch.Tensor): Tensor of individual states [batch, n_agents, state_size]
            agent_mask (torch.Tensor, optional): [batch, n_agents], 1 for real trains,
                0 for padding rows. Defaults to all ones.
        Returns:
            policies (torch.Tensor): Agent policy outputs [batch, n_agents, action_size]
            estimations (torch.Tensor): Agent estimation outputs [batch, n_agents, action_size]
            values (torch.Tensor): Per-train value output [batch, n_agents]
        """
        # 1. Process all agent states through the shared Actor
        # x is [batch_size, n_agents, state_size], Linear layers broadcast the computation automatically!
        actor_logits = self.actor(x) 
        policies = F.softmax(actor_logits, dim=-1) # [batch_size, n_agents, action_size]

        # Valid action estimation
        estimator_logits = self.estimator(x) # [batch_size, n_agents, action_size]

        # Centralized Critic (permutation-invariant pooling over real trains)
        if agent_mask is None:
            agent_mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        m      = agent_mask.unsqueeze(-1)                           # [B, A, 1]
        n      = m.sum(dim=1).clamp(min=1.0)                        # [B, 1]
        enc    = self.critic_enc(x)                                 # [B, A, 256]
        mean_p = (enc * m).sum(dim=1) / n                           # [B, 256]
        max_p  = enc.masked_fill(m == 0, -1e9).max(dim=1).values    # [B, 256]
        max_p  = torch.where(n > 0.5, max_p, torch.zeros_like(max_p))
        pooled = torch.cat([mean_p, max_p, agent_mask.sum(dim=1, keepdim=True) / 100.0], dim=-1)

        ctx = self.critic_net(pooled)                               # [B, 256]
        ctx = ctx.unsqueeze(1).expand(-1, x.shape[1], -1)           # [B, A, 256]
        v = self.value_head(torch.cat([enc, ctx], dim=-1)).squeeze(-1)   # [B, A]
        
        return policies, estimator_logits, v


class CostCritic(nn.Module):
    """Centralised critic for the constraint costs (training only, never used by
    the submission). Same masked mean+max pooling as ACNet's critic, one value
    per train and cost. Kept separate so ACNet checkpoints stay loadable by my_policy.py."""

    def __init__(self, state_size, n_costs):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(2 * 128 + 1, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 128, 128), nn.ReLU(),
                                  nn.Linear(128, n_costs))

    def forward(self, x, agent_mask):
        m      = agent_mask.unsqueeze(-1)                           # [B, A, 1]
        n      = m.sum(dim=1).clamp(min=1.0)                        # [B, 1]
        enc    = self.enc(x)                                        # [B, A, 128]
        mean_p = (enc * m).sum(dim=1) / n
        max_p  = enc.masked_fill(m == 0, -1e9).max(dim=1).values
        max_p  = torch.where(n > 0.5, max_p, torch.zeros_like(max_p))
        pooled = torch.cat([mean_p, max_p, agent_mask.sum(dim=1, keepdim=True) / 100.0], dim=-1)
        ctx    = self.ctx(pooled).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.head(torch.cat([enc, ctx], dim=-1))             # [B, A, n_costs]
