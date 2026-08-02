"""State-value networks used as a learned REINFORCE baseline (actor-critic).

Same feature encoding as the policy networks, but output a single scalar
V(s) -- an estimate of the expected round score from this decision point
onward -- rather than an action distribution. Trained via Monte-Carlo
regression toward the round's actual score (consistent with the rest of
this framework's "each round is an independent episode" design), not via
TD(0) bootstrapping -- that's Q-learning's distinguishing feature
(qtrain.py); keeping the critic here MC-based keeps a clean conceptual line
between "REINFORCE with a learned baseline" and "true" TD actor-critic.

Conditioning the baseline on the actual state (hand strength, bids so far,
etc.) instead of just a per-hand-size average removes far more variance
from the policy gradient than the previous scalar EMA baseline could.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import BIDDING_FEATURE_DIM, TRICK_FEATURE_DIM


class BiddingValueNet(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BIDDING_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class CardValueNet(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TRICK_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
