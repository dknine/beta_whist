"""Q-value networks for bidding and card-play decisions.

Same feature encoding and I/O shapes as the policy networks in policy.py so
the two approaches are directly comparable, but these output raw Q-values
(one per possible action) rather than a probability distribution -- there's
no softmax here, since action selection is masked-argmax (or epsilon-greedy
during training), not sampling.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import BIDDING_FEATURE_DIM, NUM_BID_ACTIONS, NUM_CARDS, TRICK_FEATURE_DIM


class BiddingQNetwork(nn.Module):
    """Maps a bidding feature vector to Q-values for bids 0..MAX_HAND_SIZE."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BIDDING_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_BID_ACTIONS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CardQNetwork(nn.Module):
    """Maps a trick feature vector to Q-values for the 52 card identities."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TRICK_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_CARDS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> int:
    """Index of the highest-Q legal action."""
    return int(torch.argmax(q_values.masked_fill(~mask, float("-inf"))).item())


def masked_max(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Highest Q-value among legal actions, as a scalar tensor (used for TD targets)."""
    return q_values.masked_fill(~mask, float("-inf")).max()
