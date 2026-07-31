"""Neural network policies for bidding and card-play decisions."""

from __future__ import annotations

import torch
from torch import nn

from .features import BIDDING_FEATURE_DIM, NUM_BID_ACTIONS, NUM_CARDS, TRICK_FEATURE_DIM


class BiddingPolicy(nn.Module):
    """Maps a bidding feature vector to logits over bids 0..MAX_HAND_SIZE."""

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


class CardPlayPolicy(nn.Module):
    """Maps a trick feature vector to logits over the 52 card identities."""

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


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """A Categorical distribution restricted to the actions allowed by `mask`."""
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return torch.distributions.Categorical(logits=masked_logits)
