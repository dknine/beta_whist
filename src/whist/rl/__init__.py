"""Reinforcement-learning framework for training Contract Whist bots.

- `features`: state -> fixed-size tensor encoding, plus legal-action masks.
- `policy`: PyTorch policy networks for bidding and card play.
- `agent.RLPlayer`: a Player driven by those policies (training or frozen).
- `train`: self-play REINFORCE training loop with a past-version opponent pool.
- `evaluate`: benchmark a trained policy against the heuristic bots.
"""

from .agent import RLPlayer
from .policy import BiddingPolicy, CardPlayPolicy
from .train import load_policies, save_policies, train

__all__ = [
    "RLPlayer",
    "BiddingPolicy",
    "CardPlayPolicy",
    "train",
    "save_policies",
    "load_policies",
]
