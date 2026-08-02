"""Reinforcement-learning framework for training Contract Whist bots.

Two algorithms, sharing the same feature encoding (features.py) and
opponent-pool self-play scaffold (train.py's OpponentPool), for a fair
comparison:

- REINFORCE (actor-critic policy gradient): `policy`, `critic` (learned
  value baseline), `agent.RLPlayer`, `train`.
- Q-learning (value-based, off-policy): `qnetwork`, `qagent.QAgent`, `qtrain`.

Plus:
- `features`: state -> fixed-size tensor encoding, plus legal-action masks.
- `evaluate`: benchmark one trained agent against the heuristic bots.
- `compare`: head-to-head comparison of an arbitrary mix of heuristic bots
  and trained REINFORCE/Q-learning agents in the same games.
"""

from .agent import RLPlayer
from .compare import compare
from .critic import BiddingValueNet, CardValueNet
from .policy import BiddingPolicy, CardPlayPolicy
from .qagent import QAgent
from .qnetwork import BiddingQNetwork, CardQNetwork
from .qtrain import load_qnetworks, save_qnetworks
from .qtrain import train as train_qlearning
from .train import load_critics, load_policies, save_critics, save_policies
from .train import train as train_reinforce

__all__ = [
    "RLPlayer",
    "BiddingPolicy",
    "CardPlayPolicy",
    "BiddingValueNet",
    "CardValueNet",
    "train_reinforce",
    "save_policies",
    "load_policies",
    "save_critics",
    "load_critics",
    "QAgent",
    "BiddingQNetwork",
    "CardQNetwork",
    "train_qlearning",
    "save_qnetworks",
    "load_qnetworks",
    "compare",
]
