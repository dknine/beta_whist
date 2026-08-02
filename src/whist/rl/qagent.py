"""QAgent: a Player driven by epsilon-greedy Q-learning.

Contrast with RLPlayer (policy-gradient / REINFORCE): QAgent is off-policy
and value-based. During training it acts epsilon-greedily (random legal
action with probability `epsilon`, else the masked-argmax over Q-values)
and records each decision's (features, action, mask) -- not a log-prob --
since Q-learning trains via regression against a target computed later
(qtrain.py), not via the action's own log-probability.

No gradients are computed at action-selection time here; unlike RLPlayer,
every choose_bid/choose_card call runs under no_grad. Training gradients
come from a separate forward pass over the recorded (features, action)
pairs against a regression target, done by the training loop.

Trump selection is not learned, same as RLPlayer -- the "most-held suit"
heuristic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..cards import Card, Suit
from ..player import BiddingState, Player, TrickState
from .features import (
    ALL_CARDS,
    bidding_action_mask,
    card_action_mask,
    encode_bidding_state,
    encode_trick_state,
)
from .qnetwork import BiddingQNetwork, CardQNetwork, masked_argmax


@dataclass
class QStep:
    kind: str  # "bid" or "card"
    features: torch.Tensor
    action: int
    mask: torch.Tensor


class QAgent(Player):
    def __init__(
        self,
        name: str,
        bidding_q: BiddingQNetwork,
        card_q: CardQNetwork,
        training: bool = False,
        epsilon: float = 0.0,
        rng: random.Random | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(name)
        self.bidding_q = bidding_q
        self.card_q = card_q
        self.training_mode = training
        self.epsilon = epsilon
        self._rng = rng or random.Random()
        self._round_steps: list[QStep] = []
        self.device = torch.device(device)

    def pop_round_steps(self) -> list[QStep]:
        """Return and clear this agent's recorded decisions for the round
        just played, in play order (bid first, then each card played).
        Only non-empty when training_mode is True."""
        steps, self._round_steps = self._round_steps, []
        return steps

    def choose_trump(self, hand: list[Card]) -> Suit:
        counts: dict[Suit, int] = {s: 0 for s in Suit}
        for card in hand:
            counts[card.suit] += 1
        best = max(counts.values())
        candidates = [s for s, n in counts.items() if n == best]
        return self._rng.choice(candidates)

    def choose_bid(self, state: BiddingState) -> int:
        features = encode_bidding_state(state, self.consecutive_zero_bids).to(self.device)
        mask = bidding_action_mask(state).to(self.device)
        legal = mask.nonzero(as_tuple=True)[0].tolist()

        if self.training_mode and self._rng.random() < self.epsilon:
            action = self._rng.choice(legal)
        else:
            with torch.no_grad():
                action = masked_argmax(self.bidding_q(features), mask)

        if self.training_mode:
            self._round_steps.append(QStep(kind="bid", features=features, action=action, mask=mask))
        return action

    def choose_card(self, state: TrickState) -> Card:
        features = encode_trick_state(state, self.bid, self.tricks_won).to(self.device)
        mask = card_action_mask(state.valid_cards).to(self.device)
        legal = mask.nonzero(as_tuple=True)[0].tolist()

        if self.training_mode and self._rng.random() < self.epsilon:
            action = self._rng.choice(legal)
        else:
            with torch.no_grad():
                action = masked_argmax(self.card_q(features), mask)

        if self.training_mode:
            self._round_steps.append(QStep(kind="card", features=features, action=action, mask=mask))
        return ALL_CARDS[action]
