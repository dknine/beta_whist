"""RLPlayer: a Player driven by the bidding/card-play policy networks.

Two modes:
- training=True: samples stochastically from the (masked) policy
  distribution and records each decision's log-prob/entropy (with the
  autograd graph attached) into a per-round buffer for the training loop to
  turn into a REINFORCE loss.
- training=False: no recording, no grad. Samples stochastically by default
  (useful as a training opponent for diversity) or greedily if
  `deterministic=True` (useful for evaluating a trained policy's best play).

Trump selection is not learned -- it uses the same "most-held suit"
heuristic as SimpleBot, since it's a much smaller, less interesting decision
than bidding or card play.
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
from .policy import BiddingPolicy, CardPlayPolicy, masked_categorical


@dataclass
class TrajectoryStep:
    kind: str  # "bid" or "card"
    log_prob: torch.Tensor  # scalar tensor, still attached to the autograd graph
    entropy: torch.Tensor  # scalar tensor


class RLPlayer(Player):
    def __init__(
        self,
        name: str,
        bidding_policy: BiddingPolicy,
        card_policy: CardPlayPolicy,
        training: bool = False,
        deterministic: bool = False,
        rng: random.Random | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(name)
        self.bidding_policy = bidding_policy
        self.card_policy = card_policy
        self.training_mode = training
        self.deterministic = deterministic
        self._rng = rng or random.Random()
        self._round_steps: list[TrajectoryStep] = []
        self.device = torch.device(device)

    def pop_round_steps(self) -> list[TrajectoryStep]:
        """Return and clear this player's recorded decisions for the round
        just played. Only non-empty when training_mode is True."""
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

        if self.training_mode:
            logits = self.bidding_policy(features)
            dist = masked_categorical(logits, mask)
            action = dist.sample()
            self._round_steps.append(
                TrajectoryStep(kind="bid", log_prob=dist.log_prob(action), entropy=dist.entropy())
            )
        else:
            with torch.no_grad():
                logits = self.bidding_policy(features)
                if self.deterministic:
                    action = torch.argmax(logits.masked_fill(~mask, float("-inf")))
                else:
                    action = masked_categorical(logits, mask).sample()

        return int(action.item())

    def choose_card(self, state: TrickState) -> Card:
        features = encode_trick_state(state, self.bid, self.tricks_won).to(self.device)
        mask = card_action_mask(state.valid_cards).to(self.device)

        if self.training_mode:
            logits = self.card_policy(features)
            dist = masked_categorical(logits, mask)
            action = dist.sample()
            self._round_steps.append(
                TrajectoryStep(kind="card", log_prob=dist.log_prob(action), entropy=dist.entropy())
            )
        else:
            with torch.no_grad():
                logits = self.card_policy(features)
                if self.deterministic:
                    action = torch.argmax(logits.masked_fill(~mask, float("-inf")))
                else:
                    action = masked_categorical(logits, mask).sample()

        return ALL_CARDS[int(action.item())]
