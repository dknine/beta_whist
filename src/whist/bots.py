"""Simple AI opponents.

These are intentionally lightweight heuristics, not a strong AI. `RandomBot`
plays completely randomly (useful for testing). `SimpleBot` makes a rough
trick-count estimate for bidding and then tries to win exactly its bid: it
wins tricks as cheaply as possible while under its bid, and dumps its
highest safe (non-winning) card once it has made its bid.
"""

from __future__ import annotations

import random

from .cards import Card, Rank, Suit
from .player import BiddingState, Player, TrickState


class RandomBot(Player):
    """Bids and plays uniformly at random among legal options."""

    def __init__(self, name: str, rng: random.Random | None = None) -> None:
        super().__init__(name)
        self._rng = rng or random.Random()

    def choose_trump(self, hand: list[Card]) -> Suit:
        return self._rng.choice(list(Suit))

    def choose_bid(self, state: BiddingState) -> int:
        options = [b for b in range(0, state.hand_size + 1) if b != state.forbidden_bid]
        return self._rng.choice(options)

    def choose_card(self, state: TrickState) -> Card:
        return self._rng.choice(state.valid_cards)


class SimpleBot(Player):
    """A basic heuristic bot: estimates tricks for bidding, then tries to
    land exactly on its bid during play."""

    def __init__(self, name: str, rng: random.Random | None = None) -> None:
        super().__init__(name)
        self._rng = rng or random.Random()

    def choose_trump(self, hand: list[Card]) -> Suit:
        counts: dict[Suit, int] = {s: 0 for s in Suit}
        for card in hand:
            counts[card.suit] += 1
        best = max(counts.values())
        candidates = [s for s, n in counts.items() if n == best]
        return self._rng.choice(candidates)

    def choose_bid(self, state: BiddingState) -> int:
        estimate = 0.0
        for card in state.hand:
            if card.suit is state.trump:
                if card.rank.value >= Rank.QUEEN.value:
                    estimate += 1.0
                elif card.rank.value >= Rank.TEN.value:
                    estimate += 0.5
                else:
                    estimate += 0.2
            else:
                if card.rank is Rank.ACE:
                    estimate += 1.0
                elif card.rank is Rank.KING:
                    estimate += 0.7
                elif card.rank is Rank.QUEEN:
                    estimate += 0.3

        bid = max(0, min(state.hand_size, round(estimate)))
        if state.forbidden_bid is not None and bid == state.forbidden_bid:
            if bid == state.hand_size:
                bid -= 1
            elif bid == 0:
                bid += 1
            elif estimate - bid < 0:
                bid -= 1
            else:
                bid += 1
        return bid

    def choose_card(self, state: TrickState) -> Card:
        valid = state.valid_cards
        if len(valid) == 1:
            return valid[0]

        wants_to_win = self.tricks_won < (self.bid or 0)

        if not state.cards_played:
            # Leading the trick: lead high if chasing the bid, low otherwise.
            key = lambda c: c.rank.value
            return max(valid, key=key) if wants_to_win else min(valid, key=key)

        led_suit = state.led_suit
        assert led_suit is not None
        best_so_far: Card | None = None
        for _, played in state.cards_played:
            if best_so_far is None or played.beats(best_so_far, led_suit, state.trump):
                best_so_far = played

        winning = [c for c in valid if c.beats(best_so_far, led_suit, state.trump)]
        losing = [c for c in valid if c not in winning]

        key = lambda c: c.rank.value
        if wants_to_win:
            return min(winning, key=key) if winning else min(valid, key=key)
        return max(losing, key=key) if losing else min(winning, key=key)
