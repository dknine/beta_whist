"""Player interface plus a console-driven human player."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .cards import Card, Suit


@dataclass
class BiddingState:
    """Everything a player needs to decide on a bid."""

    hand: list[Card]
    hand_size: int
    trump: Suit
    dealer_name: str
    seat_position: int
    num_players: int
    bids_so_far: list[tuple[str, int]]  # (player_name, bid) in bidding order
    is_dealer: bool
    forbidden_bid: int | None  # bid the dealer may not make (screw-the-dealer), else None


@dataclass
class TrickState:
    """Everything a player needs to decide which card to play."""

    hand: list[Card]
    valid_cards: list[Card]
    trump: Suit
    led_suit: Suit | None
    cards_played: list[tuple[str, Card]]  # (player_name, card) so far this trick
    trick_number: int  # 0-indexed within the round


class Player(ABC):
    """Base class for anything that can bid and play cards.

    Round-scoped state (hand, bid, tricks_won) is managed by the game engine
    directly on this object; total_score persists across the whole game.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.hand: list[Card] = []
        self.bid: int | None = None
        self.tricks_won: int = 0
        self.total_score: int = 0

    @abstractmethod
    def choose_trump(self, hand: list[Card]) -> Suit:
        """Return the trump suit for the round. Only called on the dealer."""

    @abstractmethod
    def choose_bid(self, state: BiddingState) -> int:
        """Return a bid between 0 and state.hand_size, honoring forbidden_bid if set."""

    @abstractmethod
    def choose_card(self, state: TrickState) -> Card:
        """Return a card from state.valid_cards to play."""

    def __str__(self) -> str:
        return self.name


class HumanPlayer(Player):
    """A player controlled via terminal input."""

    def choose_trump(self, hand: list[Card]) -> Suit:
        print(f"\n{self.name}, you are the dealer. Your hand: {_format_hand(hand)}")
        suits = {s.value: s for s in Suit}
        while True:
            raw = input(f"Choose trump suit ({'/'.join(suits)}): ").strip().upper()
            if raw in suits:
                return suits[raw]
            print(f"Please enter one of: {', '.join(suits)}")

    def choose_bid(self, state: BiddingState) -> int:
        print(f"\n{self.name}, your hand: {_format_hand(self.hand)}")
        print(f"Trump suit: {state.trump}")
        if state.bids_so_far:
            bid_summary = ", ".join(f"{n}={b}" for n, b in state.bids_so_far)
            print(f"Bids so far: {bid_summary}")
        while True:
            prompt = f"Enter your bid (0-{state.hand_size})"
            if state.forbidden_bid is not None:
                prompt += f", may not be {state.forbidden_bid}"
            prompt += ": "
            raw = input(prompt).strip()
            try:
                bid = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue
            if not (0 <= bid <= state.hand_size):
                print(f"Bid must be between 0 and {state.hand_size}.")
                continue
            if state.forbidden_bid is not None and bid == state.forbidden_bid:
                print(f"As dealer you cannot bid {state.forbidden_bid} (total would equal tricks available).")
                continue
            return bid

    def choose_card(self, state: TrickState) -> Card:
        print(f"\n{self.name}, your hand: {_format_hand(self.hand)}")
        if state.cards_played:
            played_summary = ", ".join(f"{n}: {c}" for n, c in state.cards_played)
            print(f"Played so far: {played_summary}")
        led = f" (must follow {state.led_suit})" if state.led_suit else " (you lead)"
        print(f"Valid cards{led}: {_format_hand(state.valid_cards)}")
        while True:
            raw = input("Enter card to play (e.g. 'AS' for Ace of Spades): ").strip().upper()
            for card in state.valid_cards:
                if raw == str(card.rank).upper() + card.suit.value:
                    return card
                # Also accept the printed unicode form, e.g. "A♠"
                if raw == str(card).upper():
                    return card
            print("That's not a valid card from your hand. Try again.")


def _format_hand(cards: list[Card]) -> str:
    return ", ".join(f"{c.rank}{c.suit.value}" for c in sorted(cards, key=lambda c: (c.suit.value, c.rank.value)))
