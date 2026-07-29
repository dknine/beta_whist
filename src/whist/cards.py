"""Card, Suit, Rank and Deck primitives for a standard 52-card deck."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"

    def __str__(self) -> str:
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return symbols[self]


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        names = {
            Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
            Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
            Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
            Rank.ACE: "A",
        }
        return names[self]


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"

    def _sort_key(self) -> tuple[str, int]:
        # Plain Enum members aren't orderable, so sort by suit then rank value.
        return (self.suit.value, self.rank.value)

    def __lt__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    def beats(self, other: "Card", led_suit: Suit, trump: Suit | None) -> bool:
        """Return True if this card wins against `other` in a trick.

        Assumes both cards are legal plays in the same trick (`other` was
        played earlier). A card can only win if it is trump or matches the
        led suit; anything else was a discard that can never win the trick.
        """
        self_is_trump = trump is not None and self.suit is trump
        other_is_trump = trump is not None and other.suit is trump

        if self_is_trump and not other_is_trump:
            return True
        if other_is_trump and not self_is_trump:
            return False
        if self_is_trump and other_is_trump:
            return self.rank.value > other.rank.value

        # Neither is trump: only the led suit can win.
        self_follows = self.suit is led_suit
        other_follows = other.suit is led_suit
        if self_follows and not other_follows:
            return True
        if other_follows and not self_follows:
            return False
        if self_follows and other_follows:
            return self.rank.value > other.rank.value
        return False


class Deck:
    """A standard 52-card deck that can be shuffled and dealt from."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self.cards: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def deal(self, num_players: int, hand_size: int) -> list[list[Card]]:
        """Deal `hand_size` cards to each of `num_players` players.

        Raises ValueError if there are not enough cards in the deck.
        """
        needed = num_players * hand_size
        if needed > len(self.cards):
            raise ValueError(
                f"Cannot deal {hand_size} cards to {num_players} players: "
                f"needs {needed} cards but deck only has {len(self.cards)}"
            )
        hands: list[list[Card]] = [[] for _ in range(num_players)]
        for i in range(needed):
            hands[i % num_players].append(self.cards[i])
        return hands

    def __len__(self) -> int:
        return len(self.cards)
