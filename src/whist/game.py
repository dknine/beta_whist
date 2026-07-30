"""Contract Whist game engine.

Rules implemented:
- 3 to 7 players, individually scored (no partnerships).
- Hand size sequence descends from 7 down to 1, then ascends back up to 7
  (7,6,5,4,3,2,1,2,3,4,5,6,7), regardless of player count.
- The dealer rotates each round and chooses the trump suit for that round.
- Players bid the number of tricks they expect to take, in turn starting
  left of the dealer, dealer bidding last. The dealer may not bid the value
  that would make total bids equal the number of tricks available
  ("screw the dealer" / "hook" rule) -- someone must always be wrong.
- A player may not bid 0 three rounds in a row, except when they are the
  dealer in the 1-card round and forbidden (by the rule above) from
  bidding 1 -- in that situation 0 is their only legal bid.
- Standard follow-suit trick taking; trump beats non-trump; highest card of
  the led suit wins if no trump is played.
- Scoring: a player who takes exactly their bid scores 10 + tricks taken;
  a player who misses their bid still scores the number of tricks taken
  (just without the bonus).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .cards import Card, Deck, Suit
from .player import BiddingState, Player, TrickState

MIN_PLAYERS = 3
MAX_PLAYERS = 7
MAX_HAND_SIZE = 7


@dataclass
class RoundResult:
    hand_size: int
    trump: Suit
    dealer: str
    bids: dict[str, int]
    tricks_won: dict[str, int]
    scores: dict[str, int]


class WhistGame:
    def __init__(
        self,
        players: list[Player],
        rng: random.Random | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        if not (MIN_PLAYERS <= len(players) <= MAX_PLAYERS):
            raise ValueError(
                f"Contract Whist requires between {MIN_PLAYERS} and {MAX_PLAYERS} "
                f"players, got {len(players)}"
            )
        names = [p.name for p in players]
        if len(set(names)) != len(names):
            raise ValueError(f"Player names must be unique, got {names}")

        self.players = players
        self._rng = rng or random.Random()
        self._notify = on_event or (lambda msg: None)
        self.dealer_index = 0
        self.round_results: list[RoundResult] = []

    @property
    def max_hand_size(self) -> int:
        return MAX_HAND_SIZE

    def round_sequence(self) -> list[int]:
        """Hand sizes for a full game: descend to 1, then ascend back up."""
        max_h = self.max_hand_size
        down = list(range(max_h, 0, -1))
        up = list(range(2, max_h + 1))
        return down + up

    def play_game(self) -> list[RoundResult]:
        for hand_size in self.round_sequence():
            self.play_round(hand_size)
        self._notify(self._final_standings_message())
        return self.round_results

    def play_round(self, hand_size: int) -> RoundResult:
        n = len(self.players)
        deck = Deck(self._rng)
        deck.shuffle()
        hands = deck.deal(n, hand_size)
        for player, hand in zip(self.players, hands):
            player.hand = sorted(hand)
            player.bid = None
            player.tricks_won = 0

        dealer = self.players[self.dealer_index]
        trump = dealer.choose_trump(list(dealer.hand))
        self._notify(
            f"\n=== Round: {hand_size} card(s) each | Dealer: {dealer.name} | Trump: {trump} ==="
        )

        self._run_bidding(hand_size, trump, dealer)
        self._run_tricks(hand_size, trump)

        result = self._score_round(hand_size, trump, dealer)
        self.round_results.append(result)
        self._notify(self._round_summary_message(result))

        self.dealer_index = (self.dealer_index + 1) % n
        return result

    def _bidding_order(self) -> list[Player]:
        n = len(self.players)
        start = (self.dealer_index + 1) % n
        return [self.players[(start + i) % n] for i in range(n)]

    def _run_bidding(self, hand_size: int, trump: Suit, dealer: Player) -> None:
        order = self._bidding_order()
        bids_so_far: list[tuple[str, int]] = []
        for i, player in enumerate(order):
            is_dealer = player is dealer
            forbidden: int | None = None
            if is_dealer:
                candidate = hand_size - sum(b for _, b in bids_so_far)
                if 0 <= candidate <= hand_size:
                    forbidden = candidate

            # A player may not bid 0 three rounds in a row, except when they are
            # the dealer in the 1-card round and forbidden (screw-the-dealer) from
            # bidding 1 -- then 0 is their only legal bid.
            dealer_forced_to_zero = is_dealer and hand_size == 1 and forbidden == 1
            zero_bid_forbidden = player.consecutive_zero_bids >= 2 and not dealer_forced_to_zero

            state = BiddingState(
                hand=list(player.hand),
                hand_size=hand_size,
                trump=trump,
                dealer_name=dealer.name,
                seat_position=i,
                num_players=len(order),
                bids_so_far=list(bids_so_far),
                is_dealer=is_dealer,
                forbidden_bid=forbidden,
                zero_bid_forbidden=zero_bid_forbidden,
            )
            bid = player.choose_bid(state)
            if not (0 <= bid <= hand_size):
                raise ValueError(f"{player.name} made an out-of-range bid: {bid}")
            if forbidden is not None and bid == forbidden:
                raise ValueError(
                    f"{player.name} (dealer) made a forbidden bid of {bid} "
                    f"(would make total bids equal {hand_size})"
                )
            if zero_bid_forbidden and bid == 0:
                raise ValueError(
                    f"{player.name} bid 0 for the third round in a row, which is not allowed"
                )
            player.bid = bid
            player.consecutive_zero_bids = player.consecutive_zero_bids + 1 if bid == 0 else 0
            bids_so_far.append((player.name, bid))
            self._notify(f"{player.name} bids {bid}")

    def _run_tricks(self, hand_size: int, trump: Suit) -> None:
        n = len(self.players)
        leader_index = (self.dealer_index + 1) % n
        for trick_no in range(hand_size):
            order = [self.players[(leader_index + i) % n] for i in range(n)]
            cards_played: list[tuple[str, Card]] = []
            led_suit: Suit | None = None
            for player in order:
                valid = self._valid_cards(player.hand, led_suit)
                state = TrickState(
                    hand=list(player.hand),
                    valid_cards=valid,
                    trump=trump,
                    led_suit=led_suit,
                    cards_played=list(cards_played),
                    trick_number=trick_no,
                )
                card = player.choose_card(state)
                if card not in valid:
                    raise ValueError(f"{player.name} played an illegal card: {card}")
                player.hand.remove(card)
                cards_played.append((player.name, card))
                if led_suit is None:
                    led_suit = card.suit

            assert led_suit is not None
            winner_name = self._trick_winner(cards_played, led_suit, trump)
            winner = next(p for p in self.players if p.name == winner_name)
            winner.tricks_won += 1
            self._notify(self._trick_summary_message(trick_no, cards_played, winner_name))
            leader_index = self.players.index(winner)

    @staticmethod
    def _valid_cards(hand: list[Card], led_suit: Suit | None) -> list[Card]:
        if led_suit is None:
            return list(hand)
        following = [c for c in hand if c.suit is led_suit]
        return following if following else list(hand)

    @staticmethod
    def _trick_winner(cards_played: list[tuple[str, Card]], led_suit: Suit, trump: Suit) -> str:
        best_name, best_card = cards_played[0]
        for name, card in cards_played[1:]:
            if card.beats(best_card, led_suit, trump):
                best_name, best_card = name, card
        return best_name

    def _score_round(self, hand_size: int, trump: Suit, dealer: Player) -> RoundResult:
        bids: dict[str, int] = {}
        tricks: dict[str, int] = {}
        scores: dict[str, int] = {}
        for player in self.players:
            assert player.bid is not None
            bids[player.name] = player.bid
            tricks[player.name] = player.tricks_won
            made_it = player.bid == player.tricks_won
            points = player.tricks_won + (10 if made_it else 0)
            scores[player.name] = points
            player.total_score += points
        return RoundResult(
            hand_size=hand_size,
            trump=trump,
            dealer=dealer.name,
            bids=bids,
            tricks_won=tricks,
            scores=scores,
        )

    def standings(self) -> list[tuple[str, int]]:
        return sorted(((p.name, p.total_score) for p in self.players), key=lambda t: -t[1])

    def _round_summary_message(self, result: RoundResult) -> str:
        lines = [f"--- Round result ({result.hand_size} card(s), trump {result.trump}) ---"]
        for name in result.bids:
            made = "made it" if result.bids[name] == result.tricks_won[name] else "missed"
            lines.append(
                f"  {name}: bid {result.bids[name]}, won {result.tricks_won[name]} "
                f"({made}) -> +{result.scores[name]} pts"
            )
        return "\n".join(lines)

    def _trick_summary_message(
        self, trick_no: int, cards_played: list[tuple[str, Card]], winner_name: str
    ) -> str:
        played = ", ".join(f"{n}: {c}" for n, c in cards_played)
        return f"Trick {trick_no + 1}: {played} -> {winner_name} wins"

    def _final_standings_message(self) -> str:
        lines = ["\n=== Final standings ==="]
        for rank, (name, score) in enumerate(self.standings(), start=1):
            lines.append(f"  {rank}. {name}: {score}")
        return "\n".join(lines)
