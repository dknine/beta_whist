"""Fixed-size feature encoding and legal-action masks for the RL policies.

Two decisions get encoded:
- Bidding: a discrete choice of 0..MAX_HAND_SIZE tricks.
- Card play: a discrete choice over the 52 possible card identities (masked
  down to whatever's actually in hand and legal to play).
"""

from __future__ import annotations

import torch

from ..cards import Card, Rank, Suit
from ..game import MAX_HAND_SIZE, MAX_PLAYERS
from ..player import BiddingState, TrickState, legal_bids

ALL_CARDS: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]
CARD_INDEX: dict[Card, int] = {card: i for i, card in enumerate(ALL_CARDS)}
NUM_CARDS = len(ALL_CARDS)  # 52

SUITS: list[Suit] = list(Suit)
SUIT_INDEX: dict[Suit, int] = {s: i for i, s in enumerate(SUITS)}
NUM_SUITS = len(SUITS)  # 4
SUIT_ONEHOT_DIM = NUM_SUITS + 1  # +1 slot for "no suit" (no card led yet)

NUM_BID_ACTIONS = MAX_HAND_SIZE + 1  # bids 0..7

_BIDDING_SCALAR_DIM = 10
BIDDING_FEATURE_DIM = NUM_CARDS + SUIT_ONEHOT_DIM + _BIDDING_SCALAR_DIM

_TRICK_SCALAR_DIM = 5
TRICK_FEATURE_DIM = 2 * NUM_CARDS + 2 * SUIT_ONEHOT_DIM + _TRICK_SCALAR_DIM


def _hand_multihot(cards: list[Card]) -> torch.Tensor:
    vec = torch.zeros(NUM_CARDS)
    for card in cards:
        vec[CARD_INDEX[card]] = 1.0
    return vec


def _suit_onehot(suit: Suit | None) -> torch.Tensor:
    vec = torch.zeros(SUIT_ONEHOT_DIM)
    vec[SUIT_INDEX[suit] if suit is not None else NUM_SUITS] = 1.0
    return vec


def encode_bidding_state(state: BiddingState, consecutive_zero_bids: int) -> torch.Tensor:
    """Encode a BiddingState (plus the bidder's own zero-bid streak, which
    lives on the Player object rather than the state) into a fixed-size
    feature vector of length BIDDING_FEATURE_DIM."""
    seats_before_dealer = max(state.num_players - 1, 1)
    bids_sum = sum(b for _, b in state.bids_so_far)
    scalars = torch.tensor(
        [
            state.hand_size / MAX_HAND_SIZE,
            state.seat_position / seats_before_dealer,
            state.num_players / MAX_PLAYERS,
            1.0 if state.is_dealer else 0.0,
            len(state.bids_so_far) / seats_before_dealer,
            (bids_sum / state.hand_size) if state.hand_size else 0.0,
            (state.forbidden_bid / state.hand_size)
            if state.forbidden_bid is not None and state.hand_size
            else 0.0,
            1.0 if state.forbidden_bid is not None else 0.0,
            1.0 if state.zero_bid_forbidden else 0.0,
            min(consecutive_zero_bids, 2) / 2.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([_hand_multihot(state.hand), _suit_onehot(state.trump), scalars])


def bidding_action_mask(state: BiddingState) -> torch.Tensor:
    mask = torch.zeros(NUM_BID_ACTIONS, dtype=torch.bool)
    for bid in legal_bids(state):
        mask[bid] = True
    return mask


def encode_trick_state(state: TrickState, own_bid: int | None, own_tricks_won: int) -> torch.Tensor:
    """Encode a TrickState (plus the player's own bid/tricks-won-so-far,
    which live on the Player object) into a fixed-size feature vector of
    length TRICK_FEATURE_DIM."""
    cards_played_vec = torch.zeros(NUM_CARDS)
    for _, card in state.cards_played:
        cards_played_vec[CARD_INDEX[card]] = 1.0
    seats_before_me = max(state.num_players - 1, 1)
    scalars = torch.tensor(
        [
            state.trick_number / MAX_HAND_SIZE,
            len(state.cards_played) / seats_before_me,
            (own_bid or 0) / MAX_HAND_SIZE,
            own_tricks_won / MAX_HAND_SIZE,
            state.num_players / MAX_PLAYERS,
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [
            _hand_multihot(state.hand),
            cards_played_vec,
            _suit_onehot(state.trump),
            _suit_onehot(state.led_suit),
            scalars,
        ]
    )


def card_action_mask(valid_cards: list[Card]) -> torch.Tensor:
    mask = torch.zeros(NUM_CARDS, dtype=torch.bool)
    for card in valid_cards:
        mask[CARD_INDEX[card]] = True
    return mask
