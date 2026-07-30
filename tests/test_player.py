from whist.cards import Suit
from whist.player import BiddingState, legal_bids


def _state(hand_size, forbidden_bid=None, zero_bid_forbidden=False):
    return BiddingState(
        hand=[],
        hand_size=hand_size,
        trump=Suit.SPADES,
        dealer_name="Dealer",
        seat_position=0,
        num_players=3,
        bids_so_far=[],
        is_dealer=False,
        forbidden_bid=forbidden_bid,
        zero_bid_forbidden=zero_bid_forbidden,
    )


def test_legal_bids_no_restrictions():
    assert legal_bids(_state(3)) == [0, 1, 2, 3]


def test_legal_bids_excludes_forbidden_bid():
    assert legal_bids(_state(3, forbidden_bid=2)) == [0, 1, 3]


def test_legal_bids_excludes_zero_when_forbidden():
    assert legal_bids(_state(3, zero_bid_forbidden=True)) == [1, 2, 3]


def test_legal_bids_combines_both_restrictions():
    assert legal_bids(_state(3, forbidden_bid=2, zero_bid_forbidden=True)) == [1, 3]


def test_legal_bids_one_card_hand_with_zero_forbidden_and_one_available():
    # The screw-the-dealer exception scenario: forbidden_bid=1 means 0 is the
    # only option, so zero_bid_forbidden must not also exclude it here -- the
    # game engine is responsible for not setting zero_bid_forbidden in that case.
    assert legal_bids(_state(1, forbidden_bid=1, zero_bid_forbidden=False)) == [0]
