import random

import pytest

from whist.cards import Card, Deck, Rank, Suit


def test_deck_has_52_unique_cards():
    deck = Deck()
    assert len(deck) == 52
    assert len(set(deck.cards)) == 52


def test_shuffle_preserves_all_cards():
    deck = Deck(random.Random(42))
    before = set(deck.cards)
    deck.shuffle()
    assert set(deck.cards) == before


def test_deal_distributes_correct_counts():
    deck = Deck(random.Random(1))
    deck.shuffle()
    hands = deck.deal(4, 7)
    assert len(hands) == 4
    assert all(len(h) == 7 for h in hands)
    all_cards = [c for hand in hands for c in hand]
    assert len(set(all_cards)) == 28  # no duplicates dealt


def test_deal_raises_when_not_enough_cards():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.deal(7, 8)  # needs 56 cards, only 52 available


def test_sorting_hand_does_not_raise():
    deck = Deck(random.Random(0))
    deck.shuffle()
    hand = deck.cards[:13]
    sorted(hand)  # must not raise TypeError


def test_higher_rank_same_suit_wins():
    king = Card(Rank.KING, Suit.HEARTS)
    ten = Card(Rank.TEN, Suit.HEARTS)
    assert king.beats(ten, led_suit=Suit.HEARTS, trump=Suit.SPADES)
    assert not ten.beats(king, led_suit=Suit.HEARTS, trump=Suit.SPADES)


def test_trump_beats_higher_non_trump():
    low_trump = Card(Rank.TWO, Suit.SPADES)
    ace_off_suit_follow = Card(Rank.ACE, Suit.HEARTS)
    assert low_trump.beats(ace_off_suit_follow, led_suit=Suit.HEARTS, trump=Suit.SPADES)
    assert not ace_off_suit_follow.beats(low_trump, led_suit=Suit.HEARTS, trump=Suit.SPADES)


def test_off_suit_discard_never_wins():
    led = Card(Rank.TWO, Suit.HEARTS)
    discard = Card(Rank.ACE, Suit.CLUBS)  # neither trump nor led suit
    assert not discard.beats(led, led_suit=Suit.HEARTS, trump=Suit.SPADES)


def test_higher_trump_wins_trump_battle():
    king_trump = Card(Rank.KING, Suit.SPADES)
    ace_trump = Card(Rank.ACE, Suit.SPADES)
    assert ace_trump.beats(king_trump, led_suit=Suit.HEARTS, trump=Suit.SPADES)
