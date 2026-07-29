import random

import pytest

from whist.bots import SimpleBot
from whist.cards import Suit
from whist.game import WhistGame
from whist.player import BiddingState, Player, TrickState


class ScriptedPlayer(Player):
    """A player whose bids are pre-scripted; card play falls back to the
    first legal card. Records the forbidden_bid it was shown each time it
    bids, so tests can assert on the screw-the-dealer restriction."""

    def __init__(self, name: str, bids: list[int], trump: Suit = Suit.SPADES) -> None:
        super().__init__(name)
        self._bids = list(bids)
        self._trump = trump
        self.forbidden_seen: list[int | None] = []

    def choose_trump(self, hand):
        return self._trump

    def choose_bid(self, state: BiddingState) -> int:
        self.forbidden_seen.append(state.forbidden_bid)
        return self._bids.pop(0)

    def choose_card(self, state: TrickState):
        return state.valid_cards[0]


@pytest.mark.parametrize("num_players", [3, 4, 5, 6, 7])
def test_round_sequence_always_descends_from_seven_then_ascends(num_players):
    players = [SimpleBot(f"Bot{i}") for i in range(num_players)]
    game = WhistGame(players, rng=random.Random(0))
    seq = game.round_sequence()
    assert seq[0] == 7
    assert seq[-1] == 7
    assert min(seq) == 1
    assert len(seq) == 13
    assert seq == list(range(7, 0, -1)) + list(range(2, 8))


def test_rejects_out_of_range_player_counts():
    with pytest.raises(ValueError):
        WhistGame([SimpleBot("A"), SimpleBot("B")])  # only 2 players


def test_rejects_duplicate_player_names():
    with pytest.raises(ValueError):
        WhistGame([SimpleBot("A"), SimpleBot("A"), SimpleBot("A")])


def test_screw_the_dealer_forbids_exact_total():
    # 3 players, hand_size 3. First two bid 1 each (sum=2), so the dealer
    # bidding 1 would make the total equal 3 (all tricks available) -- forbidden.
    p1 = ScriptedPlayer("P1", bids=[1])  # dealer, will attempt the forbidden bid
    p2 = ScriptedPlayer("P2", bids=[1])
    p3 = ScriptedPlayer("P3", bids=[1])
    game = WhistGame([p1, p2, p3], rng=random.Random(0))
    game.dealer_index = 0  # p1 is dealer; bidding order is p2, p3, p1

    with pytest.raises(ValueError, match="forbidden bid"):
        game.play_round(3)


def test_dealer_sees_correct_forbidden_bid_and_may_avoid_it():
    p1 = ScriptedPlayer("P1", bids=[0])  # dealer avoids the forbidden bid of 1
    p2 = ScriptedPlayer("P2", bids=[1])
    p3 = ScriptedPlayer("P3", bids=[1])
    game = WhistGame([p1, p2, p3], rng=random.Random(0))
    game.dealer_index = 0

    result = game.play_round(3)

    assert p1.forbidden_seen == [1]
    assert result.bids["P1"] == 0


def test_scoring_matches_bid_exactness():
    players = [SimpleBot(f"Bot{i}", rng=random.Random(i)) for i in range(4)]
    game = WhistGame(players, rng=random.Random(7))
    result = game.play_round(5)

    assert sum(result.tricks_won.values()) == 5  # every trick has exactly one winner
    for name in result.bids:
        made_it = result.bids[name] == result.tricks_won[name]
        expected = (10 + result.tricks_won[name]) if made_it else 0
        assert result.scores[name] == expected

    for player in players:
        assert player.total_score == result.scores[player.name]


def test_full_game_runs_to_completion_for_various_player_counts():
    for n in (3, 4, 5, 7):
        players = [SimpleBot(f"Bot{i}", rng=random.Random(i)) for i in range(n)]
        game = WhistGame(players, rng=random.Random(42))
        results = game.play_game()
        assert len(results) == len(game.round_sequence())
        for result in results:
            assert sum(result.tricks_won.values()) == result.hand_size
        standings = game.standings()
        assert len(standings) == n
        assert standings == sorted(standings, key=lambda t: -t[1])
