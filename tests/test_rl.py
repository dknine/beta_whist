import random

import pytest

torch = pytest.importorskip("torch")

from whist.cards import Suit
from whist.game import WhistGame
from whist.player import BiddingState, TrickState, legal_bids
from whist.rl.agent import RLPlayer
from whist.rl.features import (
    ALL_CARDS,
    BIDDING_FEATURE_DIM,
    TRICK_FEATURE_DIM,
    bidding_action_mask,
    card_action_mask,
    encode_bidding_state,
    encode_trick_state,
)
from whist.rl.policy import BiddingPolicy, CardPlayPolicy
from whist.rl.train import OpponentPool, RunningBaseline, load_policies, save_policies, train


def _bidding_state(**overrides):
    defaults = dict(
        hand=ALL_CARDS[:5],
        hand_size=5,
        trump=Suit.SPADES,
        dealer_name="Dealer",
        seat_position=1,
        num_players=4,
        bids_so_far=[("P0", 2)],
        is_dealer=False,
        forbidden_bid=None,
        zero_bid_forbidden=False,
    )
    defaults.update(overrides)
    return BiddingState(**defaults)


def _trick_state(**overrides):
    hand = ALL_CARDS[:5]
    defaults = dict(
        hand=hand,
        valid_cards=hand[:2],
        trump=Suit.SPADES,
        led_suit=None,
        cards_played=[],
        trick_number=0,
        num_players=4,
    )
    defaults.update(overrides)
    return TrickState(**defaults)


def test_encode_bidding_state_matches_declared_dim():
    features = encode_bidding_state(_bidding_state(), consecutive_zero_bids=1)
    assert features.shape == (BIDDING_FEATURE_DIM,)


def test_encode_trick_state_matches_declared_dim():
    features = encode_trick_state(_trick_state(), own_bid=2, own_tricks_won=1)
    assert features.shape == (TRICK_FEATURE_DIM,)


def test_bidding_action_mask_matches_legal_bids():
    state = _bidding_state(hand_size=5, forbidden_bid=2, zero_bid_forbidden=True)
    mask = bidding_action_mask(state)
    allowed = {i for i, ok in enumerate(mask.tolist()) if ok}
    assert allowed == set(legal_bids(state))


def test_card_action_mask_matches_valid_cards():
    state = _trick_state()
    mask = card_action_mask(state.valid_cards)
    allowed_cards = {ALL_CARDS[i] for i, ok in enumerate(mask.tolist()) if ok}
    assert allowed_cards == set(state.valid_cards)


def test_rlplayer_bid_and_card_are_always_legal():
    bidding_policy = BiddingPolicy()
    card_policy = CardPlayPolicy()
    rng = random.Random(0)
    for training in (True, False):
        player = RLPlayer("P", bidding_policy, card_policy, training=training, rng=rng)
        for _ in range(20):
            state = _bidding_state(forbidden_bid=random.choice([None, 0, 1]), zero_bid_forbidden=random.choice([True, False]))
            bid = player.choose_bid(state)
            assert bid in legal_bids(state)

            trick_state = _trick_state(valid_cards=random.sample(ALL_CARDS, 3))
            player.tricks_won = 0
            player.bid = bid
            card = player.choose_card(trick_state)
            assert card in trick_state.valid_cards


def test_training_mode_records_and_pops_trajectory_steps():
    bidding_policy = BiddingPolicy()
    card_policy = CardPlayPolicy()
    player = RLPlayer("P", bidding_policy, card_policy, training=True, rng=random.Random(0))

    assert player.pop_round_steps() == []
    player.choose_bid(_bidding_state())
    player.choose_card(_trick_state())
    steps = player.pop_round_steps()
    assert len(steps) == 2
    assert {s.kind for s in steps} == {"bid", "card"}
    assert all(s.log_prob.requires_grad for s in steps)
    assert player.pop_round_steps() == []  # buffer cleared after pop


def test_frozen_player_does_not_record_steps():
    bidding_policy = BiddingPolicy()
    card_policy = CardPlayPolicy()
    player = RLPlayer("P", bidding_policy, card_policy, training=False, rng=random.Random(0))
    player.choose_bid(_bidding_state())
    assert player.pop_round_steps() == []


def test_full_game_with_rl_players_runs_without_error():
    bidding_policy = BiddingPolicy()
    card_policy = CardPlayPolicy()
    for n in (3, 4, 5):
        players = [
            RLPlayer(f"P{i}", bidding_policy, card_policy, training=(i == 0), rng=random.Random(i))
            for i in range(n)
        ]
        game = WhistGame(players, rng=random.Random(42))
        results = game.play_round(4)
        assert sum(results.tricks_won.values()) == 4


def test_running_baseline_tracks_per_hand_size():
    baseline = RunningBaseline(momentum=0.5)
    assert baseline.get(3) == 0.0
    baseline.update(3, 10.0)
    assert baseline.get(3) == pytest.approx(10.0)  # first observation initializes, no bogus blend from 0
    baseline.update(3, 20.0)
    assert baseline.get(3) == pytest.approx(15.0)  # 0.5 * 10 + 0.5 * 20
    assert baseline.get(5) == 0.0  # different hand_size is independent


def test_opponent_pool_freezes_snapshots():
    bidding_policy = BiddingPolicy()
    card_policy = CardPlayPolicy()
    pool = OpponentPool(max_size=2)
    pool.add(bidding_policy, card_policy)
    frozen_bid, frozen_card = pool.snapshots[0]
    assert all(not p.requires_grad for p in frozen_bid.parameters())
    assert all(not p.requires_grad for p in frozen_card.parameters())

    # Mutating the live policy afterward must not affect the frozen snapshot.
    with torch.no_grad():
        for p in bidding_policy.parameters():
            p.add_(1.0)
    live_param = next(bidding_policy.parameters())
    frozen_param = next(frozen_bid.parameters())
    assert not torch.equal(live_param, frozen_param)

    pool.add(bidding_policy, card_policy)
    pool.add(bidding_policy, card_policy)
    assert len(pool.snapshots) == 2  # max_size enforced, oldest evicted


def test_short_training_run_completes_and_saves(tmp_path):
    bidding_policy, card_policy = train(
        iterations=2,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=1,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        on_log=lambda msg: None,
    )
    for p in bidding_policy.parameters():
        assert torch.isfinite(p).all()
    for p in card_policy.parameters():
        assert torch.isfinite(p).all()

    assert (tmp_path / "bidding_policy.pt").exists()
    assert (tmp_path / "card_policy.pt").exists()

    loaded_bid, loaded_card = load_policies(tmp_path)
    for a, b in zip(bidding_policy.parameters(), loaded_bid.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(card_policy.parameters(), loaded_card.parameters()):
        assert torch.equal(a, b)
