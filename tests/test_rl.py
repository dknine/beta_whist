import inspect
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
from whist.rl.critic import BiddingValueNet, CardValueNet
from whist.rl.policy import BiddingPolicy, CardPlayPolicy
from whist.rl.train import (
    OpponentPool,
    RunningBaseline,
    load_critics,
    load_policies,
    save_critics,
    save_policies,
    train,
)


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


def test_running_baseline_get_std_has_floor_and_grows_with_variance():
    baseline = RunningBaseline(momentum=0.5)
    assert baseline.get_std(3) == 1.0  # no data yet -> floor

    baseline.update(3, 10.0)
    assert baseline.get_std(3) == pytest.approx(1.0)  # single sample, zero deviation -> floor

    baseline.update(3, 0.0)  # far from the running mean -> variance should grow past the floor
    assert baseline.get_std(3) > 1.0

    baseline_low_var = RunningBaseline(momentum=0.5)
    baseline_low_var.update(3, 10.0)
    baseline_low_var.update(3, 10.5)  # small deviation -> still floored, shouldn't dip below min_std
    assert baseline_low_var.get_std(3) == pytest.approx(1.0)


def test_default_entropy_coef_is_not_negligible_relative_to_normalized_advantage():
    # Regression guard: with normalized (~O(1)) advantages, entropy_coef needs to be large enough
    # that the exploration bonus isn't swamped -- see the entropy-collapse writeup in train.py's
    # module docstring. 0.01 was observed to let the policy collapse to a degenerate strategy.
    default = inspect.signature(train).parameters["entropy_coef"].default
    assert default >= 0.03


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
    assert (tmp_path / "bidding_critic.pt").exists()
    assert (tmp_path / "card_critic.pt").exists()

    loaded_bid, loaded_card = load_policies(tmp_path)
    for a, b in zip(bidding_policy.parameters(), loaded_bid.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(card_policy.parameters(), loaded_card.parameters()):
        assert torch.equal(a, b)


def test_critic_value_networks_output_scalars():
    bidding_critic = BiddingValueNet()
    card_critic = CardValueNet()
    assert bidding_critic(torch.zeros(BIDDING_FEATURE_DIM)).shape == ()
    assert card_critic(torch.zeros(TRICK_FEATURE_DIM)).shape == ()


def test_save_and_load_critics_round_trip(tmp_path):
    bidding_critic = BiddingValueNet()
    card_critic = CardValueNet()
    save_critics(bidding_critic, card_critic, tmp_path)
    assert (tmp_path / "bidding_critic.pt").exists()
    assert (tmp_path / "card_critic.pt").exists()

    loaded_bid, loaded_card = load_critics(tmp_path)
    for a, b in zip(bidding_critic.parameters(), loaded_bid.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(card_critic.parameters(), loaded_card.parameters()):
        assert torch.equal(a, b)


def test_load_critics_falls_back_to_fresh_when_missing(tmp_path):
    # Simulate a checkpoint written before critics existed: only the
    # policy weights are on disk.
    save_policies(BiddingPolicy(), CardPlayPolicy(), tmp_path)
    bidding_critic, card_critic = load_critics(tmp_path)  # must not raise
    assert isinstance(bidding_critic, BiddingValueNet)
    assert isinstance(card_critic, CardValueNet)


def test_resume_works_against_a_pre_critic_checkpoint(tmp_path):
    # A checkpoint saved via save_policies() alone (no critics, no
    # training_state.pt) should still be resumable -- critics and their
    # optimizers just start fresh, same as train()'s non-resume path.
    save_policies(BiddingPolicy(), CardPlayPolicy(), tmp_path)
    bidding_policy, card_policy = train(
        iterations=1,
        games_per_iteration=1,
        num_players=3,
        resume_from=tmp_path,
        save_dir=tmp_path,
        log_every=0,
        on_log=lambda msg: None,
    )
    for p in bidding_policy.parameters():
        assert torch.isfinite(p).all()


def test_critic_regression_moves_prediction_toward_target():
    # Isolated, deterministic check of the actual mechanism train() relies on:
    # (prediction - target)**2 gradient descent should converge the critic's
    # output toward the target. This is what proves gradients correctly flow
    # into the critic -- a bug that silently detached it (making it dead
    # weight in the loss) would leave the prediction unchanged instead.
    critic = BiddingValueNet()
    opt = torch.optim.Adam(critic.parameters(), lr=0.1)
    features = torch.rand(BIDDING_FEATURE_DIM)
    target = torch.tensor(15.0)

    initial_error = abs(critic(features).item() - 15.0)
    for _ in range(30):
        pred = critic(features)
        loss = (pred - target) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_error = abs(critic(features).item() - 15.0)

    assert final_error < initial_error


def test_actor_critic_training_run_produces_finite_trained_critic(tmp_path):
    train(
        iterations=5,
        games_per_iteration=4,
        num_players=3,
        snapshot_every=5,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        on_log=lambda msg: None,
    )
    bidding_critic, card_critic = load_critics(tmp_path)
    for p in bidding_critic.parameters():
        assert torch.isfinite(p).all()
    for p in card_critic.parameters():
        assert torch.isfinite(p).all()


def test_resume_from_continues_absolute_iteration_count(tmp_path):
    train(
        iterations=2,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=1,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        on_log=lambda msg: None,
    )
    assert (tmp_path / "training_state.pt").exists()
    state = torch.load(tmp_path / "training_state.pt", weights_only=True)
    assert state["iteration"] == 2

    logged = []
    train(
        iterations=3,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=1,
        seed=1,
        save_dir=tmp_path,
        resume_from=tmp_path,
        log_every=1,
        on_log=logged.append,
    )
    state = torch.load(tmp_path / "training_state.pt", weights_only=True)
    assert state["iteration"] == 5  # 2 (previous run) + 3 (this run)
    assert any(msg.startswith("iter 3 ") for msg in logged)  # first resumed iteration is 3, not 1
    assert any(msg.startswith("iter 5 ") for msg in logged)


def test_resume_from_weights_only_checkpoint_starts_at_iteration_zero(tmp_path):
    bidding_policy, card_policy = train(
        iterations=1, games_per_iteration=1, num_players=3, seed=0, log_every=0, on_log=lambda m: None
    )
    save_policies(bidding_policy, card_policy, tmp_path)  # no training_state.pt written

    logged = []
    train(
        iterations=1,
        games_per_iteration=1,
        num_players=3,
        seed=0,
        resume_from=tmp_path,
        log_every=1,
        on_log=logged.append,
    )
    assert any(msg.startswith("iter 1 ") for msg in logged)  # missing training_state.pt -> resumes at 0


def test_eval_every_logs_and_appends_csv(tmp_path):
    logged = []
    train(
        iterations=4,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=2,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        eval_every=2,
        eval_games=2,
        on_log=logged.append,
    )
    assert sum(1 for msg in logged if "eval @ iter" in msg) == 2  # iterations 2 and 4

    log_path = tmp_path / "eval_log.csv"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert lines[0] == "iteration,avg_score,avg_rank"
    assert len(lines) == 3  # header + 2 eval points
