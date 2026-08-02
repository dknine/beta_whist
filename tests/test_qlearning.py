import random

import pytest

torch = pytest.importorskip("torch")

from whist.cards import Suit
from whist.game import WhistGame
from whist.player import BiddingState, TrickState, legal_bids
from whist.rl.features import ALL_CARDS, BIDDING_FEATURE_DIM, NUM_BID_ACTIONS, NUM_CARDS, TRICK_FEATURE_DIM
from whist.rl.qagent import QAgent
from whist.rl.qnetwork import BiddingQNetwork, CardQNetwork, masked_argmax, masked_max
from whist.rl.qtrain import card_td_targets, epsilon_at, load_qnetworks, save_qnetworks, train


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


def test_qnetwork_output_shapes():
    bid_q = BiddingQNetwork()
    card_q = CardQNetwork()
    bid_features = torch.zeros(BIDDING_FEATURE_DIM)
    trick_features = torch.zeros(TRICK_FEATURE_DIM)
    assert bid_q(bid_features).shape == (NUM_BID_ACTIONS,)
    assert card_q(trick_features).shape == (NUM_CARDS,)


def test_masked_argmax_only_considers_legal_actions():
    q_values = torch.tensor([10.0, 0.0, 5.0, 100.0])
    mask = torch.tensor([True, True, True, False])  # index 3 has the highest Q but is illegal
    assert masked_argmax(q_values, mask) == 0  # highest among legal (10.0)


def test_masked_max_matches_masked_argmax_value():
    q_values = torch.tensor([1.0, 7.0, 3.0])
    mask = torch.tensor([True, False, True])  # index 1 excluded despite highest Q
    assert masked_max(q_values, mask).item() == pytest.approx(3.0)


def test_qagent_bid_and_card_are_always_legal():
    bidding_q = BiddingQNetwork()
    card_q = CardQNetwork()
    rng = random.Random(0)
    for training in (True, False):
        agent = QAgent("P", bidding_q, card_q, training=training, epsilon=0.5, rng=rng)
        for _ in range(20):
            state = _bidding_state(
                forbidden_bid=random.choice([None, 0, 1]), zero_bid_forbidden=random.choice([True, False])
            )
            bid = agent.choose_bid(state)
            assert bid in legal_bids(state)

            trick_state = _trick_state(valid_cards=random.sample(ALL_CARDS, 3))
            agent.bid = bid
            agent.tricks_won = 0
            card = agent.choose_card(trick_state)
            assert card in trick_state.valid_cards


def test_training_mode_records_qsteps_in_play_order():
    bidding_q = BiddingQNetwork()
    card_q = CardQNetwork()
    agent = QAgent("P", bidding_q, card_q, training=True, epsilon=0.0, rng=random.Random(0))

    assert agent.pop_round_steps() == []
    agent.choose_bid(_bidding_state())
    agent.choose_card(_trick_state())
    agent.choose_card(_trick_state(trick_number=1))
    steps = agent.pop_round_steps()
    assert [s.kind for s in steps] == ["bid", "card", "card"]
    assert agent.pop_round_steps() == []  # cleared after pop


def test_frozen_qagent_does_not_record_steps():
    bidding_q = BiddingQNetwork()
    card_q = CardQNetwork()
    agent = QAgent("P", bidding_q, card_q, training=False, rng=random.Random(0))
    agent.choose_bid(_bidding_state())
    assert agent.pop_round_steps() == []


def test_full_game_with_qagents_runs_without_error():
    bidding_q = BiddingQNetwork()
    card_q = CardQNetwork()
    for n in (3, 4, 5):
        players = [
            QAgent(f"P{i}", bidding_q, card_q, training=(i == 0), epsilon=0.3, rng=random.Random(i))
            for i in range(n)
        ]
        game = WhistGame(players, rng=random.Random(42))
        result = game.play_round(4)
        assert sum(result.tricks_won.values()) == 4


def test_epsilon_at_linear_decay():
    assert epsilon_at(1, 10, 1.0, 0.0) == pytest.approx(1.0)
    assert epsilon_at(10, 10, 1.0, 0.0) == pytest.approx(0.0)
    assert epsilon_at(5, 9, 1.0, 0.0) == pytest.approx(0.5)  # 4/8 of the way through
    assert epsilon_at(1, 1, 0.7, 0.1) == pytest.approx(0.1)  # single-iteration run -> end value


def test_card_td_targets_bootstraps_except_last_step():
    card_q = CardQNetwork()

    class FakeStep:
        def __init__(self, features, mask):
            self.features = features
            self.mask = mask

    f0, f1 = torch.rand(TRICK_FEATURE_DIM), torch.rand(TRICK_FEATURE_DIM)
    mask0 = torch.ones(NUM_CARDS, dtype=torch.bool)
    mask1 = torch.zeros(NUM_CARDS, dtype=torch.bool)
    mask1[:3] = True  # only first 3 cards legal at the next step

    steps = [FakeStep(f0, mask0), FakeStep(f1, mask1)]
    targets = card_td_targets(steps, ret=12.0, gamma=1.0, card_target_q=card_q, device=torch.device("cpu"))

    assert len(targets) == 2
    # Non-terminal step bootstraps: gamma * max legal Q of the *next* step's state.
    with torch.no_grad():
        expected_bootstrap = masked_max(card_q(f1), mask1)
    assert targets[0].item() == pytest.approx(expected_bootstrap.item())
    # Terminal (last) step's target is just the round's return, not a bootstrap.
    assert targets[1].item() == pytest.approx(12.0)


def test_card_td_targets_applies_gamma_discount():
    card_q = CardQNetwork()
    f0, f1 = torch.rand(TRICK_FEATURE_DIM), torch.rand(TRICK_FEATURE_DIM)
    mask = torch.ones(NUM_CARDS, dtype=torch.bool)

    class FakeStep:
        def __init__(self, features, mask):
            self.features = features
            self.mask = mask

    steps = [FakeStep(f0, mask), FakeStep(f1, mask)]
    targets_full = card_td_targets(steps, ret=10.0, gamma=1.0, card_target_q=card_q, device=torch.device("cpu"))
    targets_half = card_td_targets(steps, ret=10.0, gamma=0.5, card_target_q=card_q, device=torch.device("cpu"))
    assert targets_half[0].item() == pytest.approx(targets_full[0].item() * 0.5)


def test_single_card_round_target_is_just_the_return():
    card_q = CardQNetwork()
    f0 = torch.rand(TRICK_FEATURE_DIM)
    mask = torch.ones(NUM_CARDS, dtype=torch.bool)

    class FakeStep:
        def __init__(self, features, mask):
            self.features = features
            self.mask = mask

    targets = card_td_targets([FakeStep(f0, mask)], ret=7.0, gamma=1.0, card_target_q=card_q, device=torch.device("cpu"))
    assert len(targets) == 1
    assert targets[0].item() == pytest.approx(7.0)


def test_short_qlearning_training_run_completes_and_saves(tmp_path):
    bidding_q, card_q = train(
        iterations=2,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=1,
        target_sync_every=1,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        on_log=lambda msg: None,
    )
    for p in bidding_q.parameters():
        assert torch.isfinite(p).all()
    for p in card_q.parameters():
        assert torch.isfinite(p).all()

    assert (tmp_path / "bidding_q.pt").exists()
    assert (tmp_path / "card_q.pt").exists()

    loaded_bid, loaded_card = load_qnetworks(tmp_path)
    for a, b in zip(bidding_q.parameters(), loaded_bid.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(card_q.parameters(), loaded_card.parameters()):
        assert torch.equal(a, b)


def test_qlearning_resume_continues_absolute_iteration_count(tmp_path):
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
    assert state["iteration"] == 5
    assert any(msg.startswith("iter 3 ") for msg in logged)


def test_qlearning_eval_every_appends_csv(tmp_path):
    train(
        iterations=2,
        games_per_iteration=2,
        num_players=3,
        snapshot_every=1,
        seed=0,
        save_dir=tmp_path,
        log_every=0,
        eval_every=1,
        eval_games=2,
        on_log=lambda m: None,
    )
    log_path = tmp_path / "eval_log.csv"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert lines[0] == "iteration,avg_score,avg_rank"
    assert len(lines) == 3  # header + iterations 1 and 2
