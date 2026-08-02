import random

import pytest

torch = pytest.importorskip("torch")

from whist.bots import RandomBot, SimpleBot
from whist.rl.agent import RLPlayer
from whist.rl.compare import compare, make_seat_factory
from whist.rl.qagent import QAgent
from whist.rl.qtrain import train as train_qlearning
from whist.rl.train import resolve_device, train as train_reinforce


@pytest.fixture(scope="module")
def reinforce_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("reinforce_ckpt")
    train_reinforce(iterations=1, games_per_iteration=1, num_players=3, save_dir=save_dir, log_every=0, on_log=lambda m: None)
    return save_dir


@pytest.fixture(scope="module")
def qlearning_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("qlearning_ckpt")
    train_qlearning(iterations=1, games_per_iteration=1, num_players=3, save_dir=save_dir, log_every=0, on_log=lambda m: None)
    return save_dir


def test_make_seat_factory_simple_and_random():
    device = resolve_device("cpu")
    simple_factory = make_seat_factory("simple", device)
    assert simple_factory.label == "Simple"
    assert isinstance(simple_factory.build("X", random.Random(0)), SimpleBot)

    random_factory = make_seat_factory("random", device)
    assert random_factory.label == "Random"
    assert isinstance(random_factory.build("X", random.Random(0)), RandomBot)


def test_make_seat_factory_reinforce_and_qlearning(reinforce_dir, qlearning_dir):
    device = resolve_device("cpu")
    reinforce_factory = make_seat_factory(f"reinforce:{reinforce_dir}", device)
    assert reinforce_factory.label == "Reinforce"
    assert isinstance(reinforce_factory.build("X", random.Random(0)), RLPlayer)

    qlearning_factory = make_seat_factory(f"qlearning:{qlearning_dir}", device)
    assert qlearning_factory.label == "QLearn"
    assert isinstance(qlearning_factory.build("X", random.Random(0)), QAgent)


def test_make_seat_factory_rejects_unknown_spec():
    with pytest.raises(ValueError, match="Unknown seat spec"):
        make_seat_factory("nonsense", resolve_device("cpu"))


def test_make_seat_factory_requires_path_for_trained_agents():
    with pytest.raises(ValueError, match="checkpoint directory"):
        make_seat_factory("reinforce", resolve_device("cpu"))
    with pytest.raises(ValueError, match="checkpoint directory"):
        make_seat_factory("qlearning", resolve_device("cpu"))


def test_compare_rejects_out_of_range_seat_count():
    with pytest.raises(ValueError):
        compare(["simple", "simple"], num_games=1)  # only 2 seats


def test_compare_heuristics_only():
    results = compare(["simple", "random", "simple"], num_games=3, seed=0)
    assert len(results) == 3
    assert [r["spec"] for r in results] == ["simple", "random", "simple"]
    for r in results:
        assert 1 <= r["avg_rank"] <= 3
        assert r["avg_score"] >= 0


def test_compare_mixed_reinforce_qlearning_and_heuristics(reinforce_dir, qlearning_dir):
    specs = [f"reinforce:{reinforce_dir}", f"qlearning:{qlearning_dir}", "simple", "simple"]
    results = compare(specs, num_games=2, seed=0)
    assert len(results) == 4
    assert [r["spec"] for r in results] == specs
    ranks = [r["avg_rank"] for r in results]
    assert all(1 <= r <= 4 for r in ranks)
