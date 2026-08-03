import random

import pytest

torch = pytest.importorskip("torch")

from whist.bots import SimpleBot
from whist.cli import _RLBotLoader, _setup_players
from whist.player import HumanPlayer
from whist.rl.agent import RLPlayer
from whist.rl.qagent import QAgent
from whist.rl.qtrain import train as train_qlearning
from whist.rl.train import train as train_reinforce


@pytest.fixture(scope="module")
def reinforce_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("cli_reinforce_ckpt")
    train_reinforce(
        iterations=1, games_per_iteration=1, num_players=3, save_dir=save_dir, log_every=0, on_log=lambda m: None
    )
    return str(save_dir)


@pytest.fixture(scope="module")
def qlearning_dir(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("cli_qlearning_ckpt")
    train_qlearning(
        iterations=1, games_per_iteration=1, num_players=3, save_dir=save_dir, log_every=0, on_log=lambda m: None
    )
    return str(save_dir)


def _feed(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_setup_players_human_and_bot(monkeypatch):
    # MIN_PLAYERS is 3: human "Alice", then two plain bots with default names.
    _feed(monkeypatch, ["3", "h", "Alice", "b", "", "b", ""])
    loader = _RLBotLoader("nonexistent", "nonexistent", "cpu")
    players = _setup_players(loader)
    assert len(players) == 3
    assert isinstance(players[0], HumanPlayer)
    assert players[0].name == "Alice"
    assert isinstance(players[1], SimpleBot)
    assert players[1].name == "Bot2"
    assert isinstance(players[2], SimpleBot)


def test_setup_players_reinforce_and_qlearning(reinforce_dir, qlearning_dir, monkeypatch):
    _feed(monkeypatch, ["3", "r", "", "q", "", "b", ""])
    loader = _RLBotLoader(reinforce_dir, qlearning_dir, "cpu")
    players = _setup_players(loader)
    assert len(players) == 3
    assert isinstance(players[0], RLPlayer)
    assert players[0].name == "Reinforce1"
    assert isinstance(players[1], QAgent)
    assert players[1].name == "QLearn2"
    assert isinstance(players[2], SimpleBot)


def test_setup_players_retries_after_missing_checkpoint(monkeypatch):
    # Player 1 picks 'r' with a bad directory (name prompt still fires, then the
    # load fails and they're re-prompted for a kind); falls back to a plain bot.
    _feed(monkeypatch, ["3", "r", "Bot", "b", "", "b", "", "b", ""])
    loader = _RLBotLoader("nonexistent_dir", "nonexistent_dir", "cpu")
    players = _setup_players(loader)
    assert len(players) == 3
    assert all(isinstance(p, SimpleBot) for p in players)


def test_setup_players_rejects_duplicate_names(monkeypatch):
    # Rejecting a duplicate name re-prompts for kind, not just the name.
    _feed(monkeypatch, ["3", "b", "Same", "b", "Same", "b", "Other", "b", ""])
    loader = _RLBotLoader("nonexistent", "nonexistent", "cpu")
    players = _setup_players(loader)
    assert [p.name for p in players] == ["Same", "Other", "Bot3"]


def test_rl_bot_loader_caches_loaded_weights(reinforce_dir, qlearning_dir):
    loader = _RLBotLoader(reinforce_dir, qlearning_dir, "cpu")
    rng = random.Random(0)
    p1 = loader.build_reinforce_bot("A", rng)
    p2 = loader.build_reinforce_bot("B", rng)
    assert p1.bidding_policy is p2.bidding_policy  # loaded once, shared, not reloaded per seat

    q1 = loader.build_qlearning_bot("C", rng)
    q2 = loader.build_qlearning_bot("D", rng)
    assert q1.bidding_q is q2.bidding_q


def test_rl_bot_loader_raises_on_missing_checkpoint():
    loader = _RLBotLoader("nonexistent_dir", "nonexistent_dir", "cpu")
    with pytest.raises(FileNotFoundError):
        loader.build_reinforce_bot("A", random.Random(0))
    with pytest.raises(FileNotFoundError):
        loader.build_qlearning_bot("B", random.Random(0))
