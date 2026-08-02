"""Pit any mix of heuristic bots and trained RL agents against each other.

Each seat is specified as one of:
- "simple"              -- SimpleBot
- "random"               -- RandomBot
- "reinforce:<dir>"      -- RLPlayer loaded from a whist.rl.train checkpoint dir, playing greedily
- "qlearning:<dir>"      -- QAgent loaded from a whist.rl.qtrain checkpoint dir, playing greedily (epsilon=0)

This is the head-to-head tool: unlike evaluate.py (one RL agent vs N copies
of one heuristic type), compare() supports an arbitrary lineup, e.g. one
REINFORCE-trained seat, one Q-learning-trained seat, and two SimpleBots, to
see how the two RL approaches actually stack up against each other and
against the heuristics in the same games.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ..bots import RandomBot, SimpleBot
from ..game import MAX_PLAYERS, MIN_PLAYERS, WhistGame
from ..player import Player
from .agent import RLPlayer
from .qagent import QAgent
from .qtrain import load_qnetworks
from .train import load_policies, resolve_device


@dataclass
class SeatFactory:
    label: str
    build: Callable[[str, random.Random], Player]


def _parse_seat_spec(spec: str) -> tuple[str, str | None]:
    kind, sep, path = spec.partition(":")
    return kind, (path if sep else None)


def make_seat_factory(spec: str, device: torch.device) -> SeatFactory:
    kind, path = _parse_seat_spec(spec)

    if kind == "simple":
        return SeatFactory("Simple", lambda name, rng: SimpleBot(name, rng=random.Random(rng.randrange(2**32))))

    if kind == "random":
        return SeatFactory("Random", lambda name, rng: RandomBot(name, rng=random.Random(rng.randrange(2**32))))

    if kind == "reinforce":
        if path is None:
            raise ValueError("'reinforce:<dir>' needs a checkpoint directory, e.g. 'reinforce:models'")
        bidding_policy, card_policy = load_policies(path, device=device)
        return SeatFactory(
            "Reinforce",
            lambda name, rng: RLPlayer(
                name, bidding_policy, card_policy, training=False, deterministic=True, rng=rng, device=device
            ),
        )

    if kind == "qlearning":
        if path is None:
            raise ValueError("'qlearning:<dir>' needs a checkpoint directory, e.g. 'qlearning:models_qlearning'")
        bidding_q, card_q = load_qnetworks(path, device=device)
        return SeatFactory(
            "QLearn",
            lambda name, rng: QAgent(name, bidding_q, card_q, training=False, epsilon=0.0, rng=rng, device=device),
        )

    raise ValueError(
        f"Unknown seat spec {spec!r}: expected 'simple', 'random', 'reinforce:<dir>', or 'qlearning:<dir>'"
    )


def compare(
    seat_specs: list[str],
    num_games: int = 100,
    seed: int | None = None,
    device: str | torch.device | None = "cpu",
) -> list[dict]:
    """Play `num_games` games with one seat per entry in `seat_specs` and
    return a list of {seat, spec, avg_score, avg_rank} dicts, one per seat,
    in the same order as `seat_specs`. Checkpoints are loaded once up front,
    not reloaded every game."""
    if not (MIN_PLAYERS <= len(seat_specs) <= MAX_PLAYERS):
        raise ValueError(f"Need {MIN_PLAYERS}-{MAX_PLAYERS} seat specs, got {len(seat_specs)}")

    rng = random.Random(seed)
    device = resolve_device(device)
    factories = [make_seat_factory(spec, device) for spec in seat_specs]
    seat_names = [f"seat{i}_{f.label}" for i, f in enumerate(factories)]

    total_scores = [0.0] * len(seat_specs)
    total_ranks = [0.0] * len(seat_specs)

    for _ in range(num_games):
        players = [factories[i].build(seat_names[i], rng) for i in range(len(factories))]
        game = WhistGame(players, rng=random.Random(rng.randrange(2**32)))
        game.play_game()
        standings = dict(game.standings())  # name -> score
        ranked_names = sorted(seat_names, key=lambda n: -standings[n])
        for i, name in enumerate(seat_names):
            total_scores[i] += standings[name]
            total_ranks[i] += ranked_names.index(name) + 1

    return [
        {
            "seat": seat_names[i],
            "spec": seat_specs[i],
            "avg_score": total_scores[i] / num_games,
            "avg_rank": total_ranks[i] / num_games,
        }
        for i in range(len(seat_specs))
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heuristic bots and trained RL agents head-to-head.",
        epilog="Example: python -m whist.rl.compare --seats reinforce:models qlearning:models_qlearning simple simple",
    )
    parser.add_argument(
        "--seats",
        nargs="+",
        required=True,
        help="3-7 seat specs: 'simple', 'random', 'reinforce:<dir>', or 'qlearning:<dir>'.",
    )
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = compare(args.seats, num_games=args.num_games, seed=args.seed, device=args.device)
    print(f"Results over {args.num_games} games:")
    for r in sorted(results, key=lambda r: r["avg_rank"]):
        print(f"  {r['seat']:20s} ({r['spec']}): avg score {r['avg_score']:.1f}, avg rank {r['avg_rank']:.2f}")


if __name__ == "__main__":
    main()
