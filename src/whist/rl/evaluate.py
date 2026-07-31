"""Benchmark trained RL policies against the existing heuristic bots."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch

from ..bots import RandomBot, SimpleBot
from ..game import WhistGame
from ..player import Player
from .agent import RLPlayer
from .policy import BiddingPolicy, CardPlayPolicy
from .train import load_policies, resolve_device

OPPONENT_TYPES = {"simple": SimpleBot, "random": RandomBot}


def build_players(
    bidding_policy: BiddingPolicy,
    card_policy: CardPlayPolicy,
    num_players: int,
    opponent: str,
    rng: random.Random,
    device: torch.device,
) -> list[Player]:
    opponent_cls = OPPONENT_TYPES[opponent]

    players: list[Player] = [
        RLPlayer(
            "RLBot", bidding_policy, card_policy, training=False, deterministic=True, rng=rng, device=device
        )
    ]
    for i in range(num_players - 1):
        players.append(opponent_cls(f"{opponent.capitalize()}{i}", rng=random.Random(rng.randrange(2**32))))
    return players


def evaluate(
    save_dir: str | Path,
    num_games: int = 100,
    num_players: int = 4,
    opponent: str = "simple",
    seed: int | None = None,
    device: str | torch.device | None = "cpu",
) -> dict[str, float]:
    """Play `num_games` full games (RLBot vs `num_players - 1` heuristic
    bots) and return average total score and average final rank per
    player name (lower rank is better; 1 = first place)."""
    rng = random.Random(seed)
    device = resolve_device(device)
    bidding_policy, card_policy = load_policies(save_dir, device=device)
    total_scores: dict[str, float] = defaultdict(float)
    total_ranks: dict[str, float] = defaultdict(float)

    for _ in range(num_games):
        players = build_players(bidding_policy, card_policy, num_players, opponent, rng, device)
        game = WhistGame(players, rng=random.Random(rng.randrange(2**32)))
        game.play_game()
        standings = game.standings()  # [(name, score), ...] best first
        for rank, (name, score) in enumerate(standings, start=1):
            total_scores[name] += score
            total_ranks[name] += rank

    results = {}
    for name in total_scores:
        results[name] = {
            "avg_score": total_scores[name] / num_games,
            "avg_rank": total_ranks[name] / num_games,
        }
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL Whist bot against heuristic bots.")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--opponent", choices=sorted(OPPONENT_TYPES), default="simple")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = evaluate(
        save_dir=args.save_dir,
        num_games=args.num_games,
        num_players=args.num_players,
        opponent=args.opponent,
        seed=args.seed,
        device=args.device,
    )
    print(f"Results over {args.num_games} games ({args.num_players} players, opponent={args.opponent}):")
    for name, stats in sorted(results.items(), key=lambda kv: kv[1]["avg_rank"]):
        print(f"  {name}: avg score {stats['avg_score']:.1f}, avg rank {stats['avg_rank']:.2f}")


if __name__ == "__main__":
    main()
