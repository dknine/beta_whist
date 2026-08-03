"""Interactive terminal entry point for playing Contract Whist."""

from __future__ import annotations

import argparse
import random

from .bots import SimpleBot
from .game import MAX_PLAYERS, MIN_PLAYERS, WhistGame
from .player import HumanPlayer, Player

DEFAULT_REINFORCE_DIR = "models_reinforce_actorcritic"
DEFAULT_QLEARNING_DIR = "models_qlearning"


def _prompt_int(prompt: str, low: int, high: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if not (low <= value <= high):
            print(f"Please enter a number between {low} and {high}.")
            continue
        return value


class _RLBotLoader:
    """Loads the trained REINFORCE/Q-learning checkpoints at most once each
    (multiple seats can share the same loaded weights), and only imports
    whist.rl / torch if an RL bot is actually requested -- the base CLI
    doesn't otherwise need PyTorch installed."""

    def __init__(self, reinforce_dir: str, qlearning_dir: str, device: str) -> None:
        self.reinforce_dir = reinforce_dir
        self.qlearning_dir = qlearning_dir
        self.device = device
        self._reinforce_policies = None
        self._qlearning_networks = None

    def build_reinforce_bot(self, name: str, rng: random.Random) -> Player:
        from .rl.agent import RLPlayer
        from .rl.train import load_policies

        if self._reinforce_policies is None:
            self._reinforce_policies = load_policies(self.reinforce_dir, device=self.device)
        bidding_policy, card_policy = self._reinforce_policies
        return RLPlayer(
            name, bidding_policy, card_policy, training=False, deterministic=True, rng=rng, device=self.device
        )

    def build_qlearning_bot(self, name: str, rng: random.Random) -> Player:
        from .rl.qagent import QAgent
        from .rl.qtrain import load_qnetworks

        if self._qlearning_networks is None:
            self._qlearning_networks = load_qnetworks(self.qlearning_dir, device=self.device)
        bidding_q, card_q = self._qlearning_networks
        return QAgent(name, bidding_q, card_q, training=False, epsilon=0.0, rng=rng, device=self.device)


def _setup_players(rl_loader: _RLBotLoader) -> list[Player]:
    print("=== Contract Whist setup ===")
    n = _prompt_int(
        f"How many players ({MIN_PLAYERS}-{MAX_PLAYERS})? ", MIN_PLAYERS, MAX_PLAYERS
    )
    players: list[Player] = []
    used_names: set[str] = set()
    rng = random.Random()
    for i in range(1, n + 1):
        while True:
            kind = input(
                f"Player {i}: human, bot, reinforce, or qlearning? [h/b/r/q]: "
            ).strip().lower()

            if kind in ("h", "human"):
                default_name = f"Player{i}"
                name = input(f"  Name (default '{default_name}'): ").strip() or default_name
                if name in used_names:
                    print("  That name is already taken.")
                    continue
                used_names.add(name)
                players.append(HumanPlayer(name))
                break

            if kind in ("b", "bot"):
                default_name = f"Bot{i}"
                name = input(f"  Bot name (default '{default_name}'): ").strip() or default_name
                if name in used_names:
                    print("  That name is already taken.")
                    continue
                used_names.add(name)
                players.append(SimpleBot(name))
                break

            if kind in ("r", "reinforce"):
                default_name = f"Reinforce{i}"
                name = input(f"  Bot name (default '{default_name}'): ").strip() or default_name
                if name in used_names:
                    print("  That name is already taken.")
                    continue
                try:
                    player = rl_loader.build_reinforce_bot(name, rng)
                except (ImportError, FileNotFoundError) as e:
                    print(
                        f"  Couldn't load a REINFORCE checkpoint from '{rl_loader.reinforce_dir}' ({e}).\n"
                        f"  Train one first with: python -m whist.rl.train --save-dir "
                        f"{rl_loader.reinforce_dir}\n"
                        f"  Or install the rl extra: pip install -e .[rl]"
                    )
                    continue
                used_names.add(name)
                players.append(player)
                break

            if kind in ("q", "qlearning"):
                default_name = f"QLearn{i}"
                name = input(f"  Bot name (default '{default_name}'): ").strip() or default_name
                if name in used_names:
                    print("  That name is already taken.")
                    continue
                try:
                    player = rl_loader.build_qlearning_bot(name, rng)
                except (ImportError, FileNotFoundError) as e:
                    print(
                        f"  Couldn't load a Q-learning checkpoint from '{rl_loader.qlearning_dir}' ({e}).\n"
                        f"  Train one first with: python -m whist.rl.qtrain --save-dir "
                        f"{rl_loader.qlearning_dir}\n"
                        f"  Or install the rl extra: pip install -e .[rl]"
                    )
                    continue
                used_names.add(name)
                players.append(player)
                break

            print("  Please enter 'h' (human), 'b' (bot), 'r' (reinforce), or 'q' (qlearning).")
    return players


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Contract Whist interactively in the terminal.")
    parser.add_argument(
        "--reinforce-dir",
        type=str,
        default=DEFAULT_REINFORCE_DIR,
        help="Checkpoint directory for the 'reinforce' opponent (from whist.rl.train).",
    )
    parser.add_argument(
        "--qlearning-dir",
        type=str,
        default=DEFAULT_QLEARNING_DIR,
        help="Checkpoint directory for the 'qlearning' opponent (from whist.rl.qtrain).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto' (for RL opponents).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rl_loader = _RLBotLoader(args.reinforce_dir, args.qlearning_dir, args.device)
    players = _setup_players(rl_loader)
    game = WhistGame(players, rng=random.Random(), on_event=print)
    game.play_game()


if __name__ == "__main__":
    main()
