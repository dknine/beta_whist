"""Interactive terminal entry point for playing Contract Whist."""

from __future__ import annotations

import random

from .bots import SimpleBot
from .game import MAX_PLAYERS, MIN_PLAYERS, WhistGame
from .player import HumanPlayer, Player


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


def _setup_players() -> list[Player]:
    print("=== Contract Whist setup ===")
    n = _prompt_int(
        f"How many players ({MIN_PLAYERS}-{MAX_PLAYERS})? ", MIN_PLAYERS, MAX_PLAYERS
    )
    players: list[Player] = []
    used_names: set[str] = set()
    for i in range(1, n + 1):
        while True:
            kind = input(f"Player {i}: human or bot? [h/b]: ").strip().lower()
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
            print("  Please enter 'h' or 'b'.")
    return players


def main() -> None:
    players = _setup_players()
    game = WhistGame(players, rng=random.Random(), on_event=print)
    game.play_game()


if __name__ == "__main__":
    main()
