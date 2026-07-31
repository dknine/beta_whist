"""Self-play REINFORCE training loop for the bidding and card-play policies.

Each round of Contract Whist is treated as an independent episode: a fresh
deal, independent of every other round except for the running total score.
So every decision (the single bid, and each of the hand_size card plays) a
player makes during a round shares that round's score as its Monte-Carlo
return -- there's no cross-round credit assignment to worry about.

Self-play: most seats each game are played by the current ("live") policy,
so gradients from every seat's experience flow into the same shared
weights. To keep the policy robust rather than overfit to beating only
itself, a configurable fraction of seats are instead played by frozen
snapshots of past policy versions, sampled from an opponent pool that's
periodically refreshed with the latest weights.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
from torch import optim

from ..game import WhistGame
from .agent import RLPlayer
from .policy import BiddingPolicy, CardPlayPolicy

DEFAULT_HIDDEN_BID = 128
DEFAULT_HIDDEN_CARD = 256


class RunningBaseline:
    """An exponential moving average return, tracked separately per
    hand_size, used as a variance-reducing baseline for REINFORCE."""

    def __init__(self, momentum: float = 0.98) -> None:
        self.momentum = momentum
        self._values: dict[int, float] = {}

    def get(self, hand_size: int) -> float:
        return self._values.get(hand_size, 0.0)

    def update(self, hand_size: int, value: float) -> None:
        old = self._values.get(hand_size, value)
        self._values[hand_size] = self.momentum * old + (1 - self.momentum) * value


class OpponentPool:
    """Frozen snapshots of past policy versions, used as training opponents."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.snapshots: list[tuple[BiddingPolicy, CardPlayPolicy]] = []

    def add(self, bidding_policy: BiddingPolicy, card_policy: CardPlayPolicy) -> None:
        frozen_bid = copy.deepcopy(bidding_policy).eval()
        frozen_card = copy.deepcopy(card_policy).eval()
        for p in frozen_bid.parameters():
            p.requires_grad_(False)
        for p in frozen_card.parameters():
            p.requires_grad_(False)
        self.snapshots.append((frozen_bid, frozen_card))
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self, rng: random.Random) -> tuple[BiddingPolicy, CardPlayPolicy] | None:
        return rng.choice(self.snapshots) if self.snapshots else None


def _build_seats(
    num_players: int,
    bidding_policy: BiddingPolicy,
    card_policy: CardPlayPolicy,
    pool: OpponentPool,
    opponent_fraction: float,
    rng: random.Random,
) -> list[RLPlayer]:
    """One seat is always the live, training policy so every game yields
    gradient data; the rest are live with probability (1 - opponent_fraction)
    and a sampled frozen snapshot otherwise."""
    guaranteed_live_seat = rng.randrange(num_players)
    seats = []
    for i in range(num_players):
        use_opponent = i != guaranteed_live_seat and pool.snapshots and rng.random() < opponent_fraction
        if use_opponent:
            frozen_bid, frozen_card = pool.sample(rng)
            seats.append(RLPlayer(f"Seat{i}", frozen_bid, frozen_card, training=False, rng=rng))
        else:
            seats.append(RLPlayer(f"Seat{i}", bidding_policy, card_policy, training=True, rng=rng))
    return seats


def train(
    iterations: int = 200,
    games_per_iteration: int = 8,
    num_players: int = 4,
    lr_bid: float = 1e-3,
    lr_card: float = 1e-3,
    entropy_coef: float = 0.01,
    opponent_fraction: float = 0.3,
    snapshot_every: int = 10,
    pool_size: int = 10,
    seed: int | None = None,
    save_dir: str | Path | None = None,
    log_every: int = 1,
    on_log: callable = print,
) -> tuple[BiddingPolicy, CardPlayPolicy]:
    rng = random.Random(seed)

    bidding_policy = BiddingPolicy(DEFAULT_HIDDEN_BID)
    card_policy = CardPlayPolicy(DEFAULT_HIDDEN_CARD)
    bidding_opt = optim.Adam(bidding_policy.parameters(), lr=lr_bid)
    card_opt = optim.Adam(card_policy.parameters(), lr=lr_card)

    pool = OpponentPool(max_size=pool_size)
    pool.add(bidding_policy, card_policy)  # seed the pool so opponent_fraction has something to sample
    baseline = RunningBaseline()

    for iteration in range(1, iterations + 1):
        bid_losses: list[torch.Tensor] = []
        card_losses: list[torch.Tensor] = []
        round_scores: list[float] = []

        for _ in range(games_per_iteration):
            seats = _build_seats(num_players, bidding_policy, card_policy, pool, opponent_fraction, rng)
            game = WhistGame(seats, rng=random.Random(rng.randrange(2**32)))

            for hand_size in game.round_sequence():
                result = game.play_round(hand_size)
                for seat in seats:
                    steps = seat.pop_round_steps()
                    if not steps:
                        continue
                    ret = float(result.scores[seat.name])
                    round_scores.append(ret)
                    advantage = ret - baseline.get(hand_size)
                    for step in steps:
                        loss = -advantage * step.log_prob - entropy_coef * step.entropy
                        (bid_losses if step.kind == "bid" else card_losses).append(loss)
                    baseline.update(hand_size, ret)

        if bid_losses:
            bidding_opt.zero_grad()
            torch.stack(bid_losses).mean().backward()
            bidding_opt.step()
        if card_losses:
            card_opt.zero_grad()
            torch.stack(card_losses).mean().backward()
            card_opt.step()

        if iteration % snapshot_every == 0:
            pool.add(bidding_policy, card_policy)

        if log_every and iteration % log_every == 0:
            avg_score = sum(round_scores) / len(round_scores) if round_scores else 0.0
            on_log(
                f"iter {iteration}/{iterations}  avg round score {avg_score:.2f}  "
                f"bid steps {len(bid_losses)}  card steps {len(card_losses)}  "
                f"pool size {len(pool.snapshots)}"
            )

    if save_dir is not None:
        save_policies(bidding_policy, card_policy, save_dir)

    return bidding_policy, card_policy


def save_policies(bidding_policy: BiddingPolicy, card_policy: CardPlayPolicy, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bidding_policy.state_dict(), save_dir / "bidding_policy.pt")
    torch.save(card_policy.state_dict(), save_dir / "card_policy.pt")


def load_policies(save_dir: str | Path) -> tuple[BiddingPolicy, CardPlayPolicy]:
    save_dir = Path(save_dir)
    bidding_policy = BiddingPolicy(DEFAULT_HIDDEN_BID)
    card_policy = CardPlayPolicy(DEFAULT_HIDDEN_CARD)
    bidding_policy.load_state_dict(torch.load(save_dir / "bidding_policy.pt", weights_only=True))
    card_policy.load_state_dict(torch.load(save_dir / "card_policy.pt", weights_only=True))
    return bidding_policy, card_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Contract Whist RL bots via self-play REINFORCE.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--lr-bid", type=float, default=1e-3)
    parser.add_argument("--lr-card", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--opponent-fraction", type=float, default=0.3)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_players=args.num_players,
        lr_bid=args.lr_bid,
        lr_card=args.lr_card,
        entropy_coef=args.entropy_coef,
        opponent_fraction=args.opponent_fraction,
        snapshot_every=args.snapshot_every,
        pool_size=args.pool_size,
        seed=args.seed,
        save_dir=args.save_dir,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
