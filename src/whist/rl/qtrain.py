"""Self-play batched Q-learning for the bidding and card-play Q-networks.

Contrast with train.py's REINFORCE: this is off-policy (epsilon-greedy
behavior, greedy target) and value-based (regression toward a target,
rather than policy-gradient ascent on log-probability).

- Bidding is a single decision per round with no follow-up within that
  round, so it's trained as one-step Monte-Carlo / contextual-bandit
  regression: target = that round's score.
- Card play is a genuine sequential decision problem within a round (up to
  MAX_HAND_SIZE plays, no reward until the round's terminal score), trained
  with proper TD bootstrapping across a player's own successive decisions:
  target_t = gamma * max_a' Q_target(s_{t+1}, a') for every play except the
  round's last, and target_T = round score for the last play (a true
  terminal target -- rounds are independent episodes, so there's nothing to
  bootstrap into the next round). Q_target is a periodically-synced frozen
  copy of the live card network, standard DQN-style target stabilization.

Reuses train.py's OpponentPool, resolve_device, checkpoint-optimizer-state
helpers, and eval-log helper directly -- those are generic to "two
optimizers training two networks with a self-play opponent pool" and don't
care whether the networks are policies or Q-networks.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
from torch import optim

from ..game import WhistGame
from .evaluate import OPPONENT_TYPES
from .qagent import QAgent, QStep
from .qnetwork import BiddingQNetwork, CardQNetwork, masked_max
from .train import OpponentPool, _append_eval_log, _load_optimizer_state, _save_optimizer_state, resolve_device

DEFAULT_HIDDEN_BID = 128
DEFAULT_HIDDEN_CARD = 256


def epsilon_at(local_iteration: int, total_iterations: int, eps_start: float, eps_end: float) -> float:
    """Linear decay from eps_start (at local_iteration=1) to eps_end (at
    local_iteration=total_iterations). "Local" because each train() call
    runs its own fresh decay schedule over its own `iterations`, rather
    than continuing a single global schedule across resumed sessions --
    simpler to reason about, at the cost of re-exploring a bit after a
    resume. Document this if you resume many times expecting epsilon to
    keep monotonically shrinking across the whole run."""
    if total_iterations <= 1:
        return eps_end
    frac = min(1.0, (local_iteration - 1) / (total_iterations - 1))
    return eps_start + frac * (eps_end - eps_start)


def _build_seats(
    num_players: int,
    bidding_q: BiddingQNetwork,
    card_q: CardQNetwork,
    pool: OpponentPool,
    opponent_fraction: float,
    opponent_epsilon: float,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> list[QAgent]:
    """One seat is always the live, training network so every game yields
    gradient data; the rest are live with probability (1 - opponent_fraction)
    and a sampled frozen snapshot (playing epsilon-greedily with a fixed,
    usually-small opponent_epsilon rather than 0, for a bit of opponent
    variety) otherwise."""
    guaranteed_live_seat = rng.randrange(num_players)
    seats = []
    for i in range(num_players):
        use_opponent = i != guaranteed_live_seat and pool.snapshots and rng.random() < opponent_fraction
        if use_opponent:
            frozen_bid, frozen_card = pool.sample(rng)
            seats.append(
                QAgent(f"Seat{i}", frozen_bid, frozen_card, training=False, epsilon=opponent_epsilon, rng=rng, device=device)
            )
        else:
            seats.append(
                QAgent(f"Seat{i}", bidding_q, card_q, training=True, epsilon=epsilon, rng=rng, device=device)
            )
    return seats


def _quick_eval(
    bidding_q: BiddingQNetwork,
    card_q: CardQNetwork,
    num_players: int,
    opponent: str,
    num_games: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[float, float]:
    """Play `num_games` games of the current Q-networks (greedy, epsilon=0)
    against a heuristic opponent and return (avg_score, avg_rank) for the
    RL seat."""
    opponent_cls = OPPONENT_TYPES[opponent]
    total_score = 0.0
    total_rank = 0.0
    for _ in range(num_games):
        players = [QAgent("RLBot", bidding_q, card_q, training=False, epsilon=0.0, rng=rng, device=device)]
        for i in range(num_players - 1):
            players.append(opponent_cls(f"{opponent.capitalize()}{i}", rng=random.Random(rng.randrange(2**32))))
        game = WhistGame(players, rng=random.Random(rng.randrange(2**32)))
        game.play_game()
        for rank, (name, score) in enumerate(game.standings(), start=1):
            if name == "RLBot":
                total_score += score
                total_rank += rank
                break
    return total_score / num_games, total_rank / num_games


def card_td_targets(
    card_steps: list[QStep],
    ret: float,
    gamma: float,
    card_target_q: CardQNetwork,
    device: torch.device,
) -> list[torch.Tensor]:
    """TD targets for a player's sequence of card plays within one round:
    gamma * max_a' Q_target(s_{t+1}, a') for every play except the last, and
    the round's final score for the last play (a true terminal target --
    rounds are independent episodes, so there's nothing to bootstrap into
    the next one). Returned in the same order as `card_steps`."""
    n = len(card_steps)
    targets = []
    for t in range(n):
        if t == n - 1:
            targets.append(torch.tensor(float(ret), device=device))
        else:
            with torch.no_grad():
                next_step = card_steps[t + 1]
                targets.append(gamma * masked_max(card_target_q(next_step.features), next_step.mask))
    return targets


def train(
    iterations: int = 200,
    games_per_iteration: int = 8,
    num_players: int = 4,
    lr_bid: float = 1e-3,
    lr_card: float = 1e-3,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    opponent_fraction: float = 0.3,
    opponent_epsilon: float = 0.1,
    snapshot_every: int = 10,
    target_sync_every: int = 10,
    pool_size: int = 10,
    seed: int | None = None,
    save_dir: str | Path | None = None,
    log_every: int = 1,
    on_log: callable = print,
    device: str | torch.device | None = "cpu",
    resume_from: str | Path | None = None,
    eval_every: int = 0,
    eval_opponent: str = "simple",
    eval_games: int = 20,
) -> tuple[BiddingQNetwork, CardQNetwork]:
    """Run self-play batched Q-learning. See train.py's `train()` for the
    resume_from / eval_every semantics -- identical here, for a fair,
    directly-comparable training regimen between the two algorithms."""
    rng = random.Random(seed)
    device = resolve_device(device)

    if resume_from is not None:
        bidding_q, card_q = load_qnetworks(resume_from, device=device)
        bidding_opt = optim.Adam(bidding_q.parameters(), lr=lr_bid)
        card_opt = optim.Adam(card_q.parameters(), lr=lr_card)
        start_iteration = _load_optimizer_state(resume_from, bidding_opt, card_opt, device)
    else:
        bidding_q = BiddingQNetwork(DEFAULT_HIDDEN_BID).to(device)
        card_q = CardQNetwork(DEFAULT_HIDDEN_CARD).to(device)
        bidding_opt = optim.Adam(bidding_q.parameters(), lr=lr_bid)
        card_opt = optim.Adam(card_q.parameters(), lr=lr_card)
        start_iteration = 0

    card_target_q = copy.deepcopy(card_q).eval()
    for p in card_target_q.parameters():
        p.requires_grad_(False)

    pool = OpponentPool(max_size=pool_size)
    pool.add(bidding_q, card_q)  # seed the pool so opponent_fraction has something to sample

    for local_iter in range(1, iterations + 1):
        iteration = start_iteration + local_iter
        epsilon = epsilon_at(local_iter, iterations, epsilon_start, epsilon_end)

        bid_losses: list[torch.Tensor] = []
        card_losses: list[torch.Tensor] = []
        round_scores: list[float] = []

        for _ in range(games_per_iteration):
            seats = _build_seats(
                num_players, bidding_q, card_q, pool, opponent_fraction, opponent_epsilon, epsilon, rng, device
            )
            game = WhistGame(seats, rng=random.Random(rng.randrange(2**32)))

            for hand_size in game.round_sequence():
                result = game.play_round(hand_size)
                for seat in seats:
                    steps = seat.pop_round_steps()
                    if not steps:
                        continue
                    ret = float(result.scores[seat.name])
                    round_scores.append(ret)

                    bid_steps = [s for s in steps if s.kind == "bid"]
                    card_steps = [s for s in steps if s.kind == "card"]

                    for s in bid_steps:
                        pred = bidding_q(s.features)[s.action]
                        target = torch.tensor(ret, device=device)
                        bid_losses.append((pred - target) ** 2)

                    targets = card_td_targets(card_steps, ret, gamma, card_target_q, device)
                    for s, target in zip(card_steps, targets):
                        pred = card_q(s.features)[s.action]
                        card_losses.append((pred - target) ** 2)

        if bid_losses:
            bidding_opt.zero_grad()
            torch.stack(bid_losses).mean().backward()
            bidding_opt.step()
        if card_losses:
            card_opt.zero_grad()
            torch.stack(card_losses).mean().backward()
            card_opt.step()

        if iteration % target_sync_every == 0:
            card_target_q.load_state_dict(card_q.state_dict())

        if iteration % snapshot_every == 0:
            pool.add(bidding_q, card_q)
            if save_dir is not None:
                save_qnetworks(bidding_q, card_q, save_dir)
                _save_optimizer_state(save_dir, bidding_opt, card_opt, iteration)

        if log_every and iteration % log_every == 0:
            avg_score = sum(round_scores) / len(round_scores) if round_scores else 0.0
            on_log(
                f"iter {iteration}  epsilon {epsilon:.3f}  avg round score {avg_score:.2f}  "
                f"bid steps {len(bid_losses)}  card steps {len(card_losses)}  "
                f"pool size {len(pool.snapshots)}"
            )

        if eval_every and iteration % eval_every == 0:
            avg_eval_score, avg_eval_rank = _quick_eval(
                bidding_q, card_q, num_players, eval_opponent, eval_games, device, rng
            )
            on_log(
                f"  eval @ iter {iteration}: vs {eval_opponent} over {eval_games} games -> "
                f"avg score {avg_eval_score:.1f}, avg rank {avg_eval_rank:.2f}"
            )
            if save_dir is not None:
                _append_eval_log(save_dir, iteration, avg_eval_score, avg_eval_rank)

    if save_dir is not None:
        save_qnetworks(bidding_q, card_q, save_dir)
        _save_optimizer_state(save_dir, bidding_opt, card_opt, start_iteration + iterations)

    return bidding_q, card_q


def save_qnetworks(bidding_q: BiddingQNetwork, card_q: CardQNetwork, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in bidding_q.state_dict().items()}, save_dir / "bidding_q.pt")
    torch.save({k: v.cpu() for k, v in card_q.state_dict().items()}, save_dir / "card_q.pt")


def load_qnetworks(
    save_dir: str | Path, device: str | torch.device | None = "cpu"
) -> tuple[BiddingQNetwork, CardQNetwork]:
    save_dir = Path(save_dir)
    device = resolve_device(device)
    bidding_q = BiddingQNetwork(DEFAULT_HIDDEN_BID)
    card_q = CardQNetwork(DEFAULT_HIDDEN_CARD)
    bidding_q.load_state_dict(torch.load(save_dir / "bidding_q.pt", map_location="cpu", weights_only=True))
    card_q.load_state_dict(torch.load(save_dir / "card_q.pt", map_location="cpu", weights_only=True))
    return bidding_q.to(device), card_q.to(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Contract Whist Q-learning bots via self-play.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--lr-bid", type=float, default=1e-3)
    parser.add_argument("--lr-card", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--opponent-fraction", type=float, default=0.3)
    parser.add_argument("--opponent-epsilon", type=float, default=0.1)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--target-sync-every", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="models_qlearning")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-opponent", choices=("simple", "random"), default="simple")
    parser.add_argument("--eval-games", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_players=args.num_players,
        lr_bid=args.lr_bid,
        lr_card=args.lr_card,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        opponent_fraction=args.opponent_fraction,
        opponent_epsilon=args.opponent_epsilon,
        snapshot_every=args.snapshot_every,
        target_sync_every=args.target_sync_every,
        pool_size=args.pool_size,
        seed=args.seed,
        save_dir=args.save_dir,
        log_every=args.log_every,
        device=args.device,
        resume_from=args.resume_from,
        eval_every=args.eval_every,
        eval_opponent=args.eval_opponent,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    main()
