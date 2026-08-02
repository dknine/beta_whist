"""Self-play REINFORCE training loop for the bidding and card-play policies.

Each round of Contract Whist is treated as an independent episode: a fresh
deal, independent of every other round except for the running total score.
So every decision (the single bid, and each of the hand_size card plays) a
player makes during a round shares that round's score as its Monte-Carlo
return -- there's no cross-round credit assignment to worry about.

The baseline subtracted from the return is a learned critic (actor-critic):
BiddingValueNet/CardValueNet predict V(s), the expected round score from
that decision point, conditioned on the actual state (hand strength, bids
so far, etc.) -- not just a per-hand_size average like the earlier scalar
EMA baseline. Conditioning on state removes far more variance from the
policy gradient, since e.g. "this specific strong hand" and "this specific
weak hand" of the same size no longer share one baseline value. The critic
is trained via Monte-Carlo regression toward the round's actual score
(consistent with this framework's "each round is an independent episode"
design), not TD(0) bootstrapping -- that's Q-learning's (qtrain.py)
distinguishing feature; keeping this critic MC-based keeps a clean
conceptual line between "REINFORCE with a learned baseline" and "true" TD
actor-critic.

The resulting advantage (return minus V(s)) is additionally normalized by a
running per-hand_size std of that residual (RunningBaseline.get_std) before
weighting the policy gradient. This still matters even with a learned
baseline: round scores range roughly 0-17, so un-normalized advantages can
be an order of magnitude larger than a modest entropy bonus -- with too
small an entropy_coef that lets the policy collapse to a degenerate
near-deterministic strategy (e.g. "always bid 0") well before entropy
regularization can push back, which then plateaus training since a policy
with ~zero entropy has nothing left to explore.

Self-play: most seats each game are played by the current ("live") policy,
so gradients from every seat's experience flow into the same shared
weights. To keep the policy robust rather than overfit to beating only
itself, a configurable fraction of seats are instead played by frozen
snapshots of past policy versions, sampled from an opponent pool that's
periodically refreshed with the latest weights. The critic is never used
for acting (RLPlayer only ever samples from the policy), so it plays no
role in the opponent pool -- only the policy weights are snapshotted there.
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
from .critic import BiddingValueNet, CardValueNet
from .policy import BiddingPolicy, CardPlayPolicy

DEFAULT_HIDDEN_BID = 128
DEFAULT_HIDDEN_CARD = 256


class RunningBaseline:
    """An exponential moving average return (and its spread), tracked
    separately per hand_size, used to normalize the REINFORCE advantage.

    Normalizing by std as well as subtracting the mean matters here: round
    scores range roughly 0-17, so raw advantages can be an order of
    magnitude larger than a small entropy bonus, which lets the policy
    collapse to a degenerate near-deterministic strategy well before
    entropy regularization has any chance to push back (see get_std)."""

    def __init__(self, momentum: float = 0.98) -> None:
        self.momentum = momentum
        self._values: dict[int, float] = {}
        self._var: dict[int, float] = {}

    def get(self, hand_size: int) -> float:
        return self._values.get(hand_size, 0.0)

    def get_std(self, hand_size: int, min_std: float = 1.0) -> float:
        """Never returns less than min_std, both to avoid dividing by ~0
        early on (before enough samples exist for a stable estimate) and to
        keep the entropy bonus from becoming irrelevant once the policy
        starts converging and true variance drops toward zero."""
        return max(self._var.get(hand_size, 0.0) ** 0.5, min_std)

    def update(self, hand_size: int, value: float) -> None:
        old_mean = self._values.get(hand_size, value)
        deviation_sq = (value - old_mean) ** 2
        old_var = self._var.get(hand_size, deviation_sq)
        self._values[hand_size] = self.momentum * old_mean + (1 - self.momentum) * value
        self._var[hand_size] = self.momentum * old_var + (1 - self.momentum) * deviation_sq


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


def resolve_device(device: str | torch.device | None) -> torch.device:
    """None/"auto" picks CUDA if available, else CPU."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_seats(
    num_players: int,
    bidding_policy: BiddingPolicy,
    card_policy: CardPlayPolicy,
    pool: OpponentPool,
    opponent_fraction: float,
    rng: random.Random,
    device: torch.device,
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
            seats.append(
                RLPlayer(f"Seat{i}", frozen_bid, frozen_card, training=False, rng=rng, device=device)
            )
        else:
            seats.append(
                RLPlayer(f"Seat{i}", bidding_policy, card_policy, training=True, rng=rng, device=device)
            )
    return seats


def _quick_eval(
    bidding_policy: BiddingPolicy,
    card_policy: CardPlayPolicy,
    num_players: int,
    opponent: str,
    num_games: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[float, float]:
    """Play `num_games` games of the current policy (deterministic) against
    heuristic bots and return (avg_score, avg_rank) for the RL seat. Lazily
    imports from .evaluate to avoid a circular import (evaluate.py imports
    from this module at load time)."""
    from .evaluate import build_players

    total_score = 0.0
    total_rank = 0.0
    for _ in range(num_games):
        players = build_players(bidding_policy, card_policy, num_players, opponent, rng, device)
        game = WhistGame(players, rng=random.Random(rng.randrange(2**32)))
        game.play_game()
        for rank, (name, score) in enumerate(game.standings(), start=1):
            if name == "RLBot":
                total_score += score
                total_rank += rank
                break
    return total_score / num_games, total_rank / num_games


def train(
    iterations: int = 200,
    games_per_iteration: int = 8,
    num_players: int = 4,
    lr_bid: float = 1e-3,
    lr_card: float = 1e-3,
    lr_critic: float = 1e-3,
    entropy_coef: float = 0.05,
    opponent_fraction: float = 0.3,
    snapshot_every: int = 10,
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
) -> tuple[BiddingPolicy, CardPlayPolicy]:
    """Run self-play REINFORCE training.

    Set `resume_from` to a directory previously written by this function
    (typically the same as `save_dir`, across two separate calls/processes)
    to continue training an existing policy instead of starting from random
    weights -- this also restores optimizer momentum and the absolute
    iteration count, so logged iteration numbers keep counting up across
    resumes rather than resetting to 1. The opponent pool itself is *not*
    persisted; it's reseeded from the resumed weights.

    Set `eval_every` > 0 to periodically play `eval_games` deterministic
    games against `eval_opponent` ("simple" or "random") during training and
    log the result -- this is what gives you a learning curve rather than
    just the noisy self-play training score. If `save_dir` is set, each
    evaluation is also appended as a CSV row to `save_dir/eval_log.csv`.
    """
    rng = random.Random(seed)
    device = resolve_device(device)

    if resume_from is not None:
        bidding_policy, card_policy = load_policies(resume_from, device=device)
        bidding_critic, card_critic = load_critics(resume_from, device=device)
        bidding_opt = optim.Adam(bidding_policy.parameters(), lr=lr_bid)
        card_opt = optim.Adam(card_policy.parameters(), lr=lr_card)
        bidding_critic_opt = optim.Adam(bidding_critic.parameters(), lr=lr_critic)
        card_critic_opt = optim.Adam(card_critic.parameters(), lr=lr_critic)
        start_iteration = _load_optimizer_state(resume_from, bidding_opt, card_opt, device)
        _load_critic_optimizer_state(resume_from, bidding_critic_opt, card_critic_opt, device)
    else:
        bidding_policy = BiddingPolicy(DEFAULT_HIDDEN_BID).to(device)
        card_policy = CardPlayPolicy(DEFAULT_HIDDEN_CARD).to(device)
        bidding_critic = BiddingValueNet(DEFAULT_HIDDEN_BID).to(device)
        card_critic = CardValueNet(DEFAULT_HIDDEN_CARD).to(device)
        bidding_opt = optim.Adam(bidding_policy.parameters(), lr=lr_bid)
        card_opt = optim.Adam(card_policy.parameters(), lr=lr_card)
        bidding_critic_opt = optim.Adam(bidding_critic.parameters(), lr=lr_critic)
        card_critic_opt = optim.Adam(card_critic.parameters(), lr=lr_critic)
        start_iteration = 0

    pool = OpponentPool(max_size=pool_size)
    pool.add(bidding_policy, card_policy)  # seed the pool so opponent_fraction has something to sample
    baseline = RunningBaseline()  # now tracks std of the (return - V(s)) residual, not the raw return

    for iteration in range(start_iteration + 1, start_iteration + iterations + 1):
        bid_losses: list[torch.Tensor] = []
        card_losses: list[torch.Tensor] = []
        bid_critic_losses: list[torch.Tensor] = []
        card_critic_losses: list[torch.Tensor] = []
        round_scores: list[float] = []

        for _ in range(games_per_iteration):
            seats = _build_seats(
                num_players, bidding_policy, card_policy, pool, opponent_fraction, rng, device
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
                    ret_tensor = torch.tensor(ret, device=device)

                    for step in steps:
                        value_net = bidding_critic if step.kind == "bid" else card_critic
                        predicted_value = value_net(step.features)
                        residual = ret - predicted_value.detach().item()
                        normalized_advantage = residual / baseline.get_std(hand_size)
                        baseline.update(hand_size, residual)

                        actor_loss = -normalized_advantage * step.log_prob - entropy_coef * step.entropy
                        critic_loss = (predicted_value - ret_tensor) ** 2
                        if step.kind == "bid":
                            bid_losses.append(actor_loss)
                            bid_critic_losses.append(critic_loss)
                        else:
                            card_losses.append(actor_loss)
                            card_critic_losses.append(critic_loss)

        if bid_losses:
            bidding_opt.zero_grad()
            torch.stack(bid_losses).mean().backward()
            bidding_opt.step()
        if card_losses:
            card_opt.zero_grad()
            torch.stack(card_losses).mean().backward()
            card_opt.step()
        if bid_critic_losses:
            bidding_critic_opt.zero_grad()
            torch.stack(bid_critic_losses).mean().backward()
            bidding_critic_opt.step()
        if card_critic_losses:
            card_critic_opt.zero_grad()
            torch.stack(card_critic_losses).mean().backward()
            card_critic_opt.step()

        if iteration % snapshot_every == 0:
            pool.add(bidding_policy, card_policy)
            if save_dir is not None:
                save_policies(bidding_policy, card_policy, save_dir)
                save_critics(bidding_critic, card_critic, save_dir)
                _save_optimizer_state(save_dir, bidding_opt, card_opt, iteration)
                _save_critic_optimizer_state(save_dir, bidding_critic_opt, card_critic_opt)

        if log_every and iteration % log_every == 0:
            avg_score = sum(round_scores) / len(round_scores) if round_scores else 0.0
            on_log(
                f"iter {iteration}  avg round score {avg_score:.2f}  "
                f"bid steps {len(bid_losses)}  card steps {len(card_losses)}  "
                f"pool size {len(pool.snapshots)}"
            )

        if eval_every and iteration % eval_every == 0:
            avg_eval_score, avg_eval_rank = _quick_eval(
                bidding_policy, card_policy, num_players, eval_opponent, eval_games, device, rng
            )
            on_log(
                f"  eval @ iter {iteration}: vs {eval_opponent} over {eval_games} games -> "
                f"avg score {avg_eval_score:.1f}, avg rank {avg_eval_rank:.2f}"
            )
            if save_dir is not None:
                _append_eval_log(save_dir, iteration, avg_eval_score, avg_eval_rank)

    if save_dir is not None:
        save_policies(bidding_policy, card_policy, save_dir)
        save_critics(bidding_critic, card_critic, save_dir)
        _save_optimizer_state(save_dir, bidding_opt, card_opt, start_iteration + iterations)
        _save_critic_optimizer_state(save_dir, bidding_critic_opt, card_critic_opt)

    return bidding_policy, card_policy


def save_policies(bidding_policy: BiddingPolicy, card_policy: CardPlayPolicy, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Always save CPU tensors so checkpoints load fine on machines without a GPU.
    torch.save({k: v.cpu() for k, v in bidding_policy.state_dict().items()}, save_dir / "bidding_policy.pt")
    torch.save({k: v.cpu() for k, v in card_policy.state_dict().items()}, save_dir / "card_policy.pt")


def load_policies(
    save_dir: str | Path, device: str | torch.device | None = "cpu"
) -> tuple[BiddingPolicy, CardPlayPolicy]:
    save_dir = Path(save_dir)
    device = resolve_device(device)
    bidding_policy = BiddingPolicy(DEFAULT_HIDDEN_BID)
    card_policy = CardPlayPolicy(DEFAULT_HIDDEN_CARD)
    bidding_policy.load_state_dict(
        torch.load(save_dir / "bidding_policy.pt", map_location="cpu", weights_only=True)
    )
    card_policy.load_state_dict(
        torch.load(save_dir / "card_policy.pt", map_location="cpu", weights_only=True)
    )
    return bidding_policy.to(device), card_policy.to(device)


def save_critics(bidding_critic: BiddingValueNet, card_critic: CardValueNet, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in bidding_critic.state_dict().items()}, save_dir / "bidding_critic.pt")
    torch.save({k: v.cpu() for k, v in card_critic.state_dict().items()}, save_dir / "card_critic.pt")


def load_critics(
    save_dir: str | Path, device: str | torch.device | None = "cpu"
) -> tuple[BiddingValueNet, CardValueNet]:
    """Loads saved critic weights if present, else returns freshly
    initialized ones -- lets resume_from work against older checkpoints
    written before critics existed (the critic just starts learning from
    scratch again, same as a fresh run's critic would)."""
    save_dir = Path(save_dir)
    device = resolve_device(device)
    bidding_critic = BiddingValueNet(DEFAULT_HIDDEN_BID)
    card_critic = CardValueNet(DEFAULT_HIDDEN_CARD)
    bid_path = save_dir / "bidding_critic.pt"
    card_path = save_dir / "card_critic.pt"
    if bid_path.exists():
        bidding_critic.load_state_dict(torch.load(bid_path, map_location="cpu", weights_only=True))
    if card_path.exists():
        card_critic.load_state_dict(torch.load(card_path, map_location="cpu", weights_only=True))
    return bidding_critic.to(device), card_critic.to(device)


def _save_optimizer_state(save_dir: str | Path, bidding_opt: optim.Optimizer, card_opt: optim.Optimizer, iteration: int) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"bidding_opt": bidding_opt.state_dict(), "card_opt": card_opt.state_dict(), "iteration": iteration},
        save_dir / "training_state.pt",
    )


def _load_optimizer_state(
    save_dir: str | Path, bidding_opt: optim.Optimizer, card_opt: optim.Optimizer, device: torch.device
) -> int:
    """Load optimizer momentum + iteration count if present; returns the
    iteration to resume from (0 if there's no saved training state, e.g. a
    checkpoint written before this feature existed, or the plain output of
    save_policies() alone)."""
    state_path = Path(save_dir) / "training_state.pt"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location=device, weights_only=True)
    bidding_opt.load_state_dict(state["bidding_opt"])
    card_opt.load_state_dict(state["card_opt"])
    return state["iteration"]


def _save_critic_optimizer_state(
    save_dir: str | Path, bidding_critic_opt: optim.Optimizer, card_critic_opt: optim.Optimizer
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"bidding_critic_opt": bidding_critic_opt.state_dict(), "card_critic_opt": card_critic_opt.state_dict()},
        save_dir / "critic_training_state.pt",
    )


def _load_critic_optimizer_state(
    save_dir: str | Path,
    bidding_critic_opt: optim.Optimizer,
    card_critic_opt: optim.Optimizer,
    device: torch.device,
) -> None:
    """No-op (fresh critic optimizer momentum) if there's no saved critic
    training state, e.g. resuming from a checkpoint written before critics
    existed."""
    state_path = Path(save_dir) / "critic_training_state.pt"
    if not state_path.exists():
        return
    state = torch.load(state_path, map_location=device, weights_only=True)
    bidding_critic_opt.load_state_dict(state["bidding_critic_opt"])
    card_critic_opt.load_state_dict(state["card_critic_opt"])


def _append_eval_log(save_dir: str | Path, iteration: int, avg_score: float, avg_rank: float) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "eval_log.csv"
    is_new = not log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("iteration,avg_score,avg_rank\n")
        f.write(f"{iteration},{avg_score},{avg_rank}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Contract Whist RL bots via self-play REINFORCE.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--lr-bid", type=float, default=1e-3)
    parser.add_argument("--lr-card", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Learning rate for the value-baseline critics.")
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Exploration bonus weight. With normalized advantages (~O(1)), too low a value "
        "(e.g. 0.01) lets the policy collapse to a degenerate near-deterministic strategy long "
        "before it's found a good one -- watch the eval_log.csv learning curve for an early "
        "plateau as a symptom.",
    )
    parser.add_argument("--opponent-fraction", type=float, default=0.3)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu', 'cuda', or 'auto' (cuda if available else cpu). Default 'cpu': these are tiny "
        "networks doing single-sample inference interleaved with game logic, so GPU kernel-launch/"
        "transfer overhead usually makes 'cuda' slower here, not faster.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Directory with a previous checkpoint (weights + optimizer state) to continue training "
        "from, instead of starting from random weights. Iteration numbers keep counting up.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Every N iterations, play --eval-games deterministic games against --eval-opponent and "
        "log avg score/rank (and append to save-dir/eval_log.csv). 0 disables periodic evaluation.",
    )
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
        lr_critic=args.lr_critic,
        entropy_coef=args.entropy_coef,
        opponent_fraction=args.opponent_fraction,
        snapshot_every=args.snapshot_every,
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
