# beta_whist

A Python implementation of Contract Whist for 3-7 players (individual
scoring, no partnerships).

## Rules implemented

- Hand sizes descend from 7 down to 1, then ascend back up to 7
  (7,6,5,4,3,2,1,2,3,4,5,6,7), regardless of player count.
- The dealer rotates each round and chooses trumps for that round.
- Players bid the number of tricks they expect to win, in turn starting to
  the dealer's left, dealer bidding last. The dealer may not make the bid
  that would force total bids to equal the tricks available ("screw the
  dealer" — someone must always be wrong).
- A player may not bid 0 three rounds in a row, except when they are the
  dealer in the 1-card round and forbidden (by the rule above) from
  bidding 1 — in that case 0 is their only legal bid.
- Standard follow-suit trick taking: trump beats non-trump, otherwise
  highest card of the suit led wins.
- Scoring: exact bid scores `10 + tricks taken`; missing the bid still
  scores the tricks taken, just without the bonus.

## Project layout

```
src/whist/
  cards.py   Suit, Rank, Card, Deck
  player.py  Player interface, BiddingState/TrickState, HumanPlayer (terminal input)
  bots.py    RandomBot, SimpleBot (heuristic AI)
  game.py    WhistGame engine: dealing, bidding, tricks, scoring
  cli.py     Interactive terminal game setup and loop
  rl/        Reinforcement-learning framework for training bots (see below)
tests/       pytest unit tests
```

## Setup

```powershell
python -m pip install --user -e .[dev]
```

(or just `python -m pip install --user pytest` — the core package has no
runtime dependencies beyond the standard library. The `rl` subpackage needs
PyTorch — see below.)

## Play interactively

```powershell
python -m whist.cli
```

You'll be asked how many players (3-7) and, for each seat, whether it's
`human`, `bot` (`SimpleBot` heuristic), `reinforce`, or `qlearning` — the
latter two load a trained agent from `whist.rl` (see below) and play it
greedily/deterministically, no exploration. Human players enter bids and
cards (e.g. `AS` for Ace of Spades) via the terminal.

`reinforce`/`qlearning` checkpoints default to `models_reinforce_actorcritic`/
`models_qlearning` (train them first — see the RL sections below); override
with `--reinforce-dir`/`--qlearning-dir`/`--device` if you keep checkpoints
elsewhere:

```powershell
python -m whist.cli --reinforce-dir my_models --qlearning-dir my_qmodels
```

If a checkpoint can't be loaded (missing directory, or PyTorch not
installed), you'll get an explanatory message and be asked to pick again for
that seat rather than crashing.

## Use as a library

```python
import random
from whist.bots import SimpleBot
from whist.game import WhistGame

players = [SimpleBot(f"Bot{i}") for i in range(4)]
game = WhistGame(players, rng=random.Random(42), on_event=print)
game.play_game()
print(game.standings())
```

## Training RL bots

`whist.rl` is a self-contained framework for training bots via reinforcement
learning instead of hand-written heuristics. It needs PyTorch:

```powershell
python -m pip install --user -e .[rl]
```

**Design:**

- **Two separate policies**, each a small PyTorch MLP: a `BiddingPolicy`
  (picks a bid, 0..7) and a `CardPlayPolicy` (picks a card, one of the 52
  identities, masked down to whatever's legal). Trump selection stays a
  fixed heuristic (most-held suit) — it's a much smaller decision than
  bidding or card play and wasn't worth a third policy.
- **`RLPlayer`** (`rl/agent.py`) is a normal `Player` driven by those
  networks. In `training=True` mode it samples from the (legal-action-masked)
  policy distribution and records each decision's log-prob for later use in
  the loss; in `training=False` mode it either samples (for use as a training
  opponent) or plays greedily via `deterministic=True` (for evaluation).
- **Training algorithm** (`rl/train.py`): self-play REINFORCE, actor-critic
  style — `BiddingValueNet`/`CardValueNet` (`rl/critic.py`) are learned value
  baselines, predicting expected round score conditioned on the actual state
  (hand strength, bids so far, etc.), trained via Monte-Carlo regression
  toward the round's real score. The advantage (return minus the critic's
  prediction) is additionally normalized by a running per-hand-size std
  before weighting the policy gradient. Each *round* is treated as an
  independent episode — a fresh deal that doesn't affect any other round —
  so every decision a player makes in a round (the bid, and each card
  played) shares that round's score as its return; the critic is never used
  for acting, only for computing the baseline during training. Most seats
  each game are played by the live policy (so gradients from every seat's
  experience update the same shared weights); a configurable fraction are
  instead played by frozen snapshots of earlier policy versions sampled from
  an opponent pool, to keep the policy robust rather than overfit to only
  beating itself.

  **Why a learned, state-conditioned baseline:** an earlier version used a
  scalar per-hand-size EMA baseline instead of a critic. That's a much
  cruder variance reducer — "the average round score for a 5-card hand"
  gives the same baseline whether you were dealt a strong hand or a weak
  one, so most of the score's actual variance leaks straight into the
  policy gradient as noise. On a 10k-game run with the EMA baseline, REINFORCE
  plateaued almost immediately (~60-65 avg score vs a heuristic bot,
  never improving) even after fixing an earlier entropy-collapse bug and
  tuning the learning rate — the credit-assignment signal was just too
  noisy for the policy to make sense of. The critic addresses this directly.

  **Why the std normalization still matters even with a critic:** round
  scores range roughly 0-17, so un-normalized advantages can be an order of
  magnitude larger than a small entropy bonus. An earlier version of this
  framework used `entropy_coef=0.01` with un-normalized advantages, which
  let training collapse the bidding policy to *always* bid 0 regardless of
  hand strength (entropy 0.000) within the first ~40 iterations — a locally
  decent exploit (better than random) but far below what an adaptive policy
  reaches, and with zero entropy left there was nothing left to explore
  afterward. If your own training run plateaus early with `eval_log.csv`
  going flat, check the policy's output entropy on a few varied hands before
  assuming you just need more iterations — it may need a larger
  `--entropy-coef` instead.
- **`rl/evaluate.py`** benchmarks a trained policy (playing deterministically)
  against `SimpleBot` or `RandomBot` opponents over many games and reports
  average score and average finishing rank.

**Run training:**

```powershell
python -m whist.rl.train --iterations 200 --games-per-iteration 8 --num-players 4 --save-dir models
```

Key flags: `--opponent-fraction` (how often a seat is a frozen past snapshot
instead of the live policy), `--snapshot-every` (how many iterations between
adding a new snapshot to the opponent pool *and* checkpointing to
`--save-dir`), `--entropy-coef` (exploration bonus), `--lr-bid`/`--lr-card`/
`--lr-critic`, `--device` (`cpu` / `cuda` / `auto`).

**Training over multiple sessions:** `train()` normally starts from random
weights every call. Pass `--resume-from <dir>` (typically the same as
`--save-dir`) to continue from a checkpoint instead — this restores both the
policy weights and the Adam optimizer momentum, and iteration numbers in the
logs keep counting up across sessions rather than resetting to 1:

```powershell
python -m whist.rl.train --iterations 1000 --save-dir models
# ...later, or in a different process...
python -m whist.rl.train --iterations 1000 --save-dir models --resume-from models
```

**Tracking a learning curve:** the per-iteration log line is the *self-play*
score, which is noisy and not directly comparable across training stages
(early on, a self-play "win" might just mean out-bidding an equally
untrained opponent). To see whether the policy is actually improving, use
`--eval-every N`: every N iterations it plays `--eval-games` deterministic
games against `--eval-opponent` (`simple` or `random`) and logs the RL bot's
average score/rank, also appending each point to `--save-dir/eval_log.csv`
so you can plot it afterward:

```powershell
python -m whist.rl.train --iterations 1000 --save-dir models --eval-every 50 --eval-games 30
```

**GPU:** `whist.rl` will use CUDA if you pass `--device cuda` (or `auto`
with a GPU present) and a CUDA build of PyTorch is installed
(`pip install torch --index-url https://download.pytorch.org/whl/cu130`, or
whichever tag matches your driver's max supported CUDA version from
`nvidia-smi`). That said, the default is `cpu` on purpose: these are small
MLPs doing single-sample inference interleaved with per-turn game logic
(batch size 1, one forward pass per bid/card decision), so GPU
transfer/kernel-launch overhead tends to make training *slower* than CPU,
not faster, at this scale. GPU would start winning if the training loop
were changed to batch many simultaneous game rollouts into one forward pass
— it doesn't currently do that.

**Evaluate a trained bot:**

```powershell
python -m whist.rl.evaluate --save-dir models --num-games 100 --num-players 4 --opponent simple
```

**Use programmatically:**

```python
from whist.rl.train import train
from whist.rl.evaluate import evaluate

bidding_policy, card_policy = train(iterations=200, games_per_iteration=8, save_dir="models")
print(evaluate("models", num_games=100, opponent="simple"))
```

This is a starting framework, not a tuned one — a few iterations won't
produce a strong bot. Notable simplifications worth knowing about if you
extend it: trump selection isn't learned, the baseline is a simple
per-hand-size EMA rather than a learned value function (no actor-critic),
and there's no opponent-modeling of what other players might hold.

### Q-learning agent (for comparison)

`whist.rl.qtrain` trains a second, algorithmically different bot on the same
game, feature encoding, and self-play scaffold, so the two are a fair,
apples-to-apples comparison rather than differing in ten confounded ways at
once:

|  | REINFORCE (`train.py`) | Q-learning (`qtrain.py`) |
|---|---|---|
| Kind | Policy-gradient, on-policy | Value-based, off-policy |
| Action selection | Sample from learned distribution | Epsilon-greedy over Q-values |
| Bidding target | Advantage-weighted log-prob | Regression to round score (single-step, since there's no follow-up bid within a round — bidding is a contextual bandit) |
| Card-play target | Same round score shared by every card played that round | Proper TD bootstrap: `target_t = gamma * max Q(s_{t+1})` for every play except the round's last, which gets the actual round score (a true terminal target — rounds are independent episodes) |
| Baseline/stabilized by | Learned, state-conditioned critic (`BiddingValueNet`/`CardValueNet`) + advantage std-normalization + entropy bonus | A periodically-synced frozen target network for the bootstrapped card-play targets |

Both share `features.py`'s state encoding, `train.py`'s `OpponentPool`
self-play mechanism, and the resume/`eval_every` machinery, via `qagent.QAgent`
(the Q-learning analog of `RLPlayer`) and `qnetwork.BiddingQNetwork`/`CardQNetwork`
(same I/O shapes as the policy nets, but output raw Q-values, no softmax).

```powershell
python -m whist.rl.qtrain --iterations 1000 --games-per-iteration 10 --save-dir models_qlearning --eval-every 100 --eval-games 30
```

Q-learning-specific flags: `--epsilon-start`/`--epsilon-end` (exploration
rate, linearly decayed over each `train()` call's own `--iterations` —
note this decay resets on `--resume-from`, it doesn't continue a single
schedule across sessions), `--opponent-epsilon` (exploration rate for
frozen opponent-pool seats), `--gamma` (discount for the TD bootstrap,
default 1.0 — rounds are short enough that discounting isn't really
necessary), `--target-sync-every` (how often the frozen target network used
for TD targets gets refreshed from the live card network). `--resume-from`,
`--eval-every`, `--device`, etc. all work the same as `whist.rl.train`.

### Comparing agents head-to-head

`whist.rl.compare` plays an arbitrary mix of heuristic bots and trained
agents in the same games — unlike `evaluate.py` (always one RL agent vs N
copies of one heuristic), this is how you actually pit REINFORCE against
Q-learning against each other:

```powershell
python -m whist.rl.compare --seats reinforce:models qlearning:models_qlearning simple simple --num-games 200
```

Each `--seats` entry is `simple`, `random`, `reinforce:<checkpoint-dir>`, or
`qlearning:<checkpoint-dir>` (3-7 entries); trained agents play
deterministically/greedily (no exploration). Reports average score and
average finishing rank per seat.

## Tests

```powershell
python -m pytest
```
