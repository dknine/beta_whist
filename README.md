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

You'll be asked how many players (3-7) and whether each is human or a bot.
Human players enter bids and cards (e.g. `AS` for Ace of Spades) via the
terminal.

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
- **Training algorithm** (`rl/train.py`): self-play REINFORCE with a
  per-hand-size running baseline. Each *round* is treated as an independent
  episode — a fresh deal that doesn't affect any other round — so every
  decision a player makes in a round (the bid, and each card played) shares
  that round's score as its return. Most seats each game are played by the
  live policy (so gradients from every seat's experience update the same
  shared weights); a configurable fraction are instead played by frozen
  snapshots of earlier policy versions sampled from an opponent pool, to
  keep the policy robust rather than overfit to only beating itself.
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
`--save-dir`), `--entropy-coef` (exploration bonus), `--lr-bid`/`--lr-card`,
`--device` (`cpu` / `cuda` / `auto`).

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

## Tests

```powershell
python -m pytest
```
