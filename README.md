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
tests/       pytest unit tests
```

## Setup

```powershell
python -m pip install --user -e .[dev]
```

(or just `python -m pip install --user pytest` — the package has no runtime
dependencies beyond the standard library.)

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

## Tests

```powershell
python -m pytest
```
