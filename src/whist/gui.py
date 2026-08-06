"""Tkinter GUI for playing Contract Whist against bots or trained RL agents.

Architecture: the game engine (`WhistGame`) runs unmodified in a background
thread. Every seat is wrapped in `_ObservedPlayer`, which reports each
decision to the GUI thread via a `queue.Queue` of small dict "events" and,
for the one human seat, blocks the game thread on a second queue until the
GUI supplies an answer (a button click). Tkinter polls the event queue with
`after()` so all widget updates happen on the main thread.
"""

from __future__ import annotations

import queue
import random
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .bots import RandomBot, SimpleBot
from .cards import Card, Suit
from .game import MAX_PLAYERS, MIN_PLAYERS, WhistGame
from .player import BiddingState, Player, TrickState, legal_bids

DEFAULT_REINFORCE_DIR = "models_reinforce_actorcritic"
DEFAULT_QLEARNING_DIR = "models_qlearning"

BOT_DELAY = 0.45  # seconds paused before a non-human decision, so the table is watchable

SUIT_SYMBOLS = {Suit.CLUBS: "♣", Suit.DIAMONDS: "♦", Suit.HEARTS: "♥", Suit.SPADES: "♠"}
RED_SUITS = {Suit.DIAMONDS, Suit.HEARTS}

KIND_LABELS = {
    "human": "You (human)",
    "simple": "Simple Bot",
    "random": "Random Bot",
    "reinforce": "REINFORCE Bot",
    "qlearning": "Q-Learning Bot",
}
LABEL_TO_KIND = {v: k for k, v in KIND_LABELS.items()}
NAME_PREFIX = {"human": "You", "simple": "Bot", "random": "Random", "reinforce": "Reinforce", "qlearning": "QLearn"}


def _card_color_for_suit(suit: Suit) -> str:
    return "#c0392b" if suit in RED_SUITS else "black"


def _card_color(card: Card) -> str:
    return _card_color_for_suit(card.suit)


def _card_text(card: Card) -> str:
    return f"{card.rank}{SUIT_SYMBOLS[card.suit]}"


def _trick_winner(cards_played: list[tuple[str, Card]], led_suit: Suit, trump: Suit | None) -> str:
    best_name, best_card = cards_played[0]
    for name, card in cards_played[1:]:
        if card.beats(best_card, led_suit, trump):
            best_name, best_card = name, card
    return best_name


class _ObservedPlayer(Player):
    """Wraps a real Player so the GUI can watch every decision (bots
    included) and, for the human seat, block the game thread until the GUI
    supplies an answer."""

    def __init__(
        self,
        name: str,
        inner: Player | None,
        notify: "queue.Queue[dict].put",
        human_inbox: "queue.Queue | None",
    ) -> None:
        super().__init__(name)
        self.inner = inner
        self._notify = notify
        self._human_inbox = human_inbox

    @property
    def is_human(self) -> bool:
        return self._human_inbox is not None

    def _sync_to_inner(self) -> None:
        if self.inner is None:
            return
        self.inner.hand = self.hand
        self.inner.bid = self.bid
        self.inner.tricks_won = self.tricks_won
        self.inner.total_score = self.total_score
        self.inner.consecutive_zero_bids = self.consecutive_zero_bids

    def choose_trump(self, hand: list[Card]) -> Suit:
        self._sync_to_inner()
        self._notify({"type": "turn_started", "player": self.name})
        if self.is_human:
            self._notify({"type": "trump_request", "player": self.name, "hand": list(hand)})
            suit = self._human_inbox.get()
        else:
            time.sleep(BOT_DELAY)
            assert self.inner is not None
            suit = self.inner.choose_trump(hand)
        self._notify({"type": "trump_chosen", "player": self.name, "suit": suit})
        return suit

    def choose_bid(self, state: BiddingState) -> int:
        self._sync_to_inner()
        self._notify({"type": "turn_started", "player": self.name})
        if self.is_human:
            self._notify({"type": "bid_request", "player": self.name, "state": state})
            bid = self._human_inbox.get()
        else:
            time.sleep(BOT_DELAY)
            assert self.inner is not None
            bid = self.inner.choose_bid(state)
        self._notify({"type": "bid_made", "player": self.name, "bid": bid})
        return bid

    def choose_card(self, state: TrickState) -> Card:
        self._sync_to_inner()
        self._notify({"type": "turn_started", "player": self.name})
        if self.is_human:
            self._notify({"type": "card_request", "player": self.name, "state": state})
            card = self._human_inbox.get()
        else:
            time.sleep(BOT_DELAY)
            assert self.inner is not None
            card = self.inner.choose_card(state)
        self._notify(
            {"type": "card_played", "player": self.name, "card": card, "trick_number": state.trick_number}
        )
        return card


class _RLBotLoader:
    """Loads each trained checkpoint at most once (multiple seats can share
    it), and only imports whist.rl / torch if an RL bot is actually
    requested. Mirrors whist.cli._RLBotLoader."""

    def __init__(self, reinforce_dir: str, qlearning_dir: str, device: str = "cpu") -> None:
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


def _run_game(game: WhistGame, event_queue: "queue.Queue[dict]") -> None:
    try:
        for hand_size in game.round_sequence():
            dealer = game.players[game.dealer_index]
            event_queue.put({"type": "round_start", "hand_size": hand_size, "dealer": dealer.name})
            result = game.play_round(hand_size)
            event_queue.put({"type": "round_end", "result": result})
        event_queue.put({"type": "game_end", "standings": game.standings()})
    except Exception as exc:  # surface engine errors in the GUI instead of dying silently
        event_queue.put({"type": "error", "message": str(exc)})


class SetupFrame(tk.Frame):
    def __init__(self, master: tk.Misc, on_start) -> None:
        super().__init__(master, padx=20, pady=20)
        self.on_start = on_start
        self.rows: list[dict] = []

        tk.Label(self, text="Contract Whist", font=("Segoe UI", 20, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 10)
        )

        count_frame = tk.Frame(self)
        count_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))
        tk.Label(count_frame, text="Number of players:").pack(side="left")
        self.count_var = tk.IntVar(value=4)
        self.count_spin = tk.Spinbox(
            count_frame, from_=MIN_PLAYERS, to=MAX_PLAYERS, width=3,
            textvariable=self.count_var, command=self._rebuild_seats,
        )
        self.count_spin.pack(side="left", padx=5)
        self.count_spin.bind("<Return>", lambda e: self._rebuild_seats())
        self.count_spin.bind("<FocusOut>", lambda e: self._rebuild_seats())

        self.seats_frame = tk.Frame(self)
        self.seats_frame.grid(row=2, column=0, columnspan=3, sticky="w")

        adv = tk.LabelFrame(self, text="RL checkpoint directories (only used if selected above)")
        adv.grid(row=3, column=0, columnspan=3, sticky="we", pady=(15, 10))
        tk.Label(adv, text="REINFORCE dir:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.reinforce_dir_var = tk.StringVar(value=DEFAULT_REINFORCE_DIR)
        tk.Entry(adv, textvariable=self.reinforce_dir_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(adv, text="Q-learning dir:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.qlearning_dir_var = tk.StringVar(value=DEFAULT_QLEARNING_DIR)
        tk.Entry(adv, textvariable=self.qlearning_dir_var, width=30).grid(row=1, column=1, padx=5, pady=2)

        self.status_label = tk.Label(self, text="", fg="#b00020")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="w")

        tk.Button(self, text="Start Game", font=("Segoe UI", 12, "bold"), command=self._start).grid(
            row=5, column=0, columnspan=3, pady=15
        )

        self._rebuild_seats()

    @staticmethod
    def _default_kind(i: int) -> str:
        return "human" if i == 0 else "simple"

    def _rebuild_seats(self) -> None:
        try:
            n = int(self.count_var.get())
        except (tk.TclError, ValueError):
            return
        n = max(MIN_PLAYERS, min(MAX_PLAYERS, n))
        self.count_var.set(n)

        for row in self.rows:
            row["frame"].destroy()
        self.rows = []

        for i in range(n):
            frame = tk.Frame(self.seats_frame)
            frame.grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(frame, text=f"Seat {i + 1}:", width=8, anchor="w").pack(side="left")

            default_kind = self._default_kind(i)
            kind_var = tk.StringVar(value=KIND_LABELS[default_kind])
            combo = ttk.Combobox(
                frame, textvariable=kind_var, values=list(KIND_LABELS.values()), state="readonly", width=16
            )
            combo.pack(side="left", padx=5)

            default_name = "You" if default_kind == "human" else f"{NAME_PREFIX[default_kind]}{i + 1}"
            name_var = tk.StringVar(value=default_name)
            tk.Entry(frame, textvariable=name_var, width=14).pack(side="left", padx=5)

            row = {"kind_var": kind_var, "name_var": name_var, "frame": frame, "last_auto": default_name}
            combo.bind("<<ComboboxSelected>>", lambda e, r=row, idx=i: self._on_kind_change(r, idx))
            self.rows.append(row)

    def _on_kind_change(self, row: dict, idx: int) -> None:
        kind = LABEL_TO_KIND[row["kind_var"].get()]
        if row["name_var"].get() == row["last_auto"] or not row["name_var"].get().strip():
            default = "You" if kind == "human" else f"{NAME_PREFIX[kind]}{idx + 1}"
            row["name_var"].set(default)
            row["last_auto"] = default

    def _start(self) -> None:
        seats: list[tuple[str, str]] = []
        names_seen: set[str] = set()
        human_count = 0
        for row in self.rows:
            kind = LABEL_TO_KIND[row["kind_var"].get()]
            name = row["name_var"].get().strip()
            if not name:
                self.status_label.config(text="Every seat needs a name.")
                return
            if name in names_seen:
                self.status_label.config(text=f"Duplicate name: {name}")
                return
            names_seen.add(name)
            human_count += kind == "human"
            seats.append((name, kind))

        if human_count > 1:
            self.status_label.config(text="Only one human seat is supported.")
            return

        self.status_label.config(text="Loading...")
        self.update_idletasks()
        try:
            self.on_start(seats, self.reinforce_dir_var.get().strip(), self.qlearning_dir_var.get().strip())
        except Exception as exc:
            self.status_label.config(text=f"Couldn't start: {exc}")


class TableFrame(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        seat_names: list[str],
        human_name: str | None,
        human_inbox: "queue.Queue | None",
        event_queue: "queue.Queue[dict]",
        on_new_game,
    ) -> None:
        super().__init__(master, padx=10, pady=10)
        self.seat_names = seat_names
        self.human_name = human_name
        self.human_inbox = human_inbox
        self.event_queue = event_queue
        self.on_new_game = on_new_game
        self._stopped = False

        self.totals = {name: 0 for name in seat_names}
        self.bid_by_name: dict[str, object] = {name: "-" for name in seat_names}
        self.tricks_by_name = {name: 0 for name in seat_names}
        self.current_trick_number = -1
        self.current_trick_cards: list[tuple[str, Card]] = []
        self.current_trump: Suit | None = None
        self.current_led_suit: Suit | None = None
        self._current_hand: list[Card] = []

        self.seat_widgets: dict[str, dict] = {}
        self._build_widgets()

    def start_polling(self) -> None:
        self._poll_events()

    def stop(self) -> None:
        self._stopped = True

    # ---------- widget construction ----------

    def _build_widgets(self) -> None:
        header = tk.Frame(self)
        header.pack(fill="x")
        self.round_label = tk.Label(header, text="Waiting for the game to start...", font=("Segoe UI", 13, "bold"))
        self.round_label.pack(side="left")
        self.trick_winner_label = tk.Label(header, text="", font=("Segoe UI", 11), fg="#1a7a1a")
        self.trick_winner_label.pack(side="right")

        seats_bar = tk.Frame(self, bd=1, relief="groove")
        seats_bar.pack(fill="x", pady=10)
        for name in self.seat_names:
            self.seat_widgets[name] = self._build_seat_panel(seats_bar, name)

        self.action_frame = tk.Frame(self, bd=1, relief="groove", height=90)
        self.action_frame.pack(fill="x", pady=5)
        self.action_frame.pack_propagate(False)
        self.action_status = tk.Label(self.action_frame, text="", font=("Segoe UI", 10, "italic"))
        self.action_status.pack(anchor="w", padx=8, pady=4)
        self.action_buttons_frame = tk.Frame(self.action_frame)
        self.action_buttons_frame.pack(fill="x", padx=8)

        tk.Label(self, text="Your hand:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.hand_frame = tk.Frame(self)
        self.hand_frame.pack(fill="x", pady=5)

        bottom = tk.Frame(self)
        bottom.pack(fill="both", expand=True, pady=(10, 0))

        score_frame = tk.Frame(bottom)
        score_frame.pack(side="left", fill="y")
        tk.Label(score_frame, text="Scoreboard", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.score_tree = ttk.Treeview(
            score_frame, columns=("bid", "tricks", "total"), show="tree headings", height=8
        )
        self.score_tree.heading("#0", text="Player")
        self.score_tree.heading("bid", text="Bid")
        self.score_tree.heading("tricks", text="Tricks")
        self.score_tree.heading("total", text="Total")
        self.score_tree.column("#0", width=120)
        self.score_tree.column("bid", width=50, anchor="center")
        self.score_tree.column("tricks", width=60, anchor="center")
        self.score_tree.column("total", width=60, anchor="center")
        for name in self.seat_names:
            self.score_tree.insert("", "end", iid=name, text=name, values=("-", 0, 0))
        self.score_tree.pack(fill="y")

        log_frame = tk.Frame(bottom)
        log_frame.pack(side="left", fill="both", expand=True, padx=(15, 0))
        tk.Label(log_frame, text="Game log", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_text = ScrolledText(log_frame, height=14, state="disabled", wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        tk.Button(self, text="Back to setup", command=self.on_new_game).pack(anchor="e", pady=(8, 0))

    def _build_seat_panel(self, parent: tk.Misc, name: str) -> dict:
        frame = tk.Frame(
            parent, padx=8, pady=6, highlightthickness=2, highlightbackground=parent.cget("bg")
        )
        frame.pack(side="left", padx=6, pady=6)
        display_name = f"{name} (you)" if name == self.human_name else name
        name_label = tk.Label(frame, text=display_name, font=("Segoe UI", 10, "bold"))
        name_label.pack()
        info_label = tk.Label(frame, text="bid: -  won: 0", font=("Segoe UI", 9))
        info_label.pack()
        card_label = tk.Label(
            frame, text="—", font=("Consolas", 16, "bold"), width=4, relief="ridge", bg="white"
        )
        card_label.pack(pady=(4, 0))
        return {"frame": frame, "info_label": info_label, "card_label": card_label}

    # ---------- event pump ----------

    def _poll_events(self) -> None:
        if self._stopped:
            return
        try:
            while True:
                self._handle_event(self.event_queue.get_nowait())
        except queue.Empty:
            pass
        if not self._stopped:
            self.after(30, self._poll_events)

    def _handle_event(self, ev: dict) -> None:
        handler = getattr(self, f"_on_{ev['type']}", None)
        if handler is not None:
            handler(ev)

    def _on_log(self, ev: dict) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", ev["message"] + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _on_round_start(self, ev: dict) -> None:
        self.current_trick_number = -1
        self.current_trick_cards = []
        self.current_trump = None
        self.current_led_suit = None
        self.trick_winner_label.config(text="")
        self.round_label.config(text=f"Round: {ev['hand_size']} card(s) | Dealer: {ev['dealer']} | Trump: ?")
        for name in self.seat_names:
            self.bid_by_name[name] = "-"
            self.tricks_by_name[name] = 0
            self._refresh_seat_info(name)
            self.seat_widgets[name]["card_label"].config(text="—", fg="black")
        self._clear_action_panel()

    def _on_turn_started(self, ev: dict) -> None:
        self._set_current_turn(ev["player"])

    def _on_trump_request(self, ev: dict) -> None:
        self._show_trump_panel(ev["hand"])

    def _on_trump_chosen(self, ev: dict) -> None:
        self.current_trump = ev["suit"]
        base = self.round_label.cget("text").split(" | Trump:")[0]
        self.round_label.config(text=f"{base} | Trump: {SUIT_SYMBOLS[ev['suit']]}")
        self._clear_action_panel()

    def _on_bid_request(self, ev: dict) -> None:
        self._show_bid_panel(ev["state"])

    def _on_bid_made(self, ev: dict) -> None:
        self.bid_by_name[ev["player"]] = ev["bid"]
        self._refresh_seat_info(ev["player"])
        self._clear_action_panel()

    def _on_card_request(self, ev: dict) -> None:
        self._show_card_panel(ev["state"])

    def _on_card_played(self, ev: dict) -> None:
        name, card, trick_no = ev["player"], ev["card"], ev["trick_number"]
        if trick_no != self.current_trick_number:
            self.current_trick_number = trick_no
            self.current_trick_cards = []
            self.current_led_suit = None
            self.trick_winner_label.config(text="")
            for n in self.seat_names:
                self.seat_widgets[n]["card_label"].config(text="—", fg="black")
        if self.current_led_suit is None:
            self.current_led_suit = card.suit
        self.current_trick_cards.append((name, card))
        self.seat_widgets[name]["card_label"].config(text=_card_text(card), fg=_card_color(card))
        self._clear_action_panel()

        if len(self.current_trick_cards) == len(self.seat_names):
            winner = _trick_winner(self.current_trick_cards, self.current_led_suit, self.current_trump)
            self.tricks_by_name[winner] += 1
            self._refresh_seat_info(winner)
            self.trick_winner_label.config(text=f"{winner} wins the trick")

    def _on_round_end(self, ev: dict) -> None:
        result = ev["result"]
        for name in self.seat_names:
            self.totals[name] += result.scores[name]
            self.bid_by_name[name] = result.bids[name]
            self.tricks_by_name[name] = result.tricks_won[name]
            self._refresh_seat_info(name)
            self.score_tree.item(name, values=(result.bids[name], result.tricks_won[name], self.totals[name]))
        self._set_current_turn(None)

    def _on_game_end(self, ev: dict) -> None:
        self._clear_action_panel()
        self._set_current_turn(None)
        lines = "\n".join(f"{i + 1}. {name}: {score}" for i, (name, score) in enumerate(ev["standings"]))
        messagebox.showinfo("Game over", f"Final standings:\n\n{lines}")

    def _on_error(self, ev: dict) -> None:
        messagebox.showerror("Game error", ev["message"])

    # ---------- seat display helpers ----------

    def _refresh_seat_info(self, name: str) -> None:
        w = self.seat_widgets[name]
        w["info_label"].config(text=f"bid: {self.bid_by_name[name]}  won: {self.tricks_by_name[name]}")

    def _set_current_turn(self, name: str | None) -> None:
        inactive = self.cget("bg")
        for seat_name, w in self.seat_widgets.items():
            color = "#2a6fdb" if seat_name == name else inactive
            w["frame"].config(highlightbackground=color, highlightcolor=color)

    # ---------- action panel (bid / trump / card) ----------

    def _clear_action_panel(self) -> None:
        for w in self.action_buttons_frame.winfo_children():
            w.destroy()
        self.action_status.config(text="")

    def _refresh_hand(self, hand: list[Card], valid: list[Card] | None, clickable: bool) -> None:
        for w in self.hand_frame.winfo_children():
            w.destroy()
        for card in sorted(hand):
            enabled = clickable and (valid is None or card in valid)
            tk.Button(
                self.hand_frame,
                text=_card_text(card),
                font=("Consolas", 14, "bold"),
                fg=_card_color(card),
                disabledforeground=_card_color(card),
                width=4,
                relief="raised" if enabled else "groove",
                state="normal" if enabled else "disabled",
                command=(lambda c=card: self._play_card(c)) if enabled else None,
            ).pack(side="left", padx=2)

    def _show_trump_panel(self, hand: list[Card]) -> None:
        self._current_hand = list(hand)
        self._refresh_hand(hand, valid=None, clickable=False)
        self._clear_action_panel()
        self.action_status.config(text="You're the dealer -- choose trump:")
        for suit in Suit:
            tk.Button(
                self.action_buttons_frame,
                text=SUIT_SYMBOLS[suit],
                font=("Consolas", 14, "bold"),
                fg=_card_color_for_suit(suit),
                width=4,
                command=lambda s=suit: self._submit_trump(s),
            ).pack(side="left", padx=3)

    def _submit_trump(self, suit: Suit) -> None:
        if self.human_inbox is not None:
            self.human_inbox.put(suit)
        self._clear_action_panel()

    def _show_bid_panel(self, state: BiddingState) -> None:
        self._current_hand = list(state.hand)
        self._refresh_hand(state.hand, valid=None, clickable=False)
        self._clear_action_panel()
        notes = []
        if state.forbidden_bid is not None:
            notes.append(f"may not bid {state.forbidden_bid}")
        if state.zero_bid_forbidden:
            notes.append("may not bid 0")
        note_text = f"  ({'; '.join(notes)})" if notes else ""
        self.action_status.config(text=f"Your turn to bid (trump is {SUIT_SYMBOLS[state.trump]}).{note_text}")
        for bid in legal_bids(state):
            tk.Button(
                self.action_buttons_frame, text=str(bid), width=4, command=lambda b=bid: self._submit_bid(b)
            ).pack(side="left", padx=3)

    def _submit_bid(self, bid: int) -> None:
        if self.human_inbox is not None:
            self.human_inbox.put(bid)
        self._clear_action_panel()

    def _show_card_panel(self, state: TrickState) -> None:
        self._current_hand = list(state.hand)
        self._refresh_hand(state.hand, valid=state.valid_cards, clickable=True)
        self._clear_action_panel()
        led = f"follow {SUIT_SYMBOLS[state.led_suit]}" if state.led_suit is not None else "you lead"
        self.action_status.config(text=f"Your turn to play a card ({led}).")

    def _play_card(self, card: Card) -> None:
        if self.human_inbox is not None:
            self.human_inbox.put(card)
        self._current_hand = [c for c in self._current_hand if c != card]
        self._refresh_hand(self._current_hand, valid=None, clickable=False)
        self._clear_action_panel()


class WhistApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Contract Whist")
        self.geometry("1080x760")
        self.minsize(920, 620)
        self.setup_frame: SetupFrame | None = None
        self.table_frame: TableFrame | None = None
        self._show_setup()

    def _show_setup(self) -> None:
        self.setup_frame = SetupFrame(self, self._start_game)
        self.setup_frame.pack(fill="both", expand=True)

    def _start_game(self, seats: list[tuple[str, str]], reinforce_dir: str, qlearning_dir: str) -> None:
        rng = random.Random()
        event_queue: "queue.Queue[dict]" = queue.Queue()
        rl_loader = _RLBotLoader(reinforce_dir or DEFAULT_REINFORCE_DIR, qlearning_dir or DEFAULT_QLEARNING_DIR)

        players: list[Player] = []
        human_name: str | None = None
        human_inbox: "queue.Queue | None" = None
        for name, kind in seats:
            if kind == "human":
                human_name = name
                human_inbox = queue.Queue()
                players.append(_ObservedPlayer(name, None, event_queue.put, human_inbox))
            elif kind == "simple":
                players.append(_ObservedPlayer(name, SimpleBot(name, rng), event_queue.put, None))
            elif kind == "random":
                players.append(_ObservedPlayer(name, RandomBot(name, rng), event_queue.put, None))
            elif kind == "reinforce":
                bot = rl_loader.build_reinforce_bot(name, rng)
                players.append(_ObservedPlayer(name, bot, event_queue.put, None))
            elif kind == "qlearning":
                bot = rl_loader.build_qlearning_bot(name, rng)
                players.append(_ObservedPlayer(name, bot, event_queue.put, None))
            else:
                raise ValueError(f"Unknown seat kind: {kind}")

        game = WhistGame(players, rng=rng, on_event=lambda msg: event_queue.put({"type": "log", "message": msg}))

        assert self.setup_frame is not None
        self.setup_frame.pack_forget()
        self.setup_frame.destroy()
        self.setup_frame = None

        self.table_frame = TableFrame(
            self, [n for n, _ in seats], human_name, human_inbox, event_queue, on_new_game=self._back_to_setup
        )
        self.table_frame.pack(fill="both", expand=True)

        threading.Thread(target=_run_game, args=(game, event_queue), daemon=True).start()
        self.table_frame.start_polling()

    def _back_to_setup(self) -> None:
        if self.table_frame is not None:
            self.table_frame.stop()
            self.table_frame.pack_forget()
            self.table_frame.destroy()
            self.table_frame = None
        self._show_setup()


def main() -> None:
    app = WhistApp()
    app.mainloop()


if __name__ == "__main__":
    main()
