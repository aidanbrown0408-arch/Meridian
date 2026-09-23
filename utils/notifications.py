"""Slack notifications for the desk -- every message Meridian sends.

`utils/slack.py` is the transport (one POST, never raises). This module is
what gets said and how it looks: Slack Block Kit messages built from plain
data, so each builder is a pure function that tests can inspect without a
network.

Message types
-------------
* **Daily standup** -- the desk's "group chat": a header, a scoreboard for
  both books, then one short line from each agent (stock desk, then options
  desk), then any lifecycle inquiries needing operator approval.
* **Stock fills** -- every buy/sell Cornelius made this run.
* **Options activity** -- every open/close/expiration Joseph booked this
  run, plus any proposal Theo turned down.
* **Halts** -- killswitch or the options bucket's loss cutoff. Urgent.
* **Halt cleared** / **operator actions** -- so the channel has a record of
  every manual override.
* **Run failures** -- anything that stopped a run or a leg of one.

Routing: routine messages go to `slack.webhook_url_env`; urgent ones
(halts, failures) go to `slack.alerts_webhook_url_env` if that is set,
otherwise to the same channel. Every type can be switched off under
`slack.notify` in config.

**Bot mode (optional).** By default every message above goes out as the one
shared webhook. If `slack.channel_id_env` resolves to a real channel ID and
at least one `slack.agent_tokens` entry has its env var set, `Notifier`
switches that message to Slack's `chat.postMessage` API instead, posting as
the owning agent's own bot user (own name, own avatar) rather than the
generic webhook sender. The standup becomes a real thread: George posts the
scoreboard and opens it, then each agent posts their own line as a threaded
reply. An agent with no bot token yet still gets their line in -- it posts
under George's identity rather than being dropped, so partial rollout (some
of the ten Slack apps not created yet) degrades gracefully. Every bot-mode
post that fails outright (bad token, bot not invited to the channel, no
channel ID at all) falls straight back to the original webhook message --
nothing in this layer can newly cause a message to go unsent. See
`docs/slack_agents_setup.md` for how to create the ten Slack apps.

Nothing here ever raises into the trading run. A formatting bug or a Slack
outage logs a warning and the run carries on.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from utils.logging_setup import get_logger
from utils.slack import post_as, post_message

log = get_logger("notifications", agent="George")

STANDUP_WEEKDAYS = range(0, 5)  # Monday=0 .. Friday=4
MAX_LIST_ROWS = 12              # rows shown per list before "…and N more"


# ------------------------------------------------------------------ data


@dataclass
class DeskLine:
    """One agent's line in the standup chat."""
    emoji: str
    agent: str
    role: str
    text: str
    tone: str = "info"  # "info" | "good" | "warn" | "bad"

    def plain(self) -> str:
        return f"{self.emoji} {self.agent} ({self.role}): {self.text}"


@dataclass
class BookSnapshot:
    """Headline numbers for one ledger, for the standup scoreboard."""
    name: str                  # "Stock desk" | "Options desk"
    equity: float
    starting_capital: float
    open_positions: int
    fills_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    extra: str = ""            # e.g. "$150.00"
    extra_label: str = ""      # e.g. "Premium at risk"

    @property
    def pnl(self) -> float:
        return self.equity - self.starting_capital

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.starting_capital if self.starting_capital else 0.0


@dataclass
class StandupPayload:
    mode: str                                   # "Paper" | "Research"
    stock_lines: list = field(default_factory=list)    # DeskLine, "Stock desk"
    options_lines: list = field(default_factory=list)  # DeskLine, "Options desk"
    inquiries: list = field(default_factory=list)      # lifecycle Recommendation
    books: list = field(default_factory=list)          # BookSnapshot
    dashboard: str = ""                          # URL or local path


# ------------------------------------------------------------------ formatting


def esc(text) -> str:
    """Slack mrkdwn escaping -- only &, <, > are special."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def money(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value > 0 else ("-" if value < 0 else "")
    return f"{sign}${abs(value):,.2f}"


def pct(value: float) -> str:
    return f"{value:+.2%}"


def _header(text: str) -> dict:
    return {"type": "header", "text": {"type": "plain_text", "text": text[:150], "emoji": True}}


def _context(*parts: str) -> dict:
    return {"type": "context",
            "elements": [{"type": "mrkdwn", "text": p[:3000]} for p in parts if p][:10]}


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:3000]}}


def _fields(pairs: list[tuple[str, str]]) -> dict:
    return {"type": "section",
            "fields": [{"type": "mrkdwn", "text": f"*{k}*\n{v}"[:2000]} for k, v in pairs[:10]]}


def _divider() -> dict:
    return {"type": "divider"}


def _stamp(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%a, %b} {now.day}, {now:%Y} · {now:%-I:%M %p}"


def _clip(rows: list[str]) -> list[str]:
    if len(rows) <= MAX_LIST_ROWS:
        return rows
    return rows[:MAX_LIST_ROWS] + [f"_…and {len(rows) - MAX_LIST_ROWS} more_"]


def _dashboard_ref(dashboard: str) -> str:
    if not dashboard:
        return ""
    if dashboard.startswith(("http://", "https://")):
        return f"<{dashboard}|Open the full dashboard>"
    return f"Full dashboard: `{esc(dashboard)}`"


TONE_MARK = {"info": "", "good": "", "warn": " :warning:", "bad": " :red_circle:"}


# ------------------------------------------------------------------ builders


def build_standup(p: StandupPayload, now: datetime | None = None) -> tuple[str, list[dict]]:
    now = now or datetime.now()
    blocks: list[dict] = [
        _header("Meridian Capital · Daily Standup"),
        _context(f"*{esc(p.mode)}* · {_stamp(now)}"),
    ]

    books = {b.name: b for b in p.books}
    desks = [("Stock desk", ":chart_with_upwards_trend:", p.stock_lines),
             ("Options desk", ":dart:", p.options_lines)]
    for name, icon, lines in desks:
        book = books.get(name)
        if book is None and not lines:
            continue
        title = f"{icon}  *{name}*"
        if book is not None:
            title += f"  ·  {money(book.equity)}  ({pct(book.pnl_pct)})"
        blocks += [_divider(), _section(title)]
        if book is not None:
            status = (f":octagonal_sign: *HALTED* — {esc(book.halt_reason)}"
                      if book.halted else ":large_green_circle: Trading")
            pairs = [
                ("P&L since inception", money(book.pnl, signed=True) if book.pnl else "$0.00"),
                ("Status", status),
                ("Open positions", str(book.open_positions)),
                ("Fills this run", str(book.fills_today)),
            ]
            if book.extra:
                pairs.append((book.extra_label or "Other", esc(book.extra)))
            blocks.append(_fields(pairs))
        if lines:
            body = "\n".join(
                f"{l.emoji}  *{esc(l.agent)}* · _{esc(l.role)}_{TONE_MARK.get(l.tone, '')}"
                f"  —  {esc(l.text)}" for l in lines)
            blocks.append(_context(body))

    if p.inquiries:
        rows = [f":raised_hand:  *{esc(r.trader)}* — recommend *{esc(r.action.upper())}* "
                f"({esc(r.trigger.replace('_', ' '))})\n{esc(r.detail)}" for r in p.inquiries]
        blocks += [_divider(),
                   _section("*Needs your approval*\n" + "\n".join(_clip(rows))),
                   _context("Nothing changes until you run `python main.py bench` / `unbench`.")]

    footer = _dashboard_ref(p.dashboard)
    blocks += [_divider(), _context(footer or "Meridian Capital", "George · Reporting")]

    summary = " · ".join(f"{b.name} {money(b.equity)} ({pct(b.pnl_pct)})" for b in p.books)
    text = f"Meridian daily standup — {summary}" if summary else "Meridian daily standup"
    return text, blocks[:50]


def build_standup_header(p: StandupPayload, now: datetime | None = None) -> tuple[str, list[dict]]:
    """Bot-mode standup opener: the scoreboard and inquiries, no per-agent
    lines -- those each post as their own threaded reply (`build_agent_line`)
    under this message. Posted as George; its `ts` becomes the thread."""
    now = now or datetime.now()
    blocks: list[dict] = [
        _header("Meridian Capital · Daily Standup"),
        _context(f"*{esc(p.mode)}* · {_stamp(now)}"),
    ]

    for book in p.books:
        blocks += [_divider(),
                   _section(f"*{esc(book.name)}*  ·  {money(book.equity)}  ({pct(book.pnl_pct)})")]
        status = (f":octagonal_sign: *HALTED* — {esc(book.halt_reason)}"
                  if book.halted else ":large_green_circle: Trading")
        pairs = [
            ("P&L since inception", money(book.pnl, signed=True) if book.pnl else "$0.00"),
            ("Status", status),
            ("Open positions", str(book.open_positions)),
            ("Fills this run", str(book.fills_today)),
        ]
        if book.extra:
            pairs.append((book.extra_label or "Other", esc(book.extra)))
        blocks.append(_fields(pairs))

    if p.inquiries:
        rows = [f":raised_hand:  *{esc(r.trader)}* — recommend *{esc(r.action.upper())}* "
                f"({esc(r.trigger.replace('_', ' '))})\n{esc(r.detail)}" for r in p.inquiries]
        blocks += [_divider(),
                   _section("*Needs your approval*\n" + "\n".join(_clip(rows))),
                   _context("Nothing changes until you run `python main.py bench` / `unbench`.")]

    footer = _dashboard_ref(p.dashboard)
    blocks += [_divider(),
              _context(footer or "Meridian Capital", "George · Reporting",
                       "Replies below are each agent's line from today's run.")]

    summary = " · ".join(f"{b.name} {money(b.equity)} ({pct(b.pnl_pct)})" for b in p.books)
    text = f"Meridian daily standup — {summary}" if summary else "Meridian daily standup"
    return text, blocks[:50]


def build_agent_line(line: DeskLine, now: datetime | None = None) -> tuple[str, list[dict]]:
    """One `DeskLine` as its own threaded reply, posted as that agent."""
    blocks = [
        _section(f"{esc(line.text)}{TONE_MARK.get(line.tone, '')}"),
        _context(f"{esc(line.role)} · {_stamp(now or datetime.now())}"),
    ]
    return line.plain(), blocks


def build_stock_fills(trades: list, equity: float, cash: float,
                      now: datetime | None = None) -> tuple[str, list[dict]]:
    rows = []
    for t in trades:
        icon = ":chart_with_upwards_trend:" if t.side == "buy" else ":chart_with_downwards_trend:"
        rows.append(f"{icon}  *{t.side.upper()}* {esc(t.symbol)} — {t.shares:,.4f} @ "
                    f"{money(t.price)}  ·  {money(t.shares * t.price)}"
                    f"  ·  _{esc(t.reason)}_")
    blocks = [
        _header(f"Stock desk · {len(trades)} fill{'s' if len(trades) != 1 else ''}"),
        _context(f"Cornelius · Execution (Paper) · {_stamp(now)}"),
        _section("\n".join(_clip(rows))),
        _context(f"Equity {money(equity)} · Cash {money(cash)}"),
    ]
    first = trades[0]
    text = (f"Stock desk: {first.side} {first.symbol}"
            + (f" (+{len(trades) - 1} more)" if len(trades) > 1 else ""))
    return text, blocks


def build_options_activity(trades: list, rejections: list, equity: float, cash: float,
                           open_premium: float, now: datetime | None = None
                           ) -> tuple[str, list[dict]]:
    rows = []
    for t in trades:
        kind = "C" if t.option_type == "long_call" else "P"
        label = f"{t.underlying} {t.expiration} {t.strike:g}{kind}"
        if t.side == "open":
            rows.append(f":large_green_circle:  *OPENED* {esc(label)} — {t.contracts} × "
                        f"{money(t.premium_per_contract)}  ·  "
                        f"{money(t.contracts * t.premium_per_contract)} at risk"
                        f"\n_{esc(t.reason)}_")
        else:
            verb = "EXPIRED" if t.reason == "expired" else "CLOSED"
            icon = ":white_check_mark:" if t.realized_pnl >= 0 else ":small_red_triangle_down:"
            rows.append(f"{icon}  *{verb}* {esc(label)} — {t.contracts} × "
                        f"{money(t.premium_per_contract)}  ·  P&L "
                        f"*{money(t.realized_pnl, signed=True)}*\n_{esc(t.reason)}_")
    for d in rejections:
        pr = d.proposal
        kind = "C" if pr.option_type == "long_call" else "P"
        rows.append(f":no_entry_sign:  *REJECTED by Theo* {esc(pr.underlying)} "
                    f"{esc(pr.expiration)} {pr.strike:g}{kind} — {esc(d.reason)}")

    n = len(trades)
    title = (f"Options desk · {n} trade{'s' if n != 1 else ''}" if n
             else "Options desk · proposal rejected")
    blocks = [
        _header(title),
        _context(f"Augustus → Theo → Joseph · {_stamp(now)}"),
        _section("\n".join(_clip(rows))),
        _context(f"Bucket equity {money(equity)} · Cash {money(cash)} · "
                 f"Premium at risk {money(open_premium)}"),
    ]
    if trades:
        t = trades[0]
        text = (f"Options desk: {t.side} {t.underlying} {t.strike:g}"
                + (f" (+{n - 1} more)" if n > 1 else ""))
    else:
        text = f"Options desk: Theo rejected {len(rejections)} proposal(s)"
    return text, blocks


def build_halt(desk: str, reason: str, detail: str = "", manual: bool = False,
               now: datetime | None = None) -> tuple[str, list[dict]]:
    command = "python main.py clear-halt" + (
        " --options" if desk.lower().startswith("options") else "")
    cause = "Killswitch pulled by operator" if manual else "Automatic halt"
    blocks = [
        _header(f":rotating_light: {desk} HALTED"),
        _context(f"{cause} · {_stamp(now)}"),
        _section(f"*Reason:* {esc(reason)}" + (f"\n{esc(detail)}" if detail else "")),
        _section("All positions on this book were flattened. Nothing trades again until you "
                 f"clear it:\n`{command}`"),
    ]
    return f"🚨 {desk} halted: {reason}", blocks


def build_halt_cleared(desk: str, now: datetime | None = None) -> tuple[str, list[dict]]:
    blocks = [
        _section(f":large_green_circle:  *{esc(desk)} halt cleared* — trading resumes on the next run."),
        _context(f"Operator action · {_stamp(now)}"),
    ]
    return f"{desk} halt cleared", blocks


def build_operator_action(action: str, detail: str,
                          now: datetime | None = None) -> tuple[str, list[dict]]:
    blocks = [
        _section(f":bust_in_silhouette:  *{esc(action)}*\n{esc(detail)}"),
        _context(f"Operator action · {_stamp(now)}"),
    ]
    return f"{action}: {detail}", blocks


def build_failure(stage: str, exc: BaseException, impact: str = "",
                  now: datetime | None = None) -> tuple[str, list[dict]]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tail = tb.strip().splitlines()[-8:]
    blocks = [
        _header(f":x: {stage} failed"),
        _context(_stamp(now)),
        _section(f"*{esc(type(exc).__name__)}:* {esc(exc)}"
                 + (f"\n{esc(impact)}" if impact else "")),
        _section("```" + esc("\n".join(tail))[:2900] + "```"),
        _context("Full traceback in `reports/meridian.log`."),
    ]
    return f"❌ {stage} failed: {type(exc).__name__}: {exc}", blocks


# ------------------------------------------------------------------ sender


class Notifier:
    """The only thing in Meridian that decides whether a message is sent."""

    URGENT = {"halts", "failures"}

    #: Which agent's bot identity owns each desk's halt/halt-cleared message.
    _DESK_AGENT = {"stock desk": "Cornelius", "options desk": "Joseph"}

    def __init__(self, config, enabled: bool = True):
        self.config = config
        self.enabled = enabled
        main_env = config.get("slack.webhook_url_env", "MERIDIAN_SLACK_WEBHOOK_URL")
        alerts_env = config.get("slack.alerts_webhook_url_env",
                                "MERIDIAN_SLACK_ALERTS_WEBHOOK_URL")
        self.webhook_url = os.environ.get(main_env)
        self.alerts_webhook_url = os.environ.get(alerts_env) or self.webhook_url
        self.dashboard = str(config.get("slack.dashboard_url", "") or "")
        self.sent: list[tuple[str, str]] = []  # (kind, text) -- for tests/console

        # Bot mode: per-agent Slack apps, layered on top of the webhook
        # above. Off unless a channel ID and at least one agent token
        # resolve from the environment -- when it's off, behavior is
        # byte-for-byte the same as before this existed.
        channel_env = config.get("slack.channel_id_env", "MERIDIAN_SLACK_CHANNEL_ID")
        alerts_channel_env = config.get("slack.alerts_channel_id_env",
                                        "MERIDIAN_SLACK_ALERTS_CHANNEL_ID")
        self.channel_id = os.environ.get(channel_env)
        self.alerts_channel_id = os.environ.get(alerts_channel_env) or self.channel_id
        self._agent_tokens: dict[str, str] = {}
        for agent, env_var in (config.get("slack.agent_tokens", {}) or {}).items():
            token = os.environ.get(env_var)
            if token:
                self._agent_tokens[agent] = token
        self.bot_mode = bool(self.channel_id and self._agent_tokens)

    def allowed(self, kind: str) -> bool:
        if not self.enabled:
            return False
        if kind == "standup":
            return bool(self.config.get("slack.post_daily_standup", True))
        return bool(self.config.get(f"slack.notify.{kind}", True))

    def _desk_agent(self, desk: str) -> str:
        return self._DESK_AGENT.get(desk.strip().lower(), "George")

    def _fallback_webhook(self, kind: str, text: str, blocks: list[dict]) -> bool:
        """The original, bot-mode-free path: post the shared webhook message
        and record it. Used both when bot mode is off and whenever a
        bot-mode post fails outright."""
        url = self.alerts_webhook_url if kind in self.URGENT else self.webhook_url
        try:
            ok = post_message(url, text, blocks)
        except Exception as exc:  # e.g. a malformed webhook URL
            log.warning("Slack %s post failed (%s).", kind, exc)
            ok = False
        self.sent.append((kind, text))
        return ok

    def _send(self, kind: str, agent: str, build, *args, **kwargs) -> bool:
        """Build one message and send it as `agent`'s bot identity when bot
        mode can, otherwise (or on any bot-mode failure) over the shared
        webhook. A formatting bug never stops a run."""
        if not self.allowed(kind):
            return False
        try:
            text, blocks = build(*args, **kwargs)
        except Exception as exc:
            log.warning("Could not build %s Slack message (%s) -- skipped.", kind, exc)
            return False

        if self.bot_mode:
            token = self._agent_tokens.get(agent)
            channel = self.alerts_channel_id if kind in self.URGENT else self.channel_id
            if token and channel:
                posted = post_as(token, channel, text, blocks)
                if posted is not None:
                    self.sent.append((kind, text))
                    return True
                log.warning("Bot-mode %s post as %s failed -- falling back to webhook.",
                           kind, agent)

        return self._fallback_webhook(kind, text, blocks)

    def _standup_bot_mode(self, payload: StandupPayload, now: datetime) -> bool:
        """Real Slack thread: George posts the header and opens the thread,
        then each agent posts their own line as a threaded reply under it.
        An agent with no bot token yet posts under George's identity instead
        of being dropped. Returns False on any failure posting the header --
        the caller then falls back to the single webhook message, untouched
        by anything attempted here."""
        try:
            text, blocks = build_standup_header(payload, now)
        except Exception as exc:
            log.warning("Could not build standup header (%s) -- bot mode skipped.", exc)
            return False

        george_token = self._agent_tokens.get("George")
        header = post_as(george_token, self.channel_id, text, blocks)
        if header is None:
            return False
        thread_ts = header.get("ts")

        for line in list(payload.stock_lines) + list(payload.options_lines):
            try:
                line_text, line_blocks = build_agent_line(line, now)
            except Exception as exc:
                log.warning("Could not build standup line for %s (%s) -- skipped.",
                           line.agent, exc)
                continue
            token = self._agent_tokens.get(line.agent)
            posted = post_as(token or george_token, self.channel_id, line_text,
                             line_blocks, thread_ts=thread_ts)
            if posted is None and token:
                # That agent's own bot post failed -- still get the line
                # into the thread under George rather than dropping it.
                post_as(george_token, self.channel_id, line_text, line_blocks,
                       thread_ts=thread_ts)

        self.sent.append(("standup", text))
        return True

    # -- public API, one method per message type

    def standup(self, payload: StandupPayload, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        if now.weekday() not in STANDUP_WEEKDAYS:
            log.info("Weekend -- standup post skipped (weekdays only per spec).")
            return False
        if not self.allowed("standup"):
            return False
        if not payload.dashboard:
            payload.dashboard = self.dashboard

        if self.bot_mode and self._standup_bot_mode(payload, now):
            return True

        try:
            text, blocks = build_standup(payload, now)
        except Exception as exc:
            log.warning("Could not build standup Slack message (%s) -- skipped.", exc)
            return False
        return self._fallback_webhook("standup", text, blocks)

    def stock_fills(self, trades: list, equity: float, cash: float) -> bool:
        if not trades:
            return False
        return self._send("trades", "Cornelius", build_stock_fills, trades, equity, cash)

    def options_activity(self, trades: list, rejections: list, equity: float,
                         cash: float, open_premium: float) -> bool:
        if not trades and not rejections:
            return False
        return self._send("trades", "Joseph", build_options_activity, trades, rejections,
                          equity, cash, open_premium)

    def halt(self, desk: str, reason: str, detail: str = "", manual: bool = False) -> bool:
        return self._send("halts", self._desk_agent(desk), build_halt,
                          desk, reason, detail, manual)

    def halt_cleared(self, desk: str) -> bool:
        return self._send("operator_actions", self._desk_agent(desk), build_halt_cleared, desk)

    def operator_action(self, action: str, detail: str) -> bool:
        return self._send("operator_actions", "George", build_operator_action, action, detail)

    def failure(self, stage: str, exc: BaseException, impact: str = "") -> bool:
        return self._send("failures", "George", build_failure, stage, exc, impact)
