# Per-agent Slack identities — setup

By default every Meridian message comes from one shared Slack webhook. Bot
mode gives each agent its own Slack app instead — its own name, its own
avatar, showing up in the channel's member list like a real teammate. The
daily standup becomes a real thread: George posts the scoreboard and opens
it, then each agent replies with their own line.

You don't have to do all ten at once. Any agent without a bot token just
posts under George's identity instead — set up as many as you want, whenever
you want, and the rest degrade gracefully. Two-way chat (@mentioning an
agent and getting a reply back) is **not** part of this — that needs a
standing listener process and is future work.

## 1. Get the channel ID

Open the Slack channel Meridian should post in, click the channel name at
the top, and copy the **Channel ID** near the bottom of the "About" panel
(starts with `C`, e.g. `C0123ABCDEF`). You'll set this once, in
`MERIDIAN_SLACK_CHANNEL_ID`.

## 2. Create one Slack app per agent

Repeat this for each agent you want to give an identity to. Suggested name
and icon (matching the emoji already used in the standup/dashboard):

| Agent | Role | Suggested name | Emoji |
|---|---|---|---|
| Wong | Data | Wong — Data | 🔍 |
| David | Compliance | David — Compliance | 🛑 |
| Leo | Backtest | Leo — Backtest | 📈 |
| Charles | Risk | Charles — Risk | ⚖️ |
| Greg | Regime | Greg — Regime | 🧭 |
| Cornelius | Execution (stocks) | Cornelius — Execution | 💼 |
| George | Reporting | George — Reporting | 📊 |
| Augustus | Options Data | Augustus — Options Data | 📡 |
| Theo | Options Risk | Theo — Options Risk | 🛡️ |
| Joseph | Options Execution | Joseph — Options Execution | 🎯 |

For each one:

1. Go to <https://api.slack.com/apps> → **Create New App** → **From
   scratch**.
2. Name it (e.g. "Wong — Data") and pick your Meridian workspace.
3. **OAuth & Permissions** (left sidebar) → under **Scopes** → **Bot Token
   Scopes** → **Add an OAuth Scope** → add `chat:write`.
4. Still on **OAuth & Permissions**, click **Install to Workspace** (top of
   the page) and approve it.
5. Copy the **Bot User OAuth Token** (starts with `xoxb-`) — this is the
   agent's token.
6. Optional but recommended: **Basic Information** → **Display Information**
   → set the app icon (an emoji-based avatar works fine) and short
   description, so the bot reads clearly in the channel.
7. Back in the target Slack channel, `/invite @Wong — Data` (or whatever you
   named it) so the bot can actually post there. A bot that's never invited
   will fail to post and Meridian will just fall back to the shared webhook
   message for that agent — nothing breaks, but you won't see that agent's
   own identity until it's invited.

## 3. Set the environment variables

In `scripts/slack_env.sh` (already gitignored — never commit real tokens),
set the channel ID plus a token per agent you created:

```bash
export MERIDIAN_SLACK_CHANNEL_ID="C0123ABCDEF"

export MERIDIAN_SLACK_TOKEN_WONG="xoxb-..."
export MERIDIAN_SLACK_TOKEN_DAVID="xoxb-..."
export MERIDIAN_SLACK_TOKEN_LEO="xoxb-..."
export MERIDIAN_SLACK_TOKEN_CHARLES="xoxb-..."
export MERIDIAN_SLACK_TOKEN_GREG="xoxb-..."
export MERIDIAN_SLACK_TOKEN_CORNELIUS="xoxb-..."
export MERIDIAN_SLACK_TOKEN_GEORGE="xoxb-..."
export MERIDIAN_SLACK_TOKEN_AUGUSTUS="xoxb-..."
export MERIDIAN_SLACK_TOKEN_THEO="xoxb-..."
export MERIDIAN_SLACK_TOKEN_JOSEPH="xoxb-..."
```

Only set the ones for agents you actually created an app for — leave the
rest unset. `scripts/run_daily_options.sh` already sources this file before
running, so the scheduled run picks up bot mode automatically once it's
there. For a manual run in your own Terminal, `source scripts/slack_env.sh`
first.

## 4. Turn it on

Bot mode switches on by itself once `MERIDIAN_SLACK_CHANNEL_ID` and at least
one `MERIDIAN_SLACK_TOKEN_<AGENT>` are set — no config change needed. Run
`python main.py paper` and check the channel: George should open a new
thread for the standup, with each configured agent replying under it in
their own name. Anything without a token yet still shows up, just under
George's name, so partial setup never loses a message.

If a bot's post ever fails (bad token, not invited to the channel, token
revoked), Meridian logs a warning and falls straight back to the plain
webhook message — nothing here can newly cause a message to go unsent.

## Reference: which agent posts what

| Message | Posts as |
|---|---|
| Standup header (scoreboard, inquiries) | George |
| Each standup desk line | That line's own agent (Wong, David, Charles, Greg, Cornelius, George, Augustus, Theo, Joseph) |
| Stock fills | Cornelius |
| Options activity (opens/closes/expiries/Theo rejections) | Joseph |
| Stock desk halt / halt cleared | Cornelius |
| Options desk halt / halt cleared | Joseph |
| Bench / unbench / other operator actions | George |
| Run failures | George |

Leo doesn't currently post a standup line (backtesting is research-mode
only, not part of the live paper run), but his token env var is wired up
and ready for when that changes.
