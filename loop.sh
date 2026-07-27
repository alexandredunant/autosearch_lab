#!/usr/bin/env bash
# Autoresearch driver. Each iteration launches a FRESH agent session that runs
# exactly one experiment (see program.md). State lives in git + the ledger, so
# no chat context accumulates across experiments.
#
# Usage:            ./loop.sh [max_experiments]
# By default the loop runs until the search CONVERGES: it stops when
# search.discard_budget consecutive experiments fail to improve the best
# score (the quality criterion, author-set in config.yml), or when you
# touch .autoresearch_done. The optional max_experiments argument is only a
# safety cap; it counts LOGGED experiments (new ledger rows), not agent
# launches — sessions that crash or get rate-limited before logging anything
# are retried after a backoff and do not consume the cap.
#
# Default agent:    codex. For a zero-API-cost run, anyone who installs
# opencode (https://opencode.ai) can use its free hosted DeepSeek model.
# Override with env vars:
#   AGENT=opencode OPENCODE_MODEL=opencode/deepseek-v4-flash-free ./loop.sh 50
#   AGENT=gemini GEMINI_MODEL=gemini-2.5-flash ./loop.sh 50
#   AGENT=claude ./loop.sh 50
#   MAX_WASTED=20 WASTED_SLEEP=120 ./loop.sh 50   # tolerance for dead sessions
#   SESSION_TIMEOUT=30m ./loop.sh                 # per-session hang guard
#
# Fully local, zero-cost, offline agent via Ollama (drives opencode against a
# local open-weight model — no API key, no rate limits, data never leaves the
# machine). One-off setup, then run:
#   ollama serve &                       # start the local server
#   ollama pull devstral:24b             # agent-first ~14GB model (needs ~20GB VRAM)
#   # raise the context window so the agent can hold the ledger + lab notes:
#   printf 'FROM devstral:24b\nPARAMETER num_ctx 32768\n' > Modelfile.autoresearch
#   ollama create devstral-autoresearch -f Modelfile.autoresearch
#   AGENT=ollama OLLAMA_MODEL=ollama/devstral-autoresearch ./loop.sh

set -u
cd "$(dirname "$0")"

AGENT=${AGENT:-codex}
OPENCODE_MODEL=${OPENCODE_MODEL:-opencode/deepseek-v4-flash-free}
GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash}
# Local Ollama model, referenced as opencode sees it (provider/tag). MUST be a
# num_ctx-raised variant (see header): at the 4k default context the tool
# schemas don't fit and the model silently loses tool-calling. Requires
# opencode's ollama provider to use the native driver (ollama-ai-provider-v2),
# not the OpenAI-compatible one, whose /v1 endpoint drops tool calls.
OLLAMA_MODEL=${OLLAMA_MODEL:-ollama/devstral-autoresearch}
CODEX_MODEL=${CODEX_MODEL:-config.toml default}  # codex has no -m flag here; whatever ~/.codex/config.toml specifies
CLAUDE_MODEL=${CLAUDE_MODEL:-CLI default}        # claude has no -m flag here; whatever the claude CLI defaults to

# Recorded in every ledger row (log_experiment() reads these, not the agent's
# own self-report) so a mid-run agent switch -- e.g. hitting one provider's
# usage limit and falling back to another -- is visible in experiments.tsv
# without depending on the model to correctly identify itself.
export AUTORESEARCH_AGENT="$AGENT"
case "$AGENT" in
    codex)    export AUTORESEARCH_MODEL="$CODEX_MODEL" ;;
    gemini)   export AUTORESEARCH_MODEL="$GEMINI_MODEL" ;;
    opencode) export AUTORESEARCH_MODEL="$OPENCODE_MODEL" ;;
    ollama)   export AUTORESEARCH_MODEL="$OLLAMA_MODEL" ;;
    claude)   export AUTORESEARCH_MODEL="$CLAUDE_MODEL" ;;
    *)        export AUTORESEARCH_MODEL="unknown" ;;
esac

MAX_EXPERIMENTS=${1:-0}           # 0 = no cap: run until the search converges
MAX_WASTED=${MAX_WASTED:-10}      # consecutive sessions that log nothing
WASTED_SLEEP=${WASTED_SLEEP:-60}  # backoff seconds after a wasted session
SESSION_TIMEOUT=${SESSION_TIMEOUT:-45m}  # kill a session that runs longer (silent gateway hangs)
LEDGER=artifacts/experiments/experiments.tsv
BEST_SCORE=artifacts/experiments/best_score.txt
# Imperative, tool-forcing phrasing: weak local models (e.g. devstral) answer
# passive prompts conversationally ("I can't read files") instead of calling
# tools. Telling them to ACT with their tools is harmless for strong models and
# essential for weak ones.
PROMPT='You are an autonomous coding agent with bash and file tools. ACT using your tools at every step — do not explain, summarize, or ask permission; execute. Read program.md and follow it exactly: recover state from git and the ledger, then run experiments continuously — keep going experiment after experiment, and do NOT stop or hand back control until the discard budget is exhausted (create .autoresearch_done at that point). This outer loop will relaunch you if your session ends early, and you resume from git + the ledger. Use tools for every action.'

# Sessions can hang silently (gateway stream stalls without erroring), so every
# agent runs under timeout: SIGTERM at SESSION_TIMEOUT, SIGKILL 30s later.
# --foreground keeps the agent in the terminal's foreground process group;
# without it, timeout runs the agent in a background group and any TTY access
# (e.g. gemini's interactive UI) triggers SIGTTIN and suspends the session.
run_agent() {
    case "$AGENT" in
        # Codex non-interactive: --dangerously-bypass-approvals-and-sandbox runs
        # fully autonomously (edit/bash/git without prompts). The outer loop
        # relaunches Codex each time its session ends, so the search runs to
        # convergence instead of stopping after one ~30-min Codex session.
        codex)    timeout --foreground -k 30 "$SESSION_TIMEOUT" codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "$PROMPT" ;;
        gemini)   timeout --foreground -k 30 "$SESSION_TIMEOUT" gemini -m "$GEMINI_MODEL" --yolo -p "$PROMPT" ;;
        opencode) timeout --foreground -k 30 "$SESSION_TIMEOUT" opencode run -m "$OPENCODE_MODEL" "$PROMPT" ;;
        # Local Ollama: reuse opencode's agentic harness (tool-calling loop)
        # pointed at a model served by a local `ollama serve`. Fully offline.
        ollama)   timeout --foreground -k 30 "$SESSION_TIMEOUT" opencode run -m "$OLLAMA_MODEL" "$PROMPT" ;;
        claude)   timeout --foreground -k 30 "$SESSION_TIMEOUT" claude -p "$PROMPT" --permission-mode acceptEdits ;;
        *)        echo "[loop] unknown AGENT: $AGENT" >&2; return 1 ;;
    esac
}

ledger_rows() {
    if [ -f "$LEDGER" ]; then wc -l < "$LEDGER"; else echo 0; fi
}

# A session can die mid-experiment (rate limit, crash). Recover to a clean
# state so the next fresh session starts from a consistent git + ledger:
# - uncommitted ledger row -> the experiment finished; commit its log
# - other dirty tracked files -> the experiment never finished; reset them
recover() {
    if ! git diff --quiet -- artifacts/experiments 2>/dev/null; then
        git add artifacts/experiments
        git commit -q -m "chore(loop): commit experiment log left by interrupted session"
    fi
    if ! git diff --quiet; then
        git reset --hard -q
    fi
}

budget_exhausted() {
    LEDGER="$LEDGER" python - <<'PY'
import os
import sys
from pathlib import Path

from prepare import load_config

budget = load_config("config.yml").discard_budget  # author-set, search: in config.yml
ledger = Path(os.environ["LEDGER"])
if not ledger.exists():
    sys.exit(1)
lines = ledger.read_text().strip().splitlines()
if len(lines) < 2:
    sys.exit(1)
header = lines[0].split("\t")
si = header.index("status")  # locate by name (robust to added columns like val_roc_auc)
consecutive = 0
for line in lines[1:]:
    cols = line.split("\t")
    status = cols[si] if len(cols) > si else ""
    if status in ("keep", "baseline"):
        consecutive = 0
    elif status in ("discard", "crash"):
        consecutive += 1
sys.exit(0 if consecutive >= budget else 1)
PY
}

# Pre-build every discovered dynamic variable's full-archive mean/min/max/std
# aggregates HERE, in this plain shell process, before any agent session
# starts. This is deliberate: an agent's own tool-calling harness may kill a
# single command well before loop.sh's SESSION_TIMEOUT, and a first-time full
# NetCDF archive scan (hundreds of files) can take minutes. Building it here
# means it's never at risk of that shorter timeout; whatever's already cached
# is a near-instant no-op on every subsequent run. Month-filtered DERIVED
# entries are NOT pre-built (unbounded combinations) and still carry the
# timeout exposure documented in program.md's Run step.
echo "[loop] pre-building DERIVED climate aggregates (mean/min/max/std, no month filter)..."
python prepare.py --warm-derived-cache

experiments=0
wasted=0
launch=0
while [ "$MAX_EXPERIMENTS" -eq 0 ] || [ "$experiments" -lt "$MAX_EXPERIMENTS" ]; do
    if [ -f .autoresearch_done ]; then
        echo "[loop] .autoresearch_done flag found; stopping."
        break
    fi
    if budget_exhausted; then
        echo "[loop] discard budget exhausted (search.discard_budget); stopping."
        break
    fi
    launch=$((launch + 1))
    if [ "$MAX_EXPERIMENTS" -eq 0 ]; then
        echo "[loop] experiment $((experiments + 1)) (launch $launch, running to convergence): fresh $AGENT session"
    else
        echo "[loop] experiment $((experiments + 1))/$MAX_EXPERIMENTS (launch $launch): fresh $AGENT session"
    fi
    before=$(ledger_rows)
    run_agent || echo "[loop] agent exited non-zero"
    recover
    after=$(ledger_rows)
    if [ "$after" -gt "$before" ]; then
        experiments=$((experiments + after - before))
        wasted=0
    else
        wasted=$((wasted + 1))
        if [ "$wasted" -ge "$MAX_WASTED" ]; then
            echo "[loop] $MAX_WASTED consecutive sessions logged nothing; stopping."
            break
        fi
        echo "[loop] session logged nothing ($wasted/$MAX_WASTED consecutive); retrying in ${WASTED_SLEEP}s"
        sleep "$WASTED_SLEEP"
    fi
done

echo "[loop] finished after $experiments experiments ($launch launches). Best score:"
cat "$BEST_SCORE" 2>/dev/null || echo "(no best score yet)"
