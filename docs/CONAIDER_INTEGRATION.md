# conaider integration guide (agent mode + GitLab)

This fork of aider adds two capabilities on top of vanilla aider 0.86.x:

1. **Agent mode** — a multi-turn tool-calling loop (`--agent`), designed to work
   with small local models via Ollama (including `ollama_chat/gemma3`, which has
   **no native tool-calling API** — see below).
2. **GitLab issue-to-MR** — fetch a GitLab issue, make the edit, push a branch,
   open a draft MR (`/gitlab-issue <iid>` or `--gitlab-issue <iid>`).

It is meant to drop into [conaider](../../conaider) as a replacement for the
`aider-chat` package. This document is the integration contract.

---

## 1. Compatibility guarantees

The changes are additive and isolated in new modules (`aider/agent/`,
`aider/gitlab.py`, `aider/coders/agent_coder.py`). Shared files got minimal,
backwards-compatible edits (`args.py`, `main.py`, `models.py`,
`coders/__init__.py`, `repo.py`). Specifically:

- The `aider` console entrypoint and **all existing behavior are unchanged.**
  Without `--agent`, aider behaves exactly as vanilla 0.86.x.
- **Defaults are unchanged.** No model's default edit format changes; agent mode
  is opt-in via a flag.
- **Auto-commit still fires** the same way (agent mode makes exactly one commit
  per turn). conaider's `post-commit` push hook keeps working unchanged.
- **No new mandatory dependencies.** `python-gitlab` is an optional extra; if it
  is not installed, only the GitLab feature is unavailable (with a clear
  message). Agent mode needs nothing beyond aider's existing deps.
- **No new network calls** are made at startup or during a normal chat. Agent
  tools only touch the local repo and the local Ollama.

Existing test suites (`test_editblock`, `test_models`, `test_main`,
`test_commands`, `test_repo`, `test_coder`, `test_sendchat`) pass unchanged, plus
new suites `test_agent_*` and `test_gitlab`.

---

## 2. Installing the fork in conaider

Replace the aider install line in conaider's `Dockerfile` (currently
`RUN pip install --no-cache-dir aider-chat mcp httpx`) with this fork, **including
the gitlab extra**:

```dockerfile
# from a git URL:
RUN pip install --no-cache-dir "aider-chat[gitlab] @ git+https://github.com/binRick/aider@agent-mode" mcp httpx
# or, for the air-gapped build, from a locally-built wheel + wheelhouse:
RUN pip install --no-cache-dir "aider-chat[gitlab]" mcp httpx \
      --find-links /wheels --no-index
```

### Offline bundle (OFFLINE.md)

`python-gitlab` (and its transitive deps `requests-toolbelt`; `requests` is
already an aider dep) must be added to the offline wheelhouse. **Runtime
`pip install` is not possible in the air-gapped container** — the code detects a
missing `python-gitlab` and prints an install hint rather than attempting to
fetch it. Add to the pip manifest:

```
python-gitlab>=4,<9
requests-toolbelt
```

---

## 3. Wiring the model-settings file (do this — the template is currently unwired)

conaider ships `conventions/aider.model.settings.yml` but **does not currently
wire it** (its own `docs/INDEX.md` marks it "Template (not wired)"; the
Dockerfile does not `COPY` it and `conaider-tui.sh` does not pass
`--model-settings-file`). aider **silently ignores** a missing settings file, so
without wiring, gemma3 runs with the wrong `num_ctx` and no agent tuning.

Wire it:

```dockerfile
# conaider Dockerfile, after the existing COPY block:
COPY conventions/aider.model.settings.yml /etc/conaider/aider.model.settings.yml
RUN chmod 0444 /etc/conaider/aider.model.settings.yml
```

```sh
# conaider scripts/conaider-tui.sh, the launch line:
aider --model "$MODEL" \
      --model-settings-file /etc/conaider/aider.model.settings.yml \
      --agent \
      --no-check-update "$@" || true
```

Recommended settings block for agent mode with gemma3:

```yaml
- name: ollama_chat/gemma3
  edit_format: agent          # use the tool-calling loop
  tool_protocol: text         # gemma3 has no native tools API (see §5)
  use_repo_map: true          # repo map flows into the agent's opening context
  streaming: true
  extra_params:
    num_ctx: 8192             # THE fix for silent truncation; size to the box's RAM
```

> `tool_protocol` is a new per-model field. It is behavior-neutral for non-agent
> modes, so shipping it does not affect vanilla conaider sessions.

**Verify it took** (aider does not error on a missing settings file):
- the banner shows `... with agent edit format`;
- `docker exec conaider-ollama ollama ps` shows the larger CONTEXT (not 2048);
- Ollama logs stop printing `truncating input`.

---

## 4. Using agent mode

CLI flags (each also available as `AIDER_*` env var and `.aider.conf.yml` key):

| flag | default | meaning |
| --- | --- | --- |
| `--agent` | off | select the agent edit format (`--edit-format agent`) |
| `--agent-tool-protocol {auto,native,text}` | `auto` | how tool calls are requested (§5) |
| `--agent-max-iterations N` | 20 | max tool-loop iterations per turn |
| `--agent-allow-shell` / `--no-agent-allow-shell` | off | permit the `bash` tool |
| `--agent-auto-approve-edits` / `--no-...` | off | apply edits without a prompt |

Tools available to the model: `read_file`, `list_files`, `grep`, `edit_file`
(reuses aider's SEARCH/REPLACE matching), `write_file`, `bash` (opt-in), `git`
(read-only subcommands), and `done`.

**Permissions.** Reads are auto-allowed. Edits prompt (batchable with "All")
unless `--agent-auto-approve-edits`. `bash` is off unless `--agent-allow-shell`
and always requires an explicit `y` — `--yes-always` alone will **not**
auto-approve shell commands, matching aider's existing shell policy. Under ttyd,
these prompts work the same as any other aider prompt.

**Commit / undo.** Each agent turn makes exactly one commit over the files it
edited (so conaider pushes once per turn), and `/undo` reverts it.

---

## 5. The gemma3 tool-calling reality (read before choosing a model)

Ollama's stock **gemma3 does not support the tools API**: a `tools=` request
returns HTTP 400 `"...gemma3 does not support tools"`. So agent mode uses a
**text protocol** for gemma3: tools are described in the system prompt and the
model emits one fenced ` ```tool ` block per turn (with large file bodies passed
in a `<<<PAYLOAD … PAYLOAD>>>` block so code never has to survive JSON escaping).
`--agent-tool-protocol auto` (the default) picks `text` for Ollama models and
`native` for tool-trained models; it also auto-downgrades to `text` if a model
rejects `tools=` mid-session.

**Reliability is model-dependent and must be measured.** Run the Phase 0 spike
before trusting gemma3:4b in production:

```sh
docker exec -e OLLAMA_API_BASE=http://ollama:11434 conaider-tui \
    python -m benchmark.agent_spike --model ollama_chat/gemma3 --trials 30
```

It prints parse rate, correct-tool rate, and a PROCEED / DO-NOT-rely
recommendation, measured at realistic context depth. If gemma3:4b misses the
gate, step up to `gemma3:12b`, or a tools-native **non-Chinese** model
(`llama3.1`, `gpt-oss:20b`, `granite4.1`) — record the numbers here.

> **Native path + Ollama caveat.** If you adopt a tools-native model, prefer the
> `ollama_chat/` prefix (aider sizes `num_ctx` for it). If you instead route
> through Ollama's OpenAI-compatible `/v1` endpoint, aider cannot set `num_ctx`
> per request — set `OLLAMA_CONTEXT_LENGTH` on the `ollama` service in
> `compose.yaml` or the model is silently truncated.

Keep the `ollama_chat/` prefix (not `ollama/`) — it uses the proper chat
template.

---

## 6. GitLab issue-to-MR

Config resolves from flags, then env, then the git remote. In conaider the
**zero-config path works with no consumer changes**: aider recovers the token and
base URL from the `oauth2:<token>@gitlab/...` origin remote that conaider's clone
already sets.

Config knobs (flag / env):

- URL: `--gitlab-url` / `GITLAB_URL` (conaider already sets `GITLAB_URL=http://gitlab`)
- token: `--gitlab-token` / `GITLAB_TOKEN` / `GITLAB_PRIVATE_TOKEN` (falls back to
  the oauth2 remote, i.e. conaider's `CONAIDER_PAT`)
- project: `--gitlab-project` / `CI_PROJECT_PATH` (falls back to the remote path)
- `--gitlab-close-keyword` (default `Closes`), `--gitlab-target-branch`

Usage:

- Interactive: `/gitlab-issue 42`
- Batch: `aider --gitlab-issue 42 --agent`

Flow: fetch issue → (reuse the current conaider session branch, or create
`<iid>-<slug>`) → run the coder on the issue text → push → open a **draft** MR
with `Closes #<iid>` in the description → print the MR URL. If no commits were
made, MR creation is skipped.

To make the token explicit rather than relying on the remote, add one line to
`conaider-tui.sh` before the aider launch:

```sh
export GITLAB_TOKEN="$CONAIDER_PAT"
```

Because conaider already checks out a `conaider/<timestamp>` session branch and
pushes via the post-commit hook, `/gitlab-issue` reuses that branch as the MR
source (no second branch, no double-push).

---

## 7. Invariants to preserve in conaider

- Keep `--auto-commits` on (agent mode relies on it; clones are ephemeral).
- Keep the `ollama_chat/` model prefix.
- Agent mode commits once per turn → one push per turn.
- Permission prompts must reach the user (works under ttyd today).

---

## 8. Testing locally

```sh
pip install -e '.[gitlab,dev]'
pytest tests/basic/test_agent_protocol.py tests/basic/test_agent_tools.py \
       tests/basic/test_agent_permissions.py tests/basic/test_agent_coder.py \
       tests/basic/test_gitlab.py
```

The issue-to-MR live test is most faithful **inside the conaider-tui container**
(the fork installed in the image). From the host, use
`GITLAB_URL=http://localhost:8080`; note API-returned URLs will reference
`http://gitlab` (the in-network hostname).
