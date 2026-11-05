# Run notes: compatibility check and model-switching checklist

Written 2026-11-06. Everything here was checked by reading code or by a live test; items I could not verify are
listed in section 6. Companion to `ARCHITECTURE.md`.

---

## 1. Verdict

**The old client in this repo (`src/assigner.py`, `src/client/task.py`, `HTTPAgent`) cannot run the FC tasks.**
It speaks the protocol of the old (v0.2) controller, and the controller the FC stack uses is a different program
(`jingbh/agentrl-controller`, a compiled Go binary that listens on port 5020).

**Use AgentRL's own evaluation client instead:** `examples/eval/server_agent.py` in https://github.com/THUDM/AgentRL.
It speaks the new protocol, accepts any OpenAI-compatible endpoint (so it works with the Colab/Ollama tunnel), and
sends the tools. A copy was downloaded to read it (scratchpad, not in this repo).

## 2. Evidence (old client vs. new controller)

| Area | Old client in this repo | New controller / workers | How verified |
|---|---|---|---|
| Controller port | `localhost:5000` (`configs/assignments/definition.yaml`) | **5020** (image `EXPOSE`, server log "HTTP server started at :5020") | image metadata + live run |
| Interact request | `{session_id, agent_response:{content:"text"}}` | headers `session_id`; body `{messages:[<OpenAI assistant msg>]}` | worker source (`InteractRequest`), `server_agent.py` |
| Start response | body has `session_id` and `output.history` | `session_id` is an **HTTP header**; body has `messages`, `tools`, `status`, `finish`, `reward`, `metric` | worker source, `server_agent.py` |
| History | full history each turn | **incremental** messages; the client keeps the history | worker source |
| Tools | never sent; reply is a text string | `tools` returned by the task; reply must be an assistant message that may contain `tool_calls` | worker source; `http_agent.py` has no `tools` |
| Final scoring | `POST /calculate_overall` on the controller | route does **not exist** (live test: HTTP 404; string absent from binary) | live test |
| Old field name | `agent_response` | string absent from the controller binary | binary search |

Live networking test (Docker Desktop, macOS):
- `network_mode: host` as in `extra/docker-compose.yml`: controller starts but `localhost:5020` from the Mac is
  **refused** ("host" = Docker's hidden Linux VM, not the Mac).
- `-p 5020:5020`: works. `GET /api/version` returned `{"variant":"nova","version":1}`.

## 3. Recommended way to run (no rewrite of the old client)

```
# one-time: tooling for server_agent.py (use a separate venv or the project .venv)
pip install pandas aiohttp openai tqdm

# run (example: 5 OS samples, 1 run, 1 concurrent request, Colab model)
python server_agent.py \
  -m qwen3:8b \
  -u https://<TUNNEL>.trycloudflare.com/v1 \
  -c http://localhost:5020/api \
  -j 1 -n 1 -r 0,5 -t 0 \
  os-std

# summarise
python check.py ./results/<file>.jsonl
```

Flag notes (from the script): `-u` must end at `/v1` with no trailing slash (it appends `/chat/completions`);
`-j` default is **64**; `-n` default is **4 runs per sample**; `-t` default is **0.8** (the paper used 0);
`-r a,b` selects indices; the score per sample is the `reward` returned by the task.

## 4. Changes required before the first run

### Blockers
1. **Docker Compose for macOS: DONE** as `extra/docker-compose.mac.yml` (original untouched). Verified 2026-11-06: controller,
   redis, dbbench-std and os-std start; workers show ALIVE; controller lists 300 dbbench-std and 144 os-std samples (OS
   indices are integers 0..143). Start: `docker compose -f extra/docker-compose.mac.yml up -d`; stop: `... down`.
   What it changes versus the original:
   - remove `network_mode: host` from `controller` and `redis`; add `ports: ["5020:5020"]` to the controller.
   - change every worker's `--controller http://172.17.0.1:5020/api` to `--controller http://controller:5020/api`
     (service name on the Compose network).
   - give workers a Redis address that is not `localhost` (needs a `state_options.connection.host` entry in the
     task YAML, which can then be overridden by env vars; the env-var naming is in
     `agentrl/worker/config/loader.py`). *Not tested.*
   - start only: `controller redis dbbench-std os_interaction-std`.
2. **Copy `server_agent.py` and `check.py`** into the project (suggest a new `eval/` folder) so they are versioned.
3. **Patch the copy of `server_agent.py`** (small edits):
   - add `"reasoning_effort": "none"` to the payload for Qwen3 (otherwise thinking is on: slow, and risks the
     100 s Cloudflare timeout, HTTP 524, which already happened once on a cold start).
   - `max_completion_tokens` is hard-coded to 1024; make it a flag if needed.
   - `-T` (`--truncate-multiple-toolcall`) does `message["tool_calls"][:1]`, which raises `KeyError` when the model
     replies without tool calls. Don't use `-T` until patched.
   - consider dropping non-standard keys (e.g. `reasoning`) from the assistant message before posting it back.
4. **MySQL memory**: already lowered `innodb_buffer_pool_size` to 1G in `src/server/tasks/dbbench/environment.py`;
   `innodb_log_file_size=1G` in the same file also lives in RAM (tmpfs), so reduce it too if the DB task is killed.

### Housekeeping
5. Add `.venv/` and `results/` to `.gitignore`.
6. `src/assigner.py`, `src/client/*`, `configs/assignments/*` are legacy for the FC stack. Don't delete them; just
   don't rely on them. (Their `gpt-3.5-turbo-0613` defaults are irrelevant when using `server_agent.py`.)

## 5. Switching models: checklist

### What stays the same for every model served through Ollama on Colab
- The Colab cells (Ollama install, `ollama serve`, tunnel) are unchanged. The tunnel URL changes each time you
  restart the tunnel cell; pass the new one to `-u`.
- Pull the new model: `!ollama pull <tag>`, then pass `-m <tag>` (or omit `-m` and the script uses the first model
  listed by `/v1/models`, which is wrong if more than one model is pulled).
- Run one model at a time on a T4 (about 15 GB). Free the old one: `!ollama stop <old-tag>`.

### Per model

| Switch to | Change | Risk / what to check |
|---|---|---|
| **Qwen3-8B** (current) | `-m qwen3:8b`, `reasoning_effort: none` | Verified working with chat; tool calls through the controller **not yet tested**. |
| **Qwen3-4B** | `!ollama pull qwen3:4b`, `-m qwen3:4b` | You wrote "qwen 3b": Qwen3 has no 3B size (sizes are 0.6B, 1.7B, 4B, 8B, 14B, 32B). Qwen2.5 has a 3B (`qwen2.5:3b`). Confirm which you want. Same thinking handling as 8B. |
| **Llama 3.1 8B** | `!ollama pull llama3.1:8b`, `-m llama3.1:8b`; **remove** `reasoning_effort` | Not a thinking model, so the flag is unnecessary and may be rejected. Should support tools in Ollama; confirm with `ollama show llama3.1:8b` (look for a `tools` capability). |
| **Vicuna** (or any 2023-era model) | **Not directly possible with the FC tasks** | Vicuna v1.5 has no function-calling template; Ollama is expected to reject a request that includes `tools`. Options: (a) use a model with tool support; (b) write a text-to-tool adapter (put the tool schema in the prompt, parse the reply into `tool_calls`); (c) use AgentBench tag `v0.2`, the text-prompt version the paper used, for a like-for-like reproduction. Option (c) means a different stack. |
| **Any other model** | `ollama pull`, then check capabilities | Needs a tool-capable chat template. Small models often emit malformed tool calls; expect low scores (the paper's open models scored 3 to 15 percent on OS). |
| **gpt-oss-20b** | via Ollama or Transformers | Memory is tight on a T4; confirm it loads first. Its reasoning control may differ. |

### Settings that interact with the model choice
- **Context length**: Colab cell sets `OLLAMA_CONTEXT_LENGTH=8192`. FC histories are not truncated like the paper's
  (3500 tokens), so long OS/DB episodes may exceed it. If you see truncation or odd replies, raise to 16384 and
  watch GPU memory with `nvidia-smi`. Larger models leave less room.
- **Temperature**: `server_agent.py` defaults to 0.8; pass `-t 0` to match the paper's greedy decoding.
- **Concurrency** (`-j`): keep 1 to 2 on a free T4; the default 64 will time out.
- **Cloudflare limit**: any single request over about 100 s returns 524. Keep thinking off, or switch tunnels.
- **If you later move to vLLM instead of Ollama**: tool calling must be enabled per server with
  `--enable-auto-tool-choice --tool-call-parser <name>`, and the parser name depends on the model family
  (Qwen uses `hermes`; others differ). Vicuna still has no tool template.
- **Result files** are named `<task>-<model>-<temp>-<MMddHHmm>.jsonl`; model tags with `:` are fine on macOS.
- **Comparing models**: keep `-r`, `-n`, `-t` identical across runs, or the numbers aren't comparable.
- **Paper comparison**: the paper's table came from v0.2 (text prompts). Treat it as context, not as a baseline for
  these FC runs.

## 6. Not verified (do not assume)
- A full end-to-end run (worker + controller + `server_agent.py` + Colab model). The controller was started and
  queried alone; no worker was running.
- Whether workers can reach Redis: they registered with the controller (so that link works) and logged no errors, but
  Redis is only used when a sample creates its container, so the first real sample is the test.
- Whether the controller accepts the extra `reasoning` key in assistant messages.
- `agentrl-eval` (the experimental CLI in the AgentRL repo): its arguments were not read.
- Qwen3-4B / Llama 3.1 tool-calling behaviour through Ollama on a T4.
