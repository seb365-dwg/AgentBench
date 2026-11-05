# AgentBench (FC version): Project Architecture

This repo is a clone of **AgentBench** (THUDM, [arXiv 2308.03688](https://arxiv.org/abs/2308.03688)). AgentBench is a benchmark that tests **LLMs acting as autonomous agents**. A model is placed in an interactive environment (a Linux shell, a SQL database, a knowledge graph, a text-based house, a web shop) and has to solve multi-turn tasks by calling tools. The model is then scored on how well it does.

The checked-out branch is the **"FC" (Function Calling) release from Oct 2025**. The task environments were rewritten to use OpenAI-style function-calling tools and to run on top of the external **AgentRL** framework (`agentrl-worker` on PyPI). Some of the older v0.2 code (the client/assigner side) is still in the repo. See [Known inconsistencies](#known-inconsistencies) below.

---

## 1. High-level architecture

The system has three decoupled parts that talk over HTTP:

```
                   ┌──────────────────────────────┐
                   │   Agent Server (the LLM)     │   OpenAI / Claude API,
                   │                              │   or a local model via FastChat
                   └──────────────▲───────────────┘
                                  │ inference(history)
┌─────────────────────────────────┴──────────────────────────────────┐
│ CLIENT  (src/assigner.py + src/client/)                            │
│  Assigner ── builds agent×task graph, runs max-flow scheduling     │
│     └─ spawns workers, each with an AgentClient + TaskClient       │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │ POST /api/start_sample, /api/interact
                   ┌──────────────▼───────────────┐
                   │   Task Controller            │   one global instance
                   │   (AgentRL controller)       │   (Docker: jingbh/agentrl-controller)
                   └──────┬───────┬───────┬───────┘
                          │       │       │  routes sessions to free workers
                ┌─────────▼┐ ┌────▼────┐ ┌▼──────────┐
                │ os worker│ │db worker│ │ kg worker │ ...  (src/server/tasks/*)
                └────┬─────┘ └────┬────┘ └────┬──────┘
                     │            │           │
               Ubuntu docker   MySQL docker  Freebase SPARQL (Virtuoso)
               containers      containers    server
                                    ▲
                              Redis (tracks container allocation)
```

### Flow of one test case
1. The **Assigner** reads `configs/assignments/*.yaml` to find out which agents to run on which tasks and how much concurrency is allowed.
2. It builds a bipartite graph (agents ↔ tasks) with source and sink nodes. The edge capacities are the agent concurrency, the task concurrency and the number of pending samples. It then runs **max-flow** (`src/utils/max_flow.py`) to decide how many workers to start for each agent/task pair.
3. Each worker calls `TaskClient.run_sample(index, agent)`:
   - `POST /start_sample` to the controller. The controller picks a free task worker, which returns the initial prompt plus the tool definitions.
   - The loop is: `agent.inference(history)` produces a tool call, then `POST /interact` sends it to the environment, which returns an observation. This repeats until the task finishes or the round limit is hit.
4. The result (status, history, score) is written to `outputs/{TIMESTAMP}/<agent>/<task>/`.
5. `src/analysis.py` collects the outputs into per-task metrics (success rate, F1, win rate…).

### Why it's split this way
Each environment needs very different resources. WebShop needs about 15 GB of RAM, OS and DB need Docker-in-Docker, and KG needs a large Freebase database. Running each one as its own worker process or container lets you scale, restart or move it to another machine without touching the others. Separate processes also get around Python's GIL.

---

## 2. Folder-by-folder guide

### Root files
| File               | Purpose                                                                                                                                                                                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`        | Upstream docs: FC quick start (Docker Compose), the v0.2 quick start and the leaderboard.                                                                                                                                                                            |
| `requirements.txt` | Python deps for the **client side** (pydantic 1.x, fastapi, requests, fschat, anthropic, transformers…). Use **Python 3.9**. It does **not** include `agentrl-worker` or `openai`, which the task workers need. Those get installed inside the worker Docker images. |
| `LICENSE`          | Apache-2.0.                                                                                                                                                                                                                                                          |
| `.gitignore`       | Ignores outputs, caches and so on.                                                                                                                                                                                                                                   |

### `src/`: all Python source
| Path              | What it does                                                                                                                                                                                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/assigner.py` | **Entry point for running an evaluation** (`python -m src.assigner [--config ...]`). Loads the assignment config, checks which tasks and agents are alive, schedules work with max-flow, runs worker threads, retries failures and writes results and `overall.json`. |
| `src/configs.py`  | YAML config loader. Handles `import:` (pulling in other YAML files), `default:` blocks and `overwrite:` merging. Every config in `configs/` goes through it.                                                                                                          |
| `src/analysis.py` | Post-processing (`python -m src.analysis`). Walks an `outputs/` directory and pulls out each task's main metric through per-task `TaskHandler` classes (OS, DB, KG, DCG, HH, WS, LTP, M2W). It builds the summary tables.                                             |

#### `src/client/`: the "Client" layer (talks to models and to the controller)
| Path                        | What it does                                                                                                                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent.py`                  | Abstract `AgentClient` base class with one method: `inference(history) -> str`.                                                                                                                                                            |
| `task.py`                   | `TaskClient`, the HTTP client for the Task Controller (`get_indices`, `list_workers`, `start_sample`, `interact`). `run_sample()` is the main agent↔environment loop.                                                                      |
| `agent_test.py`             | `python -m src.client.agent_test --config ... --agent ...`. Lets you chat with a configured agent interactively to check that your API key and config work.                                                                                |
| `agents/http_agent.py`      | `HTTPAgent`, a generic agent for any HTTP LLM API (OpenAI chat or completions, etc.). Configurable URL, headers, body template, a "prompter" that converts the history to the API's format, and a `return_format` that extracts the reply. |
| `agents/claude_agent.py`    | Agent backed by the Anthropic SDK (old `anthropic~=0.4` completion API).                                                                                                                                                                   |
| `agents/fastchat_client.py` | `FastChatAgent` for **locally hosted open models** (Vicuna, WizardLM, ChatGLM…) served by a FastChat controller. It includes per-model prompt templates.                                                                                   |
| `agents/test_agent.py`      | Dummy agent used for testing the pipeline without a real LLM.                                                                                                                                                                              |

#### `src/server/tasks/`: the task environments (one folder per benchmark)
Each folder is a **self-contained AgentRL task worker**. It subclasses `agentrl.worker.task.Task`, defines its tools in the matching `configs/tasks/*.yaml`, and has its own `Dockerfile` that installs `agentrl-worker`, copies only its own code and data, and starts with `extra/worker-entrypoint.sh`.

| Folder | Environment | Agent's tools | Metric |
|---|---|---|---|
| `os_interaction/` | Real Ubuntu bash inside Docker containers it starts. `task.py` (~740 lines) handles container lifecycle, running init scripts, executing the agent's bash and running check scripts. `environment.py` manages containers through AgentRL's environment controller. | `bash_action`, `answer_action`, `finish_action` | Success rate |
| `dbbench/` | A real MySQL (or SQLite) database per sample. `interaction.py` wraps the DB connections, `environment.py` starts MySQL containers, and `result_processor.py` normalises and compares query results and final answers. | `execute_sql`, `commit_final_answer` | Success rate |
| `knowledgegraph/` | Question answering over **Freebase** through SPARQL. `api.py` provides the KG navigation actions (get relations, get neighbours, intersection, count, argmax…). `utils/` handles SPARQL execution and logical-form parsing. `ontology/` holds the Freebase schema files (types, roles, reverse properties, vocab). `const.py` holds the prompts and tool specs. | relation/neighbour/intersection… functions | Answer F1 |
| `alfworld/` | **ALFWorld** text-based household simulator (TextWorld + PDDL). `environment.py` wraps the ALFWorld env, `configs/base_config.yaml` configures it, `prompts/` holds the prompt variants (react, CoT, plan-first, single/multi-turn) and `utils.py` handles parsing. | `take_action` | Success rate |
| `webshop/` | **WebShop** simulated e-commerce site (Princeton). `webshop.patch` patches the upstream WebShop code, and `requirements.txt` lists its deps. Needs about 16 GB of RAM. | `search_action`, `click_action` | Reward score |

### `src/typings/`: shared data models (pydantic)
| File           | Contents                                                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`    | Schemas for assignment, agent, task and concurrency configs.                                                                                                  |
| `general.py`   | Basic types: `SampleIndex`, `ChatHistoryItem`, `Assignment`, and `InstanceFactory` (builds an object from `module` + `parameters` in YAML).                   |
| `request.py`   | HTTP request bodies (`StartSampleRequest`, `InteractRequest`, `CancelRequest`, `RegisterRequest`, `HeartbeatRequest`, …).                                     |
| `output.py`    | Result objects (`TaskOutput`, `TaskSampleExecutionResult`, `AgentOutput`, `TaskClientOutput`).                                                                |
| `status.py`    | Enums: `SampleStatus` (completed, agent context limit, task limit reached…), `WorkerStatus` and `AgentOutputStatus`.                                          |
| `exception.py` | Exception hierarchy: `AgentBenchException` → client/server exceptions → agent context-limit, timeout and network errors, and task timeout and network errors. |

### `src/utils/`: helpers
| File | Contents |
|---|---|
| `max_flow.py` | Max-flow implementation (augmenting paths) used by the Assigner for scheduling. |
| `rules.py` | Small composable predicate rules (`ContainRule`, `AndRule`, `OrRule`, `NotRule`) used to filter tasks and agents. |
| `others.py` | Misc: `ColorMessage` (coloured terminal output), `JsonEncoder` and `serialize()` (for writing results to JSON). |

### `configs/`: everything is configured in YAML
| Path                                       | Purpose                                                                                                                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agents/openai-chat.yaml`                  | Template for OpenAI chat models through `HTTPAgent`. **Your API key goes here.**                                                                                                                        |
| `agents/openai-text.yaml`                  | Same, for the old completions endpoint (davinci).                                                                                                                                                       |
| `agents/api_agents.yaml`                   | Named API agents (e.g. `gpt-3.5-turbo-0613`), each importing one of the templates above with a model name.                                                                                              |
| `agents/fs_agent.yaml`                     | Named local models served through FastChat (vicuna-13b/33b, wizard-30b…).                                                                                                                               |
| `tasks/*.yaml`                             | One per environment. Each gives the Python `module` to load, its parameters (concurrency, round limits, data paths, **tool/function schemas**) and its splits (`*-std` = test, `*-dev`, `*-env_train`). |
| `tasks/task_assembly.yaml`                 | Imports all the task YAMLs into one catalogue.                                                                                                                                                          |
| `assignments/definition.yaml`              | Tells the client where the controller is (`localhost:5000/api`) and which agent and task catalogues to load.                                                                                            |
| `assignments/default.yaml`                 | Default run: `gpt-3.5-turbo-0613` on `dbbench-std` + `os-std`, concurrency 5.                                                                                                                           |
| `assignments/lite.yaml`                    | Same with concurrency 1 (for laptops).                                                                                                                                                                  |
| `assignments/test_avalon.yaml`             | Example assignment for the Avalon multi-agent game.                                                                                                                                                     |
| `start_task.yaml` / `start_task_lite.yaml` | How many task workers to launch per task (5 or 1) for the old `src.start_task` launcher.                                                                                                                |

### `data/`: benchmark datasets
| Folder | Contents |
|---|---|
| `os_interaction/` | `data/1…7/*.json` + `dev.json`: the OS questions (description, init script, checking method). `scripts/<n>/{init,check,example}`: shell and Python scripts that set up the container and check answers (`integer-match.py`, `string-match.py`, `size-match.py`…). `res/dockerfiles/{default,packages,ubuntu}`: Dockerfiles for the `local-os/*` images the agent runs inside. `train_0317/`: training split. |
| `dbbench/` | `standard.jsonl` (test), `dev.jsonl`, `db_out_new.jsonl`, `extend_std.json`. Each entry has a table schema and rows, a question and the expected answer or table state. |
| `knowledgegraph/` | `std.json` (test), `dev.json`, `kg_rl_all.json` (RL training). Questions with gold Freebase answers. |
| `alfworld/` | `standard.json`, `new_std.json`, `dev.json`, `train_valid.json` (game file lists) + `logic/` (PDDL domain `alfred.pddl` and grammar `alfred.twl2`). |
| `mind2web/` | Only `prompt/llm_prompt*.json`. The actual Mind2Web data comes with the `longinyu/agentbench-mind2web` Docker image. |
| `lateralthinkingpuzzle/` + `_zh/` | Puzzle sets (`standard.xlsx`, `dev.xlsx`) in English and Chinese. |
| `avalon/` | `dev.json`: game setups for AvalonBench. |

### `extra/`: deployment
| File | Purpose |
|---|---|
| `docker-compose.yml` | **One-command FC stack**: AgentRL controller + one worker each for alfworld, dbbench, kg, os, webshop + Freebase server + Redis. Workers connect to the controller at `172.17.0.1:5020`. |
| `worker-entrypoint.sh` | Container entrypoint: runs `python -m agentrl.worker --self http://<ip>:5021/api <task-name>`. |
| `freebase.Dockerfile` / `freebase-entrypoint.sh` | Builds a Virtuoso SPARQL server that serves the Freebase DB (you must download `virtuoso.db` separately into `extra/virtuoso_db/`). |

### `scripts/`
| File | Purpose |
|---|---|
| `validate_lite_configs.py` | Quick check that only needs PyYAML (no Docker). It verifies that the lite configs parse and only reference tasks that exist in `task_assembly.yaml`. |

### `docs/`
English (`_en`) and Chinese (`_cn`) guides: `Introduction` (framework design, the same content as §1 above), `Config` (YAML syntax: `import`, `default`, `overwrite`), `Entrance` (CLI entry points and flags) and `Extension` (how to add a new task).

### `assets/`
Images used by the README: leaderboards, the architecture diagram, dataset statistics and the logo.

### `.github/`
- `workflows/lite-configs.yml`: CI job that runs `scripts/validate_lite_configs.py` on every push or PR.
- `ISSUE_TEMPLATE/`: bug report and feature request templates.

---

## 3. Known inconsistencies

These matter if you plan to run or extend the code:

1. **`src/start_task.py` and the in-repo Task Controller no longer exist.** The README's v0.2 instructions (`python -m src.start_task -a`) will fail. In the FC version the controller is the external AgentRL image (`jingbh/agentrl-controller`), and workers are started by `docker compose -f extra/docker-compose.yml up`.
2. **Port mismatch:** the client (`configs/assignments/definition.yaml`) expects the controller at `localhost:5000/api`, while the Compose stack's workers register with the controller on port **5020**. You'll probably need to change the address in `definition.yaml`.
3. **Task configs without code:** `task_assembly.yaml` imports `mind2web`, `card_game`, `ltp` and `avalon`, but `src/server/tasks/` only contains `os_interaction`, `dbbench`, `knowledgegraph`, `alfworld` and `webshop`. The other four come from v0.2 and only work through their old Docker images (or by checking out the `v0.2` tag).
4. **Two dependency worlds:** the client (`requirements.txt`, Python 3.9, pydantic 1) and the workers (Python 3.10 in Docker, `agentrl-worker`, `openai` types) are installed separately. The workers' code isn't meant to run in your local venv.
5. **macOS:** port 5000 is used by AirPlay Receiver. Turn it off in System Settings → General → AirDrop & Handoff, or change the port.

---

## 4. How to add a new task (summary of `docs/Extension_en.md`)
1. Create `src/server/tasks/<name>/` with a class that subclasses the AgentRL `Task` and implements index listing, starting a session, and the interaction loop.
2. Add `configs/tasks/<name>.yaml` with the `module`, parameters, tool schemas and splits, then import it in `task_assembly.yaml`.
3. Put the data in `data/<name>/` and write a `Dockerfile` modelled on an existing task.
4. Add a `TaskHandler` in `src/analysis.py` so the metrics show up in the summary.
5. Reference the task in an assignment YAML and run the assigner.
