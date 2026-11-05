#!/usr/bin/env python
"""Run the benchmark with whatever model is written in eval/model.yaml.

  python eval/run.py check              test that the model link in model.yaml works
  python eval/run.py dbbench-std 1      run the first 1 sample of a task
  python eval/run.py os-std 10          run the first 10 samples of a task
  python eval/run.py dbbench-std        run ALL samples of a task

Optional settings in model.yaml: temperature (default 0), concurrency (default 1),
runs (default 1), controller (default http://localhost:5020/api), extra_body (dict).
"""
import json
import os
import subprocess
import sys
import time

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    with open(os.path.join(HERE, "model.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("url", "model"):
        if not cfg.get(key):
            sys.exit(f"model.yaml: '{key}' is missing")
    url = str(cfg["url"]).strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if url.count("/") == 2:  # only scheme://host, no path -> assume /v1
        url += "/v1"
    cfg["url"] = url
    extra = dict(cfg.get("extra_body") or {})
    if cfg.get("thinking_off"):
        extra["reasoning_effort"] = "none"
    cfg["extra"] = extra
    return cfg


def check(cfg):
    headers = {"Authorization": "Bearer " + str(cfg.get("api_key", "placeholder")),
               "Content-Type": "application/json"}
    print(f"link : {cfg['url']}\nmodel: {cfg['model']}\n")
    try:
        r = requests.get(cfg["url"] + "/models", headers=headers, timeout=30)
        print(f"GET /models -> HTTP {r.status_code}")
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print("models available on that server:", ids)
            if cfg["model"] not in ids:
                print(f"WARNING: '{cfg['model']}' is not in that list (on Ollama run: ollama pull {cfg['model']})")
    except Exception as e:
        sys.exit(f"FAILED: cannot reach the link ({type(e).__name__}). Is the Colab tunnel still running?")
    body = {"model": cfg["model"], "max_tokens": 50, "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with just: pong"}]}
    body.update(cfg["extra"])
    t = time.time()
    try:
        r = requests.post(cfg["url"] + "/chat/completions", headers=headers, json=body, timeout=95)
    except Exception as e:
        sys.exit(f"FAILED: chat request error ({type(e).__name__}). First call after a model switch can be slow; try again.")
    dt = time.time() - t
    if r.status_code != 200:
        hint = " (timeout: model probably still loading, run check again)" if r.status_code == 524 else ""
        sys.exit(f"FAILED: chat -> HTTP {r.status_code}{hint}\n{r.text[:300]}")
    msg = r.json()["choices"][0]["message"]
    print(f"chat -> OK in {dt:.1f}s, reply: {msg.get('content')!r}")
    print("\nThe model link works.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cfg = load_cfg()
    if sys.argv[1] == "check":
        return check(cfg)
    task = sys.argv[1]
    cmd = [sys.executable, os.path.join(HERE, "server_agent.py"),
           "-m", str(cfg["model"]), "-u", cfg["url"],
           "--model-api-key", str(cfg.get("api_key", "placeholder")),
           "-c", str(cfg.get("controller", "http://localhost:5020/api")),
           "-j", str(cfg.get("concurrency", 1)),
           "-n", str(cfg.get("runs", 1)),
           "-t", str(cfg.get("temperature", 0)),
           "--extra-body", json.dumps(cfg["extra"])]
    if len(sys.argv) > 2:
        cmd += ["--limit", sys.argv[2]]
    cmd.append(task)
    print("running:", " ".join(cmd[:2]), "...", f"(model={cfg['model']}, task={task})")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
