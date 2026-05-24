# agent-state-checkpoint

[![PyPI](https://img.shields.io/pypi/v/agent-state-checkpoint.svg)](https://pypi.org/project/agent-state-checkpoint/)
[![Python](https://img.shields.io/pypi/pyversions/agent-state-checkpoint.svg)](https://pypi.org/project/agent-state-checkpoint/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Durable JSON checkpoint for long-running AI agents.**

Long-running agents need a way to save state and pick back up after a
crash, restart, or sleep. This library is a small, zero-dependency
checkpoint file with atomic writes, schema versioning, and optional
rotating snapshots.

JSON-only by design: portable across processes and Python versions,
human-inspectable, no `pickle` security risk.

## Install

```bash
pip install agent-state-checkpoint
```

## Use

```python
from agent_state_checkpoint import Checkpoint

ck = Checkpoint("~/agent-runs/run-2026-05-24.json", schema_version=1)

# Save current state any time the agent advances.
ck.save({
    "step": 12,
    "messages": [{"role": "user", "content": "..."}],
    "tool_calls_made": ["search", "summarize"],
    "cost_usd": 0.42,
})
```

Later, after a crash, restart, or planned resume:

```python
ck = Checkpoint("~/agent-runs/run-2026-05-24.json", schema_version=1)
state = ck.load()
if state is not None:
    resume_from(state)
```

`load()` returns `None` when no checkpoint exists yet, so the same code
handles fresh starts and resumes.

## Atomic writes

`.save()` writes to `path.tmp` and then `os.replace(path.tmp, path)`. If
the process crashes mid-write, the previous good checkpoint stays intact.
You will never see a half-written JSON file.

## Schema versioning and migrations

When you change the shape of saved state, bump `schema_version` and
provide migration callables keyed by from-version:

```python
from agent_state_checkpoint import Checkpoint

def v1_to_v2(state: dict) -> dict:
    # rename "messages" to "history"
    state["history"] = state.pop("messages", [])
    return state

def v2_to_v3(state: dict) -> dict:
    state.setdefault("cost_usd", 0.0)
    return state

ck = Checkpoint(
    "~/agent-runs/run.json",
    schema_version=3,
    migrations={1: v1_to_v2, 2: v2_to_v3},
)

state = ck.load()  # auto-migrates 1 -> 2 -> 3 if needed
```

If a step has no migration, `MigrationMissingError` is raised so you can
catch it during deploy. Newer checkpoints than the running code are
rejected to avoid silent downgrade.

## Other methods

```python
ck.exists()     # True/False
ck.delete()     # remove the file (no error if absent)
ck.history(10)  # last N rotating snapshots (opt in via keep_history=True)
```

To keep rotating snapshots alongside the current file:

```python
ck = Checkpoint("~/agent-runs/run.json", schema_version=1, keep_history=True)
ck.save({"step": 1})
ck.save({"step": 2})
ck.save({"step": 3})
ck.history(10)  # [{"step": 3}, {"step": 2}, {"step": 1}]
```

Snapshots live in `path.history/` next to the main file.

## What it does NOT do

- No `pickle`. JSON only. If your state has non-JSON objects (numpy,
  tensors, custom classes), serialize them yourself before `.save()`.
- No background thread or autosave. You call `.save()` when your agent
  has reached a checkpoint-worthy moment.
- No multi-process locking. One checkpoint file per agent run.
- No remote storage. The file lives on the local disk. Wrap it with
  rsync or an S3 upload if you want offsite copies.

## Siblings

- [`agent-step-log`](https://github.com/MukundaKatta/agent-step-log) for
  an append-only JSONL log of every step alongside this checkpoint.
- [`llm-context-rotate`](https://github.com/MukundaKatta/llm-context-rotate)
  for an in-memory rolling message window that you would persist into
  this checkpoint.

## License

MIT
