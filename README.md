# agent-state-checkpoint

Durable JSON checkpoint for long-running agent jobs. Atomic writes (write-to-temp + rename) prevent corruption on crash.

```python
from agent_state_checkpoint import StateCheckpoint

cp = StateCheckpoint("/tmp/my-agent")

# save
cp.save("run-1", {"step": 3, "items_processed": 42})

# resume
state = cp.load("run-1")
print(state.version)        # 1
print(state.get("step"))    # 3

# incremental update
cp.update("run-1", {"step": 4})

# tag-based listing
cp.save("run-2", {}, tags=["production"])
prod = cp.list_checkpoints(tag="production")
```

Zero dependencies. Each checkpoint versioned with auto-incrementing version number and stable `created_at`.
