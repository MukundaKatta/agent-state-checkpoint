"""agent-state-checkpoint - durable JSON checkpoint for long-running AI agents.

Long-running agents need to save and resume state across crashes,
restarts, and sleeps. This library is a small, zero-dependency checkpoint
file with atomic writes, schema versioning, and optional rotating
snapshots. JSON-only for portability and safety.

    from agent_state_checkpoint import Checkpoint

    ck = Checkpoint("~/agent-runs/run-2026-05-24.json", schema_version=1)
    ck.save({"step": 12, "messages": [...], "cost_usd": 0.42})

    # later, after crash or restart:
    state = ck.load()
    if state is not None:
        resume_from(state)

Use the `migrations` keyword to evolve the saved shape across releases.

Siblings: `agent-step-log` (per-step JSONL log), `llm-context-rotate`
(in-memory rolling message window).
"""

from agent_state_checkpoint.checkpoint import (
    Checkpoint,
    CheckpointEnvelope,
    CorruptCheckpointError,
    MigrationMissingError,
)

__version__ = "0.1.0"

__all__ = [
    "Checkpoint",
    "CheckpointEnvelope",
    "CorruptCheckpointError",
    "MigrationMissingError",
    "__version__",
]
