"""
agent-state-checkpoint: Durable JSON checkpoint for long-running agent jobs.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional


class CheckpointNotFound(KeyError):
    pass


@dataclass
class CheckpointMeta:
    key: str
    created_at: float
    updated_at: float
    version: int
    tags: list[str] = field(default_factory=list)

    def age_seconds(self) -> float:
        return time.time() - self.updated_at


@dataclass
class Checkpoint:
    key: str
    state: dict[str, Any]
    meta: CheckpointMeta

    @property
    def version(self) -> int:
        return self.meta.version

    def get(self, field_name: str, default: Any = None) -> Any:
        return self.state.get(field_name, default)


class StateCheckpoint:
    """
    Durable JSON checkpoint store backed by a local directory.

    Writes are atomic (write-to-temp then os.replace) so a crash mid-write
    never corrupts the previous state.
    """

    def __init__(self, directory: str, create: bool = True) -> None:
        self._dir = directory
        if create:
            os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace("..", "_")
        return os.path.join(self._dir, f"{safe}.json")

    def _load_raw(self, key: str) -> Optional[dict[str, Any]]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, key: str, state: dict[str, Any], tags: Optional[list[str]] = None) -> Checkpoint:
        """Save (create or overwrite) a checkpoint atomically."""
        now = time.time()
        existing = self._load_raw(key)
        version = (existing["meta"]["version"] + 1) if existing else 1
        meta = CheckpointMeta(
            key=key,
            created_at=existing["meta"]["created_at"] if existing else now,
            updated_at=now,
            version=version,
            tags=tags if tags is not None else (existing["meta"].get("tags", []) if existing else []),
        )
        payload = {
            "key": key,
            "state": state,
            "meta": {
                "key": meta.key,
                "created_at": meta.created_at,
                "updated_at": meta.updated_at,
                "version": meta.version,
                "tags": meta.tags,
            },
        }
        # atomic write via temp file + rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self._dir, delete=False, suffix=".tmp", encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, self._path(key))
        return Checkpoint(key=key, state=state, meta=meta)

    def update(self, key: str, updates: dict[str, Any]) -> Checkpoint:
        """Merge updates into an existing checkpoint's state."""
        existing = self._load_raw(key)
        if existing is None:
            raise CheckpointNotFound(key)
        state = {**existing["state"], **updates}
        return self.save(key, state, tags=existing["meta"].get("tags"))

    def load(self, key: str) -> Checkpoint:
        """Load a checkpoint; raises CheckpointNotFound if missing."""
        raw = self._load_raw(key)
        if raw is None:
            raise CheckpointNotFound(key)
        m = raw["meta"]
        meta = CheckpointMeta(
            key=m["key"],
            created_at=m["created_at"],
            updated_at=m["updated_at"],
            version=m["version"],
            tags=m.get("tags", []),
        )
        return Checkpoint(key=raw["key"], state=raw["state"], meta=meta)

    def load_or_none(self, key: str) -> Optional[Checkpoint]:
        try:
            return self.load(key)
        except CheckpointNotFound:
            return None

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def keys(self) -> list[str]:
        return [f[:-5] for f in os.listdir(self._dir) if f.endswith(".json")]

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear(self) -> int:
        count = 0
        for k in self.keys():
            self.delete(k)
            count += 1
        return count

    def list_checkpoints(self, tag: Optional[str] = None) -> list[Checkpoint]:
        result = []
        for k in self.keys():
            cp = self.load_or_none(k)
            if cp is None:
                continue
            if tag is not None and tag not in cp.meta.tags:
                continue
            result.append(cp)
        return result

    def __len__(self) -> int:
        return len(self.keys())


__all__ = ["StateCheckpoint", "Checkpoint", "CheckpointMeta", "CheckpointNotFound"]
