"""Core Checkpoint implementation."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# JSON envelope key names. Centralized so migrations and tests can refer
# to the canonical strings.
_KEY_SCHEMA = "_schema_version"
_KEY_SAVED_AT = "_saved_at"
_KEY_STATE = "state"

# Suffix used for the on-disk temp file. We write `path.tmp` and then
# os.replace it onto `path`, so a crash mid-write leaves the previous
# good `path` intact.
_TMP_SUFFIX = ".tmp"


class MigrationMissingError(Exception):
    """Raised when load() needs to step from version X to X+1 but the
    caller did not provide a `migrations[X]` callable.

    Attributes:
        from_version: the version we tried to migrate from
        to_version: the version we tried to migrate to (always from_version + 1)
    """

    def __init__(self, from_version: int, to_version: int):
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"no migration registered from schema version {from_version} "
            f"to {to_version}"
        )


class CorruptCheckpointError(Exception):
    """Raised when load() finds a file that exists but cannot be parsed as
    a valid checkpoint envelope (bad JSON, missing keys, wrong types).

    Attributes:
        path: the on-disk path that failed to parse
        original_exc: the underlying exception (json.JSONDecodeError,
            KeyError, TypeError, etc.) for debugging
    """

    def __init__(self, path: Path, original_exc: Exception):
        self.path = path
        self.original_exc = original_exc
        super().__init__(f"corrupt checkpoint at {path}: {original_exc}")


@dataclass(frozen=True)
class CheckpointEnvelope:
    """On-disk envelope wrapping the user's state dict.

    The envelope is what actually lives in the JSON file. Callers of
    `Checkpoint.load()` only see the inner `state` dict (already migrated
    to current schema_version); the envelope is exposed for tooling that
    wants to inspect the schema version or save timestamp.
    """

    schema_version: int
    saved_at: str  # ISO 8601, always UTC
    state: dict[str, Any]


# Migration type: a callable that takes a state dict at version N and
# returns a state dict at version N+1. The callable owns the meaning of
# the upgrade; the library only sequences the calls.
Migration = Callable[[dict[str, Any]], dict[str, Any]]


class Checkpoint:
    """JSON checkpoint file with atomic writes, schema versioning, and
    optional rotating snapshots.

    Args:
        path: file path for the checkpoint. `~` is expanded. Parent dirs
            are created on first save.
        schema_version: integer version that the *current* code understands.
            Saved into every envelope and used as the migration target.
        migrations: optional mapping from from-version to a callable that
            upgrades state by one step. `migrations[N]` is called when a
            file with `_schema_version == N` is loaded under a code that
            wants `N+1` (or higher, chained).
        keep_history: if True, every save also drops a timestamped copy
            into `path.history/`. Read back with `.history(n)`.

    The checkpoint is JSON-only by design: portable, inspectable, no
    `pickle` security risk. If your state contains non-JSON values
    (numpy arrays, tensors, custom objects), serialize them yourself.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        schema_version: int = 1,
        migrations: Mapping[int, Migration] | None = None,
        keep_history: bool = False,
    ) -> None:
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        # expanduser handles ~; resolve() is intentionally NOT used here
        # because we want to keep a literal user-provided path (symlinks,
        # relative paths) intact.
        self._path = Path(os.fspath(path)).expanduser()
        self._schema_version = schema_version
        self._migrations: dict[int, Migration] = dict(migrations or {})
        self._keep_history = keep_history
        # History dir is derived from the main path so two checkpoints in
        # the same dir do not collide.
        self._history_dir = self._path.with_suffix(self._path.suffix + ".history")

    # ---- public properties ----

    @property
    def path(self) -> Path:
        """Resolved on-disk path of the main checkpoint file."""
        return self._path

    @property
    def schema_version(self) -> int:
        """Schema version this Checkpoint instance targets on save and
        migrates to on load."""
        return self._schema_version

    # ---- write ----

    def save(self, state: Mapping[str, Any]) -> None:
        """Write `state` atomically to the checkpoint file.

        The save sequence is:
        1. Build the envelope dict.
        2. Write to a temp file in the same directory (so os.replace is
           guaranteed to be atomic on the same filesystem).
        3. fsync the temp file to flush kernel buffers to disk.
        4. os.replace the temp onto the target path.

        If the process crashes between (2) and (4), the previous good
        checkpoint at `path` is untouched. The temp file is left on disk
        and will be overwritten by the next save attempt.
        """
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be a mapping, got {type(state).__name__}")

        envelope = {
            _KEY_SCHEMA: self._schema_version,
            _KEY_SAVED_AT: _utc_now_iso(),
            # Copy to a plain dict so callers cannot mutate it under us
            # mid-write, and so json.dumps does not choke on non-dict
            # mappings.
            _KEY_STATE: dict(state),
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # tempfile.NamedTemporaryFile in the target directory keeps the
        # rename on the same filesystem. delete=False lets us close it,
        # write, fsync, then rename without it being yanked.
        fd, tmp_name = tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=_TMP_SUFFIX,
            dir=str(self._path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(envelope, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            # Best-effort cleanup; do not mask the original error.
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise

        # History snapshot is opt-in and best-effort: failures here must
        # not break the primary save. We copy AFTER the atomic rename so
        # the snapshot reflects the on-disk file exactly.
        if self._keep_history:
            self._write_history_snapshot(envelope)

    # ---- read ----

    def load(self) -> dict[str, Any] | None:
        """Load the saved state dict, migrating to current schema_version.

        Returns None if no checkpoint file exists or the file is empty.
        Raises CorruptCheckpointError on JSON parse failure or envelope
        shape mismatch. Raises MigrationMissingError if the file is at a
        version that needs migrating and no migration is registered for
        some step in the chain. Raises ValueError if the file is at a
        version NEWER than this Checkpoint's schema_version (refusing to
        silently downgrade).
        """
        env = self._read_envelope()
        if env is None:
            return None
        return self._migrate(env.state, env.schema_version)

    def load_envelope(self) -> CheckpointEnvelope | None:
        """Like load() but returns the raw envelope (no migration). Useful
        for tooling that wants to inspect the schema version or save
        timestamp without paying the migration cost."""
        return self._read_envelope()

    def exists(self) -> bool:
        """True if the checkpoint file exists on disk."""
        return self._path.is_file()

    def delete(self) -> None:
        """Remove the checkpoint file. No-op if it does not exist. Does
        NOT remove the history directory; call history_clear() for that."""
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()

    # ---- history ----

    def history(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the most recent N rotating snapshots, newest first.

        Returns an empty list if `keep_history=False` was passed to the
        constructor or no snapshots have been written yet. Snapshots that
        cannot be parsed are silently skipped (a corrupt history entry
        should not block reading the rest).
        """
        if n <= 0:
            return []
        if not self._history_dir.is_dir():
            return []
        # Sort by filename descending; we use ISO timestamps in names so
        # lexicographic order matches chronological order.
        snapshots = sorted(self._history_dir.iterdir(), reverse=True)
        out: list[dict[str, Any]] = []
        for p in snapshots:
            if len(out) >= n:
                break
            if not p.is_file():
                continue
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                state = raw.get(_KEY_STATE)
                if isinstance(state, dict):
                    out.append(state)
            except (OSError, json.JSONDecodeError, TypeError):
                # Skip unparseable snapshots without raising.
                continue
        return out

    def history_clear(self) -> None:
        """Remove the entire history directory (if any)."""
        if self._history_dir.is_dir():
            shutil.rmtree(self._history_dir)

    # ---- internal ----

    def _read_envelope(self) -> CheckpointEnvelope | None:
        if not self._path.is_file():
            return None
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as e:
            raise CorruptCheckpointError(self._path, e) from e
        if not raw.strip():
            # Empty file is treated as "no checkpoint yet" rather than
            # corrupt: it can happen if a process crashed *before* writing
            # any bytes into the temp file in older versions of the
            # library, or if the user truncated the file by hand.
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CorruptCheckpointError(self._path, e) from e
        if not isinstance(data, dict):
            raise CorruptCheckpointError(
                self._path, TypeError(f"envelope must be a JSON object, got {type(data).__name__}")
            )
        try:
            schema = data[_KEY_SCHEMA]
            saved_at = data[_KEY_SAVED_AT]
            state = data[_KEY_STATE]
        except KeyError as e:
            raise CorruptCheckpointError(self._path, e) from e
        if not isinstance(schema, int) or schema < 1:
            raise CorruptCheckpointError(
                self._path, ValueError(f"invalid {_KEY_SCHEMA}: {schema!r}")
            )
        if not isinstance(saved_at, str):
            raise CorruptCheckpointError(
                self._path,
                TypeError(
                    f"{_KEY_SAVED_AT} must be a string, got {type(saved_at).__name__}"
                ),
            )
        if not isinstance(state, dict):
            raise CorruptCheckpointError(
                self._path,
                TypeError(
                    f"{_KEY_STATE} must be a JSON object, got {type(state).__name__}"
                ),
            )
        return CheckpointEnvelope(schema_version=schema, saved_at=saved_at, state=state)

    def _migrate(self, state: dict[str, Any], from_version: int) -> dict[str, Any]:
        """Apply migrations from `from_version` up to `self._schema_version`.

        At each step we look up `migrations[v]` and call it on the current
        state. Missing steps raise MigrationMissingError so the operator
        sees the gap loudly at deploy time rather than silently running
        the agent on a half-upgraded state.
        """
        if from_version > self._schema_version:
            # Refusing to downgrade is intentional: a newer file may
            # contain keys that this code does not know how to interpret,
            # and silently throwing them away would lose user data.
            raise ValueError(
                f"checkpoint schema version {from_version} is newer than this code's "
                f"schema_version {self._schema_version}; refusing to downgrade"
            )
        current = state
        v = from_version
        while v < self._schema_version:
            mig = self._migrations.get(v)
            if mig is None:
                raise MigrationMissingError(v, v + 1)
            current = mig(current)
            if not isinstance(current, dict):
                # Defensive: a buggy migration that returns a non-dict
                # would corrupt the next save. Fail loudly here.
                raise TypeError(
                    f"migration {v} -> {v + 1} returned {type(current).__name__}, "
                    "expected dict"
                )
            v += 1
        return current

    def _write_history_snapshot(self, envelope: dict[str, Any]) -> None:
        """Drop a timestamped copy of the envelope into the history dir.

        Failures are swallowed: history is opt-in and best-effort, and a
        full disk on the history side should not break a working save.
        """
        try:
            self._history_dir.mkdir(parents=True, exist_ok=True)
            # ISO timestamp + microseconds keeps filenames sortable and
            # unique within the same save() call. We also include a small
            # PID-derived suffix so two processes writing the same path
            # do not collide on the snapshot name.
            ts = _safe_timestamp_for_filename()
            snap_path = self._history_dir / f"{ts}-{os.getpid()}.json"
            snap_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # Best-effort: do not let history failures shadow a successful
            # primary save.
            pass


# ---- helpers ----


def _utc_now_iso() -> str:
    """Return the current UTC time as a stable ISO 8601 string.

    Always UTC, always with timezone info, always seconds precision plus
    microseconds so two saves in the same millisecond do not collide on
    the rotating snapshot filename.
    """
    return datetime.now(timezone.utc).isoformat()


def _safe_timestamp_for_filename() -> str:
    """ISO timestamp with characters that are safe across filesystems.

    `:` is fine on POSIX but bad on Windows, so we replace it. We keep
    the rest of the ISO format so files sort chronologically with plain
    string comparison.
    """
    return _utc_now_iso().replace(":", "-")
