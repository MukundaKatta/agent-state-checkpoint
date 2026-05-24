import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from agent_state_checkpoint import (
    Checkpoint,
    CheckpointEnvelope,
    CorruptCheckpointError,
    MigrationMissingError,
)

# ---------- save + load round-trip ----------


def test_save_then_load_round_trip(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    state = {"step": 12, "messages": [{"role": "user", "content": "hi"}], "cost_usd": 0.42}
    ck.save(state)
    loaded = ck.load()
    assert loaded == state


def test_load_returns_none_when_missing(tmp_path: Path):
    ck = Checkpoint(tmp_path / "absent.json", schema_version=1)
    assert ck.load() is None


def test_load_returns_none_when_file_empty(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text("")
    ck = Checkpoint(p, schema_version=1)
    assert ck.load() is None


def test_load_returns_none_when_file_whitespace(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text("   \n\n  ")
    ck = Checkpoint(p, schema_version=1)
    assert ck.load() is None


def test_exists_reports_presence(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    assert ck.exists() is False
    ck.save({"a": 1})
    assert ck.exists() is True


def test_delete_removes_file(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"a": 1})
    assert ck.exists() is True
    ck.delete()
    assert ck.exists() is False
    # double delete is safe
    ck.delete()


# ---------- atomic writes ----------


def test_atomic_write_leaves_no_tmp_on_success(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"a": 1})
    # After a successful save the directory should contain only `run.json`,
    # not any leftover `*.tmp` artifacts.
    contents = list(tmp_path.iterdir())
    assert [p.name for p in contents] == ["run.json"]


def test_atomic_write_preserves_old_file_on_crash(tmp_path: Path, monkeypatch):
    """Simulate a crash between writing the temp file and renaming it.
    The previous good checkpoint must remain readable."""
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"step": 1})

    # Make os.replace raise to simulate a crash at the rename step.
    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated power loss at rename")

    monkeypatch.setattr("agent_state_checkpoint.checkpoint.os.replace", boom)
    with pytest.raises(OSError, match="simulated power loss"):
        ck.save({"step": 999})

    # Restore so the next save would work; verify the old state survived.
    monkeypatch.setattr("agent_state_checkpoint.checkpoint.os.replace", real_replace)
    assert ck.load() == {"step": 1}
    # The temp file should have been cleaned up by the except branch.
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_save_creates_parent_dirs(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "run.json"
    ck = Checkpoint(deep, schema_version=1)
    ck.save({"step": 1})
    assert deep.is_file()


def test_path_expands_user(tmp_path: Path, monkeypatch):
    """A leading `~` in the path is expanded to the user's home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    ck = Checkpoint("~/run.json", schema_version=1)
    assert ck.path == tmp_path / "run.json"


# ---------- envelope contents ----------


def test_envelope_contains_schema_version_and_timestamp(tmp_path: Path):
    p = tmp_path / "run.json"
    ck = Checkpoint(p, schema_version=2)
    ck.save({"step": 1})
    raw = json.loads(p.read_text())
    assert raw["_schema_version"] == 2
    assert "state" in raw and raw["state"] == {"step": 1}
    # _saved_at must be ISO 8601 with a timezone offset.
    assert "_saved_at" in raw
    parsed = datetime.fromisoformat(raw["_saved_at"])
    assert parsed.tzinfo is not None


def test_load_envelope_exposes_metadata(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"step": 5})
    env = ck.load_envelope()
    assert isinstance(env, CheckpointEnvelope)
    assert env.schema_version == 1
    assert env.state == {"step": 5}
    assert isinstance(env.saved_at, str)


def test_save_rejects_non_mapping(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    with pytest.raises(TypeError):
        ck.save("not a dict")  # type: ignore[arg-type]


def test_constructor_rejects_bad_schema_version(tmp_path: Path):
    with pytest.raises(ValueError):
        Checkpoint(tmp_path / "run.json", schema_version=0)
    with pytest.raises(ValueError):
        Checkpoint(tmp_path / "run.json", schema_version=-1)


# ---------- migrations ----------


def test_migration_one_step(tmp_path: Path):
    p = tmp_path / "run.json"
    # Write a v1 file by hand to mimic an older release.
    p.write_text(
        json.dumps(
            {
                "_schema_version": 1,
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {"messages": [{"role": "user", "content": "hi"}]},
            }
        )
    )

    def v1_to_v2(state):
        state["history"] = state.pop("messages", [])
        return state

    ck = Checkpoint(p, schema_version=2, migrations={1: v1_to_v2})
    loaded = ck.load()
    assert loaded == {"history": [{"role": "user", "content": "hi"}]}


def test_migration_multi_step_chain(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(
            {
                "_schema_version": 1,
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {"messages": [1, 2, 3]},
            }
        )
    )

    def v1_to_v2(state):
        state["history"] = state.pop("messages")
        return state

    def v2_to_v3(state):
        state["cost_usd"] = 0.0
        return state

    ck = Checkpoint(p, schema_version=3, migrations={1: v1_to_v2, 2: v2_to_v3})
    loaded = ck.load()
    assert loaded == {"history": [1, 2, 3], "cost_usd": 0.0}


def test_migration_missing_step_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(
            {
                "_schema_version": 1,
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {"a": 1},
            }
        )
    )

    def v2_to_v3(state):
        return state

    # We registered 2->3 but not 1->2, so loading should raise.
    ck = Checkpoint(p, schema_version=3, migrations={2: v2_to_v3})
    with pytest.raises(MigrationMissingError) as exc:
        ck.load()
    assert exc.value.from_version == 1
    assert exc.value.to_version == 2


def test_migration_when_already_current_is_noop(tmp_path: Path):
    """If the file is already at the target schema_version, no migration
    runs even if migrations are registered."""
    p = tmp_path / "run.json"
    called = []

    def v1_to_v2(state):
        called.append(1)
        return state

    ck = Checkpoint(p, schema_version=2, migrations={1: v1_to_v2})
    ck.save({"step": 1})
    loaded = ck.load()
    assert loaded == {"step": 1}
    assert called == []


def test_newer_file_than_code_refuses_to_downgrade(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(
            {
                "_schema_version": 5,
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {"future_key": "value"},
            }
        )
    )
    ck = Checkpoint(p, schema_version=2)
    with pytest.raises(ValueError, match="newer"):
        ck.load()


def test_migration_returning_non_dict_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(
            {
                "_schema_version": 1,
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {"a": 1},
            }
        )
    )

    def bad_migration(state):
        return [1, 2, 3]  # not a dict

    ck = Checkpoint(p, schema_version=2, migrations={1: bad_migration})
    with pytest.raises(TypeError):
        ck.load()


# ---------- corrupt files ----------


def test_corrupt_json_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text("{this is not json")
    ck = Checkpoint(p, schema_version=1)
    with pytest.raises(CorruptCheckpointError) as exc:
        ck.load()
    assert exc.value.path == p
    assert exc.value.original_exc is not None


def test_corrupt_envelope_missing_keys_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"state": {"a": 1}}))  # no _schema_version
    ck = Checkpoint(p, schema_version=1)
    with pytest.raises(CorruptCheckpointError):
        ck.load()


def test_corrupt_envelope_wrong_type_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps([1, 2, 3]))  # not a dict
    ck = Checkpoint(p, schema_version=1)
    with pytest.raises(CorruptCheckpointError):
        ck.load()


def test_corrupt_envelope_bad_schema_type_raises(tmp_path: Path):
    p = tmp_path / "run.json"
    p.write_text(
        json.dumps(
            {
                "_schema_version": "one",  # should be int
                "_saved_at": "2026-05-24T12:00:00+00:00",
                "state": {},
            }
        )
    )
    ck = Checkpoint(p, schema_version=1)
    with pytest.raises(CorruptCheckpointError):
        ck.load()


# ---------- history ----------


def test_history_disabled_by_default(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"step": 1})
    ck.save({"step": 2})
    # No history dir should have been created.
    assert ck.history(10) == []
    history_dir = tmp_path / "run.json.history"
    assert not history_dir.exists()


def test_history_keeps_rotating_snapshots(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1, keep_history=True)
    ck.save({"step": 1})
    ck.save({"step": 2})
    ck.save({"step": 3})
    snaps = ck.history(10)
    # Newest first.
    assert snaps == [{"step": 3}, {"step": 2}, {"step": 1}]


def test_history_respects_n_limit(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1, keep_history=True)
    for i in range(5):
        ck.save({"step": i})
    snaps = ck.history(2)
    assert len(snaps) == 2
    assert snaps[0] == {"step": 4}
    assert snaps[1] == {"step": 3}


def test_history_n_zero_or_negative_returns_empty(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1, keep_history=True)
    ck.save({"step": 1})
    assert ck.history(0) == []
    assert ck.history(-1) == []


def test_history_skips_corrupt_snapshots(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1, keep_history=True)
    ck.save({"step": 1})
    # Corrupt a snapshot file by hand.
    history_dir = tmp_path / "run.json.history"
    snap_files = list(history_dir.iterdir())
    assert len(snap_files) == 1
    snap_files[0].write_text("not json")
    # Read should not raise; it just returns an empty list.
    assert ck.history(10) == []


def test_history_clear_removes_dir(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1, keep_history=True)
    ck.save({"step": 1})
    history_dir = tmp_path / "run.json.history"
    assert history_dir.is_dir()
    ck.history_clear()
    assert not history_dir.exists()
    # Idempotent.
    ck.history_clear()


# ---------- saved_at format ----------


def test_saved_at_is_iso8601_utc(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"step": 1})
    env = ck.load_envelope()
    assert env is not None
    # Loosely: must parse as ISO 8601 with timezone.
    parsed = datetime.fromisoformat(env.saved_at)
    assert parsed.tzinfo is not None
    # And the offset should be zero (UTC) since we always emit UTC.
    assert parsed.utcoffset().total_seconds() == 0
    # Also sanity-check the textual pattern.
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", env.saved_at)


# ---------- overwriting + path semantics ----------


def test_repeated_saves_overwrite(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    ck.save({"step": 1})
    ck.save({"step": 2})
    ck.save({"step": 3})
    assert ck.load() == {"step": 3}


def test_path_property_returns_resolved_path(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    assert isinstance(ck.path, Path)
    assert ck.path == tmp_path / "run.json"


def test_schema_version_property(tmp_path: Path):
    ck = Checkpoint(tmp_path / "run.json", schema_version=7)
    assert ck.schema_version == 7


def test_state_copy_is_independent(tmp_path: Path):
    """Mutating the dict after save() must not affect what was written."""
    ck = Checkpoint(tmp_path / "run.json", schema_version=1)
    state = {"step": 1, "nested": {"a": 1}}
    ck.save(state)
    # Mutating the top-level dict is the part we explicitly guard against
    # by copying into the envelope.
    state["step"] = 999
    loaded = ck.load()
    assert loaded["step"] == 1
