"""Tests for agent-state-checkpoint."""
import time
import pytest
from agent_state_checkpoint import StateCheckpoint, Checkpoint, CheckpointMeta, CheckpointNotFound


def test_save_and_load(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {"step": 1, "done": False})
    loaded = cp.load("run-1")
    assert loaded.state["step"] == 1
    assert loaded.state["done"] is False


def test_load_missing(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    with pytest.raises(CheckpointNotFound):
        cp.load("missing")


def test_load_or_none_missing(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    assert cp.load_or_none("missing") is None


def test_exists_true(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {})
    assert cp.exists("run-1") is True


def test_exists_false(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    assert cp.exists("nope") is False


def test_version_increments(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {"step": 1})
    cp.save("run-1", {"step": 2})
    loaded = cp.load("run-1")
    assert loaded.version == 2


def test_created_at_stable(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    c1 = cp.save("run-1", {"step": 1})
    time.sleep(0.01)
    c2 = cp.save("run-1", {"step": 2})
    assert c1.meta.created_at == c2.meta.created_at


def test_updated_at_changes(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    c1 = cp.save("run-1", {"step": 1})
    time.sleep(0.01)
    c2 = cp.save("run-1", {"step": 2})
    assert c2.meta.updated_at >= c1.meta.updated_at


def test_update_merges(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {"step": 1, "name": "test"})
    cp.update("run-1", {"step": 2})
    loaded = cp.load("run-1")
    assert loaded.state["step"] == 2
    assert loaded.state["name"] == "test"


def test_update_missing(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    with pytest.raises(CheckpointNotFound):
        cp.update("nope", {"x": 1})


def test_tags(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {}, tags=["production", "v2"])
    loaded = cp.load("run-1")
    assert "production" in loaded.meta.tags
    assert "v2" in loaded.meta.tags


def test_delete(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {})
    assert cp.delete("run-1") is True
    assert not cp.exists("run-1")


def test_delete_missing(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    assert cp.delete("nope") is False


def test_keys(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("a", {})
    cp.save("b", {})
    keys = cp.keys()
    assert "a" in keys
    assert "b" in keys


def test_clear(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("a", {})
    cp.save("b", {})
    count = cp.clear()
    assert count == 2
    assert len(cp) == 0


def test_len(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("a", {})
    cp.save("b", {})
    assert len(cp) == 2


def test_list_checkpoints(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("a", {"x": 1})
    cp.save("b", {"x": 2})
    all_cps = cp.list_checkpoints()
    assert len(all_cps) == 2


def test_list_checkpoints_by_tag(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("a", {}, tags=["prod"])
    cp.save("b", {}, tags=["dev"])
    prod = cp.list_checkpoints(tag="prod")
    assert len(prod) == 1
    assert prod[0].key == "a"


def test_checkpoint_get_helper(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    cp.save("run-1", {"step": 5})
    c = cp.load("run-1")
    assert c.get("step") == 5
    assert c.get("missing", 99) == 99


def test_checkpoint_not_found_is_key_error(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    exc = CheckpointNotFound("run-99")
    assert isinstance(exc, KeyError)


def test_age_seconds(tmp_path):
    cp = StateCheckpoint(str(tmp_path))
    c = cp.save("run-1", {})
    assert c.meta.age_seconds() >= 0.0


def test_atomic_write_directory_created(tmp_path):
    new_dir = str(tmp_path / "nested" / "dir")
    cp = StateCheckpoint(new_dir, create=True)
    import os
    assert os.path.isdir(new_dir)
