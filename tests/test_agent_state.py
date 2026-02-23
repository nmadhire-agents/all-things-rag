"""Tests for agent_state.py — AgentState, Checkpoint, StateManager.

No external calls are made; these tests are fully in-process.
"""
from __future__ import annotations

import pytest

from rag_tutorials.agent_state import AgentState, Checkpoint, StateManager


# ---------------------------------------------------------------------------
# AgentState dataclass
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_defaults(self):
        state = AgentState(question="How many leave days?")
        assert state.steps == []
        assert state.status == "running"
        assert state.current_answer == ""

    def test_mutable_steps(self):
        state = AgentState(question="q")
        state.steps.append({"action": "retrieve"})
        assert len(state.steps) == 1

    def test_custom_status(self):
        state = AgentState(question="q", status="paused")
        assert state.status == "paused"


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_fields(self):
        state = AgentState(question="q")
        cp = Checkpoint(checkpoint_id="abc123", step_number=0, label="start", state=state)
        assert cp.checkpoint_id == "abc123"
        assert cp.step_number == 0
        assert cp.label == "start"
        assert cp.state is state


# ---------------------------------------------------------------------------
# StateManager — save_checkpoint
# ---------------------------------------------------------------------------


class TestStateManagerSave:
    def test_returns_string_id(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state)
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_snapshot_is_independent_copy(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state)
        # Mutate original state after saving
        state.steps.append({"action": "retrieve"})
        # Loaded checkpoint must not reflect the mutation
        loaded = manager.load_checkpoint(cid)
        assert loaded.steps == []

    def test_label_stored(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state, label="after_step_1")
        cp = next(c for c in manager.list_checkpoints() if c.checkpoint_id == cid)
        assert cp.label == "after_step_1"

    def test_default_label_uses_step_count(self):
        manager = StateManager()
        state = AgentState(question="q")
        state.steps = [{"a": 1}, {"b": 2}]
        cid = manager.save_checkpoint(state)
        cp = next(c for c in manager.list_checkpoints() if c.checkpoint_id == cid)
        assert "2" in cp.label


# ---------------------------------------------------------------------------
# StateManager — load_checkpoint
# ---------------------------------------------------------------------------


class TestStateManagerLoad:
    def test_load_returns_correct_state(self):
        manager = StateManager()
        state = AgentState(question="What is the policy?", current_answer="14 days")
        cid = manager.save_checkpoint(state)
        loaded = manager.load_checkpoint(cid)
        assert loaded.question == "What is the policy?"
        assert loaded.current_answer == "14 days"

    def test_load_unknown_id_raises_key_error(self):
        manager = StateManager()
        with pytest.raises(KeyError):
            manager.load_checkpoint("does_not_exist")

    def test_load_returns_deep_copy(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state)
        loaded1 = manager.load_checkpoint(cid)
        loaded2 = manager.load_checkpoint(cid)
        loaded1.steps.append({"x": 1})
        assert loaded2.steps == []


# ---------------------------------------------------------------------------
# StateManager — list_checkpoints
# ---------------------------------------------------------------------------


class TestStateManagerList:
    def test_empty_by_default(self):
        manager = StateManager()
        assert manager.list_checkpoints() == []

    def test_ordered_by_step_number(self):
        manager = StateManager()
        state0 = AgentState(question="q")
        state2 = AgentState(question="q", steps=[{"a": 1}, {"b": 2}])
        state1 = AgentState(question="q", steps=[{"a": 1}])
        manager.save_checkpoint(state0, label="step0")
        manager.save_checkpoint(state2, label="step2")
        manager.save_checkpoint(state1, label="step1")
        labels = [c.label for c in manager.list_checkpoints()]
        assert labels == ["step0", "step1", "step2"]

    def test_returns_checkpoint_objects(self):
        manager = StateManager()
        manager.save_checkpoint(AgentState(question="q"))
        checkpoints = manager.list_checkpoints()
        assert all(isinstance(c, Checkpoint) for c in checkpoints)


# ---------------------------------------------------------------------------
# StateManager — rewind_to (time travel)
# ---------------------------------------------------------------------------


class TestStateManagerRewind:
    def test_rewind_restores_correct_state(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state, label="before_error")
        state.steps.append({"action": "bad_tool", "observation": "wrong"})
        state.current_answer = "incorrect"

        rewound = manager.rewind_to(cid)
        assert rewound.steps == []
        assert rewound.current_answer == ""

    def test_rewind_unknown_id_raises_key_error(self):
        manager = StateManager()
        with pytest.raises(KeyError):
            manager.rewind_to("missing_id")

    def test_rewind_returns_independent_copy(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid = manager.save_checkpoint(state)
        rewound = manager.rewind_to(cid)
        rewound.steps.append({"x": "y"})
        # A second rewind should still give a clean copy
        rewound2 = manager.rewind_to(cid)
        assert rewound2.steps == []

    def test_multiple_checkpoints_at_different_steps(self):
        manager = StateManager()
        state = AgentState(question="q")
        cid0 = manager.save_checkpoint(state, label="start")
        state.steps.append({"action": "retrieve", "observation": "context A"})
        cid1 = manager.save_checkpoint(state, label="after_retrieve")
        state.steps.append({"action": "retrieve", "observation": "context B"})

        rewound0 = manager.rewind_to(cid0)
        assert len(rewound0.steps) == 0

        rewound1 = manager.rewind_to(cid1)
        assert len(rewound1.steps) == 1
        assert rewound1.steps[0]["observation"] == "context A"
