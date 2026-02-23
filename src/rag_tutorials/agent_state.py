"""State management module for the agent tutorials extension.

Implements checkpointing and time-travel for agent loops:
  - AgentState captures a full snapshot of an agent run at any point.
  - Checkpoint wraps a state snapshot with metadata (id, label, step number).
  - StateManager stores multiple checkpoints and supports rewind.

Key operations:
  save_checkpoint  - snapshot the current state and store it
  load_checkpoint  - retrieve a snapshot by id
  list_checkpoints - enumerate all stored snapshots
  rewind_to        - restore a previous snapshot for inspection or replay
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Full snapshot of an agent's execution state at a single point in time."""

    question: str
    steps: list[dict] = field(default_factory=list)
    status: str = "running"
    current_answer: str = ""


@dataclass
class Checkpoint:
    """Immutable snapshot of an AgentState with metadata for time travel."""

    checkpoint_id: str
    step_number: int
    label: str
    state: AgentState


class StateManager:
    """Manages checkpoints for a single agent run.

    Usage
    -----
    manager = StateManager()
    state = AgentState(question="How many days?")

    # After step 1
    state.steps.append({"thought": "...", "action": "retrieve", "observation": "..."})
    cid = manager.save_checkpoint(state, label="after_step_1")

    # After step 2 (something went wrong)
    state.steps.append({"thought": "...", "action": "retrieve", "observation": "bad"})

    # Rewind to the checkpoint taken after step 1
    state = manager.rewind_to(cid)
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: AgentState, label: str = "") -> str:
        """Snapshot current state and store it.

        Args:
            state: AgentState to snapshot (deep-copied for immutability).
            label: Human-readable label, e.g. 'after_step_2'.

        Returns:
            checkpoint_id string that can be passed to load_checkpoint/rewind_to.
        """
        checkpoint_id = uuid.uuid4().hex[:8]
        snapshot = copy.deepcopy(state)
        self._checkpoints[checkpoint_id] = Checkpoint(
            checkpoint_id=checkpoint_id,
            step_number=len(state.steps),
            label=label or f"step_{len(state.steps)}",
            state=snapshot,
        )
        return checkpoint_id

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_id: str) -> AgentState:
        """Return a deep copy of the stored state for the given checkpoint id.

        Args:
            checkpoint_id: Id returned by save_checkpoint.

        Returns:
            A fresh deep copy of the snapshotted AgentState.

        Raises:
            KeyError: If checkpoint_id is not found.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")
        return copy.deepcopy(self._checkpoints[checkpoint_id].state)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[Checkpoint]:
        """Return all stored checkpoints ordered by step number.

        Returns:
            List of Checkpoint objects sorted by step_number ascending.
        """
        return sorted(self._checkpoints.values(), key=lambda c: c.step_number)

    # ------------------------------------------------------------------
    # Time travel
    # ------------------------------------------------------------------

    def rewind_to(self, checkpoint_id: str) -> AgentState:
        """Restore state to the snapshot identified by checkpoint_id.

        This is an alias for load_checkpoint that communicates the intent
        of replaying or inspecting from a past point.

        Args:
            checkpoint_id: Id of the checkpoint to rewind to.

        Returns:
            A fresh deep copy of the snapshotted AgentState.

        Raises:
            KeyError: If checkpoint_id is not found.
        """
        return self.load_checkpoint(checkpoint_id)
