"""
FLUX-A2A Co-iteration Engine.

Models how multiple agents can simultaneously traverse and modify the same
Signal program — like pair programming, except the "programmers" are AI agents
and the "code" is a Signal JSON program.

Core concepts:
  - SharedProgram: a program that multiple agents can traverse simultaneously
  - AgentCursor: each agent's position in the shared program
  - ConflictResolution: what happens when agents modify the same location
  - MergeStrategy: how to combine divergent execution paths
  - ConsensusModel: when agents agree/disagree on next action
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from flux_a2a.schema import (
    Agent,
    ConfidenceScore,
    Expression,
    Result,
    Message,
    MessageBus,
)


# ---------------------------------------------------------------------------
# Conflict resolution strategies
# ---------------------------------------------------------------------------

class ConflictResolutionStrategy(str, Enum):
    """How to resolve when agents modify the same program location."""
    PRIORITY = "priority"
    MERGE = "merge"
    VOTE = "vote"
    LAST_WRITER = "last_writer"
    REJECT = "reject"
    BRANCH = "branch"


# ---------------------------------------------------------------------------
# Merge strategies for co-iteration paths
# ---------------------------------------------------------------------------

class MergeStrategy(str, Enum):
    """How to combine divergent co-iteration execution paths."""
    SEQUENTIAL_CONSENSUS = "sequential_consensus"
    PARALLEL_MERGE = "parallel_merge"
    MAJORITY_VOTE = "majority_vote"
    TRUST_WEIGHTED = "trust_weighted"


# ---------------------------------------------------------------------------
# Agent cursor — tracks position in the shared program
# ---------------------------------------------------------------------------

@dataclass
class AgentCursor:
    """
    Each agent's position and state within a SharedProgram.

    Attributes:
        agent_id: The agent's identity.
        position: Current index in the shared program's body.
        step_count: Total steps taken by this agent.
        local_modifications: Number of modifications made by this agent.
        suggestions_pending: Outstanding suggestions from other agents.
        confidence_at_position: Confidence of the agent's evaluation at current position.
        permissions: What this agent can do (read, write, suggest, branch).
        role: The agent's role in co-iteration (modifier, auditor, observer).
        evaluation_result: The result of evaluating the current position.
        blocked: Whether the agent is blocked (e.g., waiting for conflict resolution).
    """
    agent_id: str
    position: int = 0
    step_count: int = 0
    local_modifications: int = 0
    suggestions_pending: int = 0
    confidence_at_position: float = 1.0
    permissions: list[str] = field(default_factory=lambda: ["read", "write"])
    role: str = "modifier"
    evaluation_result: Optional[Result] = None
    blocked: bool = False
    priority: int = 0  # Higher = more priority in conflicts
    meta: dict[str, Any] = field(default_factory=dict)

    def advance(self, steps: int = 1) -> None:
        """Move the cursor forward."""
        self.position += steps
        self.step_count += steps

    def can_read(self) -> bool:
        return "read" in self.permissions

    def can_write(self) -> bool:
        return "write" in self.permissions

    def can_suggest(self) -> bool:
        return "suggest" in self.permissions

    def can_branch(self) -> bool:
        return "branch" in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "step_count": self.step_count,
            "local_modifications": self.local_modifications,
            "suggestions_pending": self.suggestions_pending,
            "confidence_at_position": self.confidence_at_position,
            "permissions": self.permissions,
            "role": self.role,
            "blocked": self.blocked,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Conflict event
# ---------------------------------------------------------------------------

@dataclass
class ConflictEvent:
    """A conflict that occurred during co-iteration."""
    id: str = ""
    position: int = 0
    agents: list[str] = field(default_factory=list)
    proposed_values: dict[str, Any] = field(default_factory=dict)
    resolution: str = ""  # The strategy used to resolve
    resolved_value: Any = None
    resolved_by: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Shared program — a program multiple agents traverse simultaneously
# ---------------------------------------------------------------------------

@dataclass
class SharedProgram:
    """
    A Signal program that can be traversed and modified by multiple agents
    simultaneously.

    The body is a list of Expressions.  Each agent has a cursor tracking their
    position.  When agents converge on the same position, conflicts may arise.
    """
    id: str = ""
    body: list[Expression] = field(default_factory=list)
    modifications: list[dict[str, Any]] = field(default_factory=list)
    conflict_log: list[ConflictEvent] = field(default_factory=list)
    _cursors: dict[str, AgentCursor] = field(default_factory=dict, init=False, repr=False)
    _evaluated: dict[int, Result] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def length(self) -> int:
        return len(self.body)

    # -- Cursor management ------------------------------------------------

    def add_cursor(self, agent_id: str, agent: Optional[Agent] = None, priority: int = 0) -> AgentCursor:
        """Register an agent cursor on this shared program."""
        permissions = list(agent.capabilities) if agent else ["read", "write"]
        role = agent.role if agent else "modifier"
        cursor = AgentCursor(
            agent_id=agent_id,
            position=0,
            permissions=permissions,
            role=role,
            priority=priority,
        )
        self._cursors[agent_id] = cursor
        return cursor

    def get_cursor(self, agent_id: str) -> Optional[AgentCursor]:
        """Get an agent's cursor, if registered."""
        return self._cursors.get(agent_id)

    def get_all_cursors(self) -> dict[str, AgentCursor]:
        """Get all registered cursors."""
        return dict(self._cursors)

    def remove_cursor(self, agent_id: str) -> Optional[AgentCursor]:
        """Remove an agent's cursor."""
        return self._cursors.pop(agent_id, None)

    # -- Program access ---------------------------------------------------

    def get_expression(self, index: int) -> Optional[Expression]:
        """Get the expression at a given index."""
        if 0 <= index < len(self.body):
            return self.body[index]
        return None

    def set_expression(self, index: int, expr: Expression) -> None:
        """Replace the expression at a given index (tracked modification)."""
        if 0 <= index < len(self.body):
            old = self.body[index]
            self.body[index] = expr
            self.modifications.append({
                "position": index,
                "old": old.to_dict(),
                "new": expr.to_dict(),
                "action": "replace",
            })

    def insert_expression(self, index: int, expr: Expression) -> None:
        """Insert a new expression at a given index."""
        if 0 <= index <= len(self.body):
            self.body.insert(index, expr)
            self.modifications.append({
                "position": index,
                "expression": expr.to_dict(),
                "action": "insert",
            })
            # Adjust all cursors past the insertion point
            for cursor in self._cursors.values():
                if cursor.position >= index:
                    cursor.position += 1

    def remove_expression(self, index: int) -> Optional[Expression]:
        """Remove the expression at a given index."""
        if 0 <= index < len(self.body):
            removed = self.body.pop(index)
            self.modifications.append({
                "position": index,
                "removed": removed.to_dict(),
                "action": "remove",
            })
            # Adjust all cursors past the removal point
            for cursor in self._cursors.values():
                if cursor.position > index:
                    cursor.position -= 1
                elif cursor.position == index and cursor.position >= len(self.body):
                    cursor.position = max(0, len(self.body) - 1)
            return removed
        return None

    def set_evaluated(self, index: int, result: Result) -> None:
        """Record the evaluation result at a given index."""
        self._evaluated[index] = result

    def get_evaluated(self, index: int) -> Optional[Result]:
        """Get the evaluation result at a given index, if any."""
        return self._evaluated.get(index)

    # -- Conflict detection -----------------------------------------------

    def agents_at_position(self, position: int) -> list[AgentCursor]:
        """Get all agent cursors currently at a given position."""
        return [
            c for c in self._cursors.values()
            if c.position == position and not c.blocked
        ]

    def detect_conflicts(self) -> list[ConflictEvent]:
        """Detect positions where multiple agents' cursors overlap."""
        position_agents: dict[int, list[str]] = {}
        for cursor in self._cursors.values():
            pos = cursor.position
            if pos not in position_agents:
                position_agents[pos] = []
            position_agents[pos].append(cursor.agent_id)

        conflicts: list[ConflictEvent] = []
        for pos, agents in position_agents.items():
            if len(agents) > 1:
                # Check if any agent wants to write
                writers = [
                    a for a in agents
                    if self._cursors[a].can_write()
                ]
                if len(writers) > 1:
                    proposed = {}
                    for a in writers:
                        cursor = self._cursors[a]
                        if cursor.evaluation_result:
                            proposed[a] = cursor.evaluation_result.value
                    conflict = ConflictEvent(
                        position=pos,
                        agents=agents,
                        proposed_values=proposed,
                    )
                    conflicts.append(conflict)
                    self.conflict_log.append(conflict)

        return conflicts

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "body": [e.to_dict() for e in self.body],
            "cursors": {aid: c.to_dict() for aid, c in self._cursors.items()},
            "modifications": self.modifications,
            "conflict_count": len(self.conflict_log),
        }


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------

@dataclass
class ConflictResolver:
    """
    Resolves conflicts when multiple agents converge on the same program
    location during co-iteration.
    """
    strategy: str = ConflictResolutionStrategy.PRIORITY.value
    priority_order: list[str] = field(default_factory=list)

    def resolve(self, conflict: ConflictEvent, shared: SharedProgram) -> Any:
        """Resolve a conflict and return the winning value."""
        strategy = ConflictResolutionStrategy(self.strategy)

        if strategy == ConflictResolutionStrategy.PRIORITY:
            return self._resolve_priority(conflict)
        elif strategy == ConflictResolutionStrategy.LAST_WRITER:
            return self._resolve_last_writer(conflict)
        elif strategy == ConflictResolutionStrategy.VOTE:
            return self._resolve_vote(conflict)
        elif strategy == ConflictResolutionStrategy.MERGE:
            return self._resolve_merge(conflict)
        elif strategy == ConflictResolutionStrategy.REJECT:
            return self._resolve_reject(conflict)
        elif strategy == ConflictResolutionStrategy.BRANCH:
            return self._resolve_branch(conflict)
        else:
            return self._resolve_priority(conflict)

    def _resolve_priority(self, conflict: ConflictEvent) -> Any:
        """Higher-priority agent wins."""
        agents = self.priority_order if self.priority_order else conflict.agents
        for agent_id in agents:
            if agent_id in conflict.proposed_values:
                conflict.resolution = "priority"
                conflict.resolved_value = conflict.proposed_values[agent_id]
                conflict.resolved_by = agent_id
                return conflict.resolved_value
        # Fallback: first agent
        if conflict.proposed_values:
            first = next(iter(conflict.proposed_values))
            conflict.resolution = "priority:default"
            conflict.resolved_value = conflict.proposed_values[first]
            conflict.resolved_by = first
            return conflict.resolved_value
        return None

    def _resolve_last_writer(self, conflict: ConflictEvent) -> Any:
        """Most recent write wins."""
        if conflict.proposed_values:
            last_agent = list(conflict.proposed_values.keys())[-1]
            conflict.resolution = "last_writer"
            conflict.resolved_value = conflict.proposed_values[last_agent]
            conflict.resolved_by = last_agent
            return conflict.resolved_value
        return None

    def _resolve_vote(self, conflict: ConflictEvent) -> Any:
        """Majority vote among agents."""
        if not conflict.proposed_values:
            return None
        values = list(conflict.proposed_values.values())
        counts = Counter(values)
        winner = counts.most_common(1)[0][0]
        conflict.resolution = "vote"
        conflict.resolved_value = winner
        return winner

    def _resolve_merge(self, conflict: ConflictEvent) -> Any:
        """Attempt to merge all proposed values."""
        if not conflict.proposed_values:
            return None
        values = list(conflict.proposed_values.values())
        if len(values) == 1:
            conflict.resolution = "merge:trivial"
            conflict.resolved_value = values[0]
            return values[0]

        # Try to combine: if all are numbers, average; if all are lists, concatenate
        if all(isinstance(v, (int, float)) for v in values):
            avg = sum(values) / len(values)
            conflict.resolution = "merge:average"
            conflict.resolved_value = avg
            return avg
        if all(isinstance(v, list) for v in values):
            merged = []
            for v in values:
                merged.extend(v)
            conflict.resolution = "merge:concat"
            conflict.resolved_value = merged
            return merged
        if all(isinstance(v, dict) for v in values):
            merged = {}
            for v in values:
                merged.update(v)
            conflict.resolution = "merge:dict_update"
            conflict.resolved_value = merged
            return merged

        # Can't merge — fall back to first
        conflict.resolution = "merge:fallback_first"
        conflict.resolved_value = values[0]
        return values[0]

    def _resolve_reject(self, conflict: ConflictEvent) -> None:
        """Block conflicting writes — require explicit resolution."""
        conflict.resolution = "rejected"
        conflict.resolved_value = None
        return None

    def _resolve_branch(self, conflict: ConflictEvent) -> Any:
        """Create a branch for each conflicting agent."""
        conflict.resolution = "branched"
        conflict.resolved_value = conflict.proposed_values
        return conflict.proposed_values


# ---------------------------------------------------------------------------
# Consensus model
# ---------------------------------------------------------------------------

@dataclass
class ConsensusModel:
    """
    Determines when agents agree or disagree on the next action during
    co-iteration.
    """
    threshold: float = 0.8  # Agreement threshold (0.0-1.0)

    def check_agreement(
        self,
        agent_results: dict[str, Result],
    ) -> tuple[bool, float, Any]:
        """
        Check if agents agree on a result.

        Returns:
            (agreed, agreement_level, consensus_value)
        """
        if not agent_results:
            return True, 1.0, None

        values = [r.value for r in agent_results.values()]
        counts = Counter(values)

        # If all values are the same, full agreement
        if len(counts) == 1:
            return True, 1.0, values[0]

        # Calculate agreement level
        most_common_value, most_common_count = counts.most_common(1)[0]
        agreement_level = most_common_count / len(values)

        # Check against threshold
        agreed = agreement_level >= self.threshold

        # Consensus value: use most common if agreed, None if not
        consensus_value = most_common_value if agreed else None

        return agreed, agreement_level, consensus_value

    def check_confidence_agreement(
        self,
        agent_results: dict[str, Result],
    ) -> tuple[bool, float]:
        """Check if agents' confidence levels are within acceptable range."""
        if not agent_results:
            return True, 1.0

        confidences = [r.confidence for r in agent_results.values()]
        avg_conf = sum(confidences) / len(confidences)
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        # Agreement if standard deviation is low
        agreement = max(0.0, 1.0 - std_dev)
        return agreement >= self.threshold, agreement


# ---------------------------------------------------------------------------
# Co-iteration engine — orchestrates multi-agent program traversal
# ---------------------------------------------------------------------------

@dataclass
class CoIterationEngine:
    """
    Orchestrates co-iteration: multiple agents traversing the same
    Signal program simultaneously.
    """
    conflict_resolver: ConflictResolver = field(default_factory=ConflictResolver)
    consensus_model: ConsensusModel = field(default_factory=ConsensusModel)
    message_bus: Optional[MessageBus] = field(default=None, init=False, repr=False)
    _step_log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.message_bus = MessageBus()

    def execute(self, shared: SharedProgram) -> Result:
        """
        Execute a shared program with all registered agent cursors.

        The algorithm:
        1. While any agent has not reached the end:
           a. Detect conflicts at current positions
           b. Resolve any conflicts
           c. Advance all unblocked agents
           d. Evaluate current position for each agent
           e. Log step
        2. Merge final results
        """
        from flux_a2a.interpreter import Interpreter

        cursors = shared.get_all_cursors()
        if not cursors:
            return Result(
                value=None,
                confidence=0.0,
                source="co_iterate",
                error="No agents registered",
            )

        # Create interpreters for each agent
        interpreters: dict[str, Interpreter] = {}
        for agent_id in cursors:
            interp = Interpreter(agent_id=agent_id, message_bus=self.message_bus)
            interpreters[agent_id] = interp

        step = 0
        max_steps = shared.length * len(cursors) * 10  # Safety limit

        while step < max_steps:
            step += 1
            active_agents = [
                cid for cid, cursor in cursors.items()
                if cursor.position < shared.length and not cursor.blocked
            ]

            if not active_agents:
                break

            # Detect conflicts
            conflicts = shared.detect_conflicts()
            for conflict in conflicts:
                resolved = self.conflict_resolver.resolve(conflict, shared)
                # Unblock agents after resolution
                for agent_id in conflict.agents:
                    cursor = cursors.get(agent_id)
                    if cursor:
                        cursor.blocked = False

            # Evaluate current position for each active agent
            agent_results: dict[str, Result] = {}
            for agent_id in active_agents:
                cursor = cursors[agent_id]
                expr = shared.get_expression(cursor.position)
                if expr is None:
                    continue

                interp = interpreters[agent_id]
                result = interp.evaluate(expr)
                cursor.evaluation_result = result
                cursor.confidence_at_position = result.confidence
                shared.set_evaluated(cursor.position, result)
                agent_results[agent_id] = result

            # Check consensus
            if agent_results:
                agreed, agreement_level, consensus_value = self.consensus_model.check_agreement(agent_results)

                self._step_log.append({
                    "step": step,
                    "active_agents": active_agents,
                    "positions": {aid: cursors[aid].position for aid in active_agents},
                    "agreement": agreed,
                    "agreement_level": agreement_level,
                    "conflicts": len(conflicts),
                })

            # Advance agents
            for agent_id in active_agents:
                cursor = cursors[agent_id]
                cursor.advance(1)

        # Gather final results
        final_results: list[Result] = []
        for agent_id, cursor in cursors.items():
            if cursor.evaluation_result:
                final_results.append(cursor.evaluation_result)

        if not final_results:
            return Result(
                value=None,
                confidence=0.0,
                source="co_iterate",
                agent="system",
                meta={"steps": step, "agents": list(cursors.keys())},
            )

        # Merge final results
        avg_confidence = sum(r.confidence for r in final_results) / len(final_results)
        last_result = final_results[-1]

        return Result(
            value=last_result.value,
            confidence=avg_confidence,
            source="co_iterate",
            agent="consensus",
            children=final_results,
            meta={
                "steps": step,
                "agents": list(cursors.keys()),
                "modifications": len(shared.modifications),
                "conflicts": len(shared.conflict_log),
            },
        )

    def get_step_log(self) -> list[dict[str, Any]]:
        """Return the execution step log for debugging."""
        return list(self._step_log)
