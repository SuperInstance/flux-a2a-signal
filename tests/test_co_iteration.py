"""Tests for FLUX-A2A co-iteration engine."""

from __future__ import annotations

from flux_a2a.schema import (
    Agent,
    ConfidenceScore,
    Expression,
    Result,
)
from flux_a2a.co_iteration import (
    AgentCursor,
    CoIterationEngine,
    ConflictEvent,
    ConflictResolver,
    ConflictResolutionStrategy,
    ConsensusModel,
    MergeStrategy,
    SharedProgram,
)


# ---------------------------------------------------------------------------
# AgentCursor
# ---------------------------------------------------------------------------

class TestAgentCursor:
    def test_creation(self) -> None:
        c = AgentCursor(agent_id="writer", position=0)
        assert c.agent_id == "writer"
        assert c.position == 0
        assert c.step_count == 0
        assert not c.blocked

    def test_advance(self) -> None:
        c = AgentCursor(agent_id="test")
        c.advance(3)
        assert c.position == 3
        assert c.step_count == 3

    def test_permissions(self) -> None:
        c = AgentCursor(agent_id="test", permissions=["read", "write"])
        assert c.can_read()
        assert c.can_write()
        assert not c.can_suggest()
        assert not c.can_branch()

    def test_to_dict(self) -> None:
        c = AgentCursor(agent_id="test", position=5, step_count=10)
        d = c.to_dict()
        assert d["agent_id"] == "test"
        assert d["position"] == 5
        assert d["step_count"] == 10


# ---------------------------------------------------------------------------
# SharedProgram
# ---------------------------------------------------------------------------

class TestSharedProgram:
    def _make_program(self) -> SharedProgram:
        return SharedProgram(
            id="test-prog",
            body=[
                Expression.from_dict({"op": "let", "name": "x", "value": 0}),
                Expression.from_dict({"op": "add", "args": [1, 2]}),
                Expression.from_dict({"op": "get", "name": "x"}),
            ],
        )

    def test_length(self) -> None:
        sp = self._make_program()
        assert sp.length == 3

    def test_add_cursor(self) -> None:
        sp = self._make_program()
        c = sp.add_cursor("writer", Agent(id="writer", role="modifier"))
        assert sp.get_cursor("writer") is c
        assert sp.get_cursor("nonexistent") is None

    def test_get_expression(self) -> None:
        sp = self._make_program()
        e = sp.get_expression(1)
        assert e is not None
        assert e.op == "add"

    def test_get_expression_out_of_bounds(self) -> None:
        sp = self._make_program()
        assert sp.get_expression(99) is None

    def test_set_expression(self) -> None:
        sp = self._make_program()
        new_expr = Expression(op="mul", params={"args": [3, 4]})
        sp.set_expression(1, new_expr)
        assert sp.get_expression(1).op == "mul"
        assert len(sp.modifications) == 1

    def test_insert_expression(self) -> None:
        sp = self._make_program()
        new_expr = Expression(op="literal", params={"value": "inserted"})
        sp.insert_expression(1, new_expr)
        assert sp.length == 4
        assert sp.get_expression(1).op == "literal"

    def test_insert_adjusts_cursors(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        cursor = sp.get_cursor("a")
        assert cursor is not None
        cursor.position = 2
        sp.insert_expression(1, Expression(op="nop"))
        assert cursor.position == 3  # adjusted past insertion

    def test_remove_expression(self) -> None:
        sp = self._make_program()
        removed = sp.remove_expression(1)
        assert removed is not None
        assert removed.op == "add"
        assert sp.length == 2

    def test_remove_adjusts_cursors(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        cursor = sp.get_cursor("a")
        assert cursor is not None
        cursor.position = 2
        sp.remove_expression(0)
        assert cursor.position == 1  # adjusted past removal

    def test_agents_at_position(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        sp.add_cursor("b", Agent(id="b"))
        cursor_a = sp.get_cursor("a")
        cursor_b = sp.get_cursor("b")
        assert cursor_a is not None
        assert cursor_b is not None
        cursor_a.position = 1
        cursor_b.position = 1
        agents = sp.agents_at_position(1)
        assert len(agents) == 2

    def test_detect_conflicts_no_conflict(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        sp.add_cursor("b", Agent(id="b"))
        cursor_a = sp.get_cursor("a")
        cursor_b = sp.get_cursor("b")
        assert cursor_a is not None
        assert cursor_b is not None
        cursor_a.position = 0
        cursor_b.position = 2
        conflicts = sp.detect_conflicts()
        assert len(conflicts) == 0

    def test_detect_conflicts_with_overlap(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a", capabilities=["write"]))
        sp.add_cursor("b", Agent(id="b", capabilities=["write"]))
        cursor_a = sp.get_cursor("a")
        cursor_b = sp.get_cursor("b")
        assert cursor_a is not None
        assert cursor_b is not None
        cursor_a.position = 1
        cursor_b.position = 1
        conflicts = sp.detect_conflicts()
        assert len(conflicts) == 1

    def test_set_and_get_evaluated(self) -> None:
        sp = self._make_program()
        r = Result(value=42, confidence=0.9)
        sp.set_evaluated(1, r)
        retrieved = sp.get_evaluated(1)
        assert retrieved is not None
        assert retrieved.value == 42

    def test_remove_cursor(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        removed = sp.remove_cursor("a")
        assert removed is not None
        assert sp.get_cursor("a") is None

    def test_to_dict(self) -> None:
        sp = self._make_program()
        sp.add_cursor("a", Agent(id="a"))
        d = sp.to_dict()
        assert d["id"] == "test-prog"
        assert "body" in d
        assert "cursors" in d


# ---------------------------------------------------------------------------
# ConflictEvent
# ---------------------------------------------------------------------------

class TestConflictEvent:
    def test_auto_ids(self) -> None:
        c = ConflictEvent(position=3, agents=["a", "b"])
        assert c.id
        assert c.timestamp
        assert c.position == 3

    def test_proposed_values(self) -> None:
        c = ConflictEvent(
            position=1,
            agents=["a", "b"],
            proposed_values={"a": 10, "b": 20},
        )
        assert c.proposed_values == {"a": 10, "b": 20}


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------

class TestConflictResolver:
    def _make_conflict(self) -> ConflictEvent:
        return ConflictEvent(
            position=1,
            agents=["writer", "reviewer"],
            proposed_values={"writer": 42, "reviewer": 99},
        )

    def test_priority_writer_wins(self) -> None:
        resolver = ConflictResolver(
            strategy="priority",
            priority_order=["writer", "reviewer"],
        )
        conflict = self._make_conflict()
        result = resolver.resolve(conflict, SharedProgram())
        assert result == 42
        assert conflict.resolved_by == "writer"

    def test_priority_reviewer_missing(self) -> None:
        resolver = ConflictResolver(
            strategy="priority",
            priority_order=["reviewer"],
        )
        conflict = self._make_conflict()
        result = resolver.resolve(conflict, SharedProgram())
        assert result == 99

    def test_last_writer(self) -> None:
        resolver = ConflictResolver(strategy="last_writer")
        conflict = self._make_conflict()
        result = resolver.resolve(conflict, SharedProgram())
        assert result == 99

    def test_vote(self) -> None:
        conflict = ConflictEvent(
            position=1,
            agents=["a", "b", "c"],
            proposed_values={"a": 42, "b": 42, "c": 99},
        )
        resolver = ConflictResolver(strategy="vote")
        result = resolver.resolve(conflict, SharedProgram())
        assert result == 42

    def test_merge_numbers(self) -> None:
        resolver = ConflictResolver(strategy="merge")
        conflict = ConflictEvent(
            position=1,
            agents=["a", "b"],
            proposed_values={"a": 10.0, "b": 20.0},
        )
        result = resolver.resolve(conflict, SharedProgram())
        assert result == 15.0

    def test_merge_lists(self) -> None:
        resolver = ConflictResolver(strategy="merge")
        conflict = ConflictEvent(
            position=1,
            agents=["a", "b"],
            proposed_values={"a": [1, 2], "b": [3, 4]},
        )
        result = resolver.resolve(conflict, SharedProgram())
        assert result == [1, 2, 3, 4]

    def test_reject(self) -> None:
        resolver = ConflictResolver(strategy="reject")
        conflict = self._make_conflict()
        result = resolver.resolve(conflict, SharedProgram())
        assert result is None

    def test_branch(self) -> None:
        resolver = ConflictResolver(strategy="branch")
        conflict = self._make_conflict()
        result = resolver.resolve(conflict, SharedProgram())
        assert result == {"writer": 42, "reviewer": 99}


# ---------------------------------------------------------------------------
# ConsensusModel
# ---------------------------------------------------------------------------

class TestConsensusModel:
    def test_full_agreement(self) -> None:
        model = ConsensusModel(threshold=0.8)
        results = {
            "a": Result(value=42, confidence=0.9),
            "b": Result(value=42, confidence=0.85),
        }
        agreed, level, value = model.check_agreement(results)
        assert agreed is True
        assert level == 1.0
        assert value == 42

    def test_disagreement(self) -> None:
        model = ConsensusModel(threshold=0.8)
        results = {
            "a": Result(value=42, confidence=0.9),
            "b": Result(value=99, confidence=0.85),
            "c": Result(value=42, confidence=0.8),
        }
        agreed, level, value = model.check_agreement(results)
        # 2/3 agree, threshold is 0.8 → agreed (0.67 < 0.8? depends)
        # Actually 2/3 ≈ 0.667 which is < 0.8, so not agreed
        assert agreed is False

    def test_empty(self) -> None:
        model = ConsensusModel()
        agreed, level, value = model.check_agreement({})
        assert agreed is True
        assert level == 1.0

    def test_confidence_agreement(self) -> None:
        model = ConsensusModel(threshold=0.8)
        results = {
            "a": Result(value=1, confidence=0.9),
            "b": Result(value=1, confidence=0.88),
        }
        agreed, level = model.check_confidence_agreement(results)
        assert agreed is True


# ---------------------------------------------------------------------------
# CoIterationEngine
# ---------------------------------------------------------------------------

class TestCoIterationEngine:
    def test_single_agent_execution(self) -> None:
        sp = SharedProgram(
            body=[
                Expression.from_dict({"op": "let", "name": "x", "value": 10}),
                Expression.from_dict({"op": "add", "args": [{"op": "get", "name": "x"}, 5]}),
            ],
        )
        sp.add_cursor("agent-a", Agent(id="agent-a", capabilities=["read", "write"]))

        engine = CoIterationEngine()
        result = engine.execute(sp)

        assert result.source == "co_iterate"
        assert result.value == 15

    def test_multi_agent_execution(self) -> None:
        sp = SharedProgram(
            body=[
                Expression.from_dict({"op": "let", "name": "counter", "value": 0}),
                Expression.from_dict({"op": "literal", "value": "done"}),
            ],
        )
        sp.add_cursor("writer", Agent(id="writer", capabilities=["read", "write"], role="modifier"))
        sp.add_cursor("reader", Agent(id="reader", capabilities=["read", "suggest"], role="auditor"))

        engine = CoIterationEngine()
        result = engine.execute(sp)

        assert result.source == "co_iterate"
        assert result.meta["agents"] == ["writer", "reader"]

    def test_conflict_resolution_during_co_iteration(self) -> None:
        sp = SharedProgram(
            body=[
                Expression.from_dict({"op": "literal", "value": 42}),
            ],
        )
        sp.add_cursor("a", Agent(id="a", capabilities=["read", "write"]))
        sp.add_cursor("b", Agent(id="b", capabilities=["read", "write"]))

        engine = CoIterationEngine()
        engine.conflict_resolver = ConflictResolver(
            strategy="priority",
            priority_order=["a", "b"],
        )
        result = engine.execute(sp)

        # Should complete without error
        assert not result.is_error()

    def test_step_log(self) -> None:
        sp = SharedProgram(
            body=[
                Expression.from_dict({"op": "literal", "value": 1}),
                Expression.from_dict({"op": "literal", "value": 2}),
            ],
        )
        sp.add_cursor("agent", Agent(id="agent"))

        engine = CoIterationEngine()
        engine.execute(sp)
        log = engine.get_step_log()
        assert len(log) > 0

    def test_empty_program(self) -> None:
        sp = SharedProgram(body=[])
        sp.add_cursor("agent", Agent(id="agent"))

        engine = CoIterationEngine()
        result = engine.execute(sp)
        assert result.value is None

    def test_no_agents(self) -> None:
        sp = SharedProgram(
            body=[Expression.from_dict({"op": "literal", "value": 1})],
        )
        engine = CoIterationEngine()
        result = engine.execute(sp)
        assert result.is_error()
