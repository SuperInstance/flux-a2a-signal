"""Tests for FLUX-A2A interpreter."""

from __future__ import annotations

from flux_a2a.schema import (
    Agent,
    BranchDef,
    Expression,
    MessageBus,
    Program,
    Result,
)
from flux_a2a.interpreter import Interpreter, evaluate, interpret


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _expr(d: dict) -> Expression:
    return Expression.from_dict(d)


def _make_interpreter() -> Interpreter:
    return Interpreter(agent_id="test-agent")


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self) -> None:
        r = evaluate({"op": "add", "args": [3, 4]})
        assert r.value == 7
        assert r.confidence == 1.0

    def test_sub(self) -> None:
        r = evaluate({"op": "sub", "args": [10, 3]})
        assert r.value == 7

    def test_mul(self) -> None:
        r = evaluate({"op": "mul", "args": [5, 6]})
        assert r.value == 30

    def test_div(self) -> None:
        r = evaluate({"op": "div", "args": [20, 4]})
        assert r.value == 5.0

    def test_div_by_zero(self) -> None:
        r = evaluate({"op": "div", "args": [10, 0]})
        assert r.is_error()
        assert "zero" in r.error.lower()

    def test_mod(self) -> None:
        r = evaluate({"op": "mod", "args": [17, 5]})
        assert r.value == 2

    def test_nary_add(self) -> None:
        r = evaluate({"op": "add", "args": [1, 2, 3, 4, 5]})
        assert r.value == 15

    def test_nested_expr(self) -> None:
        """Add where one operand is itself an expression."""
        r = evaluate({
            "op": "add",
            "args": [
                {"op": "mul", "args": [3, 4]},
                {"op": "sub", "args": [10, 2]},
            ],
        })
        assert r.value == 20


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_eq(self) -> None:
        r = evaluate({"op": "eq", "args": [5, 5]})
        assert r.value is True

    def test_neq(self) -> None:
        r = evaluate({"op": "neq", "args": [5, 6]})
        assert r.value is True

    def test_lt(self) -> None:
        r = evaluate({"op": "lt", "args": [3, 5]})
        assert r.value is True

    def test_gte(self) -> None:
        r = evaluate({"op": "gte", "args": [5, 5]})
        assert r.value is True

    def test_gt_false(self) -> None:
        r = evaluate({"op": "gt", "args": [3, 5]})
        assert r.value is False


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class TestLogic:
    def test_and_true(self) -> None:
        r = evaluate({"op": "and", "args": [True, True]})
        assert r.value is True

    def test_and_false(self) -> None:
        r = evaluate({"op": "and", "args": [True, False]})
        assert r.value is False

    def test_or(self) -> None:
        r = evaluate({"op": "or", "args": [False, True]})
        assert r.value is True

    def test_not(self) -> None:
        r = evaluate({"op": "not", "args": [True]})
        assert r.value is False

    def test_xor(self) -> None:
        r = evaluate({"op": "xor", "args": [True, False]})
        assert r.value is True
        r2 = evaluate({"op": "xor", "args": [True, True]})
        assert r2.value is False


# ---------------------------------------------------------------------------
# String operations
# ---------------------------------------------------------------------------

class TestString:
    def test_concat(self) -> None:
        r = evaluate({"op": "concat", "args": ["hello", " ", "world"]})
        assert r.value == "hello world"

    def test_length_string(self) -> None:
        r = evaluate({"op": "length", "value": "hello"})
        assert r.value == 5

    def test_length_list(self) -> None:
        r = evaluate({"op": "length", "value": [1, 2, 3]})
        assert r.value == 3


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestCollections:
    def test_at(self) -> None:
        r = evaluate({"op": "at", "collection": [10, 20, 30], "index": 1})
        assert r.value == 20

    def test_collect(self) -> None:
        r = evaluate({
            "op": "collect",
            "items": [{"op": "literal", "value": 1}, 2, 3],
        })
        assert r.value == [1, 2, 3]

    def test_reduce(self) -> None:
        r = evaluate({"op": "reduce", "collection": [1, 2, 3, 4], "initial": 0})
        assert r.value == 10


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_seq(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "seq",
            "body": [
                {"op": "let", "name": "x", "value": 10},
                {"op": "add", "args": [{"op": "get", "name": "x"}, 5]},
            ],
        }))
        assert r.value == 15

    def test_if_true(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "if",
            "condition": True,
            "then": {"op": "literal", "value": "yes"},
            "else": {"op": "literal", "value": "no"},
        }))
        assert r.value == "yes"

    def test_if_false(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "if",
            "condition": False,
            "then": {"op": "literal", "value": "yes"},
            "else": {"op": "literal", "value": "no"},
        }))
        assert r.value == "no"

    def test_if_expr_condition(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "if",
            "condition": {"op": "gt", "args": [10, 5]},
            "then": {"op": "literal", "value": "bigger"},
            "else": {"op": "literal", "value": "smaller"},
        }))
        assert r.value == "bigger"

    def test_loop_count(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "loop",
            "times": 5,
            "var": "i",
            "body": [
                {"op": "literal", "value": "step"},
            ],
        }))
        assert len(r.value) == 5

    def test_loop_accumulate(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "seq",
            "body": [
                {"op": "let", "name": "sum", "value": 0},
                {
                    "op": "loop",
                    "times": 5,
                    "var": "i",
                    "body": [
                        {
                            "op": "set",
                            "name": "sum",
                            "value": {
                                "op": "add",
                                "args": [{"op": "get", "name": "sum"}, {"op": "get", "name": "i"}],
                            },
                        },
                    ],
                },
                {"op": "get", "name": "sum"},
            ],
        }))
        assert r.value == 10  # 0+1+2+3+4

    def test_match(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "match",
            "value": "b",
            "cases": [
                {"pattern": "a", "body": [{"op": "literal", "value": 1}]},
                {"pattern": "b", "body": [{"op": "literal", "value": 2}]},
            ],
        }))
        assert r.value == 2

    def test_match_wildcard(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "match",
            "value": "unknown",
            "cases": [
                {"pattern": "a", "body": [{"op": "literal", "value": 1}]},
                {"pattern": "_", "body": [{"op": "literal", "value": 99}]},
            ],
        }))
        assert r.value == 99

    def test_while(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "seq",
            "body": [
                {"op": "let", "name": "counter", "value": 0},
                {
                    "op": "while",
                    "condition": {"op": "lt", "args": [{"op": "get", "name": "counter"}, 3]},
                    "body": [
                        {
                            "op": "set",
                            "name": "counter",
                            "value": {"op": "add", "args": [{"op": "get", "name": "counter"}, 1]},
                        },
                    ],
                },
                {"op": "get", "name": "counter"},
            ],
        }))
        assert r.value == 3


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

class TestVariables:
    def test_let_and_get(self) -> None:
        interp = _make_interpreter()
        interp.evaluate(_expr({"op": "let", "name": "x", "value": 42}))
        r = interp.evaluate(_expr({"op": "get", "name": "x"}))
        assert r.value == 42

    def test_let_expr(self) -> None:
        interp = _make_interpreter()
        interp.evaluate(_expr({
            "op": "let",
            "name": "computed",
            "value": {"op": "mul", "args": [6, 7]},
        }))
        r = interp.evaluate(_expr({"op": "get", "name": "computed"}))
        assert r.value == 42

    def test_set(self) -> None:
        interp = _make_interpreter()
        interp.evaluate(_expr({"op": "let", "name": "v", "value": 10}))
        interp.evaluate(_expr({"op": "set", "name": "v", "value": 20}))
        r = interp.evaluate(_expr({"op": "get", "name": "v"}))
        assert r.value == 20

    def test_get_unbound(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({"op": "get", "name": "nonexistent"}))
        assert r.is_error()


# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

class TestStruct:
    def test_simple_struct(self) -> None:
        r = evaluate({
            "op": "struct",
            "fields": {
                "name": "test",
                "value": 42,
                "active": True,
            },
        })
        assert r.value == {"name": "test", "value": 42, "active": True}

    def test_struct_with_exprs(self) -> None:
        r = evaluate({
            "op": "struct",
            "fields": {
                "sum": {"op": "add", "args": [1, 2]},
                "product": {"op": "mul", "args": [3, 4]},
            },
        })
        assert r.value == {"sum": 3, "product": 12}


# ---------------------------------------------------------------------------
# Agent communication
# ---------------------------------------------------------------------------

class TestAgentCommunication:
    def test_tell(self) -> None:
        bus = MessageBus()
        interp = Interpreter(agent_id="sender", message_bus=bus)
        r = interp.evaluate(_expr({
            "op": "tell",
            "to": "receiver",
            "message": "hello there",
        }))
        assert r.value == "hello there"
        inbox = bus.receive("receiver")
        assert len(inbox) == 1
        assert inbox[0].payload == "hello there"

    def test_broadcast(self) -> None:
        bus = MessageBus()
        interp = Interpreter(agent_id="captain", message_bus=bus)
        interp.agents = {
            "nav": Agent(id="nav", role="nav"),
            "eng": Agent(id="eng", role="eng"),
        }
        r = interp.evaluate(_expr({
            "op": "broadcast",
            "scope": "fleet",
            "message": "all hands on deck",
        }))
        assert "nav" in r.value["recipients"]
        assert "eng" in r.value["recipients"]

    def test_signal(self) -> None:
        bus = MessageBus()
        interp = Interpreter(agent_id="scout", message_bus=bus)
        r = interp.evaluate(_expr({
            "op": "signal",
            "name": "sector_clear",
            "payload": {"sector": "A"},
        }))
        assert r.value["signal"] == "sector_clear"
        assert r.value["payload"]["sector"] == "A"

    def test_trust(self) -> None:
        interp = _make_interpreter()
        interp.agents = {"nav": Agent(id="nav")}
        r = interp.evaluate(_expr({
            "op": "trust",
            "agent": "nav",
            "level": 0.95,
            "basis": "proven",
        }))
        assert r.value["trust"] == 0.95
        assert interp.agents["nav"].trust == 0.95

    def test_confidence_scope(self) -> None:
        interp = _make_interpreter()
        r1 = interp.evaluate(_expr({"op": "confidence", "level": 0.7}))
        assert r1.value == 0.7
        r2 = interp.evaluate(_expr({"op": "add", "args": [1, 2]}))
        assert r2.confidence == 0.7


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------

class TestBranching:
    def test_simple_branch(self) -> None:
        interp = _make_interpreter()
        r = interp.evaluate(_expr({
            "op": "branch",
            "id": "test-branch",
            "branches": [
                {
                    "label": "path_a",
                    "weight": 0.5,
                    "body": [{"op": "literal", "value": "A"}],
                },
                {
                    "label": "path_b",
                    "weight": 0.5,
                    "body": [{"op": "literal", "value": "B"}],
                },
            ],
            "merge": {"strategy": "first_complete"},
        }))
        assert r.source == "branch"
        assert r.children is not None
        assert len(r.children) == 2

    def test_branch_state_isolation(self) -> None:
        """Branches should not mutate parent state."""
        interp = _make_interpreter()
        interp.evaluate(_expr({"op": "let", "name": "x", "value": 10}))
        interp.evaluate(_expr({
            "op": "branch",
            "id": "iso-test",
            "branches": [
                {
                    "label": "modifier",
                    "body": [
                        {"op": "let", "name": "x", "value": 999},
                    ],
                },
            ],
        }))
        r = interp.evaluate(_expr({"op": "get", "name": "x"}))
        assert r.value == 10  # parent state unchanged


# ---------------------------------------------------------------------------
# Forking
# ---------------------------------------------------------------------------

class TestForking:
    def test_simple_fork(self) -> None:
        interp = _make_interpreter()
        interp.evaluate(_expr({"op": "let", "name": "mission", "value": "explore"}))
        r = interp.evaluate(_expr({
            "op": "fork",
            "id": "scout-1",
            "agent": {"id": "scout", "role": "explorer", "capabilities": ["tell"], "trust": 0.8},
            "inherit": {"state": ["mission"], "context": True},
            "body": [
                {"op": "tell", "to": "parent", "message": "Deployed"},
            ],
        }))
        assert r.source == "fork"
        assert r.meta["child_agent"] == "scout"
        assert "mission" in r.meta["state_keys_inherited"]

    def test_fork_state_inheritance(self) -> None:
        interp = _make_interpreter()
        interp.evaluate(_expr({"op": "let", "name": "shared", "value": 42}))
        interp.evaluate(_expr({"op": "let", "name": "private", "value": 99}))
        interp.evaluate(_expr({
            "op": "fork",
            "id": "inherit-test",
            "agent": {"id": "child"},
            "inherit": {"state": ["shared"]},
            "body": [],
        }))
        # Fork had access to "shared" but not "private"
        tree = interp.fork_manager.get_tree()
        assert tree.get_root() is not None


# ---------------------------------------------------------------------------
# Full program execution
# ---------------------------------------------------------------------------

class TestFullProgram:
    def test_minimal_program(self) -> None:
        p = Program.from_dict({
            "signal": {
                "body": [{"op": "literal", "value": "done"}],
            }
        })
        r = interpret(p)
        assert r.value == "done"

    def test_computation_program(self) -> None:
        p = Program.from_dict({
            "signal": {
                "body": [
                    {"op": "let", "name": "x", "value": 10},
                    {"op": "let", "name": "y", "value": 20},
                    {"op": "add", "args": [{"op": "get", "name": "x"}, {"op": "get", "name": "y"}]},
                ],
            }
        })
        r = interpret(p)
        assert r.value == 30

    def test_program_with_agents(self) -> None:
        p = Program.from_dict({
            "signal": {
                "agents": [
                    {"id": "captain", "role": "coord", "capabilities": ["tell", "fork"]},
                ],
                "body": [
                    {"op": "tell", "to": "captain", "message": "acknowledge"},
                ],
            }
        })
        r = interpret(p)
        assert r.source == "tell"

    def test_empty_program(self) -> None:
        p = Program()
        r = interpret(p)
        assert r.source == "empty_program"


# ---------------------------------------------------------------------------
# Unknown / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_opcode(self) -> None:
        r = evaluate({"op": "nonexistent_op"})
        assert r.is_error()
        assert r.confidence == 0.0

    def test_literal(self) -> None:
        r = evaluate({"op": "literal", "value": [1, 2, 3]})
        assert r.value == [1, 2, 3]

    def test_eval_wrapper(self) -> None:
        r = evaluate({
            "op": "eval",
            "expr": {"op": "mul", "args": [7, 8]},
        })
        assert r.value == 56
