"""Tests for FLUX-A2A schema definitions."""

from __future__ import annotations

import json
import uuid

from flux_a2a.schema import (
    Agent,
    BranchDef,
    ConfidenceScore,
    ConflictResolution,
    CoIterateDef,
    Expression,
    ForkDef,
    LanguageTag,
    MergePolicy,
    Message,
    MessageBus,
    MessageType,
    Program,
    ProgramMeta,
    Result,
    TrustEntry,
)


# ---------------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------------

class TestConfidenceScore:
    def test_creation_clamps_to_range(self) -> None:
        c = ConfidenceScore(1.5)
        assert c.value == 1.0
        c2 = ConfidenceScore(-0.5)
        assert c2.value == 0.0

    def test_combine_min(self) -> None:
        a = ConfidenceScore(0.8)
        b = ConfidenceScore(0.6)
        assert a.combine_min(b).value == 0.6

    def test_combine_weighted(self) -> None:
        a = ConfidenceScore(0.7)
        b = ConfidenceScore(0.3)
        result = a.combine_weighted(b, 0.6, 0.4)
        assert abs(result.value - 0.54) < 1e-9

    def test_combine_geometric(self) -> None:
        a = ConfidenceScore(0.8)
        b = ConfidenceScore(0.5)
        result = a.combine_geometric([b])
        expected = (0.8 * 0.5) ** 0.5
        assert abs(result.value - expected) < 1e-9

    def test_bool_conversion(self) -> None:
        assert bool(ConfidenceScore(0.9)) is True
        assert bool(ConfidenceScore(0.3)) is False


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------

class TestExpression:
    def test_from_dict_minimal(self) -> None:
        e = Expression.from_dict({"op": "add"})
        assert e.op == "add"
        assert e.lang == "flux"
        assert e.confidence == 1.0

    def test_from_dict_full(self) -> None:
        e = Expression.from_dict({
            "op": "mul",
            "args": [3, 4],
            "lang": "zho",
            "confidence": 0.85,
            "meta": {"source": "test"},
        })
        assert e.op == "mul"
        assert e.params["args"] == [3, 4]
        assert e.lang == "zho"
        assert e.confidence == 0.85
        assert e.meta["source"] == "test"

    def test_to_dict_minimal(self) -> None:
        e = Expression(op="literal", params={"value": 42})
        d = e.to_dict()
        assert d == {"op": "literal", "value": 42}

    def test_to_dict_with_lang(self) -> None:
        e = Expression(op="add", lang="deu", params={"args": [1, 2]})
        d = e.to_dict()
        assert d["lang"] == "deu"

    def test_get_has(self) -> None:
        e = Expression(op="test", params={"key": "value", "other": 42})
        assert e.get("key") == "value"
        assert e.has("other") is True
        assert e.has("missing") is False

    def test_confidence_score(self) -> None:
        e = Expression(op="test", confidence=0.75)
        cs = e.confidence_score()
        assert cs.value == 0.75

    def test_confidence_clamps(self) -> None:
        e = Expression(op="test", confidence=2.0)
        assert e.confidence == 1.0


# ---------------------------------------------------------------------------
# BranchDef
# ---------------------------------------------------------------------------

class TestBranchDef:
    def test_from_dict(self) -> None:
        bd = BranchDef.from_dict({
            "label": "fast",
            "weight": 0.6,
            "body": [{"op": "literal", "value": 1}],
        })
        assert bd.label == "fast"
        assert bd.weight == 0.6
        assert len(bd.body) == 1
        assert isinstance(bd.body[0], Expression)

    def test_to_dict(self) -> None:
        bd = BranchDef(label="slow", weight=0.4, body=[])
        d = bd.to_dict()
        assert d["label"] == "slow"
        assert d["weight"] == 0.4
        assert d["body"] == []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def test_creation(self) -> None:
        a = Agent(id="nav", role="navigator", trust=0.9)
        assert a.id == "nav"
        assert a.role == "navigator"
        assert a.trust == 0.9

    def test_trust_clamps(self) -> None:
        a = Agent(id="test", trust=2.0)
        assert a.trust == 1.0

    def test_has_capability(self) -> None:
        a = Agent(id="test", capabilities=["tell", "branch"])
        assert a.has_capability("tell") is True
        assert a.has_capability("fork") is False


# ---------------------------------------------------------------------------
# TrustEntry
# ---------------------------------------------------------------------------

class TestTrustEntry:
    def test_creation(self) -> None:
        t = TrustEntry(level=0.8, basis="proven")
        assert t.level == 0.8
        assert t.basis == "proven"

    def test_decay(self) -> None:
        t = TrustEntry(level=0.9, decay_rate=0.1)
        decayed = t.decayed(3)
        assert abs(decayed.level - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_creation_auto_ids(self) -> None:
        m = Message(from_agent="a", to_agent="b", msg_type="tell", payload="hello")
        assert m.id  # should be auto-generated
        assert m.timestamp  # should be auto-generated
        assert m.from_agent == "a"
        assert m.to_agent == "b"

    def test_from_dict(self) -> None:
        m = Message.from_dict({
            "from": "captain",
            "to": "navigator",
            "type": "ask",
            "payload": "position?",
            "confidence": 0.9,
        })
        assert m.from_agent == "captain"
        assert m.to_agent == "navigator"
        assert m.msg_type == "ask"
        assert m.confidence == 0.9

    def test_to_dict(self) -> None:
        m = Message(from_agent="a", to_agent="b", payload=42)
        d = m.to_dict()
        assert d["from"] == "a"
        assert d["payload"] == 42
        assert "id" in d
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class TestMessageBus:
    def test_send_and_receive(self) -> None:
        bus = MessageBus()
        m = Message(from_agent="a", to_agent="b", payload="hello")
        bus.send(m)
        inbox = bus.receive("b")
        assert len(inbox) == 1
        assert inbox[0].payload == "hello"

    def test_receive_clears_inbox(self) -> None:
        bus = MessageBus()
        bus.send(Message(from_agent="a", to_agent="b", payload="msg1"))
        bus.send(Message(from_agent="a", to_agent="b", payload="msg2"))
        first = bus.receive("b")
        assert len(first) == 2
        second = bus.receive("b")
        assert len(second) == 0

    def test_broadcast(self) -> None:
        bus = MessageBus()
        msg = Message(from_agent="captain", to_agent="", payload="all hands", scope="fleet")
        bus.broadcast(msg, ["nav", "eng", "sci"])
        assert len(bus.peek("nav")) == 1
        assert len(bus.peek("eng")) == 1
        assert len(bus.peek("sci")) == 1

    def test_log(self) -> None:
        bus = MessageBus()
        bus.send(Message(from_agent="a", to_agent="b", payload="x"))
        bus.send(Message(from_agent="a", to_agent="c", payload="y"))
        assert len(bus.log()) == 2

    def test_clear(self) -> None:
        bus = MessageBus()
        bus.send(Message(from_agent="a", to_agent="b", payload="x"))
        bus.clear()
        assert len(bus.log()) == 0
        assert len(bus.peek("b")) == 0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class TestResult:
    def test_creation(self) -> None:
        r = Result(value=42, confidence=0.9, source="test", agent="me")
        assert r.value == 42
        assert r.confidence == 0.9
        assert r.timestamp  # auto-generated

    def test_is_error(self) -> None:
        r_ok = Result(value=1)
        r_err = Result(value=None, error="boom")
        assert r_ok.is_error() is False
        assert r_err.is_error() is True

    def test_confidence_clamps(self) -> None:
        r = Result(value=1, confidence=5.0)
        assert r.confidence == 1.0

    def test_to_dict(self) -> None:
        r = Result(value="hello", confidence=0.8, source="tell", agent="bot")
        d = r.to_dict()
        assert d["value"] == "hello"
        assert d["confidence"] == 0.8


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

class TestProgram:
    def test_from_dict_signal_wrapper(self) -> None:
        p = Program.from_dict({
            "signal": {
                "id": "test-1",
                "version": "0.1.0",
                "body": [
                    {"op": "literal", "value": 42},
                    {"op": "add", "args": [1, 2]},
                ],
            }
        })
        assert p.id == "test-1"
        assert p.version == "0.1.0"
        assert len(p.body) == 2
        assert isinstance(p.body[0], Expression)
        assert p.body[1].op == "add"

    def test_from_dict_flat(self) -> None:
        p = Program.from_dict({
            "body": [{"op": "literal", "value": "hi"}],
        })
        assert len(p.body) == 1

    def test_auto_generates_id(self) -> None:
        p = Program()
        assert p.id
        uuid.UUID(p.id)  # must be valid UUID

    def test_to_dict_and_back(self) -> None:
        p = Program(
            version="0.2.0",
            agents=[Agent(id="nav", role="navigator")],
            body=[Expression(op="literal", params={"value": 1})],
        )
        d = p.to_dict()
        p2 = Program.from_dict(d)
        assert p2.version == "0.2.0"
        assert len(p2.agents) == 1
        assert p2.agents[0].id == "nav"

    def test_from_json(self) -> None:
        json_str = json.dumps({
            "signal": {
                "body": [{"op": "add", "args": [1, 2]}],
            }
        })
        p = Program.from_json(json_str)
        assert len(p.body) == 1
        assert p.body[0].op == "add"

    def test_to_json(self) -> None:
        p = Program(body=[Expression(op="hello")])
        j = p.to_json()
        parsed = json.loads(j)
        assert "signal" in parsed

    def test_with_agents(self) -> None:
        p = Program.from_dict({
            "signal": {
                "agents": [
                    {"id": "captain", "role": "coordinator", "capabilities": ["fork", "merge"], "trust": 1.0},
                    {"id": "nav", "role": "executor", "capabilities": ["tell"], "trust": 0.8},
                ],
                "body": [],
            }
        })
        assert len(p.agents) == 2
        assert p.agents[0].has_capability("fork")
        assert p.agents[1].trust == 0.8

    def test_with_trust_graph(self) -> None:
        p = Program.from_dict({
            "signal": {
                "trust_graph": {
                    "captain": {"nav": {"level": 0.9, "basis": "proven"}},
                },
                "body": [],
            }
        })
        assert "captain" in p.trust_graph


# ---------------------------------------------------------------------------
# LanguageTag
# ---------------------------------------------------------------------------

class TestLanguageTag:
    def test_all_values(self) -> None:
        vals = LanguageTag.values()
        assert "flux" in vals
        assert "zho" in vals
        assert "deu" in vals
        assert "kor" in vals
        assert "san" in vals
        assert "wen" in vals
        assert "lat" in vals

    def test_enum_members(self) -> None:
        assert LanguageTag.FLUX == "flux"
        assert LanguageTag.ZHO == "zho"


# ---------------------------------------------------------------------------
# MergePolicy
# ---------------------------------------------------------------------------

class TestMergePolicy:
    def test_default(self) -> None:
        mp = MergePolicy()
        assert mp.strategy == "weighted_confidence"

    def test_from_dict(self) -> None:
        mp = MergePolicy.from_dict({"strategy": "best_confidence", "timeout_ms": 5000})
        assert mp.strategy == "best_confidence"
        assert mp.timeout_ms == 5000

    def test_to_dict(self) -> None:
        mp = MergePolicy(strategy="consensus")
        d = mp.to_dict()
        assert d["strategy"] == "consensus"


# ---------------------------------------------------------------------------
# ConflictResolution
# ---------------------------------------------------------------------------

class TestConflictResolution:
    def test_default(self) -> None:
        cr = ConflictResolution()
        assert cr.strategy == "priority"

    def test_from_dict(self) -> None:
        cr = ConflictResolution.from_dict({
            "strategy": "vote",
            "priority_order": ["a", "b"],
            "on_conflict": "block",
        })
        assert cr.strategy == "vote"
        assert cr.priority_order == ["a", "b"]
        assert cr.on_conflict == "block"
