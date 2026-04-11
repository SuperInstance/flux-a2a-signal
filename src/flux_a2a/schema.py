"""
FLUX-A2A Schema Definitions.

All data structures are Python dataclasses with JSON serialization/deserialization.
These define the complete type system for the Signal protocol.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Language tags
# ---------------------------------------------------------------------------

class LanguageTag(str, Enum):
    """Supported language tags for compilation paths."""
    FLUX = "flux"
    ZHO = "zho"     # 中文 — Chinese (imperative-ideographic)
    DEU = "deu"     # Deutsch — German (compound-word composition)
    KOR = "kor"     # 한국어 — Korean (agglutinative)
    SAN = "san"     # संस्कृतम् — Sanskrit (inflectional)
    WEN = "wen"     # 文言 — Classical Chinese (concise-semantic)
    LAT = "lat"     # Latina — Latin (synthetic-case)

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


# ---------------------------------------------------------------------------
# Confidence score — a native type
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConfidenceScore:
    """Confidence in [0.0, 1.0]. Every result carries one."""
    value: float

    def __post_init__(self) -> None:
        self.value = max(0.0, min(1.0, float(self.value)))

    # Arithmetic ---------------------------------------------------------
    def combine_min(self, other: ConfidenceScore) -> ConfidenceScore:
        """Propagation: uncertain input makes output uncertain."""
        return ConfidenceScore(min(self.value, other.value))

    def combine_weighted(
        self, other: ConfidenceScore, weight_self: float, weight_other: float
    ) -> ConfidenceScore:
        """Weighted average — used in branch merging."""
        total = weight_self + weight_other
        if total == 0:
            return ConfidenceScore(0.0)
        return ConfidenceScore(
            (weight_self * self.value + weight_other * other.value) / total
        )

    def combine_geometric(self, others: list[ConfidenceScore]) -> ConfidenceScore:
        """Geometric mean — used in co-iteration consensus."""
        scores = [self.value] + [o.value for o in others]
        product = 1.0
        for s in scores:
            if s <= 0:
                return ConfidenceScore(0.0)
            product *= s
        return ConfidenceScore(product ** (1.0 / len(scores)))

    def __bool__(self) -> bool:
        return self.value > 0.5

    def __repr__(self) -> str:
        return f"Confidence({self.value:.2f})"


# ---------------------------------------------------------------------------
# Program metadata
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProgramMeta:
    """Metadata attached to a Signal program."""
    author: str = ""
    created: str = ""
    trust_level: float = 1.0
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Agent identity & capabilities
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Agent:
    """An agent that participates in a Signal program."""
    id: str
    role: str = "executor"
    capabilities: list[str] = field(default_factory=lambda: ["tell", "ask"])
    trust: float = 0.5
    lang: str = "flux"
    endpoint: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trust = max(0.0, min(1.0, float(self.trust)))

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities


# ---------------------------------------------------------------------------
# Trust graph
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrustEntry:
    """A directed trust relationship between two agents."""
    level: float
    basis: str = "unknown"  # proven | collaborative | unproven | authority | unknown
    decay_rate: float = 0.0

    def __post_init__(self) -> None:
        self.level = max(0.0, min(1.0, float(self.level)))

    def decayed(self, steps: int = 1) -> TrustEntry:
        """Return a new TrustEntry after `steps` decay iterations."""
        new_level = max(0.0, self.level - self.decay_rate * steps)
        return TrustEntry(level=new_level, basis=self.basis, decay_rate=self.decay_rate)


# Type alias for the full trust graph
TrustGraph = dict[str, dict[str, TrustEntry]]


# ---------------------------------------------------------------------------
# Expression — the core AST node
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Expression:
    """
    A single AST node.  JSON is the universal AST — there is no parse step.

    Every expression has at minimum an ``op`` (opcode).  All other fields
    are stored in a flexible ``params`` dict to allow any combination of
    operands without coupling the schema to specific opcodes.
    """
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    lang: str = "flux"
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    # Convenience helpers ------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.params

    def confidence_score(self) -> ConfidenceScore:
        return ConfidenceScore(self.confidence)

    # Serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.params:
            d.update(self.params)
        if self.lang != "flux":
            d["lang"] = self.lang
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expression:
        """Parse a JSON dict into an Expression.  ``op`` is mandatory;
        everything else goes into ``params``.

        Note: this does **not** mutate the input dict — safe to call
        multiple times on the same dict (e.g., in loop bodies).
        """
        op = data.get("op", "")
        lang = data.get("lang", "flux")
        confidence = data.get("confidence", 1.0)
        meta = data.get("meta", {})
        reserved = {"op", "lang", "confidence", "meta"}
        params = {k: v for k, v in data.items() if k not in reserved}
        return cls(op=op, params=params, lang=lang, confidence=confidence, meta=meta)


# ---------------------------------------------------------------------------
# Branch definition (inside a ``branch`` expression)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BranchDef:
    """A single branch inside a ``branch`` opcode."""
    label: str
    weight: float = 1.0
    body: list[Expression] = field(default_factory=list)
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(1.0, float(self.weight)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "weight": self.weight,
            "body": [e.to_dict() for e in self.body],
        }
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchDef:
        body_raw = data.pop("body", [])
        body = [Expression.from_dict(e) if isinstance(e, dict) else e for e in body_raw]
        return cls(**data, body=body)


# ---------------------------------------------------------------------------
# Merge policy
# ---------------------------------------------------------------------------

class MergePolicyType(str, Enum):
    """Strategies for merging branch results."""
    LAST_WRITER_WINS = "last_writer_wins"
    CONSENSUS = "consensus"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    FIRST_COMPLETE = "first_complete"
    BEST_CONFIDENCE = "best_confidence"
    VOTE = "vote"
    CUSTOM = "custom"


@dataclass(slots=True)
class MergePolicy:
    """How branches rejoin."""
    strategy: str = MergePolicyType.WEIGHTED_CONFIDENCE.value
    timeout_ms: int = 30000
    fallback: str = MergePolicyType.FIRST_COMPLETE.value

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "timeout_ms": self.timeout_ms, "fallback": self.fallback}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergePolicy:
        return cls(**data)


# ---------------------------------------------------------------------------
# Fork definition
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ForkDef:
    """Configuration for a ``fork`` opcode — spawning a child agent."""
    agent: Agent = field(default_factory=lambda: Agent(id="child"))
    inherit_state: list[str] = field(default_factory=list)
    inherit_context: bool = True
    inherit_trust_graph: bool = False
    body: list[Expression] = field(default_factory=list)
    on_result: str = "collect"  # collect | discard | signal
    on_complete: Optional[MergePolicy] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "agent": asdict(self.agent),
            "inherit": {
                "state": self.inherit_state,
                "context": self.inherit_context,
                "trust_graph": self.inherit_trust_graph,
            },
            "body": [e.to_dict() for e in self.body],
        }
        if self.on_result != "collect":
            d["on_result"] = self.on_result
        if self.on_complete:
            d["on_complete"] = self.on_complete.to_dict()
        return d


# ---------------------------------------------------------------------------
# Co-iteration definition
# ---------------------------------------------------------------------------

class ConflictResolutionStrategy(str, Enum):
    """How to resolve when agents modify the same location."""
    PRIORITY = "priority"
    MERGE = "merge"
    VOTE = "vote"
    LAST_WRITER = "last_writer"
    REJECT = "reject"
    BRANCH = "branch"


@dataclass(slots=True)
class ConflictResolution:
    """Configuration for co-iteration conflict resolution."""
    strategy: str = ConflictResolutionStrategy.PRIORITY.value
    priority_order: list[str] = field(default_factory=list)
    on_conflict: str = "notify"  # notify | block | auto_resolve

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"strategy": self.strategy}
        if self.priority_order:
            d["priority_order"] = self.priority_order
        d["on_conflict"] = self.on_conflict
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConflictResolution:
        return cls(**data)


class MergeStrategy(str, Enum):
    """How to combine divergent co-iteration paths."""
    SEQUENTIAL_CONSENSUS = "sequential_consensus"
    PARALLEL_MERGE = "parallel_merge"
    MAJORITY_VOTE = "majority_vote"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass(slots=True)
class CoIterateDef:
    """Configuration for a ``co_iterate`` opcode."""
    program: list[Expression] = field(default_factory=list)
    agents: list[Agent] = field(default_factory=list)
    conflict_resolution: ConflictResolution = field(default_factory=ConflictResolution)
    merge_strategy: str = MergeStrategy.SEQUENTIAL_CONSENSUS.value
    require_agreement: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": [e.to_dict() for e in self.program],
            "agents": [asdict(a) for a in self.agents],
            "conflict_resolution": self.conflict_resolution.to_dict(),
            "merge_strategy": self.merge_strategy,
            "require_agreement": self.require_agreement,
        }


# ---------------------------------------------------------------------------
# Program — top-level
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """A complete Signal program.  This is the top-level container."""
    id: str = ""
    version: str = "0.1.0"
    meta: ProgramMeta = field(default_factory=ProgramMeta)
    agents: list[Agent] = field(default_factory=list)
    trust_graph: dict[str, dict[str, Any]] = field(default_factory=dict)
    body: list[Expression] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    # Serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "signal": {
                "id": self.id,
                "version": self.version,
                "meta": asdict(self.meta),
                "body": [e.to_dict() for e in self.body],
            }
        }
        if self.agents:
            d["signal"]["agents"] = [asdict(a) for a in self.agents]
        if self.trust_graph:
            d["signal"]["trust_graph"] = self.trust_graph
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
        """Parse a JSON dict into a Program.  Accepts both
        ``{signal: {...}}`` and flat ``{...}`` forms."""
        inner = data.get("signal", data)
        meta_data = inner.get("meta", {})
        meta = ProgramMeta(**meta_data) if isinstance(meta_data, dict) else ProgramMeta()

        agents_raw = inner.get("agents", [])
        agents = [Agent(**a) if isinstance(a, dict) else a for a in agents_raw]

        body_raw = inner.get("body", [])
        body = [Expression.from_dict(e) if isinstance(e, dict) else e for e in body_raw]

        return cls(
            id=inner.get("id", ""),
            version=inner.get("version", "0.1.0"),
            meta=meta,
            agents=agents,
            trust_graph=inner.get("trust_graph", {}),
            body=body,
        )

    @classmethod
    def from_json(cls, json_str: str) -> Program:
        """Parse a JSON string into a Program."""
        import json
        return cls.from_dict(json.loads(json_str))

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# A2A Message
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    TELL = "tell"
    ASK = "ask"
    DELEGATE = "delegate"
    BROADCAST = "broadcast"
    SIGNAL = "signal"
    RESPONSE = "response"
    ERROR = "error"


@dataclass(slots=True)
class Message:
    """An A2A communication message between agents."""
    id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    msg_type: str = MessageType.TELL.value
    payload: Any = None
    confidence: float = 1.0
    timestamp: str = ""
    in_reply_to: str = ""
    ttl_ms: int = 0
    scope: str = ""  # for broadcast: "fleet", "group:<name>"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.msg_type,
            "payload": self.payload,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
        if self.in_reply_to:
            d["in_reply_to"] = self.in_reply_to
        if self.ttl_ms:
            d["ttl_ms"] = self.ttl_ms
        if self.scope:
            d["scope"] = self.scope
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            id=data.get("id", ""),
            from_agent=data.get("from", data.get("from_agent", "")),
            to_agent=data.get("to", data.get("to_agent", "")),
            msg_type=data.get("type", MessageType.TELL.value),
            payload=data.get("payload"),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", ""),
            in_reply_to=data.get("in_reply_to", ""),
            ttl_ms=data.get("ttl_ms", 0),
            scope=data.get("scope", ""),
        )


# ---------------------------------------------------------------------------
# In-process message bus
# ---------------------------------------------------------------------------

@dataclass
class MessageBus:
    """Simple in-process message bus for A2A communication within a
    single interpreter session.  Production deployments would replace
    this with a proper message broker."""

    _inbox: dict[str, list[Message]] = field(
        default_factory=dict, init=False, repr=False
    )
    _log: list[Message] = field(default_factory=list, init=False, repr=False)

    def send(self, message: Message) -> None:
        """Route a message to the target agent's inbox."""
        self._log.append(message)
        target = message.to_agent
        if target not in self._inbox:
            self._inbox[target] = []
        self._inbox[target].append(message)

    def broadcast(self, message: Message, agents: list[str]) -> None:
        """Send a message to all listed agents."""
        for agent_id in agents:
            msg = Message(
                from_agent=message.from_agent,
                to_agent=agent_id,
                msg_type=message.msg_type,
                payload=message.payload,
                confidence=message.confidence,
                scope=message.scope,
            )
            self.send(msg)

    def receive(self, agent_id: str) -> list[Message]:
        """Return and clear the inbox for an agent."""
        messages = self._inbox.pop(agent_id, [])
        return messages

    def peek(self, agent_id: str) -> list[Message]:
        """Look at the inbox without clearing it."""
        return list(self._inbox.get(agent_id, []))

    def find_by_reply(self, in_reply_to: str) -> list[Message]:
        """Find all messages that are replies to a given message id."""
        return [m for m in self._log if m.in_reply_to == in_reply_to]

    def log(self) -> list[Message]:
        """Return full message history."""
        return list(self._log)

    def clear(self) -> None:
        """Clear all messages and history."""
        self._inbox.clear()
        self._log.clear()


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Result:
    """The result of evaluating an expression or executing a program."""
    value: Any = None
    confidence: float = 1.0
    source: str = ""
    timestamp: str = ""
    agent: str = ""
    branch: str = ""
    children: list[Result] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_error(self) -> bool:
        return bool(self.error)

    def confidence_score(self) -> ConfidenceScore:
        return ConfidenceScore(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
            "agent": self.agent,
        }
        if self.branch:
            d["branch"] = self.branch
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.meta:
            d["meta"] = self.meta
        if self.error:
            d["error"] = self.error
        return d
