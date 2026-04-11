"""
FLUX-A2A Protocol Specification — Agent-First-Class Language Primitives.

This module implements the six core A2A primitives defined in the Signal
Protocol design research (Rounds 1-3):

  1. Branch  — parallel exploration with configurable merge
  2. Fork    — agent inheritance with fine-grained state control
  3. CoIterate — multi-agent shared program traversal
  4. Discuss — structured agent discourse (debate, brainstorm, review, negotiate)
  5. Synthesize — result combination (map_reduce, ensemble, chain, vote)
  6. Reflect — meta-cognition (self-assessment, strategy adjustment)

Plus the ExecutionMode system (script / compile / meta_compile) from Round 3.

Design principles:
  - All primitives are dataclasses with to_dict() / from_dict() for JSON round-trip.
  - Every primitive carries a confidence score — uncertainty is first-class.
  - Every primitive has a meta dict for extensibility without schema breakage.
  - Schema versioning uses the $schema pattern (AT Protocol Lexicon style).
  - Backward compatibility: unknown fields go into meta, not errors.
  - Forward compatibility: reflect primitive handles unknown concepts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional


# ===========================================================================
# Enums
# ===========================================================================

class BranchStrategy(str, Enum):
    """How branch sub-paths execute."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COMPETITIVE = "competitive"


class MergeStrategyType(str, Enum):
    """How branch/fork/co-iterate results are merged."""
    CONSENSUS = "consensus"
    VOTE = "vote"
    BEST = "best"
    ALL = "all"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    FIRST_COMPLETE = "first_complete"
    LAST_WRITER_WINS = "last_writer_wins"
    CUSTOM = "custom"


class ForkOnComplete(str, Enum):
    """What to do when a fork completes."""
    COLLECT = "collect"
    DISCARD = "discard"
    SIGNAL = "signal"
    MERGE = "merge"


class ForkConflictMode(str, Enum):
    """How to handle merge conflicts between fork and parent."""
    PARENT_WINS = "parent_wins"
    CHILD_WINS = "child_wins"
    NEGOTIATE = "negotiate"


class SharedStateMode(str, Enum):
    """How co-iterating agents share state."""
    CONFLICT = "conflict"
    MERGE = "merge"
    PARTITIONED = "partitioned"
    ISOLATED = "isolated"


class ConflictResolutionType(str, Enum):
    """How to resolve co-iteration write conflicts."""
    PRIORITY = "priority"
    VOTE = "vote"
    LAST_WRITER = "last_writer"
    REJECT = "reject"
    BRANCH = "branch"


class CoIterateMergeType(str, Enum):
    """How co-iteration final results are merged."""
    SEQUENTIAL_CONSENSUS = "sequential_consensus"
    PARALLEL_MERGE = "parallel_merge"
    MAJORITY_VOTE = "majority_vote"
    TRUST_WEIGHTED = "trust_weighted"


class ConvergenceMetric(str, Enum):
    """Metric for detecting co-iteration convergence."""
    AGREEMENT = "agreement"
    CONFIDENCE_DELTA = "confidence_delta"
    VALUE_STABILITY = "value_stability"


class DiscussFormat(str, Enum):
    """Format for agent discussions."""
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    REVIEW = "review"
    NEGOTIATE = "negotiate"
    PEER_REVIEW = "peer_review"


class ParticipantStance(str, Enum):
    """A participant's stance in a discussion."""
    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"
    DEVILS_ADVOCATE = "devil's_advocate"
    MODERATOR = "moderator"


class TurnOrder(str, Enum):
    """How discussion turns are ordered."""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    FREE_FOR_ALL = "free_for_all"
    MODERATED = "moderated"


class DiscussUntilCondition(str, Enum):
    """When a discussion should terminate."""
    CONSENSUS = "consensus"
    TIMEOUT = "timeout"
    ROUNDS = "rounds"
    MAJORITY = "majority"
    BEST_ARGUMENT = "best_argument"


class SynthesisMethod(str, Enum):
    """Method for synthesizing multiple results."""
    MAP_REDUCE = "map_reduce"
    ENSEMBLE = "ensemble"
    CHAIN = "chain"
    VOTE = "vote"
    WEIGHTED_MERGE = "weighted_merge"
    BEST_EFFORT = "best_effort"


class SourceType(str, Enum):
    """Type of a synthesis source."""
    BRANCH_RESULT = "branch_result"
    FORK_RESULT = "fork_result"
    DISCUSS_RESULT = "discuss_result"
    EXTERNAL = "external"
    VARIABLE = "variable"


class SynthesisOutputType(str, Enum):
    """Output type of a synthesis."""
    CODE = "code"
    SPEC = "spec"
    QUESTION = "question"
    DECISION = "decision"
    SUMMARY = "summary"
    VALUE = "value"


class ReflectTarget(str, Enum):
    """What to reflect on."""
    STRATEGY = "strategy"
    PROGRESS = "progress"
    UNCERTAINTY = "uncertainty"
    CONFIDENCE = "confidence"
    ALL = "all"


class AnalysisMethod(str, Enum):
    """How to perform reflection analysis."""
    INTROSPECTION = "introspection"
    BENCHMARK = "benchmark"
    COMPARISON = "comparison"
    STATISTICAL = "statistical"


class ReflectOutputType(str, Enum):
    """What a reflection produces."""
    ADJUSTMENT = "adjustment"
    QUESTION = "question"
    BRANCH = "branch"
    LOG = "log"
    SIGNAL = "signal"


class ExecutionMode(str, Enum):
    """The three execution modes in the script-compile spectrum."""
    SCRIPT = "script"
    COMPILE = "compile"
    META_COMPILE = "meta_compile"


class ConfidenceMode(str, Enum):
    """How confidence propagates through synthesis."""
    PROPAGATE = "propagate"
    MIN = "min"
    MAX = "max"
    AVERAGE = "average"


# ===========================================================================
# Helper
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


# ===========================================================================
# 1. Branch — Parallel Exploration
# ===========================================================================

@dataclass(slots=True)
class BranchBody:
    """One arm of a branch primitive."""
    label: str = ""
    weight: float = 1.0
    agent: Optional[dict[str, Any]] = None
    body: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = _clamp(self.weight)
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"label": self.label, "weight": self.weight, "body": self.body}
        if self.agent:
            d["agent"] = self.agent
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchBody:
        return cls(
            label=data.get("label", ""),
            weight=data.get("weight", 1.0),
            agent=data.get("agent"),
            body=data.get("body", []),
            confidence=data.get("confidence", 1.0),
            meta=data.get("meta", {}),
        )


@dataclass(slots=True)
class MergeConfig:
    """Merge configuration for branch/fork results."""
    strategy: str = MergeStrategyType.WEIGHTED_CONFIDENCE.value
    timeout_ms: int = 30000
    fallback: str = MergeStrategyType.FIRST_COMPLETE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "timeout_ms": self.timeout_ms,
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConfig:
        return cls(**{k: v for k, v in data.items() if k in {"strategy", "timeout_ms", "fallback"}})


@dataclass(slots=True)
class BranchPrimitive:
    """
    Spawn parallel (or sequential/competitive) exploration paths.

    JSON schema:
      {
        "op": "branch",
        "id": "...",
        "strategy": "parallel|sequential|competitive",
        "branches": [...],
        "merge": {...},
        "confidence": 0.8
      }
    """
    id: str = ""
    strategy: str = BranchStrategy.PARALLEL.value
    branches: list[BranchBody] = field(default_factory=list)
    merge: MergeConfig = field(default_factory=MergeConfig)
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.branch/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        self.confidence = _clamp(self.confidence)
        # Auto-convert dict branches to BranchBody
        self.branches = [
            b if isinstance(b, BranchBody) else BranchBody.from_dict(b)
            for b in self.branches
        ]
        # Auto-convert dict merge to MergeConfig
        if not isinstance(self.merge, MergeConfig):
            self.merge = MergeConfig.from_dict(self.merge) if isinstance(self.merge, dict) else MergeConfig()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "branch",
            "$schema": self.schema_version,
            "id": self.id,
            "strategy": self.strategy,
            "branches": [b.to_dict() for b in self.branches],
            "merge": self.merge.to_dict(),
        }
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchPrimitive:
        branches = [BranchBody.from_dict(b) for b in data.get("branches", [])]
        merge = MergeConfig.from_dict(data["merge"]) if "merge" in data else MergeConfig()
        return cls(
            id=data.get("id", ""),
            strategy=data.get("strategy", BranchStrategy.PARALLEL.value),
            branches=branches,
            merge=merge,
            confidence=data.get("confidence", 1.0),
            meta=data.get("meta", {}),
        )

    # Bytecode encoding ---------------------------------------------------

    def to_bytecode(self) -> list[list[Any]]:
        """Encode this branch as a sequence of bytecode instructions."""
        ops: list[list[Any]] = []
        ops.append(["BRANCH_START", self.id, len(self.branches), self.merge.strategy])
        for b in self.branches:
            ops.append(["PUSH_LABEL", b.label])
            ops.append(["PUSH_WEIGHT", b.weight])
            if b.agent:
                ops.append(["PUSH_AGENT", b.agent.get("id", "unknown")])
            # Body expressions would be compiled here in the full compiler
            ops.append(["BRANCH_END_SEGMENT", b.label])
        ops.append(["BRANCH_MERGE", self.merge.strategy, self.merge.timeout_ms])
        return ops


# ===========================================================================
# 2. Fork — Agent Inheritance
# ===========================================================================

@dataclass(slots=True)
class ForkMutation:
    """Describes how a fork differs from its parent."""
    type: str = "strategy"  # prompt | context | strategy | capability
    changes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "changes": self.changes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkMutation:
        return cls(type=data.get("type", "strategy"), changes=data.get("changes", {}))


@dataclass(slots=True)
class ForkInherit:
    """Fine-grained control over what a fork inherits."""
    state: list[str] = field(default_factory=list)
    context: bool = True
    trust_graph: bool = False
    message_history: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "context": self.context,
            "trust_graph": self.trust_graph,
            "message_history": self.message_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkInherit:
        return cls(
            state=data.get("state", []),
            context=data.get("context", True),
            trust_graph=data.get("trust_graph", False),
            message_history=data.get("message_history", False),
        )


@dataclass(slots=True)
class ForkMergePolicy:
    """How fork results merge back to the parent."""
    strategy: str = MergeStrategyType.CONSENSUS.value
    conflict: str = ForkConflictMode.PARENT_WINS.value

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "conflict": self.conflict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkMergePolicy:
        return cls(**{k: v for k, v in data.items() if k in {"strategy", "conflict"}})


@dataclass(slots=True)
class ForkPrimitive:
    """
    Create a child agent with inherited state.

    JSON schema:
      {
        "op": "fork",
        "id": "...",
        "from": "parent_agent_id",
        "mutation": {...},
        "inherit": {...},
        "body": [...],
        "on_complete": "collect|discard|signal|merge",
        "merge_policy": {...}
      }
    """
    id: str = ""
    from_agent: str = "self"
    mutation: ForkMutation = field(default_factory=ForkMutation)
    inherit: ForkInherit = field(default_factory=ForkInherit)
    body: list[dict[str, Any]] = field(default_factory=list)
    on_complete: str = ForkOnComplete.COLLECT.value
    merge_policy: ForkMergePolicy = field(default_factory=ForkMergePolicy)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.fork/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "fork",
            "$schema": self.schema_version,
            "id": self.id,
            "from": self.from_agent,
            "mutation": self.mutation.to_dict(),
            "inherit": self.inherit.to_dict(),
            "body": self.body,
            "on_complete": self.on_complete,
            "merge_policy": self.merge_policy.to_dict(),
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForkPrimitive:
        mutation = ForkMutation.from_dict(data["mutation"]) if "mutation" in data else ForkMutation()
        inherit = ForkInherit.from_dict(data["inherit"]) if "inherit" in data else ForkInherit()
        mp = ForkMergePolicy.from_dict(data["merge_policy"]) if "merge_policy" in data else ForkMergePolicy()
        return cls(
            id=data.get("id", ""),
            from_agent=data.get("from", "self"),
            mutation=mutation,
            inherit=inherit,
            body=data.get("body", []),
            on_complete=data.get("on_complete", ForkOnComplete.COLLECT.value),
            merge_policy=mp,
            meta=data.get("meta", {}),
        )

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        ops.append(["FORK", self.id, self.from_agent, self.mutation.type])
        ops.append(["INHERIT_STATE", len(self.inherit.state)] + self.inherit.state)
        ops.append(["INHERIT_CONTEXT", self.inherit.context])
        ops.append(["INHERIT_TRUST", self.inherit.trust_graph])
        # Body would be compiled here
        ops.append(["FORK_COMPLETE", self.id, self.on_complete, self.merge_policy.strategy])
        return ops


# ===========================================================================
# 3. Co-Iterate — Multi-Agent Shared Program
# ===========================================================================

@dataclass(slots=True)
class CoIterAgent:
    """An agent participating in co-iteration."""
    id: str = ""
    role: str = "modifier"  # modifier | auditor | observer | reviewer
    capabilities: list[str] = field(default_factory=lambda: ["read", "write"])
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "capabilities": self.capabilities,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoIterAgent:
        return cls(
            id=data.get("id", ""),
            role=data.get("role", "modifier"),
            capabilities=data.get("capabilities", ["read", "write"]),
            priority=data.get("priority", 0),
        )


@dataclass(slots=True)
class ConflictResolutionConfig:
    """Configuration for co-iteration conflict resolution."""
    strategy: str = ConflictResolutionType.PRIORITY.value
    priority_order: list[str] = field(default_factory=list)
    on_conflict: str = "notify"  # notify | block | auto_resolve

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"strategy": self.strategy, "on_conflict": self.on_conflict}
        if self.priority_order:
            d["priority_order"] = self.priority_order
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConflictResolutionConfig:
        return cls(
            strategy=data.get("strategy", ConflictResolutionType.PRIORITY.value),
            priority_order=data.get("priority_order", []),
            on_conflict=data.get("on_conflict", "notify"),
        )


@dataclass(slots=True)
class ConvergenceConfig:
    """Configuration for co-iteration convergence detection."""
    metric: str = ConvergenceMetric.AGREEMENT.value
    threshold: float = 0.9
    max_rounds: int = 100

    def __post_init__(self) -> None:
        self.threshold = _clamp(self.threshold, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "max_rounds": self.max_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceConfig:
        return cls(
            metric=data.get("metric", ConvergenceMetric.AGREEMENT.value),
            threshold=data.get("threshold", 0.9),
            max_rounds=data.get("max_rounds", 100),
        )


@dataclass(slots=True)
class CoIteratePrimitive:
    """
    Multiple agents traverse the same program simultaneously.

    JSON schema:
      {
        "op": "co_iterate",
        "id": "...",
        "rounds": 10 | "until_convergence",
        "agents": [...],
        "program": {"body": [...]},
        "shared_state": "conflict|merge|partitioned|isolated",
        "conflict_resolution": {...},
        "merge_strategy": "sequential_consensus|...",
        "convergence": {...}
      }
    """
    id: str = ""
    rounds: Any = "until_convergence"  # int or "until_convergence"
    agents: list[CoIterAgent] = field(default_factory=list)
    program: dict[str, Any] = field(default_factory=dict)
    shared_state: str = SharedStateMode.CONFLICT.value
    conflict_resolution: ConflictResolutionConfig = field(default_factory=ConflictResolutionConfig)
    merge_strategy: str = CoIterateMergeType.SEQUENTIAL_CONSENSUS.value
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.co_iterate/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "co_iterate",
            "$schema": self.schema_version,
            "id": self.id,
            "rounds": self.rounds,
            "agents": [a.to_dict() for a in self.agents],
            "program": self.program,
            "shared_state": self.shared_state,
            "conflict_resolution": self.conflict_resolution.to_dict(),
            "merge_strategy": self.merge_strategy,
            "convergence": self.convergence.to_dict(),
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoIteratePrimitive:
        agents = [CoIterAgent.from_dict(a) for a in data.get("agents", [])]
        cr = ConflictResolutionConfig.from_dict(data["conflict_resolution"]) if "conflict_resolution" in data else ConflictResolutionConfig()
        conv = ConvergenceConfig.from_dict(data["convergence"]) if "convergence" in data else ConvergenceConfig()
        return cls(
            id=data.get("id", ""),
            rounds=data.get("rounds", "until_convergence"),
            agents=agents,
            program=data.get("program", {}),
            shared_state=data.get("shared_state", SharedStateMode.CONFLICT.value),
            conflict_resolution=cr,
            merge_strategy=data.get("merge_strategy", CoIterateMergeType.SEQUENTIAL_CONSENSUS.value),
            convergence=conv,
            meta=data.get("meta", {}),
        )

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        rounds_val = self.rounds if isinstance(self.rounds, int) else 0
        ops.append(["CO_ITERATE_START", self.id, len(self.agents), self.shared_state])
        for a in self.agents:
            ops.append(["AGENT_CURSOR", a.id, a.role, a.priority, ",".join(a.capabilities)])
        # Program body compiled here
        ops.append(["CO_ITERATE_ROUND", rounds_val])
        ops.append(["CONFLICT_DETECT"])
        ops.append(["CONFLICT_RESOLVE", self.conflict_resolution.strategy])
        ops.append(["CONSENSUS_CHECK", self.convergence.metric, self.convergence.threshold])
        ops.append(["CO_ITERATE_END", self.merge_strategy, rounds_val])
        return ops


# ===========================================================================
# 4. Discuss — Structured Agent Discourse
# ===========================================================================

@dataclass(slots=True)
class DiscussParticipant:
    """A participant in a discussion."""
    id: str = ""
    stance: str = ParticipantStance.NEUTRAL.value
    expertise: list[str] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.weight = max(0.0, float(self.weight))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "stance": self.stance, "weight": self.weight}
        if self.expertise:
            d["expertise"] = self.expertise
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussParticipant:
        return cls(
            id=data.get("id", ""),
            stance=data.get("stance", ParticipantStance.NEUTRAL.value),
            expertise=data.get("expertise", []),
            weight=data.get("weight", 1.0),
        )


@dataclass(slots=True)
class DiscussContext:
    """Background and goals for a discussion."""
    background: Any = None
    constraints: list[str] = field(default_factory=list)
    goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"constraints": self.constraints, "goal": self.goal}
        if self.background is not None:
            d["background"] = self.background
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussContext:
        return cls(
            background=data.get("background"),
            constraints=data.get("constraints", []),
            goal=data.get("goal", ""),
        )


@dataclass(slots=True)
class DiscussStructure:
    """Structural constraints on a discussion."""
    max_rounds: int = 10
    time_per_round_ms: int = 0
    turn_order: str = TurnOrder.MODERATED.value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"max_rounds": self.max_rounds, "turn_order": self.turn_order}
        if self.time_per_round_ms:
            d["time_per_round_ms"] = self.time_per_round_ms
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussStructure:
        return cls(
            max_rounds=data.get("max_rounds", 10),
            time_per_round_ms=data.get("time_per_round_ms", 0),
            turn_order=data.get("turn_order", TurnOrder.MODERATED.value),
        )


@dataclass(slots=True)
class DiscussUntil:
    """Termination conditions for a discussion."""
    condition: str = DiscussUntilCondition.CONSENSUS.value
    consensus_threshold: float = 0.8
    max_rounds: int = 10
    timeout_ms: int = 60000

    def __post_init__(self) -> None:
        self.consensus_threshold = _clamp(self.consensus_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "consensus_threshold": self.consensus_threshold,
            "max_rounds": self.max_rounds,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussUntil:
        return cls(**{k: v for k, v in data.items() if k in {
            "condition", "consensus_threshold", "max_rounds", "timeout_ms"
        }})


@dataclass(slots=True)
class DiscussOutput:
    """What a discussion produces."""
    format: str = "decision"  # decision | options | summary | transcript
    include_reasoning: bool = True
    include_confidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "include_reasoning": self.include_reasoning,
            "include_confidence": self.include_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussOutput:
        return cls(
            format=data.get("format", "decision"),
            include_reasoning=data.get("include_reasoning", True),
            include_confidence=data.get("include_confidence", True),
        )


@dataclass(slots=True)
class DiscussPrimitive:
    """
    Structured agent discourse.

    JSON schema:
      {
        "op": "discuss",
        "id": "...",
        "topic": "...",
        "context": {...},
        "participants": [...],
        "format": "debate|brainstorm|review|negotiate|peer_review",
        "structure": {...},
        "until": {...},
        "output": {...}
      }
    """
    id: str = ""
    topic: str = ""
    context: DiscussContext = field(default_factory=DiscussContext)
    participants: list[DiscussParticipant] = field(default_factory=list)
    format: str = DiscussFormat.DEBATE.value
    structure: DiscussStructure = field(default_factory=DiscussStructure)
    until: DiscussUntil = field(default_factory=DiscussUntil)
    output: DiscussOutput = field(default_factory=DiscussOutput)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.discuss/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "discuss",
            "$schema": self.schema_version,
            "id": self.id,
            "topic": self.topic,
            "context": self.context.to_dict(),
            "participants": [p.to_dict() for p in self.participants],
            "format": self.format,
            "structure": self.structure.to_dict(),
            "until": self.until.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussPrimitive:
        ctx = DiscussContext.from_dict(data["context"]) if "context" in data else DiscussContext()
        parts = [DiscussParticipant.from_dict(p) for p in data.get("participants", [])]
        struct = DiscussStructure.from_dict(data["structure"]) if "structure" in data else DiscussStructure()
        until = DiscussUntil.from_dict(data["until"]) if "until" in data else DiscussUntil()
        out = DiscussOutput.from_dict(data["output"]) if "output" in data else DiscussOutput()
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            context=ctx,
            participants=parts,
            format=data.get("format", DiscussFormat.DEBATE.value),
            structure=struct,
            until=until,
            output=out,
            meta=data.get("meta", {}),
        )

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        ops.append(["DISCUSS_START", self.id, self.format, self.until.condition])
        for p in self.participants:
            ops.append(["PARTICIPANT", p.id, p.stance, p.weight])
        ops.append(["SET_TOPIC", self.topic])
        for r in range(self.until.max_rounds):
            ops.append(["DISCUSS_ROUND", r])
            for p in self.participants:
                ops.append(["TURN", p.id])
                # ARGUMENT would be emitted by the participant agent
            ops.append(["DISCUSS_CHECK", self.until.condition])
        ops.append(["DISCUSS_END", self.output.format])
        return ops


# ===========================================================================
# 5. Synthesize — Result Combination
# ===========================================================================

@dataclass(slots=True)
class SynthesisSource:
    """A source for synthesis."""
    id: str = ""
    type: str = SourceType.VARIABLE.value
    ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "ref": self.ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesisSource:
        return cls(
            id=data.get("id", ""),
            type=data.get("type", SourceType.VARIABLE.value),
            ref=data.get("ref", ""),
        )


@dataclass(slots=True)
class SynthesisConfig:
    """Configuration for a synthesis method."""
    map_fn: Optional[dict[str, Any]] = None
    reduce_fn: Optional[dict[str, Any]] = None
    weights: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.threshold = _clamp(self.threshold)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"threshold": self.threshold}
        if self.map_fn:
            d["map_fn"] = self.map_fn
        if self.reduce_fn:
            d["reduce_fn"] = self.reduce_fn
        if self.weights:
            d["weights"] = self.weights
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesisConfig:
        return cls(
            map_fn=data.get("map_fn"),
            reduce_fn=data.get("reduce_fn"),
            weights=data.get("weights", {}),
            threshold=data.get("threshold", 0.5),
        )


@dataclass(slots=True)
class SynthesisOutput:
    """Output specification for synthesis."""
    type: str = SynthesisOutputType.DECISION.value
    format: str = "json"  # json | text | bytecode | signal_program
    confidence: str = ConfidenceMode.PROPAGATE.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "format": self.format, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesisOutput:
        return cls(**{k: v for k, v in data.items() if k in {"type", "format", "confidence"}})


@dataclass(slots=True)
class SynthesizePrimitive:
    """
    Combine results from multiple sources.

    JSON schema:
      {
        "op": "synthesize",
        "id": "...",
        "sources": [...],
        "method": "map_reduce|ensemble|chain|vote|weighted_merge|best_effort",
        "config": {...},
        "output": {...}
      }
    """
    id: str = ""
    sources: list[SynthesisSource] = field(default_factory=list)
    method: str = SynthesisMethod.WEIGHTED_MERGE.value
    config: SynthesisConfig = field(default_factory=SynthesisConfig)
    output: SynthesisOutput = field(default_factory=SynthesisOutput)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.synthesize/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "synthesize",
            "$schema": self.schema_version,
            "id": self.id,
            "sources": [s.to_dict() for s in self.sources],
            "method": self.method,
            "config": self.config.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesizePrimitive:
        sources = [SynthesisSource.from_dict(s) for s in data.get("sources", [])]
        cfg = SynthesisConfig.from_dict(data["config"]) if "config" in data else SynthesisConfig()
        out = SynthesisOutput.from_dict(data["output"]) if "output" in data else SynthesisOutput()
        return cls(
            id=data.get("id", ""),
            sources=sources,
            method=data.get("method", SynthesisMethod.WEIGHTED_MERGE.value),
            config=cfg,
            output=out,
            meta=data.get("meta", {}),
        )

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        ops.append(["SYNTH_START", self.id, self.method, len(self.sources)])
        for s in self.sources:
            ops.append(["SOURCE_REF", s.id, s.type, s.ref])
        if self.config.map_fn:
            ops.append(["MAP_FN"])  # map body would follow
        if self.config.reduce_fn:
            ops.append(["REDUCE_FN"])  # reduce body would follow
        ops.append(["SYNTH_MERGE", self.method, self.config.to_dict()])
        ops.append(["SYNTH_OUTPUT", self.output.type, self.output.format, self.output.confidence])
        return ops


# ===========================================================================
# 6. Reflect — Meta-Cognition
# ===========================================================================

@dataclass(slots=True)
class ReflectScope:
    """Scope of a reflection."""
    from_step: int = 0
    to_step: int = -1  # -1 means current
    focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from_step": self.from_step, "to_step": self.to_step}
        if self.focus:
            d["focus"] = self.focus
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectScope:
        return cls(
            from_step=data.get("from_step", 0),
            to_step=data.get("to_step", -1),
            focus=data.get("focus", []),
        )


@dataclass(slots=True)
class ReflectAnalysis:
    """How to perform the reflection analysis."""
    method: str = AnalysisMethod.INTROSPECTION.value
    baseline: Any = None
    metrics: list[str] = field(default_factory=lambda: [
        "confidence_trend", "state_change_rate", "loop_depth", "branch_divergence"
    ])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "metrics": self.metrics}
        if self.baseline is not None:
            d["baseline"] = self.baseline
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectAnalysis:
        return cls(
            method=data.get("method", AnalysisMethod.INTROSPECTION.value),
            baseline=data.get("baseline"),
            metrics=data.get("metrics", ["confidence_trend"]),
        )


@dataclass(slots=True)
class ReflectOutput:
    """What a reflection produces and how."""
    type: str = ReflectOutputType.LOG.value
    action: dict[str, Any] = field(default_factory=dict)
    min_confidence: float = 0.6
    report_to: str = ""

    def __post_init__(self) -> None:
        self.min_confidence = _clamp(self.min_confidence)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "action": self.action,
            "min_confidence": self.min_confidence,
        }
        if self.report_to:
            d["report_to"] = self.report_to
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectOutput:
        return cls(
            type=data.get("type", ReflectOutputType.LOG.value),
            action=data.get("action", {}),
            min_confidence=data.get("min_confidence", 0.6),
            report_to=data.get("report_to", ""),
        )


@dataclass(slots=True)
class ReflectPrimitive:
    """
    Agent self-reflection and meta-cognition.

    JSON schema:
      {
        "op": "reflect",
        "id": "...",
        "on": "strategy|progress|uncertainty|confidence|all",
        "scope": {...},
        "analysis": {...},
        "output": {...}
      }
    """
    id: str = ""
    on: str = ReflectTarget.UNCERTAINTY.value
    scope: ReflectScope = field(default_factory=ReflectScope)
    analysis: ReflectAnalysis = field(default_factory=ReflectAnalysis)
    output: ReflectOutput = field(default_factory=ReflectOutput)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "flux.a2a.reflect/v1"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "reflect",
            "$schema": self.schema_version,
            "id": self.id,
            "on": self.on,
            "scope": self.scope.to_dict(),
            "analysis": self.analysis.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectPrimitive:
        scope = ReflectScope.from_dict(data["scope"]) if "scope" in data else ReflectScope()
        analysis = ReflectAnalysis.from_dict(data["analysis"]) if "analysis" in data else ReflectAnalysis()
        out = ReflectOutput.from_dict(data["output"]) if "output" in data else ReflectOutput()
        return cls(
            id=data.get("id", ""),
            on=data.get("on", ReflectTarget.UNCERTAINTY.value),
            scope=scope,
            analysis=analysis,
            output=out,
            meta=data.get("meta", {}),
        )

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        ops.append(["REFLECT", self.id, self.on, self.scope.from_step, self.scope.to_step])
        ops.append(["ANALYZE", self.analysis.method, self.analysis.metrics])
        ops.append(["REFLECT_OUTPUT", self.output.type])
        if self.output.type == ReflectOutputType.ADJUSTMENT.value:
            ops.append(["ADJUST_STRATEGY", self.output.action])
        elif self.output.type == ReflectOutputType.BRANCH.value:
            ops.append(["SPAWN_BRANCH", self.output.action])
        elif self.output.type == ReflectOutputType.QUESTION.value:
            ops.append(["ASK_AGENT", self.output.action.get("ask_agent", ""), ""])
        elif self.output.type == ReflectOutputType.SIGNAL.value:
            ops.append(["SIGNAL_REPORT", self.output.report_to])
        return ops


# ===========================================================================
# Execution Mode System
# ===========================================================================

@dataclass(slots=True)
class ExecutionModeConfig:
    """
    The script-compile-meta_compile spectrum.

    Modes:
      - script:        Interpret JSON directly, no optimization. Full introspection.
      - compile:       Optimize hot paths, type-specialize. Production performance.
      - meta_compile:  Observe patterns, generate specialized interpreter. Self-improvement.
    """
    mode: str = ExecutionMode.SCRIPT.value
    optimizations: list[str] = field(default_factory=lambda: [
        "dead_branch_elim", "cse", "constant_fold"
    ])
    observations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"mode": self.mode}
        if self.mode != ExecutionMode.SCRIPT.value:
            d["optimizations"] = self.optimizations
        if self.mode == ExecutionMode.META_COMPILE.value:
            d["observations"] = self.observations
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionModeConfig:
        return cls(
            mode=data.get("mode", ExecutionMode.SCRIPT.value),
            optimizations=data.get("optimizations", ["dead_branch_elim", "cse", "constant_fold"]),
            observations=data.get("observations", {}),
        )


@dataclass(slots=True)
class ModeTransition:
    """A mode switch within a program body."""
    mode: str = ExecutionMode.SCRIPT.value
    optimizations: list[str] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    body: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": "mode", "value": self.mode, "body": self.body}
        if self.optimizations:
            d["optimizations"] = self.optimizations
        if self.observations:
            d["observations"] = self.observations
        return d

    def to_bytecode(self) -> list[list[Any]]:
        ops: list[list[Any]] = []
        opts = ",".join(self.optimizations) if self.optimizations else ""
        ops.append(["MODE_SWITCH", self.mode, opts])
        return ops


# ===========================================================================
# Protocol Registry — wire primitives into the interpreter
# ===========================================================================

class ProtocolRegistry:
    """
    Registry of all A2A protocol primitives.

    Maps opcode strings to their Python classes, enabling the interpreter
    to dynamically dispatch based on the ``op`` field of an Expression.

    Usage:
        registry = ProtocolRegistry()
        cls = registry.get_class("branch")
        primitive = cls.from_dict(expression_dict)
    """

    def __init__(self) -> None:
        self._classes: dict[str, type] = {
            "branch": BranchPrimitive,
            "fork": ForkPrimitive,
            "co_iterate": CoIteratePrimitive,
            "discuss": DiscussPrimitive,
            "synthesize": SynthesizePrimitive,
            "reflect": ReflectPrimitive,
            "mode": ModeTransition,
        }

    def get_class(self, opcode: str) -> Optional[type]:
        """Get the primitive class for an opcode."""
        return self._classes.get(opcode)

    def register(self, opcode: str, cls: type) -> None:
        """Register a new primitive class."""
        self._classes[opcode] = cls

    def list_opcodes(self) -> list[str]:
        """List all registered opcodes."""
        return sorted(self._classes.keys())

    def is_primitive(self, opcode: str) -> bool:
        """Check if an opcode is a known A2A primitive."""
        return opcode in self._classes

    def parse(self, data: dict[str, Any]):
        """
        Parse a JSON dict into the appropriate primitive class.
        Falls back to returning the raw dict if the opcode is unknown.
        """
        opcode = data.get("op", "")
        cls = self._classes.get(opcode)
        if cls is not None:
            return cls.from_dict(data)
        return data


# ===========================================================================
# New bytecode opcodes introduced by the A2A protocol
# ===========================================================================

NEW_OPCODES: list[str] = [
    # Branch
    "BRANCH_START", "BRANCH_END_SEGMENT", "BRANCH_MERGE",
    # Fork
    "FORK", "INHERIT_STATE", "INHERIT_CONTEXT", "INHERIT_TRUST", "FORK_COMPLETE",
    # Co-iterate
    "CO_ITERATE_START", "AGENT_CURSOR", "CO_ITERATE_ROUND",
    "CONFLICT_DETECT", "CONFLICT_RESOLVE", "CONSENSUS_CHECK", "CO_ITERATE_END",
    # Discuss
    "DISCUSS_START", "PARTICIPANT", "SET_TOPIC",
    "DISCUSS_ROUND", "TURN", "ARGUMENT", "DISCUSS_CHECK", "DISCUSS_END",
    # Synthesize
    "SYNTH_START", "SOURCE_REF", "MAP_FN", "REDUCE_FN", "SYNTH_MERGE", "SYNTH_OUTPUT",
    # Reflect / Meta
    "REFLECT", "ANALYZE", "REFLECT_OUTPUT", "ADJUST_STRATEGY", "SPAWN_BRANCH",
    # Mode
    "MODE_SWITCH",
]

# Complete list of all opcodes in the Signal ISA
ALL_OPCODES: list[str] = [
    # Original interpreter opcodes
    "PUSH", "POP", "DUP", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "MOD", "NEG",
    "EQ", "NEQ", "LT", "LTE", "GT", "GTE",
    "AND", "OR", "NOT",
    "CONCAT", "LENGTH",
    "AT", "COLLECT", "REDUCE",
    "LOAD", "STORE",
    "JUMP", "JUMP_IF", "JUMP_IF_NOT", "LABEL", "CALL", "RET", "HALT",
    "STRUCT",
    "TELL", "ASK", "DELEGATE", "BROADCAST", "SIGNAL", "AWAIT",
    "BRANCH", "FORK", "MERGE", "CO_ITERATE",
    "TRUST", "CONFIDENCE",
    "LANG_TAG", "NOP",
] + NEW_OPCODES
