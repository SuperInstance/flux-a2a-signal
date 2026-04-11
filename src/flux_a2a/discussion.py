"""
FLUX-A2A Discussion Protocol — Structured Agent Discourse (Round 4).

The discuss primitive from Round 2 was designed. This module IMPLEMENTS it with
concrete strategies for each discussion format:

  - DebateStrategy:     agents argue opposing positions, converge on truth
  - BrainstormStrategy: agents freely generate ideas, no judgment, then evaluate
  - ReviewStrategy:     systematic critique using checklist
  - NegotiationStrategy: multi-objective optimization with utility functions
  - PeerReviewStrategy: structured evaluation with rubric

Design principles:
  - Each strategy is a concrete implementation of DiscussionStrategy protocol.
  - DiscussionTurn is the universal unit of discourse — a single contribution.
  - Strategies manage the lifecycle: initialization → turn management → conclusion.
  - All data is JSON-serializable via to_dict() / from_dict().
  - Confidence and references are first-class on every turn.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


# ===========================================================================
# Enums
# ===========================================================================

class DiscussionFormat(str, Enum):
    """The five supported discussion formats."""
    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    REVIEW = "review"
    NEGOTIATION = "negotiation"
    PEER_REVIEW = "peer_review"


class Phase(str, Enum):
    """Phase of a multi-phase discussion (e.g. brainstorm has generate + evaluate)."""
    GENERATE = "generate"
    EVALUATE = "evaluate"
    ARGUE = "argue"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    PROPOSE = "propose"
    COUNTER = "counter"
    SINGLE = "single"


class ConvergenceStatus(str, Enum):
    """Status of a discussion's convergence toward resolution."""
    DIVERGING = "diverging"
    STABLE = "stable"
    CONVERGING = "converging"
    CONVERGED = "converged"
    STALEMATE = "stalemate"


class DiscussionOutcome(str, Enum):
    """Possible outcomes of a discussion."""
    CONSENSUS = "consensus"
    MAJORITY = "majority"
    COMPROMISE = "compromise"
    NO_AGREEMENT = "no_agreement"
    DEFERRED = "deferred"


# ===========================================================================
# Helpers
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ===========================================================================
# DiscussionTurn — the universal unit of discourse
# ===========================================================================

@dataclass(slots=True)
class DiscussionTurn:
    """
    A single turn in a discussion.

    Every turn captures who spoke, what they said, how confident they are,
    what they're responding to, and optionally what they're challenging.
    This is the atomic unit of A2A collective intelligence.
    """
    turn_id: str = ""
    agent_id: str = ""
    content: Any = None
    confidence: float = 0.8
    references: list[str] = field(default_factory=list)
    challenge_to: Optional[str] = None
    turn_number: int = 0
    round_number: int = 0
    phase: str = Phase.SINGLE.value
    stance: str = "neutral"
    timestamp: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = str(uuid.uuid4())
        self.confidence = _clamp(self.confidence)
        if not self.timestamp:
            self.timestamp = _now()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "confidence": self.confidence,
            "turn_number": self.turn_number,
            "round_number": self.round_number,
            "phase": self.phase,
            "stance": self.stance,
            "timestamp": self.timestamp,
        }
        if self.references:
            d["references"] = self.references
        if self.challenge_to:
            d["challenge_to"] = self.challenge_to
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussionTurn:
        return cls(
            turn_id=data.get("turn_id", ""),
            agent_id=data.get("agent_id", ""),
            content=data.get("content"),
            confidence=data.get("confidence", 0.8),
            references=data.get("references", []),
            challenge_to=data.get("challenge_to"),
            turn_number=data.get("turn_number", 0),
            round_number=data.get("round_number", 0),
            phase=data.get("phase", Phase.SINGLE.value),
            stance=data.get("stance", "neutral"),
            timestamp=data.get("timestamp", ""),
            meta=data.get("meta", {}),
        )


# ===========================================================================
# Review Rubric and Checklist
# ===========================================================================

@dataclass(slots=True)
class ReviewCriterion:
    """A single criterion for review/peer-review evaluation."""
    name: str = ""
    description: str = ""
    weight: float = 1.0
    score: float = 0.0  # 0.0–1.0, filled during evaluation
    notes: str = ""

    def __post_init__(self) -> None:
        self.weight = max(0.0, float(self.weight))
        self.score = _clamp(self.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "score": self.score,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewCriterion:
        return cls(**{k: v for k, v in data.items() if k in {
            "name", "description", "weight", "score", "notes"
        }})


# Default review checklist used by ReviewStrategy
DEFAULT_REVIEW_CHECKLIST: list[ReviewCriterion] = [
    ReviewCriterion(name="correctness", description="Is the proposal technically correct?", weight=1.5),
    ReviewCriterion(name="completeness", description="Does it cover all required aspects?", weight=1.0),
    ReviewCriterion(name="clarity", description="Is it clear and unambiguous?", weight=0.8),
    ReviewCriterion(name="efficiency", description="Is it efficient in time/space complexity?", weight=1.0),
    ReviewCriterion(name="maintainability", description="Is it easy to maintain and extend?", weight=0.7),
]


# ===========================================================================
# AgentPosition — for consensus detection integration
# ===========================================================================

@dataclass(slots=True)
class AgentPosition:
    """
    An agent's position in multi-dimensional opinion space.

    Dimensions:
      - approach:    how (methodology, algorithm, strategy)
      - goal:        what (outcome, objective, target)
      - priority:    importance weighting of different concerns
      - confidence:  certainty in this position
    """
    agent_id: str = ""
    approach: list[float] = field(default_factory=list)
    goal: list[float] = field(default_factory=list)
    priority: list[float] = field(default_factory=list)
    confidence: float = 0.5
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "agent_id": self.agent_id,
            "approach": self.approach,
            "goal": self.goal,
            "priority": self.priority,
            "confidence": self.confidence,
        }
        if self.label:
            d["label"] = self.label
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentPosition:
        return cls(
            agent_id=data.get("agent_id", ""),
            approach=data.get("approach", []),
            goal=data.get("goal", []),
            priority=data.get("priority", []),
            confidence=data.get("confidence", 0.5),
            label=data.get("label", ""),
            meta=data.get("meta", {}),
        )


# ===========================================================================
# DiscussionResult — what a completed discussion produces
# ===========================================================================

@dataclass(slots=True)
class DiscussionResult:
    """The outcome of a completed discussion."""
    discussion_id: str = ""
    format: str = ""
    outcome: str = DiscussionOutcome.NO_AGREEMENT.value
    decision: Any = None
    confidence: float = 0.0
    turns: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discussion_id": self.discussion_id,
            "format": self.format,
            "outcome": self.outcome,
            "decision": self.decision,
            "confidence": self.confidence,
            "turns": self.turns,
            "positions": self.positions,
            "reasoning": self.reasoning,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussionResult:
        return cls(
            discussion_id=data.get("discussion_id", ""),
            format=data.get("format", ""),
            outcome=data.get("outcome", DiscussionOutcome.NO_AGREEMENT.value),
            decision=data.get("decision"),
            confidence=data.get("confidence", 0.0),
            turns=data.get("turns", []),
            positions=data.get("positions", []),
            reasoning=data.get("reasoning", ""),
            meta=data.get("meta", {}),
        )


# ===========================================================================
# DiscussionStrategy — abstract base for all formats
# ===========================================================================

class DiscussionStrategy(ABC):
    """
    Abstract base class for discussion format strategies.

    Each strategy manages the lifecycle of a structured discussion:
    1. initialize() — set up participants, context, and initial state
    2. next_turn() — determine what the next turn should be
    3. process_turn() — ingest a completed turn and update state
    4. check_completion() — determine if the discussion should end
    5. conclude() — produce the final DiscussionResult
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8):
        self.topic = topic
        self.participants = participants
        self.context = context or {}
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.discussion_id = str(uuid.uuid4())
        self.turns: list[DiscussionTurn] = []
        self.current_round = 0
        self.current_phase = Phase.SINGLE
        self.is_complete = False
        self._positions: dict[str, AgentPosition] = {}
        self._turn_counter = 0

    # -- Abstract interface -------------------------------------------------

    @abstractmethod
    def initialize(self) -> list[DiscussionTurn]:
        """Create initial turns to start the discussion."""
        ...

    @abstractmethod
    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a completed turn and update internal state."""
        ...

    @abstractmethod
    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if discussion is complete. Returns (is_done, outcome)."""
        ...

    @abstractmethod
    def conclude(self) -> DiscussionResult:
        """Produce the final result of the discussion."""
        ...

    # -- Shared helpers -----------------------------------------------------

    def add_turn(self, turn: DiscussionTurn) -> None:
        self._turn_counter += 1
        turn.turn_number = self._turn_counter
        turn.round_number = self.current_round
        turn.phase = self.current_phase.value
        self.turns.append(turn)

    def get_positions(self) -> list[AgentPosition]:
        return list(self._positions.values())

    def get_turn_history(self) -> list[DiscussionTurn]:
        return list(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discussion_id": self.discussion_id,
            "format": self.__class__.__name__.replace("Strategy", "").lower(),
            "topic": self.topic,
            "participants": self.participants,
            "context": self.context,
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "current_round": self.current_round,
            "current_phase": self.current_phase.value,
            "is_complete": self.is_complete,
            "turn_count": len(self.turns),
        }


# ===========================================================================
# DebateStrategy — thesis/antithesis → synthesis
# ===========================================================================

class DebateStrategy(DiscussionStrategy):
    """
    Structured debate: agents argue opposing positions, converge on truth.

    Tracks thesis (pro) and antithesis (con) arguments. Detects synthesis
    opportunities when agents start acknowledging valid points from the
    other side.
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8):
        super().__init__(topic, participants, context, max_rounds, consensus_threshold)
        self.thesis_points: list[str] = []
        self.antithesis_points: list[str] = []
        self.synthesis_candidates: list[str] = []
        self.concession_count: int = 0
        self.current_phase = Phase.ARGUE

    def initialize(self) -> list[DiscussionTurn]:
        """Create opening arguments for pro and con sides."""
        initial: list[DiscussionTurn] = []
        for p in self.participants:
            stance = p.get("stance", "neutral")
            if stance not in ("pro", "con", "neutral", "devil's_advocate"):
                stance = "neutral"
            turn = DiscussionTurn(
                agent_id=p.get("id", "unknown"),
                content={"type": "opening", "position": f"Initial {stance} position on: {self.topic}"},
                confidence=p.get("weight", 1.0),
                stance=stance,
                turn_number=len(initial) + 1,
                round_number=0,
                phase=self.current_phase.value,
            )
            initial.append(turn)
            self.add_turn(turn)

            # Initialize position
            self._positions[p.get("id", "unknown")] = AgentPosition(
                agent_id=p.get("id", "unknown"),
                confidence=0.5,
                label=stance,
            )

        self.current_round = 1
        return initial

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a debate turn, tracking thesis/antithesis/synthesis."""
        self.add_turn(turn)

        # Track argument points by stance and content type
        if isinstance(turn.content, dict):
            ctype = turn.content.get("type", "")
            point = turn.content.get("argument", turn.content.get("point", ""))

            # Check content type FIRST, before stance-based routing
            if ctype == "concession":
                self.concession_count += 1
                if point:
                    self.thesis_points.append(str(point))
            elif ctype == "synthesis":
                self.synthesis_candidates.append(str(point))
            elif point:
                if turn.stance == "pro":
                    self.thesis_points.append(str(point))
                elif turn.stance == "con":
                    self.antithesis_points.append(str(point))

        # Update agent position confidence
        if turn.agent_id in self._positions:
            pos = self._positions[turn.agent_id]
            pos.confidence = turn.confidence
            if isinstance(turn.content, dict) and "position_vector" in turn.content:
                pos.approach = turn.content["position_vector"]

        # Check for phase transitions
        if self.concession_count >= 2 and self.current_phase == Phase.ARGUE:
            self.current_phase = Phase.SYNTHESIZE

    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if debate has reached resolution."""
        if self.current_round >= self.max_rounds:
            if self.synthesis_candidates:
                return True, DiscussionOutcome.CONSENSUS
            return True, DiscussionOutcome.NO_AGREEMENT

        if self.current_phase == Phase.SYNTHESIZE and len(self.synthesis_candidates) >= 2:
            return True, DiscussionOutcome.CONSENSUS

        if self.current_phase == Phase.SYNTHESIZE and len(self.synthesis_candidates) >= 1:
            return True, DiscussionOutcome.COMPROMISE

        if self.concession_count >= 3:
            return True, DiscussionOutcome.MAJORITY

        return False, DiscussionOutcome.NO_AGREEMENT

    def conclude(self) -> DiscussionResult:
        """Produce debate result with synthesis if possible."""
        is_done, outcome = self.check_completion()
        self.is_complete = True

        decision = None
        confidence = 0.0
        reasoning = ""

        if outcome == DiscussionOutcome.CONSENSUS and self.synthesis_candidates:
            decision = {
                "type": "synthesis",
                "thesis_points": self.thesis_points,
                "antithesis_points": self.antithesis_points,
                "synthesis": self.synthesis_candidates[-1],
            }
            confidence = _safe_mean([t.confidence for t in self.turns[-3:]])
            reasoning = (
                f"Debate reached synthesis after {self.current_round} rounds. "
                f"{len(self.thesis_points)} thesis points, {len(self.antithesis_points)} "
                f"antithesis points, {self.concession_count} concessions."
            )
        elif outcome == DiscussionOutcome.COMPROMISE:
            decision = {
                "type": "compromise",
                "thesis_summary": self.thesis_points[-3:] if self.thesis_points else [],
                "antithesis_summary": self.antithesis_points[-3:] if self.antithesis_points else [],
                "partial_synthesis": self.synthesis_candidates,
            }
            confidence = _safe_mean([t.confidence for t in self.turns[-3:]]) * 0.8
            reasoning = (
                f"Partial compromise reached after {self.current_round} rounds with "
                f"{self.concession_count} concessions."
            )
        else:
            decision = {
                "type": "no_agreement",
                "thesis_points": self.thesis_points,
                "antithesis_points": self.antithesis_points,
            }
            confidence = _safe_mean([t.confidence for t in self.turns])
            reasoning = f"No agreement after {self.current_round} rounds."

        return DiscussionResult(
            discussion_id=self.discussion_id,
            format="debate",
            outcome=outcome.value,
            decision=decision,
            confidence=confidence,
            turns=[t.to_dict() for t in self.turns],
            positions=[p.to_dict() for p in self.get_positions()],
            reasoning=reasoning,
        )


# ===========================================================================
# BrainstormStrategy — generate without judgment, then evaluate
# ===========================================================================

class BrainstormStrategy(DiscussionStrategy):
    """
    Two-phase brainstorm: generate ideas freely (no judgment), then evaluate.

    Phase 1 (GENERATE): All participants contribute ideas. No criticism allowed.
    Phase 2 (EVALUATE): Ideas are scored and ranked. Best ideas emerge.
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8):
        super().__init__(topic, participants, context, max_rounds, consensus_threshold)
        self.ideas: list[dict[str, Any]] = []
        self.evaluations: dict[str, list[dict[str, Any]]] = {}
        self.generate_rounds: int = max(2, max_rounds // 2)
        self.current_phase = Phase.GENERATE

    def initialize(self) -> list[DiscussionTurn]:
        """Start the brainstorm with a generation prompt."""
        initial: list[DiscussionTurn] = []
        for p in self.participants:
            turn = DiscussionTurn(
                agent_id=p.get("id", "unknown"),
                content={
                    "type": "brainstorm_start",
                    "prompt": f"Generate ideas for: {self.topic}",
                    "constraints": self.context.get("constraints", []),
                },
                confidence=1.0,
                stance="neutral",
                turn_number=len(initial) + 1,
                round_number=0,
                phase=self.current_phase.value,
            )
            initial.append(turn)
            self.add_turn(turn)
        self.current_round = 1
        return initial

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a brainstorm turn."""
        self.add_turn(turn)

        if self.current_phase == Phase.GENERATE:
            if isinstance(turn.content, dict) and turn.content.get("type") == "idea":
                idea_id = f"idea-{len(self.ideas)}"
                self.ideas.append({
                    "id": idea_id,
                    "content": turn.content.get("content", ""),
                    "agent_id": turn.agent_id,
                    "confidence": turn.confidence,
                    "round": self.current_round,
                })
                # Initialize position for this idea
                self._positions[turn.agent_id] = AgentPosition(
                    agent_id=turn.agent_id,
                    confidence=turn.confidence,
                    label="generating",
                )
        elif self.current_phase == Phase.EVALUATE:
            if isinstance(turn.content, dict) and turn.content.get("type") == "evaluation":
                target = turn.content.get("target_idea", "")
                if target not in self.evaluations:
                    self.evaluations[target] = []
                self.evaluations[target].append({
                    "agent_id": turn.agent_id,
                    "score": turn.content.get("score", 0.5),
                    "comment": turn.content.get("comment", ""),
                    "confidence": turn.confidence,
                })

        # Phase transition: after generate_rounds, switch to evaluate
        if self.current_phase == Phase.GENERATE and self.current_round >= self.generate_rounds:
            self.current_phase = Phase.EVALUATE

    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if brainstorm is complete."""
        if self.current_round >= self.max_rounds:
            return True, DiscussionOutcome.MAJORITY

        if self.current_phase == Phase.EVALUATE:
            all_evaluated = all(
                idea["id"] in self.evaluations
                for idea in self.ideas
            ) if self.ideas else False
            if all_evaluated:
                return True, DiscussionOutcome.CONSENSUS

            # If we've done enough evaluation rounds
            eval_rounds = self.current_round - self.generate_rounds
            if eval_rounds >= len(self.participants):
                return True, DiscussionOutcome.MAJORITY

        return False, DiscussionOutcome.NO_AGREEMENT

    def conclude(self) -> DiscussionResult:
        """Rank ideas and produce the best result."""
        self.is_complete = True
        is_done, outcome = self.check_completion()

        # Score each idea
        scored_ideas: list[dict[str, Any]] = []
        for idea in self.ideas:
            evals = self.evaluations.get(idea["id"], [])
            if evals:
                avg_score = _safe_mean([e["score"] for e in evals])
                avg_conf = _safe_mean([e["confidence"] for e in evals])
            else:
                avg_score = idea["confidence"]
                avg_conf = idea["confidence"]
            scored_ideas.append({
                **idea,
                "avg_score": avg_score,
                "eval_count": len(evals),
                "composite_score": avg_score * avg_conf,
            })

        scored_ideas.sort(key=lambda x: x["composite_score"], reverse=True)

        best = scored_ideas[0] if scored_ideas else None
        confidence = best["composite_score"] if best else 0.0
        reasoning = (
            f"Brainstorm produced {len(self.ideas)} ideas across "
            f"{self.generate_rounds} generation rounds. "
            f"Top idea scored {confidence:.2f}."
        )

        return DiscussionResult(
            discussion_id=self.discussion_id,
            format="brainstorm",
            outcome=outcome.value,
            decision={
                "type": "ranked_ideas",
                "total_ideas": len(self.ideas),
                "best_idea": best,
                "ranked": scored_ideas[:10],  # top 10
            },
            confidence=confidence,
            turns=[t.to_dict() for t in self.turns],
            positions=[p.to_dict() for p in self.get_positions()],
            reasoning=reasoning,
        )


# ===========================================================================
# ReviewStrategy — systematic critique using checklist
# ===========================================================================

class ReviewStrategy(DiscussionStrategy):
    """
    Systematic review: agents critique a proposal using a structured checklist.

    Each criterion is evaluated independently by each reviewer. The strategy
    aggregates scores and identifies strengths/weaknesses.
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8,
                 checklist: Optional[list[ReviewCriterion]] = None):
        super().__init__(topic, participants, context, max_rounds, consensus_threshold)
        self.checklist = checklist or list(DEFAULT_REVIEW_CHECKLIST)
        self.current_phase = Phase.CRITIQUE
        self.review_results: dict[str, list[dict[str, Any]]] = {}
        self._current_criterion_idx = 0

    def initialize(self) -> list[DiscussionTurn]:
        """Start the review with the proposal context."""
        initial: list[DiscussionTurn] = []
        for p in self.participants:
            turn = DiscussionTurn(
                agent_id=p.get("id", "unknown"),
                content={
                    "type": "review_start",
                    "proposal": self.topic,
                    "criteria": [c.name for c in self.checklist],
                    "context": self.context.get("background", {}),
                },
                confidence=0.8,
                stance="reviewer",
                turn_number=len(initial) + 1,
                round_number=0,
                phase=self.current_phase.value,
            )
            initial.append(turn)
            self.add_turn(turn)
            self._positions[p.get("id", "unknown")] = AgentPosition(
                agent_id=p.get("id", "unknown"),
                confidence=0.5,
                label="reviewer",
            )
        self.current_round = 1
        return initial

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a review turn."""
        self.add_turn(turn)

        if isinstance(turn.content, dict):
            if turn.content.get("type") == "criterion_review":
                criterion_name = turn.content.get("criterion", "")
                if turn.agent_id not in self.review_results:
                    self.review_results[turn.agent_id] = []
                self.review_results[turn.agent_id].append({
                    "criterion": criterion_name,
                    "score": turn.content.get("score", 0.5),
                    "notes": turn.content.get("notes", ""),
                    "suggestion": turn.content.get("suggestion", ""),
                    "confidence": turn.confidence,
                })
                # Update position
                if turn.agent_id in self._positions:
                    self._positions[turn.agent_id].confidence = turn.confidence
            elif turn.content.get("type") == "overall_review":
                if turn.agent_id not in self.review_results:
                    self.review_results[turn.agent_id] = []
                self.review_results[turn.agent_id].append({
                    "criterion": "__overall__",
                    "score": turn.content.get("score", 0.5),
                    "notes": turn.content.get("notes", ""),
                    "suggestion": turn.content.get("suggestion", ""),
                    "confidence": turn.confidence,
                })

    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if all reviewers have completed all criteria."""
        if self.current_round >= self.max_rounds:
            return True, DiscussionOutcome.MAJORITY

        expected_reviews = len(self.checklist) * len(self.participants)
        actual_reviews = sum(len(revs) for revs in self.review_results.values())
        if actual_reviews >= expected_reviews:
            return True, DiscussionOutcome.CONSENSUS

        # Check if all agents have submitted at least some reviews
        if len(self.review_results) >= len(self.participants):
            all_substantial = all(
                len(revs) >= len(self.checklist) // 2
                for revs in self.review_results.values()
            )
            if all_substantial:
                return True, DiscussionOutcome.MAJORITY

        return False, DiscussionOutcome.NO_AGREEMENT

    def conclude(self) -> DiscussionResult:
        """Aggregate reviews and produce final assessment."""
        self.is_complete = True
        is_done, outcome = self.check_completion()

        # Aggregate per-criterion scores
        criterion_scores: dict[str, list[float]] = {}
        for criterion in self.checklist:
            criterion_scores[criterion.name] = []

        for agent_id, reviews in self.review_results.items():
            for review in reviews:
                cname = review["criterion"]
                if cname in criterion_scores:
                    criterion_scores[cname].append(review["score"])

        aggregated: list[dict[str, Any]] = []
        for criterion in self.checklist:
            scores = criterion_scores.get(criterion.name, [])
            if scores:
                aggregated.append({
                    "name": criterion.name,
                    "description": criterion.description,
                    "weight": criterion.weight,
                    "avg_score": _safe_mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "reviewer_count": len(scores),
                    "passing": _safe_mean(scores) >= 0.6,
                })

        overall_scores = [a["avg_score"] for a in aggregated if a["reviewer_count"] > 0]
        overall = _safe_mean(overall_scores) if overall_scores else 0.0
        passing = all(a["passing"] for a in aggregated) if aggregated else False

        return DiscussionResult(
            discussion_id=self.discussion_id,
            format="review",
            outcome=outcome.value,
            decision={
                "type": "review_result",
                "overall_score": overall,
                "passing": passing,
                "criteria": aggregated,
                "strengths": [a["name"] for a in aggregated if a["avg_score"] >= 0.8],
                "weaknesses": [a["name"] for a in aggregated if a["avg_score"] < 0.6],
                "suggestions": [
                    r.get("suggestion", "")
                    for revs in self.review_results.values()
                    for r in revs
                    if r.get("suggestion")
                ],
            },
            confidence=overall,
            turns=[t.to_dict() for t in self.turns],
            positions=[p.to_dict() for p in self.get_positions()],
            reasoning=(
                f"Review completed with {len(self.review_results)} reviewers. "
                f"Overall score: {overall:.2f}. "
                f"{'PASS' if passing else 'FAIL'} — "
                f"Strengths: {len([a for a in aggregated if a['avg_score'] >= 0.8])}, "
                f"Weaknesses: {len([a for a in aggregated if a['avg_score'] < 0.6])}."
            ),
        )


# ===========================================================================
# NegotiationStrategy — multi-objective optimization
# ===========================================================================

class NegotiationStrategy(DiscussionStrategy):
    """
    Multi-objective negotiation: agents have conflicting goals, find compromise.

    Each agent has a utility function. The strategy finds Pareto-optimal
    compromises that maximize total utility.
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8):
        super().__init__(topic, participants, context, max_rounds, consensus_threshold)
        self.current_phase = Phase.PROPOSE
        self.proposals: list[dict[str, Any]] = []
        self.utility_functions: dict[str, Callable] = {}
        self.agent_utilities: dict[str, list[float]] = {}
        self.compromises: list[dict[str, Any]] = []

    def initialize(self) -> list[DiscussionTurn]:
        """Start negotiation with initial positions."""
        initial: list[DiscussionTurn] = []
        for p in self.participants:
            agent_id = p.get("id", "unknown")
            turn = DiscussionTurn(
                agent_id=agent_id,
                content={
                    "type": "initial_position",
                    "topic": self.topic,
                    "constraints": self.context.get("constraints", []),
                    "goals": p.get("goals", []),
                },
                confidence=p.get("weight", 1.0),
                stance=p.get("stance", "neutral"),
                turn_number=len(initial) + 1,
                round_number=0,
                phase=self.current_phase.value,
            )
            initial.append(turn)
            self.add_turn(turn)

            # Initialize agent position and utility tracking
            self._positions[agent_id] = AgentPosition(
                agent_id=agent_id,
                confidence=0.5,
                label=p.get("stance", "negotiator"),
                priority=p.get("priorities", []),
            )
            self.agent_utilities[agent_id] = [0.5]  # Start at neutral utility

            # Register utility function (default: linear combination of priorities)
            priorities = p.get("priorities", [1.0])
            weight = p.get("weight", 1.0)

            def make_utility(prios: list[float], w: float) -> Callable:
                def utility(proposal_vector: list[float]) -> float:
                    if not prios or not proposal_vector:
                        return 0.5
                    min_len = min(len(prios), len(proposal_vector))
                    score = sum(prios[i] * proposal_vector[i] for i in range(min_len))
                    max_possible = sum(abs(p) for p in prios[:min_len]) or 1.0
                    return _clamp(score / max_possible) * w
                return utility

            self.utility_functions[agent_id] = make_utility(priorities, weight)

        self.current_round = 1
        return initial

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a negotiation turn."""
        self.add_turn(turn)

        if isinstance(turn.content, dict):
            ctype = turn.content.get("type", "")

            if ctype == "proposal":
                proposal_vector = turn.content.get("proposal_vector", [])
                self.proposals.append({
                    "id": f"proposal-{len(self.proposals)}",
                    "agent_id": turn.agent_id,
                    "vector": proposal_vector,
                    "confidence": turn.confidence,
                    "round": self.current_round,
                    "description": turn.content.get("description", ""),
                })

                # Evaluate this proposal against all utility functions
                utilities = {
                    aid: self.utility_functions[aid](proposal_vector)
                    for aid in self.utility_functions
                }
                total_utility = sum(utilities.values()) / max(len(utilities), 1)

                # Update agent positions
                if turn.agent_id in self._positions:
                    self._positions[turn.agent_id].approach = proposal_vector
                    self._positions[turn.agent_id].confidence = total_utility

                # Track utilities
                for aid, util in utilities.items():
                    if aid in self.agent_utilities:
                        self.agent_utilities[aid].append(util)

            elif ctype == "counter":
                counter_to = turn.content.get("counter_to", "")
                proposal_vector = turn.content.get("proposal_vector", [])
                self.current_phase = Phase.COUNTER

                if turn.agent_id in self._positions:
                    self._positions[turn.agent_id].approach = proposal_vector

            elif ctype == "accept" or ctype == "compromise":
                self.compromises.append({
                    "agent_id": turn.agent_id,
                    "type": ctype,
                    "accepted_proposal": turn.content.get("proposal_id", ""),
                    "confidence": turn.confidence,
                    "round": self.current_round,
                })

    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if negotiation has reached compromise."""
        if self.current_round >= self.max_rounds:
            if self.compromises:
                return True, DiscussionOutcome.COMPROMISE
            return True, DiscussionOutcome.NO_AGREEMENT

        # All agents accepted
        if len(self.compromises) >= len(self.participants):
            accepted_ids = {c["accepted_proposal"] for c in self.compromises}
            if len(accepted_ids) == 1:
                return True, DiscussionOutcome.CONSENSUS
            return True, DiscussionOutcome.COMPROMISE

        # Check if utilities are converging (all trending up)
        all_converging = True
        for aid, utils in self.agent_utilities.items():
            if len(utils) >= 3:
                recent_avg = _safe_mean(utils[-3:])
                earlier_avg = _safe_mean(utils[-6:-3]) if len(utils) >= 6 else _safe_mean(utils[:-3])
                if recent_avg < earlier_avg:
                    all_converging = False
                    break

        if all_converging and len(self.compromises) >= len(self.participants) // 2 + 1:
            return True, DiscussionOutcome.MAJORITY

        return False, DiscussionOutcome.NO_AGREEMENT

    def conclude(self) -> DiscussionResult:
        """Produce negotiation result."""
        self.is_complete = True
        is_done, outcome = self.check_completion()

        # Find the proposal with highest total utility
        best_proposal = None
        best_total_utility = -1.0
        for proposal in self.proposals:
            utilities = {
                aid: self.utility_functions[aid](proposal["vector"])
                for aid in self.utility_functions
            }
            total = sum(utilities.values()) / max(len(utilities), 1)
            if total > best_total_utility:
                best_total_utility = total
                best_proposal = {**proposal, "agent_utilities": utilities, "total_utility": total}

        # Calculate min utility (worst-off agent) for fairness
        min_utility = 0.0
        if best_proposal and "agent_utilities" in best_proposal:
            utils = best_proposal["agent_utilities"]
            min_utility = min(utils.values()) if utils else 0.0

        return DiscussionResult(
            discussion_id=self.discussion_id,
            format="negotiation",
            outcome=outcome.value,
            decision={
                "type": "negotiation_result",
                "best_proposal": best_proposal,
                "total_proposals": len(self.proposals),
                "compromises": self.compromises,
                "fairness_score": min_utility,
                "efficiency_score": best_total_utility,
            },
            confidence=best_total_utility,
            turns=[t.to_dict() for t in self.turns],
            positions=[p.to_dict() for p in self.get_positions()],
            reasoning=(
                f"Negotiation produced {len(self.proposals)} proposals. "
                f"Best proposal utility: {best_total_utility:.2f}, "
                f"min agent utility: {min_utility:.2f}. "
                f"Outcome: {outcome.value}."
            ),
        )


# ===========================================================================
# PeerReviewStrategy — structured evaluation with rubric
# ===========================================================================

class PeerReviewStrategy(DiscussionStrategy):
    """
    Academic-style peer review: independent evaluation then discussion.

    Phase 1: Each reviewer independently evaluates against a rubric.
    Phase 2: Reviewers discuss discrepancies and produce consensus.
    """

    def __init__(self, topic: str, participants: list[dict[str, Any]],
                 context: Optional[dict[str, Any]] = None,
                 max_rounds: int = 10, consensus_threshold: float = 0.8,
                 rubric: Optional[list[ReviewCriterion]] = None):
        super().__init__(topic, participants, context, max_rounds, consensus_threshold)
        self.rubric = rubric or list(DEFAULT_REVIEW_CHECKLIST)
        self.current_phase = Phase.SINGLE
        self.independent_reviews: dict[str, dict[str, Any]] = {}
        self.discussion_turns: list[dict[str, Any]] = []
        self.revision_requests: list[str] = []

    def initialize(self) -> list[DiscussionTurn]:
        """Start independent review phase."""
        initial: list[DiscussionTurn] = []
        for p in self.participants:
            agent_id = p.get("id", "unknown")
            turn = DiscussionTurn(
                agent_id=agent_id,
                content={
                    "type": "independent_review",
                    "submission": self.topic,
                    "rubric": [
                        {"name": c.name, "description": c.description, "weight": c.weight}
                        for c in self.rubric
                    ],
                    "instructions": (
                        "Evaluate independently. Score each criterion 0.0-1.0. "
                        "Provide detailed justification."
                    ),
                },
                confidence=0.8,
                stance="reviewer",
                turn_number=len(initial) + 1,
                round_number=0,
                phase=self.current_phase.value,
            )
            initial.append(turn)
            self.add_turn(turn)
            self._positions[agent_id] = AgentPosition(
                agent_id=agent_id,
                confidence=0.5,
                label="independent_reviewer",
            )
        self.current_round = 1
        return initial

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Process a peer review turn."""
        self.add_turn(turn)

        if isinstance(turn.content, dict):
            ctype = turn.content.get("type", "")

            if ctype == "independent_review_result":
                # Store the full independent review
                self.independent_reviews[turn.agent_id] = {
                    "scores": turn.content.get("scores", {}),
                    "summary": turn.content.get("summary", ""),
                    "recommendation": turn.content.get("recommendation", ""),
                    "confidence": turn.confidence,
                    "strengths": turn.content.get("strengths", []),
                    "weaknesses": turn.content.get("weaknesses", []),
                }
                # Transition to discussion phase once all reviews are in
                if len(self.independent_reviews) >= len(self.participants):
                    self.current_phase = Phase.EVALUATE

                # Update position
                if turn.agent_id in self._positions:
                    scores = turn.content.get("scores", {})
                    avg = _safe_mean(list(scores.values())) if scores else 0.5
                    self._positions[turn.agent_id].confidence = avg
                    self._positions[turn.agent_id].label = "reviewed"

            elif ctype == "reviewer_comment":
                self.discussion_turns.append({
                    "agent_id": turn.agent_id,
                    "comment": turn.content.get("comment", ""),
                    "target": turn.content.get("target_reviewer", ""),
                    "round": self.current_round,
                    "confidence": turn.confidence,
                })

            elif ctype == "revision_request":
                self.revision_requests.append(turn.content.get("description", ""))

    def check_completion(self) -> tuple[bool, DiscussionOutcome]:
        """Check if peer review is complete."""
        if self.current_round >= self.max_rounds:
            return True, DiscussionOutcome.MAJORITY

        # All independent reviews submitted and at least some discussion
        if len(self.independent_reviews) >= len(self.participants):
            if self.current_phase == Phase.EVALUATE:
                # After discussion phase, check if reviewers have converged
                scores = [
                    _safe_mean(list(r["scores"].values()))
                    for r in self.independent_reviews.values()
                    if r.get("scores")
                ]
                if scores:
                    score_range = max(scores) - min(scores)
                    if score_range <= 0.2:  # Tight convergence
                        return True, DiscussionOutcome.CONSENSUS
                    elif score_range <= 0.4:
                        return True, DiscussionOutcome.MAJORITY

                # Enough discussion rounds
                if len(self.discussion_turns) >= len(self.participants) * 2:
                    return True, DiscussionOutcome.MAJORITY

        return False, DiscussionOutcome.NO_AGREEMENT

    def conclude(self) -> DiscussionResult:
        """Produce peer review consensus result."""
        self.is_complete = True
        is_done, outcome = self.check_completion()

        # Aggregate scores per criterion
        criterion_agg: dict[str, list[float]] = {}
        for review in self.independent_reviews.values():
            for cname, score in review.get("scores", {}).items():
                if cname not in criterion_agg:
                    criterion_agg[cname] = []
                criterion_agg[cname].append(float(score))

        aggregated_criteria = []
        for criterion in self.rubric:
            scores = criterion_agg.get(criterion.name, [])
            if scores:
                aggregated_criteria.append({
                    "name": criterion.name,
                    "weight": criterion.weight,
                    "avg_score": _safe_mean(scores),
                    "std_dev": (
                        (sum((s - _safe_mean(scores)) ** 2 for s in scores) / len(scores)) ** 0.5
                        if len(scores) > 1 else 0.0
                    ),
                    "reviewer_count": len(scores),
                    "agreement": 1.0 - min(1.0, (
                        (max(scores) - min(scores)) if len(scores) > 1 else 0.0
                    )),
                })

        overall = _safe_mean([a["avg_score"] for a in aggregated_criteria]) if aggregated_criteria else 0.0
        recommendations = [r["recommendation"] for r in self.independent_reviews.values()]

        return DiscussionResult(
            discussion_id=self.discussion_id,
            format="peer_review",
            outcome=outcome.value,
            decision={
                "type": "peer_review_result",
                "overall_score": overall,
                "criteria": aggregated_criteria,
                "recommendations": recommendations,
                "review_count": len(self.independent_reviews),
                "revision_needed": len(self.revision_requests) > 0,
                "revision_requests": self.revision_requests,
                "discussion_highlights": self.discussion_turns[:5],
            },
            confidence=overall,
            turns=[t.to_dict() for t in self.turns],
            positions=[p.to_dict() for p in self.get_positions()],
            reasoning=(
                f"Peer review with {len(self.independent_reviews)} reviewers. "
                f"Overall score: {overall:.2f}. "
                f"Recommendations: {', '.join(recommendations)}. "
                f"Outcome: {outcome.value}."
            ),
        )


# ===========================================================================
# DiscussionProtocol — the main orchestrator
# ===========================================================================

@dataclass(slots=True)
class DiscussionConfig:
    """Configuration for a DiscussionProtocol instance."""
    format: str = DiscussionFormat.DEBATE.value
    topic: str = ""
    participants: list[dict[str, Any]] = field(default_factory=list)
    context: Optional[dict[str, Any]] = None
    max_rounds: int = 10
    consensus_threshold: float = 0.8
    checklist: Optional[list[dict[str, Any]]] = None
    rubric: Optional[list[dict[str, Any]]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "format": self.format,
            "topic": self.topic,
            "participants": self.participants,
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
        }
        if self.context:
            d["context"] = self.context
        if self.checklist:
            d["checklist"] = self.checklist
        if self.rubric:
            d["rubric"] = self.rubric
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscussionConfig:
        checklist = None
        if data.get("checklist"):
            checklist = [ReviewCriterion.from_dict(c) for c in data["checklist"]]
        rubric = None
        if data.get("rubric"):
            rubric = [ReviewCriterion.from_dict(r) for r in data["rubric"]]
        return cls(
            format=data.get("format", DiscussionFormat.DEBATE.value),
            topic=data.get("topic", ""),
            participants=data.get("participants", []),
            context=data.get("context"),
            max_rounds=data.get("max_rounds", 10),
            consensus_threshold=data.get("consensus_threshold", 0.8),
            checklist=checklist,
            rubric=rubric,
            meta=data.get("meta", {}),
        )


class DiscussionProtocol:
    """
    Structured agent discussion — the core of A2A intelligence.

    The DiscussionProtocol is the main entry point for running structured
    multi-agent discussions. It selects the appropriate strategy based on
    the format and manages the full discussion lifecycle.

    Usage:
        config = DiscussionConfig(
            format="debate",
            topic="Should we use microservices?",
            participants=[{"id": "pro-agent", "stance": "pro"}, {"id": "con-agent", "stance": "con"}],
        )
        protocol = DiscussionProtocol(config)
        result = await protocol.run(turn_generator=my_agent_turn_fn)
    """

    STRATEGY_MAP: dict[str, type[DiscussionStrategy]] = {
        DiscussionFormat.DEBATE.value: DebateStrategy,
        DiscussionFormat.BRAINSTORM.value: BrainstormStrategy,
        DiscussionFormat.REVIEW.value: ReviewStrategy,
        DiscussionFormat.NEGOTIATION.value: NegotiationStrategy,
        DiscussionFormat.PEER_REVIEW.value: PeerReviewStrategy,
    }

    def __init__(self, config: DiscussionConfig):
        self.config = config
        self.strategy: Optional[DiscussionStrategy] = None
        self._result: Optional[DiscussionResult] = None
        self._initialize_strategy()

    def _initialize_strategy(self) -> None:
        """Create the appropriate strategy based on format."""
        fmt = self.config.format
        strategy_cls = self.STRATEGY_MAP.get(fmt)
        if strategy_cls is None:
            raise ValueError(f"Unknown discussion format: {fmt}. "
                             f"Supported: {list(self.STRATEGY_MAP.keys())}")

        kwargs: dict[str, Any] = {
            "topic": self.config.topic,
            "participants": self.config.participants,
            "context": self.config.context,
            "max_rounds": self.config.max_rounds,
            "consensus_threshold": self.config.consensus_threshold,
        }

        # Format-specific kwargs
        if fmt == DiscussionFormat.REVIEW.value and self.config.checklist:
            kwargs["checklist"] = [
                c if isinstance(c, ReviewCriterion) else ReviewCriterion.from_dict(c)
                for c in self.config.checklist
            ]
        if fmt == DiscussionFormat.PEER_REVIEW.value and self.config.rubric:
            kwargs["rubric"] = [
                r if isinstance(r, ReviewCriterion) else ReviewCriterion.from_dict(r)
                for r in self.config.rubric
            ]

        self.strategy = strategy_cls(**kwargs)

    def initialize(self) -> list[DiscussionTurn]:
        """Start the discussion. Returns initial turns."""
        if self.strategy is None:
            raise RuntimeError("Strategy not initialized")
        return self.strategy.initialize()

    def process_turn(self, turn: DiscussionTurn) -> None:
        """Submit a turn from an agent."""
        if self.strategy is None:
            raise RuntimeError("Strategy not initialized")
        self.strategy.process_turn(turn)
        self.strategy.current_round += 1

    def check_completion(self) -> tuple[bool, str]:
        """Check if discussion should end. Returns (is_done, outcome_str)."""
        if self.strategy is None:
            raise RuntimeError("Strategy not initialized")
        is_done, outcome = self.strategy.check_completion()
        return is_done, outcome.value

    def conclude(self) -> DiscussionResult:
        """End the discussion and get the result."""
        if self.strategy is None:
            raise RuntimeError("Strategy not initialized")
        self._result = self.strategy.conclude()
        return self._result

    async def run(self, turn_generator: Callable) -> DiscussionResult:
        """
        Run the full discussion lifecycle.

        Args:
            turn_generator: async function(discussion_state) -> DiscussionTurn | None
                Called each round to get the next turn from agents.

        Returns:
            DiscussionResult with the outcome.
        """
        if self.strategy is None:
            raise RuntimeError("Strategy not initialized")

        # Initialize
        self.initialize()

        # Run discussion rounds
        while not self.strategy.is_complete:
            is_done, _ = self.check_completion()
            if is_done:
                break

            # Get next turn from the agent turn generator
            state = self.get_state()
            turn = await turn_generator(state)
            if turn is None:
                break
            self.process_turn(turn)

        return self.conclude()

    def get_state(self) -> dict[str, Any]:
        """Get current discussion state for the turn generator."""
        if self.strategy is None:
            return {"error": "not initialized"}
        return {
            "discussion_id": self.strategy.discussion_id,
            "format": self.config.format,
            "topic": self.config.topic,
            "current_round": self.strategy.current_round,
            "current_phase": self.strategy.current_phase.value,
            "is_complete": self.strategy.is_complete,
            "turn_count": len(self.strategy.turns),
            "last_turn": self.strategy.turns[-1].to_dict() if self.strategy.turns else None,
            "positions": [p.to_dict() for p in self.strategy.get_positions()],
        }

    def get_result(self) -> Optional[DiscussionResult]:
        """Get the result if discussion has concluded."""
        return self._result

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "state": self.get_state(),
            "result": self._result.to_dict() if self._result else None,
        }
