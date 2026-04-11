"""
FLUX-A2A Consensus Detection — Knowing When Agents Agree (Round 5).

This module implements the ConsensusDetector, which determines when agents
have reached agreement during discussions. It supports six consensus types:

  - unanimous:    all agents agree on the same answer
  - majority:     >50% agree
  - supermajority: >66% agree
  - convergence:  agents are trending toward agreement (distance shrinking)
  - compromise:   agents accept a middle ground (utility increase)
  - stalemate:    agents are not converging (time to branch or escalate)

AgentPosition models each agent's stance in a multi-dimensional opinion space.
Positions are compared using cosine similarity and Euclidean distance.

Design principles:
  - Consensus is not binary — it's a spectrum from stalemate to unanimous.
  - Convergence is measured over time, not at a single point.
  - Stalemate detection triggers actionable resolution strategies.
  - All metrics are explainable — agents can understand WHY consensus failed.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ===========================================================================
# Enums
# ===========================================================================

class ConsensusType(str, Enum):
    """Types of consensus that can be detected."""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    CONVERGENCE = "convergence"
    COMPROMISE = "compromise"
    STALEMATE = "stalemate"


class ConvergenceTrend(str, Enum):
    """Trend of agent positions over time."""
    DIVERGING = "diverging"
    STABLE = "stable"
    SLOWLY_CONVERGING = "slowly_converging"
    CONVERGING = "converging"
    CONVERGED = "converged"


class ResolutionType(str, Enum):
    """Strategies for resolving stalemates."""
    REBRANCH = "rebranch"
    ESCALATE = "escalate"
    VOTE = "vote"
    COMPROMISE_FIND = "compromise_find"
    DEFER = "defer"
    SPLIT_DIFFERENCE = "split_difference"
    RANDOM_ARBITRATOR = "random_arbitrator"


class SimilarityMetric(str, Enum):
    """How to measure similarity between agent positions."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


# ===========================================================================
# Helpers
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ===========================================================================
# Vector similarity functions
# ===========================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if not a or not b or len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute Manhattan distance between two vectors."""
    if not a or not b or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b))


def normalize_vector(v: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    if not v:
        return []
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return [0.0] * len(v)
    return [x / norm for x in v]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the mean of a list of vectors."""
    if not vectors:
        return []
    n = min(len(v) for v in vectors) if vectors else 0
    if n == 0:
        return []
    return [_safe_mean([v[i] for v in vectors if i < len(v)]) for i in range(n)]


# ===========================================================================
# AgentPosition — multi-dimensional opinion space
# ===========================================================================

@dataclass(slots=True)
class AgentPosition:
    """
    An agent's position in multi-dimensional opinion space.

    Dimensions:
      - approach:  how (methodology, algorithm, strategy) — e.g. [0.8, 0.2, 0.5]
      - goal:      what (outcome, objective, target) — e.g. [1.0, 0.0]
      - priority:  importance weighting of concerns — e.g. [0.7, 0.3, 0.9, 0.1]
      - confidence: certainty in this position (0.0–1.0)

    Positions can be compared using cosine_similarity() or euclidean_distance().
    """
    agent_id: str = ""
    approach: list[float] = field(default_factory=list)
    goal: list[float] = field(default_factory=list)
    priority: list[float] = field(default_factory=list)
    confidence: float = 0.5
    label: str = ""
    timestamp: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)
        if not self.timestamp:
            self.timestamp = _now()

    def to_vector(self) -> list[float]:
        """Flatten all dimensions into a single vector for comparison."""
        return self.approach + self.goal + self.priority + [self.confidence]

    def to_normalized_vector(self) -> list[float]:
        """Flatten and normalize."""
        return normalize_vector(self.to_vector())

    def distance_to(self, other: AgentPosition,
                    metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """Distance between two positions (0.0 = identical, 1.0 = opposite)."""
        v1 = self.to_normalized_vector()
        v2 = other.to_normalized_vector()
        if not v1 or not v2:
            return 1.0
        if metric == SimilarityMetric.COSINE:
            return _clamp(1.0 - cosine_similarity(v1, v2))
        elif metric == SimilarityMetric.EUCLIDEAN:
            max_dist = math.sqrt(len(v1)) if v1 else 1.0
            return _clamp(euclidean_distance(v1, v2) / max_dist)
        elif metric == SimilarityMetric.MANHATTAN:
            max_dist = len(v1) if v1 else 1.0
            return _clamp(manhattan_distance(v1, v2) / max_dist)
        return 1.0

    def similarity_to(self, other: AgentPosition,
                      metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """Similarity between two positions (1.0 = identical, 0.0 = opposite)."""
        return 1.0 - self.distance_to(other, metric)

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
        if self.timestamp:
            d["timestamp"] = self.timestamp
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
            timestamp=data.get("timestamp", ""),
            meta=data.get("meta", {}),
        )


# ===========================================================================
# AgreementMetrics — snapshot of current agreement state
# ===========================================================================

@dataclass(slots=True)
class AgreementMetrics:
    """
    Metrics measuring how much agents agree at a point in time.
    """
    consensus_type: str = ConsensusType.STALEMATE.value
    agreement_score: float = 0.0  # 0.0–1.0 overall agreement
    pairwise_similarities: dict[str, float] = field(default_factory=dict)
    cluster_count: int = 0
    majority_fraction: float = 0.0
    mean_distance: float = 1.0
    min_distance: float = 1.0
    max_distance: float = 1.0
    confidence_alignment: float = 0.0  # How aligned are confidence levels?
    timestamp: str = ""

    def __post_init__(self) -> None:
        self.agreement_score = _clamp(self.agreement_score)
        self.majority_fraction = _clamp(self.majority_fraction)
        self.confidence_alignment = _clamp(self.confidence_alignment)
        if not self.timestamp:
            self.timestamp = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_type": self.consensus_type,
            "agreement_score": self.agreement_score,
            "pairwise_similarities": self.pairwise_similarities,
            "cluster_count": self.cluster_count,
            "majority_fraction": self.majority_fraction,
            "mean_distance": self.mean_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "confidence_alignment": self.confidence_alignment,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgreementMetrics:
        return cls(**{k: v for k, v in data.items() if k in {
            "consensus_type", "agreement_score", "pairwise_similarities",
            "cluster_count", "majority_fraction", "mean_distance",
            "min_distance", "max_distance", "confidence_alignment", "timestamp",
        }})


# ===========================================================================
# ConvergenceHistory — tracking positions over time
# ===========================================================================

@dataclass(slots=True)
class HistorySnapshot:
    """A snapshot of agreement metrics at a point in time."""
    round_number: int = 0
    metrics: Optional[AgreementMetrics] = None
    positions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "positions": self.positions,
        }


@dataclass(slots=True)
class ConvergenceHistory:
    """Time-series of agreement snapshots for trend detection."""
    snapshots: list[HistorySnapshot] = field(default_factory=list)
    max_length: int = 50

    def add_snapshot(self, round_number: int, metrics: AgreementMetrics,
                     positions: list[AgentPosition]) -> None:
        snapshot = HistorySnapshot(
            round_number=round_number,
            metrics=metrics,
            positions=[p.to_dict() for p in positions],
        )
        self.snapshots.append(snapshot)
        # Trim to max length
        if len(self.snapshots) > self.max_length:
            self.snapshots = self.snapshots[-self.max_length:]

    def get_agreement_scores(self) -> list[float]:
        return [
            s.metrics.agreement_score
            for s in self.snapshots
            if s.metrics is not None
        ]

    def get_mean_distances(self) -> list[float]:
        return [
            s.metrics.mean_distance
            for s in self.snapshots
            if s.metrics is not None
        ]

    def get_confidence_alignments(self) -> list[float]:
        return [
            s.metrics.confidence_alignment
            for s in self.snapshots
            if s.metrics is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "total_rounds": len(self.snapshots),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceHistory:
        history = cls()
        for sd in data.get("snapshots", []):
            metrics = AgreementMetrics.from_dict(sd["metrics"]) if sd.get("metrics") else None
            history.snapshots.append(HistorySnapshot(
                round_number=sd.get("round_number", 0),
                metrics=metrics,
                positions=sd.get("positions", []),
            ))
        return history


# ===========================================================================
# Stalemate — when agents can't agree
# ===========================================================================

@dataclass(slots=True)
class Stalemate:
    """Description of a detected stalemate and suggested resolution."""
    stalemate_id: str = ""
    detected_at_round: int = 0
    severity: float = 0.0  # 0.0–1.0, how stuck are we?
    reason: str = ""
    diverging_agents: list[str] = field(default_factory=list)
    cluster_info: dict[str, Any] = field(default_factory=dict)
    suggested_resolution: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.severity = _clamp(self.severity)
        if not self.stalemate_id:
            self.stalemate_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stalemate_id": self.stalemate_id,
            "detected_at_round": self.detected_at_round,
            "severity": self.severity,
            "reason": self.reason,
            "diverging_agents": self.diverging_agents,
            "cluster_info": self.cluster_info,
            "suggested_resolution": self.suggested_resolution,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stalemate:
        return cls(**{k: v for k, v in data.items() if k in {
            "stalemate_id", "detected_at_round", "severity", "reason",
            "diverging_agents", "cluster_info", "suggested_resolution", "meta",
        }})


# ===========================================================================
# ResolutionStrategy — actionable ways to resolve stalemates
# ===========================================================================

@dataclass(slots=True)
class ResolutionStrategy:
    """An actionable strategy for resolving a stalemate."""
    type: str = ResolutionType.REBRANCH.value
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    confidence: float = 0.5

    def __post_init__(self) -> None:
        self.expected_improvement = _clamp(self.expected_improvement)
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "params": self.params,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolutionStrategy:
        return cls(**{k: v for k, v in data.items() if k in {
            "type", "description", "params", "expected_improvement", "confidence",
        }})


# ===========================================================================
# ConsensusDetector — the main detector
# ===========================================================================

class ConsensusDetector:
    """
    Detect when agents have reached agreement.

    The ConsensusDetector provides multiple ways to measure and detect
    consensus, from simple majority voting to sophisticated convergence
    trend analysis over time.

    Usage:
        detector = ConsensusDetector(threshold=0.8)
        positions = [AgentPosition(...), AgentPosition(...)]
        metrics = detector.measure_agreement(positions)
        if metrics.consensus_type == "unanimous":
            # Act on consensus
        elif metrics.consensus_type == "stalemate":
            stalemate = detector.detect_stalemate(positions, metrics)
            resolution = detector.suggest_resolution(stalemate)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        convergence_window: int = 5,
        stalemate_rounds: int = 3,
    ):
        self.threshold = _clamp(threshold)
        self.similarity_metric = similarity_metric
        self.convergence_window = convergence_window
        self.stalemate_rounds = stalemate_rounds
        self.history = ConvergenceHistory()
        self._round_counter = 0

    # -- Core measurement ---------------------------------------------------

    def measure_agreement(self, positions: list[AgentPosition]) -> AgreementMetrics:
        """
        Measure how much agents agree at the current point in time.

        Computes pairwise similarity, clustering, majority fraction,
        and determines the consensus type.
        """
        self._round_counter += 1

        if not positions or len(positions) < 2:
            return AgreementMetrics(
                consensus_type=ConsensusType.STALEMATE.value,
                agreement_score=0.0,
                timestamp=_now(),
            )

        # 1. Compute pairwise similarities
        pairwise: dict[str, float] = {}
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                key = f"{positions[i].agent_id}:{positions[j].agent_id}"
                sim = positions[i].similarity_to(positions[j], self.similarity_metric)
                pairwise[key] = sim
                # Also store reverse for easy lookup
                rev_key = f"{positions[j].agent_id}:{positions[i].agent_id}"
                pairwise[rev_key] = sim

        # 2. Compute distance statistics
        distances: list[float] = [
            1.0 - sim for sim in pairwise.values()
        ]
        mean_dist = _safe_mean(distances) if distances else 1.0
        min_dist = min(distances) if distances else 1.0
        max_dist = max(distances) if distances else 1.0

        # 3. Confidence alignment (how similar are confidence levels?)
        confidences = [p.confidence for p in positions]
        confidence_alignment = 1.0 - min(1.0, _safe_std(confidences) * 2)

        # 4. Clustering (simple: positions within threshold distance form a cluster)
        clusters = self._find_clusters(positions)
        cluster_count = len(clusters)

        # 5. Majority fraction (fraction of agents in the largest cluster)
        largest_cluster_size = max(len(c) for c in clusters) if clusters else 0
        majority_fraction = largest_cluster_size / len(positions)

        # 6. Determine consensus type
        agreement_score = _clamp(1.0 - mean_dist)
        consensus_type = self._determine_consensus_type(
            positions, agreement_score, majority_fraction, cluster_count, pairwise
        )

        metrics = AgreementMetrics(
            consensus_type=consensus_type,
            agreement_score=agreement_score,
            pairwise_similarities=pairwise,
            cluster_count=cluster_count,
            majority_fraction=majority_fraction,
            mean_distance=mean_dist,
            min_distance=min_dist,
            max_distance=max_dist,
            confidence_alignment=confidence_alignment,
        )

        # Record in history
        self.history.add_snapshot(self._round_counter, metrics, positions)

        return metrics

    def _determine_consensus_type(
        self,
        positions: list[AgentPosition],
        agreement_score: float,
        majority_fraction: float,
        cluster_count: int,
        pairwise: dict[str, float],
    ) -> str:
        """Determine the type of consensus based on metrics."""
        n = len(positions)

        # Unanimous: all pairs very similar
        if agreement_score >= self.threshold and cluster_count == 1:
            return ConsensusType.UNANIMOUS.value

        # Supermajority: >66% in one cluster
        if majority_fraction > 0.66 and agreement_score >= self.threshold * 0.8:
            return ConsensusType.SUPERMAJORITY.value

        # Majority: >50% in one cluster
        if majority_fraction > 0.50 and agreement_score >= self.threshold * 0.6:
            return ConsensusType.MAJORITY.value

        # Check for compromise: moderate agreement, clusters exist
        if 0.3 <= agreement_score < self.threshold and cluster_count <= 2:
            return ConsensusType.COMPROMISE.value

        # Check for convergence based on history
        if len(self.history.snapshots) >= 3:
            scores = self.history.get_agreement_scores()
            recent = scores[-3:]
            if len(recent) >= 2 and recent[-1] > recent[0] + 0.05:
                return ConsensusType.CONVERGENCE.value

        return ConsensusType.STALEMATE.value

    def _find_clusters(self, positions: list[AgentPosition]) -> list[list[str]]:
        """
        Simple distance-based clustering.

        Two agents are in the same cluster if their distance is below
        the threshold. This is a basic union-find approach.
        """
        threshold_dist = 1.0 - self.threshold
        n = len(positions)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                dist = positions[i].distance_to(positions[j], self.similarity_metric)
                if dist <= threshold_dist:
                    union(i, j)

        clusters: dict[int, list[str]] = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(positions[i].agent_id)

        return list(clusters.values())

    # -- Convergence trend detection -----------------------------------------

    def detect_convergence_trend(
        self, history: Optional[ConvergenceHistory] = None
    ) -> ConvergenceTrend:
        """
        Analyze the trend of agreement scores over time.

        Returns a ConvergenceTrend indicating whether agents are
        moving toward or away from agreement.
        """
        h = history or self.history
        scores = h.get_agreement_scores()

        if len(scores) < 3:
            return ConvergenceTrend.STABLE

        recent = scores[-self.convergence_window:]
        earlier = scores[-self.convergence_window * 2:-self.convergence_window] \
            if len(scores) >= self.convergence_window * 2 else scores[:-self.convergence_window]

        if not earlier:
            return ConvergenceTrend.STABLE

        recent_mean = _safe_mean(recent)
        earlier_mean = _safe_mean(earlier)
        delta = recent_mean - earlier_mean

        # Also check variance (converging = variance decreasing)
        recent_std = _safe_std(recent)
        earlier_std = _safe_std(earlier) if len(earlier) >= 2 else 0.0
        variance_delta = recent_std - earlier_std

        # Check if already converged
        if recent_mean >= self.threshold and recent_std < 0.05:
            return ConvergenceTrend.CONVERGED

        # Strong convergence
        if delta > 0.1 and variance_delta < 0:
            return ConvergenceTrend.CONVERGING

        # Slow convergence
        if delta > 0.03 and variance_delta <= 0:
            return ConvergenceTrend.SLOWLY_CONVERGING

        # Diverging
        if delta < -0.05:
            return ConvergenceTrend.DIVERGING

        return ConvergenceTrend.STABLE

    # -- Stalemate detection ------------------------------------------------

    def detect_stalemate(
        self,
        positions: list[AgentPosition],
        metrics: Optional[AgreementMetrics] = None,
    ) -> Optional[Stalemate]:
        """
        Detect if agents are in a stalemate.

        A stalemate is detected when:
        1. Agreement score is low AND not improving over time
        2. Multiple distinct clusters exist
        3. Confidence levels are high but disagreeing (entrenched positions)
        """
        if metrics is None:
            metrics = self.measure_agreement(positions)

        # Need history to detect stalemate
        if len(self.history.snapshots) < self.stalemate_rounds:
            return None

        # Check if metrics indicate stalemate
        if metrics.consensus_type != ConsensusType.STALEMATE.value:
            return None

        # Check trend: are we stuck?
        trend = self.detect_convergence_trend()
        if trend in (ConvergenceTrend.CONVERGING, ConvergenceTrend.SLOWLY_CONVERGING):
            return None  # Still making progress

        # Calculate severity
        severity = _clamp(1.0 - metrics.agreement_score)
        if trend == ConvergenceTrend.DIVERGING:
            severity = min(1.0, severity + 0.2)

        # Identify diverging agents (those most distant from the group centroid)
        group_vector = mean_vector([p.to_vector() for p in positions])
        diverging: list[str] = []
        for p in positions:
            dist = euclidean_distance(p.to_vector(), group_vector)
            if dist > metrics.mean_distance * 1.5:
                diverging.append(p.agent_id)

        # Cluster info
        clusters = self._find_clusters(positions)
        cluster_info = {
            "count": len(clusters),
            "sizes": [len(c) for c in clusters],
            "agents_per_cluster": clusters,
        }

        # Build stalemate
        stalemate = Stalemate(
            detected_at_round=self._round_counter,
            severity=severity,
            reason=(
                f"Agents have not converged after {self._round_counter} rounds. "
                f"Agreement score: {metrics.agreement_score:.2f}, "
                f"Trend: {trend.value}, Clusters: {len(clusters)}."
            ),
            diverging_agents=diverging,
            cluster_info=cluster_info,
        )

        # Attach suggested resolution
        stalemate.suggested_resolution = self.suggest_resolution(stalemate).to_dict()

        return stalemate

    # -- Resolution suggestions ----------------------------------------------

    def suggest_resolution(self, stalemate: Stalemate) -> ResolutionStrategy:
        """
        Suggest a strategy for resolving a stalemate.

        The suggestion is based on the stalemate's characteristics:
        - Severity, cluster count, trend, etc.
        """
        severity = stalemate.severity
        cluster_count = stalemate.cluster_info.get("count", 1)
        trend = self.detect_convergence_trend()

        # High severity with many clusters: re-branch
        if severity > 0.8 and cluster_count >= 3:
            return ResolutionStrategy(
                type=ResolutionType.REBRANCH.value,
                description=(
                    "Strong disagreement detected across multiple positions. "
                    "Re-branch the discussion into parallel explorations "
                    "of each position, then synthesize."
                ),
                params={
                    "branch_per_cluster": True,
                    "refined_prompts": True,
                    "include_dissenting_views": True,
                },
                expected_improvement=0.3,
                confidence=0.7,
            )

        # Diverging trend: escalate to higher authority
        if trend == ConvergenceTrend.DIVERGING:
            return ResolutionStrategy(
                type=ResolutionType.ESCALATE.value,
                description=(
                    "Agents are actively diverging. Escalate to a "
                    "moderator or higher-level agent for arbitration."
                ),
                params={
                    "escalate_to": "moderator",
                    "provide_summary": True,
                },
                expected_improvement=0.2,
                confidence=0.6,
            )

        # Two clusters: try compromise or split difference
        if cluster_count == 2:
            return ResolutionStrategy(
                type=ResolutionType.SPLIT_DIFFERENCE.value,
                description=(
                    "Two distinct positions detected. Try splitting the "
                    "difference or finding a compromise position."
                ),
                params={
                    "method": "weighted_midpoint",
                    "consider_priorities": True,
                },
                expected_improvement=0.4,
                confidence=0.65,
            )

        # Moderate stalemate: vote
        if severity > 0.5:
            return ResolutionStrategy(
                type=ResolutionType.VOTE.value,
                description=(
                    "Moderate disagreement. Conduct a weighted vote "
                    "to break the stalemate."
                ),
                params={
                    "method": "weighted_confidence",
                    "threshold": 0.5,
                },
                expected_improvement=0.3,
                confidence=0.7,
            )

        # Low severity: defer and gather more information
        return ResolutionStrategy(
            type=ResolutionType.DEFER.value,
            description=(
                "Mild disagreement. Defer the decision and gather "
                "more information to reduce ambiguity."
            ),
            params={
                "gather_from": "all_agents",
                "focus_on": stalemate.diverging_agents,
            },
            expected_improvement=0.2,
            confidence=0.5,
        )

    # -- Convenience methods ------------------------------------------------

    def check_consensus(
        self,
        positions: list[AgentPosition],
        required_type: ConsensusType = ConsensusType.MAJORITY,
    ) -> tuple[bool, AgreementMetrics, Optional[Stalemate]]:
        """
        Quick check: do positions meet the required consensus type?

        Returns (has_consensus, metrics, stalemate_if_any).
        """
        metrics = self.measure_agreement(positions)

        type_priority = {
            ConsensusType.UNANIMOUS: 5,
            ConsensusType.SUPERMAJORITY: 4,
            ConsensusType.MAJORITY: 3,
            ConsensusType.COMPROMISE: 2,
            ConsensusType.CONVERGENCE: 1,
            ConsensusType.STALEMATE: 0,
        }

        current_level = type_priority.get(
            ConsensusType(metrics.consensus_type), 0
        )
        required_level = type_priority.get(required_type, 0)

        has_consensus = current_level >= required_level

        stalemate = None
        if not has_consensus and current_level == 0:
            stalemate = self.detect_stalemate(positions, metrics)

        return has_consensus, metrics, stalemate

    def get_consensus_summary(self) -> dict[str, Any]:
        """Get a summary of the consensus detection state."""
        trend = self.detect_convergence_trend()
        scores = self.history.get_agreement_scores()

        return {
            "total_rounds": self._round_counter,
            "current_trend": trend.value,
            "latest_agreement_score": scores[-1] if scores else 0.0,
            "agreement_score_history": scores,
            "threshold": self.threshold,
            "similarity_metric": self.similarity_metric.value,
            "snapshots": len(self.history.snapshots),
        }

    def reset(self) -> None:
        """Reset the detector's history and counter."""
        self.history = ConvergenceHistory()
        self._round_counter = 0
