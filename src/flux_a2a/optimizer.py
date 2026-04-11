"""
Cross-Language Optimization Engine for FLUX Multilingual Runtime.

Analyzes programs at the TYPE level and optimizes execution by leveraging
paradigm-specific strengths across the six FLUX languages (ZHO, DEU, KOR,
SAN, LAT, WEN).  The optimizer splits work across paradigms, minimizes
bridge costs, and maximizes information preservation.

Architecture:
    ParadigmProfiler        — Profiles which paradigms suit different operations
    CrossLanguageOptimizer   — Splits programs across paradigms by strategy
    BridgeOptimizer          — Optimizes bridge paths and caches results

Paradigm Strengths (from type system analysis):
    ZHO — Classifiers: data classification, shape-based dispatch
    DEU — Case system (4): structural typing, scope-based control
    KOR — Honorifics: concurrency, access control, social hierarchy
    SAN — 8 cases: formal reasoning, multi-level scope nesting
    LAT — 6 tenses: temporal reasoning, lifecycle management
    WEN — Context stack: implicit computation, contextual inference

Reference: type_safe_bridge.py (TypeAlgebra, BridgeCostMatrix, TypeWitness),
           cross_compiler.py (CompilationResult), types.py (FluxType)
"""

from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
)

from flux_a2a.types import (
    ConstraintKind,
    FluxBaseType,
    FluxConstraint,
    FluxType,
    _PARADIGM_TO_BASE,
)
from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    BridgeCostReport,
    PreservationDegree,
    TypeAlgebra,
    TypeWitness,
    WitnessConstraint,
)
from flux_a2a.cross_compiler import (
    CompilationResult,
)


# ══════════════════════════════════════════════════════════════════
# Constants & Type Aliases
# ══════════════════════════════════════════════════════════════════

LangTag: TypeAlias = str

SUPPORTED_LANGUAGES: FrozenSet[LangTag] = frozenset(
    {"zho", "deu", "kor", "san", "lat", "wen"}
)

# Operations that the optimizer can reason about at the type level.
# These map to combinations of FluxBaseType + ConstraintKind.
KNOWN_OPERATIONS: Tuple[str, ...] = (
    "data_classification",    # Classifying data by shape/animacy (ZHO)
    "structural_typing",      # Strong structural type discipline (DEU)
    "access_control",         # Honorific-based permission hierarchy (KOR)
    "formal_reasoning",       # Multi-level logical scope nesting (SAN)
    "temporal_reasoning",     # Time-based lifecycle and scheduling (LAT)
    "contextual_inference",   # Implicit computation from context (WEN)
    "scope_management",       # Controlling visibility/access (DEU/SAN)
    "concurrency",            # Parallel/async execution patterns (KOR)
    "mutation",               # In-place state changes (DEU/LAT)
    "pure_transform",         # Stateless data transformation (SAN)
    "pattern_matching",       # Classifier-based pattern dispatch (ZHO)
    "capability_check",       # Permission/capability verification (KOR)
    "deferred_computation",   # Lazy/future computation (LAT)
    "stateless_abstraction",  # High-level abstract composition (WEN/SAN)
    "imperative_sequence",    # Ordered step-by-step execution (LAT/DEU)
    "container_ops",          # Collection manipulation (ZHO/DEU)
)


# ══════════════════════════════════════════════════════════════════
# 1. ParadigmProfile — Language Capability Model
# ══════════════════════════════════════════════════════════════════

@dataclass
class ParadigmProfile:
    """Capability profile for a single FLUX language paradigm.

    Attributes:
        lang: Language tag (zho, deu, kor, san, lat, wen).
        strengths: operation -> suitability score [0.0, 1.0].
        weaknesses: operation -> weakness score [0.0, 1.0].
        latency_estimate: operation -> estimated relative latency (lower = faster).
        memory_estimate: operation -> estimated relative memory usage.
    """
    lang: str
    strengths: Dict[str, float] = field(default_factory=dict)
    weaknesses: Dict[str, float] = field(default_factory=dict)
    latency_estimate: Dict[str, float] = field(default_factory=dict)
    memory_estimate: Dict[str, float] = field(default_factory=dict)

    def suitability(self, operation: str) -> float:
        """Net suitability score for an operation (strength - weakness)."""
        s = self.strengths.get(operation, 0.3)
        w = self.weaknesses.get(operation, 0.1)
        return max(0.0, min(1.0, s - w))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "strengths": {k: round(v, 4) for k, v in self.strengths.items()},
            "weaknesses": {k: round(v, 4) for k, v in self.weaknesses.items()},
            "latency_estimate": {
                k: round(v, 4) for k, v in self.latency_estimate.items()
            },
            "memory_estimate": {
                k: round(v, 4) for k, v in self.memory_estimate.items()
            },
        }


# ══════════════════════════════════════════════════════════════════
# 2. ParadigmProfiler — Capability Analysis Engine
# ══════════════════════════════════════════════════════════════════

class ParadigmProfiler:
    """Profiles which paradigms are best suited for different operations.

    Produces ParadigmProfile instances for each of the 6 FLUX languages,
    encoding their strengths and weaknesses based on type-system analysis.

    Pre-defined profiles:
      - ZHO: Strong in data classification, pattern matching, container ops.
             Weak in precise scope control, mutation.
      - DEU: Strong in structural typing, scope management, imperative seq.
             Weak in temporal flexibility, concurrency.
      - KOR: Strong in concurrency, access control, capability check.
             Weak in pure FP (pure_transform), formal reasoning.
      - SAN: Strong in formal reasoning, multi-level scope, pure transform.
             Weak in imperative mutation, concurrency.
      - LAT: Strong in temporal reasoning, deferred computation, imperative seq.
             Weak in parallel execution, contextual inference.
      - WEN: Strong in contextual inference, stateless abstraction.
             Weak in explicit typing, precise scope control.

    Usage:
        profiler = ParadigmProfiler()
        profile = profiler.profile("zho")
        print(profile.suitability("data_classification"))  # 0.90
        print(profiler.optimal_lang_for("temporal_reasoning"))  # [("lat", 0.85), ...]
    """

    # Pre-defined strength/weakness profiles for all 6 languages
    _PREDEFINED_STRENGTHS: Dict[str, Dict[str, float]] = {
        "zho": {
            "data_classification": 0.95,
            "pattern_matching": 0.90,
            "container_ops": 0.80,
            "stateless_abstraction": 0.70,
            "structural_typing": 0.50,
            "scope_management": 0.35,
            "access_control": 0.30,
            "formal_reasoning": 0.30,
            "temporal_reasoning": 0.35,
            "contextual_inference": 0.60,
            "concurrency": 0.50,
            "mutation": 0.40,
            "pure_transform": 0.50,
            "capability_check": 0.30,
            "deferred_computation": 0.30,
            "imperative_sequence": 0.45,
        },
        "deu": {
            "structural_typing": 0.90,
            "scope_management": 0.85,
            "imperative_sequence": 0.80,
            "mutation": 0.75,
            "container_ops": 0.70,
            "data_classification": 0.55,
            "pattern_matching": 0.50,
            "access_control": 0.45,
            "formal_reasoning": 0.50,
            "temporal_reasoning": 0.40,
            "contextual_inference": 0.25,
            "concurrency": 0.35,
            "pure_transform": 0.45,
            "capability_check": 0.40,
            "deferred_computation": 0.40,
            "stateless_abstraction": 0.40,
        },
        "kor": {
            "concurrency": 0.90,
            "access_control": 0.90,
            "capability_check": 0.85,
            "data_classification": 0.60,
            "pattern_matching": 0.50,
            "structural_typing": 0.45,
            "scope_management": 0.50,
            "formal_reasoning": 0.30,
            "temporal_reasoning": 0.40,
            "contextual_inference": 0.50,
            "mutation": 0.55,
            "pure_transform": 0.25,
            "deferred_computation": 0.45,
            "imperative_sequence": 0.55,
            "container_ops": 0.50,
            "stateless_abstraction": 0.45,
        },
        "san": {
            "formal_reasoning": 0.95,
            "scope_management": 0.95,
            "pure_transform": 0.90,
            "stateless_abstraction": 0.85,
            "structural_typing": 0.70,
            "data_classification": 0.40,
            "pattern_matching": 0.35,
            "access_control": 0.50,
            "temporal_reasoning": 0.45,
            "contextual_inference": 0.30,
            "concurrency": 0.25,
            "mutation": 0.20,
            "capability_check": 0.45,
            "deferred_computation": 0.35,
            "imperative_sequence": 0.30,
            "container_ops": 0.40,
        },
        "lat": {
            "temporal_reasoning": 0.95,
            "deferred_computation": 0.90,
            "imperative_sequence": 0.80,
            "mutation": 0.70,
            "structural_typing": 0.55,
            "scope_management": 0.60,
            "data_classification": 0.35,
            "pattern_matching": 0.30,
            "access_control": 0.35,
            "formal_reasoning": 0.50,
            "contextual_inference": 0.25,
            "concurrency": 0.20,
            "pure_transform": 0.50,
            "capability_check": 0.30,
            "container_ops": 0.45,
            "stateless_abstraction": 0.55,
        },
        "wen": {
            "contextual_inference": 0.95,
            "stateless_abstraction": 0.90,
            "pattern_matching": 0.60,
            "data_classification": 0.50,
            "structural_typing": 0.20,
            "scope_management": 0.25,
            "access_control": 0.40,
            "formal_reasoning": 0.55,
            "temporal_reasoning": 0.35,
            "concurrency": 0.30,
            "mutation": 0.25,
            "pure_transform": 0.65,
            "capability_check": 0.35,
            "deferred_computation": 0.40,
            "imperative_sequence": 0.30,
            "container_ops": 0.45,
        },
    }

    _PREDEFINED_WEAKNESSES: Dict[str, Dict[str, float]] = {
        "zho": {
            "scope_management": 0.60,
            "formal_reasoning": 0.60,
            "mutation": 0.50,
            "access_control": 0.50,
            "temporal_reasoning": 0.55,
            "deferred_computation": 0.60,
        },
        "deu": {
            "concurrency": 0.60,
            "temporal_reasoning": 0.55,
            "contextual_inference": 0.65,
            "stateless_abstraction": 0.50,
            "pure_transform": 0.45,
        },
        "kor": {
            "pure_transform": 0.65,
            "formal_reasoning": 0.60,
            "structural_typing": 0.50,
            "contextual_inference": 0.45,
            "scope_management": 0.45,
        },
        "san": {
            "mutation": 0.75,
            "concurrency": 0.70,
            "contextual_inference": 0.65,
            "imperative_sequence": 0.60,
            "data_classification": 0.55,
        },
        "lat": {
            "concurrency": 0.75,
            "contextual_inference": 0.65,
            "pure_transform": 0.40,
            "access_control": 0.55,
            "pattern_matching": 0.60,
        },
        "wen": {
            "structural_typing": 0.70,
            "scope_management": 0.65,
            "mutation": 0.65,
            "access_control": 0.55,
            "imperative_sequence": 0.60,
            "capability_check": 0.55,
        },
    }

    # Latency estimates: lower = faster.  1.0 = average, < 1.0 = fast, > 1.0 = slow.
    _PREDEFINED_LATENCY: Dict[str, Dict[str, float]] = {
        "zho": {
            "data_classification": 0.5,
            "pattern_matching": 0.6,
            "container_ops": 0.7,
            "scope_management": 1.5,
            "mutation": 1.3,
        },
        "deu": {
            "structural_typing": 0.5,
            "scope_management": 0.6,
            "imperative_sequence": 0.6,
            "concurrency": 1.8,
            "temporal_reasoning": 1.5,
        },
        "kor": {
            "concurrency": 0.4,
            "access_control": 0.5,
            "capability_check": 0.5,
            "pure_transform": 1.6,
            "formal_reasoning": 1.5,
        },
        "san": {
            "formal_reasoning": 0.4,
            "scope_management": 0.5,
            "pure_transform": 0.5,
            "concurrency": 2.0,
            "mutation": 1.9,
        },
        "lat": {
            "temporal_reasoning": 0.4,
            "deferred_computation": 0.5,
            "imperative_sequence": 0.6,
            "concurrency": 2.2,
            "contextual_inference": 1.8,
        },
        "wen": {
            "contextual_inference": 0.4,
            "stateless_abstraction": 0.5,
            "pattern_matching": 0.8,
            "structural_typing": 1.7,
            "scope_management": 1.6,
        },
    }

    # Memory estimates: relative to a baseline. 1.0 = average.
    _PREDEFINED_MEMORY: Dict[str, Dict[str, float]] = {
        "zho": {
            "data_classification": 0.8,
            "pattern_matching": 0.9,
            "container_ops": 1.3,
            "scope_management": 1.5,
        },
        "deu": {
            "structural_typing": 0.7,
            "scope_management": 0.8,
            "imperative_sequence": 0.9,
            "concurrency": 1.5,
        },
        "kor": {
            "concurrency": 1.4,
            "access_control": 0.8,
            "capability_check": 0.7,
            "pure_transform": 1.3,
        },
        "san": {
            "formal_reasoning": 0.9,
            "scope_management": 0.6,
            "pure_transform": 0.7,
            "mutation": 1.5,
        },
        "lat": {
            "temporal_reasoning": 0.8,
            "deferred_computation": 1.2,
            "imperative_sequence": 0.8,
            "concurrency": 1.8,
        },
        "wen": {
            "contextual_inference": 1.5,
            "stateless_abstraction": 0.6,
            "pattern_matching": 1.0,
            "structural_typing": 1.4,
        },
    }

    def __init__(self) -> None:
        self._profiles: Dict[str, ParadigmProfile] = {}
        self._build_profiles()

    def _build_profiles(self) -> None:
        """Build all 6 paradigm profiles from pre-defined data."""
        for lang in SUPPORTED_LANGUAGES:
            strengths = dict(self._PREDEFINED_STRENGTHS.get(lang, {}))
            weaknesses = dict(self._PREDEFINED_WEAKNESSES.get(lang, {}))
            latency = dict(self._PREDEFINED_LATENCY.get(lang, {}))
            memory = dict(self._PREDEFINED_MEMORY.get(lang, {}))
            self._profiles[lang] = ParadigmProfile(
                lang=lang,
                strengths=strengths,
                weaknesses=weaknesses,
                latency_estimate=latency,
                memory_estimate=memory,
            )

    def profile(self, lang: str) -> ParadigmProfile:
        """Get the capability profile for a language.

        Args:
            lang: Language tag (zho, deu, kor, san, lat, wen).

        Returns:
            The ParadigmProfile for the given language.

        Raises:
            ValueError: If the language is not supported.
        """
        if lang not in self._profiles:
            raise ValueError(
                f"Unsupported language: {lang}. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
            )
        return self._profiles[lang]

    def compare(self, a: str, b: str, operation: str) -> str:
        """Compare two languages for a specific operation.

        Args:
            a: First language tag.
            b: Second language tag.
            operation: The operation to compare on.

        Returns:
            The language tag with higher suitability, or 'tie' if equal.
        """
        prof_a = self.profile(a)
        prof_b = self.profile(b)
        score_a = prof_a.suitability(operation)
        score_b = prof_b.suitability(operation)
        if abs(score_a - score_b) < 0.01:
            return "tie"
        return a if score_a > score_b else b

    def optimal_lang_for(
        self, operation: str
    ) -> List[Tuple[str, float]]:
        """Rank all languages by suitability for an operation.

        Args:
            operation: The operation to rank for.

        Returns:
            List of (language, suitability_score) tuples, sorted descending.
        """
        scores: List[Tuple[str, float]] = []
        for lang in SUPPORTED_LANGUAGES:
            prof = self.profile(lang)
            scores.append((lang, prof.suitability(operation)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def all_profiles(self) -> List[ParadigmProfile]:
        """Return profiles for all supported languages."""
        return [self._profiles[lang] for lang in sorted(SUPPORTED_LANGUAGES)]

    def infer_operations(self, types: List[FluxType]) -> Dict[str, float]:
        """Infer which operations a set of types requires.

        Analyzes the base types and constraints to determine what operations
        the program needs to perform, returning a relevance score for each.

        Args:
            types: List of FluxTypes from a program.

        Returns:
            Dict mapping operation name to relevance score [0.0, 1.0].
        """
        if not types:
            return {}

        op_scores: Dict[str, float] = {}

        for ft in types:
            # Classify based on base type
            bt = ft.base_type
            base_op = self._base_type_to_operation(bt)
            if base_op:
                op_scores[base_op] = op_scores.get(base_op, 0.0) + 0.5

            # Classify based on constraints
            for c in ft.constraints:
                constraint_ops = self._constraint_to_operations(c.kind, c.language)
                for op, weight in constraint_ops:
                    op_scores[op] = op_scores.get(op, 0.0) + weight * ft.confidence

            # Weight by confidence
            for op in list(op_scores.keys()):
                op_scores[op] *= ft.confidence

        # Normalize to [0, 1]
        if op_scores:
            max_score = max(op_scores.values())
            if max_score > 0:
                op_scores = {k: v / max_score for k, v in op_scores.items()}

        return op_scores

    def _base_type_to_operation(self, bt: FluxBaseType) -> Optional[str]:
        """Map a FluxBaseType to its primary operation."""
        mapping = {
            FluxBaseType.VALUE: "data_classification",
            FluxBaseType.ACTIVE: "pattern_matching",
            FluxBaseType.CONTAINER: "container_ops",
            FluxBaseType.SCOPE: "scope_management",
            FluxBaseType.CAPABILITY: "capability_check",
            FluxBaseType.MODAL: "temporal_reasoning",
            FluxBaseType.UNCERTAIN: "pattern_matching",
            FluxBaseType.CONTEXTUAL: "contextual_inference",
        }
        return mapping.get(bt)

    def _constraint_to_operations(
        self, kind: ConstraintKind, language: str
    ) -> List[Tuple[str, float]]:
        """Map a constraint kind to relevant operations with weights."""
        ops: List[Tuple[str, float]] = []

        if kind in (ConstraintKind.CLASSIFIER_SHAPE, ConstraintKind.CLASSIFIER_ANIMACY):
            ops.append(("data_classification", 1.0))
            ops.append(("pattern_matching", 0.7))
        elif kind == ConstraintKind.CLASSIFIER_COUNTABILITY:
            ops.append(("container_ops", 1.0))
        elif kind == ConstraintKind.GENDER_AGREEMENT:
            ops.append(("structural_typing", 1.0))
        elif kind == ConstraintKind.CASE_MARKING:
            ops.append(("scope_management", 1.0))
            ops.append(("formal_reasoning", 0.5))
        elif kind == ConstraintKind.NUMBER_AGREEMENT:
            ops.append(("structural_typing", 0.6))
            ops.append(("container_ops", 0.4))
        elif kind == ConstraintKind.HONORIFIC_LEVEL:
            ops.append(("access_control", 1.0))
            ops.append(("capability_check", 0.8))
        elif kind == ConstraintKind.SPEECH_ACT:
            ops.append(("imperative_sequence", 0.7))
        elif kind == ConstraintKind.SCOPE_LEVEL:
            ops.append(("scope_management", 1.0))
            ops.append(("formal_reasoning", 0.8))
        elif kind == ConstraintKind.SCOPE_ACCESS:
            ops.append(("access_control", 0.8))
        elif kind == ConstraintKind.TEMPORAL_ASPECT:
            ops.append(("temporal_reasoning", 1.0))
            ops.append(("deferred_computation", 0.6))
        elif kind == ConstraintKind.EXECUTION_MODE:
            ops.append(("imperative_sequence", 0.8))
            ops.append(("concurrency", 0.4))
        elif kind == ConstraintKind.CONTEXT_DOMAIN:
            ops.append(("contextual_inference", 1.0))
        elif kind == ConstraintKind.TOPIC_REGISTER:
            ops.append(("contextual_inference", 0.8))
            ops.append(("stateless_abstraction", 0.5))
        elif kind == ConstraintKind.CONFIDENCE_BOUND:
            ops.append(("pattern_matching", 0.5))
        elif kind == ConstraintKind.TRUST_REQUIREMENT:
            ops.append(("capability_check", 1.0))
            ops.append(("access_control", 0.8))
        elif kind == ConstraintKind.EFFECT_SET:
            ops.append(("pure_transform", 0.6))
            ops.append(("mutation", 0.4))

        return ops


# ══════════════════════════════════════════════════════════════════
# 3. Optimization Strategy & Data Structures
# ══════════════════════════════════════════════════════════════════

class OptimizationStrategy(str, Enum):
    """Strategy for cross-language optimization.

    PARADIGM_SPECIALIST:   Delegate each operation to the best-suited paradigm.
                           Maximizes quality, may increase bridge calls.
    MINIMUM_BRIDGING:      Minimize the number of cross-language calls.
                           Prefers keeping types in the source paradigm.
    MAXIMUM_PRESERVATION:  Maximize information preservation across the pipeline.
                           May sacrifice speed for fidelity.
    LOAD_BALANCED:         Distribute work evenly across paradigms.
                           Good for avoiding hot-spots in concurrent execution.
    """
    PARADIGM_SPECIALIST = "paradigm_specialist"
    MINIMUM_BRIDGING = "minimum_bridging"
    MAXIMUM_PRESERVATION = "maximum_preservation"
    LOAD_BALANCED = "load_balanced"


@dataclass
class CodeSegment:
    """A segment of code assigned to a specific paradigm.

    Attributes:
        lang: The target paradigm for this segment.
        types: The FluxTypes that belong in this segment.
        rationale: Why this paradigm was chosen for these types.
    """
    lang: str
    types: List[FluxType] = field(default_factory=list)
    rationale: str = ""

    def type_count(self) -> int:
        """Number of types in this segment."""
        return len(self.types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "type_count": self.type_count(),
            "types": [
                {"base_type": t.base_type.name, "confidence": round(t.confidence, 3),
                 "paradigm": t.paradigm_source}
                for t in self.types
            ],
            "rationale": self.rationale,
        }


@dataclass
class OptimizationPlan:
    """A complete optimization plan for cross-language execution.

    Attributes:
        strategy: The optimization strategy used.
        segments: Code segments assigned to different paradigms.
        estimated_speedup: Estimated speedup factor (>1.0 means faster).
        estimated_loss: Estimated information loss [0.0, 1.0].
        bridge_calls: Number of cross-paradigm bridge calls needed.
        witness_chain: Type witnesses proving each transformation.
    """
    strategy: OptimizationStrategy
    segments: List[CodeSegment] = field(default_factory=list)
    estimated_speedup: float = 1.0
    estimated_loss: float = 0.0
    bridge_calls: int = 0
    witness_chain: List[TypeWitness] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "segments": [s.to_dict() for s in self.segments],
            "estimated_speedup": round(self.estimated_speedup, 4),
            "estimated_loss": round(self.estimated_loss, 4),
            "bridge_calls": self.bridge_calls,
            "witness_count": len(self.witness_chain),
        }


# ══════════════════════════════════════════════════════════════════
# 4. CrossLanguageOptimizer — Program Splitting Engine
# ══════════════════════════════════════════════════════════════════

class CrossLanguageOptimizer:
    """Optimizes programs by splitting work across paradigms.

    Analyzes a set of FluxTypes and produces an OptimizationPlan that
    assigns segments to the best-suited paradigms based on the chosen
    strategy.

    The optimizer works at the TYPE level — it does not analyze source
    code directly.  Instead, it reasons about the base types and
    constraints to determine which paradigm can handle each type most
    efficiently.

    Usage:
        optimizer = CrossLanguageOptimizer()
        plan = optimizer.optimize(
            types=my_types,
            source_lang="zho",
            strategy=OptimizationStrategy.PARADIGM_SPECIALIST,
        )
        for seg in plan.segments:
            print(f"  {seg.lang}: {seg.type_count()} types - {seg.rationale}")
        print(f"  Speedup: {plan.estimated_speedup:.2f}x")
    """

    def __init__(
        self,
        profiler: Optional[ParadigmProfiler] = None,
        cost_matrix: Optional[BridgeCostMatrix] = None,
        algebra: Optional[TypeAlgebra] = None,
    ) -> None:
        self.profiler = profiler or ParadigmProfiler()
        self.cost_matrix = cost_matrix or BridgeCostMatrix(
            algebra or TypeAlgebra()
        )
        self.algebra = algebra or TypeAlgebra()

    def optimize(
        self,
        types: List[FluxType],
        source_lang: str,
        strategy: OptimizationStrategy,
    ) -> OptimizationPlan:
        """Optimize a program by splitting types across paradigms.

        Args:
            types: The FluxTypes to optimize.
            source_lang: The originating paradigm.
            strategy: Which optimization strategy to apply.

        Returns:
            An OptimizationPlan with segments, speedup estimate, and witnesses.
        """
        if not types:
            return OptimizationPlan(
                strategy=strategy,
                estimated_speedup=1.0,
                estimated_loss=0.0,
            )

        # Infer what operations the program needs
        op_scores = self.profiler.infer_operations(types)

        # Analyze distribution of types across paradigms
        dist = self.analyze_operation_distribution(types)

        # Generate segments based on strategy
        if strategy == OptimizationStrategy.PARADIGM_SPECIALIST:
            segments = self._strategy_paradigm_specialist(types, op_scores, source_lang)
        elif strategy == OptimizationStrategy.MINIMUM_BRIDGING:
            segments = self._strategy_minimum_bridging(types, source_lang)
        elif strategy == OptimizationStrategy.MAXIMUM_PRESERVATION:
            segments = self._strategy_maximum_preservation(types, source_lang)
        elif strategy == OptimizationStrategy.LOAD_BALANCED:
            segments = self._strategy_load_balanced(types, op_scores, source_lang)
        else:
            segments = [CodeSegment(lang=source_lang, types=types,
                                    rationale="fallback: keep all in source")]

        # Count bridge calls (transitions between different langs)
        bridge_calls = self._count_bridge_transitions(segments)

        # Estimate speedup and loss
        speedup = self._estimate_speedup(types, segments, source_lang)
        loss = self._estimate_information_loss(types, segments, source_lang)

        # Generate witnesses
        witnesses = self._generate_witnesses(types, segments, source_lang)

        return OptimizationPlan(
            strategy=strategy,
            segments=segments,
            estimated_speedup=speedup,
            estimated_loss=loss,
            bridge_calls=bridge_calls,
            witness_chain=witnesses,
        )

    def analyze_operation_distribution(
        self, types: List[FluxType]
    ) -> Dict[str, int]:
        """Analyze how many types map to each base-type operation category.

        Args:
            types: List of FluxTypes to analyze.

        Returns:
            Dict mapping operation name to count of types requiring it.
        """
        dist: Dict[str, int] = {}
        for ft in types:
            op = self.profiler._base_type_to_operation(ft.base_type)
            if op:
                dist[op] = dist.get(op, 0) + 1
            # Also count constraint-derived operations
            for c in ft.constraints:
                for cop, _ in self.profiler._constraint_to_operations(c.kind, c.language):
                    dist[cop] = dist.get(cop, 0) + 1
        return dist

    def suggest_paradigm_split(self, types: List[FluxType]) -> List[CodeSegment]:
        """Suggest a paradigm split using the PARADIGM_SPECIALIST strategy.

        This is a convenience method that uses the most aggressive
        optimization strategy.

        Args:
            types: List of FluxTypes to split.

        Returns:
            List of CodeSegments, one per target paradigm.
        """
        if not types:
            return []
        # Infer source from the most common paradigm
        lang_counts: Dict[str, int] = {}
        for ft in types:
            lang_counts[ft.paradigm_source] = lang_counts.get(ft.paradigm_source, 0) + 1
        source_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "flux"
        return self._strategy_paradigm_specialist(
            types, self.profiler.infer_operations(types), source_lang
        )

    # ── Strategy Implementations ─────────────────────────────────

    def _strategy_paradigm_specialist(
        self,
        types: List[FluxType],
        op_scores: Dict[str, float],
        source_lang: str,
    ) -> List[CodeSegment]:
        """Delegate each type to the best-suited paradigm for its operation."""
        segments_by_lang: Dict[str, List[FluxType]] = {}
        rationale_map: Dict[str, str] = {}

        for ft in types:
            best_lang = self._find_best_lang_for_type(ft, op_scores)
            segments_by_lang.setdefault(best_lang, []).append(ft)

            if best_lang not in rationale_map:
                profile = self.profiler.profile(best_lang)
                # Find the dominant operation
                ops = self.profiler.infer_operations([ft])
                dominant_op = max(ops, key=ops.get) if ops else "general"
                rationale_map[best_lang] = (
                    f"Best paradigm for {dominant_op} "
                    f"(suitability={profile.suitability(dominant_op):.2f})"
                )

        segments = []
        for lang, lang_types in segments_by_lang.items():
            segments.append(CodeSegment(
                lang=lang,
                types=lang_types,
                rationale=rationale_map.get(lang, "specialist delegation"),
            ))
        return segments

    def _strategy_minimum_bridging(
        self,
        types: List[FluxType],
        source_lang: str,
    ) -> List[CodeSegment]:
        """Minimize cross-language calls by keeping types in source paradigm."""
        # Check if source can handle all types adequately
        source_profile = self.profiler.profile(source_lang)
        op_scores = self.profiler.infer_operations(types)

        # Find any operations where source is very weak (< 0.2 suitability)
        weak_ops = []
        for op, score in op_scores.items():
            if score > 0.3 and source_profile.suitability(op) < 0.2:
                weak_ops.append((op, score))

        if not weak_ops:
            # Source paradigm handles everything well enough
            return [CodeSegment(
                lang=source_lang,
                types=types,
                rationale="Source paradigm handles all operations adequately "
                         f"(suitability >= 0.2 for all required ops)",
            )]

        # Only offload types that require weak operations
        source_types: List[FluxType] = []
        offload_types: List[FluxType] = []
        offload_lang: str = source_lang

        for ft in types:
            ft_ops = self.profiler.infer_operations([ft])
            needs_offload = False
            for op, weight in ft_ops.items():
                for weak_op, weak_score in weak_ops:
                    if op == weak_op and weight > 0.3:
                        needs_offload = True
                        # Find the best language for this weak op
                        best = self.profiler.optimal_lang_for(weak_op)[0]
                        offload_lang = best[0] if best else source_lang
                        break
                if needs_offload:
                    break

            if needs_offload:
                offload_types.append(ft)
            else:
                source_types.append(ft)

        segments = []
        if source_types:
            segments.append(CodeSegment(
                lang=source_lang,
                types=source_types,
                rationale="Kept in source paradigm to minimize bridging",
            ))
        if offload_types:
            best_profile = self.profiler.profile(offload_lang)
            segments.append(CodeSegment(
                lang=offload_lang,
                types=offload_types,
                rationale=f"Offloaded weak operations to {offload_lang} "
                         f"(minimizing bridge calls)",
            ))
        return segments

    def _strategy_maximum_preservation(
        self,
        types: List[FluxType],
        source_lang: str,
    ) -> List[CodeSegment]:
        """Maximize information preservation by choosing lowest-cost bridges."""
        # For each type, find the paradigm that preserves the most information
        segments_by_lang: Dict[str, List[FluxType]] = {}
        rationale_map: Dict[str, str] = {}

        for ft in types:
            best_lang = source_lang
            best_cost = self.cost_matrix.compute(source_lang, source_lang).total_cost

            for lang in SUPPORTED_LANGUAGES:
                if lang == source_lang:
                    cost = 0.0
                else:
                    report = self.cost_matrix.compute(source_lang, lang)
                    cost = report.total_cost
                    # Penalize information_loss more heavily
                    cost += report.information_loss * 0.5

                if cost < best_cost:
                    best_cost = cost
                    best_lang = lang

            segments_by_lang.setdefault(best_lang, []).append(ft)

            if best_lang not in rationale_map:
                if best_lang == source_lang:
                    rationale_map[best_lang] = (
                        "Lowest information loss: keep in source paradigm (zero bridge cost)"
                    )
                else:
                    bridge = self.cost_matrix.compute(source_lang, best_lang)
                    rationale_map[best_lang] = (
                        f"Lowest bridge cost from source "
                        f"(cost={bridge.total_cost:.3f}, "
                        f"info_loss={bridge.information_loss:.3f})"
                    )

        segments = []
        for lang, lang_types in segments_by_lang.items():
            segments.append(CodeSegment(
                lang=lang,
                types=lang_types,
                rationale=rationale_map.get(lang, "information preservation"),
            ))
        return segments

    def _strategy_load_balanced(
        self,
        types: List[FluxType],
        op_scores: Dict[str, float],
        source_lang: str,
    ) -> List[CodeSegment]:
        """Distribute types evenly across paradigms that can handle them."""
        # Score each language for the overall workload
        lang_scores: List[Tuple[str, float]] = []
        for lang in SUPPORTED_LANGUAGES:
            profile = self.profiler.profile(lang)
            total_suitability = 0.0
            for op, relevance in op_scores.items():
                total_suitability += profile.suitability(op) * relevance
            # Penalize bridge cost from source
            bridge_cost = self.cost_matrix.compute(source_lang, lang).total_cost
            # Prefer source lang slightly (no bridge needed)
            if lang == source_lang:
                total_suitability += 0.1
            adjusted_score = total_suitability * (1.0 - bridge_cost * 0.3)
            lang_scores.append((lang, adjusted_score))

        lang_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top languages and distribute types round-robin
        qualified = [(lang, score) for lang, score in lang_scores if score > 0.05]
        if not qualified:
            return [CodeSegment(lang=source_lang, types=types,
                                rationale="No qualified paradigms found")]

        # Use at most 3 paradigms for load balancing
        selected = qualified[:3]

        # Assign types to paradigms using round-robin weighted by suitability
        segments_by_lang: Dict[str, List[FluxType]] = {lang: [] for lang, _ in selected}
        rationale_map: Dict[str, str] = {}

        for i, ft in enumerate(types):
            # Round-robin among selected paradigms
            idx = i % len(selected)
            chosen_lang = selected[idx][0]
            segments_by_lang[chosen_lang].append(ft)

        for lang, _ in selected:
            profile = self.profiler.profile(lang)
            avg_suit = (
                sum(profile.suitability(op) * rel for op, rel in op_scores.items())
                / max(len(op_scores), 1)
            )
            bridge = self.cost_matrix.compute(source_lang, lang)
            rationale_map[lang] = (
                f"Load-balanced distribution "
                f"(avg_suitability={avg_suit:.2f}, bridge_cost={bridge.total_cost:.3f})"
            )

        segments = []
        for lang in [l for l, _ in selected]:
            if segments_by_lang[lang]:
                segments.append(CodeSegment(
                    lang=lang,
                    types=segments_by_lang[lang],
                    rationale=rationale_map.get(lang, "load balancing"),
                ))
        return segments

    # ── Helper Methods ───────────────────────────────────────────

    def _find_best_lang_for_type(
        self,
        ft: FluxType,
        op_scores: Dict[str, float],
    ) -> str:
        """Find the paradigm best suited for a single FluxType."""
        ft_ops = self.profiler.infer_operations([ft])
        if not ft_ops:
            return ft.paradigm_source

        # Weight each language by how well it handles the type's operations
        lang_scores: List[Tuple[str, float]] = []
        for lang in SUPPORTED_LANGUAGES:
            profile = self.profiler.profile(lang)
            score = 0.0
            for op, relevance in ft_ops.items():
                score += profile.suitability(op) * relevance
            # Small preference for the type's native paradigm
            if lang == ft.paradigm_source:
                score += 0.05
            lang_scores.append((lang, score))

        lang_scores.sort(key=lambda x: x[1], reverse=True)
        return lang_scores[0][0]

    def _count_bridge_transitions(self, segments: List[CodeSegment]) -> int:
        """Count the number of cross-paradigm bridge calls needed."""
        if len(segments) <= 1:
            return 0
        # Each segment boundary requires a bridge call
        return len(segments) - 1

    def _estimate_speedup(
        self,
        types: List[FluxType],
        segments: List[CodeSegment],
        source_lang: str,
    ) -> float:
        """Estimate the speedup factor from paradigm splitting.

        Compares the estimated execution time with all types in the source
        paradigm vs. the optimized distribution.
        """
        if not types or not segments:
            return 1.0

        op_scores = self.profiler.infer_operations(types)
        source_profile = self.profiler.profile(source_lang)

        # Estimate baseline time: all in source paradigm
        baseline_time = 0.0
        for op, relevance in op_scores.items():
            lat = source_profile.latency_estimate.get(op, 1.0)
            baseline_time += lat * relevance * len(types)

        # Estimate optimized time
        optimized_time = 0.0
        for seg in segments:
            seg_profile = self.profiler.profile(seg.lang)
            seg_ops = self.profiler.infer_operations(seg.types)
            for op, relevance in seg_ops.items():
                lat = seg_profile.latency_estimate.get(op, 1.0)
                optimized_time += lat * relevance * len(seg.types)

        # Add bridge overhead
        bridge_overhead = len(segments) * 0.1  # Each bridge ~10% of a type operation
        optimized_time += bridge_overhead

        if optimized_time <= 0:
            return 1.0

        speedup = baseline_time / optimized_time
        # Clamp to reasonable range [0.5, 5.0]
        return max(0.5, min(5.0, speedup))

    def _estimate_information_loss(
        self,
        types: List[FluxType],
        segments: List[CodeSegment],
        source_lang: str,
    ) -> float:
        """Estimate total information loss from paradigm splitting."""
        if not types:
            return 0.0

        total_loss = 0.0
        for seg in segments:
            if seg.lang == source_lang:
                continue
            report = self.cost_matrix.compute(source_lang, seg.lang)
            type_weight = len(seg.types) / len(types)
            total_loss += report.information_loss * type_weight

        return min(total_loss, 1.0)

    def _generate_witnesses(
        self,
        types: List[FluxType],
        segments: List[CodeSegment],
        source_lang: str,
    ) -> List[TypeWitness]:
        """Generate TypeWitnesses for each cross-paradigm transformation."""
        witnesses: List[TypeWitness] = []

        for seg in segments:
            if seg.lang == source_lang:
                continue

            for ft in seg.types:
                # Find equivalence class if possible
                source_tag = self._extract_native_tag(ft)
                equiv_class = None
                target_tag = ""

                if source_tag:
                    equiv_class = self.algebra.find_class(source_lang, source_tag)
                    if equiv_class:
                        target_slot = equiv_class.get_slot(seg.lang)
                        if target_slot:
                            target_tag = target_slot.native_tag

                preservation = PreservationDegree.DEGRADED
                if equiv_class:
                    preservation = equiv_class.degree

                witness = TypeWitness(
                    witness_id=str(uuid.uuid4())[:8],
                    source_lang=source_lang,
                    source_tag=source_tag,
                    target_lang=seg.lang,
                    target_tag=target_tag,
                    source_type=ft,
                    preservation=preservation,
                    equivalence_class_id=equiv_class.class_id if equiv_class else "",
                    constraints=[
                        WitnessConstraint(
                            name="base_type_preserved",
                            expected=ft.base_type.name,
                            actual=ft.base_type.name,
                            satisfied=True,
                        ),
                        WitnessConstraint(
                            name="paradigm_valid",
                            expected=seg.lang,
                            actual=seg.lang,
                            satisfied=True,
                        ),
                    ],
                )
                witnesses.append(witness)

        return witnesses

    def _extract_native_tag(self, ft: FluxType) -> str:
        """Extract the native tag from a FluxType's constraints."""
        for c in ft.constraints:
            if c.language == ft.paradigm_source:
                return c.value
        return ""


# ══════════════════════════════════════════════════════════════════
# 5. BridgeOptimizer — Bridge Path Optimization & Caching
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BridgeCacheKey:
    """Key for the bridge result cache."""
    source: str
    target: str
    type_fingerprint: str


class BridgeOptimizer:
    """Optimizes bridge paths between paradigms.

    Provides three optimization capabilities:

    1. **Path optimization**: Find the cheapest route between two paradigms,
       potentially using intermediate languages as stepping stones.

    2. **Information loss minimization**: Find routes that preserve the most
       type information, possibly at the cost of higher latency.

    3. **Result caching**: Cache bridge results to avoid redundant computation
       when the same types are bridged repeatedly.

    Usage:
        optimizer = BridgeOptimizer()
        path = optimizer.optimize_bridge_sequence(["zho", "san", "lat"])
        # → ["zho", "deu", "lat"] if that's cheaper

        route, loss = optimizer.minimize_information_loss("zho", "lat")
        # → (["zho", "san", "lat"], 0.15)  # SAN preserves scope info

        optimizer.cache_bridge_result("zho", "deu", types, result)
        cached = optimizer.get_cached("zho", "deu")
    """

    def __init__(
        self,
        cost_matrix: Optional[BridgeCostMatrix] = None,
        algebra: Optional[TypeAlgebra] = None,
    ) -> None:
        self.cost_matrix = cost_matrix or BridgeCostMatrix(
            algebra or TypeAlgebra()
        )
        self._cache: Dict[BridgeCacheKey, CompilationResult] = {}

    def optimize_bridge_sequence(self, path: List[str]) -> List[str]:
        """Optimize a bridge path to minimize total cost.

        Uses a modified Dijkstra algorithm to find the cheapest path
        from the first to the last language in the sequence, considering
        all intermediate nodes.

        Args:
            path: The original bridge path (list of language tags).

        Returns:
            The optimized path.  May be shorter or use different intermediates.
        """
        if len(path) <= 2:
            return path

        source = path[0]
        target = path[-1]

        if source == target:
            return [source]

        # Use Dijkstra to find cheapest path from source to target
        # through the graph of all 6 paradigms
        return self._dijkstra_shortest_path(source, target)

    def minimize_information_loss(
        self,
        source: str,
        target: str,
        via: Optional[List[str]] = None,
    ) -> Tuple[List[str], float]:
        """Find the route from source to target with minimum information loss.

        Unlike optimize_bridge_sequence which minimizes total cost, this
        specifically minimizes information_loss even if total_cost is higher.

        Args:
            source: Source paradigm.
            target: Target paradigm.
            via: Optional intermediate nodes to prefer (soft constraint).

        Returns:
            Tuple of (optimized_path, total_information_loss).
        """
        if source == target:
            return [source], 0.0

        # Use modified Dijkstra minimizing information_loss instead of total_cost
        distances: Dict[str, float] = {lang: float('inf') for lang in SUPPORTED_LANGUAGES}
        distances[source] = 0.0
        previous: Dict[str, str] = {}
        visited: Set[str] = set()

        # Priority queue: (info_loss, language)
        pq: List[Tuple[float, str]] = [(0.0, source)]

        while pq:
            current_loss, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == target:
                break

            for neighbor in SUPPORTED_LANGUAGES:
                if neighbor in visited:
                    continue
                report = self.cost_matrix.compute(current, neighbor)
                # Use information_loss as the edge weight
                edge_loss = report.information_loss
                new_loss = current_loss + edge_loss

                # Soft preference for 'via' nodes (slight penalty for not using them)
                if via and neighbor in via:
                    new_loss *= 0.95  # 5% bonus for preferred intermediates

                if new_loss < distances[neighbor]:
                    distances[neighbor] = new_loss
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_loss, neighbor))

        # Reconstruct path
        if distances[target] == float('inf'):
            # No path found, return direct route
            report = self.cost_matrix.compute(source, target)
            return [source, target], report.information_loss

        path = self._reconstruct_path(previous, source, target)
        return path, distances[target]

    def cache_bridge_result(
        self,
        src: str,
        tgt: str,
        types: List[FluxType],
        result: CompilationResult,
    ) -> None:
        """Cache a bridge result for later reuse.

        Args:
            src: Source language.
            tgt: Target language.
            types: The FluxTypes that were bridged.
            result: The compilation result to cache.
        """
        fingerprint = self._compute_type_fingerprint(types)
        key = BridgeCacheKey(source=src, target=tgt, type_fingerprint=fingerprint)
        self._cache[key] = result

    def get_cached(
        self, src: str, tgt: str, types: Optional[List[FluxType]] = None
    ) -> Optional[CompilationResult]:
        """Retrieve a cached bridge result.

        Args:
            src: Source language.
            tgt: Target language.
            types: If provided, match against type fingerprint.  If None,
                   return any cached result for the (src, tgt) pair.

        Returns:
            The cached CompilationResult, or None if not found.
        """
        if types is not None:
            fingerprint = self._compute_type_fingerprint(types)
            key = BridgeCacheKey(source=src, target=tgt, type_fingerprint=fingerprint)
            return self._cache.get(key)

        # Return the most recent cache entry for this pair
        matches = [
            self._cache[k] for k in self._cache
            if k.source == src and k.target == tgt
        ]
        return matches[-1] if matches else None

    def clear_cache(self) -> None:
        """Clear all cached bridge results."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Return the number of cached bridge results."""
        return len(self._cache)

    def find_cheaper_intermediate(
        self, source: str, target: str
    ) -> Optional[Tuple[str, float]]:
        """Check if routing through an intermediate language is cheaper.

        Args:
            source: Source paradigm.
            target: Target paradigm.

        Returns:
            (intermediate_lang, savings) if a cheaper route exists, else None.
        """
        direct_cost = self.cost_matrix.compute(source, target).total_cost

        best_intermediate: Optional[str] = None
        best_savings = 0.0

        for intermediate in SUPPORTED_LANGUAGES:
            if intermediate == source or intermediate == target:
                continue

            cost1 = self.cost_matrix.compute(source, intermediate).total_cost
            cost2 = self.cost_matrix.compute(intermediate, target).total_cost
            indirect_cost = cost1 + cost2

            savings = direct_cost - indirect_cost
            if savings > best_savings and savings > 0.01:
                best_savings = savings
                best_intermediate = intermediate

        if best_intermediate:
            return best_intermediate, best_savings
        return None

    # ── Private Methods ──────────────────────────────────────────

    def _dijkstra_shortest_path(self, source: str, target: str) -> List[str]:
        """Find cheapest path using Dijkstra's algorithm on total_cost."""
        distances: Dict[str, float] = {lang: float('inf') for lang in SUPPORTED_LANGUAGES}
        distances[source] = 0.0
        previous: Dict[str, str] = {}
        visited: Set[str] = set()

        pq: List[Tuple[float, str]] = [(0.0, source)]

        while pq:
            current_cost, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == target:
                break

            for neighbor in SUPPORTED_LANGUAGES:
                if neighbor in visited:
                    continue
                report = self.cost_matrix.compute(current, neighbor)
                new_cost = current_cost + report.total_cost

                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

        if distances[target] == float('inf'):
            return [source, target]  # Fallback to direct path

        return self._reconstruct_path(previous, source, target)

    def _reconstruct_path(
        self, previous: Dict[str, str], source: str, target: str
    ) -> List[str]:
        """Reconstruct a path from Dijkstra's previous map."""
        path: List[str] = []
        current = target
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(source)
        path.reverse()
        return path

    def _compute_type_fingerprint(self, types: List[FluxType]) -> str:
        """Compute a lightweight fingerprint for a list of FluxTypes.

        Used as cache key to avoid redundant bridge computations.
        """
        if not types:
            return ""

        parts: List[str] = []
        for ft in types:
            parts.append(f"{ft.base_type.name}:{ft.confidence:.2f}:{ft.paradigm_source}")
            for c in ft.constraints:
                parts.append(f"{c.kind.value}={c.value}")
        return "|".join(parts)
