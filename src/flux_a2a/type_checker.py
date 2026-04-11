"""
Universal Type Checker, Type Bridge, and Type Compatibility for FUTS.

This module implements the cross-language type checking and translation
machinery for the Flux Universal Type System (FUTS).  It provides:

  - TypeCompatibility: measures how compatible two FluxTypes are [0.0, 1.0]
  - UniversalTypeChecker: checks types across language boundaries
  - TypeBridge: translates types between language paradigms

The checker does NOT enforce binary pass/fail type checking.  Instead, it
produces a compatibility score and a list of warnings/recommendations.
This is "gradual typing" taken to its logical conclusion: types are not
enforced, they are ADVISORY, and the compatibility score tells you how
much trust to place in a cross-paradigm type assignment.

Design Decisions (from Round 10 research):
  - Gradual typing → compatibility scores instead of pass/fail
  - Row polymorphism → extensible constraint rows
  - Dependent types → CONTEXTUAL types resolved at runtime
  - Algebraic effects → MODAL types carry effect sets
  - Session types → CAPABILITY types encode channel protocols
  - Rust traits → SCOPE types work like trait bounds on visibility
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from flux_a2a.types import (
    ConstraintKind,
    FluxBaseType,
    FluxConstraint,
    FluxType,
    FluxTypeRegistry,
    FluxTypeSignature,
    QuantumTypeState,
    _PARADIGM_TO_BASE,
    build_default_registry,
)


# ══════════════════════════════════════════════════════════════════
# 1. TypeCompatibility — Measuring Cross-Language Type Compatibility
# ══════════════════════════════════════════════════════════════════

class CompatibilityLevel(str, Enum):
    """Categorical compatibility levels for human readability."""
    IDENTICAL = "identical"        # Same type, same constraints
    COMPATIBLE = "compatible"      # Semantically equivalent, different surface
    CONVERTIBLE = "convertible"    # Lossy conversion possible
    WEAKLY_COMPATIBLE = "weakly_compatible"  # Partial overlap
    INCOMPATIBLE = "incompatible"  # No meaningful correspondence
    CONTRADICTORY = "contradictory"  # Explicitly contradictory


@dataclass
class CompatibilityReport:
    """Detailed compatibility analysis between two FluxTypes.

    Attributes:
        score: Overall compatibility [0.0, 1.0].
        level: Categorical compatibility level.
        warnings: List of human-readable warnings.
        suggestions: List of suggestions for improving compatibility.
        base_compatibility: Compatibility of base types alone.
        constraint_compatibility: Compatibility of constraint sets.
        confidence_factor: How confidence scores affect overall compatibility.
    """
    score: float = 0.0
    level: CompatibilityLevel = CompatibilityLevel.INCOMPATIBLE
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    base_compatibility: float = 0.0
    constraint_compatibility: float = 0.0
    confidence_factor: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "level": self.level.value,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "base_compatibility": round(self.base_compatibility, 6),
            "constraint_compatibility": round(self.constraint_compatibility, 6),
            "confidence_factor": round(self.confidence_factor, 6),
        }


class TypeCompatibility:
    """Measures compatibility between two FluxTypes.

    Compatibility is NOT binary.  It's a real-valued score in [0.0, 1.0]
    that accounts for:
      1. Base type alignment (spectrum distance)
      2. Constraint overlap (shared constraints increase score)
      3. Confidence agreement (similar confidence → higher score)
      4. Paradigm distance (closer paradigms → easier bridging)
      5. Quantum state alignment (for UNCERTAIN types)

    The score is computed as a weighted product:
        score = base_w * constraint_w * confidence_w * paradigm_w

    Where each factor is in [0.0, 1.0].
    """

    def __init__(
        self,
        base_weight: float = 0.35,
        constraint_weight: float = 0.30,
        confidence_weight: float = 0.15,
        paradigm_weight: float = 0.20,
    ) -> None:
        self.base_weight = base_weight
        self.constraint_weight = constraint_weight
        self.confidence_weight = confidence_weight
        self.paradigm_weight = paradigm_weight

    def compute(self, a: FluxType, b: FluxType) -> CompatibilityReport:
        """Compute full compatibility report between two FluxTypes.

        Args:
            a: First FluxType.
            b: Second FluxType.

        Returns:
            A CompatibilityReport with score, level, warnings, suggestions.
        """
        # 1. Base type compatibility
        base_compat = self._base_compatibility(a, b)

        # 2. Constraint compatibility
        constraint_compat = self._constraint_compatibility(a, b)

        # 3. Confidence agreement
        confidence_compat = self._confidence_compatibility(a, b)

        # 4. Paradigm distance
        paradigm_compat = self._paradigm_compatibility(a, b)

        # Weighted product (geometric mean with weights)
        raw_score = (
            (base_compat ** self.base_weight) *
            (constraint_compat ** self.constraint_weight) *
            (confidence_compat ** self.confidence_weight) *
            (paradigm_compat ** self.paradigm_weight)
        )

        # Adjust for quantum state
        quantum_adjust = self._quantum_adjustment(a, b)
        score = min(1.0, raw_score * quantum_adjust)

        # Determine level
        level = self._classify_level(score, base_compat, constraint_compat)

        # Generate warnings and suggestions
        warnings, suggestions = self._generate_feedback(a, b, score, level)

        return CompatibilityReport(
            score=score,
            level=level,
            warnings=warnings,
            suggestions=suggestions,
            base_compatibility=base_compat,
            constraint_compatibility=constraint_compat,
            confidence_factor=confidence_compat,
        )

    def _base_compatibility(self, a: FluxType, b: FluxType) -> float:
        """Compute base type compatibility [0.0, 1.0].

        Uses spectrum distance: identical types → 1.0, antipodal → 0.0.
        """
        a_eff = a.effective_base_type()
        b_eff = b.effective_base_type()

        if a_eff == b_eff:
            return 1.0

        dist = a_eff.spectrum_distance(b_eff)
        max_dist = float(FluxBaseType.CONTEXTUAL)

        # Linear falloff with a minimum floor
        compat = 1.0 - (dist / max_dist)
        return max(0.0, compat)

    def _constraint_compatibility(self, a: FluxType, b: FluxType) -> float:
        """Compute constraint set compatibility [0.0, 1.0].

        For each constraint in the SMALLER set, find the best match in
        the LARGER set.  Score is the average of best matches.
        """
        if not a.constraints and not b.constraints:
            return 1.0  # No constraints = fully compatible
        if not a.constraints or not b.constraints:
            return 0.7  # One-sided constraints = moderate compatibility

        # Use the smaller set as reference
        if len(a.constraints) <= len(b.constraints):
            source, target = a.constraints, b.constraints
        else:
            source, target = b.constraints, a.constraints

        if not source:
            return 0.8

        scores = []
        for sc in source:
            best = max((sc.compatible_with(tc) for tc in target), default=0.0)
            scores.append(best)

        return sum(scores) / len(scores) if scores else 0.5

    def _confidence_compatibility(self, a: FluxType, b: FluxType) -> float:
        """Compute confidence agreement [0.0, 1.0].

        Types with similar confidence are more compatible.
        Two low-confidence types can still be compatible (both uncertain).
        """
        diff = abs(a.confidence - b.confidence)
        # Max diff is 1.0 (0.0 vs 1.0)
        return 1.0 - diff

    def _paradigm_compatibility(self, a: FluxType, b: FluxType) -> float:
        """Compute paradigm-based compatibility [0.0, 1.0].

        Same paradigm → 1.0.  Known bridge distances for different paradigms.
        Unknown paradigms get a default of 0.6.
        """
        if a.paradigm_source == b.paradigm_source:
            return 1.0

        # Paradigm bridge distances (from Round 4-6 simulations)
        _PARADIGM_DISTANCES: Dict[Tuple[str, str], float] = {
            ("deu", "lat"): 0.041, ("lat", "deu"): 0.041,
            ("san", "lat"): 0.077, ("lat", "san"): 0.077,
            ("kor", "san"): 0.099, ("san", "kor"): 0.099,
            ("zho", "san"): 0.104, ("san", "zho"): 0.104,
            ("kor", "deu"): 0.122, ("deu", "kor"): 0.122,
            ("wen", "lat"): 0.124, ("lat", "wen"): 0.124,
            ("zho", "deu"): 0.154, ("deu", "zho"): 0.154,
            ("wen", "zho"): 0.156, ("zho", "wen"): 0.156,
            ("zho", "lat"): 0.163, ("lat", "zho"): 0.163,
            ("wen", "san"): 0.180, ("san", "wen"): 0.180,
            ("zho", "kor"): 0.202, ("kor", "zho"): 0.202,
            ("wen", "kor"): 0.207, ("kor", "wen"): 0.207,
            ("deu", "wen"): 0.231, ("wen", "deu"): 0.231,
        }

        max_distance = 0.231  # deu-wen (hardest bridge)
        dist = _PARADIGM_DISTANCES.get(
            (a.paradigm_source, b.paradigm_source), 0.15
        )
        return max(0.0, 1.0 - dist / max_distance)

    def _quantum_adjustment(self, a: FluxType, b: FluxType) -> float:
        """Adjust compatibility for quantum superposition states.

        If both types are in superposition, check if their possibility
        sets overlap.  Overlap increases compatibility.
        """
        a_q = a.quantum_state
        b_q = b.quantum_state

        if not a_q and not b_q:
            return 1.0

        if a_q and b_q:
            a_types = {bt for bt, _ in a_q.possibilities}
            b_types = {bt for bt, _ in b_q.possibilities}
            if a_types & b_types:
                return 1.2  # Boost for shared possibilities
            return 0.8  # Penalty for non-overlapping superpositions

        # One is quantum, one is definite
        definite_type = a.effective_base_type() if not a_q else b.effective_base_type()
        quantum = b_q if a_q else a_q
        if quantum:
            for bt, amp in quantum.possibilities:
                if bt == definite_type:
                    return 1.0 + amp * 0.2  # Small boost
        return 0.9

    def _classify_level(
        self,
        score: float,
        base_compat: float,
        constraint_compat: float,
    ) -> CompatibilityLevel:
        """Map score to categorical compatibility level."""
        if score >= 0.95 and base_compat == 1.0 and constraint_compat >= 0.9:
            return CompatibilityLevel.IDENTICAL
        if score >= 0.8:
            return CompatibilityLevel.COMPATIBLE
        if score >= 0.6:
            return CompatibilityLevel.CONVERTIBLE
        if score >= 0.35:
            return CompatibilityLevel.WEAKLY_COMPATIBLE
        if score >= 0.15:
            return CompatibilityLevel.INCOMPATIBLE
        return CompatibilityLevel.CONTRADICTORY

    def _generate_feedback(
        self,
        a: FluxType,
        b: FluxType,
        score: float,
        level: CompatibilityLevel,
    ) -> Tuple[List[str], List[str]]:
        """Generate human-readable warnings and suggestions."""
        warnings: List[str] = []
        suggestions: List[str] = []

        # Base type warnings
        if a.effective_base_type() != b.effective_base_type():
            warnings.append(
                f"Base type mismatch: {a.effective_base_type().name} vs "
                f"{b.effective_base_type().name} "
                f"(spectrum distance: {a.effective_base_type().spectrum_distance(b.effective_base_type()):.0f})"
            )

        # Confidence warnings
        if abs(a.confidence - b.confidence) > 0.3:
            low = a if a.confidence < b.confidence else b
            warnings.append(
                f"Confidence gap: {a.confidence:.2f} vs {b.confidence:.2f} "
                f"({low.paradigm_source} type is uncertain)"
            )

        # Paradigm warnings
        if a.paradigm_source != b.paradigm_source:
            warnings.append(
                f"Cross-paradigm type check: {a.paradigm_source} ↔ {b.paradigm_source}"
            )

        # Quantum state warnings
        if a.quantum_state and not a.quantum_state.collapsed:
            warnings.append(
                f"Type {a.name or a.id} is in quantum superposition "
                f"(entropy={a.quantum_state.entropy():.2f})"
            )

        # Contextual type warnings
        if a.base_type == FluxBaseType.CONTEXTUAL or b.base_type == FluxBaseType.CONTEXTUAL:
            warnings.append(
                "Context-dependent type: compatibility may change at runtime"
            )

        # Suggestions for low compatibility
        if score < 0.5:
            suggestions.append(
                "Consider using TypeBridge to convert between paradigms"
            )
            if a.paradigm_source != b.paradigm_source:
                suggestions.append(
                    f"Route through Latin (lat) hub for {a.paradigm_source} ↔ "
                    f"{b.paradigm_source} bridging"
                )

        if a.base_type == FluxBaseType.CONTEXTUAL:
            suggestions.append(
                "Use runtime context resolution before type checking"
            )

        if score >= 0.6 and score < 0.8:
            suggestions.append(
                f"Acceptable compatibility ({score:.2f}); verify semantics "
                f"are preserved in the target paradigm"
            )

        return warnings, suggestions


# ══════════════════════════════════════════════════════════════════
# 2. TypeBridge — Cross-Paradigm Type Translation
# ══════════════════════════════════════════════════════════════════

class BridgeStrategy(str, Enum):
    """Strategies for bridging types between paradigms."""
    DIRECT = "direct"                # Map directly (same base type)
    VIA_HUB = "via_hub"              # Route through Latin hub
    CONSTRAINT_PRESERVATION = "constraint_preservation"  # Keep all constraints
    CONSTRAINT_STRIPPING = "constraint_stripping"        # Drop language-specific
    UPCAST = "upcast"                # Widen to more general type
    DOWNCAST = "downcast"            # Narrow to more specific type
    QUANTUM_DEFER = "quantum_defer"  # Create superposition of candidates


@dataclass
class BridgeResult:
    """Result of a type bridge translation.

    Attributes:
        target_type: The translated FluxType in the target paradigm.
        strategy: The strategy used for bridging.
        fidelity: How faithfully the source semantics are preserved [0.0, 1.0].
        warnings: Any warnings about information loss.
        cost: The "bridge cost" (higher = more expensive translation).
    """
    target_type: FluxType
    strategy: BridgeStrategy
    fidelity: float = 1.0
    warnings: List[str] = field(default_factory=list)
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type.to_dict(),
            "strategy": self.strategy.value,
            "fidelity": round(self.fidelity, 6),
            "warnings": self.warnings,
            "cost": round(self.cost, 6),
        }


class TypeBridge:
    """Translates FluxTypes between language paradigms.

    The bridge supports multiple strategies:
      - DIRECT: when both paradigms share the same base type
      - VIA_HUB: route through the Latin (lat) hub paradigm
      - CONSTRAINT_PRESERVATION: keep source constraints in target
      - CONSTRAINT_STRIPPING: remove source-specific constraints
      - UPCAST/DOWNCAST: widen or narrow the type
      - QUANTUM_DEFER: create superposition when uncertain

    Usage:
        bridge = TypeBridge(registry)
        result = bridge.translate(source_type, target_lang="deu")
        print(result.target_type, result.fidelity)
    """

    # Known cross-paradigm base type mappings
    # Maps (source_lang, source_tag) → (target_lang, target_tag)
    _CROSS_MAPPINGS: Dict[Tuple[str, str], Tuple[str, str]] = {
        # German gender ↔ Latin gender
        ("deu", "maskulinum"): ("lat", "maskulinum"),
        ("deu", "femininum"):  ("lat", "femininum"),
        ("deu", "neutrum"):    ("lat", "neutrum"),
        ("lat", "maskulinum"): ("deu", "maskulinum"),
        ("lat", "femininum"):  ("deu", "femininum"),
        ("lat", "neutrum"):    ("deu", "neutrum"),
        # Sanskrit gender ↔ Latin gender
        ("san", "linga_pushkara"): ("lat", "maskulinum"),
        ("san", "linga_stri"):     ("lat", "femininum"),
        ("san", "linga_napumsaka"):("lat", "neutrum"),
        # German case ↔ Sanskrit case (approximate)
        ("deu", "nominativ"):  ("san", "prathama"),
        ("deu", "akkusativ"):  ("san", "dvitiya"),
        ("deu", "dativ"):      ("san", "chaturthi"),
        ("deu", "genitiv"):    ("san", "shashthi"),
        # Chinese classifiers → German gender (shape-based heuristics)
        ("zho", "person"):     ("deu", "maskulinum"),   # active
        ("zho", "animal"):     ("deu", "maskulinum"),   # active
        ("zho", "machine"):    ("deu", "maskulinum"),   # active
        ("zho", "collective"): ("deu", "femininum"),    # container
        ("zho", "pair"):       ("deu", "femininum"),    # container
        ("zho", "volume"):     ("deu", "femininum"),    # container
        ("zho", "flat_object"):("deu", "neutrum"),      # value
        ("zho", "long_flexible"):("deu", "neutrum"),    # value
        ("zho", "small_round"):("deu", "neutrum"),      # value
        ("zho", "generic"):    ("deu", "neutrum"),      # value (default)
    }

    def __init__(
        self,
        registry: Optional[FluxTypeRegistry] = None,
        hub_paradigm: str = "lat",
    ) -> None:
        self.registry = registry or build_default_registry()
        self.hub_paradigm = hub_paradigm
        self._compatibility = TypeCompatibility()

    def translate(
        self,
        source: FluxType,
        target_lang: str,
        strategy: Optional[BridgeStrategy] = None,
    ) -> BridgeResult:
        """Translate a FluxType to the target paradigm.

        Args:
            source: The source FluxType.
            target_lang: The target language paradigm.
            strategy: Bridge strategy (auto-detected if None).

        Returns:
            A BridgeResult with the translated type and metadata.
        """
        # Same paradigm → identity
        if source.paradigm_source == target_lang:
            return BridgeResult(
                target_type=source,
                strategy=BridgeStrategy.DIRECT,
                fidelity=1.0,
                cost=0.0,
            )

        # Auto-detect strategy
        if strategy is None:
            strategy = self._detect_strategy(source, target_lang)

        return self._execute_bridge(source, target_lang, strategy)

    def translate_native(
        self,
        source_lang: str,
        native_tag: str,
        target_lang: str,
        confidence: float = 1.0,
    ) -> BridgeResult:
        """Convenience: translate a native language tag to another paradigm.

        Args:
            source_lang: Source paradigm.
            native_tag: Native type tag.
            target_lang: Target paradigm.
            confidence: Initial confidence.

        Returns:
            A BridgeResult.
        """
        source_type = FluxType.from_paradigm(
            source_lang, native_tag, confidence=confidence
        )
        return self.translate(source_type, target_lang)

    def _detect_strategy(
        self, source: FluxType, target_lang: str
    ) -> BridgeStrategy:
        """Auto-detect the best bridging strategy."""
        # Check for direct mapping
        for c in source.constraints:
            key = (c.language, c.value)
            if key in self._CROSS_MAPPINGS:
                target_key = self._CROSS_MAPPINGS[key]
                if target_key[0] == target_lang:
                    return BridgeStrategy.DIRECT

        # Check if both share the same base type
        target_tags = _PARADIGM_TO_BASE.get(target_lang, {})
        for c in source.constraints:
            if c.value in target_tags:
                if target_tags[c.value] == source.base_type:
                    return BridgeStrategy.CONSTRAINT_PRESERVATION

        # If source is UNCERTAIN or CONTEXTUAL, defer
        if source.base_type in (FluxBaseType.UNCERTAIN, FluxBaseType.CONTEXTUAL):
            return BridgeStrategy.QUANTUM_DEFER

        # Default: via hub
        return BridgeStrategy.VIA_HUB

    def _execute_bridge(
        self,
        source: FluxType,
        target_lang: str,
        strategy: BridgeStrategy,
    ) -> BridgeResult:
        """Execute a specific bridging strategy."""
        warnings: List[str] = []

        if strategy == BridgeStrategy.DIRECT:
            return self._bridge_direct(source, target_lang)

        elif strategy == BridgeStrategy.VIA_HUB:
            return self._bridge_via_hub(source, target_lang)

        elif strategy == BridgeStrategy.CONSTRAINT_PRESERVATION:
            return self._bridge_preserve(source, target_lang)

        elif strategy == BridgeStrategy.CONSTRAINT_STRIPPING:
            return self._bridge_strip(source, target_lang)

        elif strategy == BridgeStrategy.UPCAST:
            return self._bridge_upcast(source, target_lang, warnings)

        elif strategy == BridgeStrategy.QUANTUM_DEFER:
            return self._bridge_quantum_defer(source, target_lang)

        else:
            # Fallback to via_hub
            return self._bridge_via_hub(source, target_lang)

    def _bridge_direct(
        self, source: FluxType, target_lang: str
    ) -> BridgeResult:
        """Direct cross-mapping via known translation table."""
        for c in source.constraints:
            key = (c.language, c.value)
            if key in self._CROSS_MAPPINGS:
                tgt_lang, tgt_tag = self._CROSS_MAPPINGS[key]
                if tgt_lang == target_lang:
                    target_type = FluxType.from_paradigm(
                        tgt_lang, tgt_tag,
                        confidence=source.confidence,
                        name=f"{target_lang}:{tgt_tag}",
                    )
                    return BridgeResult(
                        target_type=target_type,
                        strategy=BridgeStrategy.DIRECT,
                        fidelity=0.9,
                        cost=0.1,
                    )

        # No direct mapping found — fall back
        return self._bridge_via_hub(source, target_lang)

    def _bridge_via_hub(
        self, source: FluxType, target_lang: str
    ) -> BridgeResult:
        """Route through the hub paradigm (default: Latin)."""
        # Source → Hub
        hub_result = self._bridge_to_hub(source)
        # Hub → Target
        if hub_result.target_type.paradigm_source == target_lang:
            # Hub IS the target
            total_cost = hub_result.cost
            total_fidelity = hub_result.fidelity
        else:
            target_result = self._bridge_from_hub(
                hub_result.target_type, target_lang
            )
            total_cost = hub_result.cost + target_result.cost
            total_fidelity = hub_result.fidelity * target_result.fidelity
            hub_result = target_result

        return BridgeResult(
            target_type=hub_result.target_type,
            strategy=BridgeStrategy.VIA_HUB,
            fidelity=total_fidelity,
            warnings=[f"Routed via {self.hub_paradigm} hub"],
            cost=total_cost,
        )

    def _bridge_to_hub(self, source: FluxType) -> BridgeResult:
        """Bridge from source paradigm to hub paradigm."""
        if source.paradigm_source == self.hub_paradigm:
            return BridgeResult(
                target_type=source,
                strategy=BridgeStrategy.DIRECT,
                fidelity=1.0,
                cost=0.0,
            )

        # Try direct mapping to hub
        for c in source.constraints:
            key = (c.language, c.value)
            if key in self._CROSS_MAPPINGS:
                tgt_lang, tgt_tag = self._CROSS_MAPPINGS[key]
                if tgt_lang == self.hub_paradigm:
                    hub_type = FluxType.from_paradigm(
                        tgt_lang, tgt_tag,
                        confidence=source.confidence * 0.95,
                    )
                    return BridgeResult(
                        target_type=hub_type,
                        strategy=BridgeStrategy.DIRECT,
                        fidelity=0.95,
                        cost=0.05,
                    )

        # Fall back to base type mapping
        hub_tags = _PARADIGM_TO_BASE.get(self.hub_paradigm, {})
        for tag, base in hub_tags.items():
            if base == source.base_type:
                hub_type = FluxType.from_paradigm(
                    self.hub_paradigm, tag,
                    confidence=source.confidence * 0.8,
                )
                return BridgeResult(
                    target_type=hub_type,
                    strategy=BridgeStrategy.UPCAST,
                    fidelity=0.8,
                    cost=0.15,
                    warnings=["No direct mapping; used base type"],
                )

        # Last resort: generic
        hub_type = FluxType(
            base_type=source.base_type,
            confidence=source.confidence * 0.7,
            paradigm_source=self.hub_paradigm,
            name=f"bridged:{source.name}",
        )
        return BridgeResult(
            target_type=hub_type,
            strategy=BridgeStrategy.CONSTRAINT_STRIPPING,
            fidelity=0.7,
            cost=0.25,
            warnings=["No base type match in hub; stripped constraints"],
        )

    def _bridge_from_hub(
        self, hub_type: FluxType, target_lang: str
    ) -> BridgeResult:
        """Bridge from hub paradigm to target paradigm."""
        # Try reverse mapping
        for c in hub_type.constraints:
            key = (c.language, c.value)
            # Search reverse mappings
            for (sl, st), (tl, tt) in self._CROSS_MAPPINGS.items():
                if tl == self.hub_paradigm and tt == c.value:
                    if sl == target_lang:
                        target_type = FluxType.from_paradigm(
                            sl, st,
                            confidence=hub_type.confidence * 0.95,
                        )
                        return BridgeResult(
                            target_type=target_type,
                            strategy=BridgeStrategy.DIRECT,
                            fidelity=0.95,
                            cost=0.05,
                        )

        # Fall back to base type
        target_tags = _PARADIGM_TO_BASE.get(target_lang, {})
        for tag, base in target_tags.items():
            if base == hub_type.base_type:
                target_type = FluxType.from_paradigm(
                    target_lang, tag,
                    confidence=hub_type.confidence * 0.8,
                )
                return BridgeResult(
                    target_type=target_type,
                    strategy=BridgeStrategy.UPCAST,
                    fidelity=0.8,
                    cost=0.15,
                    warnings=["No direct reverse mapping; used base type"],
                )

        # Last resort
        target_type = FluxType(
            base_type=hub_type.base_type,
            confidence=hub_type.confidence * 0.7,
            paradigm_source=target_lang,
            name=f"bridged:{hub_type.name}",
        )
        return BridgeResult(
            target_type=target_type,
            strategy=BridgeStrategy.CONSTRAINT_STRIPPING,
            fidelity=0.7,
            cost=0.25,
            warnings=["No match in target; stripped constraints"],
        )

    def _bridge_preserve(
        self, source: FluxType, target_lang: str
    ) -> BridgeResult:
        """Bridge by preserving all constraints and changing paradigm source."""
        target = FluxType(
            base_type=source.base_type,
            constraints=[
                FluxConstraint(
                    kind=c.kind,
                    language=c.language,  # Keep original language
                    value=c.value,
                    confidence=c.confidence * 0.9,
                    meta=c.meta,
                )
                for c in source.constraints
            ],
            confidence=source.confidence * 0.9,
            paradigm_source=target_lang,
            name=f"preserved:{source.name}",
        )
        return BridgeResult(
            target_type=target,
            strategy=BridgeStrategy.CONSTRAINT_PRESERVATION,
            fidelity=0.9,
            cost=0.08,
        )

    def _bridge_strip(
        self, source: FluxType, target_lang: str
    ) -> BridgeResult:
        """Bridge by stripping all language-specific constraints."""
        target = FluxType(
            base_type=source.base_type,
            confidence=source.confidence * 0.85,
            paradigm_source=target_lang,
            name=f"stripped:{source.name}",
        )
        return BridgeResult(
            target_type=target,
            strategy=BridgeStrategy.CONSTRAINT_STRIPPING,
            fidelity=0.75,
            cost=0.2,
            warnings=[f"Stripped {len(source.constraints)} constraints"],
        )

    def _bridge_upcast(
        self, source: FluxType, target_lang: str,
        warnings: List[str],
    ) -> BridgeResult:
        """Bridge by widening to a more general type."""
        # Upcast rules: ACTIVE → VALUE, CONTAINER → VALUE, SCOPE → VALUE
        upcast_map = {
            FluxBaseType.ACTIVE: FluxBaseType.VALUE,
            FluxBaseType.CONTAINER: FluxBaseType.VALUE,
            FluxBaseType.SCOPE: FluxBaseType.VALUE,
            FluxBaseType.CAPABILITY: FluxBaseType.SCOPE,
            FluxBaseType.MODAL: FluxBaseType.SCOPE,
            FluxBaseType.UNCERTAIN: FluxBaseType.VALUE,
            FluxBaseType.CONTEXTUAL: FluxBaseType.VALUE,
        }
        new_base = upcast_map.get(source.base_type, source.base_type)
        if new_base != source.base_type:
            warnings.append(
                f"Upcast {source.base_type.name} → {new_base.name}"
            )
        target = FluxType(
            base_type=new_base,
            confidence=source.confidence * 0.85,
            paradigm_source=target_lang,
            name=f"upcast:{source.name}",
        )
        return BridgeResult(
            target_type=target,
            strategy=BridgeStrategy.UPCAST,
            fidelity=0.7,
            cost=0.2,
            warnings=warnings,
        )

    def _bridge_quantum_defer(
        self, source: FluxType, target_lang: str
    ) -> BridgeResult:
        """Bridge by creating a superposition of possible target types."""
        target_tags = _PARADIGM_TO_BASE.get(target_lang, {})

        # Gather all base types that match the source
        candidates: List[Tuple[FluxBaseType, float]] = []
        for tag, base in target_tags.items():
            if base == source.base_type:
                candidates.append((base, 1.0))
            elif source.base_type == FluxBaseType.UNCERTAIN:
                # For uncertain sources, all target types are candidates
                candidates.append((base, 0.5))
            elif source.base_type == FluxBaseType.CONTEXTUAL:
                # For contextual, match by constraint
                for c in source.constraints:
                    if c.value == tag:
                        candidates.append((base, 0.8))

        if not candidates:
            # Fallback: add base type with low confidence
            candidates = [(source.base_type, 0.3)]

        # Deduplicate by base type, keeping highest amplitude
        seen: Dict[FluxBaseType, float] = {}
        for bt, amp in candidates:
            if bt not in seen or amp > seen[bt]:
                seen[bt] = amp
        candidates = [(bt, amp) for bt, amp in seen.items()]

        target = FluxType.uncertain(
            possibilities=candidates,
            paradigm_source=target_lang,
            name=f"quantum_defer:{source.name}",
        )
        target.confidence = source.confidence * 0.6

        return BridgeResult(
            target_type=target,
            strategy=BridgeStrategy.QUANTUM_DEFER,
            fidelity=0.5,
            cost=0.3,
            warnings=[
                f"Created superposition of {len(candidates)} types; "
                f"requires runtime resolution"
            ],
        )


# ══════════════════════════════════════════════════════════════════
# 3. UniversalTypeChecker — Cross-Language Type Checking
# ══════════════════════════════════════════════════════════════════

@dataclass
class TypeCheckResult:
    """Result of a type check operation.

    Attributes:
        is_compatible: Whether the types are compatible (score >= threshold).
        score: The raw compatibility score.
        report: Detailed compatibility report.
        required_bridge: Whether a bridge translation is needed.
        bridge_result: The bridge result if translation was applied.
        errors: Type errors (empty if compatible).
    """
    is_compatible: bool = False
    score: float = 0.0
    report: Optional[CompatibilityReport] = None
    required_bridge: bool = False
    bridge_result: Optional[BridgeResult] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compatible": self.is_compatible,
            "score": round(self.score, 6),
            "report": self.report.to_dict() if self.report else None,
            "required_bridge": self.required_bridge,
            "bridge_result": self.bridge_result.to_dict() if self.bridge_result else None,
            "errors": self.errors,
        }


class UniversalTypeChecker:
    """Cross-language type checker for FUTS.

    The checker validates type compatibility across paradigm boundaries.
    Unlike traditional type checkers, it does NOT reject programs with
    type mismatches.  Instead, it produces a compatibility score and
    optional bridge translations.

    Usage:
        checker = UniversalTypeChecker(registry)

        # Check if a Chinese classifier type is compatible with German gender
        result = checker.check(zho_type, deu_type)
        if result.is_compatible:
            print(f"Compatible with score {result.score:.2f}")
        else:
            print(f"Incompatible: {result.errors}")

        # Check a type signature
        sig_result = checker.check_signature(func_sig, arg_types)

    Design Principles:
      1. Types are ADVISORY, not mandatory
      2. Compatibility is GRADUAL (0.0 to 1.0, not pass/fail)
      3. Bridges are SUGGESTED, not enforced
      4. Contextual types are DEFERRED to runtime
      5. Quantum types are PROPAGATED (not collapsed prematurely)
    """

    def __init__(
        self,
        registry: Optional[FluxTypeRegistry] = None,
        compatibility_threshold: float = 0.5,
        auto_bridge: bool = True,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.compatibility = TypeCompatibility()
        self.bridge = TypeBridge(self.registry)
        self.threshold = compatibility_threshold
        self.auto_bridge = auto_bridge

    def check(
        self,
        expected: FluxType,
        actual: FluxType,
    ) -> TypeCheckResult:
        """Check if actual type is compatible with expected type.

        Args:
            expected: The type we expect.
            actual: The type we actually have.

        Returns:
            A TypeCheckResult with compatibility information.
        """
        report = self.compatibility.compute(expected, actual)
        errors = []

        # Low score → errors
        if report.score < self.threshold:
            errors.append(
                f"Type compatibility {report.score:.2f} below threshold "
                f"{self.threshold:.2f}"
            )
            if report.level in (CompatibilityLevel.CONTRADICTORY,
                                CompatibilityLevel.INCOMPATIBLE):
                errors.append(
                    f"Types are {report.level.value}: "
                    f"{expected.base_type.name} vs {actual.base_type.name}"
                )

        # Contextual types → defer
        if (expected.base_type == FluxBaseType.CONTEXTUAL or
                actual.base_type == FluxBaseType.CONTEXTUAL):
            errors.append(
                "Context-dependent type: deferring to runtime resolution"
            )

        # Quantum types → propagate
        if (expected.quantum_state and not expected.quantum_state.collapsed):
            errors.append(
                f"Expected type is in superposition "
                f"(entropy={expected.quantum_state.entropy():.2f})"
            )

        # Try auto-bridge if not compatible
        bridge_result = None
        required_bridge = False
        if (not errors or report.score < 0.7) and self.auto_bridge:
            if expected.paradigm_source != actual.paradigm_source:
                bridge_result = self.bridge.translate(
                    actual, expected.paradigm_source
                )
                required_bridge = True

        return TypeCheckResult(
            is_compatible=report.score >= self.threshold,
            score=report.score,
            report=report,
            required_bridge=required_bridge,
            bridge_result=bridge_result,
            errors=errors,
        )

    def check_signature(
        self,
        signature: FluxTypeSignature,
        arguments: List[FluxType],
    ) -> TypeCheckResult:
        """Check if a list of argument types matches a function signature.

        Args:
            signature: The expected type signature.
            arguments: The actual argument types.

        Returns:
            A TypeCheckResult for the whole signature.
        """
        errors: List[str] = []
        scores: List[float] = []

        # Check arity
        if len(arguments) != len(signature.inputs):
            errors.append(
                f"Arity mismatch: expected {len(signature.inputs)}, "
                f"got {len(arguments)}"
            )
            return TypeCheckResult(
                is_compatible=False,
                score=0.0,
                errors=errors,
            )

        # Check each argument
        for i, (expected, actual) in enumerate(
            zip(signature.inputs, arguments)
        ):
            result = self.check(expected, actual)
            scores.append(result.score)
            if not result.is_compatible:
                errors.append(f"Argument {i}: {result.errors}")
            if result.report:
                errors.extend(result.report.warnings)

        # Check capability requirements
        for req in signature.requires:
            satisfied = False
            for arg in arguments:
                if arg.has_constraint(req.kind):
                    satisfied = True
                    break
            if not satisfied:
                errors.append(
                    f"Unsatisfied requirement: {req.kind.value} "
                    f"(from {req.language})"
                )

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return TypeCheckResult(
            is_compatible=avg_score >= self.threshold and not errors,
            score=avg_score,
            errors=errors,
        )

    def suggest_bridge(
        self,
        source: FluxType,
        target_lang: str,
    ) -> List[BridgeResult]:
        """Suggest all possible bridge translations, ranked by fidelity.

        Args:
            source: The source FluxType.
            target_lang: The target paradigm.

        Returns:
            List of BridgeResults sorted by fidelity (best first).
        """
        results: List[BridgeResult] = []

        for strategy in BridgeStrategy:
            try:
                result = self.bridge.translate(
                    source, target_lang, strategy=strategy
                )
                results.append(result)
            except Exception:
                continue

        results.sort(key=lambda r: r.fidelity, reverse=True)
        return results

    def check_all_pairs(
        self,
        types: List[FluxType],
    ) -> Dict[Tuple[str, str], CompatibilityReport]:
        """Check all pairs of types in a list.

        Returns:
            Dict mapping (name_a, name_b) → CompatibilityReport.
        """
        results: Dict[Tuple[str, str], CompatibilityReport] = {}
        for i, a in enumerate(types):
            for b in types[i + 1:]:
                name_a = a.name or a.id
                name_b = b.name or b.id
                report = self.compatibility.compute(a, b)
                results[(name_a, name_b)] = report
                results[(name_b, name_a)] = report
        return results

    def summary(self) -> str:
        """Generate a human-readable summary of the checker configuration."""
        type_counts: Dict[str, int] = {}
        for t in self.registry.all_types():
            type_counts[t.paradigm_source] = type_counts.get(t.paradigm_source, 0) + 1

        lines = [
            "UniversalTypeChecker — FUTS Cross-Language Type System",
            f"  Registry: {len(self.registry.all_types())} types from "
            f"{len(type_counts)} paradigms",
            f"  Paradigm breakdown: {type_counts}",
            f"  Compatibility threshold: {self.threshold:.2f}",
            f"  Auto-bridge: {self.auto_bridge}",
            f"  Hub paradigm: {self.bridge.hub_paradigm}",
        ]
        return "\n".join(lines)
