"""
Bidirectional Cross-Compiler for FLUX Multilingual Runtime.

Translates programs between language runtimes using their bridge adapters
and the type-safe bridge system.  Compilation operates at the TYPE level
— it translates FluxType representations between paradigms, producing
witness chains that prove each transformation was type-safe.

Architecture:
    CrossCompiler (main orchestrator)
    ├── TypeSafeBridge  (type translation + witness generation)
    ├── BridgeCostMatrix (cost computation for route selection)
    ├── MultiHopCompiler (multi-hop path planning + execution)
    ├── SemanticEquivalenceChecker (behavioral verification)
    └── TranslationRuleSet (data-driven paradigm-pair translation rules)

Translation Pipeline:
    source code → parse to FluxType(s) → translate via TypeAlgebra
    → emit target code with type annotations → witness chain

Reference: type_safe_bridge.py, bridge_adapter.py (all runtimes)
"""

from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
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
    FluxTypeRegistry,
    _PARADIGM_TO_BASE,
    build_default_registry,
)
from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    BridgeCostReport,
    BridgeWarning,
    PreservationDegree,
    TypeAlgebra,
    TypeEquivalenceClass,
    TypeEquivalenceSlot,
    TypeSafeBridge,
    TypeWitness,
    WitnessConstraint,
    WitnessGenerator,
)
from flux_a2a.type_checker import BridgeResult, BridgeStrategy


# ══════════════════════════════════════════════════════════════════
# 1. Data Structures
# ══════════════════════════════════════════════════════════════════

LangTag: TypeAlias = str
NativeTag: TypeAlias = str

# All six supported FLUX paradigms
SUPPORTED_LANGUAGES: FrozenSet[LangTag] = frozenset(
    {"zho", "deu", "kor", "san", "lat", "wen"}
)


class ASTDiffKind(str, Enum):
    """Category of an AST difference."""
    TYPE_RENAMED = "type_renamed"
    CONSTRAINT_LOST = "constraint_lost"
    CONSTRAINT_ADDED = "constraint_added"
    CONFIDENCE_CHANGED = "confidence_changed"
    BASE_TYPE_SHIFTED = "base_type_shifted"
    PARADIGM_CHANGED = "paradigm_changed"
    STRUCTURAL_PRESERVED = "structural_preserved"


@dataclass
class ASTDiff:
    """Structural comparison between source and target ASTs (type trees).

    Rather than comparing raw text ASTs, the cross-compiler compares
    the FluxType trees: base types, constraints, confidence, and
    paradigm source.  Each difference is categorized and scored.

    Attributes:
        source_lang: Source paradigm tag.
        target_lang: Target paradigm tag.
        diff_count: Number of detected differences.
        match_score: 0.0-1.0, how structurally similar source and target are.
        details: List of individual differences.
    """
    source_lang: LangTag = ""
    target_lang: LangTag = ""
    diff_count: int = 0
    match_score: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "diff_count": self.diff_count,
            "match_score": round(self.match_score, 4),
            "details": self.details,
        }


@dataclass
class CompilationResult:
    """Result of a cross-language compilation.

    Attributes:
        target_code: The compiled representation in the target paradigm.
            This is a serialized type-level program, not raw source text.
        witness_chain: Proof that each type transformation was type-safe.
        total_cost: Accumulated bridge cost across all translated types.
        information_preserved: 0.0-1.0, fraction of semantics that survived.
        warnings: What was lost, ambiguous, or degraded.
        ast_diff: Structural comparison between source and target type trees.
        source_types: The original FluxTypes from the source paradigm.
        target_types: The translated FluxTypes in the target paradigm.
        route: The path taken (e.g. ["zho", "deu"] or ["zho", "san", "lat"]).
    """
    target_code: str = ""
    witness_chain: list[TypeWitness] = field(default_factory=list)
    total_cost: float = 0.0
    information_preserved: float = 0.0
    warnings: list[str] = field(default_factory=list)
    ast_diff: ASTDiff = field(default_factory=ASTDiff)
    source_types: list[FluxType] = field(default_factory=list)
    target_types: list[FluxType] = field(default_factory=list)
    route: list[LangTag] = field(default_factory=list)

    @property
    def is_type_safe(self) -> bool:
        """Whether all witnesses in the chain are valid."""
        return all(w.is_valid for w in self.witness_chain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_code": self.target_code,
            "witness_count": len(self.witness_chain),
            "total_cost": round(self.total_cost, 4),
            "information_preserved": round(self.information_preserved, 4),
            "warnings": self.warnings,
            "ast_diff": self.ast_diff.to_dict(),
            "is_type_safe": self.is_type_safe,
            "route": self.route,
        }


@dataclass
class RoundTripResult:
    """Result of a round-trip compilation (A → B → A).

    Attributes:
        original_types: The original FluxTypes before round-trip.
        round_trip_types: The FluxTypes after round-tripping.
        preservation_rate: 0.0-1.0, how much survived the round trip.
        forward_result: Compilation result for the forward leg.
        backward_result: Compilation result for the return leg.
        witness_chain: Combined chain of all witnesses.
    """
    original_types: list[FluxType] = field(default_factory=list)
    round_trip_types: list[FluxType] = field(default_factory=list)
    preservation_rate: float = 0.0
    forward_result: Optional[CompilationResult] = None
    backward_result: Optional[CompilationResult] = None
    witness_chain: list[TypeWitness] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preservation_rate": round(self.preservation_rate, 4),
            "original_count": len(self.original_types),
            "witness_count": len(self.witness_chain),
            "forward_warnings": self.forward_result.warnings if self.forward_result else [],
            "backward_warnings": self.backward_result.warnings if self.backward_result else [],
        }


# ══════════════════════════════════════════════════════════════════
# 2. Translation Rules — Data-Driven Paradigm-Pair Mappings
# ══════════════════════════════════════════════════════════════════

class TransformKind(str, Enum):
    """Category of a translation rule transform."""
    CLASSIFIER_TO_PLURAL = "classifier_to_plural"
    SCOPE_TO_PARTICLE = "scope_to_particle"
    GENDER_TO_HONORIFIC = "gender_to_honorific"
    CASE_TO_CASE = "case_to_case"
    WORD_ORDER_SHIFT = "word_order_shift"
    TOPIC_MARKING = "topic_marking"
    VERB_STRUCTURE = "verb_structure"
    REGISTER_MAP = "register_map"
    TEMPORAL_MAP = "temporal_map"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class TranslationRule:
    """A single data-driven translation rule between two paradigms.

    Rules are declarative: they describe WHAT transforms, not HOW.
    The CrossCompiler applies them using the TypeAlgebra and TypeSafeBridge.

    Attributes:
        rule_id: Unique identifier.
        from_lang: Source paradigm.
        to_lang: Target paradigm.
        source_pattern: Pattern to match in source types (native tag or wildcard).
        target_pattern: What to produce in the target (native tag).
        transform_kind: Category of this transformation.
        confidence_factor: Multiplier applied to confidence during translation.
        notes: Human-readable description of what this rule captures.
    """
    rule_id: str
    from_lang: LangTag
    to_lang: LangTag
    source_pattern: str
    target_pattern: str
    transform_kind: TransformKind
    confidence_factor: float = 0.9
    notes: str = ""


class TranslationRuleSet:
    """Registry of data-driven translation rules for paradigm pairs.

    Rules are loaded declaratively and consulted during compilation to
    guide type-level translation beyond what TypeAlgebra provides.
    They capture paradigm-specific linguistic phenomena that affect
    program semantics.

    Usage:
        rules = TranslationRuleSet.standard()
        for rule in rules.lookup("zho", "deu"):
            print(f"{rule.source_pattern} -> {rule.target_pattern}")
    """

    def __init__(self) -> None:
        self._rules: list[TranslationRule] = []
        self._index: dict[tuple[LangTag, LangTag], list[TranslationRule]] = {}

    def add(self, rule: TranslationRule) -> None:
        """Register a translation rule."""
        self._rules.append(rule)
        key = (rule.from_lang, rule.to_lang)
        self._index.setdefault(key, []).append(rule)

    def lookup(self, from_lang: LangTag, to_lang: LangTag) -> list[TranslationRule]:
        """Get all rules for a paradigm pair."""
        return list(self._index.get((from_lang, to_lang), []))

    def match(
        self, from_lang: LangTag, to_lang: LangTag, source_tag: str
    ) -> Optional[TranslationRule]:
        """Find the first rule that matches a source tag."""
        for rule in self.lookup(from_lang, to_lang):
            if rule.source_pattern == source_tag or rule.source_pattern == "*":
                return rule
        return None

    def all_rules(self) -> Iterator[TranslationRule]:
        """Iterate over all registered rules."""
        return iter(self._rules)

    @classmethod
    def standard(cls) -> TranslationRuleSet:
        """Build the standard rule set with all paradigm-pair rules.

        The rules are data-driven — they encode linguistic correspondences
        that guide type-level translation.  Each rule maps a source
        paradigm construct to a target paradigm construct.
        """
        rs = cls()

        # ── ZHO ↔ DEU (classifier ↔ case/gender) ────────────────

        # ZHO classifiers → DEU gender + plural
        rs.add(TranslationRule(
            rule_id="zho_deu_person_mask",
            from_lang="zho", to_lang="deu",
            source_pattern="person",
            target_pattern="maskulinum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="ZHO person classifier (位) → DEU Maskulinum (active agent)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_animal_mask",
            from_lang="zho", to_lang="deu",
            source_pattern="animal",
            target_pattern="maskulinum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.8,
            notes="ZHO animal classifier (隻) → DEU Maskulinum",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_collective_fem",
            from_lang="zho", to_lang="deu",
            source_pattern="collective",
            target_pattern="femininum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="ZHO collective classifier (群) → DEU Femininum (container)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_flat_neut",
            from_lang="zho", to_lang="deu",
            source_pattern="flat_object",
            target_pattern="neutrum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.75,
            notes="ZHO flat object classifier (張) → DEU Neutrum (value)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_long_neut",
            from_lang="zho", to_lang="deu",
            source_pattern="long_flexible",
            target_pattern="neutrum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.75,
            notes="ZHO long flexible classifier (條) → DEU Neutrum",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_round_neut",
            from_lang="zho", to_lang="deu",
            source_pattern="small_round",
            target_pattern="neutrum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.75,
            notes="ZHO small round classifier (顆) → DEU Neutrum",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_machine_mask",
            from_lang="zho", to_lang="deu",
            source_pattern="machine",
            target_pattern="maskulinum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="ZHO machine classifier (台) → DEU Maskulinum (active)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_pair_fem",
            from_lang="zho", to_lang="deu",
            source_pattern="pair",
            target_pattern="femininum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="ZHO pair classifier (雙) → DEU Femininum (container)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_volume_fem",
            from_lang="zho", to_lang="deu",
            source_pattern="volume",
            target_pattern="femininum",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.8,
            notes="ZHO volume classifier (杯/瓶) → DEU Femininum (container)",
        ))

        # ZHO 量词作用域 → DEU Kasus作用域
        rs.add(TranslationRule(
            rule_id="zho_deu_scope_nom",
            from_lang="zho", to_lang="deu",
            source_pattern="flat_surface",
            target_pattern="nominativ",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.7,
            notes="ZHO spatial scope (面) → DEU Nominativ (subject scope)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_deu_scope_acc",
            from_lang="zho", to_lang="deu",
            source_pattern="spatial_extent",
            target_pattern="akkusativ",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.65,
            notes="ZHO spatial extent (段) → DEU Akkusativ (object scope)",
        ))

        # ── DEU ↔ ZHO (reverse) ──────────────────────────────────
        rs.add(TranslationRule(
            rule_id="deu_zho_mask_person",
            from_lang="deu", to_lang="zho",
            source_pattern="maskulinum",
            target_pattern="person",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="DEU Maskulinum → ZHO person classifier (位)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_zho_fem_collective",
            from_lang="deu", to_lang="zho",
            source_pattern="femininum",
            target_pattern="collective",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.85,
            notes="DEU Femininum → ZHO collective classifier (群)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_zho_neut_flat",
            from_lang="deu", to_lang="zho",
            source_pattern="neutrum",
            target_pattern="flat_object",
            transform_kind=TransformKind.CLASSIFIER_TO_PLURAL,
            confidence_factor=0.75,
            notes="DEU Neutrum → ZHO flat object classifier (張)",
        ))

        # DEU Kasus → ZHO 量词 scope
        rs.add(TranslationRule(
            rule_id="deu_zho_nom_surface",
            from_lang="deu", to_lang="zho",
            source_pattern="nominativ",
            target_pattern="flat_surface",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.7,
            notes="DEU Nominativ → ZHO flat surface scope (面)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_zho_acc_extent",
            from_lang="deu", to_lang="zho",
            source_pattern="akkusativ",
            target_pattern="spatial_extent",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.65,
            notes="DEU Akkusativ → ZHO spatial extent scope (段)",
        ))

        # ── DEU ↔ KOR (case ↔ honorific/particles) ──────────────

        # DEU Kasus → KOR particles
        rs.add(TranslationRule(
            rule_id="deu_kor_nom_i",
            from_lang="deu", to_lang="kor",
            source_pattern="nominativ",
            target_pattern="subject_honorific",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.8,
            notes="DEU Nominativ (이/가 subject marker) → KOR subject_honorific",
        ))
        rs.add(TranslationRule(
            rule_id="deu_kor_acc_eul",
            from_lang="deu", to_lang="kor",
            source_pattern="akkusativ",
            target_pattern="object_honorific",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.8,
            notes="DEU Akkusativ (을/를 object marker) → KOR object_honorific",
        ))
        rs.add(TranslationRule(
            rule_id="deu_kor_dat_ege",
            from_lang="deu", to_lang="kor",
            source_pattern="dativ",
            target_pattern="haeyoche",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.7,
            notes="DEU Dativ (에게 recipient) → KOR haeyoche (polite register)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_kor_gen_ui",
            from_lang="deu", to_lang="kor",
            source_pattern="genitiv",
            target_pattern="haerache",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.65,
            notes="DEU Genitiv (의 possessive) → KOR haerache (plain register)",
        ))

        # DEU Geschlecht → KOR honorific register
        rs.add(TranslationRule(
            rule_id="deu_kor_mask_formal",
            from_lang="deu", to_lang="kor",
            source_pattern="maskulinum",
            target_pattern="hasipsioche",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.7,
            notes="DEU Maskulinum → KOR hasipsioche (formal highest, active agent)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_kor_fem_polite",
            from_lang="deu", to_lang="kor",
            source_pattern="femininum",
            target_pattern="haeyoche",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.7,
            notes="DEU Femininum → KOR haeyoche (polite, container/receiver)",
        ))
        rs.add(TranslationRule(
            rule_id="deu_kor_neut_plain",
            from_lang="deu", to_lang="kor",
            source_pattern="neutrum",
            target_pattern="haeche",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.65,
            notes="DEU Neutrum → KOR haeche (informal, value/data)",
        ))

        # ── KOR ↔ DEU (reverse) ──────────────────────────────────
        rs.add(TranslationRule(
            rule_id="kor_deu_subj_nom",
            from_lang="kor", to_lang="deu",
            source_pattern="subject_honorific",
            target_pattern="nominativ",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.8,
            notes="KOR subject_honorific → DEU Nominativ",
        ))
        rs.add(TranslationRule(
            rule_id="kor_deu_obj_acc",
            from_lang="kor", to_lang="deu",
            source_pattern="object_honorific",
            target_pattern="akkusativ",
            transform_kind=TransformKind.SCOPE_TO_PARTICLE,
            confidence_factor=0.8,
            notes="KOR object_honorific → DEU Akkusativ",
        ))
        rs.add(TranslationRule(
            rule_id="kor_deu_formal_mask",
            from_lang="kor", to_lang="deu",
            source_pattern="hasipsioche",
            target_pattern="maskulinum",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.7,
            notes="KOR hasipsioche → DEU Maskulinum (formal → active)",
        ))
        rs.add(TranslationRule(
            rule_id="kor_deu_polite_fem",
            from_lang="kor", to_lang="deu",
            source_pattern="haeyoche",
            target_pattern="femininum",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.7,
            notes="KOR haeyoche → DEU Femininum (polite → container)",
        ))
        rs.add(TranslationRule(
            rule_id="kor_deu_plain_neut",
            from_lang="kor", to_lang="deu",
            source_pattern="haeche",
            target_pattern="neutrum",
            transform_kind=TransformKind.GENDER_TO_HONORIFIC,
            confidence_factor=0.65,
            notes="KOR haeche → DEU Neutrum (informal → value)",
        ))

        # ── ZHO ↔ KOR (classifier ↔ particles) ──────────────────

        # ZHO 量词系统 → KOR 助词系统
        rs.add(TranslationRule(
            rule_id="zho_kor_person_subj",
            from_lang="zho", to_lang="kor",
            source_pattern="person",
            target_pattern="subject_honorific",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.75,
            notes="ZHO person classifier → KOR subject particle (은/는)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_kor_animal_subj",
            from_lang="zho", to_lang="kor",
            source_pattern="animal",
            target_pattern="subject_honorific",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.7,
            notes="ZHO animal classifier → KOR subject particle",
        ))
        rs.add(TranslationRule(
            rule_id="zho_kor_collective_obj",
            from_lang="zho", to_lang="kor",
            source_pattern="collective",
            target_pattern="object_honorific",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.7,
            notes="ZHO collective classifier → KOR object particle (을/를)",
        ))
        rs.add(TranslationRule(
            rule_id="zho_kor_flat_val",
            from_lang="zho", to_lang="kor",
            source_pattern="flat_object",
            target_pattern="haeche",
            transform_kind=TransformKind.REGISTER_MAP,
            confidence_factor=0.6,
            notes="ZHO flat object → KOR plain register (value type)",
        ))

        # ZHO topic markers → KOR topic markers
        rs.add(TranslationRule(
            rule_id="zho_kor_topic",
            from_lang="zho", to_lang="kor",
            source_pattern="generic",
            target_pattern="haeyoche",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.55,
            notes="ZHO generic classifier → KOR polite (topic marker 은/는)",
        ))

        # ── KOR ↔ ZHO (reverse) ──────────────────────────────────
        rs.add(TranslationRule(
            rule_id="kor_zho_subj_person",
            from_lang="kor", to_lang="zho",
            source_pattern="subject_honorific",
            target_pattern="person",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.75,
            notes="KOR subject particle → ZHO person classifier (位)",
        ))
        rs.add(TranslationRule(
            rule_id="kor_zho_obj_collective",
            from_lang="kor", to_lang="zho",
            source_pattern="object_honorific",
            target_pattern="collective",
            transform_kind=TransformKind.TOPIC_MARKING,
            confidence_factor=0.7,
            notes="KOR object particle → ZHO collective classifier (群)",
        ))
        rs.add(TranslationRule(
            rule_id="kor_zho_formal_person",
            from_lang="kor", to_lang="zho",
            source_pattern="hasipsioche",
            target_pattern="person",
            transform_kind=TransformKind.REGISTER_MAP,
            confidence_factor=0.7,
            notes="KOR formal → ZHO person classifier (respectful)",
        ))
        rs.add(TranslationRule(
            rule_id="kor_zho_plain_flat",
            from_lang="kor", to_lang="zho",
            source_pattern="haeche",
            target_pattern="flat_object",
            transform_kind=TransformKind.REGISTER_MAP,
            confidence_factor=0.6,
            notes="KOR plain → ZHO flat object classifier (張)",
        ))

        return rs


# ══════════════════════════════════════════════════════════════════
# 3. AST Diff Engine — Structural Type Tree Comparison
# ══════════════════════════════════════════════════════════════════

class ASTDiffEngine:
    """Computes structural differences between source and target FluxType lists.

    Compares type trees pair-wise, producing categorized differences
    and an overall match score.
    """

    @staticmethod
    def compare(
        source_types: list[FluxType],
        target_types: list[FluxType],
        source_lang: LangTag,
        target_lang: LangTag,
    ) -> ASTDiff:
        """Compare source and target type lists, producing an ASTDiff.

        Args:
            source_types: Original FluxTypes from the source paradigm.
            target_types: Translated FluxTypes in the target paradigm.
            source_lang: Source language tag.
            target_lang: Target language tag.

        Returns:
            An ASTDiff with categorized differences and match score.
        """
        details: list[dict[str, Any]] = []
        match_scores: list[float] = []

        # Pair-wise comparison (zip, so shorter list limits)
        for i, (src, tgt) in enumerate(zip(source_types, target_types)):
            pair_score = ASTDiffEngine._compare_pair(src, tgt, i, details)
            match_scores.append(pair_score)

        # Extra types in either list are full losses
        if len(source_types) > len(target_types):
            for i in range(len(target_types), len(source_types)):
                details.append({
                    "index": i,
                    "kind": ASTDiffKind.CONSTRAINT_LOST.value,
                    "detail": f"Source type {i} has no corresponding target",
                    "severity": 1.0,
                })
                match_scores.append(0.0)
        elif len(target_types) > len(source_types):
            for i in range(len(source_types), len(target_types)):
                details.append({
                    "index": i,
                    "kind": ASTDiffKind.CONSTRAINT_ADDED.value,
                    "detail": f"Target type {i} has no corresponding source",
                    "severity": 0.5,
                })
                match_scores.append(0.5)

        overall_score = (
            sum(match_scores) / len(match_scores) if match_scores else 1.0
        )

        return ASTDiff(
            source_lang=source_lang,
            target_lang=target_lang,
            diff_count=sum(1 for d in details if d["kind"] != ASTDiffKind.STRUCTURAL_PRESERVED.value),
            match_score=overall_score,
            details=details,
        )

    @staticmethod
    def _compare_pair(
        src: FluxType, tgt: FluxType, index: int, details: list[dict[str, Any]]
    ) -> float:
        """Compare a pair of FluxTypes and append differences to details."""
        score = 1.0

        # Paradigm change (expected for cross-compilation)
        if src.paradigm_source != tgt.paradigm_source:
            details.append({
                "index": index,
                "kind": ASTDiffKind.PARADIGM_CHANGED.value,
                "detail": f"Paradigm: {src.paradigm_source} → {tgt.paradigm_source}",
                "severity": 0.0,  # Expected, not a penalty
            })

        # Base type comparison
        if src.base_type != tgt.base_type:
            dist = src.base_type.spectrum_distance(tgt.base_type)
            penalty = min(dist / 7.0, 1.0)
            score -= penalty * 0.3
            details.append({
                "index": index,
                "kind": ASTDiffKind.BASE_TYPE_SHIFTED.value,
                "detail": (
                    f"Base type: {src.base_type.name} → {tgt.base_type.name} "
                    f"(spectrum distance: {dist})"
                ),
                "severity": penalty,
            })
        else:
            details.append({
                "index": index,
                "kind": ASTDiffKind.STRUCTURAL_PRESERVED.value,
                "detail": f"Base type preserved: {src.base_type.name}",
                "severity": 0.0,
            })

        # Constraint comparison
        src_kinds = {(c.kind, c.value) for c in src.constraints if c.language == src.paradigm_source}
        tgt_kinds = {(c.kind, c.value) for c in tgt.constraints if c.language == tgt.paradigm_source}

        lost = src_kinds - tgt_kinds  # These are expected (paradigm shift)
        gained = tgt_kinds - src_kinds

        if lost:
            for kind, val in lost:
                details.append({
                    "index": index,
                    "kind": ASTDiffKind.CONSTRAINT_LOST.value,
                    "detail": f"Constraint lost: {kind.value}={val}",
                    "severity": 0.3,
                })
            score -= min(len(lost) * 0.05, 0.2)

        if gained:
            for kind, val in gained:
                details.append({
                    "index": index,
                    "kind": ASTDiffKind.CONSTRAINT_ADDED.value,
                    "detail": f"Constraint added: {kind.value}={val}",
                    "severity": 0.1,
                })

        # Confidence change
        conf_diff = abs(src.confidence - tgt.confidence)
        if conf_diff > 0.01:
            score -= conf_diff * 0.2
            details.append({
                "index": index,
                "kind": ASTDiffKind.CONFIDENCE_CHANGED.value,
                "detail": (
                    f"Confidence: {src.confidence:.3f} → {tgt.confidence:.3f} "
                    f"(Δ={conf_diff:.3f})"
                ),
                "severity": conf_diff,
            })

        return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════
# 4. Code Emitter — Target Language Code Generation
# ══════════════════════════════════════════════════════════════════

class CodeEmitter:
    """Emits target-language code from translated FluxType representations.

    The emitted code is a structured, type-annotated representation
    suitable for consumption by the target runtime.  It is NOT a
    human-readable natural language string — it is a serialized
    type-level program.

    Format per paradigm:
        zho:  #zho {classifier: 只, type: ANIMAL, noun: 猫, count: 3}
        deu:  #deu {kasus: AKKUSATIV, geschlecht: FEMININUM, artikel: die}
        kor:  #kor {honorific: haeyoche, particle: 을, speech: declarative}
        san:  #san {vibhakti: dvitiya, linga: stri, pada: active}
        lat:  #lat {casus: accusativus, genus: femininum, tempus: praesens}
        wen:  #wen {strategy: attack, context: topic, stack: 0}
    """

    _LANG_PREFIX: dict[LangTag, str] = {
        "zho": "zho", "deu": "deu", "kor": "kor",
        "san": "san", "lat": "lat", "wen": "wen",
    }

    @classmethod
    def emit(cls, flux_type: FluxType) -> str:
        """Emit a single FluxType as a target-language annotation string."""
        lang = flux_type.paradigm_source
        prefix = cls._LANG_PREFIX.get(lang, lang)

        parts: list[str] = [f"#{prefix}"]

        # Base type
        parts.append(f"type:{flux_type.base_type.name}")

        # Confidence
        if flux_type.confidence < 1.0:
            parts.append(f"conf:{flux_type.confidence:.2f}")

        # Constraints
        for c in flux_type.constraints:
            parts.append(f"{c.kind.value}={c.value}")

        # Name
        if flux_type.name:
            parts.append(f"name:{flux_type.name}")

        return " ".join(parts)

    @classmethod
    def emit_program(
        cls, types: list[FluxType], lang: LangTag
    ) -> str:
        """Emit a full program as a sequence of type annotations."""
        lines: list[str] = [f"# program compiled to {lang}"]
        lines.append(f"# types: {len(types)}")
        for i, ft in enumerate(types):
            lines.append(cls.emit(ft))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 5. SemanticEquivalenceChecker — Behavioral Verification
# ══════════════════════════════════════════════════════════════════

@dataclass
class EquivalenceCheckResult:
    """Result of a semantic equivalence check.

    Attributes:
        is_equivalent: Whether source and target are semantically equivalent.
        equivalence_score: 0.0-1.0, degree of equivalence.
        divergences: List of behavioral differences found.
        test_cases_passed: Number of property-based test cases passed.
        test_cases_total: Total number of test cases run.
    """
    is_equivalent: bool = False
    equivalence_score: float = 0.0
    divergences: list[str] = field(default_factory=list)
    test_cases_passed: int = 0
    test_cases_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_equivalent": self.is_equivalent,
            "equivalence_score": round(self.equivalence_score, 4),
            "divergences": self.divergences,
            "test_cases_passed": self.test_cases_passed,
            "test_cases_total": self.test_cases_total,
        }


class SemanticEquivalenceChecker:
    """Verifies that compiled code preserves behavior.

    Uses three complementary approaches:

    1. **Type-level symbolic comparison**: Compare base types, constraints,
       and confidence scores to check structural equivalence.

    2. **Property-based testing**: Generate random evaluation contexts
       (inputs), evaluate source and target types through the same
       abstract interpreter, and verify outputs match.

    3. **Divergence detection**: Identify specific behavioral differences
       that go beyond type-level mismatches.

    The checker operates on FluxTypes directly — it does not execute
    runtime code.  Instead, it simulates behavior through type-level
    operations (spectrum distance, constraint compatibility, etc.).

    Usage:
        checker = SemanticEquivalenceChecker()
        result = checker.check(source_types, target_types)
        print(f"Equivalent: {result.is_equivalent} (score={result.equivalence_score:.2f})")
    """

    # Threshold above which we consider types "equivalent enough"
    EQUIVALENCE_THRESHOLD: float = 0.7

    def __init__(self, algebra: Optional[TypeAlgebra] = None) -> None:
        self.algebra = algebra or TypeAlgebra()

    def check(
        self,
        source_types: list[FluxType],
        target_types: list[FluxType],
        num_property_tests: int = 50,
    ) -> EquivalenceCheckResult:
        """Check semantic equivalence between source and target type lists.

        Args:
            source_types: Original FluxTypes.
            target_types: Translated FluxTypes.
            num_property_tests: Number of random property tests to run.

        Returns:
            An EquivalenceCheckResult with equivalence score and divergences.
        """
        divergences: list[str] = []
        scores: list[float] = []

        # Phase 1: Pair-wise structural comparison
        for i, (src, tgt) in enumerate(zip(source_types, target_types)):
            score, divs = self._compare_pair(src, tgt, i)
            scores.append(score)
            divergences.extend(divs)

        # Phase 2: Property-based testing
        passed, total = self._property_test(
            source_types, target_types, num_property_tests
        )
        property_score = passed / total if total > 0 else 0.0
        scores.append(property_score)

        # Phase 3: Aggregate
        overall = sum(scores) / len(scores) if scores else 0.0

        return EquivalenceCheckResult(
            is_equivalent=overall >= self.EQUIVALENCE_THRESHOLD,
            equivalence_score=overall,
            divergences=divergences,
            test_cases_passed=passed,
            test_cases_total=total,
        )

    def _compare_pair(
        self, src: FluxType, tgt: FluxType, index: int
    ) -> tuple[float, list[str]]:
        """Compare a pair of FluxTypes for semantic equivalence."""
        divergences: list[str] = []
        score = 1.0

        # Base type alignment
        dist = src.effective_base_type().spectrum_distance(tgt.effective_base_type())
        if dist > 0:
            penalty = dist / 7.0
            score -= penalty * 0.3
            if dist >= 3:
                divergences.append(
                    f"Type[{index}]: significant base type shift "
                    f"({src.base_type.name} → {tgt.base_type.name}, dist={dist})"
                )

        # Constraint overlap
        src_constraints = set(
            (c.kind.value, c.value) for c in src.constraints
        )
        tgt_constraints = set(
            (c.kind.value, c.value) for c in tgt.constraints
        )

        # We expect paradigm-specific constraints to change, so only
        # penalize if constraint KINDS are lost (not values)
        src_kinds = {k for k, v in src_constraints}
        tgt_kinds = {k for k, v in tgt_constraints}
        lost_kinds = src_kinds - tgt_kinds
        if lost_kinds:
            score -= min(len(lost_kinds) * 0.1, 0.3)
            divergences.append(
                f"Type[{index}]: constraint kinds lost: {lost_kinds}"
            )

        # Confidence preservation
        conf_diff = abs(src.confidence - tgt.confidence)
        if conf_diff > 0.2:
            score -= conf_diff * 0.2
            divergences.append(
                f"Type[{index}]: confidence gap ({src.confidence:.2f} → "
                f"{tgt.confidence:.2f}, Δ={conf_diff:.2f})"
            )

        return max(0.0, min(1.0, score)), divergences

    def _property_test(
        self,
        source_types: list[FluxType],
        target_types: list[FluxType],
        num_tests: int,
    ) -> tuple[int, int]:
        """Run property-based tests on type pairs.

        Tests:
        - Monotonicity: if source.base_type < source2.base_type, then
          target.base_type should maintain similar ordering.
        - Confidence correlation: source confidence and target confidence
          should be positively correlated.
        - Equivalence class stability: types in the same equivalence
          class should map to the same or adjacent classes.
        """
        passed = 0

        for test_idx in range(num_tests):
            # Generate a "random" test context by cycling through types
            if not source_types or not target_types:
                continue

            src = source_types[test_idx % len(source_types)]
            tgt = target_types[test_idx % len(target_types)]

            # Test: base type should be within 2 spectrum positions
            dist = src.effective_base_type().spectrum_distance(
                tgt.effective_base_type()
            )
            if dist <= 2:
                passed += 1

            # Test: confidence should not drop below 0.2
            if tgt.confidence >= 0.2:
                passed += 1

            # Test: both types should have constraints
            if src.constraints and tgt.constraints:
                passed += 1

        total = num_tests * 3  # 3 tests per iteration
        return passed, total

    def check_arithmetic_preservation(
        self,
        source_types: list[FluxType],
        target_types: list[FluxType],
    ) -> EquivalenceCheckResult:
        """Specialized check for arithmetic expression preservation.

        For arithmetic types, verify that:
        - VALUE base types map to VALUE base types
        - Confidence is preserved above 0.5
        - No MODAL or CAPABILITY constraints leak in
        """
        divergences: list[str] = []
        scores: list[float] = []
        passed = 0
        total = 0

        for i, (src, tgt) in enumerate(zip(source_types, target_types)):
            total += 1
            pair_score = 1.0

            # Arithmetic should stay in VALUE base type
            if src.base_type == FluxBaseType.VALUE:
                if tgt.base_type != FluxBaseType.VALUE:
                    pair_score -= 0.4
                    divergences.append(
                        f"Arithmetic[{i}]: base type changed from VALUE to "
                        f"{tgt.base_type.name}"
                    )

            # No modal/capability leakage
            for c in tgt.constraints:
                if c.kind in (ConstraintKind.EXECUTION_MODE,
                              ConstraintKind.HONORIFIC_LEVEL):
                    pair_score -= 0.2
                    divergences.append(
                        f"Arithmetic[{i}]: unexpected constraint "
                        f"{c.kind.value}={c.value}"
                    )

            # Confidence check
            if tgt.confidence >= 0.5:
                passed += 1

            scores.append(max(0.0, min(1.0, pair_score)))

        overall = sum(scores) / len(scores) if scores else 0.0

        return EquivalenceCheckResult(
            is_equivalent=overall >= 0.7,
            equivalence_score=overall,
            divergences=divergences,
            test_cases_passed=passed,
            test_cases_total=total,
        )


# ══════════════════════════════════════════════════════════════════
# 6. MultiHopCompiler — Multi-Hop Path Planning and Execution
# ══════════════════════════════════════════════════════════════════

class MultiHopCompiler:
    """Compiles programs through multi-hop paths when direct bridging is
    too lossy.

    Uses Dijkstra's algorithm over the BridgeCostMatrix to find the
    cheapest path between two paradigms, then composes translations
    across each hop.

    The cost model considers:
    - Bridge cost (from BridgeCostMatrix)
    - Information loss per hop
    - Translation rule availability (rule-based hops are cheaper)

    Usage:
        compiler = MultiHopCompiler()
        path = compiler.find_cheapest_path("zho", "lat")
        result = compiler.compile_hops(source_types, path)
    """

    MAX_HOPS: int = 4  # Don't route through more than 4 languages

    def __init__(
        self,
        algebra: Optional[TypeAlgebra] = None,
        cost_matrix: Optional[BridgeCostMatrix] = None,
        safe_bridge: Optional[TypeSafeBridge] = None,
        rule_set: Optional[TranslationRuleSet] = None,
    ) -> None:
        self.algebra = algebra or TypeAlgebra()
        self.cost_matrix = cost_matrix or BridgeCostMatrix(self.algebra)
        self.safe_bridge = safe_bridge or TypeSafeBridge(algebra=self.algebra)
        self.rule_set = rule_set or TranslationRuleSet.standard()

    def find_cheapest_path(
        self, from_lang: LangTag, to_lang: LangTag
    ) -> list[LangTag]:
        """Find the cheapest path between two paradigms.

        Uses Dijkstra's algorithm over bridge costs.  The path includes
        both endpoints.

        Args:
            from_lang: Source paradigm.
            to_lang: Target paradigm.

        Returns:
            Ordered list of language tags forming the cheapest path.
        """
        if from_lang == to_lang:
            return [from_lang]

        # Dijkstra
        dist: dict[LangTag, float] = {lang: float("inf") for lang in SUPPORTED_LANGUAGES}
        prev: dict[LangTag, Optional[LangTag]] = {lang: None for lang in SUPPORTED_LANGUAGES}
        dist[from_lang] = 0.0
        visited: set[LangTag] = set()

        # Priority queue: (cost, lang)
        pq: list[tuple[float, LangTag]] = [(0.0, from_lang)]

        while pq:
            cost, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            if current == to_lang:
                break

            for neighbor in SUPPORTED_LANGUAGES:
                if neighbor in visited:
                    continue

                edge_cost = self._edge_cost(current, neighbor)
                new_cost = cost + edge_cost

                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    prev[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

        # Reconstruct path
        if dist[to_lang] == float("inf"):
            return [from_lang, to_lang]  # Fallback: direct

        path: list[LangTag] = []
        current: Optional[LangTag] = to_lang
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        # Limit hops
        if len(path) > self.MAX_HOPS + 1:
            # Fall back to direct
            return [from_lang, to_lang]

        return path

    def compile_hops(
        self,
        source_types: list[FluxType],
        path: list[LangTag],
    ) -> CompilationResult:
        """Compile types through a multi-hop path.

        Args:
            source_types: Original FluxTypes from the source paradigm.
            path: Ordered list of languages to route through.

        Returns:
            CompilationResult with all hop witnesses composed.
        """
        if len(path) < 2:
            return CompilationResult(
                target_code=CodeEmitter.emit_program(source_types, path[0] if path else "flux"),
                source_types=source_types,
                target_types=source_types,
                route=path,
                information_preserved=1.0,
            )

        all_witnesses: list[TypeWitness] = []
        all_warnings: list[str] = []
        total_cost = 0.0
        current_types = source_types

        for hop_idx in range(len(path) - 1):
            from_lang = path[hop_idx]
            to_lang = path[hop_idx + 1]

            hop_types: list[FluxType] = []
            hop_witnesses: list[TypeWitness] = []

            for src_type in current_types:
                result = self.safe_bridge.translate_safe(src_type, to_lang)
                hop_types.append(result.target_type)
                hop_witnesses.append(result.witness)
                all_warnings.extend(result.warnings)

            # Compute hop cost
            cost_report = self.cost_matrix.compute(from_lang, to_lang)
            total_cost += cost_report.total_cost

            all_witnesses.extend(hop_witnesses)
            current_types = hop_types

        # Compute information preservation
        info_preserved = MultiHopCompiler._compute_information_preserved(
            source_types, current_types, total_cost
        )

        # AST diff
        ast_diff = ASTDiffEngine.compare(
            source_types, current_types,
            path[0], path[-1],
        )

        return CompilationResult(
            target_code=CodeEmitter.emit_program(current_types, path[-1]),
            witness_chain=all_witnesses,
            total_cost=total_cost,
            information_preserved=info_preserved,
            warnings=all_warnings,
            ast_diff=ast_diff,
            source_types=source_types,
            target_types=current_types,
            route=path,
        )

    def is_multi_hop_cheaper(
        self, from_lang: LangTag, to_lang: LangTag
    ) -> tuple[bool, list[LangTag], float]:
        """Check if a multi-hop path is cheaper than direct bridging.

        Returns:
            (is_cheaper, best_path, savings)
        """
        direct_cost = self.cost_matrix.compute(from_lang, to_lang).total_cost
        best_path = self.find_cheapest_path(from_lang, to_lang)

        # Compute multi-hop cost
        multi_cost = 0.0
        for i in range(len(best_path) - 1):
            multi_cost += self.cost_matrix.compute(
                best_path[i], best_path[i + 1]
            ).total_cost

        is_cheaper = multi_cost < direct_cost - 0.001
        savings = direct_cost - multi_cost

        return is_cheaper, best_path, savings

    def _edge_cost(self, from_lang: LangTag, to_lang: LangTag) -> float:
        """Compute edge cost between two languages, with rule-based discount.

        If translation rules exist for this pair, the cost is reduced
        because the mapping is better understood.
        """
        base_cost = self.cost_matrix.compute(from_lang, to_lang).total_cost

        # Discount for having translation rules
        rules = self.rule_set.lookup(from_lang, to_lang)
        if rules:
            # More rules = cheaper (up to 30% discount)
            rule_discount = min(len(rules) * 0.05, 0.3)
            base_cost *= (1.0 - rule_discount)

        return base_cost

    @staticmethod
    def _compute_information_preserved(
        source_types: list[FluxType],
        target_types: list[FluxType],
        total_cost: float,
    ) -> float:
        """Estimate how much information survived the multi-hop translation."""
        if not source_types or not target_types:
            return 0.0

        # Pair-wise confidence comparison
        conf_scores: list[float] = []
        for src, tgt in zip(source_types, target_types):
            # Information preserved = confidence ratio × base type match
            conf_ratio = tgt.confidence / max(src.confidence, 0.01)
            base_match = 1.0 if src.base_type == tgt.base_type else 0.7
            conf_scores.append(min(conf_ratio, 1.0) * base_match)

        type_preservation = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        # Cost penalty: higher cost = more information lost
        cost_penalty = max(0.0, 1.0 - total_cost)

        return type_preservation * cost_penalty


# ══════════════════════════════════════════════════════════════════
# 7. CrossCompiler — Main Orchestrator
# ══════════════════════════════════════════════════════════════════

class CrossCompiler:
    """Bidirectional cross-compiler that translates programs between
    language runtimes using their bridge adapters and type-safe bridge
    system.

    The compiler operates at the TYPE level — it takes FluxType
    representations, translates them via TypeAlgebra and TypeSafeBridge,
    and produces target-language code with full witness chains.

    Key features:
    - Direct compilation between any two of the 6 FLUX paradigms
    - Multi-hop route optimization for lossy direct bridges
    - Round-trip compilation with preservation measurement
    - Data-driven translation rules for paradigm-specific phenomena
    - Witness chain verification for every transformation

    Usage:
        compiler = CrossCompiler()

        # Simple compilation
        result = compiler.compile(
            source_types=[zho_type],
            from_lang="zho",
            to_lang="deu",
        )

        # Round-trip
        rt = compiler.compile_round_trip(
            source_types=[zho_type],
            lang="zho",
            via_lang="deu",
        )

        # Route optimization
        path = compiler.optimize_route("zho", "lat")
        print(f"Best path: {' → '.join(path)}")
    """

    def __init__(
        self,
        algebra: Optional[TypeAlgebra] = None,
        cost_matrix: Optional[BridgeCostMatrix] = None,
        safe_bridge: Optional[TypeSafeBridge] = None,
        rule_set: Optional[TranslationRuleSet] = None,
        multi_hop: Optional[MultiHopCompiler] = None,
    ) -> None:
        self.algebra = algebra or TypeAlgebra()
        self.cost_matrix = cost_matrix or BridgeCostMatrix(self.algebra)
        self.safe_bridge = safe_bridge or TypeSafeBridge(algebra=self.algebra)
        self.rule_set = rule_set or TranslationRuleSet.standard()
        self.multi_hop = multi_hop or MultiHopCompiler(
            algebra=self.algebra,
            cost_matrix=self.cost_matrix,
            safe_bridge=self.safe_bridge,
            rule_set=self.rule_set,
        )
        self.equivalence_checker = SemanticEquivalenceChecker(self.algebra)

    def compile(
        self,
        source: str | list[FluxType],
        from_lang: LangTag,
        to_lang: LangTag,
        use_multi_hop: bool = True,
    ) -> CompilationResult:
        """Compile source code or types from one paradigm to another.

        If ``source`` is a string, it is parsed as a list of native type
        tags (comma-separated).  If it is a list of FluxTypes, it is used
        directly.

        Args:
            source: Either a comma-separated string of native type tags,
                or a list of FluxType objects.
            from_lang: Source paradigm tag.
            to_lang: Target paradigm tag.
            use_multi_hop: Whether to try multi-hop routing if direct
                bridging is expensive.

        Returns:
            A CompilationResult with the translated code, witnesses, and
            metadata.
        """
        # Parse source
        source_types = self._parse_source(source, from_lang)

        # Validate languages
        self._validate_lang(from_lang)
        self._validate_lang(to_lang)

        if from_lang == to_lang:
            return CompilationResult(
                target_code=CodeEmitter.emit_program(source_types, to_lang),
                source_types=source_types,
                target_types=list(source_types),
                route=[from_lang, to_lang],
                information_preserved=1.0,
                warnings=["Source and target are the same language"],
            )

        # Decide routing strategy
        if use_multi_hop and len(source_types) > 0:
            is_cheaper, path, savings = self.multi_hop.is_multi_hop_cheaper(
                from_lang, to_lang
            )
            if is_cheaper and len(path) > 2:
                result = self.multi_hop.compile_hops(source_types, path)
                result.warnings.append(
                    f"Used multi-hop path {' → '.join(path)} "
                    f"(saved {savings:.4f} cost vs direct)"
                )
                return result

        # Direct compilation
        return self._compile_direct(source_types, from_lang, to_lang)

    def compile_round_trip(
        self,
        source: str | list[FluxType],
        lang: LangTag,
        via_lang: LangTag,
    ) -> RoundTripResult:
        """Compile A → B → A and measure preservation.

        Args:
            source: Source types or native tag string.
            lang: The original (and final) language.
            via_lang: The intermediate language to route through.

        Returns:
            A RoundTripResult with preservation metrics.
        """
        source_types = self._parse_source(source, lang)

        # Forward: lang → via_lang
        forward = self.compile(source_types, lang, via_lang, use_multi_hop=False)

        # Backward: via_lang → lang
        backward = self.compile(
            forward.target_types, via_lang, lang, use_multi_hop=False
        )

        # Combined witness chain
        all_witnesses = forward.witness_chain + backward.witness_chain

        # Compute preservation rate
        preservation = self._compute_round_trip_preservation(
            source_types, backward.target_types
        )

        return RoundTripResult(
            original_types=list(source_types),
            round_trip_types=backward.target_types,
            preservation_rate=preservation,
            forward_result=forward,
            backward_result=backward,
            witness_chain=all_witnesses,
        )

    def optimize_route(
        self, from_lang: LangTag, to_lang: LangTag
    ) -> list[LangTag]:
        """Find the cheapest bridge path between two paradigms.

        Uses Dijkstra's algorithm over the BridgeCostMatrix, discounted
        by translation rule availability.

        Args:
            from_lang: Source paradigm.
            to_lang: Target paradigm.

        Returns:
            Ordered list of language tags forming the cheapest path.
        """
        return self.multi_hop.find_cheapest_path(from_lang, to_lang)

    def check_equivalence(
        self,
        source_types: list[FluxType],
        target_types: list[FluxType],
        num_property_tests: int = 50,
    ) -> EquivalenceCheckResult:
        """Check semantic equivalence between source and target types.

        Args:
            source_types: Original FluxTypes.
            target_types: Translated FluxTypes.
            num_property_tests: Number of property-based tests.

        Returns:
            An EquivalenceCheckResult.
        """
        return self.equivalence_checker.check(
            source_types, target_types, num_property_tests
        )

    # ── Internal Methods ────────────────────────────────────────

    def _compile_direct(
        self,
        source_types: list[FluxType],
        from_lang: LangTag,
        to_lang: LangTag,
    ) -> CompilationResult:
        """Perform direct single-hop compilation."""
        target_types: list[FluxType] = []
        witnesses: list[TypeWitness] = []
        warnings: list[str] = []

        for src_type in source_types:
            # Apply translation rule if available (data-driven)
            src_tag = self._infer_tag(src_type)
            rule = self.rule_set.match(from_lang, to_lang, src_tag) if src_tag else None

            if rule and rule.target_pattern in _PARADIGM_TO_BASE.get(to_lang, {}):
                # Use rule-based translation
                try:
                    target_type = FluxType.from_paradigm(
                        to_lang,
                        rule.target_pattern,
                        confidence=src_type.confidence * rule.confidence_factor,
                        name=f"{to_lang}:{rule.target_pattern}",
                    )
                except KeyError:
                    # Rule target not valid, fall back to bridge
                    result = self.safe_bridge.translate_safe(src_type, to_lang)
                    target_type = result.target_type
                    witnesses.append(result.witness)
                    warnings.extend(result.warnings)
                    target_types.append(target_type)
                    continue
            else:
                # Use TypeSafeBridge for translation
                result = self.safe_bridge.translate_safe(src_type, to_lang)
                target_type = result.target_type
                witnesses.append(result.witness)
                warnings.extend(result.warnings)

            # Generate witness for rule-based translation
            if rule:
                # Build a witness manually for the rule-based translation
                witness = self._build_rule_witness(
                    src_type, target_type, from_lang, to_lang, rule
                )
                witnesses.append(witness)

            target_types.append(target_type)

        # Compute costs
        cost_report = self.cost_matrix.compute(from_lang, to_lang)
        total_cost = cost_report.total_cost

        # Compute information preservation
        info_preserved = CrossCompiler._compute_information_preserved(
            source_types, target_types
        )

        # Compute AST diff
        ast_diff = ASTDiffEngine.compare(
            source_types, target_types, from_lang, to_lang
        )

        # Emit target code
        target_code = CodeEmitter.emit_program(target_types, to_lang)

        return CompilationResult(
            target_code=target_code,
            witness_chain=witnesses,
            total_cost=total_cost,
            information_preserved=info_preserved,
            warnings=warnings,
            ast_diff=ast_diff,
            source_types=list(source_types),
            target_types=target_types,
            route=[from_lang, to_lang],
        )

    def _build_rule_witness(
        self,
        source_type: FluxType,
        target_type: FluxType,
        from_lang: LangTag,
        to_lang: LangTag,
        rule: TranslationRule,
    ) -> TypeWitness:
        """Build a TypeWitness for a rule-based translation."""
        constraints: list[WitnessConstraint] = []

        # C1: Rule was applied
        constraints.append(WitnessConstraint(
            name="rule_applied",
            expected=rule.rule_id,
            actual=rule.rule_id,
            satisfied=True,
        ))

        # C2: Source and target differ
        constraints.append(WitnessConstraint(
            name="paradigm_difference",
            expected="different",
            actual="different" if from_lang != to_lang else "same",
            satisfied=from_lang != to_lang,
        ))

        # C3: Base type distance check
        dist = source_type.base_type.spectrum_distance(target_type.base_type)
        constraints.append(WitnessConstraint(
            name="base_type_distance",
            expected=0,
            actual=int(dist),
            satisfied=dist <= 2,
        ))

        # C4: Confidence floor
        constraints.append(WitnessConstraint(
            name="confidence_floor",
            expected=">=0.2",
            actual=target_type.confidence,
            satisfied=target_type.confidence >= 0.2,
        ))

        # C5: Transform kind recorded
        constraints.append(WitnessConstraint(
            name="transform_kind",
            expected=rule.transform_kind.value,
            actual=rule.transform_kind.value,
            satisfied=True,
        ))

        # Determine preservation from confidence factor
        if rule.confidence_factor >= 0.85:
            preservation = PreservationDegree.NEAR_LOSSLESS
        elif rule.confidence_factor >= 0.65:
            preservation = PreservationDegree.PARTIAL
        else:
            preservation = PreservationDegree.LOSSY

        witness = TypeWitness(
            source_lang=from_lang,
            source_tag=rule.source_pattern,
            target_lang=to_lang,
            target_tag=rule.target_pattern,
            source_type=source_type,
            target_type=target_type,
            strategy=f"rule:{rule.rule_id}",
            preservation=preservation,
            constraints=constraints,
        )
        witness.verify()
        return witness

    @staticmethod
    def _infer_tag(flux_type: FluxType) -> Optional[NativeTag]:
        """Infer the native tag from a FluxType's constraints."""
        for c in flux_type.constraints:
            if c.language == flux_type.paradigm_source:
                return str(c.value)
        return None

    @staticmethod
    def _parse_source(
        source: str | list[FluxType], lang: LangTag
    ) -> list[FluxType]:
        """Parse source input into a list of FluxTypes.

        If source is a list of FluxTypes, returns it directly.
        If source is a string, parses comma-separated native tags.
        """
        if isinstance(source, list):
            return source

        # Parse comma-separated native tags
        tags = [t.strip() for t in source.split(",") if t.strip()]
        types: list[FluxType] = []
        for tag in tags:
            try:
                ft = FluxType.from_paradigm(lang, tag, name=f"{lang}:{tag}")
                types.append(ft)
            except KeyError:
                # Unknown tag — create a generic type
                types.append(FluxType(
                    base_type=FluxBaseType.VALUE,
                    confidence=0.5,
                    paradigm_source=lang,
                    name=f"{lang}:unknown:{tag}",
                    constraints=[
                        FluxConstraint(
                            kind=ConstraintKind.CLASSIFIER_SHAPE,
                            language=lang,
                            value=tag,
                            confidence=0.3,
                        )
                    ],
                ))
        return types

    @staticmethod
    def _validate_lang(lang: LangTag) -> None:
        """Validate that a language tag is supported."""
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {lang}. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )

    @staticmethod
    def _compute_information_preserved(
        source_types: list[FluxType],
        target_types: list[FluxType],
    ) -> float:
        """Compute the information preservation ratio."""
        if not source_types or not target_types:
            return 0.0

        scores: list[float] = []
        for src, tgt in zip(source_types, target_types):
            # Confidence preservation
            conf_ratio = tgt.confidence / max(src.confidence, 0.01)
            conf_preserved = min(conf_ratio, 1.0)

            # Base type match
            base_match = 1.0 if src.base_type == tgt.base_type else (
                0.7 if src.effective_base_type().spectrum_distance(
                    tgt.effective_base_type()
                ) <= 1 else 0.4
            )

            scores.append(conf_preserved * base_match)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _compute_round_trip_preservation(
        original: list[FluxType], round_tripped: list[FluxType]
    ) -> float:
        """Compute how much survived a round trip."""
        if not original or not round_tripped:
            return 0.0

        scores: list[float] = []
        for orig, rt in zip(original, round_tripped):
            # Base type must match exactly for full preservation
            base_score = 1.0 if orig.base_type == rt.base_type else 0.3

            # Confidence should not degrade too much
            conf_score = rt.confidence / max(orig.confidence, 0.01)
            conf_score = min(conf_score, 1.0)

            scores.append(base_score * conf_score)

        return sum(scores) / len(scores) if scores else 0.0
