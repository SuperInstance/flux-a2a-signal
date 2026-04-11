"""
Type-Safe Cross-Language Bridge System for FUTS.

This module provides a type-safe layer on top of the existing TypeBridge that
guarantees compile-time type safety for cross-paradigm operations.  It implements
four interconnected systems:

  1. **TypeAlgebra** — A unified type algebra that maps noun-categorization,
     register, contextual-scoping, and temporal-execution systems across all
     six FLUX languages (ZHO, DEU, KOR, SAN, LAT, WEN).

  2. **BridgeCostMatrix** — Computes the multi-factor cost of bridging between
     any two paradigms, including structural distance, expressiveness gap,
     information loss, and translation ambiguity across 8 paradigm dimensions.

  3. **TypeWitness** — A proof-carrying type witness system.  Every bridge
     transformation produces a serializable witness that can be verified
     independently, supporting audit trails and incremental refinement.

  4. **TypeSafeBridge** — The main orchestrator that combines TypeAlgebra,
     BridgeCostMatrix, and TypeWitness into a single enhanced bridge that
     tracks preservation guarantees, supports incremental refinement, and
     maintains bidirectional type maps between all language pairs.

Design Principles:
  - Type mappings are either LOSSLESS (isomorphic) or LOSSY (information-reducing).
  - Lossy mappings can be incrementally refined toward losslessness.
  - Every cross-paradigm translation produces a verifiable witness.
  - Costs are computed from four orthogonal factors, not a single number.
  - The bridge is fully bidirectional: A→B and B→A are tracked independently.

Reference: docs/round10-12_type_unification.md, type_checker.py
"""

from __future__ import annotations

import hashlib
import json
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
from flux_a2a.type_checker import (
    BridgeResult,
    BridgeStrategy,
    TypeBridge,
    TypeCompatibility,
)


# ══════════════════════════════════════════════════════════════════
# 1. TypeAlgebra — Unified Cross-Paradigm Type Algebra
# ══════════════════════════════════════════════════════════════════

# Type aliases for clarity
LangTag: TypeAlias = str
NativeTag: TypeAlias = str
AlgebraId: TypeAlias = str
TypeSlot: TypeAlias = Tuple[LangTag, NativeTag]


class PreservationDegree(str, Enum):
    """Degree to which a type mapping preserves semantics.

    LOSSLESS: The mapping is isomorphic — round-tripping recovers the original.
    NEAR_LOSSLESS: Minor information attenuation; round-trip fidelity >= 0.90.
    PARTIAL: Significant information loss; round-trip fidelity 0.60–0.90.
    LOSSY: Major information loss; round-trip fidelity < 0.60.
    DEGRADED: Only the base type survives; all language-specific detail is lost.
    """
    LOSSLESS = "lossless"
    NEAR_LOSSLESS = "near_lossless"
    PARTIAL = "partial"
    LOSSY = "lossy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class TypeEquivalenceSlot:
    """A single slot in a type-equivalence class.

    Each slot represents a (language, native_tag) pair that participates
    in a cross-paradigm equivalence.  For example, the noun-categorization
    equivalence class includes (zho, "flat_object"), (deu, "neutrum"),
    (san, "linga_napumsaka"), and (lat, "neutrum").
    """
    language: LangTag
    native_tag: NativeTag
    base_type: FluxBaseType

    def as_tuple(self) -> TypeSlot:
        return (self.language, self.native_tag)


@dataclass
class TypeEquivalenceClass:
    """A family of type slots that are semantically equivalent across paradigms.

    Attributes:
        class_id: Unique identifier for this equivalence class.
        name: Human-readable name describing the shared semantic role.
        slots: The set of (language, tag) pairs in this class.
        degree: The weakest preservation degree among all pairings.
        semantic_domain: Which semantic domain this class belongs to
            (noun_cat, register, scope, temporal).
    """
    class_id: str
    name: str
    slots: FrozenSet[TypeEquivalenceSlot] = field(default_factory=frozenset)
    degree: PreservationDegree = PreservationDegree.LOSSLESS
    semantic_domain: str = "noun_cat"
    notes: str = ""

    def get_slot(self, language: LangTag) -> Optional[TypeEquivalenceSlot]:
        """Retrieve the slot for a specific language, if present."""
        for slot in self.slots:
            if slot.language == language:
                return slot
        return None

    def has_language(self, language: LangTag) -> bool:
        """Check if this equivalence class covers a given language."""
        return any(s.language == language for s in self.slots)

    def languages(self) -> Set[LangTag]:
        """Return the set of languages covered by this class."""
        return {s.language for s in self.slots}

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "slots": [(s.language, s.native_tag, s.base_type.name) for s in self.slots],
            "degree": self.degree.value,
            "semantic_domain": self.semantic_domain,
            "notes": self.notes,
        }


class TypeAlgebra:
    """Unified type algebra mapping across all six FLUX paradigms.

    Organizes cross-language type equivalences into four semantic domains:

      1. **noun_cat** — Noun categorization systems:
         ZHO classifiers ↔ DEU Kasus ↔ SAN Vibhakti ↔ LAT Casus
      2. **register** — Register / honorific information:
         KOR honorific levels ↔ SAN person-gender-number
      3. **scope** — Contextual scoping systems:
         ZHO 量词 scope ↔ WEN 文境 stack depth
      4. **temporal** — Temporal execution systems:
         LAT Tempus modes ↔ SAN Lakara tenses

    Usage:
        algebra = TypeAlgebra()
        cls = algebra.find_class("zho", "flat_object")
        if cls:
            deu_slot = cls.get_slot("deu")
            print(deu_slot.native_tag)  # "neutrum"

    The algebra is immutable after construction; all equivalence classes
    are frozen.  To add new mappings, create a new TypeAlgebra instance
    with ``TypeAlgebra.with_extra_classes(...)``.
    """

    def __init__(self) -> None:
        self._classes: Dict[AlgebraId, TypeEquivalenceClass] = {}
        self._index: Dict[TypeSlot, AlgebraId] = {}
        self._populate()

    # ── Population ───────────────────────────────────────────────

    def _add_class(self, cls: TypeEquivalenceClass) -> None:
        """Register an equivalence class and index its slots."""
        self._classes[cls.class_id] = cls
        for slot in cls.slots:
            self._index[slot.as_tuple()] = cls.class_id

    def _slot(self, lang: LangTag, tag: NativeTag) -> TypeEquivalenceSlot:
        """Create a TypeEquivalenceSlot with the base type looked up from FUTS."""
        base = _PARADIGM_TO_BASE.get(lang, {}).get(tag, FluxBaseType.VALUE)
        return TypeEquivalenceSlot(language=lang, native_tag=tag, base_type=base)

    def _populate(self) -> None:
        """Populate all four semantic domains with equivalence classes."""

        # ── Domain 1: Noun Categorization ────────────────────────

        # Active/masculine agents
        self._add_class(TypeEquivalenceClass(
            class_id="nc_active_person",
            name="Active person / masculine / pushkara",
            slots=frozenset([
                self._slot("zho", "person"),
                self._slot("deu", "maskulinum"),
                self._slot("san", "linga_pushkara"),
                self._slot("lat", "maskulinum"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="noun_cat",
            notes="Agents that initiate actions: 位(人), der, पुंल्लिङ्ग, masculinum",
        ))

        # Active/masculine — animals
        self._add_class(TypeEquivalenceClass(
            class_id="nc_active_animal",
            name="Active animal / machine",
            slots=frozenset([
                self._slot("zho", "animal"),
                self._slot("zho", "machine"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="noun_cat",
            notes="Animals and machines as active agents: 隻, 台",
        ))

        # Container/feminine
        self._add_class(TypeEquivalenceClass(
            class_id="nc_container",
            name="Container / feminine / stri",
            slots=frozenset([
                self._slot("zho", "collective"),
                self._slot("zho", "pair"),
                self._slot("deu", "femininum"),
                self._slot("san", "linga_stri"),
                self._slot("lat", "femininum"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="noun_cat",
            notes="Containers that hold values: 群, 雙, die, स्त्रीलिङ्ग, femininum",
        ))

        # Value/neuter — flat objects
        self._add_class(TypeEquivalenceClass(
            class_id="nc_value_flat",
            name="Value / neuter / napumsaka — flat objects",
            slots=frozenset([
                self._slot("zho", "flat_object"),
                self._slot("zho", "long_flexible"),
                self._slot("zho", "small_round"),
                self._slot("deu", "neutrum"),
                self._slot("san", "linga_napumsaka"),
                self._slot("lat", "neutrum"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="noun_cat",
            notes="Flat/value objects: 張, 條, 顆 → das, नपुंसकलिङ्ग, neutrum",
        ))

        # Container — volume
        self._add_class(TypeEquivalenceClass(
            class_id="nc_value_volume",
            name="Container / volume — holding vessels",
            slots=frozenset([
                self._slot("zho", "volume"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="noun_cat",
            notes="Volume containers: 杯, 瓶",
        ))

        # Scope — case systems
        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_nominative",
            name="Scope: nominative / prathama / nominativus",
            slots=frozenset([
                self._slot("deu", "nominativ"),
                self._slot("san", "prathama"),
                self._slot("lat", "nominativus"),
            ]),
            degree=PreservationDegree.LOSSLESS,
            semantic_domain="noun_cat",
            notes="Subject case across DEU/SAN/LAT: Nominativ, प्रथमा, Nominativus",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_accusative",
            name="Scope: accusative / dvitiya / accusativus",
            slots=frozenset([
                self._slot("deu", "akkusativ"),
                self._slot("san", "dvitiya"),
                self._slot("lat", "accusativus"),
            ]),
            degree=PreservationDegree.LOSSLESS,
            semantic_domain="noun_cat",
            notes="Object case: Akkusativ, द्वितीया, Accusativus",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_dative",
            name="Scope: dative / chaturthi / dativus",
            slots=frozenset([
                self._slot("deu", "dativ"),
                self._slot("san", "chaturthi"),
                self._slot("lat", "dativus"),
            ]),
            degree=PreservationDegree.LOSSLESS,
            semantic_domain="noun_cat",
            notes="Recipient case: Dativ, चतुर्थी, Dativus",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_genitive",
            name="Scope: genitive / shashthi / genitivus",
            slots=frozenset([
                self._slot("deu", "genitiv"),
                self._slot("san", "shashthi"),
                self._slot("lat", "genitivus"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="noun_cat",
            notes="Possessive case: Genitiv, षष्ठी, Genitivus",
        ))

        # SAN-only extra cases
        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_instrumental",
            name="Scope: tritiya (instrumental) — SAN-only",
            slots=frozenset([
                self._slot("san", "tritiya"),
                self._slot("lat", "ablativus"),  # ablative overlaps functionally
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="noun_cat",
            notes="Instrumental/ablative: तृतीया, Ablativus (partial overlap)",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_locative",
            name="Scope: saptami (locative) — SAN-only",
            slots=frozenset([
                self._slot("san", "saptami"),
            ]),
            degree=PreservationDegree.DEGRADED,
            semantic_domain="noun_cat",
            notes="Locative case — no direct DEU/LAT equivalent: सप्तमी",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="nc_scope_vocative",
            name="Scope: sambodhana / vocativus — agent invocation",
            slots=frozenset([
                self._slot("san", "sambodhana"),
                self._slot("lat", "vocativus"),
            ]),
            degree=PreservationDegree.LOSSLESS,
            semantic_domain="noun_cat",
            notes="Vocative / agent invocation: संबोधन, Vocativus",
        ))

        # Generic/contextual classifier
        self._add_class(TypeEquivalenceClass(
            class_id="nc_generic",
            name="Generic / contextual — default classifier",
            slots=frozenset([
                self._slot("zho", "generic"),
            ]),
            degree=PreservationDegree.DEGRADED,
            semantic_domain="noun_cat",
            notes="Default classifier 個 — context-dependent, no direct cross-language equivalent",
        ))

        # ── Domain 2: Register / Honorific ──────────────────────

        self._add_class(TypeEquivalenceClass(
            class_id="reg_highest_formal",
            name="Highest formal register",
            slots=frozenset([
                self._slot("kor", "hasipsioche"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="register",
            notes="Korean formal highest: 하십시오체",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="reg_subject_honorific",
            name="Subject honorification",
            slots=frozenset([
                self._slot("kor", "subject_honorific"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="register",
            notes="Korean subject honorific: 주체 높임",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="reg_polite",
            name="Polite register",
            slots=frozenset([
                self._slot("kor", "haeyoche"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="register",
            notes="Korean polite: 해요체",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="reg_informal",
            name="Informal / plain register",
            slots=frozenset([
                self._slot("kor", "haeche"),
                self._slot("kor", "haerache"),
                self._slot("kor", "haeraoche"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="register",
            notes="Korean informal levels: 해체, 해라체, 해라오체",
        ))

        # Speech acts — shared across KOR and WEN
        self._add_class(TypeEquivalenceClass(
            class_id="reg_speech_declarative",
            name="Declarative speech act",
            slots=frozenset([
                self._slot("kor", "declarative"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="register",
            notes="Declarative: 평서문",
        ))

        # ── Domain 3: Contextual Scoping ─────────────────────────

        self._add_class(TypeEquivalenceClass(
            class_id="scope_surface",
            name="Surface / spatial scope",
            slots=frozenset([
                self._slot("zho", "flat_surface"),
                self._slot("zho", "spatial_extent"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="scope",
            notes="Chinese spatial classifiers: 面, 段, 節",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="scope_wen_context",
            name="WEN context stack — topic/zero-anaphora",
            slots=frozenset([
                self._slot("wen", "topic"),
                self._slot("wen", "zero_anaphora"),
                self._slot("wen", "context_resolved"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="scope",
            notes="WEN context stack: 定題, 省略, 文境",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="scope_zho_contextual",
            name="ZHO contextual types",
            slots=frozenset([
                self._slot("zho", "generic"),
                self._slot("zho", "abstract_action"),
            ]),
            degree=PreservationDegree.DEGRADED,
            semantic_domain="scope",
            notes="ZHO contextual classifiers: 個, 次",
        ))

        # ── Domain 4: Temporal Execution ─────────────────────────

        self._add_class(TypeEquivalenceClass(
            class_id="temp_present",
            name="Present / active execution",
            slots=frozenset([
                self._slot("lat", "praesens"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="temporal",
            notes="Latin present tense: Praesens — active execution loop",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="temp_imperfect",
            name="Imperfect / continuous loop",
            slots=frozenset([
                self._slot("lat", "imperfectum"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="temporal",
            notes="Latin imperfect: Imperfectum — continuous/repeated execution",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="temp_perfect",
            name="Perfect / completed",
            slots=frozenset([
                self._slot("lat", "perfectum"),
            ]),
            degree=PreservationDegree.NEAR_LOSSLESS,
            semantic_domain="temporal",
            notes="Latin perfect: Perfectum — completed cache write",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="temp_pluperfect",
            name="Pluperfect / rollback point",
            slots=frozenset([
                self._slot("lat", "plusquamperfectum"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="temporal",
            notes="Latin pluperfect: Plusquamperfectum — past-of-past rollback",
        ))

        self._add_class(TypeEquivalenceClass(
            class_id="temp_future",
            name="Future / deferred computation",
            slots=frozenset([
                self._slot("lat", "futurum"),
                self._slot("lat", "futurum_exactum"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="temporal",
            notes="Latin futures: Futurum, Futurum Exactum — deferred/scheduled",
        ))

        # WEN military strategy as temporal execution
        self._add_class(TypeEquivalenceClass(
            class_id="temp_wen_execution",
            name="WEN execution modes — military strategy opcodes",
            slots=frozenset([
                self._slot("wen", "attack"),
                self._slot("wen", "defend"),
                self._slot("wen", "advance"),
                self._slot("wen", "retreat"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="temporal",
            notes="WEN military strategy opcodes: 攻, 守, 進, 退",
        ))

        # WEN control structures
        self._add_class(TypeEquivalenceClass(
            class_id="temp_wen_control",
            name="WEN control structures",
            slots=frozenset([
                self._slot("wen", "sequence"),
                self._slot("wen", "loop"),
            ]),
            degree=PreservationDegree.PARTIAL,
            semantic_domain="temporal",
            notes="WEN control structures: 則 (sequence), 循 (loop)",
        ))

    # ── Query API ────────────────────────────────────────────────

    def find_class(
        self, language: LangTag, native_tag: NativeTag
    ) -> Optional[TypeEquivalenceClass]:
        """Find the equivalence class containing a given (language, tag) pair.

        Args:
            language: The source language tag.
            native_tag: The language-specific type tag.

        Returns:
            The TypeEquivalenceClass if found, else None.
        """
        class_id = self._index.get((language, native_tag))
        if class_id is None:
            return None
        return self._classes[class_id]

    def translate(
        self,
        source_lang: LangTag,
        source_tag: NativeTag,
        target_lang: LangTag,
    ) -> Optional[TypeEquivalenceSlot]:
        """Translate a type tag from one language to another via equivalence.

        Args:
            source_lang: Source language.
            source_tag: Source native tag.
            target_lang: Target language.

        Returns:
            The target TypeEquivalenceSlot if a mapping exists, else None.
        """
        cls = self.find_class(source_lang, source_tag)
        if cls is None:
            return None
        return cls.get_slot(target_lang)

    def classes_by_domain(self, domain: str) -> List[TypeEquivalenceClass]:
        """Return all equivalence classes in a given semantic domain."""
        return [c for c in self._classes.values() if c.semantic_domain == domain]

    def all_classes(self) -> Iterator[TypeEquivalenceClass]:
        """Iterate over all equivalence classes."""
        return iter(self._classes.values())

    def domain_coverage(self, domain: str) -> Dict[LangTag, int]:
        """Count how many equivalence classes cover each language in a domain."""
        coverage: Dict[LangTag, int] = {}
        for cls in self.classes_by_domain(domain):
            for lang in cls.languages():
                coverage[lang] = coverage.get(lang, 0) + 1
        return coverage

    def check_consistency(
        self, domain: str, lang_a: LangTag, lang_b: LangTag
    ) -> float:
        """Measure how consistently two languages map within a domain.

        Returns a score in [0.0, 1.0] where 1.0 means all classes in the
        domain that cover ``lang_a`` also cover ``lang_b`` (and vice versa).
        """
        classes = self.classes_by_domain(domain)
        if not classes:
            return 1.0  # Vacuous consistency

        a_covered: Set[str] = set()
        b_covered: Set[str] = set()
        for cls in classes:
            if cls.has_language(lang_a):
                a_covered.add(cls.class_id)
            if cls.has_language(lang_b):
                b_covered.add(cls.class_id)

        if not a_covered and not b_covered:
            return 1.0
        if not a_covered or not b_covered:
            return 0.0

        intersection = a_covered & b_covered
        union = a_covered | b_covered
        return len(intersection) / len(union)


# ══════════════════════════════════════════════════════════════════
# 2. BridgeCostMatrix — Multi-Factor Bridge Cost Computation
# ══════════════════════════════════════════════════════════════════

# The 8 paradigm dimensions (from paradigm_lattice.py)
PARADIGM_DIMENSIONS: Tuple[str, ...] = (
    "state_magnitude",
    "control_explicitness",
    "typing_strength",
    "composition_style",
    "concurrency_model",
    "effect_tracking",
    "naming_style",
    "abstraction_level",
)

# Known paradigm coordinate profiles for the 6 FLUX languages.
# Each value is in [0.0, 1.0].  Source: paradigm_lattice.py.
_LANGUAGE_COORDINATES: Dict[LangTag, Dict[str, float]] = {
    "zho": {
        "state_magnitude": 0.5, "control_explicitness": 0.4,
        "typing_strength": 0.55, "composition_style": 0.3,
        "concurrency_model": 0.5, "effect_tracking": 0.35,
        "naming_style": 0.6, "abstraction_level": 0.7,
    },
    "deu": {
        "state_magnitude": 0.6, "control_explicitness": 0.7,
        "typing_strength": 0.65, "composition_style": 0.6,
        "concurrency_model": 0.4, "effect_tracking": 0.4,
        "naming_style": 0.7, "abstraction_level": 0.55,
    },
    "kor": {
        "state_magnitude": 0.4, "control_explicitness": 0.5,
        "typing_strength": 0.5, "composition_style": 0.4,
        "concurrency_model": 0.6, "effect_tracking": 0.65,
        "naming_style": 0.75, "abstraction_level": 0.65,
    },
    "san": {
        "state_magnitude": 0.3, "control_explicitness": 0.35,
        "typing_strength": 0.7, "composition_style": 0.3,
        "concurrency_model": 0.45, "effect_tracking": 0.6,
        "naming_style": 0.85, "abstraction_level": 0.8,
    },
    "lat": {
        "state_magnitude": 0.45, "control_explicitness": 0.6,
        "typing_strength": 0.6, "composition_style": 0.45,
        "concurrency_model": 0.35, "effect_tracking": 0.55,
        "naming_style": 0.8, "abstraction_level": 0.7,
    },
    "wen": {
        "state_magnitude": 0.35, "control_explicitness": 0.25,
        "typing_strength": 0.45, "composition_style": 0.15,
        "concurrency_model": 0.35, "effect_tracking": 0.5,
        "naming_style": 0.3, "abstraction_level": 0.85,
    },
}

# Dimension weights for cost computation (from paradigm_lattice.py).
_DIMENSION_WEIGHTS: Dict[str, float] = {
    "state_magnitude": 1.5,
    "control_explicitness": 1.0,
    "typing_strength": 1.2,
    "composition_style": 0.8,
    "concurrency_model": 1.4,
    "effect_tracking": 1.3,
    "naming_style": 0.5,
    "abstraction_level": 1.0,
}


@dataclass
class BridgeWarning:
    """A specific warning about what gets lost or degraded in a bridge.

    Attributes:
        category: What kind of information is affected.
        detail: Human-readable description.
        severity: 0.0 (minor) to 1.0 (critical).
    """
    category: str
    detail: str
    severity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "detail": self.detail,
            "severity": round(self.severity, 4),
        }


@dataclass
class BridgeCostReport:
    """Full cost report for bridging between two paradigms.

    Attributes:
        source_lang: Source paradigm.
        target_lang: Target paradigm.
        structural_distance: Euclidean distance in 8-D paradigm space.
        expressiveness_gap: How much expressiveness is lost [0.0, 1.0].
        information_loss: Fraction of type information lost [0.0, 1.0].
        translation_ambiguity: Ambiguity in the mapping [0.0, 1.0].
        total_cost: Weighted combination of all factors [0.0, 1.0].
        warnings: Specific warnings about what gets lost.
    """
    source_lang: LangTag
    target_lang: LangTag
    structural_distance: float = 0.0
    expressiveness_gap: float = 0.0
    information_loss: float = 0.0
    translation_ambiguity: float = 0.0
    total_cost: float = 0.0
    warnings: List[BridgeWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_lang,
            "target": self.target_lang,
            "structural_distance": round(self.structural_distance, 4),
            "expressiveness_gap": round(self.expressiveness_gap, 4),
            "information_loss": round(self.information_loss, 4),
            "translation_ambiguity": round(self.translation_ambiguity, 4),
            "total_cost": round(self.total_cost, 4),
            "warnings": [w.to_dict() for w in self.warnings],
        }


class BridgeCostMatrix:
    """Compute the multi-factor cost of bridging between any two paradigms.

    The cost model considers four orthogonal factors:

      1. **Structural distance** — Weighted Euclidean distance in the
         8-dimensional paradigm space (from paradigm_lattice.py).
      2. **Expressiveness gap** — Fraction of source type tags that have
         no equivalent in the target paradigm.
      3. **Information loss** — Weighted sum of preservation degrees
         for matched type tags (LOSSLESS=0, DEGRADED=1).
      4. **Translation ambiguity** — Number of valid target candidates
         per source tag (more candidates = more ambiguous).

    Usage:
        cost_matrix = BridgeCostMatrix(algebra)
        report = cost_matrix.compute("zho", "san")
        print(report.total_cost)
        for w in report.warnings:
            print(f"  WARNING: {w.detail}")
    """

    def __init__(
        self,
        algebra: Optional[TypeAlgebra] = None,
        dim_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.algebra = algebra or TypeAlgebra()
        self.dim_weights = dim_weights or dict(_DIMENSION_WEIGHTS)

    def compute(self, source: LangTag, target: LangTag) -> BridgeCostReport:
        """Compute the full bridge cost report between two paradigms.

        Args:
            source: Source language tag.
            target: Target language tag.

        Returns:
            A BridgeCostReport with all cost factors and warnings.
        """
        warnings: List[BridgeWarning] = []

        # Same language → zero cost
        if source == target:
            return BridgeCostReport(
                source_lang=source,
                target_lang=target,
                total_cost=0.0,
            )

        # 1. Structural distance
        structural = self._structural_distance(source, target)

        # 2. Expressiveness gap
        expr_gap, expr_warnings = self._expressiveness_gap(source, target)
        warnings.extend(expr_warnings)

        # 3. Information loss
        info_loss, info_warnings = self._information_loss(source, target)
        warnings.extend(info_warnings)

        # 4. Translation ambiguity
        ambiguity, amb_warnings = self._translation_ambiguity(source, target)
        warnings.extend(amb_warnings)

        # Combine into total cost
        total = (
            structural * 0.25 +
            expr_gap * 0.30 +
            info_loss * 0.30 +
            ambiguity * 0.15
        )
        total = min(total, 1.0)

        return BridgeCostReport(
            source_lang=source,
            target_lang=target,
            structural_distance=structural,
            expressiveness_gap=expr_gap,
            information_loss=info_loss,
            translation_ambiguity=ambiguity,
            total_cost=total,
            warnings=warnings,
        )

    def compare(self, a: LangTag, b: LangTag, c: LangTag) -> str:
        """Compare two bridge costs: a→b vs a→c.

        Returns:
            "cheaper" if a→b is cheaper, "more_expensive" otherwise.
            In case of a tie, returns "equal".
        """
        cost_ab = self.compute(a, b).total_cost
        cost_ac = self.compute(a, c).total_cost
        if cost_ab < cost_ac - 0.001:
            return "cheaper"
        elif cost_ab > cost_ac + 0.001:
            return "more_expensive"
        return "equal"

    def _structural_distance(self, a: LangTag, b: LangTag) -> float:
        """Compute weighted Euclidean distance in 8-D paradigm space."""
        coords_a = _LANGUAGE_COORDINATES.get(a, {})
        coords_b = _LANGUAGE_COORDINATES.get(b, {})
        if not coords_a or not coords_b:
            return 1.0  # Unknown language → maximum distance

        total = 0.0
        for dim in PARADIGM_DIMENSIONS:
            delta = coords_a.get(dim, 0.5) - coords_b.get(dim, 0.5)
            w = self.dim_weights.get(dim, 1.0)
            total += (delta * w) ** 2
        return min(math.sqrt(total) / 3.0, 1.0)  # Normalize to [0, 1]

    def _expressiveness_gap(
        self, source: LangTag, target: LangTag
    ) -> Tuple[float, List[BridgeWarning]]:
        """Fraction of source type tags with no equivalent in the target."""
        source_tags = _PARADIGM_TO_BASE.get(source, {})
        if not source_tags:
            return 1.0, []

        no_match = 0
        warnings: List[BridgeWarning] = []

        for tag in source_tags:
            slot = self.algebra.translate(source, tag, target)
            if slot is None:
                no_match += 1
                warnings.append(BridgeWarning(
                    category="expressiveness",
                    detail=f"No equivalent for {source}:{tag} in {target}",
                    severity=0.6,
                ))

        gap = no_match / len(source_tags)
        return gap, warnings

    def _information_loss(
        self, source: LangTag, target: LangTag
    ) -> Tuple[float, List[BridgeWarning]]:
        """Weighted information loss based on preservation degrees."""
        total_loss = 0.0
        matched = 0
        warnings: List[BridgeWarning] = []

        source_tags = _PARADIGM_TO_BASE.get(source, {})
        for tag in source_tags:
            cls = self.algebra.find_class(source, tag)
            if cls and cls.has_language(target):
                matched += 1
                # Map preservation degree to loss: LOSSLESS=0, DEGRADED=1
                degree_loss = {
                    PreservationDegree.LOSSLESS: 0.0,
                    PreservationDegree.NEAR_LOSSLESS: 0.1,
                    PreservationDegree.PARTIAL: 0.4,
                    PreservationDegree.LOSSY: 0.7,
                    PreservationDegree.DEGRADED: 1.0,
                }.get(cls.degree, 0.5)
                total_loss += degree_loss

                if cls.degree in (PreservationDegree.LOSSY, PreservationDegree.DEGRADED):
                    warnings.append(BridgeWarning(
                        category="information_loss",
                        detail=(
                            f"{source}:{tag} → {target} is {cls.degree.value}: "
                            f"{cls.name}"
                        ),
                        severity=degree_loss,
                    ))

        if matched == 0:
            return 1.0, warnings

        avg_loss = total_loss / matched
        # Also penalize unmatched tags
        unmatched = len(source_tags) - matched
        overall = (avg_loss * matched + 1.0 * unmatched) / len(source_tags)
        return overall, warnings

    def _translation_ambiguity(
        self, source: LangTag, target: LangTag
    ) -> Tuple[float, List[BridgeWarning]]:
        """Measure how ambiguous the mapping is (multiple targets per source)."""
        source_tags = _PARADIGM_TO_BASE.get(source, {})
        if not source_tags:
            return 0.0, []

        # Count target tags per source tag via equivalence classes
        ambiguous_count = 0
        warnings: List[BridgeWarning] = []
        max_ambiguity = 0

        for tag in source_tags:
            candidates: List[TypeEquivalenceSlot] = []
            for cls in self.algebra.all_classes():
                if cls.has_language(source) and cls.has_language(target):
                    src_slot = cls.get_slot(source)
                    tgt_slot = cls.get_slot(target)
                    if src_slot and src_slot.native_tag == tag and tgt_slot:
                        candidates.append(tgt_slot)
            if len(candidates) > 1:
                ambiguous_count += 1
                max_ambiguity = max(max_ambiguity, len(candidates))
                candidate_tags = [c.native_tag for c in candidates]
                warnings.append(BridgeWarning(
                    category="ambiguity",
                    detail=(
                        f"{source}:{tag} maps to {len(candidates)} targets "
                        f"in {target}: {candidate_tags}"
                    ),
                    severity=0.3,
                ))

        ambiguity = ambiguous_count / len(source_tags) if source_tags else 0.0
        return ambiguity, warnings


# ══════════════════════════════════════════════════════════════════
# 3. TypeWitness — Proof-Carrying Type Transformation Witnesses
# ══════════════════════════════════════════════════════════════════

@dataclass
class WitnessConstraint:
    """A single constraint that a witness must satisfy."""
    name: str
    expected: Any
    actual: Any
    satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "satisfied": self.satisfied,
        }


@dataclass
class TypeWitness:
    """Proof that a bridge transformation is correct.

    Every cross-paradigm type translation produces a TypeWitness that
    captures what was transformed, how, and what guarantees hold.

    Witnesses are:
      - **Serializable**: Can be stored as JSON for later audit.
      - **Verifiable**: ``verify()`` re-checks all constraints independently.
      - **Composable**: A chain of witnesses can be concatenated for
        multi-hop bridges.

    Attributes:
        witness_id: Unique identifier (UUID4).
        source_lang: Source paradigm.
        source_tag: Source native type tag.
        target_lang: Target paradigm.
        target_tag: Target native type tag.
        source_type: The full FluxType that was translated.
        target_type: The full FluxType that was produced.
        strategy: The BridgeStrategy used.
        preservation: The PreservationDegree of the mapping.
        equivalence_class_id: The TypeAlgebra class used (if any).
        constraints: The list of verified constraints.
        timestamp: Unix timestamp of witness creation.
        verified: Whether this witness has been independently verified.
    """
    witness_id: str = ""
    source_lang: LangTag = ""
    source_tag: NativeTag = ""
    target_lang: LangTag = ""
    target_tag: NativeTag = ""
    source_type: Optional[FluxType] = None
    target_type: Optional[FluxType] = None
    strategy: str = ""
    preservation: PreservationDegree = PreservationDegree.DEGRADED
    equivalence_class_id: str = ""
    constraints: List[WitnessConstraint] = field(default_factory=list)
    timestamp: float = 0.0
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.witness_id:
            self.witness_id = str(uuid.uuid4())[:12]
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

    @property
    def is_valid(self) -> bool:
        """Whether all constraints are satisfied."""
        return all(c.satisfied for c in self.constraints)

    @property
    def failed_constraints(self) -> List[WitnessConstraint]:
        """Return only the constraints that failed verification."""
        return [c for c in self.constraints if not c.satisfied]

    def verify(self) -> bool:
        """Independently verify all constraints.

        Re-checks each constraint's ``expected`` vs ``actual`` values.
        This can be called after deserialization to re-validate a stored
        witness.

        Returns:
            True if all constraints pass, False otherwise.
        """
        for c in self.constraints:
            c.satisfied = self._check_constraint(c)
        self.verified = True
        return self.is_valid

    @staticmethod
    def _check_constraint(c: WitnessConstraint) -> bool:
        """Check a single constraint, handling various comparison forms.

        Supports:
          - Direct equality: expected == actual
          - String prefix comparisons: ">=0.3" checks actual >= 0.3
          - Boolean checks: True/False
        """
        # Handle string-based threshold comparisons (e.g., ">=0.3")
        if isinstance(c.expected, str) and c.expected.startswith(">="):
            try:
                threshold = float(c.expected[2:])
                return float(c.actual) >= threshold
            except (ValueError, TypeError):
                return c.expected == c.actual
        if isinstance(c.expected, str) and c.expected.startswith(">"):
            try:
                threshold = float(c.expected[1:])
                return float(c.actual) > threshold
            except (ValueError, TypeError):
                return c.expected == c.actual
        if isinstance(c.expected, str) and c.expected.startswith("<="):
            try:
                threshold = float(c.expected[2:])
                return float(c.actual) <= threshold
            except (ValueError, TypeError):
                return c.expected == c.actual
        if isinstance(c.expected, str) and c.expected.startswith("<"):
            try:
                threshold = float(c.expected[1:])
                return float(c.actual) < threshold
            except (ValueError, TypeError):
                return c.expected == c.actual
        return c.expected == c.actual

    def chain(self, next_witness: TypeWitness) -> TypeWitness:
        """Compose this witness with another for multi-hop bridges.

        The resulting witness covers source→intermediate→target.
        Failed constraints from either hop are carried forward.

        Args:
            next_witness: The next hop's witness (intermediate→target).

        Returns:
            A new TypeWitness covering the full chain.
        """
        # Check that the chain is valid
        assert next_witness.source_lang == self.target_lang, (
            f"Chain mismatch: this target={self.target_lang}, "
            f"next source={next_witness.source_lang}"
        )

        # Use the weakest preservation degree
        degree_order = [
            PreservationDegree.LOSSLESS,
            PreservationDegree.NEAR_LOSSLESS,
            PreservationDegree.PARTIAL,
            PreservationDegree.LOSSY,
            PreservationDegree.DEGRADED,
        ]
        worst = max(
            degree_order.index(self.preservation),
            degree_order.index(next_witness.preservation),
        )

        chained = TypeWitness(
            source_lang=self.source_lang,
            source_tag=self.source_tag,
            target_lang=next_witness.target_lang,
            target_tag=next_witness.target_tag,
            source_type=self.source_type,
            target_type=next_witness.target_type,
            strategy=f"{self.strategy}→{next_witness.strategy}",
            preservation=degree_order[worst],
            equivalence_class_id=(
                f"{self.equivalence_class_id}+{next_witness.equivalence_class_id}"
                or ""
            ),
            constraints=self.constraints + next_witness.constraints,
        )
        chained.verified = self.verified and next_witness.verified
        return chained

    def to_dict(self) -> dict[str, Any]:
        """Serialize the witness to a JSON-compatible dict."""
        return {
            "witness_id": self.witness_id,
            "source": {"lang": self.source_lang, "tag": self.source_tag},
            "target": {"lang": self.target_lang, "tag": self.target_tag},
            "strategy": self.strategy,
            "preservation": self.preservation.value,
            "equivalence_class_id": self.equivalence_class_id,
            "constraints": [c.to_dict() for c in self.constraints],
            "source_type": self.source_type.to_dict() if self.source_type else None,
            "target_type": self.target_type.to_dict() if self.target_type else None,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "is_valid": self.is_valid,
        }

    def to_json(self) -> str:
        """Serialize the witness to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypeWitness:
        """Deserialize a witness from a dict."""
        constraints = [
            WitnessConstraint(**c) for c in data.get("constraints", [])
        ]
        return cls(
            witness_id=data.get("witness_id", ""),
            source_lang=data.get("source", {}).get("lang", ""),
            source_tag=data.get("source", {}).get("tag", ""),
            target_lang=data.get("target", {}).get("lang", ""),
            target_tag=data.get("target", {}).get("tag", ""),
            strategy=data.get("strategy", ""),
            preservation=PreservationDegree(data.get("preservation", "degraded")),
            equivalence_class_id=data.get("equivalence_class_id", ""),
            constraints=constraints,
            timestamp=data.get("timestamp", 0.0),
            verified=data.get("verified", False),
        )


class WitnessGenerator:
    """Factory for generating TypeWitness instances from bridge operations.

    Each witness proves that a specific type translation was performed
    correctly according to the TypeAlgebra equivalence classes.
    """

    def __init__(self, algebra: Optional[TypeAlgebra] = None) -> None:
        self.algebra = algebra or TypeAlgebra()

    def generate(
        self,
        source_type: FluxType,
        target_type: FluxType,
        strategy: BridgeStrategy,
        source_tag: Optional[NativeTag] = None,
        target_tag: Optional[NativeTag] = None,
    ) -> TypeWitness:
        """Generate a witness for a bridge translation.

        Args:
            source_type: The source FluxType.
            target_type: The translated FluxType.
            strategy: The bridge strategy used.
            source_tag: The native tag used for lookup (inferred if None).
            target_tag: The native tag produced (inferred if None).

        Returns:
            A TypeWitness with verified constraints.
        """
        src_lang = source_type.paradigm_source
        tgt_lang = target_type.paradigm_source

        # Infer native tags from constraints
        if source_tag is None:
            for c in source_type.constraints:
                if c.language == src_lang:
                    source_tag = str(c.value)
                    break
        if target_tag is None:
            for c in target_type.constraints:
                if c.language == tgt_lang:
                    target_tag = str(c.value)
                    break

        # Look up equivalence class
        equiv_cls = None
        preservation = PreservationDegree.DEGRADED
        class_id = ""

        if source_tag:
            equiv_cls = self.algebra.find_class(src_lang, source_tag)
            if equiv_cls:
                class_id = equiv_cls.class_id
                preservation = equiv_cls.degree

        # Build constraints
        constraints = self._build_constraints(
            source_type, target_type, equiv_cls, src_lang, tgt_lang,
            source_tag, target_tag,
        )

        witness = TypeWitness(
            source_lang=src_lang,
            source_tag=source_tag or "",
            target_lang=tgt_lang,
            target_tag=target_tag or "",
            source_type=source_type,
            target_type=target_type,
            strategy=strategy.value,
            preservation=preservation,
            equivalence_class_id=class_id,
            constraints=constraints,
        )
        witness.verify()
        return witness

    def _build_constraints(
        self,
        source_type: FluxType,
        target_type: FluxType,
        equiv_cls: Optional[TypeEquivalenceClass],
        src_lang: LangTag,
        tgt_lang: LangTag,
        source_tag: Optional[NativeTag],
        target_tag: Optional[NativeTag],
    ) -> List[WitnessConstraint]:
        """Build the list of verification constraints."""
        constraints: List[WitnessConstraint] = []

        # C1: Source and target paradigms must differ
        constraints.append(WitnessConstraint(
            name="paradigm_difference",
            expected="different",
            actual="different" if src_lang != tgt_lang else "same",
            satisfied=src_lang != tgt_lang,
        ))

        # C2: Base type must be in the same or adjacent spectrum position
        src_base = source_type.effective_base_type()
        tgt_base = target_type.effective_base_type()
        dist = src_base.spectrum_distance(tgt_base)
        constraints.append(WitnessConstraint(
            name="base_type_distance",
            expected=0,  # Ideally 0 (same base type)
            actual=int(dist),
            satisfied=dist <= 2,  # Allow up to 2 spectrum positions
        ))

        # C3: If equivalence class exists, both languages must be covered
        if equiv_cls:
            constraints.append(WitnessConstraint(
                name="equivalence_class_covers_source",
                expected=True,
                actual=equiv_cls.has_language(src_lang),
                satisfied=equiv_cls.has_language(src_lang),
            ))
            constraints.append(WitnessConstraint(
                name="equivalence_class_covers_target",
                expected=True,
                actual=equiv_cls.has_language(tgt_lang),
                satisfied=equiv_cls.has_language(tgt_lang),
            ))

        # C4: Confidence must not drop below a floor
        confidence_floor = 0.3
        constraints.append(WitnessConstraint(
            name="confidence_floor",
            expected=f">={confidence_floor}",
            actual=target_type.confidence,
            satisfied=target_type.confidence >= confidence_floor,
        ))

        # C5: Preservation degree must not be DEGRADED if both tags exist
        if source_tag and target_tag and equiv_cls:
            is_ok = equiv_cls.degree != PreservationDegree.DEGRADED
            constraints.append(WitnessConstraint(
                name="preservation_not_degraded",
                expected="not_degraded",
                actual="not_degraded" if is_ok else equiv_cls.degree.value,
                satisfied=is_ok,
            ))

        # C6: Source and target types must be non-null
        constraints.append(WitnessConstraint(
            name="source_type_exists",
            expected=True,
            actual=source_type is not None,
            satisfied=source_type is not None,
        ))
        constraints.append(WitnessConstraint(
            name="target_type_exists",
            expected=True,
            actual=target_type is not None,
            satisfied=target_type is not None,
        ))

        return constraints


# ══════════════════════════════════════════════════════════════════
# 4. TypeSafeBridge — Enhanced Type-Safe Cross-Paradigm Bridge
# ══════════════════════════════════════════════════════════════════

@dataclass
class SafeBridgeResult:
    """Result of a type-safe bridge translation.

    Extends the basic BridgeResult with type-safety guarantees:

    Attributes:
        bridge_result: The underlying TypeBridge result.
        witness: The TypeWitness proving the translation's correctness.
        cost_report: The BridgeCostReport quantifying the translation cost.
        preservation: The PreservationDegree of this specific translation.
        refinement_applied: Whether an incremental refinement was used.
    """
    bridge_result: BridgeResult
    witness: TypeWitness
    cost_report: BridgeCostReport
    preservation: PreservationDegree = PreservationDegree.DEGRADED
    refinement_applied: bool = False

    @property
    def target_type(self) -> FluxType:
        """Shortcut to the translated FluxType."""
        return self.bridge_result.target_type

    @property
    def is_safe(self) -> bool:
        """Whether the translation passed all witness checks."""
        return self.witness.is_valid

    @property
    def warnings(self) -> List[str]:
        """Combined warnings from bridge, cost report, and witness."""
        all_warnings: List[str] = list(self.bridge_result.warnings)
        for w in self.cost_report.warnings:
            all_warnings.append(f"[{w.category}] {w.detail}")
        for fc in self.witness.failed_constraints:
            all_warnings.append(f"[witness] {fc.name}: expected={fc.expected}, got={fc.actual}")
        return all_warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.bridge_result.to_dict(),
            "witness": self.witness.to_dict(),
            "cost": self.cost_report.to_dict(),
            "preservation": self.preservation.value,
            "is_safe": self.is_safe,
            "refinement_applied": self.refinement_applied,
            "warnings": self.warnings,
        }


# Registry for incremental refinement rules.
# Maps (source_lang, source_tag, target_lang) → (target_tag, preservation).
RefinementRule: TypeAlias = Tuple[
    LangTag, NativeTag, LangTag, NativeTag, PreservationDegree
]


class TypeSafeBridge:
    """Type-safe cross-paradigm bridge with witness generation.

    Enhances the existing TypeBridge with:
      - Bidirectional type maps via TypeAlgebra
      - Preservation guarantees (LOSSLESS through DEGRADED)
      - Multi-factor cost computation via BridgeCostMatrix
      - Proof-carrying translations via TypeWitness
      - Incremental refinement of lossy mappings

    Usage:
        bridge = TypeSafeBridge()
        result = bridge.translate_safe(zho_type, target_lang="deu")
        if result.is_safe:
            print(f"Translated to {result.target_type} (fidelity={result.bridge_result.fidelity:.2f})")
        else:
            print(f"Warnings: {result.warnings}")

        # Serialize witness for audit
        witness_json = result.witness.to_json()
    """

    # Known bidirectional refinement rules.  These can be incrementally
    # added to improve the precision of lossy default mappings.
    _DEFAULT_REFINEMENTS: List[RefinementRule] = [
        # ZHO generic → DEU neutrum (refine from DEGRADED to PARTIAL)
        ("zho", "generic", "deu", "neutrum", PreservationDegree.PARTIAL),
        # KOR honorific → LAT indicative (refine from DEGRADED to PARTIAL)
        ("kor", "hasipsioche", "lat", "praesens", PreservationDegree.PARTIAL),
        # KOR informal → LAT imperfect (refine from DEGRADED to PARTIAL)
        ("kor", "haeche", "lat", "imperfectum", PreservationDegree.PARTIAL),
        # WEN topic → ZHO generic (contextual scoping equivalence)
        ("wen", "topic", "zho", "generic", PreservationDegree.PARTIAL),
        # WEN loop → LAT imperfect (temporal equivalence)
        ("wen", "loop", "lat", "imperfectum", PreservationDegree.PARTIAL),
    ]

    def __init__(
        self,
        registry: Optional[FluxTypeRegistry] = None,
        algebra: Optional[TypeAlgebra] = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.algebra = algebra or TypeAlgebra()
        self.cost_matrix = BridgeCostMatrix(self.algebra)
        self.witness_gen = WitnessGenerator(self.algebra)
        self._inner_bridge = TypeBridge(self.registry)
        self._refinements: Dict[
            Tuple[LangTag, NativeTag, LangTag],
            Tuple[NativeTag, PreservationDegree],
        ] = {}
        self._load_default_refinements()

    def _load_default_refinements(self) -> None:
        """Load the built-in refinement rules."""
        for src_lang, src_tag, tgt_lang, tgt_tag, degree in self._DEFAULT_REFINEMENTS:
            self._refinements[(src_lang, src_tag, tgt_lang)] = (tgt_tag, degree)

    def add_refinement(
        self,
        source_lang: LangTag,
        source_tag: NativeTag,
        target_lang: LangTag,
        target_tag: NativeTag,
        degree: PreservationDegree = PreservationDegree.NEAR_LOSSLESS,
    ) -> None:
        """Add an incremental refinement rule.

        Refinement rules override the default (often DEGRADED) mapping
        for a specific (source, tag, target) triple with a more precise
        target tag and a better preservation degree.

        Args:
            source_lang: Source language.
            source_tag: Source native type tag.
            target_lang: Target language.
            target_tag: Target native type tag.
            degree: The preservation degree of this refined mapping.
        """
        self._refinements[(source_lang, source_tag, target_lang)] = (
            target_tag, degree
        )

    def translate_safe(
        self,
        source: FluxType,
        target_lang: LangTag,
        strategy: Optional[BridgeStrategy] = None,
    ) -> SafeBridgeResult:
        """Translate a FluxType to the target paradigm with full type safety.

        This method wraps the underlying TypeBridge.translate() with:
          1. TypeAlgebra lookup for optimal mappings
          2. BridgeCostMatrix computation for cost analysis
          3. TypeWitness generation for proof-carrying output
          4. Incremental refinement application

        Args:
            source: The source FluxType.
            target_lang: The target language paradigm.
            strategy: Optional BridgeStrategy (auto-detected if None).

        Returns:
            A SafeBridgeResult with the translated type, witness, and cost.
        """
        src_lang = source.paradigm_source
        src_tag = self._infer_tag(source)

        # 1. Try TypeAlgebra first for precise mapping
        algebra_slot = None
        if src_tag:
            algebra_slot = self.algebra.translate(src_lang, src_tag, target_lang)

        # 2. Try refinement rules
        refinement = self._refinements.get((src_lang, src_tag or "", target_lang))

        # 3. Determine the best approach
        refinement_applied = False

        if algebra_slot:
            # Use algebra-based translation
            target_type = FluxType.from_paradigm(
                algebra_slot.language,
                algebra_slot.native_tag,
                confidence=source.confidence,
                name=f"{target_lang}:{algebra_slot.native_tag}",
            )
            equiv_cls = self.algebra.find_class(src_lang, src_tag or "")
            preservation = equiv_cls.degree if equiv_cls else PreservationDegree.DEGRADED
            used_strategy = BridgeStrategy.DIRECT

        elif refinement:
            # Use refinement rule
            ref_tag, ref_degree = refinement
            try:
                target_type = FluxType.from_paradigm(
                    target_lang, ref_tag,
                    confidence=source.confidence * 0.9,
                    name=f"{target_lang}:{ref_tag}",
                )
            except KeyError:
                target_type = self._fallback_translate(source, target_lang, strategy)
            preservation = ref_degree
            used_strategy = BridgeStrategy.CONSTRAINT_PRESERVATION
            refinement_applied = True

        else:
            # Fall back to inner TypeBridge
            inner_result = self._inner_bridge.translate(source, target_lang, strategy)
            target_type = inner_result.target_type
            used_strategy = inner_result.strategy
            preservation = PreservationDegree.DEGRADED

        # 4. Compute cost report
        cost_report = self.cost_matrix.compute(src_lang, target_lang)

        # 5. Build bridge result
        fid = 1.0 if preservation in (
            PreservationDegree.LOSSLESS, PreservationDegree.NEAR_LOSSLESS
        ) else (0.8 if preservation == PreservationDegree.PARTIAL else 0.6)

        bridge_result = BridgeResult(
            target_type=target_type,
            strategy=used_strategy,
            fidelity=fid,
            cost=cost_report.total_cost,
        )

        # 6. Generate witness
        tgt_tag = self._infer_tag(target_type)
        witness = self.witness_gen.generate(
            source, target_type, used_strategy, src_tag, tgt_tag,
        )

        return SafeBridgeResult(
            bridge_result=bridge_result,
            witness=witness,
            cost_report=cost_report,
            preservation=preservation,
            refinement_applied=refinement_applied,
        )

    def translate_round_trip(
        self,
        source: FluxType,
        via_lang: LangTag,
        back_lang: LangTag,
    ) -> SafeBridgeResult:
        """Perform a round-trip translation: source → via → back.

        Produces a chained witness covering the full round-trip.

        Args:
            source: The original FluxType.
            via_lang: The intermediate language.
            back_lang: The final language (typically = source's language).

        Returns:
            A SafeBridgeResult for the final leg, with a chained witness.
        """
        # Leg 1: source → via
        leg1 = self.translate_safe(source, via_lang)
        # Leg 2: via → back
        leg2 = self.translate_safe(leg1.target_type, back_lang)

        # Chain witnesses
        chained_witness = leg1.witness.chain(leg2.witness)

        # Build combined cost
        combined_cost = BridgeCostReport(
            source_lang=source.paradigm_source,
            target_lang=back_lang,
            structural_distance=leg1.cost_report.structural_distance + leg2.cost_report.structural_distance,
            expressiveness_gap=max(
                leg1.cost_report.expressiveness_gap,
                leg2.cost_report.expressiveness_gap,
            ),
            information_loss=max(
                leg1.cost_report.information_loss,
                leg2.cost_report.information_loss,
            ),
            translation_ambiguity=max(
                leg1.cost_report.translation_ambiguity,
                leg2.cost_report.translation_ambiguity,
            ),
            total_cost=leg1.cost_report.total_cost + leg2.cost_report.total_cost,
            warnings=leg1.cost_report.warnings + leg2.cost_report.warnings,
        )

        # Worst preservation degree
        degree_order = [
            PreservationDegree.LOSSLESS,
            PreservationDegree.NEAR_LOSSLESS,
            PreservationDegree.PARTIAL,
            PreservationDegree.LOSSY,
            PreservationDegree.DEGRADED,
        ]
        worst = max(
            degree_order.index(leg1.preservation),
            degree_order.index(leg2.preservation),
        )

        return SafeBridgeResult(
            bridge_result=leg2.bridge_result,
            witness=chained_witness,
            cost_report=combined_cost,
            preservation=degree_order[worst],
        )

    def get_bidirectional_map(
        self, lang_a: LangTag, lang_b: LangTag
    ) -> Dict[NativeTag, Optional[NativeTag]]:
        """Return the bidirectional type map between two languages.

        For each type tag in lang_a, returns the corresponding tag in lang_b
        (via TypeAlgebra), or None if no mapping exists.

        Args:
            lang_a: First language.
            lang_b: Second language.

        Returns:
            Dict mapping lang_a tags to lang_b tags (or None).
        """
        a_tags = _PARADIGM_TO_BASE.get(lang_a, {})
        b_map: Dict[NativeTag, Optional[NativeTag]] = {}
        for tag in a_tags:
            slot = self.algebra.translate(lang_a, tag, lang_b)
            b_map[tag] = slot.native_tag if slot else None
        return b_map

    def _infer_tag(self, flux_type: FluxType) -> Optional[NativeTag]:
        """Infer the native tag from a FluxType's constraints."""
        for c in flux_type.constraints:
            if c.language == flux_type.paradigm_source:
                return str(c.value)
        return None

    def _fallback_translate(
        self,
        source: FluxType,
        target_lang: LangTag,
        strategy: Optional[BridgeStrategy],
    ) -> FluxType:
        """Fallback translation when no algebra or refinement matches."""
        result = self._inner_bridge.translate(source, target_lang, strategy)
        return result.target_type
