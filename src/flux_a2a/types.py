"""
Flux Universal Type System (FUTS) — Cross-Language Type Unification.

Implements a unified type system that bridges 6 natural-language-derived type
paradigms into a coherent, programmable type framework.  FUTS draws on ideas
from gradual typing, session types, algebraic effects, row polymorphism,
dependent types, and Rust's trait system to solve the "paradigm gap" problem.

Core Design Principles (from Round 10 survey):
  1. Types are PROBABILISTIC, not absolute — every FluxType carries a confidence
  2. Constraints are LAYERED — language-specific constraints sit atop universal
     base types, not in competition with them
  3. The system is OPEN — new base types and constraints can be registered at
     runtime (row-polymorphic extensibility)
  4. Compatibility is MEASURED, not binary — TypeCompatibility produces a real-
     valued similarity score, not a pass/fail
  5. Translation preserves SEMANTICS — the TypeBridge accounts for paradigm
     distance when converting between language-specific type views

Architecture:
  FluxType
    ├── base_type: FluxBaseType       (8 universal categories)
    ├── constraints: [FluxConstraint] (language-specific refinements)
    ├── confidence: float              (quantum-register confidence)
    ├── paradigm_source: str           (originating NL paradigm)
    ├── quantum_state: Optional[...]   (superposition of possible types)
    └── meta: dict[str, Any]           (extensible metadata)

Reference: docs/round10-12_type_unification.md
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. FluxBaseType — Universal Type Categories
# ══════════════════════════════════════════════════════════════════

class FluxBaseType(IntEnum):
    """The 8 universal type categories of FUTS.

    Each category is derived from one or more NL paradigms but abstracted
    to a language-independent universal.  The IntEnum ordering reflects
    the "type spectrum" from pure data (VALUE) to pure context (CONTEXTUAL).

    Type Spectrum:
        VALUE ──► ACTIVE ──► CONTAINER ──► SCOPE ──►
        CAPABILITY ──► MODAL ──► UNCERTAIN ──► CONTEXTUAL

    Interpretation:
        VALUE      — Pure data with no behavior.  Maps to: Latin Neutrum,
                     Chinese 量词 for flat objects (張 for paper, 條 for rivers).
        ACTIVE     — Has agency / can initiate actions.  Maps to: Latin
                     Maskulinum, German "Active" Geschlecht, Korean subject
                     honorification.
        CONTAINER  — Holds other values.  Maps to: Latin Femininum, German
                     "Container" Geschlecht, Chinese collective 量词 (群, 組).
        SCOPE      — Defines accessibility / visibility.  Maps to: Sanskrit
                     vibhakti (8 case levels), German Kasus (4 cases).
        CAPABILITY — Encodes access permissions.  Maps to: Korean 경어
                     (7 honorific levels, compressed to 5 here), Sanskrit
                     pada (active/middle voice).
        MODAL      — Execution mode / temporal aspect.  Maps to: Latin tempus
                     (6 tenses), Sanskrit lakāra (10 tenses/moods).
        UNCERTAIN  — Superposition of multiple possible types.  Maps to:
                     Chinese quantum registers (classifier ambiguity in
                     border cases), FLUX confidence propagation.
        CONTEXTUAL — Type determined entirely by runtime context.  Maps to:
                     Classical Chinese 文境 (context-dependence), zero-
                     anaphora resolution.
    """
    VALUE = 0
    ACTIVE = 1
    CONTAINER = 2
    SCOPE = 3
    CAPABILITY = 4
    MODAL = 5
    UNCERTAIN = 6
    CONTEXTUAL = 7

    @classmethod
    def from_paradigm(cls, lang: str, native_tag: str) -> FluxBaseType:
        """Map a language-specific type tag to a universal base type.

        Args:
            lang: Language tag (zho, deu, kor, san, wen, lat).
            native_tag: Language-specific type identifier.

        Returns:
            The corresponding FluxBaseType.

        Raises:
            KeyError: If the mapping is unknown.

        Examples:
            >>> FluxBaseType.from_paradigm("lat", "maskulinum")
            <FluxBaseType.ACTIVE: 1>
            >>> FluxBaseType.from_paradigm("san", "prathama")
            <FluxBaseType.SCOPE: 3>
            >>> FluxBaseType.from_paradigm("zho", "flat_object")
            <FluxBaseType.VALUE: 0>
        """
        return _PARADIGM_TO_BASE[lang][native_tag]

    @classmethod
    def paradigm_tags(cls, lang: str) -> Dict[str, FluxBaseType]:
        """Return all known type tags for a given language paradigm."""
        return dict(_PARADIGM_TO_BASE.get(lang, {}))

    def spectrum_distance(self, other: FluxBaseType) -> float:
        """Distance on the type spectrum [0.0, 7.0].

        Adjacent types have distance 1.0; antipodal types have distance 7.0.
        This is a rough proxy for conversion cost.
        """
        return abs(int(self) - int(other))


# ── Paradigm → Base Type Mapping Tables ──────────────────────────

# Chinese 量词 (classifiers) → FluxBaseType
_ZHO_CLASSIFIERS: Dict[str, FluxBaseType] = {
    # Flat objects → VALUE
    "flat_object":       FluxBaseType.VALUE,     # 張 (paper), 片 (leaf)
    "long_flexible":     FluxBaseType.VALUE,     # 條 (river, rope)
    "small_round":       FluxBaseType.VALUE,     # 顆 (pearl, star)
    "granular":          FluxBaseType.VALUE,     # 粒 (grain)
    # Active agents → ACTIVE
    "person":            FluxBaseType.ACTIVE,    # 位 (person, honorific)
    "animal":            FluxBaseType.ACTIVE,    # 隻 (animal)
    "machine":           FluxBaseType.ACTIVE,    # 台 (machine)
    # Containers → CONTAINER
    "collective":        FluxBaseType.CONTAINER, # 群 (group), 組 (set)
    "pair":              FluxBaseType.CONTAINER, # 雙 (pair)
    "volume":            FluxBaseType.CONTAINER, # 杯 (cup), 瓶 (bottle)
    # Scope → SCOPE
    "flat_surface":      FluxBaseType.SCOPE,     # 面 (surface, face)
    "spatial_extent":    FluxBaseType.SCOPE,     # 段 (segment), 節 (section)
    # Uncertain → UNCERTAIN
    "indeterminate":     FluxBaseType.UNCERTAIN, # 些 (some, uncertain quantity)
    # Contextual → CONTEXTUAL
    "generic":           FluxBaseType.CONTEXTUAL, # 個 (generic — most common)
    "abstract_action":   FluxBaseType.CONTEXTUAL, # 次 (occurrence, action)
}

# German Geschlecht (gender) → FluxBaseType
# Following the Active/Container/Data model from the project spec
_DEU_GESCHLECHT: Dict[str, FluxBaseType] = {
    "maskulinum":        FluxBaseType.ACTIVE,    # der — active, initiates
    "femininum":         FluxBaseType.CONTAINER, # die — holds, receives
    "neutrum":           FluxBaseType.VALUE,     # das — pure data
    # Kasus (case) → SCOPE
    "nominativ":         FluxBaseType.SCOPE,     # subject
    "akkusativ":         FluxBaseType.SCOPE,     # direct object
    "dativ":             FluxBaseType.SCOPE,     # indirect object
    "genitiv":           FluxBaseType.SCOPE,     # possessive
}

# Korean 경어 (honorifics) → FluxBaseType
_KOR_HONORIFICS: Dict[str, FluxBaseType] = {
    # Honorific levels → CAPABILITY
    "hasipsioche":       FluxBaseType.CAPABILITY, # 하십시오체 — formal highest
    "haeyoche":          FluxBaseType.CAPABILITY, # 해요체 — polite
    "haeche":            FluxBaseType.CAPABILITY, # 해체 — informal
    "haerache":          FluxBaseType.CAPABILITY, # 해라체 — plain
    "haeraoche":         FluxBaseType.CAPABILITY, # 해라오체 — blunt
    # Subject markers → ACTIVE / SCOPE
    "subject_honorific": FluxBaseType.ACTIVE,     # 주체 높임
    "object_honorific":  FluxBaseType.SCOPE,      # 객체 높임
    # Speech act types → MODAL
    "declarative":       FluxBaseType.MODAL,      # 평서문
    "interrogative":     FluxBaseType.MODAL,      # 의문문
    "imperative":        FluxBaseType.MODAL,      # 명령문
    "propositive":       FluxBaseType.MODAL,      # 청유문
}

# Sanskrit vibhakti (cases) → FluxBaseType
_SAN_VIBHAKTI: Dict[str, FluxBaseType] = {
    "prathama":          FluxBaseType.SCOPE,  # प्रथमा — nominative (subject)
    "dvitiya":           FluxBaseType.SCOPE,  # द्वितीया — accusative (object)
    "tritiya":           FluxBaseType.SCOPE,  # तृतीया — instrumental (means)
    "chaturthi":         FluxBaseType.SCOPE,  # चतुर्थी — dative (purpose)
    "panchami":          FluxBaseType.SCOPE,  # पंचमी — ablative (origin)
    "shashthi":          FluxBaseType.SCOPE,  # षष्ठी — genitive (possessive)
    "saptami":           FluxBaseType.SCOPE,  # सप्तमी — locative (location)
    "sambodhana":        FluxBaseType.SCOPE,  # संबोधन — vocative (address)
    # Gender → base type mapping
    "linga_pushkara":    FluxBaseType.ACTIVE,    # पुंल्लिङ्ग — masculine
    "linga_stri":        FluxBaseType.CONTAINER, # स्त्रीलिङ्ग — feminine
    "linga_napumsaka":   FluxBaseType.VALUE,     # नपुंसकलिङ्ग — neuter
    # Number
    "ekavachana":        FluxBaseType.VALUE,     # एकवचन — singular
    "dvivachana":        FluxBaseType.CONTAINER, # द्विवचन — dual
    "bahuvachana":       FluxBaseType.CONTEXTUAL,# बहुवचन — plural (contextual)
}

# Classical Chinese 文境 → FluxBaseType
_WEN_WENJING: Dict[str, FluxBaseType] = {
    # Confucian 五常 (Five Constants) → CAPABILITY
    "ren":               FluxBaseType.CAPABILITY, # 仁 — benevolence
    "yi":                FluxBaseType.CAPABILITY, # 義 — righteousness
    "li":                FluxBaseType.CAPABILITY, # 禮 — propriety
    "zhi":               FluxBaseType.CAPABILITY, # 智 — wisdom
    "xin":               FluxBaseType.CAPABILITY, # 信 — trustworthiness
    # Military strategy → MODAL
    "attack":            FluxBaseType.MODAL,      # 攻 — attack
    "defend":            FluxBaseType.MODAL,      # 守 — defend
    "advance":           FluxBaseType.MODAL,      # 進 — advance
    "retreat":           FluxBaseType.MODAL,      # 退 — retreat
    # Control structures → MODAL
    "sequence":          FluxBaseType.MODAL,      # 則 — sequence
    "loop":              FluxBaseType.MODAL,      # 循 — loop
    # Topic-comment → CONTEXTUAL
    "topic":             FluxBaseType.CONTEXTUAL,  # 定題 — set topic
    "zero_anaphora":     FluxBaseType.CONTEXTUAL,  # 省略 — implicit reference
    "context_resolved":  FluxBaseType.CONTEXTUAL,  # 文境 — context-dependent
}

# Latin tempus (tenses) → FluxBaseType
_LAT_TEMPUS: Dict[str, FluxBaseType] = {
    "praesens":          FluxBaseType.MODAL,  # Present — active execution
    "imperfectum":       FluxBaseType.MODAL,  # Imperfect — continuous loop
    "perfectum":         FluxBaseType.MODAL,  # Perfect — completed cache
    "plusquamperfectum": FluxBaseType.MODAL,  # Pluperfect — rollback point
    "futurum":           FluxBaseType.MODAL,  # Future — deferred computation
    "futurum_exactum":   FluxBaseType.MODAL,  # Future Perfect — scheduled
    # Gender → base type
    "maskulinum":        FluxBaseType.ACTIVE,    # masculine
    "femininum":         FluxBaseType.CONTAINER, # feminine
    "neutrum":           FluxBaseType.VALUE,     # neuter
    # Case → scope
    "nominativus":       FluxBaseType.SCOPE,     # nominative
    "accusativus":       FluxBaseType.SCOPE,     # accusative
    "genitivus":         FluxBaseType.SCOPE,     # genitive
    "dativus":           FluxBaseType.SCOPE,     # dative
    "ablativus":         FluxBaseType.SCOPE,     # ablative
    "vocativus":         FluxBaseType.SCOPE,     # vocative
}

# Master mapping: paradigm → {native_tag → FluxBaseType}
_PARADIGM_TO_BASE: Dict[str, Dict[str, FluxBaseType]] = {
    "zho": _ZHO_CLASSIFIERS,
    "deu": _DEU_GESCHLECHT,
    "kor": _KOR_HONORIFICS,
    "san": _SAN_VIBHAKTI,
    "wen": _WEN_WENJING,
    "lat": _LAT_TEMPUS,
}


# ══════════════════════════════════════════════════════════════════
# 2. FluxConstraint — Language-Specific Type Constraints
# ══════════════════════════════════════════════════════════════════

class ConstraintKind(str, Enum):
    """Categories of type constraints."""
    # Classifier constraints (ZHO)
    CLASSIFIER_SHAPE = "classifier_shape"
    CLASSIFIER_ANIMACY = "classifier_animacy"
    CLASSIFIER_COUNTABILITY = "classifier_countability"
    # Gender constraints (DEU, SAN, LAT)
    GENDER_AGREEMENT = "gender_agreement"
    CASE_MARKING = "case_marking"
    NUMBER_AGREEMENT = "number_agreement"
    # Honorific constraints (KOR)
    HONORIFIC_LEVEL = "honorific_level"
    SPEECH_ACT = "speech_act"
    # Scope constraints (SAN, DEU)
    SCOPE_LEVEL = "scope_level"
    SCOPE_ACCESS = "scope_access"
    # Temporal/modal constraints (LAT)
    TEMPORAL_ASPECT = "temporal_aspect"
    EXECUTION_MODE = "execution_mode"
    # Context constraints (WEN)
    CONTEXT_DOMAIN = "context_domain"
    TOPIC_REGISTER = "topic_register"
    # Universal constraints
    CONFIDENCE_BOUND = "confidence_bound"
    TRUST_REQUIREMENT = "trust_requirement"
    EFFECT_SET = "effect_set"


@dataclass(slots=True)
class FluxConstraint:
    """A language-specific constraint layered on top of a FluxBaseType.

    Constraints refine the universal base type with language-specific
    details.  Multiple constraints can be combined (conjunctively) on
    a single FluxType.

    Attributes:
        kind: The category of constraint.
        language: The originating language paradigm.
        value: The constraint value (type-dependent).
        confidence: How certain we are about this constraint.
        meta: Extensible metadata.

    Examples:
        >>> c = FluxConstraint(
        ...     kind=ConstraintKind.CLASSIFIER_SHAPE,
        ...     language="zho",
        ...     value="flat_object",
        ...     confidence=0.95,
        ... )
    """
    kind: ConstraintKind
    language: str
    value: Any
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def compatible_with(self, other: FluxConstraint) -> float:
        """Measure compatibility between two constraints [0.0, 1.0].

        Returns 1.0 if identical, 0.0 if contradictory, and intermediate
        values for partially compatible constraints.
        """
        if self.kind != other.kind:
            return 0.5  # Different constraint kinds are orthogonal
        if self.value == other.value:
            return 1.0
        if self.language == other.language:
            # Same language, different value → likely contradictory
            return 0.1
        # Different languages, same kind → may be compatible
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "language": self.language,
            "value": self.value,
            "confidence": round(self.confidence, 6),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FluxConstraint:
        return cls(
            kind=ConstraintKind(data["kind"]),
            language=data["language"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            meta=data.get("meta", {}),
        )


# ══════════════════════════════════════════════════════════════════
# 3. QuantumTypeState — Superposition of Possible Types
# ══════════════════════════════════════════════════════════════════

@dataclass
class QuantumTypeState:
    """Represents a superposition of possible FluxTypes (quantum registers).

    In FLUX, types can be in superposition when the classifier or grammar
    is ambiguous.  This is analogous to quantum computing's superposition:
    the type is not definitively VALUE, ACTIVE, or CONTAINER until
    observation (runtime evidence) collapses the state.

    Attributes:
        possibilities: List of (FluxBaseType, amplitude) pairs.
                      Amplitude is the probability weight [0.0, 1.0].
        collapsed: Whether the state has been observed/collapsed.
        collapsed_type: The type chosen after collapse.
    """
    possibilities: List[Tuple[FluxBaseType, float]] = field(default_factory=list)
    collapsed: bool = False
    collapsed_type: Optional[FluxBaseType] = None

    def __post_init__(self) -> None:
        # Normalize amplitudes to sum to 1.0
        total = sum(amp for _, amp in self.possibilities)
        if total > 0:
            self.possibilities = [
                (bt, amp / total) for bt, amp in self.possibilities
            ]

    def add_possibility(self, base_type: FluxBaseType, amplitude: float = 1.0) -> None:
        """Add a possible type to the superposition."""
        self.possibilities.append((base_type, amplitude))
        total = sum(amp for _, amp in self.possibilities)
        if total > 0:
            self.possibilities = [
                (bt, amp / total) for bt, amp in self.possibilities
            ]
        self.collapsed = False
        self.collapsed_type = None

    def observe(self, evidence: Optional[Dict[FluxBaseType, float]] = None) -> FluxBaseType:
        """Collapse the superposition based on evidence.

        Args:
            evidence: Optional evidence weights for each type.

        Returns:
            The collapsed FluxBaseType.

        If evidence is provided, amplitudes are updated Bayesian-style
        before collapse.  If not, the type with highest amplitude wins.
        """
        if self.collapsed and self.collapsed_type is not None:
            return self.collapsed_type

        if not self.possibilities:
            self.collapsed_type = FluxBaseType.VALUE
            self.collapsed = True
            return self.collapsed_type

        if evidence:
            # Bayesian update: multiply amplitude by evidence weight
            updated = []
            for bt, amp in self.possibilities:
                ev = evidence.get(bt, 0.5)
                updated.append((bt, amp * ev))
            total = sum(a for _, a in updated)
            if total > 0:
                self.possibilities = [
                    (bt, a / total) for bt, a in updated
                ]

        # Collapse: highest amplitude wins
        best_type, _ = max(self.possibilities, key=lambda x: x[1])
        self.collapsed_type = best_type
        self.collapsed = True
        return best_type

    def entropy(self) -> float:
        """Shannon entropy of the superposition [0.0, log2(n)].

        0.0 = fully collapsed (one type has probability 1.0).
        Higher values = more uncertainty.
        """
        if self.collapsed or len(self.possibilities) <= 1:
            return 0.0
        h = 0.0
        for _, amp in self.possibilities:
            if amp > 0:
                h -= amp * math.log2(amp)
        return h

    def to_dict(self) -> dict[str, Any]:
        return {
            "possibilities": [
                {"type": bt.name, "amplitude": round(amp, 6)}
                for bt, amp in self.possibilities
            ],
            "collapsed": self.collapsed,
            "collapsed_type": self.collapsed_type.name if self.collapsed_type else None,
            "entropy": round(self.entropy(), 6),
        }


# ══════════════════════════════════════════════════════════════════
# 4. FluxType — The Universal Type
# ══════════════════════════════════════════════════════════════════

@dataclass
class FluxType:
    """The universal type for all FLUX languages.

    A FluxType is the intersection of:
      - A universal base type (FluxBaseType)
      - Zero or more language-specific constraints (FluxConstraint)
      - A confidence score (from quantum registers)
      - Optionally, a quantum superposition state (QuantumTypeState)

    Design rationale (Round 10 survey influences):
      - Gradual typing: types are optional; confidence < 1.0 means "unsure"
      - Row polymorphism: constraints form an extensible row of annotations
      - Dependent types: CONTEXTUAL types depend on runtime values
      - Algebraic effects: MODAL types encode execution effects
      - Session types: CAPABILITY types encode communication channels
      - Rust traits: SCOPE types work like trait bounds

    Attributes:
        base_type: The universal base category.
        constraints: Language-specific refinements.
        confidence: Type certainty [0.0, 1.0].
        paradigm_source: Which NL paradigm this type originates from.
        quantum_state: Optional superposition of possible types.
        name: Optional human-readable name.
        meta: Extensible metadata.

    Examples:
        >>> t = FluxType(
        ...     base_type=FluxBaseType.CONTAINER,
        ...     constraints=[
        ...         FluxConstraint(
        ...             kind=ConstraintKind.CLASSIFIER_SHAPE,
        ...             language="zho",
        ...             value="collective",
        ...         ),
        ...     ],
        ...     confidence=0.92,
        ...     paradigm_source="zho",
        ...     name="ChineseCollectiveClassifier",
        ... )
    """
    base_type: FluxBaseType
    constraints: List[FluxConstraint] = field(default_factory=list)
    confidence: float = 1.0
    paradigm_source: str = "flux"
    quantum_state: Optional[QuantumTypeState] = None
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    # ── Construction helpers ────────────────────────────────────

    @classmethod
    def from_paradigm(
        cls,
        lang: str,
        native_tag: str,
        confidence: float = 1.0,
        extra_constraints: Optional[List[FluxConstraint]] = None,
        name: str = "",
    ) -> FluxType:
        """Create a FluxType from a language-specific type tag.

        Args:
            lang: Language tag (zho, deu, kor, san, wen, lat).
            native_tag: Language-specific type identifier.
            confidence: Initial confidence.
            extra_constraints: Additional constraints.
            name: Optional human-readable name.

        Returns:
            A FluxType with the base type mapped from the paradigm tag.

        Raises:
            KeyError: If the lang/tag combination is unknown.
        """
        base = FluxBaseType.from_paradigm(lang, native_tag)
        constraints = [
            FluxConstraint(
                kind=_infer_constraint_kind(lang, native_tag),
                language=lang,
                value=native_tag,
                confidence=confidence,
            )
        ]
        if extra_constraints:
            constraints.extend(extra_constraints)

        return cls(
            base_type=base,
            constraints=constraints,
            confidence=confidence,
            paradigm_source=lang,
            name=name or f"{lang}:{native_tag}",
        )

    @classmethod
    def uncertain(
        cls,
        possibilities: List[Tuple[FluxBaseType, float]],
        paradigm_source: str = "flux",
        name: str = "",
    ) -> FluxType:
        """Create a FluxType in superposition (UNCERTAIN).

        Args:
            possibilities: List of (base_type, amplitude) pairs.
            paradigm_source: Originating paradigm.
            name: Optional name.

        Returns:
            A FluxType with UNCERTAIN base and quantum superposition.
        """
        quantum = QuantumTypeState(possibilities=possibilities)
        return cls(
            base_type=FluxBaseType.UNCERTAIN,
            confidence=quantum.entropy() / max(math.log2(len(possibilities)), 1.0),
            paradigm_source=paradigm_source,
            quantum_state=quantum,
            name=name or "uncertain",
        )

    @classmethod
    def contextual(
        cls,
        paradigm_source: str = "wen",
        resolver: Optional[str] = None,
        name: str = "",
    ) -> FluxType:
        """Create a context-dependent FluxType.

        Args:
            paradigm_source: Originating paradigm (default: wen for 文境).
            resolver: Optional description of how context resolves this type.
            name: Optional name.

        Returns:
            A FluxType with CONTEXTUAL base type.
        """
        constraints = []
        if resolver:
            constraints.append(
                FluxConstraint(
                    kind=ConstraintKind.CONTEXT_DOMAIN,
                    language=paradigm_source,
                    value=resolver,
                    confidence=0.5,
                )
            )
        return cls(
            base_type=FluxBaseType.CONTEXTUAL,
            confidence=0.5,
            paradigm_source=paradigm_source,
            constraints=constraints,
            name=name or "contextual",
        )

    # ── Query methods ───────────────────────────────────────────

    def has_constraint(self, kind: ConstraintKind, language: str = "") -> bool:
        """Check if this type has a constraint of the given kind."""
        for c in self.constraints:
            if c.kind == kind:
                if not language or c.language == language:
                    return True
        return False

    def get_constraints(self, kind: Optional[ConstraintKind] = None,
                        language: str = "") -> List[FluxConstraint]:
        """Get constraints, optionally filtered by kind and language."""
        result = []
        for c in self.constraints:
            if kind and c.kind != kind:
                continue
            if language and c.language != language:
                continue
            result.append(c)
        return result

    def effective_base_type(self) -> FluxBaseType:
        """Get the effective base type, collapsing quantum state if needed."""
        if self.quantum_state and not self.quantum_state.collapsed:
            return self.quantum_state.observe()
        if self.quantum_state and self.quantum_state.collapsed_type:
            return self.quantum_state.collapsed_type
        return self.base_type

    def is_compatible_base(self, other_base: FluxBaseType) -> float:
        """Quick compatibility check against a base type.

        Returns:
            1.0 if same base type.
            0.0 if antipodal on spectrum.
            Linear interpolation otherwise.
        """
        max_dist = float(FluxBaseType.CONTEXTUAL)
        dist = self.effective_base_type().spectrum_distance(other_base)
        return max(0.0, 1.0 - dist / max_dist)

    # ── Mutation ────────────────────────────────────────────────

    def add_constraint(self, constraint: FluxConstraint) -> None:
        """Add a language-specific constraint."""
        self.constraints.append(constraint)

    def set_confidence(self, confidence: float) -> None:
        """Update the confidence score."""
        self.confidence = max(0.0, min(1.0, float(confidence)))

    def collapse(self, evidence: Optional[Dict[FluxBaseType, float]] = None) -> FluxBaseType:
        """Collapse quantum state with optional evidence.

        Returns:
            The collapsed base type.
        """
        if self.quantum_state:
            result = self.quantum_state.observe(evidence)
            self.base_type = result
            self.confidence = max(self.confidence, 0.8)
            return result
        return self.base_type

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "base_type": self.base_type.name,
            "confidence": round(self.confidence, 6),
            "paradigm_source": self.paradigm_source,
            "name": self.name,
            "constraints": [c.to_dict() for c in self.constraints],
        }
        if self.quantum_state:
            d["quantum_state"] = self.quantum_state.to_dict()
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FluxType:
        quantum = None
        if "quantum_state" in data:
            qs = data["quantum_state"]
            quantum = QuantumTypeState(
                possibilities=[
                    (FluxBaseType[p["type"]], p["amplitude"])
                    for p in qs.get("possibilities", [])
                ],
                collapsed=qs.get("collapsed", False),
                collapsed_type=FluxBaseType[qs["collapsed_type"]] if qs.get("collapsed_type") else None,
            )
        return cls(
            base_type=FluxBaseType[data["base_type"]],
            constraints=[FluxConstraint.from_dict(c) for c in data.get("constraints", [])],
            confidence=data.get("confidence", 1.0),
            paradigm_source=data.get("paradigm_source", "flux"),
            quantum_state=quantum,
            name=data.get("name", ""),
            meta=data.get("meta", {}),
            id=data.get("id", ""),
        )

    def __repr__(self) -> str:
        q = f" (q={self.quantum_state.entropy():.2f})" if self.quantum_state else ""
        return (f"FluxType({self.base_type.name}, conf={self.confidence:.2f}, "
                f"src={self.paradigm_source}, "
                f"constraints={len(self.constraints)}{q})")


# ══════════════════════════════════════════════════════════════════
# 5. FluxTypeSignature — Function/Operation Type Signature
# ══════════════════════════════════════════════════════════════════

@dataclass
class FluxTypeSignature:
    """Type signature for a function or operation.

    Combines input types, output type, and effect constraints into
    a single signature — inspired by algebraic effect types and
    Rust's fn trait bounds.

    Attributes:
        inputs: List of input FluxTypes.
        output: Output FluxType.
        effects: Effect constraints (what side effects this may produce).
        requires: Capability requirements (from Korean honorific system).
        scope: Scope level (from Sanskrit vibhakti).
    """
    inputs: List[FluxType] = field(default_factory=list)
    output: Optional[FluxType] = None
    effects: List[FluxConstraint] = field(default_factory=list)
    requires: List[FluxConstraint] = field(default_factory=list)
    scope: Optional[FluxBaseType] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [t.to_dict() for t in self.inputs],
            "output": self.output.to_dict() if self.output else None,
            "effects": [e.to_dict() for e in self.effects],
            "requires": [r.to_dict() for r in self.requires],
            "scope": self.scope.name if self.scope else None,
        }


# ══════════════════════════════════════════════════════════════════
# 6. Internal helpers
# ══════════════════════════════════════════════════════════════════

def _infer_constraint_kind(lang: str, native_tag: str) -> ConstraintKind:
    """Infer the ConstraintKind from a language tag and native type name."""
    # Classifier-related tags
    if lang == "zho":
        if "flat" in native_tag or "long" in native_tag or "round" in native_tag:
            return ConstraintKind.CLASSIFIER_SHAPE
        if "person" in native_tag or "animal" in native_tag:
            return ConstraintKind.CLASSIFIER_ANIMACY
        if "collective" in native_tag or "pair" in native_tag or "volume" in native_tag:
            return ConstraintKind.CLASSIFIER_COUNTABILITY
        if "indeterminate" in native_tag:
            return ConstraintKind.CLASSIFIER_SHAPE
        return ConstraintKind.CLASSIFIER_SHAPE

    # Gender/case tags
    if lang == "deu":
        if native_tag in ("nominativ", "akkusativ", "dativ", "genitiv"):
            return ConstraintKind.CASE_MARKING
        if native_tag in ("maskulinum", "femininum", "neutrum"):
            return ConstraintKind.GENDER_AGREEMENT
        return ConstraintKind.GENDER_AGREEMENT

    # Honorific tags
    if lang == "kor":
        if "che" in native_tag:
            return ConstraintKind.HONORIFIC_LEVEL
        if "honorific" in native_tag:
            return ConstraintKind.HONORIFIC_LEVEL
        if native_tag in ("declarative", "interrogative", "imperative", "propositive"):
            return ConstraintKind.SPEECH_ACT
        return ConstraintKind.HONORIFIC_LEVEL

    # Sanskrit tags
    if lang == "san":
        if native_tag in ("prathama", "dvitiya", "tritiya", "chaturthi",
                          "panchami", "shashthi", "saptami", "sambodhana"):
            return ConstraintKind.SCOPE_LEVEL
        if "linga" in native_tag:
            return ConstraintKind.GENDER_AGREEMENT
        if "vachana" in native_tag:
            return ConstraintKind.NUMBER_AGREEMENT
        return ConstraintKind.SCOPE_LEVEL

    # Classical Chinese tags
    if lang == "wen":
        if native_tag in ("attack", "defend", "advance", "retreat",
                          "sequence", "loop"):
            return ConstraintKind.EXECUTION_MODE
        if native_tag in ("ren", "yi", "li", "zhi", "xin"):
            return ConstraintKind.TRUST_REQUIREMENT
        if native_tag in ("topic", "zero_anaphora", "context_resolved"):
            return ConstraintKind.TOPIC_REGISTER
        return ConstraintKind.CONTEXT_DOMAIN

    # Latin tags
    if lang == "lat":
        if native_tag in ("praesens", "imperfectum", "perfectum",
                          "plusquamperfectum", "futurum", "futurum_exactum"):
            return ConstraintKind.TEMPORAL_ASPECT
        if "tius" in native_tag:
            return ConstraintKind.CASE_MARKING
        if native_tag in ("maskulinum", "femininum", "neutrum"):
            return ConstraintKind.GENDER_AGREEMENT
        return ConstraintKind.TEMPORAL_ASPECT

    return ConstraintKind.CONFIDENCE_BOUND


# ══════════════════════════════════════════════════════════════════
# 7. Type Registry — Central catalog of all known types
# ══════════════════════════════════════════════════════════════════

class FluxTypeRegistry:
    """Central registry for FluxType definitions.

    Provides:
      - Named type registration and lookup
      - Paradigm-specific type catalogs
      - Type alias management
      - JSON serialization/deserialization of the full registry
    """

    def __init__(self) -> None:
        self._types: Dict[str, FluxType] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, flux_type: FluxType, alias: str = "") -> None:
        """Register a FluxType, optionally with an alias."""
        name = alias or flux_type.name or flux_type.id
        self._types[name] = flux_type
        if alias:
            self._aliases[alias] = name

    def get(self, name: str) -> Optional[FluxType]:
        """Look up a type by name or alias."""
        resolved = self._aliases.get(name, name)
        return self._types.get(resolved)

    def get_by_paradigm(self, lang: str) -> List[FluxType]:
        """Get all types originating from a specific paradigm."""
        return [t for t in self._types.values() if t.paradigm_source == lang]

    def all_types(self) -> List[FluxType]:
        """Return all registered types."""
        return list(self._types.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "types": {name: t.to_dict() for name, t in self._types.items()},
            "aliases": dict(self._aliases),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FluxTypeRegistry:
        reg = cls()
        for name, td in data.get("types", {}).items():
            ft = FluxType.from_dict(td)
            ft.name = name
            reg.register(ft, alias=name)
        for alias, target in data.get("aliases", {}).items():
            reg._aliases[alias] = target
        return reg


# ══════════════════════════════════════════════════════════════════
# 8. Pre-built Type Catalog
# ══════════════════════════════════════════════════════════════════

def build_default_registry() -> FluxTypeRegistry:
    """Build a registry pre-populated with canonical types from all 6 paradigms."""
    reg = FluxTypeRegistry()

    # ── Chinese (ZHO) types ──
    for tag, base in _ZHO_CLASSIFIERS.items():
        reg.register(FluxType.from_paradigm("zho", tag, name=f"zho:{tag}"))

    # ── German (DEU) types ──
    for tag, base in _DEU_GESCHLECHT.items():
        reg.register(FluxType.from_paradigm("deu", tag, name=f"deu:{tag}"))

    # ── Korean (KOR) types ──
    for tag, base in _KOR_HONORIFICS.items():
        reg.register(FluxType.from_paradigm("kor", tag, name=f"kor:{tag}"))

    # ── Sanskrit (SAN) types ──
    for tag, base in _SAN_VIBHAKTI.items():
        reg.register(FluxType.from_paradigm("san", tag, name=f"san:{tag}"))

    # ── Classical Chinese (WEN) types ──
    for tag, base in _WEN_WENJING.items():
        reg.register(FluxType.from_paradigm("wen", tag, name=f"wen:{tag}"))

    # ── Latin (LAT) types ──
    for tag, base in _LAT_TEMPUS.items():
        reg.register(FluxType.from_paradigm("lat", tag, name=f"lat:{tag}"))

    return reg
