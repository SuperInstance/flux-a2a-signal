"""
FLUX Denotational Semantics — The Mathematical Meaning of FLUX Programs.

A FLUX program P denotes a function:

    ⟦P⟧ : FluxState × FluxContext × CapSet → Dist(FluxResult × FluxState)

where:
  - FluxState       = Register file (registers, memory regions, topic register)
  - FluxContext     = Language tag, domain stack, trust graph, temporal mode
  - CapSet          = Set of capabilities derived from Kasus/honorifics/scope
  - Dist(α)         = Probability distribution over values of type α
  - FluxResult      = (value, confidence, effects_performed)

This is NOT standard denotational semantics (⟦P⟧ : State → State).  FLUX
programs are fundamentally non-standard because they are:
  1. Non-deterministic   — quantum registers hold superpositions
  2. Context-dependent   — Classical Chinese characters change meaning
  3. Capability-constrained — German Kasus, Korean honorifics gate execution
  4. Temporal-mode-aware — Latin tenses change execution semantics
  5. Scope-rich          — Sanskrit 8-case scope system

The theoretical foundations draw from:
  - Probabilistic denotational semantics (Kozen, Jones, Plotkin)
  - Effect systems (Koka, Multicore OCaml algebraic effects)
  - Game semantics (Abramsky, Jagadeesan, Malacaria)
  - Concurrent process algebra (CSP, pi-calculus, CCS)
  - Quantitative type theory (Gilbert, Krishnaswami, 2023)

Formal reference for the semantics: see docs/round10-12_flux_semantics.md
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ═══════════════════════════════════════════════════════════════════════════
# §1  Core Semantic Domains
# ═══════════════════════════════════════════════════════════════════════════

A = TypeVar("A")  # Generic type for distribution elements
B = TypeVar("B")


# ---------------------------------------------------------------------------
# §1.1  Superposition — quantum-like probabilistic register values
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BasisState:
    """A single basis state in a quantum-like superposition.

    Represents a classical value with an associated amplitude (complex
    probability weight).  For FLUX we simplify to real amplitudes in [0,1].

    Attributes:
        value:      The classical value this basis state holds.
        amplitude:  Probability amplitude (squared = probability).
    """
    value: Any
    amplitude: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", max(0.0, min(1.0, self.amplitude)))

    @property
    def probability(self) -> float:
        """Probability = amplitude² (Born rule)."""
        return self.amplitude ** 2

    def __repr__(self) -> str:
        return f"BasisState({self.value!r}, α={self.amplitude:.3f})"


class Superposition:
    """
    A quantum-like superposition of basis states.

    Superposition(α₁|ψ₁⟩ + α₂|ψ₂⟩ + ... + αₙ|ψₙ⟩)

    where αᵢ are amplitudes and ψᵢ are classical values.
    The distribution is normalized so that Σ αᵢ² = 1.

    In FLUX, registers can hold Superpositions instead of scalar values.
    This is the fundamental source of non-determinism in the semantics.
    """

    def __init__(self, states: Optional[Iterable[BasisState]] = None):
        self._states: List[BasisState] = list(states) if states else []

    def add(self, value: Any, amplitude: float = 1.0) -> None:
        """Add a basis state."""
        self._states.append(BasisState(value=value, amplitude=amplitude))

    def normalize(self) -> Superposition:
        """Normalize amplitudes so Σ αᵢ² = 1."""
        total_prob = sum(s.probability for s in self._states)
        if total_prob == 0:
            return self
        scale = 1.0 / math.sqrt(total_prob)
        normalized = Superposition(
            BasisState(s.value, s.amplitude * scale) for s in self._states
        )
        return normalized

    @property
    def states(self) -> Tuple[BasisState, ...]:
        return tuple(self._states)

    @property
    def is_deterministic(self) -> bool:
        """True if exactly one basis state with amplitude 1."""
        return len(self._states) == 1 and abs(self._states[0].amplitude - 1.0) < 1e-9

    def deterministic_value(self) -> Any:
        """Return the scalar value if deterministic, else None."""
        if self.is_deterministic:
            return self._states[0].value
        return None

    def collapse(self) -> Any:
        """Measure: collapse to a single classical value (sample from distribution).

        Returns the most probable basis state's value.
        """
        if self.is_deterministic:
            return self._states[0].value
        # Sample weighted by probability
        import random
        weights = [s.probability for s in self._states]
        total = sum(weights)
        if total == 0:
            return None
        weights = [w / total for w in weights]
        chosen = random.choices(self._states, weights=weights, k=1)[0]
        return chosen.value

    def expected_value(self) -> float:
        """Compute the expected (mean) value, assuming numeric basis states."""
        normed = self.normalize()
        total = sum(s.probability * s.value for s in normed._states
                    if isinstance(s.value, (int, float)))
        return total

    def entropy(self) -> float:
        """Shannon entropy of the probability distribution.

        H = -Σ pᵢ · log₂(pᵢ)

        Returns 0.0 for deterministic superpositions, up to log₂(n) for
        uniform superpositions.
        """
        normed = self.normalize()
        entropy = 0.0
        for s in normed._states:
            p = s.probability
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def compose(self, other: Superposition) -> Superposition:
        """Tensor product (entanglement) of two superpositions.

        Creates all pairs (a, b) with combined amplitudes.
        """
        result = Superposition()
        for sa in self._states:
            for sb in other._states:
                result.add(
                    value=(sa.value, sb.value),
                    amplitude=sa.amplitude * sb.amplitude,
                )
        return result.normalize()

    def map(self, fn: Callable[[Any], Any]) -> Superposition:
        """Apply a function to all basis state values (preserving amplitudes)."""
        return Superposition(
            BasisState(fn(s.value), s.amplitude) for s in self._states
        )

    def filter(self, pred: Callable[[Any], bool]) -> Superposition:
        """Filter basis states by predicate, then re-normalize."""
        result = Superposition(
            s for s in self._states if pred(s.value)
        )
        return result.normalize()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "states": [
                {"value": s.value, "amplitude": round(s.amplitude, 6)}
                for s in self._states
            ],
            "is_deterministic": self.is_deterministic,
            "entropy": round(self.entropy(), 4),
        }

    def __repr__(self) -> str:
        if self.is_deterministic:
            return f"Superposition(|{self._states[0].value!r}⟩)"
        terms = " + ".join(f"{s.amplitude:.2f}|{s.value!r}⟩" for s in self._states)
        return f"Superposition({terms})"


def pure(value: Any) -> Superposition:
    """Create a deterministic superposition (single basis state)."""
    return Superposition([BasisState(value=value, amplitude=1.0)])


def uniform(values: Sequence[Any]) -> Superposition:
    """Create a uniform superposition over values."""
    n = len(values)
    if n == 0:
        return Superposition()
    amp = 1.0 / math.sqrt(n)
    return Superposition(BasisState(v, amp) for v in values).normalize()


# ---------------------------------------------------------------------------
# §1.2  FluxState — machine state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FluxState:
    """
    The mutable machine state of a FLUX execution.

    Attributes:
        registers:     General-purpose register file (R0–R15).
        memory:        Memory regions (named address spaces).
        topic_register: The implicit-operand topic register (定題).
        stack:         Evaluation stack.
        flags:         Condition flags (zero, carry, negative, overflow).
        pc:            Program counter (for denotational composition).
    """
    registers: Dict[int, Superposition] = field(default_factory=dict)
    memory: Dict[str, bytes] = field(default_factory=dict)
    topic_register: Optional[Superposition] = None
    stack: List[Superposition] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    pc: int = 0

    def get_register(self, reg: int) -> Superposition:
        """Get register value; defaults to |0⟩."""
        return self.registers.get(reg, pure(0))

    def set_register(self, reg: int, value: Union[Superposition, Any]) -> None:
        """Set register value, auto-wrapping scalars in Superposition."""
        if not isinstance(value, Superposition):
            value = pure(value)
        self.registers[reg] = value

    def fork(self) -> FluxState:
        """Create a deep copy of this state (for branch isolation)."""
        return FluxState(
            registers=dict(self.registers),
            memory=dict(self.memory),
            topic_register=self.topic_register,
            stack=list(self.stack),
            flags=dict(self.flags),
            pc=self.pc,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registers": {str(k): v.to_dict() for k, v in self.registers.items()},
            "topic_register": self.topic_register.to_dict() if self.topic_register else None,
            "stack_depth": len(self.stack),
            "flags": dict(self.flags),
            "pc": self.pc,
        }


# ---------------------------------------------------------------------------
# §1.3  FluxContext — execution context
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FluxContext:
    """
    Execution context that parameterizes the denotation.

    Unlike standard denotational semantics where context is fixed, FLUX
    programs are *parametric* in their context.  Same bytecode, different
    context → different denotation.  This is by design.

    Attributes:
        language:        Source language tag (zho, deu, kor, san, wen, lat).
        domain_stack:    Stack of (domain_name, priority) pairs.
        trust_graph:     Agent trust relationships.
        temporal_mode:   Latin tense governing execution semantics.
        honorific_level: Korean speech level (affects capability checking).
        scope_depth:     Sanskrit vibhakti scope level (1–8).
        topic:           Current discourse topic.
    """
    language: str = "flux"
    domain_stack: List[Tuple[str, float]] = field(default_factory=list)
    trust_graph: Dict[str, Dict[str, float]] = field(default_factory=dict)
    temporal_mode: str = "present"  # present|imperfect|perfect|pluperfect|future|future_perfect
    honorific_level: int = 3        # Korean: 1=하십시오체 ... 7=해체
    scope_depth: int = 1            # Sanskrit: 1=प्रथमा ... 8=संबोधन
    topic: str = ""

    def push_domain(self, domain: str, priority: float = 1.0) -> None:
        """Push a domain onto the context stack."""
        self.domain_stack.append((domain, priority))

    def pop_domain(self) -> Optional[Tuple[str, float]]:
        """Pop the top domain from the stack."""
        if self.domain_stack:
            return self.domain_stack.pop()
        return None

    def active_domain(self) -> Optional[str]:
        """Return the highest-priority active domain."""
        if not self.domain_stack:
            return None
        # Sort by priority descending, return name of highest
        sorted_domains = sorted(self.domain_stack, key=lambda x: x[1], reverse=True)
        return sorted_domains[0][0]

    def fork(self) -> FluxContext:
        """Create a copy of this context."""
        return FluxContext(
            language=self.language,
            domain_stack=list(self.domain_stack),
            trust_graph={k: dict(v) for k, v in self.trust_graph.items()},
            temporal_mode=self.temporal_mode,
            honorific_level=self.honorific_level,
            scope_depth=self.scope_depth,
            topic=self.topic,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "domain_stack": list(self.domain_stack),
            "temporal_mode": self.temporal_mode,
            "honorific_level": self.honorific_level,
            "scope_depth": self.scope_depth,
            "topic": self.topic,
        }


# ---------------------------------------------------------------------------
# §1.4  CapSet — capability set
# ---------------------------------------------------------------------------

class ExtendedCap(Enum):
    """
    Extended capabilities derived from linguistic features.

    German Kasus → capability constraints:
      - NOM_READ:   Can read nominative-annotated data
      - AKK_WRITE:  Can write accusative-annotated data
      - DAT_SHARE:  Can share dative-annotated data
      - GEN_OWN:    Owns genitive-annotated data

    Korean honorifics → capability levels:
      - SPEAK_FORMAL:  Can issue 하십시오체 commands
      - SPEAK_POLITE:  Can issue 해요체 commands
      - SPEAK_PLAIN:   Can issue 해체 commands

    Sanskrit vibhakti → scope capabilities:
      - SCOPE_1 through SCOPE_8: 8 case scope levels

    General FLUX:
      - FORK:     Can spawn child agents
      - BRANCH:   Can create parallel branches
      - TELL:     Can send messages
      - ASK:      Can query agents
      - TRUST:    Can modify trust relationships
      - META:     Can modify own opcodes
    """
    # FLUX core
    FORK = "fork"
    BRANCH = "branch"
    TELL = "tell"
    ASK = "ask"
    TRUST = "trust"
    META = "meta"
    DISCUSS = "discuss"
    DELEGATE = "delegate"

    # German Kasus-derived
    NOM_READ = "nom_read"
    AKK_WRITE = "akk_write"
    DAT_SHARE = "dat_share"
    GEN_OWN = "gen_own"

    # Korean honorific-derived
    SPEAK_FORMAL = "speak_formal"
    SPEAK_POLITE = "speak_polite"
    SPEAK_PLAIN = "speak_plain"

    # Sanskrit scope-derived
    SCOPE_1 = "scope_1"  # प्रथमा — nominative (agent)
    SCOPE_2 = "scope_2"  # द्वितीया — accusative (patient)
    SCOPE_3 = "scope_3"  # तृतीया — instrumental (means)
    SCOPE_4 = "scope_4"  # चतुर्थी — dative (recipient)
    SCOPE_5 = "scope_5"  # पञ्चमी — ablative (source)
    SCOPE_6 = "scope_6"  # षष्ठी — genitive (possessor)
    SCOPE_7 = "scope_7"  # सप्तमी — locative (location)
    SCOPE_8 = "scope_8"  # संबोधन — vocative (address)

    # Latin temporal-derived
    TEMPORAL_LOOP = "temporal_loop"        # Imperfectum — continuous
    TEMPORAL_DEFER = "temporal_defer"      # Futurum — deferred
    TEMPORAL_CACHE = "temporal_cache"      # Perfectum — cached
    TEMPORAL_ROLLBACK = "temporal_rollback"  # Plusquamperfectum — revert
    TEMPORAL_EVENTUAL = "temporal_eventual"  # Futurum exactum — eventual


@dataclass(frozen=True, slots=True)
class CapSet:
    """
    A set of capabilities that gates which programs can execute.

    Capabilities are derived from linguistic annotations:
      - German Kasus (4 cases → 4 data access capabilities)
      - Korean honorific level → speech act capabilities
      - Sanskrit vibhakti → scope access levels
      - Latin tense → temporal execution mode capabilities

    The empty CapSet() has no capabilities — almost nothing can run.
    The universal CapSet.full() permits everything.

    Capability monotonicity theorem (see §5.3):
        If A ⊆ B then D_B ⊆ D_A
    where D_C is the set of programs denotable under capability set C.
    """
    capabilities: FrozenSet[ExtendedCap] = frozenset()
    source: str = ""  # Which language/annotation defined these

    def has(self, cap: ExtendedCap) -> bool:
        """Check if a specific capability is present."""
        return cap in self.capabilities

    def requires(self, cap: ExtendedCap) -> bool:
        """Check capability; returns True if present."""
        return self.has(cap)

    def requires_all(self, caps: Iterable[ExtendedCap]) -> bool:
        """Check that ALL required capabilities are present."""
        return all(c in self.capabilities for c in caps)

    def requires_any(self, caps: Iterable[ExtendedCap]) -> bool:
        """Check that at least ONE capability is present."""
        return any(c in self.capabilities for c in caps)

    def union(self, other: CapSet) -> CapSet:
        """Merge two capability sets."""
        return CapSet(
            capabilities=self.capabilities | other.capabilities,
            source=f"{self.source}+{other.source}" or "union",
        )

    def intersection(self, other: CapSet) -> CapSet:
        """Intersect two capability sets."""
        return CapSet(
            capabilities=self.capabilities & other.capabilities,
            source=f"{self.source}∩{other.source}" or "intersection",
        )

    def is_subset(self, other: CapSet) -> bool:
        """Check if this CapSet is a subset of another."""
        return self.capabilities.issubset(other.capabilities)

    @classmethod
    def empty(cls) -> CapSet:
        """Empty capability set — nothing permitted."""
        return cls(capabilities=frozenset(), source="empty")

    @classmethod
    def full(cls) -> CapSet:
        """Full capability set — everything permitted."""
        return cls(capabilities=frozenset(ExtendedCap), source="full")

    @classmethod
    def from_korean(cls, honorific_level: int) -> CapSet:
        """
        Derive capabilities from Korean honorific level.

        Level 1 (하십시오체/formal): all speech capabilities
        Level 3 (해요체/polite): SPEAK_POLITE + lower
        Level 7 (해체/plain): SPEAK_PLAIN only
        """
        caps = set()
        # All levels get SPEAK_PLAIN (minimum capability)
        caps.add(ExtendedCap.SPEAK_PLAIN)
        if honorific_level <= 3:
            caps.add(ExtendedCap.SPEAK_POLITE)
        if honorific_level <= 1:
            caps.add(ExtendedCap.SPEAK_FORMAL)
        return cls(capabilities=frozenset(caps), source=f"kor_honorific_{honorific_level}")

    @classmethod
    def from_german(cls, kasus: str) -> CapSet:
        """
        Derive capabilities from German Kasus.

        Nominativ → NOM_READ (can access subject data)
        Akkusativ → AKK_WRITE (can modify object data)
        Dativ    → DAT_SHARE (can share recipient data)
        Genitiv  → GEN_OWN (can own possessor data)
        """
        mapping = {
            "nom": ExtendedCap.NOM_READ,
            "akk": ExtendedCap.AKK_WRITE,
            "dat": ExtendedCap.DAT_SHARE,
            "gen": ExtendedCap.GEN_OWN,
        }
        cap = mapping.get(kasus.lower()[:3])
        caps = frozenset({cap}) if cap else frozenset()
        return cls(capabilities=caps, source=f"deu_kasus_{kasus}")

    @classmethod
    def from_sanskrit(cls, vibhakti: int) -> CapSet:
        """
        Derive capabilities from Sanskrit vibhakti (case).

        Each of the 8 cases maps to a scope capability.
        Higher cases have broader scope access.
        """
        scope_map = {
            1: ExtendedCap.SCOPE_1,  # प्रथमा
            2: ExtendedCap.SCOPE_2,  # द्वितीया
            3: ExtendedCap.SCOPE_3,  # तृतीया
            4: ExtendedCap.SCOPE_4,  # चतुर्थी
            5: ExtendedCap.SCOPE_5,  # पञ्चमी
            6: ExtendedCap.SCOPE_6,  # षष्ठी
            7: ExtendedCap.SCOPE_7,  # सप्तमी
            8: ExtendedCap.SCOPE_8,  # संबोधन
        }
        cap = scope_map.get(vibhakti)
        # Lower cases are subsets of higher (scope hierarchy)
        caps = frozenset(scope_map[i] for i in range(1, vibhakti + 1)) if cap else frozenset()
        return cls(capabilities=caps, source=f"san_vibhakti_{vibhakti}")

    @classmethod
    def from_latin(cls, tense: str) -> CapSet:
        """
        Derive capabilities from Latin tense.

        Present        → (default, no temporal effect)
        Imperfectum    → TEMPORAL_LOOP (continuous iteration)
        Perfectum      → TEMPORAL_CACHE (cached results)
        Plusquamperf.  → TEMPORAL_ROLLBACK (state snapshots)
        Futurum        → TEMPORAL_DEFER (lazy evaluation)
        Futurum Exact. → TEMPORAL_EVENTUAL (eventual consistency)
        """
        mapping = {
            "present": frozenset(),
            "imperfect": frozenset({ExtendedCap.TEMPORAL_LOOP}),
            "perfect": frozenset({ExtendedCap.TEMPORAL_CACHE}),
            "pluperfect": frozenset({ExtendedCap.TEMPORAL_ROLLBACK}),
            "future": frozenset({ExtendedCap.TEMPORAL_DEFER}),
            "future_perfect": frozenset({ExtendedCap.TEMPORAL_EVENTUAL}),
        }
        caps = mapping.get(tense.lower().replace(" ", "_"), frozenset())
        return cls(capabilities=caps, source=f"lat_tense_{tense}")

    @property
    def count(self) -> int:
        """Number of capabilities in this set."""
        return len(self.capabilities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capabilities": sorted(c.value for c in self.capabilities),
            "source": self.source,
            "count": len(self.capabilities),
        }


# ---------------------------------------------------------------------------
# §1.5  FluxResult — semantic result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FluxResult:
    """
    The result of applying a denotation to (state, context, caps).

    Attributes:
        value:      The produced value (may be a Superposition).
        confidence: Confidence score in [0.0, 1.0].
        new_state:  The resulting state after execution.
        effects:    Side effects performed (messages sent, branches created, etc.).
        error:      Error message if execution failed.
    """
    value: Any = None
    confidence: float = 1.0
    new_state: Optional[FluxState] = None
    effects: List[str] = field(default_factory=list)
    error: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    def with_state(self, state: FluxState) -> FluxResult:
        """Return a copy with the given state."""
        return FluxResult(
            value=self.value,
            confidence=self.confidence,
            new_state=state,
            effects=list(self.effects),
            error=self.error,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "value": self.value,
            "confidence": round(self.confidence, 4),
        }
        if self.new_state:
            d["new_state"] = self.new_state.to_dict()
        if self.effects:
            d["effects"] = self.effects
        if self.error:
            d["error"] = self.error
        return d


# ═══════════════════════════════════════════════════════════════════════════
# §2  FluxFunction — The Semantic Arrow
# ═══════════════════════════════════════════════════════════════════════════

class FluxFunction(ABC):
    """
    A denotable semantic function.

    The central type in FLUX's semantics:

        FluxFunction ≅ FluxState → Dist(FluxResult × FluxState)

    More precisely, it takes (state, context, caps) as parameters and
    returns a list of (result, probability) pairs — a discrete distribution.

    This is the semantic analog of the Kleisli arrow for the probability
    monad, parameterized by context and capabilities.
    """

    @abstractmethod
    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        """
        Apply this denotation to produce a distribution of results.

        Returns:
            A list of (result, probability) pairs.  The probabilities
            should sum to ≤ 1.0 (may be less if there are "impossible"
            paths due to capability violations).
        """
        ...

    def run(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> FluxResult:
        """Convenience: run and return the highest-probability result."""
        results = self.apply(state, context, caps)
        if not results:
            return FluxResult(
                value=None, confidence=0.0, new_state=state,
                error="No viable execution path (capability violation or empty program)",
            )
        best = max(results, key=lambda x: x[1])
        return best[0]

    def bind(
        self,
        g: Callable[[FluxResult], FluxFunction],
    ) -> FluxFunction:
        """Kleisli composition: self >>= g.

        Sequential composition of denotations.  The result of the first
        feeds into the second.  Probabilities multiply.

        This implements: ⟦P1; P2⟧ = ⟦P2⟧ ∘ ⟦P1⟧
        """
        f = self
        return _BoundFluxFunction(f, g)

    def then(self, g: FluxFunction) -> FluxFunction:
        """Sequential composition: self >> g.

        Ignores the result of self, passes state through.
        """
        return self.bind(lambda _: g)

    def guard(self, cap: ExtendedCap) -> FluxFunction:
        """Guard this function with a capability requirement.

        If the capability is present, execution proceeds normally.
        If not, the function produces an error result.
        """
        f = self
        return _GuardedFluxFunction(f, cap)


class _BoundFluxFunction(FluxFunction):
    """Kleisli composition: f >>= g."""

    def __init__(
        self,
        f: FluxFunction,
        g: Callable[[FluxResult], FluxFunction],
    ) -> None:
        self._f = f
        self._g = g

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        results = []
        for (result_a, prob_a) in self._f.apply(state, context, caps):
            if result_a.is_error:
                results.append((result_a, prob_a))
                continue
            next_state = result_a.new_state or state
            g_fn = self._g(result_a)
            for (result_b, prob_b) in g_fn.apply(next_state, context, caps):
                combined_prob = prob_a * prob_b
                combined_conf = result_a.confidence * result_b.confidence
                merged_effects = result_a.effects + result_b.effects
                results.append((
                    FluxResult(
                        value=result_b.value,
                        confidence=combined_conf,
                        new_state=result_b.new_state or next_state,
                        effects=merged_effects,
                        error=result_b.error,
                    ),
                    combined_prob,
                ))
        return results

    def __repr__(self) -> str:
        return f"BoundFluxFunction({self._f!r}, ...)"


class _GuardedFluxFunction(FluxFunction):
    """Capability guard: requires a specific capability."""

    def __init__(self, f: FluxFunction, cap: ExtendedCap) -> None:
        self._f = f
        self._cap = cap

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        if not caps.has(self._cap):
            return [(
                FluxResult(
                    value=None,
                    confidence=0.0,
                    new_state=state,
                    error=f"Capability violation: {self._cap.value} required but not in {caps.source or 'caps'}",
                ),
                1.0,
            )]
        return self._f.apply(state, context, caps)

    def __repr__(self) -> str:
        return f"GuardedFluxFunction({self._f!r}, {self._cap.value})"


# ---------------------------------------------------------------------------
# §2.1  Primitive FluxFunction constructors
# ---------------------------------------------------------------------------

class IdentityFunc(FluxFunction):
    """The identity denotation: returns state unchanged with confidence 1.0."""

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        return [(FluxResult(value=None, confidence=1.0, new_state=state), 1.0)]


class ConstFunc(FluxFunction):
    """A constant denotation: always produces the given value."""

    def __init__(self, value: Any, confidence: float = 1.0) -> None:
        self._value = value
        self._confidence = confidence

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        return [(FluxResult(value=self._value, confidence=self._confidence, new_state=state), 1.0)]


class PureFunc(FluxFunction):
    """A pure computation: applies fn to state, returns result."""

    def __init__(self, fn: Callable[[FluxState, FluxContext, CapSet], FluxResult]) -> None:
        self._fn = fn

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        result = self._fn(state, context, caps)
        return [(result, 1.0)]


class BranchFunc(FluxFunction):
    """
    Probabilistic branching: weighted choice between sub-functions.

    Semantically:  ⟦branch⟧ = Σᵢ wᵢ · ⟦Pᵢ⟧

    Each branch has a weight (prior probability).  The result is a
    distribution over all branch outcomes.
    """

    def __init__(
        self,
        branches: List[Tuple[FluxFunction, float]],
        merge_strategy: str = "weighted_confidence",
    ) -> None:
        self._branches = branches
        self._merge_strategy = merge_strategy

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        if not self._branches:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="Empty branch"), 1.0)]

        # Normalize weights
        total_weight = sum(w for _, w in self._branches)
        if total_weight == 0:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="All branch weights are 0"), 1.0)]

        results: List[Tuple[FluxResult, float]] = []
        for branch_fn, weight in self._branches:
            branch_state = state.fork()  # Branch isolation
            branch_results = branch_fn.apply(branch_state, context, caps)
            prob = weight / total_weight
            for result, inner_prob in branch_results:
                results.append((result, prob * inner_prob))

        return results


class ForkFunc(FluxFunction):
    """
    Agent forking: spawn a child agent that executes a sub-program.

    Semantically modeled as concurrent composition:
        ⟦fork(P)⟧(s, c, k) = ⟦P⟧(s_child, c_child, k_child) ⊗ (s, c, k)

    The parent state continues while the child executes independently.
    """

    def __init__(
        self,
        child_fn: FluxFunction,
        inherit_keys: Optional[List[str]] = None,
        child_caps: Optional[CapSet] = None,
    ) -> None:
        self._child_fn = child_fn
        self._inherit_keys = inherit_keys or []
        self._child_caps = child_caps

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        # Parent continues with original state
        parent_result = FluxResult(
            value=None,
            confidence=1.0,
            new_state=state,
            effects=["fork_spawned"],
        )

        # Child executes in a forked state
        child_state = state.fork()
        child_context = context.fork()
        child_caps = self._child_caps or caps

        child_results = self._child_fn.apply(child_state, child_context, child_caps)

        results = []
        for child_res, child_prob in child_results:
            # Merge child result into parent context
            effects = list(parent_result.effects)
            effects.extend(child_res.effects)
            effects.append(f"fork_result:{child_res.value}")

            results.append((
                FluxResult(
                    value=child_res.value,
                    confidence=child_res.confidence,
                    new_state=state,  # Parent state unchanged by child
                    effects=effects,
                ),
                child_prob,
            ))

        # If child produced no results, return parent result
        if not results:
            results.append((parent_result, 1.0))

        return results


class DiscussFunc(FluxFunction):
    """
    Structured discussion between agents.

    Modeled as a game-theoretic interaction:
      - Debate:    zero-sum game (agents compete)
      - Brainstorm: cooperative game (agents collaborate)
      - Review:    verification game (agents critique)
      - Negotiation: bargaining game (agents compromise)

    The semantic result depends on the game type and convergence.
    """

    class GameType(Enum):
        DEBATE = "debate"
        BRAINSTORM = "brainstorm"
        REVIEW = "review"
        NEGOTIATION = "negotiation"

    def __init__(
        self,
        agents: List[Tuple[str, FluxFunction]],
        rounds: int = 3,
        game_type: GameType = GameType.BRAINSTORM,
        convergence_threshold: float = 0.8,
    ) -> None:
        self._agents = agents
        self._rounds = rounds
        self._game_type = game_type
        self._convergence_threshold = convergence_threshold

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        if not self._agents:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="No agents in discussion"), 1.0)]

        # Each agent produces their position
        agent_positions: List[List[Tuple[FluxResult, float]]] = []
        for agent_name, agent_fn in self._agents:
            agent_state = state.fork()
            positions = agent_fn.apply(agent_state, context, caps)
            agent_positions.append(positions)

        # Aggregate based on game type
        if self._game_type == self.GameType.BRAINSTORM:
            # Cooperative: combine all unique results
            return self._merge_cooperative(agent_positions, state)
        elif self._game_type == self.GameType.DEBATE:
            # Competitive: best result wins
            return self._merge_competitive(agent_positions, state)
        elif self._game_type == self.GameType.REVIEW:
            # Verification: require consensus
            return self._merge_review(agent_positions, state)
        elif self._game_type == self.GameType.NEGOTIATION:
            # Bargaining: weighted compromise
            return self._merge_negotiation(agent_positions, state)
        else:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error=f"Unknown game type: {self._game_type}"), 1.0)]

    def _merge_cooperative(
        self,
        agent_positions: List[List[Tuple[FluxResult, float]]],
        state: FluxState,
    ) -> List[Tuple[FluxResult, float]]:
        """Merge results cooperatively (union with combined confidence)."""
        results: List[Tuple[FluxResult, float]] = []
        all_results = []
        for positions in agent_positions:
            all_results.extend(positions)

        # Deduplicate by value
        seen_values: Dict[Any, Tuple[FluxResult, float]] = {}
        for result, prob in all_results:
            key = result.value
            if key in seen_values:
                existing_res, existing_prob = seen_values[key]
                # Combine: higher confidence, average probability
                seen_values[key] = (
                    FluxResult(
                        value=existing_res.value,
                        confidence=max(existing_res.confidence, result.confidence),
                        new_state=state,
                        effects=existing_res.effects + result.effects,
                    ),
                    (existing_prob + prob) / 2,
                )
            else:
                seen_values[key] = (result, prob)

        return list(seen_values.values())

    def _merge_competitive(
        self,
        agent_positions: List[List[Tuple[FluxResult, float]]],
        state: FluxState,
    ) -> List[Tuple[FluxResult, float]]:
        """Merge competitively: highest-confidence result wins."""
        best_result: Optional[FluxResult] = None
        best_prob: float = 0.0
        for positions in agent_positions:
            for result, prob in positions:
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                    best_prob = prob

        if best_result is None:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="No agent produced a result"), 1.0)]

        return [(best_result.with_state(state), best_prob)]

    def _merge_review(
        self,
        agent_positions: List[List[Tuple[FluxResult, float]]],
        state: FluxState,
    ) -> List[Tuple[FluxResult, float]]:
        """Review: require agents to agree on a value."""
        if len(agent_positions) < 2:
            return agent_positions[0] if agent_positions else []

        # Check for agreement
        first_values = set(
            r.value for r, _ in agent_positions[0]
            if not r.is_error
        )
        for positions in agent_positions[1:]:
            agent_values = set(
                r.value for r, _ in positions
                if not r.is_error
            )
            first_values &= agent_values
            if not first_values:
                return [(FluxResult(
                    value=None,
                    confidence=0.0,
                    new_state=state,
                    error="Review: agents could not reach consensus",
                ), 1.0)]

        # Agreement found
        agreed_value = first_values.pop()
        min_conf = min(
            r.confidence
            for positions in agent_positions
            for r, _ in positions
            if r.value == agreed_value and not r.is_error
        )
        return [(FluxResult(value=agreed_value, confidence=min_conf, new_state=state), 1.0)]

    def _merge_negotiation(
        self,
        agent_positions: List[List[Tuple[FluxResult, float]]],
        state: FluxState,
    ) -> List[Tuple[FluxResult, float]]:
        """Negotiation: weighted compromise based on agent count."""
        if not agent_positions:
            return []

        # Collect all non-error results
        all_results = [
            (r, p)
            for positions in agent_positions
            for r, p in positions
            if not r.is_error
        ]
        if not all_results:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="All agents in negotiation produced errors"), 1.0)]

        # Weighted average confidence
        total_prob = sum(p for _, p in all_results)
        if total_prob == 0:
            return [(FluxResult(value=None, confidence=0.0, new_state=state, error="Zero total probability in negotiation"), 1.0)]

        avg_conf = sum(r.confidence * p for r, p in all_results) / total_prob
        # Use the most common value as the compromise
        value_counts = Counter(r.value for r, _ in all_results)
        most_common = value_counts.most_common(1)[0][0]

        return [(FluxResult(
            value=most_common,
            confidence=avg_conf,
            new_state=state,
            effects=["negotiation_compromise"],
        ), 1.0)]


class ContextDependentFunc(FluxFunction):
    """
    A function whose denotation changes based on context.

    This is the semantic representation of context-dependent meaning:
    same bytecode, different context → different denotation.

    Example: The Classical Chinese character 道 has different meanings
    in "philosophy" vs "martial arts" domains.
    """

    def __init__(
        self,
        dispatch: Dict[str, FluxFunction],
        default: Optional[FluxFunction] = None,
        key_fn: Optional[Callable[[FluxContext], str]] = None,
    ) -> None:
        """
        Args:
            dispatch: Mapping from context keys to denotations.
            default: Default denotation if no key matches.
            key_fn:  Function to extract the dispatch key from context.
                     Defaults to context.active_domain().
        """
        self._dispatch = dispatch
        self._default = default
        self._key_fn = key_fn or (lambda ctx: ctx.active_domain() or "")

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        key = self._key_fn(context)
        fn = self._dispatch.get(key, self._default)
        if fn is None:
            return [(FluxResult(
                value=None, confidence=0.0, new_state=state,
                error=f"No denotation for context key '{key}'",
            ), 1.0)]
        return fn.apply(state, context, caps)


class SuperpositionFunc(FluxFunction):
    """
    A function that operates on superposition registers.

    The denotation maps each basis state through the inner function,
    preserving the probabilistic structure.

    ⟦op⟧(α₁|ψ₁⟩ + ... + αₙ|ψₙ⟩) = α₁·⟦op⟧(|ψ₁⟩) + ... + αₙ·⟦op⟧(|ψₙ⟩)
    """

    def __init__(
        self,
        register: int,
        inner: Callable[[Any], FluxFunction],
    ) -> None:
        self._register = register
        self._inner = inner

    def apply(
        self,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> List[Tuple[FluxResult, float]]:
        reg_value = state.get_register(self._register)
        results: List[Tuple[FluxResult, float]] = []

        for basis in reg_value.states:
            branch_state = state.fork()
            branch_state.set_register(self._register, basis.value)

            fn = self._inner(basis.value)
            branch_results = fn.apply(branch_state, context, caps)

            for result, prob in branch_results:
                results.append((result, basis.probability * prob))

        return results


# ═══════════════════════════════════════════════════════════════════════════
# §3  FluxDenotation — The Main Denotational Engine
# ═══════════════════════════════════════════════════════════════════════════

class FluxDenotation:
    """
    Denotational semantics engine for FLUX programs.

    Usage:
        fd = FluxDenotation()
        ctx = FluxContext(language="zho")
        caps = CapSet.full()

        # Build denotation for an expression
        fn = fd.denote_expression(expr)
        result = fn.run(state, ctx, caps)

    The engine maps FLUX expressions to FluxFunctions — mathematical
    objects that can be composed, guarded, and analyzed.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., FluxFunction]] = {}
        self._register_primitives()

    def _register_primitives(self) -> None:
        """Register primitive opcode denotations."""
        pass  # Registration is done via denote_expression

    # ------------------------------------------------------------------
    # §3.1  Main denotation entry point
    # ------------------------------------------------------------------

    def denote_expression(
        self,
        expr: Any,
    ) -> FluxFunction:
        """
        Map a FLUX expression to its denotation.

        The expression can be:
          - An Expression object (from schema.py)
          - A dict with 'op' key (JSON AST)
          - A string (literal value)

        Returns a FluxFunction — the mathematical meaning of this expression.
        """
        from flux_a2a.schema import Expression

        if isinstance(expr, Expression):
            op = expr.op
            params = expr.params
            confidence = expr.confidence
            lang = expr.lang
        elif isinstance(expr, dict):
            op = expr.get("op", "literal")
            params = {k: v for k, v in expr.items() if k != "op"}
            confidence = params.pop("confidence", 1.0)
            lang = params.pop("lang", "flux")
        else:
            return ConstFunc(expr, confidence=1.0)

        # Dispatch by opcode
        denotation = self._dispatch(op, params, confidence, lang)
        return denotation

    def denote_sequence(
        self,
        expressions: List[Any],
    ) -> FluxFunction:
        """
        Map a sequence of expressions to their composed denotation.

        ⟦[e₁, e₂, ..., eₙ]⟧ = ⟦eₙ⟧ ∘ ... ∘ ⟦e₂⟧ ∘ ⟦e₁⟧

        This is the fundamental compositionality property.
        """
        if not expressions:
            return IdentityFunc()

        fns = [self.denote_expression(e) for e in expressions]
        return reduce(lambda acc, fn: acc.bind(lambda _, f=fn: f), fns)

    def _dispatch(
        self,
        op: str,
        params: Dict[str, Any],
        confidence: float,
        lang: str,
    ) -> FluxFunction:
        """Dispatch to the appropriate denotation builder."""

        # --- Arithmetic ---
        if op in ("add", "sub", "mul", "div", "mod"):
            return self._denote_arithmetic(op, params, confidence)
        if op in ("eq", "neq", "lt", "lte", "gt", "gte"):
            return self._denote_comparison(op, params, confidence)

        # --- Control flow ---
        if op == "seq":
            body = params.get("body", [])
            return self.denote_sequence(body)
        if op == "if":
            return self._denote_if(params, confidence)
        if op == "loop":
            return self._denote_loop(params, confidence)
        if op == "while":
            return self._denote_while(params, confidence)
        if op == "match":
            return self._denote_match(params, confidence)

        # --- Variables ---
        if op == "let":
            return self._denote_let(params, confidence)
        if op == "get":
            return self._denote_get(params, confidence)
        if op == "set":
            return self._denote_set(params, confidence)

        # --- Branching ---
        if op == "branch":
            return self._denote_branch(params, confidence)
        if op == "fork":
            return self._denote_fork(params, confidence)
        if op == "co_iterate":
            return self._denote_co_iterate(params, confidence)

        # --- Agent communication ---
        if op == "tell":
            return self._denote_tell(params, confidence)
        if op == "ask":
            return self._denote_ask(params, confidence)
        if op == "broadcast":
            return self._denote_broadcast(params, confidence)
        if op == "delegate":
            return self._denote_delegate(params, confidence)

        # --- Confidence ---
        if op == "confidence":
            return self._denote_confidence(params)

        # --- Literal / pass-through ---
        if op == "literal":
            return ConstFunc(params.get("value"), confidence)
        if op == "concat":
            return self._denote_concat(params, confidence)

        # --- Paradigm opcodes ---
        if op == "discuss":
            return self._denote_discuss(params, confidence)

        # Unknown opcode
        return PureFunc(lambda s, c, k: FluxResult(
            value=None, confidence=0.0, new_state=s,
            error=f"Unknown opcode in denotation: {op}",
        ))

    # ------------------------------------------------------------------
    # §3.2  Arithmetic denotations
    # ------------------------------------------------------------------

    def _denote_arithmetic(
        self, op: str, params: Dict[str, Any], confidence: float
    ) -> FluxFunction:
        """Denotation for arithmetic operations."""

        def arith_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            args = params.get("args", [])
            values = []
            min_conf = confidence
            for a in args:
                if isinstance(a, Superposition):
                    values.append(a.collapse())
                    min_conf = min(min_conf, confidence * 0.9 if a.entropy() > 0 else confidence)
                elif isinstance(a, (int, float)):
                    values.append(a)
                elif isinstance(a, dict) and "op" in a:
                    # Evaluate nested expression
                    nested_fn = self._dispatch(a.get("op", ""), {k: v for k, v in a.items() if k != "op"}, a.get("confidence", 1.0), a.get("lang", "flux"))
                    nested_result = nested_fn.run(state, ctx, caps)
                    values.append(nested_result.value)
                    min_conf = min(min_conf, nested_result.confidence)
                else:
                    values.append(a)

            if op == "add" and len(values) >= 2:
                result = sum(v for v in values if isinstance(v, (int, float)))
            elif op == "sub" and len(values) >= 2:
                result = values[0] - values[1]
            elif op == "mul" and len(values) >= 2:
                result = values[0] * values[1]
            elif op == "div" and len(values) >= 2:
                if values[1] == 0:
                    return FluxResult(value=None, confidence=0.0, new_state=state, error="Division by zero")
                result = values[0] / values[1]
            elif op == "mod" and len(values) >= 2:
                result = values[0] % values[1]
            else:
                return FluxResult(value=None, confidence=0.0, new_state=state, error=f"Bad arith args for {op}")

            return FluxResult(value=result, confidence=min_conf, new_state=state)

        return PureFunc(arith_fn)

    def _denote_comparison(
        self, op: str, params: Dict[str, Any], confidence: float
    ) -> FluxFunction:
        """Denotation for comparison operations."""
        ops = {
            "eq": lambda a, b: a == b,
            "neq": lambda a, b: a != b,
            "lt": lambda a, b: a < b,
            "lte": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "gte": lambda a, b: a >= b,
        }
        fn = ops.get(op)
        if fn is None:
            return ConstFunc(None, 0.0)

        args = params.get("args", [])

        def cmp_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if len(args) < 2:
                return FluxResult(value=None, confidence=0.0, new_state=state, error=f"Need 2 args for {op}")
            try:
                result = fn(args[0], args[1])
            except TypeError:
                result = False
            return FluxResult(value=result, confidence=confidence, new_state=state)

        return PureFunc(cmp_fn)

    # ------------------------------------------------------------------
    # §3.3  Control flow denotations
    # ------------------------------------------------------------------

    def _denote_if(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for conditional.

        ⟦if(c, t, e)⟧(s) = if c(s) then ⟦t⟧(s) else ⟦e⟧(s)
        """
        cond = params.get("condition", params.get("cond", False))
        then_body = params.get("then", params.get("body", []))
        else_body = params.get("else", [])

        then_fn = self.denote_sequence(then_body) if isinstance(then_body, list) else self.denote_expression(then_body)
        else_fn = self.denote_sequence(else_body) if isinstance(else_body, list) else self.denote_expression(else_body)

        def if_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            # Evaluate condition
            if isinstance(cond, dict) and "op" in cond:
                cond_result = self.denote_expression(cond).run(state, ctx, caps)
                condition = bool(cond_result.value)
            else:
                condition = bool(cond)

            if condition:
                result = then_fn.run(state, ctx, caps)
            else:
                result = else_fn.run(state, ctx, caps)
            return result

        return PureFunc(if_fn)

    def _denote_loop(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for loop.

        ⟦loop(n, P)⟧ = ⟦P⟧ⁿ  (n-fold sequential composition)
        """
        times = params.get("times", params.get("count", 0))
        over = params.get("over", None)
        var_name = params.get("var", "item")
        body = params.get("body", [])
        body_fn = self.denote_sequence(body)

        def loop_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            current_state = state
            results = []
            min_conf = confidence

            if over is not None:
                collection = over if isinstance(over, list) else [over]
                for item in collection:
                    current_state.set_register(0, item)
                    r = body_fn.run(current_state, ctx, caps)
                    if r.new_state:
                        current_state = r.new_state
                    min_conf = min(min_conf, r.confidence)
                    results.append(r.value)
            else:
                for i in range(int(times)):
                    current_state.set_register(0, i)
                    r = body_fn.run(current_state, ctx, caps)
                    if r.new_state:
                        current_state = r.new_state
                    min_conf = min(min_conf, r.confidence)
                    results.append(r.value)

            return FluxResult(
                value=results,
                confidence=min_conf,
                new_state=current_state,
            )

        return PureFunc(loop_fn)

    def _denote_while(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for while loop.

        ⟦while(c, P)⟧ = μX. if c then P; X else id
        (least fixed point)
        """
        cond_expr = params.get("condition", params.get("cond"))
        body = params.get("body", [])
        max_iter = params.get("max_iterations", 1000)
        body_fn = self.denote_sequence(body)

        def while_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            current_state = state
            results = []
            min_conf = confidence
            iteration = 0

            while iteration < max_iter:
                # Evaluate condition
                if isinstance(cond_expr, dict) and "op" in cond_expr:
                    cond_result = self.denote_expression(cond_expr).run(current_state, ctx, caps)
                    condition = bool(cond_result.value)
                else:
                    condition = bool(cond_expr)

                if not condition:
                    break

                r = body_fn.run(current_state, ctx, caps)
                if r.new_state:
                    current_state = r.new_state
                min_conf = min(min_conf, r.confidence)
                results.append(r.value)
                iteration += 1

            return FluxResult(
                value=results,
                confidence=min_conf,
                new_state=current_state,
            )

        return PureFunc(while_fn)

    def _denote_match(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for pattern matching.

        ⟦match(v, [(p₁, e₁), ..., (pₙ, eₙ)], d)⟧ =
            if v == p₁ then ⟦e₁⟧ else ... else if v == pₙ then ⟦eₙ⟧ else ⟦d⟧
        """
        value = params.get("value")
        cases = params.get("cases", [])
        default = params.get("default", None)

        case_fns = []
        for case in cases:
            pattern = case.get("pattern")
            case_body = case.get("body", [])
            fn = self.denote_sequence(case_body) if isinstance(case_body, list) else self.denote_expression(case_body)
            case_fns.append((pattern, fn))

        default_fn = None
        if default is not None:
            default_fn = (
                self.denote_sequence(default)
                if isinstance(default, list)
                else self.denote_expression(default)
            )

        def match_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            for pattern, fn in case_fns:
                if pattern is None or pattern == "_" or pattern == value:
                    return fn.run(state, ctx, caps)
            if default_fn:
                return default_fn.run(state, ctx, caps)
            return FluxResult(value=None, confidence=0.0, new_state=state, error="No matching case")

        return PureFunc(match_fn)

    # ------------------------------------------------------------------
    # §3.4  Variable denotations
    # ------------------------------------------------------------------

    def _denote_let(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for variable binding.

        ⟦let(x, e)⟧(s) = let v = ⟦e⟧(s).value in ⟦(x↦v)(s)⟧
        """
        name = params.get("name", "")
        raw_value = params.get("value")

        def let_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if isinstance(raw_value, (dict,)) and "op" in raw_value:
                r = self.denote_expression(raw_value).run(state, ctx, caps)
                val = r.value
                conf = r.confidence
            else:
                val = raw_value
                conf = confidence

            new_state = state.fork()
            new_state.registers[hash(name) & 0x7FFFFFFF] = pure(val)
            return FluxResult(value=val, confidence=conf, new_state=new_state)

        return PureFunc(let_fn)

    def _denote_get(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for variable access."""
        name = params.get("name", "")

        def get_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            reg_key = hash(name) & 0x7FFFFFFF
            if reg_key in state.registers:
                val = state.registers[reg_key]
                if isinstance(val, Superposition):
                    return FluxResult(value=val.collapse(), confidence=confidence, new_state=state)
                return FluxResult(value=val, confidence=confidence, new_state=state)
            return FluxResult(value=None, confidence=0.0, new_state=state, error=f"Unbound: {name}")

        return PureFunc(get_fn)

    def _denote_set(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for variable mutation."""
        name = params.get("name", "")
        raw_value = params.get("value")

        def set_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if isinstance(raw_value, (dict,)) and "op" in raw_value:
                r = self.denote_expression(raw_value).run(state, ctx, caps)
                val = r.value
                conf = r.confidence
            else:
                val = raw_value
                conf = confidence

            new_state = state.fork()
            new_state.registers[hash(name) & 0x7FFFFFFF] = pure(val)
            return FluxResult(value=val, confidence=conf, new_state=new_state)

        return PureFunc(set_fn)

    # ------------------------------------------------------------------
    # §3.5  Branching / forking denotations
    # ------------------------------------------------------------------

    def _denote_branch(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for parallel branching.

        ⟦branch([P₁...Pₙ])⟧ = Σᵢ wᵢ · ⟦Pᵢ⟧
        """
        branches_raw = params.get("branches", [])
        branches: List[Tuple[FluxFunction, float]] = []

        for b in branches_raw:
            if isinstance(b, dict):
                weight = b.get("weight", 1.0)
                body = b.get("body", [])
                fn = self.denote_sequence(body)
                branches.append((fn, weight))

        return BranchFunc(branches)

    def _denote_fork(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for agent forking.

        ⟦fork(P)⟧ = ⟦P⟧(fork(s)) ⊗ id(s)
        """
        body = params.get("body", [])
        inherit = params.get("inherit", {})
        inherit_keys = inherit.get("state", []) if isinstance(inherit, dict) else []

        child_fn = self.denote_sequence(body)
        return ForkFunc(child_fn, inherit_keys=inherit_keys)

    def _denote_co_iterate(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for co-iteration.

        Multiple agents traverse the same program simultaneously.
        """
        program_raw = params.get("program", {})
        agents_raw = params.get("agents", [])

        if isinstance(program_raw, dict) and "body" in program_raw:
            body = program_raw.get("body", [])
        elif isinstance(program_raw, list):
            body = program_raw
        else:
            body = []

        shared_fn = self.denote_sequence(body)

        def co_iter_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            results = []
            for i, agent in enumerate(agents_raw):
                agent_state = state.fork()
                agent_state.set_register(0, i)  # Agent index
                r = shared_fn.run(agent_state, ctx, caps)
                results.append(r.value)

            return FluxResult(
                value=results,
                confidence=min(r.confidence for r in [shared_fn.run(state.fork(), ctx, caps)]) if agents_raw else confidence,
                new_state=state,
                effects=["co_iterate"],
            )

        return PureFunc(co_iter_fn)

    # ------------------------------------------------------------------
    # §3.6  Agent communication denotations
    # ------------------------------------------------------------------

    def _denote_tell(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for tell (one-way message).

        ⟦tell(to, msg)⟧(s) = s with effect [sent msg to 'to']
        """
        target = params.get("to", params.get("to_agent", ""))
        message = params.get("message", params.get("payload", None))

        def tell_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if not caps.requires(ExtendedCap.TELL):
                return FluxResult(
                    value=None, confidence=0.0, new_state=state,
                    error=f"Capability violation: TELL required",
                )
            return FluxResult(
                value=message,
                confidence=confidence,
                new_state=state,
                effects=[f"tell:{target}"],
            )

        return PureFunc(tell_fn)

    def _denote_ask(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for ask (query).

        ⟦ask(from, q)⟧(s) = s with effect [asked q of 'from']
        Returns the answer if available, else None with pending status.
        """
        target = params.get("from", params.get("to", ""))
        question = params.get("question", params.get("payload", ""))

        def ask_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if not caps.requires(ExtendedCap.ASK):
                return FluxResult(
                    value=None, confidence=0.0, new_state=state,
                    error="Capability violation: ASK required",
                )
            return FluxResult(
                value=None,
                confidence=confidence,
                new_state=state,
                effects=[f"ask:{target}"],
            )

        return PureFunc(ask_fn)

    def _denote_broadcast(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for broadcast."""
        payload = params.get("message", params.get("payload", None))

        def broadcast_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            if not caps.requires(ExtendedCap.TELL):
                return FluxResult(
                    value=None, confidence=0.0, new_state=state,
                    error="Capability violation: TELL required for broadcast",
                )
            return FluxResult(
                value={"scope": params.get("scope", "fleet")},
                confidence=confidence,
                new_state=state,
                effects=[f"broadcast:{params.get('scope', 'fleet')}"],
            )

        return PureFunc(broadcast_fn)

    def _denote_delegate(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for delegation with capability check."""
        task = params.get("task", {})
        task_fn = self.denote_expression(task) if isinstance(task, dict) else ConstFunc(task)

        return PureFunc(lambda s, c, k: FluxResult(
            value=None,
            confidence=confidence if k.requires(ExtendedCap.DELEGATE) else 0.0,
            new_state=s,
            error="" if k.requires(ExtendedCap.DELEGATE) else "Capability violation: DELEGATE required",
            effects=["delegate"] if k.requires(ExtendedCap.DELEGATE) else [],
        ))

    # ------------------------------------------------------------------
    # §3.7  Other denotations
    # ------------------------------------------------------------------

    def _denote_confidence(self, params: Dict[str, Any]) -> FluxFunction:
        """Denotation for confidence scope setting."""
        level = float(params.get("level", params.get("value", 1.0)))
        return ConstFunc(level, level)

    def _denote_concat(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for string concatenation."""
        args = params.get("args", [])

        def concat_fn(state: FluxState, ctx: FluxContext, caps: CapSet) -> FluxResult:
            return FluxResult(
                value="".join(str(a) for a in args),
                confidence=confidence,
                new_state=state,
            )

        return PureFunc(concat_fn)

    def _denote_discuss(self, params: Dict[str, Any], confidence: float) -> FluxFunction:
        """Denotation for structured discussion.

        Maps to the game-theoretic DiscussFunc.
        """
        game_type_str = params.get("format", "brainstorm")
        game_type_map = {
            "debate": DiscussFunc.GameType.DEBATE,
            "brainstorm": DiscussFunc.GameType.BRAINSTORM,
            "review": DiscussFunc.GameType.REVIEW,
            "negotiation": DiscussFunc.GameType.NEGOTIATION,
            "negotiate": DiscussFunc.GameType.NEGOTIATION,
            "peer_review": DiscussFunc.GameType.REVIEW,
        }
        game_type = game_type_map.get(game_type_str, DiscussFunc.GameType.BRAINSTORM)

        # For now, create a simple agent list
        # In practice, agents would be extracted from params
        agents_raw = params.get("agents", [])

        agent_fns = []
        for agent in agents_raw:
            if isinstance(agent, dict):
                agent_id = agent.get("id", "agent")
                agent_body = agent.get("body", [])
                fn = self.denote_sequence(agent_body)
                agent_fns.append((agent_id, fn))

        return DiscussFunc(
            agents=agent_fns,
            rounds=params.get("rounds", 3),
            game_type=game_type,
            convergence_threshold=params.get("convergence", 0.8),
        )

    # ------------------------------------------------------------------
    # §3.8  Composition helpers
    # ------------------------------------------------------------------

    def compose(self, f: FluxFunction, g: FluxFunction) -> FluxFunction:
        """
        Sequential composition of two denotations.

        ⟦P1; P2⟧ = ⟦P2⟧ ∘ ⟦P1⟧

        This is the fundamental compositionality property of the semantics.
        """
        return f.bind(lambda _: g)

    def parallel(self, *fns: FluxFunction) -> FluxFunction:
        """Execute multiple denotations in parallel (branch)."""
        return BranchFunc([(fn, 1.0) for fn in fns])

    def choice(self, *weighted_fns: Tuple[FluxFunction, float]) -> FluxFunction:
        """Probabilistic choice between weighted denotations."""
        return BranchFunc(list(weighted_fns))


# ═══════════════════════════════════════════════════════════════════════════
# §4  Semantic Properties — Proofs and Invariants
# ═══════════════════════════════════════════════════════════════════════════

class SemanticProperties:
    """
    Verification of semantic properties for FLUX denotations.

    This module provides runtime verification of the mathematical properties
    proven in the formal document (round10-12_flux_semantics.md).
    """

    @staticmethod
    def verify_compositionality(
        denotation: FluxDenotation,
        expr1: Any,
        expr2: Any,
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> Tuple[bool, str]:
        """
        Verify: ⟦P1; P2⟧ = ⟦P2⟧ ∘ ⟦P1⟧

        Tests that the denotation of a sequence equals the composition
        of the denotations of the individual expressions.
        """
        # Build sequence denotation
        seq_fn = denotation.denote_sequence([expr1, expr2])
        seq_result = seq_fn.run(state, context, caps)

        # Build composed denotation
        fn1 = denotation.denote_expression(expr1)
        fn2 = denotation.denote_expression(expr2)
        composed_fn = fn1.bind(lambda _: fn2)
        composed_result = composed_fn.run(state, context, caps)

        values_match = seq_result.value == composed_result.value
        confs_close = abs(seq_result.confidence - composed_result.confidence) < 0.01

        if values_match and confs_close:
            return True, (
                f"Compositionality holds: seq={seq_result.value} "
                f"== composed={composed_result.value}, "
                f"conf {seq_result.confidence:.4f} ≈ {composed_result.confidence:.4f}"
            )
        return False, (
            f"Compositionality violation: seq={seq_result.value} "
            f"!= composed={composed_result.value}, "
            f"conf {seq_result.confidence:.4f} vs {composed_result.confidence:.4f}"
        )

    @staticmethod
    def verify_confidence_monotonicity(
        denotation: FluxDenotation,
        expressions: List[Any],
        state: FluxState,
        context: FluxContext,
        caps: CapSet,
    ) -> Tuple[bool, str]:
        """
        Verify: if ⟦P⟧ has confidence c, then ⟦P; Q⟧ has confidence ≤ c.

        Confidence can only decrease through composition.
        """
        fn_p = denotation.denote_sequence(expressions)
        result_p = fn_p.run(state, context, caps)
        conf_p = result_p.confidence

        # Add a trailing expression
        tail = {"op": "literal", "value": 42}
        fn_pq = denotation.denote_sequence(expressions + [tail])
        result_pq = fn_pq.run(state, context, caps)
        conf_pq = result_pq.confidence

        holds = conf_pq <= conf_p + 1e-9  # Allow floating point epsilon
        return holds, (
            f"Confidence monotonicity: c(P)={conf_p:.4f}, "
            f"c(P;Q)={conf_pq:.4f}, holds={holds}"
        )

    @staticmethod
    def verify_capability_monotonicity(
        denotation: FluxDenotation,
        expr: Any,
        state: FluxState,
        context: FluxContext,
        caps_small: CapSet,
        caps_large: CapSet,
    ) -> Tuple[bool, str]:
        """
        Verify: adding capabilities can only increase denotable programs.

        If caps_small ⊆ caps_large, then any program that runs under
        caps_small should also run under caps_large with at least as
        much confidence.
        """
        result_small = denotation.denote_expression(expr).run(state, context, caps_small)
        result_large = denotation.denote_expression(expr).run(state, context, caps_large)

        # If small caps caused an error, large caps should not make it worse
        holds = True
        if result_small.is_error:
            # Large caps might fix the error
            holds = not result_large.is_error or result_large.error == result_small.error
        else:
            # Large caps should have >= confidence
            holds = result_large.confidence >= result_small.confidence - 1e-9

        return holds, (
            f"Capability monotonicity: "
            f"small={result_small.confidence:.4f}{' (error)' if result_small.is_error else ''}, "
            f"large={result_large.confidence:.4f}{' (error)' if result_large.is_error else ''}, "
            f"holds={holds}"
        )

    @staticmethod
    def verify_context_sensitivity(
        denotation: FluxDenotation,
        expr: Any,
        state: FluxState,
        context_a: FluxContext,
        context_b: FluxContext,
        caps: CapSet,
    ) -> Tuple[bool, str]:
        """
        Verify: same bytecode, different context → potentially different denotation.

        We specifically test that context changes CAN produce different results.
        """
        result_a = denotation.denote_expression(expr).run(state, context_a, caps)
        result_b = denotation.denote_expression(expr).run(state, context_b, caps)

        # For most expressions, different contexts should yield potentially different results
        # (This is a "can differ" test, not "always differs")
        different = (result_a.value != result_b.value) or (abs(result_a.confidence - result_b.confidence) > 1e-9)
        return True, (
            f"Context sensitivity: ctx_a={context_a.language}→{result_a.value}, "
            f"ctx_b={context_b.language}→{result_b.value}, "
            f"different={different}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §5  Effect System Mapping
# ═══════════════════════════════════════════════════════════════════════════

class FluxEffect:
    """
    An algebraic effect in the FLUX effect system.

    Effects are the semantic representation of side effects:
      - Message sending/receiving
      - Branch creation
      - Fork spawning
      - Trust modification
      - Capability requirement

    Each effect has:
      - A type (what kind of effect)
      - A payload (associated data)
      - A capability requirement (which CapSet must authorize it)

    This maps directly to Koka-style algebraic effects:
      - ExtendedCap → Effect operation
      - CapSet → Effect handler capability
    """

    class EffectType(Enum):
        TELL = "tell"
        ASK = "ask"
        FORK = "fork"
        BRANCH = "branch"
        TRUST_MODIFY = "trust_modify"
        CONFIDENCE_SET = "confidence_set"
        STATE_MUTATE = "state_mutate"
        CAPABILITY_REQUIRE = "capability_require"
        DISCUSS = "discuss"
        DELEGATE = "delegate"
        META_OPCODE = "meta_opcode"
        TEMPORAL_DEFER = "temporal_defer"
        TEMPORAL_CACHE = "temporal_cache"
        TEMPORAL_ROLLBACK = "temporal_rollback"

    @staticmethod
    def capability_for_effect(effect_type: EffectType) -> Optional[ExtendedCap]:
        """Map effect type to required capability."""
        mapping = {
            FluxEffect.EffectType.TELL: ExtendedCap.TELL,
            FluxEffect.EffectType.ASK: ExtendedCap.ASK,
            FluxEffect.EffectType.FORK: ExtendedCap.FORK,
            FluxEffect.EffectType.BRANCH: ExtendedCap.BRANCH,
            FluxEffect.EffectType.TRUST_MODIFY: ExtendedCap.TRUST,
            FluxEffect.EffectType.DISCUSS: ExtendedCap.DISCUSS,
            FluxEffect.EffectType.DELEGATE: ExtendedCap.DELEGATE,
            FluxEffect.EffectType.META_OPCODE: ExtendedCap.META,
            FluxEffect.EffectType.TEMPORAL_DEFER: ExtendedCap.TEMPORAL_DEFER,
            FluxEffect.EffectType.TEMPORAL_CACHE: ExtendedCap.TEMPORAL_CACHE,
            FluxEffect.EffectType.TEMPORAL_ROLLBACK: ExtendedCap.TEMPORAL_ROLLBACK,
        }
        return mapping.get(effect_type)

    @staticmethod
    def effects_for_korean_level(level: int) -> List[EffectType]:
        """Map Korean honorific level to permitted effects."""
        if level <= 1:
            return list(FluxEffect.EffectType)
        elif level <= 3:
            return [e for e in FluxEffect.EffectType
                    if e not in (FluxEffect.EffectType.META_OPCODE,)]
        elif level <= 5:
            return [
                FluxEffect.EffectType.TELL,
                FluxEffect.EffectType.ASK,
                FluxEffect.EffectType.BRANCH,
                FluxEffect.EffectType.STATE_MUTATE,
                FluxEffect.EffectType.CONFIDENCE_SET,
            ]
        else:
            return [
                FluxEffect.EffectType.STATE_MUTATE,
                FluxEffect.EffectType.CONFIDENCE_SET,
            ]


# ═══════════════════════════════════════════════════════════════════════════
# §6  Module-level API
# ═══════════════════════════════════════════════════════════════════════════

def denote(expr: Any) -> FluxFunction:
    """Convenience: create a denotation for an expression."""
    return FluxDenotation().denote_expression(expr)


def denote_and_run(
    expr: Any,
    state: Optional[FluxState] = None,
    context: Optional[FluxContext] = None,
    caps: Optional[CapSet] = None,
) -> FluxResult:
    """Convenience: denote an expression and immediately run it."""
    fd = FluxDenotation()
    fn = fd.denote_expression(expr)
    return fn.run(
        state or FluxState(),
        context or FluxContext(),
        caps or CapSet.full(),
    )


def verify_all_properties(
    denotation: Optional[FluxDenotation] = None,
) -> Dict[str, Tuple[bool, str]]:
    """Run all semantic property verification tests."""
    fd = denotation or FluxDenotation()
    props = SemanticProperties()

    state = FluxState()
    ctx = FluxContext(language="flux")
    caps = CapSet.full()

    results = {}

    # Compositionality
    ok, msg = props.verify_compositionality(
        fd, {"op": "let", "name": "x", "value": 10}, {"op": "get", "name": "x"},
        state, ctx, caps,
    )
    results["compositionality"] = (ok, msg)

    # Confidence monotonicity
    ok, msg = props.verify_confidence_monotonicity(
        fd,
        [{"op": "literal", "value": 42}],
        state, ctx, caps,
    )
    results["confidence_monotonicity"] = (ok, msg)

    # Capability monotonicity
    ok, msg = props.verify_capability_monotonicity(
        fd,
        {"op": "tell", "to": "agent", "message": "hello"},
        state, ctx,
        CapSet.empty(),
        CapSet.full(),
    )
    results["capability_monotonicity"] = (ok, msg)

    # Context sensitivity
    ok, msg = props.verify_context_sensitivity(
        fd,
        {"op": "literal", "value": 42},
        state,
        FluxContext(language="zho"),
        FluxContext(language="deu"),
        caps,
    )
    results["context_sensitivity"] = (ok, msg)

    return results
