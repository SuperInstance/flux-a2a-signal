"""
FLUX Evolution Engine — NL Programming Through Natural Selection.

Implements the meta-compilation layer: the interpreter observes its own execution,
learns hot patterns, and generates specialized code paths.  Three levels of evolution:

  1. Pattern Learning  — Observe common NL patterns, optimize compilation.
  2. Grammar Evolution  — Users introduce new constructions; the engine adapts.
  3. Paradigm Emergence — New programming paradigms emerge from usage patterns.

Theoretical foundation:
  - Self-hosting compilers (C, Rust, PyPy bootstrapping)
  - Meta-tracing JITs (PyPy's observation-driven optimization)
  - Futamura projections (partial evaluation for compiler generation)
  - Darwinian evolution applied to language grammars

Key insight from PyPy: you don't need to KNOW what to optimize —
observe execution traces and learn.  If many users write "三只猫",
the interpreter should learn to fast-path this pattern without anyone
explicitly teaching it.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ===========================================================================
# Core Data Structures
# ===========================================================================

class EvolutionLevel(str, Enum):
    """The three tiers of evolution."""
    PATTERN_LEARNING = "pattern_learning"
    GRAMMAR_EVOLUTION = "grammar_evolution"
    PARADIGM_EMERGENCE = "paradigm_emergence"


class OptimizationKind(str, Enum):
    """Categories of discovered optimizations."""
    HOT_PATH_INLINE = "hot_path_inline"
    DEAD_PATTERN_ELIM = "dead_pattern_elim"
    TYPE_SPECIALIZE = "type_specialize"
    CONSTANT_PROPAGATE = "constant_propagate"
    MACRO_EXPANSION = "macro_expansion"
    BRANCH_FUSION = "branch_fusion"
    CSE_PATTERN = "cse_pattern"
    VOCAB_FAST_PATH = "vocab_fast_path"
    GRAMMAR_EXTENSION = "grammar_extension"
    PARADIGM_SHIFT = "paradigm_shift"


# ===========================================================================
# Observation Records
# ===========================================================================

@dataclass
class NLObservation:
    """A single observation of an NL program execution."""
    id: str = ""
    timestamp: float = 0.0
    program_hash: str = ""
    op_sequence: list[str] = field(default_factory=list)
    lang_tags: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    result_confidence: float = 1.0
    variable_types: dict[str, str] = field(default_factory=dict)
    branch_decisions: list[dict[str, Any]] = field(default_factory=list)
    error_occurred: bool = False
    source_agent: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "program_hash": self.program_hash,
            "op_sequence": self.op_sequence,
            "lang_tags": self.lang_tags,
            "execution_time_ms": self.execution_time_ms,
            "result_confidence": self.result_confidence,
            "variable_types": self.variable_types,
            "branch_decisions": self.branch_decisions,
            "error_occurred": self.error_occurred,
            "source_agent": self.source_agent,
        }


@dataclass
class HotPath:
    """A frequently-executed sequence of operations."""
    sequence: tuple[str, ...]
    frequency: int = 0
    avg_execution_time_ms: float = 0.0
    total_executions: int = 0
    confidence_scores: list[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    lang_affinity: str = ""  # e.g. "zho", "deu", "eng"

    def __post_init__(self) -> None:
        now = time.time()
        if not self.first_seen:
            self.first_seen = now
        if not self.last_seen:
            self.last_seen = now

    @property
    def heat(self) -> float:
        """Heat score combining frequency, recency, and consistency."""
        if not self.total_executions:
            return 0.0
        age = time.time() - self.last_seen
        recency = math.exp(-age / 3600.0)  # 1-hour half-life
        freq_factor = min(1.0, self.total_executions / 100.0)
        conf_variance = 0.0
        if len(self.confidence_scores) > 1:
            mean_c = sum(self.confidence_scores) / len(self.confidence_scores)
            conf_variance = sum((c - mean_c) ** 2 for c in self.confidence_scores) / len(self.confidence_scores)
        consistency = math.exp(-conf_variance * 10)  # Low variance = high consistency
        return recency * freq_factor * consistency

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": list(self.sequence),
            "frequency": self.frequency,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "total_executions": self.total_executions,
            "heat": round(self.heat, 4),
            "lang_affinity": self.lang_affinity,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class NLPattern:
    """A natural language pattern observed across programs."""
    fingerprint: str = ""
    raw_forms: list[str] = field(default_factory=list)
    resolved_ops: list[str] = field(default_factory=list)
    frequency: int = 0
    confidence_avg: float = 1.0
    lang: str = ""
    grammar_slot: str = ""  # Where this pattern fits in the grammar
    is_idiomatic: bool = False  # True if this is a language-specific idiom

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "raw_forms": self.raw_forms[:5],
            "resolved_ops": self.resolved_ops,
            "frequency": self.frequency,
            "confidence_avg": round(self.confidence_avg, 3),
            "lang": self.lang,
            "grammar_slot": self.grammar_slot,
            "is_idiomatic": self.is_idiomatic,
        }


@dataclass
class CompiledPattern:
    """A specialized, pre-compiled path for a hot NL pattern."""
    pattern_fingerprint: str = ""
    bytecode: list[list[Any]] = field(default_factory=list)
    specialized_vars: dict[str, Any] = field(default_factory=dict)
    optimization_kind: str = ""
    speedup_estimate: float = 1.0
    created_at: float = 0.0
    invocation_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_fingerprint": self.pattern_fingerprint,
            "bytecode": self.bytecode,
            "specialized_vars": self.specialized_vars,
            "optimization_kind": self.optimization_kind,
            "speedup_estimate": round(self.speedup_estimate, 2),
            "created_at": self.created_at,
            "invocation_count": self.invocation_count,
        }


@dataclass
class Optimization:
    """A suggested optimization based on observed patterns."""
    id: str = ""
    kind: str = ""
    target_pattern: str = ""
    description: str = ""
    estimated_speedup: float = 1.0
    confidence: float = 0.0
    applies_to: list[str] = field(default_factory=list)  # op sequences it applies to
    bytecode_delta: list[list[Any]] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target_pattern": self.target_pattern,
            "description": self.description,
            "estimated_speedup": round(self.estimated_speedup, 2),
            "confidence": round(self.confidence, 3),
            "applies_to": self.applies_to,
            "rationale": self.rationale,
        }


@dataclass
class GrammarDelta:
    """A proposed change to the NL grammar based on observed usage."""
    id: str = ""
    change_type: str = ""  # "new_production" | "macro" | "syntax_sugar" | "idiom"
    name: str = ""
    pattern: str = ""
    expansion: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    frequency: int = 0
    languages: list[str] = field(default_factory=list)
    example_programs: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "change_type": self.change_type,
            "name": self.name,
            "pattern": self.pattern,
            "expansion": self.expansion,
            "confidence": round(self.confidence, 3),
            "frequency": self.frequency,
            "languages": self.languages,
            "example_programs": self.example_programs[:3],
            "rationale": self.rationale,
        }


@dataclass
class FitnessMetrics:
    """Measures how well the current grammar/interpreter serves users."""
    total_programs: int = 0
    successful_executions: float = 0.0
    avg_confidence: float = 1.0
    avg_execution_time_ms: float = 0.0
    error_rate: float = 0.0
    pattern_coverage: float = 0.0  # fraction of hot paths that are optimized
    grammar_utilization: float = 0.0  # fraction of grammar productions actually used
    lang_diversity: float = 0.0  # how many languages are actively used
    evolution_generation: int = 0
    paradigm_fitness: dict[str, float] = field(default_factory=dict)

    @property
    def overall_fitness(self) -> float:
        """Weighted composite fitness score."""
        return (
            0.25 * self.successful_executions
            + 0.20 * self.avg_confidence
            + 0.15 * (1.0 - self.error_rate)
            + 0.15 * self.pattern_coverage
            + 0.10 * self.grammar_utilization
            + 0.10 * self.lang_diversity
            + 0.05 * (1.0 - min(1.0, self.avg_execution_time_ms / 1000.0))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_programs": self.total_programs,
            "successful_executions": round(self.successful_executions, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "error_rate": round(self.error_rate, 3),
            "pattern_coverage": round(self.pattern_coverage, 3),
            "grammar_utilization": round(self.grammar_utilization, 3),
            "lang_diversity": round(self.lang_diversity, 3),
            "evolution_generation": self.evolution_generation,
            "overall_fitness": round(self.overall_fitness, 3),
            "paradigm_fitness": {k: round(v, 3) for k, v in self.paradigm_fitness.items()},
        }


# ===========================================================================
# Evolution Engine
# ===========================================================================

class EvolutionEngine:
    """
    The FLUX Evolution Engine — NL programming through natural selection.

    The engine operates on three levels:
      1. Pattern Learning: Observe common NL patterns, optimize compilation.
         When "三只猫" appears 500 times, it becomes a fast-path.
      2. Grammar Evolution: Users introduce new constructions; engine adapts.
         If 100 users independently write "for each X in Y", add it to the grammar.
      3. Paradigm Emergence: New paradigms emerge from aggregate patterns.
         If users consistently compose branch→co_iterate→synthesize, that's a paradigm.

    The engine is inspired by:
      - PyPy's meta-tracing JIT (observe and optimize without prior knowledge)
      - Darwinian evolution (mutation, selection, reproduction of patterns)
      - Futamura projections (compile the compiler through partial evaluation)
    """

    def __init__(
        self,
        hot_threshold: int = 10,
        max_observations: int = 10000,
        min_confidence_for_learning: float = 0.5,
    ) -> None:
        # Observation storage
        self._observations: list[NLObservation] = []
        self._max_observations = max_observations

        # Pattern tracking
        self._hot_paths: dict[tuple[str, ...], HotPath] = {}
        self._nl_patterns: dict[str, NLPattern] = {}
        self._compiled_patterns: dict[str, CompiledPattern] = {}

        # Grammar evolution
        self._grammar_deltas: list[GrammarDelta] = []
        self._adopted_grammar: list[GrammarDelta] = []

        # Configuration
        self._hot_threshold = hot_threshold
        self._min_confidence = min_confidence_for_learning

        # Counters
        self._generation = 0
        self._total_programs = 0
        self._total_errors = 0
        self._execution_times: list[float] = []
        self._confidence_history: list[float] = []
        self._ops_used: set[str] = set()
        self._langs_used: dict[str, int] = defaultdict(int)
        self._all_ops_known: set[str] = {
            "add", "sub", "mul", "div", "mod", "eq", "neq", "lt", "lte", "gt", "gte",
            "and", "or", "not", "xor", "concat", "length", "at", "collect", "reduce",
            "seq", "if", "loop", "while", "match", "let", "get", "set", "struct",
            "tell", "ask", "delegate", "broadcast", "signal", "await",
            "branch", "fork", "merge", "co_iterate", "trust", "confidence",
            "literal", "yield", "eval",
            # A2A protocol primitives
            "discuss", "synthesize", "reflect",
        }

    # ------------------------------------------------------------------
    # Level 1: Pattern Learning — Observe and Optimize
    # ------------------------------------------------------------------

    def observe(self, program: Any, execution_time_ms: float = 0.0,
                result_confidence: float = 1.0, variable_types: Optional[dict[str, str]] = None,
                branch_decisions: Optional[list[dict[str, Any]]] = None,
                error_occurred: bool = False, source_agent: str = "") -> None:
        """Record a program execution for learning.

        Parameters
        ----------
        program:
            A Program, dict, or list of expressions to observe.
        execution_time_ms:
            How long the program took to execute.
        result_confidence:
            Confidence of the result.
        variable_types:
            Observed variable types during execution.
        branch_decisions:
            Which branches were taken and why.
        error_occurred:
            Whether an error occurred during execution.
        source_agent:
            Which agent produced this program.
        """
        # Extract op sequence and lang tags
        op_sequence: list[str] = []
        lang_tags: list[str] = []
        raw_forms: list[str] = []

        expressions = self._extract_expressions(program)

        for expr in expressions:
            if isinstance(expr, dict):
                op = expr.get("op", "")
                if op:
                    op_sequence.append(op)
                    self._ops_used.add(op)
                lang = expr.get("lang", "")
                if lang and lang != "flux":
                    lang_tags.append(lang)
                    self._langs_used[lang] += 1
                # Track raw NL forms (the source text before compilation)
                raw = expr.get("source", expr.get("raw", ""))
                if raw:
                    raw_forms.append(raw)

        # Create observation
        program_hash = self._hash_ops(op_sequence)
        observation = NLObservation(
            program_hash=program_hash,
            op_sequence=op_sequence,
            lang_tags=lang_tags,
            execution_time_ms=execution_time_ms,
            result_confidence=result_confidence,
            variable_types=variable_types or {},
            branch_decisions=branch_decisions or [],
            error_occurred=error_occurred,
            source_agent=source_agent,
            meta={"raw_forms": raw_forms},
        )

        # Store observation (bounded)
        if len(self._observations) >= self._max_observations:
            self._observations.pop(0)
        self._observations.append(observation)

        # Update counters
        self._total_programs += 1
        if error_occurred:
            self._total_errors += 1
        self._execution_times.append(execution_time_ms)
        self._confidence_history.append(result_confidence)

        # Update hot paths
        self._update_hot_paths(op_sequence, execution_time_ms, result_confidence, lang_tags)

        # Update NL patterns from raw forms
        for raw in raw_forms:
            self._update_nl_patterns(raw, op_sequence, result_confidence, lang_tags)

    def hot_path(self, min_heat: float = 0.1) -> list[HotPath]:
        """Identify frequently-executed NL patterns, sorted by heat score.

        The heat score combines:
          - Frequency (how often the pattern appears)
          - Recency (recent patterns get a boost, 1-hour half-life)
          - Consistency (low variance in confidence scores = high consistency)
        """
        sorted_paths = sorted(
            [p for p in self._hot_paths.values() if p.heat >= min_heat],
            key=lambda p: p.heat,
            reverse=True,
        )
        return sorted_paths

    def specialize(self, pattern: NLPattern) -> CompiledPattern:
        """Specialize the interpreter for a hot pattern.

        This is Futamura Projection 1 in action: given a program pattern
        (partially known input), produce optimized bytecode that skips the
        generic interpretation path.

        The specialization applies these transformations:
          1. Inline the resolved op sequence (skip dispatch)
          2. Type-specialize variables if types are stable
          3. Constant-fold any known values
          4. Emit a fast-path bytecode that matches the pattern fingerprint
        """
        bytecode: list[list[Any]] = []

        # Emit pattern match header
        bytecode.append(["PATTERN_MATCH", pattern.fingerprint])
        bytecode.append(["LANG_TAG", pattern.lang])

        # Emit specialized op sequence
        for op in pattern.resolved_ops:
            bytecode.append([op.upper()])

        # Emit specialized variable types if we have type info
        # (This is type specialization — one of the most powerful JIT opts)
        # In the full implementation, we'd emit type-guarded fast paths

        # Emit pattern completion
        bytecode.append(["PATTERN_END", pattern.fingerprint])

        # Estimate speedup: inlined path vs generic dispatch
        speedup = 1.0 + len(pattern.resolved_ops) * 0.15  # ~15% per inlined op
        if pattern.is_idiomatic:
            speedup *= 1.5  # Idiomatic patterns benefit more

        compiled = CompiledPattern(
            pattern_fingerprint=pattern.fingerprint,
            bytecode=bytecode,
            optimization_kind=OptimizationKind.HOT_PATH_INLINE.value,
            speedup_estimate=speedup,
        )
        self._compiled_patterns[pattern.fingerprint] = compiled
        return compiled

    def suggest_optimization(self) -> list[Optimization]:
        """Suggest optimizations based on observed patterns.

        This is the engine's "intelligence" — it doesn't just learn
        fast paths, it reasons about what KINDS of optimization to apply:

        1. Hot Path Inline: Frequent sequences → inline them
        2. Dead Pattern Elim: Ops that are never used → remove from grammar
        3. Type Specialize: Stable variable types → monomorphize
        4. Constant Propagate: Variables that always have the same value → fold
        5. Branch Fusion: Sequential branches with same merge → combine
        6. Macro Expansion: Repeated sub-trees → extract as macros
        7. Vocab Fast Path: Common NL terms → pre-resolve
        """
        optimizations: list[Optimization] = []

        # 1. Hot path inlining
        for hp in self.hot_path(min_heat=0.3):
            if hp.frequency >= self._hot_threshold:
                opt = Optimization(
                    kind=OptimizationKind.HOT_PATH_INLINE.value,
                    target_pattern="→".join(hp.sequence),
                    description=(
                        f"Inline hot path '{' → '.join(hp.sequence)}' "
                        f"(freq={hp.frequency}, heat={hp.heat:.2f})"
                    ),
                    estimated_speedup=1.0 + len(hp.sequence) * 0.15,
                    confidence=min(1.0, hp.heat + 0.1),
                    applies_to=list(hp.sequence),
                    rationale=(
                        f"This op sequence has been executed {hp.total_executions} times "
                        f"with avg confidence {sum(hp.confidence_scores) / max(1, len(hp.confidence_scores)):.2f}. "
                        f"Inlining eliminates dispatch overhead for each op."
                    ),
                )
                optimizations.append(opt)

        # 2. Dead pattern elimination
        if self._total_programs >= 50:
            used_ops = self._ops_used
            for op in self._all_ops_known:
                if op not in used_ops:
                    opt = Optimization(
                        kind=OptimizationKind.DEAD_PATTERN_ELIM.value,
                        target_pattern=op,
                        description=f"Opcode '{op}' never observed — candidate for removal",
                        estimated_speedup=1.01,  # Marginal but reduces interpreter size
                        confidence=0.7,
                        applies_to=[],
                        rationale=f"Across {self._total_programs} programs, '{op}' was never used.",
                    )
                    optimizations.append(opt)

        # 3. Type specialization
        var_type_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for obs in self._observations:
            for var, vtype in obs.variable_types.items():
                var_type_counter[var][vtype] += 1

        for var, type_counts in var_type_counter.items():
            total = sum(type_counts.values())
            if total >= self._hot_threshold:
                dominant_type = max(type_counts, key=type_counts.get)
                dominance = type_counts[dominant_type] / total
                if dominance >= 0.95:
                    opt = Optimization(
                        kind=OptimizationKind.TYPE_SPECIALIZE.value,
                        target_pattern=f"var:{var}",
                        description=(
                            f"Variable '{var}' is always '{dominant_type}' "
                            f"({dominance:.0%} of {total} uses)"
                        ),
                        estimated_speedup=2.0,  # Type specialization is very impactful
                        confidence=dominance,
                        applies_to=[],
                        rationale=(
                            f"Type-specializing '{var}' to '{dominant_type}' eliminates "
                            f"dynamic type checks on every access."
                        ),
                    )
                    optimizations.append(opt)

        # 4. Constant propagation
        var_value_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for obs in self._observations:
            # We don't track values directly in observations, but we can
            # identify variables whose types never change
            for var in obs.variable_types:
                if obs.variable_types[var] in ("int", "float", "bool", "str"):
                    var_value_counter[var][obs.variable_types[var]] += 1

        # 5. Vocabulary fast paths
        for fp, pattern in self._nl_patterns.items():
            if pattern.frequency >= self._hot_threshold and pattern.is_idiomatic:
                opt = Optimization(
                    kind=OptimizationKind.VOCAB_FAST_PATH.value,
                    target_pattern=pattern.fingerprint,
                    description=(
                        f"Fast-path for '{pattern.raw_forms[0] if pattern.raw_forms else fp}' "
                        f"(idiomatic {pattern.lang}, freq={pattern.frequency})"
                    ),
                    estimated_speedup=3.0,  # Skipping NL→op resolution is very fast
                    confidence=min(1.0, pattern.frequency / 100.0),
                    applies_to=pattern.resolved_ops,
                    rationale=(
                        f"This NL form maps deterministically to {pattern.resolved_ops}. "
                        f"Pre-resolving eliminates the ambiguity resolution step."
                    ),
                )
                optimizations.append(opt)

        # 6. Branch fusion: detect branch→synthesize or branch→co_iterate patterns
        branch_synthesize_count = 0
        branch_coiter_count = 0
        branch_reflect_count = 0
        for hp in self._hot_paths.values():
            seq = hp.sequence
            if "branch" in seq and "synthesize" in seq:
                branch_synthesize_count += hp.total_executions
            if "branch" in seq and "co_iterate" in seq:
                branch_coiter_count += hp.total_executions
            if "branch" in seq and "reflect" in seq:
                branch_reflect_count += hp.total_executions

        if branch_synthesize_count >= self._hot_threshold:
            optimizations.append(Optimization(
                kind=OptimizationKind.BRANCH_FUSION.value,
                target_pattern="branch→synthesize",
                description="Fuse branch+synthesize into a single parallel-reduce op",
                estimated_speedup=1.5,
                confidence=0.8,
                applies_to=["branch", "synthesize"],
                rationale=(
                    f"branch→synthesize executed {branch_synthesize_count} times. "
                    f"Fusing into a single 'parallel_reduce' op eliminates intermediate results."
                ),
            ))

        return sorted(optimizations, key=lambda o: o.confidence * o.estimated_speedup, reverse=True)

    # ------------------------------------------------------------------
    # Level 2: Grammar Evolution
    # ------------------------------------------------------------------

    def evolve_grammar(self, min_frequency: int = 5) -> GrammarDelta:
        """Propose grammar extensions based on observed usage patterns.

        The grammar evolves through three mechanisms:
          1. Macro extraction: Repeated sub-trees become named macros.
          2. Syntax sugar: Common patterns get shorter syntax.
          3. Idiom adoption: Language-specific constructions become first-class.

        This is analogous to biological evolution:
          - Mutation: A user writes something novel
          - Selection: If it's useful, it survives (gets reused)
          - Reproduction: If it spreads, it becomes part of the species (grammar)
        """
        best_delta: Optional[GrammarDelta] = None
        best_score = 0.0

        # Scan NL patterns for grammar extension candidates
        for fp, pattern in self._nl_patterns.items():
            if pattern.frequency < min_frequency:
                continue

            # Score: frequency * confidence * idiomaticity bonus
            score = pattern.frequency * pattern.confidence_avg
            if pattern.is_idiomatic:
                score *= 2.0
            if len(pattern.resolved_ops) > 2:
                score *= 1.5  # Multi-op patterns are more valuable as grammar extensions

            if score > best_score:
                best_score = score
                change_type = "idiom" if pattern.is_idiomatic else "macro"
                if len(pattern.resolved_ops) == 1:
                    change_type = "syntax_sugar"

                example = pattern.raw_forms[0] if pattern.raw_forms else fp

                delta = GrammarDelta(
                    change_type=change_type,
                    name=f"auto_{pattern.lang}_{fp[:12]}",
                    pattern=example,
                    expansion=[{"op": op} for op in pattern.resolved_ops],
                    confidence=min(1.0, score / 100.0),
                    frequency=pattern.frequency,
                    languages=[pattern.lang] if pattern.lang else [],
                    example_programs=pattern.raw_forms[:3],
                    rationale=(
                        f"Pattern '{example}' observed {pattern.frequency} times "
                        f"with avg confidence {pattern.confidence_avg:.2f}. "
                        f"Proposed as {change_type} for {pattern.lang or 'all languages'}."
                    ),
                )
                best_delta = delta

        if best_delta is not None:
            self._grammar_deltas.append(best_delta)

        return best_delta or GrammarDelta(
            change_type="none",
            name="",
            pattern="",
            confidence=0.0,
            rationale="No patterns meet the frequency threshold for grammar evolution.",
        )

    # ------------------------------------------------------------------
    # Level 3: Paradigm Emergence
    # ------------------------------------------------------------------

    def detect_paradigm_shifts(self) -> list[dict[str, Any]]:
        """Detect emerging programming paradigms from usage patterns.

        A paradigm is a cluster of related patterns that co-occur frequently.
        For example:
          - "branch → co_iterate → synthesize" = Collaborative Parallelism
          - "reflect → fork → merge" = Speculative Self-Improvement
          - "discuss → branch → synthesize" = Deliberative Exploration

        This is the highest level of evolution: not just optimizing individual
        patterns, but recognizing that patterns form *paradigms* — coherent
        ways of thinking about computation.
        """
        # Count 2-grams and 3-grams of ops across all observations
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()

        for obs in self._observations:
            ops = obs.op_sequence
            for i in range(len(ops) - 1):
                bigram_counts[(ops[i], ops[i + 1])] += 1
            for i in range(len(ops) - 2):
                trigram_counts[(ops[i], ops[i + 1], ops[i + 2])] += 1

        # Known paradigm signatures
        paradigm_signatures = {
            "collaborative_parallelism": {
                "required": {"branch", "co_iterate", "synthesize"},
                "description": "Multi-agent parallel exploration with consensus",
            },
            "speculative_improvement": {
                "required": {"reflect", "fork", "merge"},
                "description": "Self-evaluation followed by speculative execution",
            },
            "deliberative_exploration": {
                "required": {"discuss", "branch", "synthesize"},
                "description": "Agent discussion before branching into exploration",
            },
            "adaptive_delegation": {
                "required": {"reflect", "delegate", "trust"},
                "description": "Self-aware task delegation with trust management",
            },
            "signal_reactive": {
                "required": {"signal", "await", "if"},
                "description": "Event-driven reactive programming",
            },
        }

        shifts: list[dict[str, Any]] = []

        for name, sig in paradigm_signatures.items():
            required = sig["required"]
            # Check if all required ops appear together in any trigram
            # (This is a simplified check — full implementation would use
            # co-occurrence analysis across programs)
            co_occurrence_count = 0
            for obs in self._observations:
                obs_ops = set(obs.op_sequence)
                if required.issubset(obs_ops):
                    co_occurrence_count += 1

            if co_occurrence_count >= self._hot_threshold // 2:
                # Find the strongest trigram
                best_trigram = None
                best_count = 0
                for trigram, count in trigram_counts.most_common(20):
                    if required.issubset(set(trigram)):
                        best_trigram = trigram
                        best_count = count
                        break

                shifts.append({
                    "paradigm": name,
                    "description": sig["description"],
                    "co_occurrence_count": co_occurrence_count,
                    "signature_trigram": list(best_trigram) if best_trigram else None,
                    "trigram_count": best_count,
                    "emergence_confidence": min(1.0, co_occurrence_count / self._hot_threshold),
                    "required_ops": list(required),
                })

        return sorted(shifts, key=lambda s: s["emergence_confidence"], reverse=True)

    # ------------------------------------------------------------------
    # Fitness Measurement
    # ------------------------------------------------------------------

    def measure_fitness(self) -> FitnessMetrics:
        """Measure how well the current grammar serves users.

        Fitness is a multi-dimensional score:
          - successful_executions: fraction of programs that ran without errors
          - avg_confidence: average result confidence
          - error_rate: fraction of programs that errored
          - pattern_coverage: fraction of hot paths that have compiled fast-paths
          - grammar_utilization: fraction of grammar productions actually used
          - lang_diversity: how many languages are actively used
        """
        if not self._total_programs:
            return FitnessMetrics()

        total = self._total_programs
        success_rate = (total - self._total_errors) / total
        avg_conf = sum(self._confidence_history) / len(self._confidence_history) if self._confidence_history else 1.0
        avg_time = sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0.0
        error_rate = self._total_errors / total

        # Pattern coverage: hot paths with compiled patterns / total hot paths
        hot_paths = self.hot_path(min_heat=0.1)
        compiled_count = sum(
            1 for hp in hot_paths
            if self._hash_ops(list(hp.sequence)) in self._compiled_patterns
        )
        pattern_coverage = compiled_count / max(1, len(hot_paths))

        # Grammar utilization: ops used / ops known
        grammar_util = len(self._ops_used) / len(self._all_ops_known)

        # Language diversity (Shannon entropy of language distribution)
        lang_counts = list(self._langs_used.values())
        lang_diversity = 0.0
        if lang_counts:
            total_lang = sum(lang_counts)
            if total_lang > 0:
                for count in lang_counts:
                    p = count / total_lang
                    if p > 0:
                        lang_diversity -= p * math.log2(p)
                # Normalize by max possible entropy (assuming 10 languages)
                lang_diversity = min(1.0, lang_diversity / math.log2(10))

        # Paradigm fitness (from detected shifts)
        paradigm_shifts = self.detect_paradigm_shifts()
        paradigm_fitness = {
            s["paradigm"]: s["emergence_confidence"]
            for s in paradigm_shifts
        }

        return FitnessMetrics(
            total_programs=total,
            successful_executions=success_rate,
            avg_confidence=avg_conf,
            avg_execution_time_ms=avg_time,
            error_rate=error_rate,
            pattern_coverage=pattern_coverage,
            grammar_utilization=grammar_util,
            lang_diversity=lang_diversity,
            evolution_generation=self._generation,
            paradigm_fitness=paradigm_fitness,
        )

    def evolve_generation(self) -> dict[str, Any]:
        """Run one generation of evolution: learn, optimize, propose grammar changes.

        Returns a summary of what happened this generation.
        """
        self._generation += 1

        # Measure current fitness
        fitness = self.measure_fitness()

        # Get hot paths
        hot_paths = self.hot_path()
        new_compiled = 0

        # Auto-specialize hot paths
        for hp in hot_paths:
            fp = self._hash_ops(list(hp.sequence))
            if fp not in self._compiled_patterns and hp.frequency >= self._hot_threshold:
                pattern = NLPattern(
                    fingerprint=fp,
                    resolved_ops=list(hp.sequence),
                    frequency=hp.frequency,
                    confidence_avg=(
                        sum(hp.confidence_scores) / len(hp.confidence_scores)
                        if hp.confidence_scores else 1.0
                    ),
                    lang=hp.lang_affinity,
                )
                self.specialize(pattern)
                new_compiled += 1

        # Suggest optimizations
        optimizations = self.suggest_optimization()

        # Propose grammar evolution
        grammar_delta = self.evolve_grammar()

        # Detect paradigm shifts
        paradigm_shifts = self.detect_paradigm_shifts()

        return {
            "generation": self._generation,
            "fitness": fitness.to_dict(),
            "hot_paths_count": len(hot_paths),
            "new_compiled_patterns": new_compiled,
            "total_compiled_patterns": len(self._compiled_patterns),
            "optimizations_suggested": len(optimizations),
            "grammar_delta": grammar_delta.to_dict(),
            "paradigm_shifts": paradigm_shifts,
            "top_optimizations": [o.to_dict() for o in optimizations[:5]],
        }

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _extract_expressions(self, program: Any) -> list[dict[str, Any]]:
        """Extract flat list of expression dicts from a program."""
        expressions: list[dict[str, Any]] = []

        if isinstance(program, dict):
            # Check if it's a Signal program wrapper
            inner = program.get("signal", program)
            if "body" in inner:
                for expr in inner["body"]:
                    if isinstance(expr, dict):
                        expressions.append(expr)
                        # Recurse into nested structures
                        if "branches" in expr:
                            for branch in expr.get("branches", []):
                                if isinstance(branch, dict):
                                    expressions.extend(
                                        e for e in branch.get("body", [])
                                        if isinstance(e, dict)
                                    )
                        if "body" in expr:
                            expressions.extend(
                                e for e in expr.get("body", [])
                                if isinstance(e, dict)
                            )
            elif "op" in inner:
                expressions.append(inner)
        elif isinstance(program, list):
            for expr in program:
                if isinstance(expr, dict):
                    expressions.append(expr)

        return expressions

    def _hash_ops(self, ops: list[str]) -> str:
        """Create a stable fingerprint for an op sequence."""
        return hashlib.sha256("→".join(ops).encode()).hexdigest()[:16]

    def _update_hot_paths(
        self,
        op_sequence: list[str],
        exec_time: float,
        confidence: float,
        lang_tags: list[str],
    ) -> None:
        """Update hot path tracking with a new observation."""
        # Track sequences of length 2-5
        for length in range(2, min(6, len(op_sequence) + 1)):
            for i in range(len(op_sequence) - length + 1):
                seq = tuple(op_sequence[i:i + length])
                if seq not in self._hot_paths:
                    self._hot_paths[seq] = HotPath(sequence=seq)
                hp = self._hot_paths[seq]
                hp.frequency += 1
                hp.total_executions += 1
                hp.avg_execution_time_ms = (
                    (hp.avg_execution_time_ms * (hp.total_executions - 1) + exec_time)
                    / hp.total_executions
                )
                hp.confidence_scores.append(confidence)
                hp.last_seen = time.time()
                if lang_tags:
                    hp.lang_affinity = lang_tags[0]  # Dominant language

    def _update_nl_patterns(
        self,
        raw_form: str,
        op_sequence: list[str],
        confidence: float,
        lang_tags: list[str],
    ) -> None:
        """Update NL pattern tracking with a new raw form."""
        if not raw_form or len(raw_form) < 2:
            return

        fp = hashlib.sha256(raw_form.encode()).hexdigest()[:16]
        if fp not in self._nl_patterns:
            self._nl_patterns[fp] = NLPattern(
                fingerprint=fp,
                raw_forms=[raw_form],
                resolved_ops=list(op_sequence),
                lang=lang_tags[0] if lang_tags else "",
            )
        else:
            pattern = self._nl_patterns[fp]
            if raw_form not in pattern.raw_forms:
                pattern.raw_forms.append(raw_form)
            pattern.frequency += 1
            # Running average of confidence
            pattern.confidence_avg = (
                (pattern.confidence_avg * (pattern.frequency - 1) + confidence)
                / pattern.frequency
            )
            if lang_tags and not pattern.lang:
                pattern.lang = lang_tags[0]

        # Detect idiomatic patterns: NL forms that appear frequently
        # with the same op resolution and are language-specific
        pattern = self._nl_patterns[fp]
        pattern.is_idiomatic = (
            bool(pattern.lang)
            and pattern.lang != "eng"
            and pattern.frequency >= 3
            and len(pattern.raw_forms) >= 2
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full evolution engine state."""
        return {
            "generation": self._generation,
            "total_programs": self._total_programs,
            "total_errors": self._total_errors,
            "observation_count": len(self._observations),
            "hot_paths": {self._hash_ops(list(k)): v.to_dict() for k, v in self._hot_paths.items()},
            "nl_patterns": {k: v.to_dict() for k, v in self._nl_patterns.items()},
            "compiled_patterns": {k: v.to_dict() for k, v in self._compiled_patterns.items()},
            "grammar_deltas": [d.to_dict() for d in self._grammar_deltas],
            "adopted_grammar": [d.to_dict() for d in self._adopted_grammar],
            "fitness": self.measure_fitness().to_dict(),
            "ops_used": sorted(self._ops_used),
            "langs_used": dict(self._langs_used),
        }
