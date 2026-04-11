"""
Ambiguous Parse System — Confident Ambiguity for Natural Language Compilation.

This module implements FLUX's core innovation: when natural language input is
ambiguous (has multiple valid interpretations), we do NOT pick the "best" parse
and hope for the best. Instead:

  1. ALL valid interpretations are compiled simultaneously
  2. Each interpretation carries a confidence score
  3. All interpretations execute in parallel
  4. Runtime evidence is accumulated to converge on the correct result

This is "confident ambiguity" — ambiguity is not an error condition to resolve
at compile time, but a first-class runtime value that resolves through evidence.

Classes:
  - Interpretation: One valid parse of an ambiguous input
  - AmbiguousParse: A set of weighted interpretations
  - ConfidencePropagation: Accumulates evidence across interpretations
  - BranchingExecutor: Executes all interpretations in parallel, merges results
"""

from __future__ import annotations

import uuid
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Interpretation — one valid reading of ambiguous input
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Interpretation:
    """One valid parse/reading of an ambiguous natural language input.

    Each interpretation has:
      - label: a human-readable name for this reading
      - weight: initial prior probability (not confidence — this is the prior)
      - confidence: how much we trust this interpretation (accumulated evidence)
      - bytecode: the compiled bytecode for this interpretation
      - metadata: arbitrary metadata about the interpretation
    """

    label: str
    weight: float = 1.0
    confidence: float = 0.5
    bytecode: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(1.0, float(self.weight)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def add_evidence(self, evidence_strength: float) -> None:
        """Bayesian-like evidence accumulation.

        Uses log-odds transformation to combine evidence:
          - Positive evidence (>0.5) increases confidence
          - Negative evidence (<0.5) decreases confidence
          - Neutral evidence (0.5) has no effect

        The formula avoids the "certainty trap" where confidence
        asymptotically approaches but never reaches 1.0.

        Args:
            evidence_strength: A value in [0.0, 1.0] representing the
                               strength of confirming/disconfirming evidence.
        """
        evidence = max(0.0, min(1.0, float(evidence_strength)))

        # Transform to log-odds space
        eps = 1e-10  # avoid log(0)
        current_odds = self.confidence / (1.0 - self.confidence + eps)
        evidence_odds = evidence / (1.0 - evidence + eps)

        # Combine: new odds = old odds * evidence odds
        new_odds = current_odds * evidence_odds

        # Transform back
        self.confidence = new_odds / (1.0 + new_odds)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def is_converged(self, threshold: float = 0.95) -> bool:
        """Check if this interpretation has converged (high confidence)."""
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "weight": self.weight,
            "confidence": round(self.confidence, 6),
            "bytecode_length": len(self.bytecode),
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AmbiguousParse — a set of weighted interpretations
# ═══════════════════════════════════════════════════════════════════════════

class AmbiguityStatus(str, Enum):
    """Status of an ambiguous parse."""
    PENDING = "pending"           # Not yet resolved
    EXECUTING = "executing"       # Running all interpretations
    CONVERGED = "converged"       # One interpretation has won
    DISPUTED = "disputed"         # Multiple interpretations remain viable
    EXHAUSTED = "exhausted"       # All evidence consumed, no convergence
    RESOLVED_BY_AGENT = "resolved_by_agent"  # External agent resolved it


@dataclass
class AmbiguousParse:
    """Represents an ambiguous natural language input with multiple valid
    interpretations.

    This is the core data structure for FLUX's "confident ambiguity" approach.
    Instead of resolving ambiguity at compile time, we carry all interpretations
    forward into execution and let runtime evidence determine the winner.

    Example:
        # "加三于四" in Classical Chinese could mean:
        #   - Math context: 3 + 4 = 7
        #   - Confucian context: distribute 3 among 4
        parse = AmbiguousParse(
            source="加三于四",
            lang="wen",
            interpretations=[
                Interpretation(label="math", weight=0.7, confidence=0.5,
                              bytecode=[("MOVI", 0, 3), ("MOVI", 1, 4), ("IADD", 0, 0, 1)]),
                Interpretation(label="confucian", weight=0.3, confidence=0.5,
                              bytecode=[("MOVI", 0, 3), ("MOVI", 1, 4), ("DISTRIBUTE", 0, 1)]),
            ],
        )
    """

    source: str
    lang: str = "flux"
    interpretations: list[Interpretation] = field(default_factory=list)
    status: AmbiguityStatus = AmbiguityStatus.PENDING
    id: str = ""
    created_at: str = ""
    resolved_at: str = ""
    resolved_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def from_binary(
        cls,
        source: str,
        lang: str,
        label_a: str,
        bytecode_a: list[Any],
        label_b: str,
        bytecode_b: list[Any],
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ) -> AmbiguousParse:
        """Create an ambiguous parse with exactly two interpretations.

        Args:
            source: The original ambiguous natural language input.
            lang: Language tag.
            label_a: Label for interpretation A.
            bytecode_a: Bytecode for interpretation A.
            label_b: Label for interpretation B.
            bytecode_b: Bytecode for interpretation B.
            weight_a: Prior weight for A (default 0.5).
            weight_b: Prior weight for B (default 0.5).

        Returns:
            An AmbiguousParse with two interpretations.
        """
        return cls(
            source=source,
            lang=lang,
            interpretations=[
                Interpretation(label=label_a, weight=weight_a, bytecode=bytecode_a),
                Interpretation(label=label_b, weight=weight_b, bytecode=bytecode_b),
            ],
        )

    @classmethod
    def from_options(
        cls,
        source: str,
        lang: str,
        options: dict[str, tuple[list[Any], float]],
    ) -> AmbiguousParse:
        """Create an ambiguous parse from a dict of label → (bytecode, weight).

        Args:
            source: The original ambiguous natural language input.
            lang: Language tag.
            options: Dict mapping label → (bytecode, weight).

        Returns:
            An AmbiguousParse with one interpretation per option.
        """
        interpretations = []
        for label, (bytecode, weight) in options.items():
            interpretations.append(
                Interpretation(label=label, weight=weight, bytecode=bytecode)
            )
        return cls(source=source, lang=lang, interpretations=interpretations)

    # ── Query ───────────────────────────────────────────────────────────

    @property
    def interpretation_count(self) -> int:
        """Number of valid interpretations."""
        return len(self.interpretations)

    @property
    def is_unambiguous(self) -> bool:
        """True if there is only one interpretation (no ambiguity)."""
        return len(self.interpretations) <= 1

    @property
    def is_resolved(self) -> bool:
        """True if ambiguity has been resolved."""
        return self.status in (
            AmbiguityStatus.CONVERGED,
            AmbiguityStatus.RESOLVED_BY_AGENT,
        )

    def best_interpretation(self) -> Optional[Interpretation]:
        """Return the interpretation with the highest confidence.

        Returns None if there are no interpretations.
        """
        if not self.interpretations:
            return None
        return max(self.interpretations, key=lambda i: i.confidence)

    def viable_interpretations(self, threshold: float = 0.3) -> list[Interpretation]:
        """Return interpretations above the viability threshold.

        Args:
            threshold: Minimum confidence to be considered viable.

        Returns:
            List of viable interpretations, sorted by confidence descending.
        """
        viable = [i for i in self.interpretations if i.confidence >= threshold]
        return sorted(viable, key=lambda i: i.confidence, reverse=True)

    def confidence_spread(self) -> float:
        """The gap between the best and second-best confidence scores.

        A high spread (>0.3) indicates clear winner. A low spread (<0.1)
        indicates the interpretations are nearly equally likely.
        """
        if len(self.interpretations) < 2:
            return 1.0
        sorted_conf = sorted(
            [i.confidence for i in self.interpretations], reverse=True
        )
        return sorted_conf[0] - sorted_conf[1]

    # ── Mutation ───────────────────────────────────────────────────────

    def add_evidence(self, evidence: dict[str, float]) -> None:
        """Add evidence to interpretations.

        Args:
            evidence: Dict mapping interpretation label → evidence strength [0.0, 1.0].
                      Evidence > 0.5 confirms the interpretation.
                      Evidence < 0.5 disconfirms it.
        """
        for interp in self.interpretations:
            if interp.label in evidence:
                interp.add_evidence(evidence[interp.label])

    def mark_resolved(self, label: str, method: str = "converged") -> None:
        """Mark this parse as resolved.

        Args:
            label: The winning interpretation's label.
            method: How it was resolved ("converged" or "agent").
        """
        self.resolved_label = label
        self.resolved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if method == "agent":
            self.status = AmbiguityStatus.RESOLVED_BY_AGENT
        else:
            self.status = AmbiguityStatus.CONVERGED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "lang": self.lang,
            "status": self.status.value,
            "interpretation_count": self.interpretation_count,
            "is_unambiguous": self.is_unambiguous,
            "interpretations": [i.to_dict() for i in self.interpretations],
            "best": self.best_interpretation().to_dict() if self.best_interpretation() else None,
            "confidence_spread": round(self.confidence_spread(), 6),
            "resolved_label": self.resolved_label,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ConfidencePropagation — runtime confidence accumulation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceRecord:
    """A single piece of evidence for or against an interpretation."""
    interpretation_label: str
    evidence_strength: float
    source: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ConfidencePropagation:
    """Runtime confidence accumulation engine.

    Tracks evidence for all interpretations of an ambiguous parse and
    determines when convergence has been achieved.

    The propagation model:
      1. Start with prior weights as initial confidence
      2. Accumulate evidence from:
         - Execution results (did the interpretation produce valid output?)
         - Type checking (did the types match expected signatures?)
         - Runtime context (does the domain context support this reading?)
         - External agents (did a human/agent resolve the ambiguity?)
      3. Check convergence criteria:
         - Best interpretation has confidence > 0.95
         - Confidence spread > 0.3 (clear winner)
         - Maximum evidence rounds exhausted

    Usage:
        cp = ConfidencePropagation(ambiguous_parse)
        cp.add_execution_result("math", valid=True, output=7)
        cp.add_execution_result("confucian", valid=False, error="type mismatch")
        cp.add_context_evidence("math", strength=0.9)
        if cp.is_converged():
            winner = cp.winner()
    """

    def __init__(
        self,
        parse: AmbiguousParse,
        convergence_threshold: float = 0.95,
        spread_threshold: float = 0.3,
        max_rounds: int = 10,
    ) -> None:
        self.parse = parse
        self.convergence_threshold = convergence_threshold
        self.spread_threshold = spread_threshold
        self.max_rounds = max_rounds
        self._evidence_log: list[EvidenceRecord] = []
        self._round: int = 0

    @property
    def round(self) -> int:
        """Current evidence accumulation round."""
        return self._round

    @property
    def evidence_log(self) -> list[EvidenceRecord]:
        """Full history of all evidence added."""
        return list(self._evidence_log)

    # ── Evidence addition ───────────────────────────────────────────────

    def add_execution_result(
        self,
        label: str,
        valid: bool,
        output: Any = None,
        error: str = "",
    ) -> None:
        """Add evidence from executing one interpretation.

        Args:
            label: The interpretation's label.
            valid: Whether the execution produced valid output.
            output: The execution output (used for value-based evidence).
            error: Error message if execution failed.
        """
        if valid:
            strength = 0.85  # Strong confirming evidence
            source = "execution:valid"
        else:
            strength = 0.15  # Strong disconfirming evidence
            source = f"execution:error:{error[:50]}"

        self._add_evidence(label, strength, source)

    def add_context_evidence(
        self,
        label: str,
        strength: float,
        context_description: str = "",
    ) -> None:
        """Add evidence from the surrounding context.

        Context evidence is weaker than execution evidence but can
        tip the balance when execution results are ambiguous.

        Args:
            label: The interpretation's label.
            strength: Evidence strength [0.0, 1.0].
            context_description: Description of the contextual evidence.
        """
        source = f"context:{context_description}" if context_description else "context"
        # Dampen context evidence (it's less reliable than execution)
        dampened = 0.5 + 0.5 * (strength - 0.5)  # Pull toward 0.5
        self._add_evidence(label, dampened, source)

    def add_type_evidence(
        self,
        label: str,
        types_match: bool,
        expected_type: str = "",
        actual_type: str = "",
    ) -> None:
        """Add evidence from type checking.

        Args:
            label: The interpretation's label.
            types_match: Whether the types matched expectations.
            expected_type: The expected type.
            actual_type: The actual type found.
        """
        if types_match:
            strength = 0.80
            source = f"type:match:{expected_type}"
        else:
            strength = 0.20
            source = f"type:mismatch:{expected_type}!={actual_type}"
        self._add_evidence(label, strength, source)

    def add_agent_resolution(self, label: str, agent_id: str = "human") -> None:
        """An external agent (or human) resolves the ambiguity.

        This immediately marks the parse as resolved with full confidence.

        Args:
            label: The winning interpretation's label.
            agent_id: The agent or human who resolved it.
        """
        # Set the winning interpretation to near-certainty
        for interp in self.parse.interpretations:
            if interp.label == label:
                interp.confidence = 0.99
            else:
                interp.confidence = 0.01

        self._evidence_log.append(
            EvidenceRecord(
                interpretation_label=label,
                evidence_strength=0.99,
                source=f"agent:{agent_id}:resolution",
            )
        )
        self.parse.mark_resolved(label, method="agent")

    def _add_evidence(
        self,
        label: str,
        strength: float,
        source: str,
    ) -> None:
        """Internal: add evidence and propagate to interpretation."""
        record = EvidenceRecord(
            interpretation_label=label,
            evidence_strength=strength,
            source=source,
        )
        self._evidence_log.append(record)

        for interp in self.parse.interpretations:
            if interp.label == label:
                interp.add_evidence(strength)
            else:
                # Counter-evidence: if label A gets confirming evidence,
                # label B gets mild disconfirming evidence
                interp.add_evidence(1.0 - strength * 0.3)  # Mild dampening

    def next_round(self) -> int:
        """Advance to the next evidence accumulation round.

        Returns:
            The new round number.
        """
        self._round += 1
        return self._round

    # ── Convergence checking ────────────────────────────────────────────

    def is_converged(self) -> bool:
        """Check if the interpretations have converged to a clear winner.

        Convergence criteria (ALL must be met):
          1. Best interpretation confidence > convergence_threshold
          2. Confidence spread > spread_threshold

        Returns:
            True if converged.
        """
        if self.parse.is_resolved:
            return True

        best = self.parse.best_interpretation()
        if best is None:
            return False

        return (
            best.confidence >= self.convergence_threshold
            and self.parse.confidence_spread() >= self.spread_threshold
        )

    def is_exhausted(self) -> bool:
        """Check if maximum evidence rounds have been consumed.

        Returns:
            True if no more rounds are available.
        """
        return self._round >= self.max_rounds

    def winner(self) -> Optional[Interpretation]:
        """Return the winning interpretation, or None if not converged.

        Returns:
            The best interpretation if converged, None otherwise.
        """
        if self.is_converged() or self.is_exhausted():
            best = self.parse.best_interpretation()
            if best and not self.parse.is_resolved:
                self.parse.mark_resolved(best.label, method="converged")
            return best
        return None

    def summary(self) -> dict[str, Any]:
        """Return a summary of the propagation state."""
        best = self.parse.best_interpretation()
        return {
            "parse_id": self.parse.id,
            "round": self._round,
            "max_rounds": self.max_rounds,
            "status": self.parse.status.value,
            "converged": self.is_converged(),
            "exhausted": self.is_exhausted(),
            "best_label": best.label if best else None,
            "best_confidence": round(best.confidence, 6) if best else None,
            "confidence_spread": round(self.parse.confidence_spread(), 6),
            "evidence_count": len(self._evidence_log),
            "interpretations": [
                {"label": i.label, "confidence": round(i.confidence, 6)}
                for i in self.parse.interpretations
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# BranchingExecutor — execute all interpretations in parallel
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    """The result of executing one interpretation."""
    label: str
    success: bool
    value: Any = None
    error: str = ""
    confidence: float = 1.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionBackend:
    """Abstract execution backend — executes bytecode for one interpretation.

    Subclass this to provide actual bytecode execution. The default
    implementation is a simple interpreter that handles a subset of
    FLUX opcodes.
    """

    def execute(self, bytecode: list[Any]) -> ExecutionResult:
        """Execute bytecode and return the result.

        Args:
            bytecode: A list of instruction tuples, e.g. [("MOVI", 0, 42), ("ADD", 0, 0, 1)].

        Returns:
            An ExecutionResult with the outcome.
        """
        raise NotImplementedError


class SimpleBackend(ExecutionBackend):
    """A minimal bytecode executor for demonstration and testing.

    Supports a subset of FLUX opcodes:
      - ("MOVI", reg, imm)       : Set register to immediate value
      - ("MOV", rd, rs)          : Copy register
      - ("ADD", rd, rs1, rs2)    : Add rs1 + rs2 → rd
      - ("SUB", rd, rs1, rs2)    : Subtract rs2 from rs1 → rd
      - ("MUL", rd, rs1, rs2)    : Multiply rs1 * rs2 → rd
      - ("DIV", rd, rs1, rs2)    : Integer divide rs1 / rs2 → rd
      - ("HALT",)                 : Stop execution
      - ("PRINT", reg)            : Print register value
      - ("NOP",)                  : No operation
      - ("IADD", a, b)            : Add two immediates → R0
      - ("DISTRIBUTE", a, b)      : Divide a among b → R0 (fractional)
    """

    def __init__(self, max_cycles: int = 10000) -> None:
        self.max_cycles = max_cycles

    def execute(self, bytecode: list[Any]) -> ExecutionResult:
        registers = [0] * 8  # R0-R7
        output: list[str] = []
        cycles = 0
        error = ""

        try:
            for instr in bytecode:
                cycles += 1
                if cycles > self.max_cycles:
                    error = "Max cycles exceeded"
                    break

                if not isinstance(instr, (list, tuple)) or len(instr) == 0:
                    continue

                op = str(instr[0]).upper()

                if op == "MOVI" and len(instr) >= 3:
                    reg = int(instr[1])
                    imm = instr[2]
                    if 0 <= reg < 8:
                        registers[reg] = imm

                elif op == "MOV" and len(instr) >= 3:
                    rd = int(instr[1])
                    rs = int(instr[2])
                    if 0 <= rd < 8 and 0 <= rs < 8:
                        registers[rd] = registers[rs]

                elif op == "ADD" and len(instr) >= 4:
                    rd = int(instr[1])
                    rs1 = int(instr[2])
                    rs2 = int(instr[3])
                    if 0 <= rd < 8 and 0 <= rs1 < 8 and 0 <= rs2 < 8:
                        registers[rd] = registers[rs1] + registers[rs2]

                elif op == "SUB" and len(instr) >= 4:
                    rd = int(instr[1])
                    rs1 = int(instr[2])
                    rs2 = int(instr[3])
                    if 0 <= rd < 8 and 0 <= rs1 < 8 and 0 <= rs2 < 8:
                        registers[rd] = registers[rs1] - registers[rs2]

                elif op == "MUL" and len(instr) >= 4:
                    rd = int(instr[1])
                    rs1 = int(instr[2])
                    rs2 = int(instr[3])
                    if 0 <= rd < 8 and 0 <= rs1 < 8 and 0 <= rs2 < 8:
                        registers[rd] = registers[rs1] * registers[rs2]

                elif op == "DIV" and len(instr) >= 4:
                    rd = int(instr[1])
                    rs1 = int(instr[2])
                    rs2 = int(instr[3])
                    if 0 <= rd < 8 and 0 <= rs1 < 8 and 0 <= rs2 < 8:
                        if registers[rs2] == 0:
                            error = "Division by zero"
                            break
                        registers[rd] = int(registers[rs1] / registers[rs2])

                elif op == "IADD" and len(instr) >= 3:
                    a = float(instr[1])
                    b = float(instr[2])
                    registers[0] = a + b

                elif op == "DISTRIBUTE" and len(instr) >= 3:
                    a = float(instr[1])
                    b = float(instr[2])
                    if b == 0:
                        error = "Division by zero in distribute"
                        break
                    registers[0] = a / b

                elif op == "PRINT" and len(instr) >= 2:
                    reg = int(instr[1])
                    if 0 <= reg < 8:
                        output.append(str(registers[reg]))

                elif op == "HALT":
                    break

                elif op == "NOP":
                    pass

        except (IndexError, TypeError, ValueError) as e:
            error = str(e)

        success = len(error) == 0
        value = registers[0] if success else None
        if output:
            value = output[-1] if not isinstance(value, int) else value

        return ExecutionResult(
            label="",
            success=success,
            value=value,
            error=error,
            confidence=1.0 if success else 0.0,
            metadata={"cycles": cycles},
        )


class BranchingExecutor:
    """Execute all interpretations of an ambiguous parse in parallel.

    The BranchingExecutor:
      1. Takes an AmbiguousParse with multiple interpretations
      2. Executes each interpretation's bytecode
      3. Feeds execution results as evidence to ConfidencePropagation
      4. Determines the winner based on accumulated confidence

    This can use actual parallelism (ThreadPoolExecutor) for production
    or sequential execution for testing.

    Usage:
        parse = AmbiguousParse.from_binary(
            source="加三于四", lang="wen",
            label_a="math", bytecode_a=[("MOVI", 0, 3), ("MOVI", 1, 4), ("IADD", 0, 0, 1)],
            label_b="confucian", bytecode_b=[("MOVI", 0, 3), ("MOVI", 1, 4), ("DISTRIBUTE", 0, 1)],
            weight_a=0.7, weight_b=0.3,
        )

        executor = BranchingExecutor(backend=SimpleBackend())
        result = executor.execute(parse)
        print(result.winner_label)  # "math" (most likely after execution)
        print(result.winner_value)  # 7
    """

    def __init__(
        self,
        backend: Optional[ExecutionBackend] = None,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.backend = backend or SimpleBackend()
        self.parallel = parallel
        self.max_workers = max_workers

    def execute(self, parse: AmbiguousParse) -> BranchingResult:
        """Execute all interpretations and determine the winner.

        Args:
            parse: The ambiguous parse to resolve.

        Returns:
            A BranchingResult with the winning interpretation and all results.
        """
        if parse.is_unambiguous:
            # No ambiguity — just execute the single interpretation
            interp = parse.interpretations[0]
            result = self._execute_one(interp)
            return BranchingResult(
                parse_id=parse.id,
                source=parse.source,
                winner_label=interp.label,
                winner_value=result.value,
                winner_confidence=1.0,
                all_results=[result],
                propagation_summary=None,
                metadata={"unambiguous": True},
            )

        # Execute all interpretations
        all_results = self._execute_all(parse.interpretations)

        # Feed results into confidence propagation
        propagation = ConfidencePropagation(parse)

        for result in all_results:
            propagation.add_execution_result(
                label=result.label,
                valid=result.success,
                output=result.value,
                error=result.error,
            )
            # Set interpretation confidence from execution result
            for interp in parse.interpretations:
                if interp.label == result.label:
                    interp.confidence = result.confidence

        propagation.next_round()

        # Determine winner
        winner = propagation.winner()
        if winner is None:
            # No convergence — use weighted confidence
            winner = parse.best_interpretation()

        if not parse.is_resolved and winner:
            parse.mark_resolved(winner.label, method="converged")

        winner_result = next(
            (r for r in all_results if r.label == winner.label),
            all_results[0],
        )

        return BranchingResult(
            parse_id=parse.id,
            source=parse.source,
            winner_label=winner.label if winner else "",
            winner_value=winner_result.value if winner else None,
            winner_confidence=winner.confidence if winner else 0.0,
            all_results=all_results,
            propagation_summary=propagation.summary(),
            metadata={
                "parallel": self.parallel,
                "interpretation_count": parse.interpretation_count,
            },
        )

    def _execute_one(self, interp: Interpretation) -> ExecutionResult:
        """Execute a single interpretation."""
        result = self.backend.execute(interp.bytecode)
        result.label = interp.label
        return result

    def _execute_all(
        self, interpretations: list[Interpretation]
    ) -> list[ExecutionResult]:
        """Execute all interpretations, optionally in parallel."""
        if self.parallel and len(interpretations) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(self._execute_one, interp): interp
                    for interp in interpretations
                }
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            return results
        else:
            return [self._execute_one(interp) for interp in interpretations]


@dataclass
class BranchingResult:
    """The result of branch execution — the resolved winner and all results."""

    parse_id: str
    source: str
    winner_label: str
    winner_value: Any = None
    winner_confidence: float = 0.0
    all_results: list[ExecutionResult] = field(default_factory=list)
    propagation_summary: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_id": self.parse_id,
            "source": self.source,
            "winner_label": self.winner_label,
            "winner_value": self.winner_value,
            "winner_confidence": round(self.winner_confidence, 6),
            "all_results": [
                {
                    "label": r.label,
                    "success": r.success,
                    "value": r.value,
                    "error": r.error,
                    "confidence": round(r.confidence, 6),
                    "duration_ms": r.duration_ms,
                }
                for r in self.all_results
            ],
            "propagation": self.propagation_summary,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def resolve_ambiguity(
    source: str,
    lang: str,
    interpretations: dict[str, tuple[list[Any], float]],
    backend: Optional[ExecutionBackend] = None,
) -> BranchingResult:
    """Convenience function: create an ambiguous parse and resolve it.

    Args:
        source: The ambiguous natural language input.
        lang: Language tag.
        interpretations: Dict of label → (bytecode, weight).
        backend: Optional execution backend.

    Returns:
        A BranchingResult with the resolved winner.
    """
    parse = AmbiguousParse.from_options(source, lang, interpretations)
    executor = BranchingExecutor(backend=backend)
    return executor.execute(parse)
