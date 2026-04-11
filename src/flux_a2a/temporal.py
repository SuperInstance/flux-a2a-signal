"""
FLUX-A2A Temporal Logic for Agent Systems (Rounds 10-11).

This module implements formal temporal reasoning for agent execution traces:

  1. **LTL (Linear Temporal Logic)**: formulas evaluated over linear traces.
     - Operators: ◇ (eventually), □ (always), ○ (next), U (until), W (weak_until), R (release)
     - Agent guarantees: "◇ consensus", "□ capability_check", "request U response"
     - Evaluation over finite execution traces with stuttering semantics.

  2. **CTL (Computational Tree Logic)**: formulas evaluated over branching computation trees.
     - Path quantifiers: A (all paths), E (exists a path)
     - Temporal operators: ◇ (eventually), □ (always), ○ (next), U (until)
     - Agent branching: "A◇convergence" (on ALL branch outcomes, eventually converge)
     - "E□conflict" (there EXISTS a path where conflict persists)

  3. **Session Types**: type-safe communication channels between agents.
     - Each session has a protocol type that determines valid message sequences.
     - Mapped to Korean honorific levels (7 levels) and German Kasus (4 cases).
     - Protocol: Send, Receive, Choice, RecvChoice, Close.
     - The type system ensures agents follow the communication protocol correctly.

  4. **AgentTrace**: execution trace with temporal annotations.
     - Records every event (action, message, state change) with timestamps.
     - Supports LTL/CTL formula evaluation against the trace.
     - Tracks confidence, agent identity, and causal ordering.

Design principles:
  - Temporal logic is not just verification — it's a SPECIFICATION language.
  - Agents can declare temporal contracts: "I guarantee ◇response within 30s"
  - Session types encode social hierarchy (honorifics) as type constraints.
  - Cross-language temporal patterns: Latin tenses → temporal aspects,
    Chinese 因果 → causal chains, Sanskrit hetu-phala → cause-effect scope.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


# ===========================================================================
# LTL: Linear Temporal Logic
# ===========================================================================

class LTLOp(str, Enum):
    """LTL temporal and boolean operators."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    IMPLIES = "IMPLIES"
    NEXT = "NEXT"               # ○ P — P holds at the next state
    EVENTUALLY = "EVENTUALLY"   # ◇ P — P holds at some future state
    ALWAYS = "ALWAYS"           # □ P — P holds at all future states
    UNTIL = "UNTIL"             # P U Q — P holds until Q holds
    WEAK_UNTIL = "W_UNTIL"      # P W Q — P holds until Q holds (P may hold forever)
    RELEASE = "RELEASE"         # P R Q — Q holds until P holds (Q may hold forever)
    ATOM = "ATOM"               # Atomic proposition


@dataclass
class LTLFormula:
    """
    An LTL formula over agent execution traces.

    LTL formulas are trees of operators applied to atomic propositions.

    Examples:
        # "eventually all agents agree"
        LTLFormula(LTLOp.EVENTUALLY, child=LTLFormula(LTLOp.ATOM, atom="consensus"))

        # "always capability checks pass before execution"
        cap = LTLFormula(LTLOp.ATOM, atom="capability_check_pass")
        exec_ = LTLFormula(LTLOp.ATOM, atom="execution_start")
        LTLFormula(LTLOp.ALWAYS, child=LTLFormula(LTLOp.IMPLIES, left=exec_, right=cap))

        # "request until response"
        LTLFormula(LTLOp.UNTIL,
                   left=LTLFormula(LTLOp.ATOM, atom="request_pending"),
                   right=LTLFormula(LTLOp.ATOM, atom="response_received"))
    """
    op: str = LTLOp.ATOM.value
    atom: str = ""               # For ATOM nodes: the proposition name
    left: Optional[LTLFormula] = None
    right: Optional[LTLFormula] = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- Constructors for common patterns ------------------------------------

    @classmethod
    def prop(cls, name: str) -> LTLFormula:
        """Atomic proposition."""
        return cls(op=LTLOp.ATOM.value, atom=name)

    @classmethod
    def eventually(cls, child: LTLFormula) -> LTLFormula:
        """◇ child — child eventually holds."""
        return cls(op=LTLOp.EVENTUALLY.value, left=child)

    @classmethod
    def always(cls, child: LTLFormula) -> LTLFormula:
        """□ child — child always holds."""
        return cls(op=LTLOp.ALWAYS.value, left=child)

    @classmethod
    def next(cls, child: LTLFormula) -> LTLFormula:
        """○ child — child holds at the next step."""
        return cls(op=LTLOp.NEXT.value, left=child)

    @classmethod
    def until(cls, left: LTLFormula, right: LTLFormula) -> LTLFormula:
        """left U right — left holds until right holds."""
        return cls(op=LTLOp.UNTIL.value, left=left, right=right)

    @classmethod
    def weak_until(cls, left: LTLFormula, right: LTLFormula) -> LTLFormula:
        """left W right — left holds until right (or forever)."""
        return cls(op=LTLOp.WEAK_UNTIL.value, left=left, right=right)

    @classmethod
    def release(cls, left: LTLFormula, right: LTLFormula) -> LTLFormula:
        """left R right — right holds until left (or forever)."""
        return cls(op=LTLOp.RELEASE.value, left=left, right=right)

    @classmethod
    def implies(cls, left: LTLFormula, right: LTLFormula) -> LTLFormula:
        """left → right."""
        return cls(op=LTLOp.IMPLIES.value, left=left, right=right)

    @classmethod
    def And(cls, *formulas: LTLFormula) -> LTLFormula:
        """Conjunction of formulas."""
        if not formulas:
            return cls(op=LTLOp.TRUE.value)
        result = formulas[0]
        for f in formulas[1:]:
            result = cls(op=LTLOp.AND.value, left=result, right=f)
        return result

    @classmethod
    def Or(cls, *formulas: LTLFormula) -> LTLFormula:
        """Disjunction of formulas."""
        if not formulas:
            return cls(op=LTLOp.FALSE.value)
        result = formulas[0]
        for f in formulas[1:]:
            result = cls(op=LTLOp.OR.value, left=result, right=f)
        return result

    @classmethod
    def Not(cls, child: LTLFormula) -> LTLFormula:
        """Negation."""
        return cls(op=LTLOp.NOT.value, left=child)

    def __str__(self) -> str:
        """Pretty-print the LTL formula."""
        if self.op == LTLOp.ATOM.value:
            return self.atom
        if self.op == LTLOp.TRUE.value:
            return "⊤"
        if self.op == LTLOp.FALSE.value:
            return "⊥"
        if self.op == LTLOp.NOT.value:
            return f"¬{self.left}"
        if self.op in (LTLOp.AND.value, LTLOp.OR.value, LTLOp.IMPLIES.value):
            sym = {"AND": "∧", "OR": "∨", "IMPLIES": "→"}[self.op]
            return f"({self.left} {sym} {self.right})"
        if self.op == LTLOp.NEXT.value:
            return f"○{self.left}"
        if self.op == LTLOp.EVENTUALLY.value:
            return f"◇{self.left}"
        if self.op == LTLOp.ALWAYS.value:
            return f"□{self.left}"
        if self.op in (LTLOp.UNTIL.value, LTLOp.WEAK_UNTIL.value, LTLOp.RELEASE.value):
            sym = {"UNTIL": "U", "W_UNTIL": "W", "RELEASE": "R"}[self.op]
            return f"({self.left} {sym} {self.right})"
        return f"?{self.op}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.atom:
            d["atom"] = self.atom
        if self.left:
            d["left"] = self.left.to_dict()
        if self.right:
            d["right"] = self.right.to_dict()
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LTLFormula:
        return cls(
            op=data["op"],
            atom=data.get("atom", ""),
            left=cls.from_dict(data["left"]) if "left" in data else None,
            right=cls.from_dict(data["right"]) if "right" in data else None,
            meta=data.get("meta", {}),
        )


@dataclass
class LTLEvaluation:
    """
    Result of evaluating an LTL formula against a trace.
    """
    formula: str                          # String representation
    result: bool                          # Whether the formula holds
    first_violation: int = -1             # Index of first violation (-1 if holds)
    evaluation_at: dict[int, bool] = field(default_factory=dict)  # Per-step truth values
    meta: dict[str, Any] = field(default_factory=dict)


class TemporalLogic:
    """
    LTL formula evaluation on agent execution traces.

    The evaluator implements standard LTL semantics over finite traces:
    - The trace is a sequence of states, each containing a set of propositions.
    - Stuttering extension: the last state repeats forever.
    - This handles ◇ (eventually) and □ (always) correctly on finite traces.

    Usage:
        tl = TemporalLogic()
        trace = [{"ready": True}, {"processing": True}, {"done": True}]
        result = tl.evaluate(LTLFormula.eventually(LTLFormula.prop("done")), trace)
        assert result.result is True
    """

    def evaluate(self, formula: LTLFormula, trace: list[dict[str, Any]]) -> LTLEvaluation:
        """
        Evaluate an LTL formula against a trace.

        Args:
            formula: The LTL formula to evaluate.
            trace: A list of state dictionaries. Each state maps proposition
                   names to boolean values (or any truthy/falsy value).

        Returns:
            LTLEvaluation with result and per-step evaluation.
        """
        if not trace:
            # Empty trace: only TRUE and ALWAYS hold; EVENTUALLY doesn't
            return LTLEvaluation(
                formula=str(formula),
                result=(formula.op == LTLOp.TRUE.value),
                first_violation=0,
            )

        # Compute per-step truth values using recursive LTL evaluation
        # For efficiency, we use a bottom-up approach with memoization
        n = len(trace)
        evaluation_at: dict[int, bool] = {}

        # Evaluate at each position with memoization
        memo: dict[tuple[str, int], bool] = {}

        for i in range(n):
            evaluation_at[i] = self._eval_at(formula, i, trace, n, memo)

        # Overall result: formula holds at position 0
        result = evaluation_at.get(0, False)

        # Find first violation
        first_violation = -1
        for i in range(n):
            if not evaluation_at.get(i, False):
                first_violation = i
                break

        return LTLEvaluation(
            formula=str(formula),
            result=result,
            first_violation=first_violation,
            evaluation_at=evaluation_at,
        )

    def evaluate_batch(
        self,
        formulas: list[LTLFormula],
        trace: list[dict[str, Any]],
    ) -> list[LTLEvaluation]:
        """Evaluate multiple formulas against the same trace."""
        return [self.evaluate(f, trace) for f in formulas]

    def _eval_at(
        self,
        formula: LTLFormula,
        index: int,
        trace: list[dict[str, Any]],
        trace_len: int,
        memo: dict[tuple[str, int], bool],
    ) -> bool:
        """Evaluate formula at a specific trace index."""
        key = (str(formula), index)
        if key in memo:
            return memo[key]

        result = self._eval_inner(formula, index, trace, trace_len, memo)
        memo[key] = result
        return result

    def _eval_inner(
        self,
        formula: LTLFormula,
        index: int,
        trace: list[dict[str, Any]],
        trace_len: int,
        memo: dict[tuple[str, int], bool],
    ) -> bool:
        """Inner evaluation logic for LTL formulas."""
        op = formula.op
        state = trace[index] if index < trace_len else trace[-1]

        # Stuttering: indices beyond trace length use the last state
        effective_index = min(index, trace_len - 1)

        if op == LTLOp.TRUE.value:
            return True
        if op == LTLOp.FALSE.value:
            return False
        if op == LTLOp.ATOM.value:
            return bool(state.get(formula.atom, False))
        if op == LTLOp.NOT.value and formula.left:
            return not self._eval_at(formula.left, index, trace, trace_len, memo)
        if op == LTLOp.AND.value and formula.left and formula.right:
            return (self._eval_at(formula.left, index, trace, trace_len, memo) and
                    self._eval_at(formula.right, index, trace, trace_len, memo))
        if op == LTLOp.OR.value and formula.left and formula.right:
            return (self._eval_at(formula.left, index, trace, trace_len, memo) or
                    self._eval_at(formula.right, index, trace, trace_len, memo))
        if op == LTLOp.IMPLIES.value and formula.left and formula.right:
            return (not self._eval_at(formula.left, index, trace, trace_len, memo) or
                    self._eval_at(formula.right, index, trace, trace_len, memo))

        # Temporal operators
        if op == LTLOp.NEXT.value and formula.left:
            next_idx = index + 1
            if next_idx >= trace_len:
                # Stuttering: □P holds at the end iff P holds at the end
                # ○P at the end evaluates P at the end (stuttering)
                return self._eval_at(formula.left, effective_index, trace, trace_len, memo)
            return self._eval_at(formula.left, next_idx, trace, trace_len, memo)

        if op == LTLOp.EVENTUALLY.value and formula.left:
            # ◇P: P holds at some future state (including now)
            for i in range(index, trace_len + 1):
                if self._eval_at(formula.left, i, trace, trace_len, memo):
                    return True
            return False

        if op == LTLOp.ALWAYS.value and formula.left:
            # □P: P holds at all future states (including now)
            for i in range(index, trace_len + 1):
                if not self._eval_at(formula.left, i, trace, trace_len, memo):
                    return False
            return True

        if op == LTLOp.UNTIL.value and formula.left and formula.right:
            # P U Q: P holds from now until Q first becomes true (Q must eventually hold)
            for i in range(index, trace_len + 1):
                if self._eval_at(formula.right, i, trace, trace_len, memo):
                    return True
                if not self._eval_at(formula.left, i, trace, trace_len, memo):
                    return False
            return False  # Q never held — until fails

        if op == LTLOp.WEAK_UNTIL.value and formula.left and formula.right:
            # P W Q: P holds until Q holds, or P holds forever
            for i in range(index, trace_len + 1):
                if self._eval_at(formula.right, i, trace, trace_len, memo):
                    return True
                if not self._eval_at(formula.left, i, trace, trace_len, memo):
                    return False
            return True  # P held forever (end of trace with stuttering)

        if op == LTLOp.RELEASE.value and formula.left and formula.right:
            # P R Q: Q holds until P holds (and P holds at or before Q stops)
            # Equivalent to: ¬(¬P U ¬Q)
            for i in range(index, trace_len + 1):
                if not self._eval_at(formula.right, i, trace, trace_len, memo):
                    return self._eval_at(formula.left, i, trace, trace_len, memo)
            return True  # Q held forever

        return False  # Unknown operator


# ===========================================================================
# CTL: Computational Tree Logic (for branching computations)
# ===========================================================================

class CTLOp(str, Enum):
    """CTL path quantifiers and temporal operators."""
    A = "A"       # All paths
    E = "E"       # Exists a path
    AF = "AF"     # All paths, Eventually (A◇)
    AG = "AG"     # All paths, Always (A□)
    AX = "AX"     # All paths, neXt (A○)
    AU = "AU"     # All paths, Until (AU)
    EF = "EF"     # Exists path, Eventually (E◇)
    EG = "EG"     # Exists path, Always (E□)
    EX = "EX"     # Exists path, neXt (E○)
    EU = "EU"     # Exists path, Until (EU)
    ATOM = "ATOM"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"


@dataclass
class CTLFormula:
    """
    A CTL formula evaluated over branching computation trees.

    CTL combines path quantifiers (A, E) with temporal operators (◇, □, ○, U).
    This is essential for agent systems where branching creates a TREE of possible
    futures.

    Examples:
        # "On ALL paths, eventually the system converges"
        CTLFormula(CTLOp.AF, child=CTLFormula(CTLOp.ATOM, atom="converged"))

        # "There EXISTS a path where conflict persists"
        CTLFormula(CTLOp.EG, child=CTLFormula(CTLOp.ATOM, atom="conflict"))

        # "On all paths, capability_check until execution"
        CTLFormula(CTLOp.AU,
                   left=CTLFormula(CTLOp.ATOM, atom="capability_check"),
                   right=CTLFormula(CTLOp.ATOM, atom="execution"))
    """
    op: str = CTLOp.ATOM.value
    atom: str = ""
    left: Optional[CTLFormula] = None
    right: Optional[CTLFormula] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def prop(cls, name: str) -> CTLFormula:
        return cls(op=CTLOp.ATOM.value, atom=name)

    @classmethod
    def AF(cls, child: CTLFormula) -> CTLFormula:
        """On ALL paths, eventually child holds."""
        return cls(op=CTLOp.AF.value, left=child)

    @classmethod
    def AG(cls, child: CTLFormula) -> CTLFormula:
        """On ALL paths, always child holds."""
        return cls(op=CTLOp.AG.value, left=child)

    @classmethod
    def EF(cls, child: CTLFormula) -> CTLFormula:
        """There EXISTS a path where eventually child holds."""
        return cls(op=CTLOp.EF.value, left=child)

    @classmethod
    def EG(cls, child: CTLFormula) -> CTLFormula:
        """There EXISTS a path where always child holds."""
        return cls(op=CTLOp.EG.value, left=child)

    @classmethod
    def AU(cls, left: CTLFormula, right: CTLFormula) -> CTLFormula:
        """On ALL paths, left holds until right holds."""
        return cls(op=CTLOp.AU.value, left=left, right=right)

    @classmethod
    def EU(cls, left: CTLFormula, right: CTLFormula) -> CTLFormula:
        """There EXISTS a path where left holds until right holds."""
        return cls(op=CTLOp.EU.value, left=left, right=right)

    def __str__(self) -> str:
        if self.op == CTLOp.ATOM.value:
            return self.atom
        if self.op in (CTLOp.NOT.value, CTLOp.AND.value, CTLOp.OR.value):
            sym = {"NOT": "¬", "AND": "∧", "OR": "∨"}[self.op]
            return f"({sym} {self.left})"
        if self.op in (CTLOp.AF.value, CTLOp.AG.value, CTLOp.EF.value, CTLOp.EG.value,
                       CTLOp.AX.value, CTLOp.EX.value):
            return f"{self.op}({self.left})"
        if self.op in (CTLOp.AU.value, CTLOp.EU.value):
            return f"{self.op}({self.left}, {self.right})"
        return f"?{self.op}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.atom:
            d["atom"] = self.atom
        if self.left:
            d["left"] = self.left.to_dict()
        if self.right:
            d["right"] = self.right.to_dict()
        return d


@dataclass
class BranchPoint:
    """
    A branching point in a computation tree.

    Each branch has a label, a set of propositions that hold at that branch,
    and children (successor branch points).
    """
    id: str = ""
    propositions: set[str] = field(default_factory=set)
    children: list[BranchPoint] = field(default_factory=list)
    parent: Optional[BranchPoint] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "propositions": list(self.propositions),
            "children": [c.to_dict() for c in self.children],
        }


class CTLEvaluator:
    """
    Evaluate CTL formulas over branching computation trees.

    Unlike LTL (which evaluates over linear traces), CTL evaluates over
    trees where each node can branch into multiple children.

    Usage:
        tree = BranchPoint("root", {"ready"}, [
            BranchPoint("branch_a", {"path_a", "success"}),
            BranchPoint("branch_b", {"path_b", "failure"}),
        ])
        eval = CTLEvaluator()
        result = eval.evaluate(CTLFormula.AF(CTLFormula.prop("success")), tree)
    """

    def evaluate(self, formula: CTLFormula, root: BranchPoint) -> bool:
        """
        Evaluate a CTL formula over a computation tree.

        Uses the standard CTL model checking algorithm with state labeling.
        """
        return self._eval(formula, root, set())

    def _eval(self, formula: CTLFormula, state: BranchPoint,
              visited: set[str]) -> bool:
        if state.id in visited:
            # Handle cycles (shouldn't happen in DAG trees)
            return False
        visited = visited | {state.id}

        op = formula.op

        if op == CTLOp.ATOM.value:
            return formula.atom in state.propositions

        if op == CTLOp.NOT.value and formula.left:
            return not self._eval(formula.left, state, visited)

        if op == CTLOp.AND.value and formula.left and formula.right:
            return (self._eval(formula.left, state, visited) and
                    self._eval(formula.right, state, visited))

        if op == CTLOp.OR.value and formula.left and formula.right:
            return (self._eval(formula.left, state, visited) or
                    self._eval(formula.right, state, visited))

        # Path quantifiers + temporal operators
        if op == CTLOp.AF.value and formula.left:
            # A◇P: On ALL paths, eventually P holds
            # Equivalent to: ¬E(¬P U ¬(P ∨ ¬P)) = ¬EG¬P
            return self._all_eventually(formula.left, state, visited)

        if op == CTLOp.AG.value and formula.left:
            # A□P: On ALL paths, always P holds
            # Equivalent to: ¬EF¬P
            return self._all_always(formula.left, state, visited)

        if op == CTLOp.EF.value and formula.left:
            # E◇P: There EXISTS a path where eventually P holds
            # Equivalent to: E(true U P)
            return self._exists_eventually(formula.left, state, visited)

        if op == CTLOp.EG.value and formula.left:
            # E□P: There EXISTS a path where always P holds
            return self._exists_always(formula.left, state, visited)

        if op == CTLOp.AX.value and formula.left:
            # A○P: On ALL paths, P holds at the next state
            for child in state.children:
                if not self._eval(formula.left, child, visited):
                    return False
            return True

        if op == CTLOp.EX.value and formula.left:
            # E○P: There EXISTS a path where P holds at the next state
            for child in state.children:
                if self._eval(formula.left, child, visited):
                    return True
            return False

        if op == CTLOp.AU.value and formula.left and formula.right:
            # AU(P, Q): On ALL paths, P holds until Q holds
            for child in state.children:
                if not self._until_on_path(formula.left, formula.right, child, visited):
                    return False
            return True

        if op == CTLOp.EU.value and formula.left and formula.right:
            # EU(P, Q): There EXISTS a path where P holds until Q holds
            for child in state.children:
                if self._until_on_path(formula.left, formula.right, child, visited):
                    return True
            return False

        return False

    def _all_eventually(self, formula: CTLFormula, state: BranchPoint,
                        visited: set[str]) -> bool:
        """A◇: On all paths from state, eventually formula holds."""
        if formula.atom in state.propositions:
            return True
        if not state.children:
            return formula.atom in state.propositions
        for child in state.children:
            if not self._all_eventually(formula, child, visited):
                return False
        return True

    def _all_always(self, formula: CTLFormula, state: BranchPoint,
                    visited: set[str]) -> bool:
        """A□: On all paths from state, always formula holds."""
        if formula.atom not in state.propositions:
            return False
        for child in state.children:
            if not self._all_always(formula, child, visited):
                return False
        return True

    def _exists_eventually(self, formula: CTLFormula, state: BranchPoint,
                           visited: set[str]) -> bool:
        """E◇: There exists a path from state where eventually formula holds."""
        if formula.atom in state.propositions:
            return True
        for child in state.children:
            if self._exists_eventually(formula, child, visited):
                return True
        return False

    def _exists_always(self, formula: CTLFormula, state: BranchPoint,
                       visited: set[str]) -> bool:
        """E□: There exists a path from state where always formula holds."""
        if formula.atom not in state.propositions:
            return False
        if not state.children:
            return True
        for child in state.children:
            if self._exists_always(formula, child, visited):
                return True
        return False

    def _until_on_path(self, left: CTLFormula, right: CTLFormula,
                       state: BranchPoint, visited: set[str]) -> bool:
        """Check if left holds until right holds on a single path."""
        if right.atom in state.propositions:
            return True
        if left.atom not in state.propositions:
            return False
        if not state.children:
            return False
        for child in state.children:
            if self._until_on_path(left, right, child, visited):
                return True
        return False


# ===========================================================================
# Session Types: Type-Safe Agent Communication Channels
# ===========================================================================

class SessionOp(str, Enum):
    """Session type operations."""
    SEND = "Send"           # !T — send a message of type T
    RECV = "Recv"           # ?T — receive a message of type T
    CHOICE = "Choice"       # &{l₁: S₁, ..., lₙ: Sₙ} — offer a choice (server)
    RECV_CHOICE = "Select"  # ⊕{l₁: S₁, ..., lₙ: Sₙ} — select an option (client)
    CLOSE = "Close"         # end — close the session
    DELEGATE = "Delegate"   # Delegate session to another agent


@dataclass
class SessionType:
    """
    A session type describing the protocol for typed agent communication.

    Session types ensure that agents follow the correct communication protocol.
    The type is a linear type — each operation consumes the type and produces
    the remainder.

    Cross-language mappings:
    - Korean honorifics: 7 levels map to 7 session privilege levels
      - 하십시오체 (formal deferential) = Close (highest authority, session end)
      - 해요체 (polite) = Send (active communication)
      - 해체 (informal) = Recv (passive reception)
      - 하오체 (authoritative) = Choice (offer options)
    - German Kasus: 4 cases map to 4 session scopes
      - Nominativ = Send (subject initiates)
      - Akkusativ = Recv (object receives)
      - Dativ = Delegate (indirect object, relay)
      - Genitiv = Close (possessive, session termination)
    - Sanskrit vibhakti: 8 cases map to 8 session operations
      - प्रथमा (Nominative) = Send
      - द्वितीया (Accusative) = Recv
      - तृतीया (Instrumental) = Choice
      - चतुर्थी (Dative) = RecvChoice
      - पञ्चमी (Ablative) = Delegate
      - षष्ठी (Genitive) = Close
      - सप्तमी (Locative) = nested session scope
      - संबोधन (Vocative) = session broadcast

    Examples:
        # A request-response protocol
        SessionType(SessionOp.SEND, remaining=SessionType(SessionOp.RECV, remaining=SessionType(SessionOp.CLOSE)))

        # Request-response using convenience constructors
        SessionType.request_response()
    """
    op: str = SessionOp.CLOSE.value
    message_type: str = "any"
    remaining: Optional[SessionType] = None
    choices: dict[str, SessionType] = field(default_factory=dict)
    honorific_level: int = 0   # Korean honorific level (0-6)
    kasus_case: str = ""       # German Kasus case
    vibhakti: str = ""         # Sanskrit vibhakti case
    meta: dict[str, Any] = field(default_factory=dict)

    # -- Convenience constructors -------------------------------------------

    @classmethod
    def close(cls) -> SessionType:
        """End the session."""
        return cls(op=SessionOp.CLOSE.value)

    @classmethod
    def send(cls, message_type: str = "any",
             remaining: Optional[SessionType] = None) -> SessionType:
        """Send a message and continue with remaining protocol."""
        return cls(op=SessionOp.SEND.value, message_type=message_type,
                   remaining=remaining or cls.close())

    @classmethod
    def recv(cls, message_type: str = "any",
             remaining: Optional[SessionType] = None) -> SessionType:
        """Receive a message and continue with remaining protocol."""
        return cls(op=SessionOp.RECV.value, message_type=message_type,
                   remaining=remaining or cls.close())

    @classmethod
    def choice(cls, choices: dict[str, SessionType]) -> SessionType:
        """Offer a choice between labeled alternatives."""
        return cls(op=SessionOp.CHOICE.value, choices=choices)

    @classmethod
    def select(cls, choices: dict[str, SessionType]) -> SessionType:
        """Select from offered alternatives."""
        return cls(op=SessionOp.RECV_CHOICE.value, choices=choices)

    @classmethod
    def request_response(cls, request_type: str = "request",
                         response_type: str = "response") -> SessionType:
        """Common request-response pattern."""
        return cls.send(request_type, cls.recv(response_type, cls.close()))

    @classmethod
    def negotiate(cls, rounds: int = 3) -> SessionType:
        """Multi-round negotiation protocol."""
        if rounds <= 0:
            return cls.close()
        return cls.send("proposal",
                        cls.recv("counter",
                                 cls.negotiate(rounds - 1)))

    @classmethod
    def peer_review(cls, criteria_count: int = 3) -> SessionType:
        """Peer review protocol: submit → receive reviews → respond → close."""
        return (cls.send("submission",
                         cls.recv("review", cls.recv("review",
                                  cls.recv("review",
                                           cls.send("response",
                                                    cls.close()))))))

    # -- Type operations ----------------------------------------------------

    def step(self, actual_op: SessionOp, actual_type: str = "any",
             selected_label: str = "") -> SessionType:
        """
        Advance the session type by one step.

        Args:
            actual_op: The actual operation performed.
            actual_type: The message type (for Send/Recv).
            selected_label: The label selected (for Select).

        Returns:
            The remaining session type after this step.

        Raises:
            TypeError: If the operation doesn't match the expected protocol.
        """
        if self.op == SessionOp.CLOSE.value:
            raise TypeError("Session is closed; no more operations allowed")

        if self.op == SessionOp.SEND.value:
            if actual_op != SessionOp.SEND:
                raise TypeError(f"Expected Send(!{self.message_type}), got {actual_op}")
            if self.remaining is None:
                return cls.close()
            return self.remaining

        if self.op == SessionOp.RECV.value:
            if actual_op != SessionOp.RECV:
                raise TypeError(f"Expected Recv(?{self.message_type}), got {actual_op}")
            if self.remaining is None:
                return cls.close()
            return self.remaining

        if self.op == SessionOp.CHOICE.value:
            if actual_op == SessionOp.CHOICE:
                # The server doesn't choose; it waits
                return self
            if selected_label not in self.choices:
                raise TypeError(f"Unknown choice label: {selected_label}. "
                                f"Available: {list(self.choices.keys())}")
            return self.choices[selected_label]

        if self.op == SessionOp.RECV_CHOICE.value:
            if actual_op != SessionOp.RECV_CHOICE:
                raise TypeError(f"Expected Select(⊕), got {actual_op}")
            if selected_label not in self.choices:
                raise TypeError(f"Unknown select label: {selected_label}. "
                                f"Available: {list(self.choices.keys())}")
            return self.choices[selected_label]

        raise TypeError(f"Unknown session operation: {self.op}")

    def is_complete(self) -> bool:
        """Check if the session protocol has been completed."""
        return self.op == SessionOp.CLOSE.value

    def depth(self) -> int:
        """Compute the depth (number of remaining steps) of the protocol."""
        if self.op == SessionOp.CLOSE.value:
            return 0
        if self.remaining:
            return 1 + self.remaining.depth()
        if self.choices:
            return 1 + max(st.depth() for st in self.choices.values())
        return 1

    def to_korean_honorific(self) -> str:
        """
        Map the current session operation to a Korean honorific level.

        Mapping:
          Close       → 하십시오체 (hasipsioche, level 6): formal deferential — session end
          Send        → 해요체 (haeyoche, level 4): polite — active communication
          Recv        → 해체 (haeche, level 2): informal — passive reception
          Choice      → 하오체 (haoche, level 3): authoritative — offer options
          RecvChoice  → 해라체 (haerache, level 1): plain — select from options
          Delegate    → 합쇼체 (hapsyoche, level 5): formal — relay authority
        """
        honorific_map = {
            SessionOp.CLOSE.value: ("하십시오체 (hasipsioche)", 6),
            SessionOp.SEND.value: ("해요체 (haeyoche)", 4),
            SessionOp.RECV.value: ("해체 (haeche)", 2),
            SessionOp.CHOICE.value: ("하오체 (haoche)", 3),
            SessionOp.RECV_CHOICE.value: ("해라체 (haerache)", 1),
            SessionOp.DELEGATE.value: ("합쇼체 (hapsyoche)", 5),
        }
        name, level = honorific_map.get(self.op, ("unknown", 0))
        self.honorific_level = level
        return name

    def to_german_kasus(self) -> str:
        """
        Map the current session operation to a German Kasus case.

        Mapping:
          Send      → Nominativ (subject initiates)
          Recv      → Akkusativ (direct object receives)
          Delegate  → Dativ (indirect object, relay)
          Close     → Genitiv (possessive, session termination)
          Choice    → Nominativ+Akkusativ (offering options)
          RecvChoice→ Akkusativ (selecting from options)
        """
        kasus_map = {
            SessionOp.SEND.value: "Nominativ",
            SessionOp.RECV.value: "Akkusativ",
            SessionOp.DELEGATE.value: "Dativ",
            SessionOp.CLOSE.value: "Genitiv",
            SessionOp.CHOICE.value: "Nominativ+Akkusativ",
            SessionOp.RECV_CHOICE.value: "Akkusativ",
        }
        case = kasus_map.get(self.op, "unknown")
        self.kasus_case = case
        return case

    def to_sanskrit_vibhakti(self) -> str:
        """
        Map the current session operation to a Sanskrit vibhakti case.

        Mapping:
          Send       → प्रथमा (prathama, 1st/Nominative): subject initiates
          Recv       → द्वितीया (dvitiya, 2nd/Accusative): object receives
          Choice     → तृतीया (tritiya, 3rd/Instrumental): offering by means
          RecvChoice → चतुर्थी (chaturthi, 4th/Dative): selection for benefit
          Delegate   → पञ्चमी (panchami, 5th/Ablative): relay from source
          Close      → षष्ठी (shashthi, 6th/Genitive): possessive termination
        """
        vibhakti_map = {
            SessionOp.SEND.value: "प्रथमा (prathamā)",
            SessionOp.RECV.value: "द्वितीया (dvitīyā)",
            SessionOp.CHOICE.value: "तृतीया (tṛtīyā)",
            SessionOp.RECV_CHOICE.value: "चतुर्थी (caturthī)",
            SessionOp.DELEGATE.value: "पञ्चमी (pañcamī)",
            SessionOp.CLOSE.value: "षष्ठी (ṣaṣṭhī)",
        }
        case = vibhakti_map.get(self.op, "unknown")
        self.vibhakti = case
        return case

    def __str__(self) -> str:
        if self.op == SessionOp.CLOSE.value:
            return "end"
        if self.op == SessionOp.SEND.value:
            rest = f" . {self.remaining}" if self.remaining else ""
            return f"!{self.message_type}{rest}"
        if self.op == SessionOp.RECV.value:
            rest = f" . {self.remaining}" if self.remaining else ""
            return f"?{self.message_type}{rest}"
        if self.op == SessionOp.CHOICE.value:
            alts = " | ".join(f"{k}: {v}" for k, v in self.choices.items())
            return f"&{{{alts}}}"
        if self.op == SessionOp.RECV_CHOICE.value:
            alts = " | ".join(f"{k}: {v}" for k, v in self.choices.items())
            return f"⊕{{{alts}}}"
        return f"?{self.op}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": self.op,
            "message_type": self.message_type,
        }
        if self.remaining:
            d["remaining"] = self.remaining.to_dict()
        if self.choices:
            d["choices"] = {k: v.to_dict() for k, v in self.choices.items()}
        d["honorific_level"] = self.honorific_level
        if self.kasus_case:
            d["kasus_case"] = self.kasus_case
        if self.vibhakti:
            d["vibhakti"] = self.vibhakti
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionType:
        return cls(
            op=data["op"],
            message_type=data.get("message_type", "any"),
            remaining=cls.from_dict(data["remaining"]) if "remaining" in data else None,
            choices={k: cls.from_dict(v) for k, v in data.get("choices", {}).items()},
            honorific_level=data.get("honorific_level", 0),
            kasus_case=data.get("kasus_case", ""),
            vibhakti=data.get("vibhakti", ""),
            meta=data.get("meta", {}),
        )


@dataclass
class SessionChannel:
    """
    An active session-typed channel between two agents.

    Tracks the current protocol state for both endpoints (dual types).
    """
    channel_id: str = ""
    sender_agent: str = ""
    receiver_agent: str = ""
    protocol: SessionType = field(default_factory=SessionType.close)
    history: list[dict[str, Any]] = field(default_factory=list)
    is_open: bool = True

    def __post_init__(self) -> None:
        if not self.channel_id:
            self.channel_id = str(uuid.uuid4())

    def send(self, message_type: str = "any") -> SessionType:
        """Perform a Send operation on this channel."""
        if not self.is_open:
            raise TypeError(f"Channel {self.channel_id} is closed")
        next_protocol = self.protocol.step(SessionOp.SEND, message_type)
        self.history.append({"op": "Send", "type": message_type,
                             "from": self.sender_agent,
                             "to": self.receiver_agent,
                             "timestamp": datetime.now(timezone.utc).isoformat()})
        self.protocol = next_protocol
        if self.protocol.is_complete():
            self.is_open = False
        return next_protocol

    def recv(self, message_type: str = "any") -> SessionType:
        """Perform a Recv operation on this channel."""
        if not self.is_open:
            raise TypeError(f"Channel {self.channel_id} is closed")
        next_protocol = self.protocol.step(SessionOp.RECV, message_type)
        self.history.append({"op": "Recv", "type": message_type,
                             "from": self.receiver_agent,
                             "to": self.sender_agent,
                             "timestamp": datetime.now(timezone.utc).isoformat()})
        self.protocol = next_protocol
        if self.protocol.is_complete():
            self.is_open = False
        return next_protocol

    def close(self) -> None:
        """Close the channel."""
        self.protocol = SessionType.close()
        self.is_open = False
        self.history.append({"op": "Close",
                             "timestamp": datetime.now(timezone.utc).isoformat()})

    def current_honorific(self) -> str:
        """Get the Korean honorific level for the current protocol state."""
        return self.protocol.to_korean_honorific()

    def current_kasus(self) -> str:
        """Get the German Kasus case for the current protocol state."""
        return self.protocol.to_german_kasus()

    def current_vibhakti(self) -> str:
        """Get the Sanskrit vibhakti case for the current protocol state."""
        return self.protocol.to_sanskrit_vibhakti()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "sender_agent": self.sender_agent,
            "receiver_agent": self.receiver_agent,
            "protocol": self.protocol.to_dict(),
            "history": self.history,
            "is_open": self.is_open,
        }


# ===========================================================================
# Agent Trace: Execution Trace with Temporal Annotations
# ===========================================================================

class TraceEventType(str, Enum):
    """Types of events in an agent execution trace."""
    AGENT_ACTION = "agent_action"
    MESSAGE_SEND = "message_send"
    MESSAGE_RECV = "message_recv"
    STATE_CHANGE = "state_change"
    BRANCH_START = "branch_start"
    BRANCH_END = "branch_end"
    FORK_START = "fork_start"
    FORK_END = "fork_end"
    MERGE = "merge"
    CONFLICT = "conflict"
    CONSENSUS = "consensus"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


@dataclass
class TraceEvent:
    """
    A single event in an agent execution trace.
    """
    event_id: str = ""
    event_type: str = TraceEventType.AGENT_ACTION.value
    agent_id: str = ""
    timestamp: str = ""
    propositions: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str = ""     # For causal ordering
    causal_clock: dict[str, int] = field(default_factory=dict)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "propositions": self.propositions,
            "confidence": self.confidence,
        }
        if self.parent_event_id:
            d["parent_event_id"] = self.parent_event_id
        if self.causal_clock:
            d["causal_clock"] = self.causal_clock
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(
            event_id=data.get("event_id", ""),
            event_type=data.get("event_type", TraceEventType.AGENT_ACTION.value),
            agent_id=data.get("agent_id", ""),
            timestamp=data.get("timestamp", ""),
            propositions=data.get("propositions", {}),
            parent_event_id=data.get("parent_event_id", ""),
            causal_clock=data.get("causal_clock", {}),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


class AgentTrace:
    """
    Execution trace for an agent system with temporal annotations.

    The trace records every event and supports LTL/CTL formula evaluation.
    It maintains a vector clock for causal ordering of events.

    Usage:
        trace = AgentTrace()
        trace.add_event(TraceEvent(agent_id="a", propositions={"ready": True}))
        trace.add_event(TraceEvent(agent_id="a", propositions={"processing": True}))
        trace.add_event(TraceEvent(agent_id="a", propositions={"done": True}))

        result = trace.evaluate_ltl(
            LTLFormula.eventually(LTLFormula.prop("done"))
        )
        assert result.result is True
    """

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.events: list[TraceEvent] = []
        self._clocks: dict[str, dict[str, int]] = {}  # agent_id -> vector clock
        self._metadata: dict[str, Any] = {}

    def add_event(self, event: TraceEvent) -> None:
        """Add an event to the trace, updating causal clocks."""
        # Update vector clock for the agent
        if event.agent_id not in self._clocks:
            self._clocks[event.agent_id] = {}
        clock = self._clocks[event.agent_id]
        clock[event.agent_id] = clock.get(event.agent_id, 0) + 1

        # Inherit from parent event's clock
        if event.parent_event_id:
            for e in self.events:
                if e.event_id == event.parent_event_id:
                    for aid, val in e.causal_clock.items():
                        clock[aid] = max(clock.get(aid, 0), val)
                    break

        event.causal_clock = dict(clock)
        self.events.append(event)

    def add_events(self, events: list[TraceEvent]) -> None:
        """Add multiple events to the trace."""
        for event in events:
            self.add_event(event)

    def get_events(self, agent_id: Optional[str] = None,
                   event_type: Optional[str] = None) -> list[TraceEvent]:
        """Filter events by agent and/or type."""
        result = self.events
        if agent_id:
            result = [e for e in result if e.agent_id == agent_id]
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        return result

    def to_proposition_trace(self, agent_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Convert the trace to a list of proposition dictionaries for LTL evaluation.

        Each state in the trace is the union of propositions from all events
        at that position (or filtered by agent_id).
        """
        events = self.get_events(agent_id=agent_id) if agent_id else self.events
        return [dict(e.propositions) for e in events]

    def evaluate_ltl(self, formula: LTLFormula,
                     agent_id: Optional[str] = None) -> LTLEvaluation:
        """
        Evaluate an LTL formula against this trace.

        Args:
            formula: The LTL formula to evaluate.
            agent_id: If provided, only consider events from this agent.
        """
        trace = self.to_proposition_trace(agent_id=agent_id)
        evaluator = TemporalLogic()
        result = evaluator.evaluate(formula, trace)
        return result

    def evaluate_ltl_batch(self, formulas: list[LTLFormula],
                           agent_id: Optional[str] = None) -> list[LTLEvaluation]:
        """Evaluate multiple LTL formulas against this trace."""
        trace = self.to_proposition_trace(agent_id=agent_id)
        evaluator = TemporalLogic()
        return evaluator.evaluate_batch(formulas, trace)

    def evaluate_ctl(self, formula: CTLFormula,
                     root: Optional[BranchPoint] = None) -> bool:
        """
        Evaluate a CTL formula against a computation tree derived from this trace.

        If no tree is provided, constructs one from the trace's branching events.
        """
        if root is None:
            root = self._build_tree_from_trace()
        evaluator = CTLEvaluator()
        return evaluator.evaluate(formula, root)

    def _build_tree_from_trace(self) -> BranchPoint:
        """Build a computation tree from the trace's branching structure."""
        # Find all branch events and build a tree
        branch_events = [e for e in self.events
                         if e.event_type == TraceEventType.BRANCH_START.value]
        if not branch_events:
            # Linear trace: single path
            all_props: set[str] = set()
            for e in self.events:
                all_props.update(e.propositions.keys())
            return BranchPoint(id="root", propositions=all_props)

        # Build tree from branch structure
        root = BranchPoint(id="root", propositions=set())
        for event in self.events:
            if event.event_type == TraceEventType.BRANCH_START.value:
                branch_id = event.metadata.get("branch_id", event.event_id)
                child = BranchPoint(id=branch_id,
                                    propositions=set(event.propositions.keys()))
                child.parent = root
                root.children.append(child)
            elif event.event_type == TraceEventType.BRANCH_END.value:
                # Collect final propositions
                for prop in event.propositions:
                    root.propositions.add(prop)
            elif not branch_events:
                root.propositions.update(event.propositions.keys())

        return root

    def causal_order(self) -> list[str]:
        """
        Return events in causal order (topological sort by vector clocks).
        """
        # Sort by vector clock (lexicographic comparison)
        def clock_key(event: TraceEvent) -> tuple:
            clock = event.causal_clock
            return tuple(sorted(clock.items()))

        sorted_events = sorted(self.events, key=clock_key)
        return [e.event_id for e in sorted_events]

    def find_concurrent_events(self) -> list[list[str]]:
        """
        Find groups of events that are causally concurrent (no ordering).
        """
        groups: list[list[str]] = []
        assigned: set[str] = set()

        for i, e1 in enumerate(self.events):
            if e1.event_id in assigned:
                continue
            group = [e1.event_id]
            for j, e2 in enumerate(self.events):
                if i >= j or e2.event_id in assigned:
                    continue
                # Check if e1 and e2 are concurrent
                c1 = e1.causal_clock
                c2 = e2.causal_clock
                all_agents = set(c1.keys()) | set(c2.keys())
                c1_before = all(c1.get(a, 0) <= c2.get(a, 0) for a in all_agents)
                c2_before = all(c2.get(a, 0) <= c1.get(a, 0) for a in all_agents)
                strict_1 = any(c1.get(a, 0) < c2.get(a, 0) for a in all_agents)
                strict_2 = any(c2.get(a, 0) < c1.get(a, 0) for a in all_agents)

                if not (c1_before and strict_2) and not (c2_before and strict_1):
                    group.append(e2.event_id)

            assigned.update(group)
            groups.append(group)

        return groups

    def summary(self) -> dict[str, Any]:
        """Get a summary of the trace."""
        agents = set(e.agent_id for e in self.events)
        types = defaultdict(int)
        for e in self.events:
            types[e.event_type] += 1

        return {
            "trace_id": self.trace_id,
            "total_events": len(self.events),
            "agents": list(agents),
            "event_types": dict(types),
            "duration": (
                f"{self.events[-1].timestamp} - {self.events[0].timestamp}"
                if len(self.events) >= 2 else "N/A"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": [e.to_dict() for e in self.events],
            "metadata": self._metadata,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentTrace:
        trace = cls(trace_id=data.get("trace_id", ""))
        trace._metadata = data.get("metadata", {})
        for event_data in data.get("events", []):
            trace.add_event(TraceEvent.from_dict(event_data))
        return trace


# ===========================================================================
# Pre-defined LTL specifications for agent systems
# ===========================================================================

# "eventually all agents agree"
LTL_EVENTUAL_CONSENSUS = LTLFormula.eventually(LTLFormula.prop("consensus"))

# "always capability checks pass before execution"
LTL_CAPABILITY_SAFETY = LTLFormula.always(
    LTLFormula.implies(
        LTLFormula.prop("execution_start"),
        LTLFormula.prop("capability_check_pass"),
    )
)

# "no conflict persists forever"
LTL_NO_PERMANENT_CONFLICT = LTLFormula.eventually(
    LTLFormula.Not(LTLFormula.prop("conflict"))
)

# "requests eventually get responses"
LTL_REQUEST_RESPONSE = LTLFormula.always(
    LTLFormula.implies(
        LTLFormula.prop("request_sent"),
        LTLFormula.eventually(LTLFormula.prop("response_received")),
    )
)

# "the discussion terminates with consensus"
LTL_DISCUSSION_TERMINATES = LTLFormula.eventually(
    LTLFormula.Or(
        LTLFormula.prop("consensus"),
        LTLFormula.prop("discussion_timeout"),
        LTLFormula.prop("stalemate_detected"),
    )
)

# "confidence never drops below threshold during execution"
def ltl_confidence_bound(threshold: float = 0.5) -> LTLFormula:
    return LTLFormula.always(
        LTLFormula.implies(
            LTLFormula.prop("execution_active"),
            LTLFormula.prop(f"confidence_ge_{threshold}"),
        )
    )

# Pre-defined CTL specifications for branching systems

# "On ALL paths, eventually the system converges"
CTL_ALL_CONVERGE = CTLFormula.AF(CTLFormula.prop("converged"))

# "There EXISTS a path where conflict persists"
CTL_EXISTS_CONFLICT_PATH = CTLFormula.EG(CTLFormula.prop("conflict"))

# "On ALL paths, capability checks always pass"
CTL_ALL_SAFE = CTLFormula.AG(CTLFormula.prop("capability_safe"))

# "There EXISTS a path to consensus"
CTL_EXISTS_CONSENSUS = CTLFormula.EF(CTLFormula.prop("consensus"))
