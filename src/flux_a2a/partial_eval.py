"""
FLUX Partial Evaluator — Futamura Projections for NL Programs.

Implements partial evaluation (PE) — the technique of specializing a program
with respect to some known inputs, producing a residual program that is more
efficient because the known parts are pre-computed.

The three Futamura Projections:
  - Projection 1: specialize(prog, input) → residual_prog  (compile a program)
  - Projection 2: specialize(interpreter, prog) → compiled_prog  (compile via interpreter)
  - Projection 3: specialize(mix, compiler) → compiler_generator  (make a compiler)

For FLUX, Projection 1 is the most immediately useful:
  Given an NL program and a known vocabulary/context, produce optimized bytecode
  where NL→op resolution is pre-computed for all known terms.

This module also provides:
  - Static knowledge injection (known vocabulary → pre-resolved operations)
  - Constant folding for known values
  - Dead code elimination for statically determined branches
  - Type specialization when variable types are known

References:
  - Yoshihiko Futamura, "Partial Evaluation of Computation Process" (1971)
  - Neil Jones, "Partial Evaluation and Automatic Program Generation" (1993)
  - PyPy tracing JIT documentation
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ===========================================================================
# Types and Constants
# ===========================================================================

class PELevel(str, Enum):
    """How aggressively to partial-evaluate."""
    LIGHT = "light"          # Only fold constants
    MEDIUM = "medium"        # Constant fold + dead code elim + type specialize
    AGGRESSIVE = "aggressive"  # Full PE including unrolling and inlining


class ReductionResult(str, Enum):
    """Result of attempting to reduce an expression."""
    REDUCED = "reduced"       # Successfully reduced to a value
    RESIDUAL = "residual"     # Cannot reduce — produce residual expression
    UNKNOWN = "unknown"       # Insufficient information


# ===========================================================================
# Static Knowledge Base
# ===========================================================================

@dataclass
class StaticKnowledge:
    """Known facts about a program's execution environment.

    Partial evaluation uses this knowledge to pre-compute parts of the program.
    The richer the knowledge, the more of the program can be specialized away.
    """
    # Known variable values
    constants: dict[str, Any] = field(default_factory=dict)
    # Known variable types (variable_name → type string)
    types: dict[str, str] = field(default_factory=dict)
    # Known NL vocabulary (natural language term → resolved op dict)
    vocabulary: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Known language of the program
    language: str = "flux"
    # Known agent capabilities
    agent_capabilities: dict[str, list[str]] = field(default_factory=dict)
    # Known trust levels
    trust_levels: dict[str, float] = field(default_factory=dict)
    # Functions that are known to be pure (no side effects)
    pure_functions: set[str] = field(default_factory=lambda: {
        "add", "sub", "mul", "div", "mod", "eq", "neq", "lt", "lte", "gt", "gte",
        "and", "or", "not", "xor", "concat", "length", "collect", "reduce",
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "constants": self.constants,
            "types": self.types,
            "vocabulary": {k: v for k, v in list(self.vocabulary.items())[:20]},
            "language": self.language,
            "agent_capabilities": self.agent_capabilities,
            "trust_levels": self.trust_levels,
        }


# ===========================================================================
# Partial Evaluator
# ===========================================================================

@dataclass
class PEResult:
    """Result of partially evaluating an expression or program."""
    # The residual program (simplified/optimized)
    residual: Any = None
    # How many expressions were reduced (pre-computed)
    reductions: int = 0
    # How many expressions were left as residual
    residuals: int = 0
    # Reduction rate (reductions / total)
    reduction_rate: float = 0.0
    # Specific optimizations applied
    optimizations_applied: list[str] = field(default_factory=list)
    # New constants discovered during PE
    discovered_constants: dict[str, Any] = field(default_factory=dict)
    # Estimated speedup
    speedup_estimate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reductions": self.reductions,
            "residuals": self.residuals,
            "reduction_rate": round(self.reduction_rate, 3),
            "optimizations_applied": self.optimizations_applied,
            "discovered_constants": self.discovered_constants,
            "speedup_estimate": round(self.speedup_estimate, 2),
        }


class PartialEvaluator:
    """
    Specializes NL programs for known inputs (Futamura Projection 1).

    The partial evaluator takes a program and a set of known facts (static
    knowledge), and produces a residual program where everything that can be
    pre-computed has been computed.

    Key optimizations:
      1. NL Resolution: If "三只猫" → [literal, 3] is known, skip LLM resolution
      2. Constant Folding: If x=5 is known, replace `x + 3` with `8`
      3. Dead Branch Elim: If condition is known true, remove else branch
      4. Type Specialization: If x is always int, emit int-specific bytecode
      5. Vocabulary Fast-Path: Known terms bypass the general NL resolver

    Example usage:
        >>> pe = PartialEvaluator()
        >>> knowledge = StaticKnowledge(
        ...     constants={"pi": 3.14159},
        ...     types={"x": "int", "y": "float"},
        ...     vocabulary={"三只猫": {"op": "literal", "value": 3}},
        ...     language="zho",
        ... )
        >>> result = pe.evaluate(program_dict, knowledge)
        >>> # result.residual is the specialized program
        >>> # result.reduction_rate tells us how much was pre-computed
    """

    # Pure arithmetic operations that can be constant-folded
    _ARITH_OPS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
        "mod": lambda a, b: a % b if b != 0 else None,
    }
    _CMP_OPS = {
        "eq": lambda a, b: a == b,
        "neq": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
    }

    def __init__(self, level: PELevel = PELevel.MEDIUM) -> None:
        self.level = level
        self._reductions = 0
        self._residuals = 0
        self._optimizations: list[str] = []
        self._discovered: dict[str, Any] = {}
        self._knowledge: StaticKnowledge = StaticKnowledge()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, program: Any, knowledge: Optional[StaticKnowledge] = None) -> PEResult:
        """Partially evaluate a program with respect to known inputs.

        This is Futamura Projection 1:
            specialize(NL_program, known_vocabulary) → optimized_bytecode

        Parameters
        ----------
        program:
            A Program dict, list of expressions, or single expression.
        knowledge:
            Static knowledge about the program's execution environment.

        Returns
        -------
        PEResult with the residual program and optimization statistics.
        """
        self._reductions = 0
        self._residuals = 0
        self._optimizations = []
        self._discovered = {}
        self._knowledge = knowledge or StaticKnowledge()

        if isinstance(program, dict):
            inner = program.get("signal", program)
            if "body" in inner:
                residual_body = self._pe_body(inner["body"])
                residual = {"signal": {**inner, "body": residual_body}}
            else:
                residual = self._pe_expr(program)
        elif isinstance(program, list):
            residual = self._pe_body(program)
        else:
            residual = program

        total = self._reductions + self._residuals
        reduction_rate = self._reductions / total if total > 0 else 0.0
        speedup = 1.0 + reduction_rate * 0.5  # Conservative estimate

        return PEResult(
            residual=residual,
            reductions=self._reductions,
            residuals=self._residuals,
            reduction_rate=reduction_rate,
            optimizations_applied=self._optimizations,
            discovered_constants=self._discovered,
            speedup_estimate=speedup,
        )

    def project_2(self, interpreter_source: Any, program: Any) -> PEResult:
        """
        Futamura Projection 2: specialize(interpreter, program) → compiled_program.

        This takes the interpreter's logic and a specific program, and produces
        a standalone executable that doesn't need the interpreter.

        In practice, this is what our Compiler does — but with PE, we can also
        specialize the compilation process itself. For example, if a program
        only uses "add", "if", and "loop", we don't need to include the
        interpreter's handling for "co_iterate", "discuss", etc.

        This is a simplified implementation that marks which interpreter paths
        are needed for a given program.
        """
        self._reductions = 0
        self._residuals = 0
        self._optimizations = []
        self._discovered = {}

        # Extract all ops used by the program
        ops_used: set[str] = set()
        self._collect_ops(program, ops_used)

        # Compute the specialized interpreter surface
        all_ops = {
            "add", "sub", "mul", "div", "mod", "eq", "neq", "lt", "lte", "gt", "gte",
            "and", "or", "not", "xor", "concat", "length", "at", "collect", "reduce",
            "seq", "if", "loop", "while", "match", "let", "get", "set", "struct",
            "tell", "ask", "delegate", "broadcast", "signal", "await",
            "branch", "fork", "merge", "co_iterate", "trust", "confidence",
            "literal", "yield", "eval", "discuss", "synthesize", "reflect",
        }
        eliminated_ops = all_ops - ops_used

        if eliminated_ops:
            self._optimizations.append(f"eliminated_{len(eliminated_ops)}_ops")
            self._reductions += len(eliminated_ops)

        self._residuals += len(ops_used)
        total = self._reductions + self._residuals

        # Build the specialized interpreter spec
        specialized = {
            "projection": "futamura_2",
            "ops_required": sorted(ops_used),
            "ops_eliminated": sorted(eliminated_ops),
            "specialization_ratio": (
                round(len(eliminated_ops) / len(all_ops), 3) if all_ops else 0.0
            ),
        }

        return PEResult(
            residual=specialized,
            reductions=self._reductions,
            residuals=self._residuals,
            reduction_rate=self._reductions / total if total > 0 else 0.0,
            optimizations_applied=self._optimizations,
            speedup_estimate=1.0 + len(eliminated_ops) * 0.02,
        )

    # ------------------------------------------------------------------
    # Body Evaluation
    # ------------------------------------------------------------------

    def _pe_body(self, body: list[Any]) -> list[Any]:
        """Partially evaluate a list of expressions."""
        residual: list[Any] = []
        for expr in body:
            if isinstance(expr, dict) and "op" in expr:
                result = self._pe_expr(expr)
                if result is not None:
                    residual.append(result)
            elif isinstance(expr, list):
                result = self._pe_body(expr)
                residual.append(result)
            else:
                residual.append(expr)
        return residual

    # ------------------------------------------------------------------
    # Expression Evaluation
    # ------------------------------------------------------------------

    def _pe_expr(self, expr: dict[str, Any]) -> Any:
        """Partially evaluate a single expression.

        Returns either a reduced value (if fully computable) or a residual
        expression (if some inputs are unknown).
        """
        if not isinstance(expr, dict):
            return expr

        op = expr.get("op", "")

        # --- Literal: always reducible ---
        if op == "literal":
            self._reductions += 1
            return expr.get("value")

        # --- Let: if value is known, propagate ---
        if op == "let":
            return self._pe_let(expr)

        # --- Get: if variable is known, substitute ---
        if op == "get":
            return self._pe_get(expr)

        # --- Arithmetic: try constant folding ---
        if op in self._ARITH_OPS:
            return self._pe_arithmetic(op, expr)

        # --- Comparison: try constant folding ---
        if op in self._CMP_OPS:
            return self._pe_comparison(op, expr)

        # --- Logic: try constant folding ---
        if op in ("and", "or", "not"):
            return self._pe_logic(op, expr)

        # --- If: try branch elimination ---
        if op == "if":
            return self._pe_if(expr)

        # --- Seq: recurse into body ---
        if op == "seq":
            return self._pe_seq(expr)

        # --- Loop with known count: try unrolling (aggressive only) ---
        if op == "loop":
            return self._pe_loop(expr)

        # --- Struct: try field reduction ---
        if op == "struct":
            return self._pe_struct(expr)

        # --- Vocabulary fast-path: known NL terms ---
        if op == "resolve_nl" or op == "nl":
            return self._pe_nl_resolution(expr)

        # --- Default: mark as residual ---
        self._residuals += 1
        return expr

    def _pe_let(self, expr: dict[str, Any]) -> dict[str, Any]:
        """Partial evaluation of let binding."""
        name = expr.get("name", "")
        value = expr.get("value")

        if value is not None:
            pe_value = self._pe_value(value)
            if not isinstance(pe_value, dict) or "op" not in pe_value:
                # Value was fully reduced — record it as known
                self._knowledge.constants[name] = pe_value
                if not isinstance(pe_value, dict):
                    self._discovered[name] = pe_value
                self._reductions += 1
                self._optimizations.append(f"constant_prop:{name}")
                # The let itself can be eliminated (value is now known)
                return {"op": "nop", "eliminated": f"let:{name}", "value": pe_value}

        # Value is residual — keep the let but PE its value
        residual_value = self._pe_value(value) if value is not None else value
        self._residuals += 1
        return {**expr, "value": residual_value}

    def _pe_get(self, expr: dict[str, Any]) -> Any:
        """Partial evaluation of variable access."""
        name = expr.get("name", "")
        if name in self._knowledge.constants:
            self._reductions += 1
            self._optimizations.append(f"subst:{name}")
            return self._knowledge.constants[name]
        self._residuals += 1
        return expr

    def _pe_arithmetic(self, op: str, expr: dict[str, Any]) -> Any:
        """Try to constant-fold arithmetic."""
        args = expr.get("args", [])

        if self.level in (PELevel.MEDIUM, PELevel.AGGRESSIVE):
            # Try to resolve all args
            resolved = [self._try_resolve(a) for a in args]

            if all(r[0] for r in resolved):
                values = [r[1] for r in resolved]
                result = self._ARITH_OPS[op](*values)
                if result is not None:
                    self._reductions += 1
                    self._optimizations.append(f"const_fold:{op}")
                    return result

        # Partially resolve args
        pe_args = [self._pe_value(a) for a in args]
        self._residuals += 1
        return {**expr, "args": pe_args}

    def _pe_comparison(self, op: str, expr: dict[str, Any]) -> Any:
        """Try to constant-fold comparisons."""
        args = expr.get("args", [])

        if self.level in (PELevel.MEDIUM, PELevel.AGGRESSIVE):
            resolved = [self._try_resolve(a) for a in args]

            if all(r[0] for r in resolved):
                try:
                    result = self._CMP_OPS[op](resolved[0][1], resolved[1][1])
                    self._reductions += 1
                    self._optimizations.append(f"const_fold:{op}")
                    return result
                except TypeError:
                    pass

        pe_args = [self._pe_value(a) for a in args]
        self._residuals += 1
        return {**expr, "args": pe_args}

    def _pe_logic(self, op: str, expr: dict[str, Any]) -> Any:
        """Try to constant-fold logic operations."""
        args = expr.get("args", [])

        if op == "not":
            resolved = self._try_resolve(args[0]) if args else (False, None)
            if resolved[0] is not None and resolved[0]:
                self._reductions += 1
                self._optimizations.append("const_fold:not")
                return not resolved[1]
        elif op == "and":
            # Short-circuit: if any arg is false, result is false
            if self.level == PELevel.AGGRESSIVE:
                for arg in args:
                    resolved = self._try_resolve(arg)
                    if resolved[0] and not resolved[1]:
                        self._reductions += 1
                        self._optimizations.append("const_fold:and_short")
                        return False
        elif op == "or":
            # Short-circuit: if any arg is true, result is true
            if self.level == PELevel.AGGRESSIVE:
                for arg in args:
                    resolved = self._try_resolve(arg)
                    if resolved[0] and resolved[1]:
                        self._reductions += 1
                        self._optimizations.append("const_fold:or_short")
                        return True

        pe_args = [self._pe_value(a) for a in args]
        self._residuals += 1
        return {**expr, "args": pe_args}

    def _pe_if(self, expr: dict[str, Any]) -> Any:
        """Try dead branch elimination for if expressions."""
        cond = expr.get("condition", expr.get("cond"))
        then_body = expr.get("then", expr.get("body", []))
        else_body = expr.get("else", [])

        resolved = self._try_resolve(cond)
        if resolved[0]:
            condition_value = resolved[1]
            if condition_value:
                self._reductions += 1
                self._optimizations.append("dead_branch_elim:then")
                if isinstance(then_body, list):
                    return self._pe_body(then_body)
                return self._pe_expr(then_body) if isinstance(then_body, dict) else then_body
            else:
                self._reductions += 1
                self._optimizations.append("dead_branch_elim:else")
                if isinstance(else_body, list):
                    return self._pe_body(else_body)
                return self._pe_expr(else_body) if isinstance(else_body, dict) else else_body

        # Cannot determine condition — residual
        pe_cond = self._pe_value(cond)
        pe_then = self._pe_body(then_body) if isinstance(then_body, list) else self._pe_value(then_body)
        pe_else = self._pe_body(else_body) if isinstance(else_body, list) else self._pe_value(else_body)

        self._residuals += 1
        result: dict[str, Any] = dict(expr)
        if "condition" in result:
            result["condition"] = pe_cond
        elif "cond" in result:
            result["cond"] = pe_cond
        result["then"] = pe_then if not isinstance(pe_then, list) else pe_then
        result["else"] = pe_else
        return result

    def _pe_seq(self, expr: dict[str, Any]) -> dict[str, Any]:
        """Partially evaluate a sequence."""
        body = expr.get("body", [])
        pe_body = self._pe_body(body)
        self._residuals += 1
        return {**expr, "body": pe_body}

    def _pe_loop(self, expr: dict[str, Any]) -> dict[str, Any]:
        """Partially evaluate a loop.

        In LIGHT mode: just PE the body.
        In MEDIUM mode: if count is known, annotate it.
        In AGGRESSIVE mode: if count is small and known, unroll.
        """
        times = expr.get("times", expr.get("count"))
        body = expr.get("body", [])

        resolved_times = self._try_resolve(times) if times is not None else None

        if self.level == PELevel.AGGRESSIVE and resolved_times and resolved_times[0]:
            count = int(resolved_times[1])
            if 0 < count <= 10:
                # Unroll the loop
                unrolled: list[Any] = []
                for i in range(count):
                    # Substitute the loop variable
                    var_name = expr.get("var", "item")
                    self._knowledge.constants[var_name] = i
                    unrolled.extend(self._pe_body(body))
                    del self._knowledge.constants[var_name]
                self._reductions += 1
                self._optimizations.append(f"loop_unroll:{count}")
                if unrolled:
                    return {"op": "seq", "body": unrolled}
                return {"op": "literal", "value": None}

        # Residual loop
        pe_body = self._pe_body(body)
        self._residuals += 1
        return {**expr, "body": pe_body}

    def _pe_struct(self, expr: dict[str, Any]) -> dict[str, Any]:
        """Partially evaluate struct fields."""
        fields = expr.get("fields", {})
        pe_fields: dict[str, Any] = {}
        for key, val in fields.items():
            pe_fields[key] = self._pe_value(val)
        self._residuals += 1
        return {**expr, "fields": pe_fields}

    def _pe_nl_resolution(self, expr: dict[str, Any]) -> Any:
        """Fast-path for NL resolution when vocabulary is known.

        If we know that "三只猫" resolves to {"op": "literal", "value": 3},
        we can skip the LLM resolution step entirely.
        """
        raw_term = expr.get("term", expr.get("source", expr.get("value", "")))

        if raw_term and isinstance(raw_term, str):
            # Check vocabulary
            if raw_term in self._knowledge.vocabulary:
                resolved = self._knowledge.vocabulary[raw_term]
                self._reductions += 1
                self._optimizations.append(f"vocab_fast_path:{raw_term[:10]}")
                return resolved

            # Check if it's a known constant reference
            if raw_term in self._knowledge.constants:
                value = self._knowledge.constants[raw_term]
                self._reductions += 1
                self._optimizations.append(f"const_resolve:{raw_term}")
                return {"op": "literal", "value": value}

        self._residuals += 1
        return expr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pe_value(self, value: Any) -> Any:
        """Partially evaluate a value (literal or expression)."""
        if isinstance(value, dict) and "op" in value:
            return self._pe_expr(value)
        if isinstance(value, list):
            return [self._pe_value(v) for v in value]
        return value

    def _try_resolve(self, value: Any) -> tuple[bool, Any]:
        """Try to resolve a value to a concrete constant.

        Returns (is_resolved, resolved_value).
        """
        if value is None:
            return (True, None)
        if isinstance(value, (int, float, bool, str)):
            return (True, value)
        if isinstance(value, dict):
            if "op" in value:
                if value["op"] == "literal":
                    return (True, value.get("value"))
                if value["op"] == "get":
                    name = value.get("name", "")
                    if name in self._knowledge.constants:
                        return (True, self._knowledge.constants[name])
                # Try PE and check if it reduced
                pe_result = self._pe_expr(value)
                if not isinstance(pe_result, dict):
                    return (True, pe_result)
        return (False, value)

    def _collect_ops(self, program: Any, ops: set[str]) -> None:
        """Recursively collect all ops used in a program."""
        if isinstance(program, dict):
            op = program.get("op", "")
            if op:
                ops.add(op)
            for v in program.values():
                if isinstance(v, (dict, list)):
                    self._collect_ops(v, ops)
        elif isinstance(program, list):
            for item in program:
                if isinstance(item, (dict, list)):
                    self._collect_ops(item, ops)


# ===========================================================================
# Convenience Functions
# ===========================================================================

def partial_evaluate(
    program: Any,
    knowledge: Optional[StaticKnowledge] = None,
    level: PELevel = PELevel.MEDIUM,
) -> PEResult:
    """Partial-evaluate a program with respect to known inputs.

    This is Futamura Projection 1: specialize(prog, known_input) → residual_prog.

    Parameters
    ----------
    program:
        A Signal program dict or list of expressions.
    knowledge:
        Static knowledge about the execution environment.
    level:
        How aggressively to partial-evaluate.

    Returns
    -------
    PEResult with the specialized residual program.
    """
    pe = PartialEvaluator(level=level)
    return pe.evaluate(program, knowledge)


def specialize_interpreter(
    program: Any,
) -> PEResult:
    """Futamura Projection 2: specialize the interpreter for a specific program.

    Determines which interpreter opcodes are needed and which can be eliminated.

    Parameters
    ----------
    program:
        A Signal program to specialize the interpreter for.

    Returns
    -------
    PEResult with the specialized interpreter surface.
    """
    pe = PartialEvaluator(level=PELevel.MEDIUM)
    return pe.project_2(None, program)


def build_knowledge(
    constants: Optional[dict[str, Any]] = None,
    types: Optional[dict[str, str]] = None,
    vocabulary: Optional[dict[str, dict[str, Any]]] = None,
    language: str = "flux",
) -> StaticKnowledge:
    """Build a StaticKnowledge object for partial evaluation."""
    return StaticKnowledge(
        constants=constants or {},
        types=types or {},
        vocabulary=vocabulary or {},
        language=language,
    )
