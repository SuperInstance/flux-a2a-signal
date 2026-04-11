"""
FLUX-A2A Compiler — JSON → FLUX Bytecode.

Compiles Signal programs to FLUX bytecode (.fluxb).  Supports language-specific
compilation paths through the six FLUX runtimes, as well as direct bytecode
emission.

The bytecode format uses a simple stack-based design:
  - Each instruction is a list: [OPCODE, ...operands]
  - Labels are strings resolved to instruction indices
  - Branch tables map label → jump targets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Bytecode instruction opcodes
# ---------------------------------------------------------------------------

class BcOp(str, Enum):
    """Bytecode opcodes for the FLUX VM."""
    # Stack
    PUSH = "PUSH"
    POP = "POP"
    DUP = "DUP"
    SWAP = "SWAP"

    # Arithmetic
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    NEG = "NEG"

    # Comparison
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    LTE = "LTE"
    GT = "GT"
    GTE = "GTE"

    # Logic
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # String
    CONCAT = "CONCAT"
    LENGTH = "LENGTH"

    # Collection
    AT = "AT"
    COLLECT = "COLLECT"
    REDUCE = "REDUCE"

    # Variables
    LOAD = "LOAD"
    STORE = "STORE"

    # Control flow
    JUMP = "JUMP"
    JUMP_IF = "JUMP_IF"
    JUMP_IF_NOT = "JUMP_IF_NOT"
    LABEL = "LABEL"
    CALL = "CALL"
    RET = "RET"
    HALT = "HALT"

    # Struct
    STRUCT = "STRUCT"

    # Agent communication
    TELL = "TELL"
    ASK = "ASK"
    DELEGATE = "DELEGATE"
    BROADCAST = "BROADCAST"
    SIGNAL = "SIGNAL"
    AWAIT = "AWAIT"

    # Agent operations
    BRANCH = "BRANCH"
    FORK = "FORK"
    MERGE = "MERGE"
    CO_ITERATE = "CO_ITERATE"

    # Trust & confidence
    TRUST = "TRUST"
    CONFIDENCE = "CONFIDENCE"

    # Metadata
    LANG_TAG = "LANG_TAG"
    NOP = "NOP"


# ---------------------------------------------------------------------------
# Bytecode chunk
# ---------------------------------------------------------------------------

@dataclass
class BytecodeChunk:
    """A compiled bytecode unit — a sequence of instructions with metadata."""
    instructions: list[list[Any]] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)
    constants: list[Any] = field(default_factory=list)
    source_map: list[Optional[str]] = field(default_factory=list)  # debug source info
    lang_tags: list[str] = field(default_factory=list)

    def emit(self, opcode: str, *operands: Any, source: str = "") -> int:
        """Emit a single instruction and return its index."""
        idx = len(self.instructions)
        instr = [opcode] + list(operands)
        self.instructions.append(instr)
        self.source_map.append(source or None)
        if not self.lang_tags:
            self.lang_tags.append("flux")
        return idx

    def emit_push(self, value: Any, source: str = "") -> int:
        """Push a constant value onto the stack."""
        if value not in self.constants:
            self.constants.append(value)
        const_idx = self.constants.index(value)
        return self.emit(BcOp.PUSH.value, const_idx, source=source)

    def emit_label(self, name: str) -> int:
        """Place a named label at the current position."""
        idx = len(self.instructions)
        self.labels[name] = idx
        self.emit(BcOp.LABEL.value, name)
        return idx

    def resolve_jump(self, instr_idx: int, target_label: str) -> None:
        """Patch a jump instruction to point to the resolved label."""
        if target_label not in self.labels:
            raise ValueError(f"Unresolved label: {target_label}")
        target = self.labels[target_label]
        if instr_idx < len(self.instructions):
            self.instructions[instr_idx][-1] = target

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            "labels": self.labels,
            "constants": self.constants,
            "source_map": self.source_map,
            "lang_tags": self.lang_tags,
        }


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class Compiler:
    """Compile Signal JSON programs to FLUX bytecode.

    Parameters
    ----------
    optimizations:
        List of optimization passes to apply.
    """

    def __init__(self, optimizations: Optional[list[str]] = None) -> None:
        self.optimizations = optimizations or []
        self._label_counter = 0

    def _next_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f"{prefix}_{self._label_counter}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_program(self, program: Any) -> BytecodeChunk:
        """Compile a Program object or raw dict to bytecode."""
        from flux_a2a.schema import Program, Expression

        if isinstance(program, Program):
            body = program.body
        elif isinstance(program, dict):
            inner = program.get("signal", program)
            body_raw = inner.get("body", [])
            body = [Expression.from_dict(e) if isinstance(e, dict) else e for e in body_raw]
        else:
            raise TypeError(f"Cannot compile {type(program)}")

        chunk = BytecodeChunk()

        # Emit program header
        if isinstance(program, Program) and program.id:
            chunk.emit(BcOp.NOP.value, f"program:{program.id}", source="header")

        # Register agents
        if isinstance(program, Program):
            for agent in program.agents:
                chunk.emit(
                    BcOp.NOP.value,
                    f"agent:{agent.id}:{agent.role}",
                    source="agents",
                )

        # Compile body
        for expr in body:
            if isinstance(expr, dict) and "op" in expr:
                self._compile_expr(Expression.from_dict(expr), chunk)
            elif isinstance(expr, Expression):
                self._compile_expr(expr, chunk)

        chunk.emit(BcOp.HALT.value, source="end")
        return chunk

    def compile_expr(self, expr: Any) -> BytecodeChunk:
        """Compile a single expression to bytecode."""
        from flux_a2a.schema import Expression

        if isinstance(expr, dict):
            expr = Expression.from_dict(expr)
        chunk = BytecodeChunk()
        self._compile_expr(expr, chunk)
        chunk.emit(BcOp.HALT.value)
        return chunk

    # ------------------------------------------------------------------
    # Expression compilation
    # ------------------------------------------------------------------

    def _compile_expr(self, expr: Any, chunk: BytecodeChunk) -> None:
        from flux_a2a.schema import Expression

        if not isinstance(expr, Expression):
            expr = Expression.from_dict(expr) if isinstance(expr, dict) else Expression(op="literal", params={"value": expr})

        # Language tag
        if expr.lang and expr.lang != "flux":
            chunk.emit(BcOp.LANG_TAG.value, expr.lang, source=f"lang:{expr.op}")
            chunk.lang_tags[-1] = expr.lang

        # Confidence
        if expr.confidence < 1.0:
            chunk.emit(BcOp.CONFIDENCE.value, expr.confidence, source=f"conf:{expr.op}")

        op = expr.op

        # --- Arithmetic ---
        if op in ("add", "sub", "mul", "div", "mod"):
            self._compile_arithmetic(op, expr, chunk)
        elif op in ("eq", "neq", "lt", "lte", "gt", "gte"):
            self._compile_comparison(op, expr, chunk)
        elif op in ("and", "or", "not", "xor"):
            self._compile_logic(op, expr, chunk)
        elif op == "concat":
            self._compile_concat(expr, chunk)
        elif op == "length":
            self._compile_length(expr, chunk)
        elif op == "at":
            self._compile_at(expr, chunk)
        elif op == "collect":
            self._compile_collect(expr, chunk)
        elif op == "reduce":
            self._compile_reduce(expr, chunk)
        elif op == "seq":
            self._compile_seq(expr, chunk)
        elif op == "if":
            self._compile_if(expr, chunk)
        elif op == "loop":
            self._compile_loop(expr, chunk)
        elif op == "while":
            self._compile_while(expr, chunk)
        elif op == "match":
            self._compile_match(expr, chunk)
        elif op == "let":
            self._compile_let(expr, chunk)
        elif op == "get":
            self._compile_get(expr, chunk)
        elif op == "set":
            self._compile_set(expr, chunk)
        elif op == "struct":
            self._compile_struct(expr, chunk)
        elif op == "tell":
            self._compile_tell(expr, chunk)
        elif op == "ask":
            self._compile_ask(expr, chunk)
        elif op == "delegate":
            self._compile_delegate(expr, chunk)
        elif op == "broadcast":
            self._compile_broadcast(expr, chunk)
        elif op == "signal":
            self._compile_signal(expr, chunk)
        elif op == "await":
            self._compile_await(expr, chunk)
        elif op == "branch":
            self._compile_branch(expr, chunk)
        elif op == "fork":
            self._compile_fork(expr, chunk)
        elif op == "merge":
            self._compile_merge(expr, chunk)
        elif op == "co_iterate":
            self._compile_co_iterate(expr, chunk)
        elif op == "trust":
            self._compile_trust(expr, chunk)
        elif op == "confidence":
            self._compile_confidence(expr, chunk)
        elif op == "literal":
            chunk.emit_push(expr.get("value"), source="literal")
        elif op == "yield":
            # Yield is a no-op in compiled mode — just keeps value on stack
            pass
        elif op == "eval":
            inner = expr.get("expr", {})
            if isinstance(inner, dict) and "op" in inner:
                self._compile_expr(inner, chunk)
            else:
                chunk.emit_push(inner, source="eval")
        else:
            # Unknown — emit as NOP for forward compatibility
            chunk.emit(BcOp.NOP.value, f"unknown:{op}", source=f"unknown_op:{op}")

    def _compile_value(self, value: Any, chunk: BytecodeChunk) -> None:
        """Compile a value (literal or expression) to bytecode."""
        if isinstance(value, dict) and "op" in value:
            self._compile_expr(value, chunk)
        else:
            chunk.emit_push(value, source="literal")

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _compile_arithmetic(self, op: str, expr: Any, chunk: BytecodeChunk) -> None:
        args = expr.get("args", [])
        for a in args:
            self._compile_value(a, chunk)
        bc_op = {
            "add": BcOp.ADD, "sub": BcOp.SUB,
            "mul": BcOp.MUL, "div": BcOp.DIV,
            "mod": BcOp.MOD,
        }
        chunk.emit(bc_op[op].value, source=f"arith:{op}")

    def _compile_comparison(self, op: str, expr: Any, chunk: BytecodeChunk) -> None:
        args = expr.get("args", [])
        for a in args[:2]:
            self._compile_value(a, chunk)
        bc_op = {
            "eq": BcOp.EQ, "neq": BcOp.NEQ,
            "lt": BcOp.LT, "lte": BcOp.LTE,
            "gt": BcOp.GT, "gte": BcOp.GTE,
        }
        chunk.emit(bc_op[op].value, source=f"cmp:{op}")

    def _compile_logic(self, op: str, expr: Any, chunk: BytecodeChunk) -> None:
        args = expr.get("args", [])
        if op == "not":
            self._compile_value(args[0], chunk)
            chunk.emit(BcOp.NOT.value, source="logic:not")
        else:
            for a in args:
                self._compile_value(a, chunk)
            bc_op = {"and": BcOp.AND, "or": BcOp.OR, "xor": BcOp.NEQ}
            chunk.emit(bc_op[op].value, source=f"logic:{op}")

    def _compile_concat(self, expr: Any, chunk: BytecodeChunk) -> None:
        args = expr.get("args", [])
        for a in args:
            self._compile_value(a, chunk)
        chunk.emit(BcOp.CONCAT.value, source="concat")

    def _compile_length(self, expr: Any, chunk: BytecodeChunk) -> None:
        self._compile_value(expr.get("value"), chunk)
        chunk.emit(BcOp.LENGTH.value, source="length")

    def _compile_at(self, expr: Any, chunk: BytecodeChunk) -> None:
        self._compile_value(expr.get("collection", expr.get("value")), chunk)
        self._compile_value(expr.get("index", 0), chunk)
        chunk.emit(BcOp.AT.value, source="at")

    def _compile_collect(self, expr: Any, chunk: BytecodeChunk) -> None:
        items = expr.get("items", [])
        chunk.emit_push(len(items), source="collect:count")
        for item in items:
            self._compile_value(item, chunk)
        chunk.emit(BcOp.COLLECT.value, source="collect")

    def _compile_reduce(self, expr: Any, chunk: BytecodeChunk) -> None:
        self._compile_value(expr.get("initial", 0), chunk)
        self._compile_value(expr.get("collection", []), chunk)
        chunk.emit(BcOp.REDUCE.value, source="reduce")

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _compile_seq(self, expr: Any, chunk: BytecodeChunk) -> None:
        body = expr.get("body", [])
        for step in body:
            self._compile_value(step, chunk)

    def _compile_if(self, expr: Any, chunk: BytecodeChunk) -> None:
        else_label = self._next_label("else")
        end_label = self._next_label("endif")

        # Compile condition
        self._compile_value(expr.get("condition", expr.get("cond", False)), chunk)
        jump_idx = chunk.emit(BcOp.JUMP_IF_NOT.value, else_label, source="if:cond")

        # Then branch
        then_body = expr.get("then", expr.get("body", []))
        self._compile_body(then_body, chunk)
        chunk.emit(BcOp.JUMP.value, end_label, source="if:skip_else")

        # Else branch
        chunk.emit_label(else_label)
        else_body = expr.get("else", [])
        self._compile_body(else_body, chunk)

        chunk.emit_label(end_label)

    def _compile_loop(self, expr: Any, chunk: BytecodeChunk) -> None:
        loop_label = self._next_label("loop")
        end_label = self._next_label("loop_end")
        var_name = expr.get("var", "item")
        body = expr.get("body", [])

        # Loop over collection or count
        collection = expr.get("over", None)
        times = expr.get("times", expr.get("count", None))

        if collection is not None:
            self._compile_value(collection, chunk)
            chunk.emit(BcOp.LOAD.value, f"_iter:{var_name}", source="loop:init_iter")
            chunk.emit_label(loop_label)
            # Check if iterator exhausted
            chunk.emit(BcOp.JUMP_IF_NOT.value, end_label, source="loop:check")
            chunk.emit(BcOp.LOAD.value, var_name, source=f"loop:load_{var_name}")
            self._compile_body(body, chunk)
            chunk.emit(BcOp.JUMP.value, loop_label, source="loop:next")
        else:
            count = int(times) if times is not None else 0
            chunk.emit_push(0, source="loop:counter_init")
            chunk.emit(BcOp.STORE.value, f"_loop_counter", source="loop:store_counter")
            chunk.emit_label(loop_label)
            # Check counter
            chunk.emit(BcOp.LOAD.value, f"_loop_counter", source="loop:load_counter")
            chunk.emit_push(count, source="loop:limit")
            chunk.emit(BcOp.LT.value, source="loop:check")
            chunk.emit(BcOp.JUMP_IF_NOT.value, end_label, source="loop:end_check")
            # Load counter as var
            chunk.emit(BcOp.LOAD.value, f"_loop_counter", source=f"loop:load_{var_name}")
            chunk.emit(BcOp.STORE.value, var_name, source=f"loop:store_{var_name}")
            self._compile_body(body, chunk)
            # Increment counter
            chunk.emit(BcOp.LOAD.value, f"_loop_counter", source="loop:inc_load")
            chunk.emit_push(1, source="loop:inc_one")
            chunk.emit(BcOp.ADD.value, source="loop:inc")
            chunk.emit(BcOp.STORE.value, f"_loop_counter", source="loop:inc_store")
            chunk.emit(BcOp.JUMP.value, loop_label, source="loop:continue")

        chunk.emit_label(end_label)

    def _compile_while(self, expr: Any, chunk: BytecodeChunk) -> None:
        cond_label = self._next_label("while")
        end_label = self._next_label("while_end")
        body = expr.get("body", [])

        chunk.emit_label(cond_label)
        self._compile_value(expr.get("condition", expr.get("cond")), chunk)
        chunk.emit(BcOp.JUMP_IF_NOT.value, end_label, source="while:check")
        self._compile_body(body, chunk)
        chunk.emit(BcOp.JUMP.value, cond_label, source="while:continue")
        chunk.emit_label(end_label)

    def _compile_match(self, expr: Any, chunk: BytecodeChunk) -> None:
        cases = expr.get("cases", [])
        end_label = self._next_label("match_end")

        self._compile_value(expr.get("value"), chunk)

        for i, case in enumerate(cases):
            if isinstance(case, dict):
                pattern = case.get("pattern")
                next_label = self._next_label(f"case_{i}_next")

                if pattern is not None and pattern != "_":
                    chunk.emit(BcOp.DUP.value, source=f"match:dup_{i}")
                    chunk.emit_push(pattern, source=f"match:pattern_{i}")
                    chunk.emit(BcOp.EQ.value, source=f"match:eq_{i}")
                    chunk.emit(BcOp.JUMP_IF_NOT.value, next_label, source=f"match:skip_{i}")
                    chunk.emit(BcOp.POP.value, source=f"match:pop_{i}")
                    self._compile_body(case.get("body", []), chunk)
                    chunk.emit(BcOp.JUMP.value, end_label, source=f"match:done_{i}")
                    chunk.emit_label(next_label)
                else:
                    # Wildcard — always matches
                    chunk.emit(BcOp.POP.value, source=f"match:pop_wild_{i}")
                    self._compile_body(case.get("body", []), chunk)
                    chunk.emit(BcOp.JUMP.value, end_label, source=f"match:wild_done_{i}")

        # Pop the value if no case matched
        chunk.emit(BcOp.POP.value, source="match:cleanup")
        chunk.emit_label(end_label)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _compile_let(self, expr: Any, chunk: BytecodeChunk) -> None:
        name = expr.get("name", "")
        self._compile_value(expr.get("value"), chunk)
        chunk.emit(BcOp.STORE.value, name, source=f"let:{name}")

    def _compile_get(self, expr: Any, chunk: BytecodeChunk) -> None:
        name = expr.get("name", "")
        chunk.emit(BcOp.LOAD.value, name, source=f"get:{name}")

    def _compile_set(self, expr: Any, chunk: BytecodeChunk) -> None:
        name = expr.get("name", "")
        self._compile_value(expr.get("value"), chunk)
        chunk.emit(BcOp.STORE.value, name, source=f"set:{name}")

    def _compile_struct(self, expr: Any, chunk: BytecodeChunk) -> None:
        fields = expr.get("fields", {})
        chunk.emit_push(len(fields), source="struct:count")
        for key, val in fields.items():
            chunk.emit_push(key, source=f"struct:key:{key}")
            self._compile_value(val, chunk)
        chunk.emit(BcOp.STRUCT.value, source="struct")

    # ------------------------------------------------------------------
    # Agent communication
    # ------------------------------------------------------------------

    def _compile_tell(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("to", ""), source="tell:to")
        self._compile_value(expr.get("message", expr.get("payload")), chunk)
        chunk.emit(BcOp.TELL.value, source="tell")

    def _compile_ask(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("from", expr.get("to", "")), source="ask:from")
        self._compile_value(expr.get("question", expr.get("payload")), chunk)
        chunk.emit_push(expr.get("timeout_ms", 5000), source="ask:timeout")
        chunk.emit(BcOp.ASK.value, source="ask")

    def _compile_delegate(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("to", ""), source="delegate:to")
        self._compile_value(expr.get("task"), chunk)
        chunk.emit(BcOp.DELEGATE.value, source="delegate")

    def _compile_broadcast(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("scope", "fleet"), source="broadcast:scope")
        self._compile_value(expr.get("message", expr.get("payload")), chunk)
        chunk.emit(BcOp.BROADCAST.value, source="broadcast")

    def _compile_signal(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("name", ""), source="signal:name")
        self._compile_value(expr.get("payload"), chunk)
        chunk.emit(BcOp.SIGNAL.value, source="signal")

    def _compile_await(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("signal", ""), source="await:signal")
        chunk.emit_push(expr.get("timeout_ms", 30000), source="await:timeout")
        chunk.emit(BcOp.AWAIT.value, source="await")

    # ------------------------------------------------------------------
    # Agent operations
    # ------------------------------------------------------------------

    def _compile_branch(self, expr: Any, chunk: BytecodeChunk) -> None:
        branch_id = expr.get("id", "branch")
        branches = expr.get("branches", [])
        merge_raw = expr.get("merge", {})

        from flux_a2a.schema import BranchDef
        branch_defs = []
        for b in branches:
            if isinstance(b, dict):
                branch_defs.append(BranchDef.from_dict(b))
            elif isinstance(b, BranchDef):
                branch_defs.append(b)

        end_label = self._next_label("branch_end")

        # Emit BRANCH with count and branch IDs
        chunk.emit_push(branch_id, source="branch:id")
        chunk.emit_push(len(branch_defs), source="branch:count")

        branch_labels = []
        for i, bd in enumerate(branch_defs):
            bl = self._next_label(f"branch_{bd.label}")
            branch_labels.append(bl)
            chunk.emit_push(bd.label, source=f"branch:label_{i}")
            chunk.emit_push(bd.weight, source=f"branch:weight_{i}")

        chunk.emit(BcOp.BRANCH.value, end_label, source="branch:start")

        # Emit each branch body
        for i, bd in enumerate(branch_defs):
            chunk.emit_label(branch_labels[i])
            self._compile_body(bd.body, chunk)
            chunk.emit(BcOp.JUMP.value, end_label, source=f"branch:done_{i}")

        chunk.emit_label(end_label)

    def _compile_fork(self, expr: Any, chunk: BytecodeChunk) -> None:
        fork_id = expr.get("id", "fork")
        agent_raw = expr.get("agent", {})
        body = expr.get("body", [])

        from flux_a2a.schema import Agent
        if isinstance(agent_raw, dict):
            agent = Agent(**agent_raw)
        else:
            agent = Agent(id="child")

        chunk.emit_push(fork_id, source="fork:id")
        chunk.emit_push(agent.id, source="fork:agent")
        chunk.emit_push(agent.role, source="fork:role")
        chunk.emit(BcOp.FORK.value, source="fork")

        # Fork body runs in child context
        self._compile_body(body, chunk)

    def _compile_merge(self, expr: Any, chunk: BytecodeChunk) -> None:
        strategy = expr.get("strategy", "weighted_confidence")
        chunk.emit_push(strategy, source="merge:strategy")
        chunk.emit(BcOp.MERGE.value, source="merge")

    def _compile_co_iterate(self, expr: Any, chunk: BytecodeChunk) -> None:
        co_id = expr.get("id", "co_iterate")
        program = expr.get("program", {})

        chunk.emit_push(co_id, source="co_iterate:id")

        # Emit program body as a callable section
        if isinstance(program, dict) and "body" in program:
            self._compile_body(program.get("body", []), chunk)
        elif isinstance(program, list):
            self._compile_body(program, chunk)
        elif isinstance(program, dict) and "op" in program:
            self._compile_expr(program, chunk)

        chunk.emit(BcOp.CO_ITERATE.value, source="co_iterate")

    def _compile_trust(self, expr: Any, chunk: BytecodeChunk) -> None:
        chunk.emit_push(expr.get("agent", ""), source="trust:agent")
        chunk.emit_push(expr.get("level", 0.5), source="trust:level")
        chunk.emit(BcOp.TRUST.value, source="trust")

    def _compile_confidence(self, expr: Any, chunk: BytecodeChunk) -> None:
        level = expr.get("level", expr.get("value", 1.0))
        chunk.emit_push(float(level), source="confidence:level")
        chunk.emit(BcOp.CONFIDENCE.value, source="confidence")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compile_body(self, body: list[Any], chunk: BytecodeChunk) -> None:
        """Compile a list of expressions/statements."""
        for step in body:
            if isinstance(step, dict) and "op" in step:
                self._compile_expr(step, chunk)
            elif isinstance(step, dict):
                chunk.emit_push(step, source="body:dict")
            else:
                chunk.emit_push(step, source="body:literal")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """Bytecode optimization passes."""

    def __init__(self, passes: Optional[list[str]] = None) -> None:
        self.passes = passes or ["dead_branch_elim", "cse", "constant_fold"]

    def optimize(self, chunk: BytecodeChunk) -> BytecodeChunk:
        """Run all configured optimization passes on a bytecode chunk."""
        if "dead_branch_elim" in self.passes:
            self._dead_branch_elimination(chunk)
        if "cse" in self.passes:
            self._common_subexpression_elimination(chunk)
        if "constant_fold" in self.passes:
            self._constant_folding(chunk)
        return chunk

    def _dead_branch_elimination(self, chunk: BytecodeChunk) -> None:
        """Remove branches whose conditions are constant."""
        # Find PUSH true/false followed by JUMP_IF / JUMP_IF_NOT
        i = 0
        while i < len(chunk.instructions) - 1:
            instr = chunk.instructions[i]
            next_instr = chunk.instructions[i + 1]
            if (
                instr[0] == BcOp.PUSH.value
                and isinstance(instr[1], int)
                and instr[1] < len(chunk.constants)
            ):
                const = chunk.constants[instr[1]]
                if const is True and next_instr[0] == BcOp.JUMP_IF.value:
                    # Always jump — remove the condition check
                    chunk.instructions[i] = [BcOp.JUMP.value, next_instr[1]]
                    chunk.instructions.pop(i + 1)
                    continue
                elif const is False and next_instr[0] == BcOp.JUMP_IF.value:
                    # Never jump — remove both
                    chunk.instructions.pop(i)
                    chunk.instructions.pop(i)
                    continue
                elif const is True and next_instr[0] == BcOp.JUMP_IF_NOT.value:
                    # Never jump — remove both
                    chunk.instructions.pop(i)
                    chunk.instructions.pop(i)
                    continue
                elif const is False and next_instr[0] == BcOp.JUMP_IF_NOT.value:
                    # Always jump
                    chunk.instructions[i] = [BcOp.JUMP.value, next_instr[1]]
                    chunk.instructions.pop(i + 1)
                    continue
            i += 1

    def _common_subexpression_elimination(self, chunk: BytecodeChunk) -> None:
        """Remove duplicate consecutive instructions (simple CSE)."""
        seen: set[tuple] = set()
        new_instrs: list[list[Any]] = []
        for instr in chunk.instructions:
            key = tuple(instr)
            if key not in seen or instr[0] in (BcOp.LOAD.value, BcOp.STORE.value, BcOp.LABEL.value):
                new_instrs.append(instr)
                seen.add(key)
        chunk.instructions = new_instrs

    def _constant_folding(self, chunk: BytecodeChunk) -> None:
        """Fold constant arithmetic: PUSH a, PUSH b, ADD → PUSH result."""
        arith_ops = {BcOp.ADD.value, BcOp.SUB.value, BcOp.MUL.value, BcOp.DIV.value, BcOp.MOD.value}
        import operator as _op

        op_funcs = {
            BcOp.ADD.value: _op.add,
            BcOp.SUB.value: _op.sub,
            BcOp.MUL.value: _op.mul,
            BcOp.DIV.value: _op.truediv,
            BcOp.MOD.value: _op.mod,
        }

        i = 0
        while i < len(chunk.instructions) - 2:
            instr_a = chunk.instructions[i]
            instr_b = chunk.instructions[i + 1]
            instr_op = chunk.instructions[i + 2]

            if (
                instr_a[0] == BcOp.PUSH.value
                and instr_b[0] == BcOp.PUSH.value
                and instr_op[0] in arith_ops
            ):
                try:
                    a_val = chunk.constants[int(instr_a[1])]
                    b_val = chunk.constants[int(instr_b[1])]
                    if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                        result = op_funcs[instr_op[0]](a_val, b_val)
                        if instr_op[0] == BcOp.DIV.value and b_val == 0:
                            i += 1
                            continue
                        if result not in chunk.constants:
                            chunk.constants.append(result)
                        const_idx = chunk.constants.index(result)
                        chunk.instructions[i] = [BcOp.PUSH.value, const_idx]
                        chunk.instructions.pop(i + 1)
                        chunk.instructions.pop(i + 1)
                        continue
                except (IndexError, TypeError, ZeroDivisionError):
                    pass
            i += 1


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def compile_program(program: Any, optimizations: Optional[list[str]] = None) -> BytecodeChunk:
    """Compile a Signal program to FLUX bytecode."""
    compiler = Compiler(optimizations=optimizations)
    chunk = compiler.compile_program(program)
    if optimizations:
        optimizer = Optimizer(passes=optimizations)
        optimizer.optimize(chunk)
    return chunk
