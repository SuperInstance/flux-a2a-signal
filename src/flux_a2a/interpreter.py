"""
FLUX-A2A Interpreter — Evaluate JSON programs directly.

The interpreter walks the JSON AST (there is no separate parse step — JSON IS the AST).
It supports:
  - Expression evaluation (arithmetic, logic, string, agent ops)
  - Branching (parallel branch evaluation with merge)
  - Forking (child agent spawning with state inheritance)
  - Co-iteration (multi-agent shared program traversal)
  - Language tag routing
  - Confidence propagation
"""

from __future__ import annotations

import math
from typing import Any, Optional

from flux_a2a.schema import (
    Agent,
    BranchDef,
    ConfidenceScore,
    Expression,
    LanguageTag,
    MergePolicy,
    MergePolicyType,
    Message,
    MessageType,
    MessageBus,
    Program,
    Result,
)
from flux_a2a.fork_manager import BranchManager, ForkManager
from flux_a2a.co_iteration import CoIterationEngine, SharedProgram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ensure_number(v: Any) -> float:
    if _is_number(v):
        return float(v)
    raise TypeError(f"Expected number, got {type(v).__name__}: {v!r}")


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """Walks a Signal Program's AST and evaluates expressions.

    Parameters
    ----------
    agent_id:
        Identity of the agent running this interpreter instance.
    message_bus:
        Shared message bus for A2A communication.
    branch_manager:
        Manages parallel branch execution.
    fork_manager:
        Manages child agent forking.
    """

    def __init__(
        self,
        agent_id: str = "interpreter",
        message_bus: Optional[MessageBus] = None,
        branch_manager: Optional[BranchManager] = None,
        fork_manager: Optional[ForkManager] = None,
    ) -> None:
        self.agent_id = agent_id
        self.message_bus = message_bus or MessageBus()
        self.branch_manager = branch_manager or BranchManager()
        self.fork_manager = fork_manager or ForkManager()
        self.co_engine = CoIterationEngine()

        # Execution state — variable bindings
        self.state: dict[str, Any] = {}
        # Registered agents (from program header)
        self.agents: dict[str, Agent] = {}
        # Confidence for current scope
        self._scope_confidence = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(self, program: Program) -> Result:
        """Execute a full Signal program and return the final result."""
        # Register agents
        for agent in program.agents:
            self.agents[agent.id] = agent

        # Execute body expressions sequentially
        last_result: Optional[Result] = None
        for expr in program.body:
            last_result = self.evaluate(expr)

        if last_result is None:
            last_result = Result(value=None, source="empty_program", agent=self.agent_id)

        return last_result

    def evaluate(self, expr: Expression) -> Result:
        """Evaluate a single expression.  This is the main dispatch."""

        # Update scope confidence if expression specifies one
        if expr.confidence < self._scope_confidence:
            self._scope_confidence = expr.confidence

        op = expr.op

        # --- Arithmetic ---
        if op in ("add", "sub", "mul", "div", "mod"):
            return self._eval_arithmetic(op, expr)

        # --- Comparison ---
        if op in ("eq", "neq", "lt", "lte", "gt", "gte"):
            return self._eval_comparison(op, expr)

        # --- Logic ---
        if op in ("and", "or", "not", "xor"):
            return self._eval_logic(op, expr)

        # --- String ---
        if op == "concat":
            return self._eval_concat(expr)
        if op == "length":
            return self._eval_length(expr)

        # --- Collection ---
        if op == "at":
            return self._eval_at(expr)
        if op == "collect":
            return self._eval_collect(expr)
        if op == "reduce":
            return self._eval_reduce(expr)

        # --- Control flow ---
        if op == "seq":
            return self._eval_seq(expr)
        if op == "if":
            return self._eval_if(expr)
        if op == "loop":
            return self._eval_loop(expr)
        if op == "while":
            return self._eval_while(expr)
        if op == "match":
            return self._eval_match(expr)
        if op == "yield":
            return self._eval_yield(expr)
        if op == "await":
            return self._eval_await(expr)

        # --- Variable bindings ---
        if op == "let":
            return self._eval_let(expr)
        if op == "get":
            return self._eval_get(expr)
        if op == "set":
            return self._eval_set(expr)

        # --- Data ---
        if op == "struct":
            return self._eval_struct(expr)

        # --- Agent communication ---
        if op == "tell":
            return self._eval_tell(expr)
        if op == "ask":
            return self._eval_ask(expr)
        if op == "delegate":
            return self._eval_delegate(expr)
        if op == "broadcast":
            return self._eval_broadcast(expr)

        # --- Agent operations ---
        if op == "branch":
            return self._eval_branch(expr)
        if op == "fork":
            return self._eval_fork(expr)
        if op == "merge":
            return self._eval_merge(expr)
        if op == "co_iterate":
            return self._eval_co_iterate(expr)

        # --- Signals & trust ---
        if op == "signal":
            return self._eval_signal(expr)
        if op == "trust":
            return self._eval_trust(expr)
        if op == "confidence":
            return self._eval_confidence_op(expr)

        # --- Literals / pass-through ---
        if op == "literal":
            return Result(
                value=expr.get("value"),
                confidence=self._scope_confidence,
                source="literal",
                agent=self.agent_id,
            )

        # --- eval wrapper ---
        if op == "eval":
            inner = expr.get("expr", {})
            if isinstance(inner, dict):
                return self.evaluate(Expression.from_dict(inner))
            return Result(value=inner, confidence=self._scope_confidence, source="eval")

        # Unknown opcode
        return Result(
            value=None,
            confidence=0.0,
            source=f"unknown_op:{op}",
            agent=self.agent_id,
            error=f"Unknown opcode: {op}",
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _eval_arithmetic(self, op: str, expr: Expression) -> Result:
        args = expr.get("args", [])
        if op == "add" and len(args) == 2:
            # Also support named operands for language-tagged expressions
            a_val = args[0] if not isinstance(args[0], dict) else self._try_eval(args[0])
            b_val = args[1] if not isinstance(args[1], dict) else self._try_eval(args[1])
            return self._arith_result(a_val + b_val, self._scope_confidence)
        if op == "sub" and len(args) == 2:
            a = _ensure_number(args[0] if not isinstance(args[0], dict) else self._try_eval(args[0]))
            b = _ensure_number(args[1] if not isinstance(args[1], dict) else self._try_eval(args[1]))
            return self._arith_result(a - b, self._scope_confidence)
        if op == "mul" and len(args) == 2:
            a = _ensure_number(args[0] if not isinstance(args[0], dict) else self._try_eval(args[0]))
            b = _ensure_number(args[1] if not isinstance(args[1], dict) else self._try_eval(args[1]))
            return self._arith_result(a * b, self._scope_confidence)
        if op == "div" and len(args) == 2:
            a = _ensure_number(args[0] if not isinstance(args[0], dict) else self._try_eval(args[0]))
            b = _ensure_number(args[1] if not isinstance(args[1], dict) else self._try_eval(args[1]))
            if b == 0:
                return Result(value=None, confidence=0.0, source="div", agent=self.agent_id, error="Division by zero")
            return self._arith_result(a / b, self._scope_confidence)
        if op == "mod" and len(args) == 2:
            a = _ensure_number(args[0] if not isinstance(args[0], dict) else self._try_eval(args[0]))
            b = _ensure_number(args[1] if not isinstance(args[1], dict) else self._try_eval(args[1]))
            return self._arith_result(a % b, self._scope_confidence)

        # n-ary add/sub/mul
        values = []
        min_conf = self._scope_confidence
        for a in args:
            if isinstance(a, dict) and "op" in a:
                r = self.evaluate(Expression.from_dict(a))
                values.append(r.value)
                min_conf = min(min_conf, r.confidence)
            else:
                values.append(a)
        if op == "add":
            return self._arith_result(sum(values), min_conf)
        if op == "mul":
            result = 1
            for v in values:
                result *= v
            return self._arith_result(result, min_conf)

        return Result(
            value=None, confidence=0.0, source=f"arith:{op}", agent=self.agent_id,
            error=f"Cannot evaluate {op} with args {args}",
        )

    def _arith_result(self, value: Any, conf: float) -> Result:
        """Create an arithmetic result with confidence propagation."""
        return Result(value=value, confidence=conf, source="arithmetic", agent=self.agent_id)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _eval_comparison(self, op: str, expr: Expression) -> Result:
        args = expr.get("args", [])
        if len(args) < 2:
            return Result(value=None, confidence=0.0, source=f"cmp:{op}", agent=self.agent_id, error="Need 2 args")
        a = self._try_eval(args[0])
        b = self._try_eval(args[1])
        ops = {
            "eq": lambda: a == b,
            "neq": lambda: a != b,
            "lt": lambda: a < b,
            "lte": lambda: a <= b,
            "gt": lambda: a > b,
            "gte": lambda: a >= b,
        }
        try:
            value = ops[op]()
        except TypeError:
            value = False
        return Result(value=value, confidence=self._scope_confidence, source=f"cmp:{op}", agent=self.agent_id)

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------

    def _eval_logic(self, op: str, expr: Expression) -> Result:
        args = expr.get("args", [])
        if op == "not":
            a = self._try_eval(args[0])
            return Result(value=not a, confidence=self._scope_confidence, source="logic:not", agent=self.agent_id)
        if op == "and":
            value = all(self._try_eval(a) for a in args)
            return Result(value=value, confidence=self._scope_confidence, source="logic:and", agent=self.agent_id)
        if op == "or":
            value = any(self._try_eval(a) for a in args)
            return Result(value=value, confidence=self._scope_confidence, source="logic:or", agent=self.agent_id)
        if op == "xor" and len(args) == 2:
            a = bool(self._try_eval(args[0]))
            b = bool(self._try_eval(args[1]))
            return Result(value=a != b, confidence=self._scope_confidence, source="logic:xor", agent=self.agent_id)
        return Result(value=None, confidence=0.0, source=f"logic:{op}", agent=self.agent_id, error=f"Unhandled: {op}")

    # ------------------------------------------------------------------
    # String & Collection
    # ------------------------------------------------------------------

    def _eval_concat(self, expr: Expression) -> Result:
        args = expr.get("args", [])
        parts = [str(self._try_eval(a)) for a in args]
        return Result(value="".join(parts), confidence=self._scope_confidence, source="concat", agent=self.agent_id)

    def _eval_length(self, expr: Expression) -> Result:
        val = self._try_eval(expr.get("value"))
        return Result(value=len(val), confidence=self._scope_confidence, source="length", agent=self.agent_id)

    def _eval_at(self, expr: Expression) -> Result:
        collection = self._try_eval(expr.get("collection", expr.get("value", [])))
        index = int(self._try_eval(expr.get("index", 0)))
        return Result(value=collection[index], confidence=self._scope_confidence, source="at", agent=self.agent_id)

    def _eval_collect(self, expr: Expression) -> Result:
        items = expr.get("items", [])
        values = []
        min_conf = self._scope_confidence
        for item in items:
            if isinstance(item, dict) and "op" in item:
                r = self.evaluate(Expression.from_dict(item))
                values.append(r.value)
                min_conf = min(min_conf, r.confidence)
            else:
                values.append(item)
        return Result(value=values, confidence=min_conf, source="collect", agent=self.agent_id)

    def _eval_reduce(self, expr: Expression) -> Result:
        collection = self._try_eval(expr.get("collection", []))
        initial = self._try_eval(expr.get("initial", 0))
        # Use a simple sum as default reducer
        acc = initial
        for item in collection:
            if _is_number(acc) and _is_number(item):
                acc = acc + item
            else:
                acc = str(acc) + str(item)
        return Result(value=acc, confidence=self._scope_confidence, source="reduce", agent=self.agent_id)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _eval_seq(self, expr: Expression) -> Result:
        body = expr.get("body", [])
        last: Optional[Result] = None
        for step in body:
            if isinstance(step, dict) and "op" in step:
                last = self.evaluate(Expression.from_dict(step))
            else:
                last = Result(value=step, confidence=self._scope_confidence, source="seq", agent=self.agent_id)
        return last or Result(value=None, confidence=self._scope_confidence, source="seq", agent=self.agent_id)

    def _eval_if(self, expr: Expression) -> Result:
        condition = self._try_eval(expr.get("condition", expr.get("cond", False)))
        then_body = expr.get("then", expr.get("body", []))
        else_body = expr.get("else", [])

        if condition:
            if isinstance(then_body, list):
                return self._eval_body(then_body)
            if isinstance(then_body, dict) and "op" in then_body:
                return self.evaluate(Expression.from_dict(then_body))
            return Result(value=then_body, confidence=self._scope_confidence, source="if:then", agent=self.agent_id)
        else:
            if isinstance(else_body, list):
                return self._eval_body(else_body)
            if isinstance(else_body, dict) and "op" in else_body:
                return self.evaluate(Expression.from_dict(else_body))
            if else_body:
                return Result(value=else_body, confidence=self._scope_confidence, source="if:else", agent=self.agent_id)
            return Result(value=None, confidence=self._scope_confidence, source="if:false", agent=self.agent_id)

    def _eval_loop(self, expr: Expression) -> Result:
        body = expr.get("body", [])
        times = int(self._try_eval(expr.get("times", expr.get("count", 0))))
        collection = expr.get("over", None)
        var_name = expr.get("var", "item")

        results: list[Result] = []
        if collection is not None:
            items = self._try_eval(collection)
            for item in items:
                self.state[var_name] = item
                r = self._eval_body(body)
                results.append(r)
        else:
            for i in range(times):
                self.state[var_name] = i
                r = self._eval_body(body)
                results.append(r)

        return Result(
            value=[r.value for r in results],
            confidence=min((r.confidence for r in results), default=self._scope_confidence),
            source="loop",
            agent=self.agent_id,
            children=results,
        )

    def _eval_while(self, expr: Expression) -> Result:
        cond_expr = expr.get("condition", expr.get("cond"))
        body = expr.get("body", [])
        max_iterations = int(expr.get("max_iterations", 1000))
        results: list[Result] = []
        iteration = 0

        while iteration < max_iterations:
            condition = self._try_eval(cond_expr)
            if not condition:
                break
            r = self._eval_body(body)
            results.append(r)
            iteration += 1

        return Result(
            value=[r.value for r in results],
            confidence=min((r.confidence for r in results), default=self._scope_confidence),
            source="while",
            agent=self.agent_id,
            children=results,
        )

    def _eval_match(self, expr: Expression) -> Result:
        value = self._try_eval(expr.get("value"))
        cases = expr.get("cases", [])
        default = expr.get("default", None)

        for case in cases:
            if isinstance(case, dict):
                pattern = case.get("pattern")
                if pattern is None or pattern == value or pattern == "_":
                    body = case.get("body", [])
                    return self._eval_body(body)

        if default is not None:
            if isinstance(default, list):
                return self._eval_body(default)
            return Result(value=default, confidence=self._scope_confidence, source="match:default", agent=self.agent_id)

        return Result(value=None, confidence=0.0, source="match:miss", agent=self.agent_id, error="No matching case")

    def _eval_yield(self, expr: Expression) -> Result:
        value = self._try_eval(expr.get("value"))
        return Result(value=value, confidence=self._scope_confidence, source="yield", agent=self.agent_id)

    def _eval_await(self, expr: Expression) -> Result:
        signal_name = expr.get("signal", "")
        timeout_ms = int(expr.get("timeout_ms", 30000))

        # In this single-threaded implementation, await is a no-op that
        # returns immediately.  Production impls would block on the signal.
        messages = self.message_bus.receive(self.agent_id)
        if messages:
            last = messages[-1]
            return Result(
                value=last.payload,
                confidence=last.confidence,
                source="await",
                agent=self.agent_id,
                meta={"signal": signal_name, "messages_received": len(messages)},
            )
        return Result(
            value=None,
            confidence=self._scope_confidence,
            source="await",
            agent=self.agent_id,
            meta={"signal": signal_name, "status": "timeout"},
        )

    # ------------------------------------------------------------------
    # Variable bindings
    # ------------------------------------------------------------------

    def _eval_let(self, expr: Expression) -> Result:
        name = expr.get("name", "")
        raw_value = expr.get("value")
        if isinstance(raw_value, dict) and "op" in raw_value:
            r = self.evaluate(Expression.from_dict(raw_value))
            self.state[name] = r.value
            return Result(value=r.value, confidence=r.confidence, source="let", agent=self.agent_id)
        self.state[name] = raw_value
        return Result(value=raw_value, confidence=self._scope_confidence, source="let", agent=self.agent_id)

    def _eval_get(self, expr: Expression) -> Result:
        name = expr.get("name", "")
        if name in self.state:
            return Result(value=self.state[name], confidence=self._scope_confidence, source="get", agent=self.agent_id)
        return Result(value=None, confidence=0.0, source="get", agent=self.agent_id, error=f"Unbound: {name}")

    def _eval_set(self, expr: Expression) -> Result:
        name = expr.get("name", "")
        raw_value = expr.get("value")
        if isinstance(raw_value, dict) and "op" in raw_value:
            r = self.evaluate(Expression.from_dict(raw_value))
            self.state[name] = r.value
            return Result(value=r.value, confidence=r.confidence, source="set", agent=self.agent_id)
        self.state[name] = raw_value
        return Result(value=raw_value, confidence=self._scope_confidence, source="set", agent=self.agent_id)

    def _eval_struct(self, expr: Expression) -> Result:
        fields = expr.get("fields", {})
        result: dict[str, Any] = {}
        min_conf = self._scope_confidence
        for key, val in fields.items():
            if isinstance(val, dict) and "op" in val:
                r = self.evaluate(Expression.from_dict(val))
                result[key] = r.value
                min_conf = min(min_conf, r.confidence)
            else:
                result[key] = val
        return Result(value=result, confidence=min_conf, source="struct", agent=self.agent_id)

    # ------------------------------------------------------------------
    # Agent communication
    # ------------------------------------------------------------------

    def _eval_tell(self, expr: Expression) -> Result:
        target = expr.get("to", expr.get("to_agent", ""))
        raw_msg = expr.get("message", expr.get("payload", None))
        message = self._try_eval(raw_msg) if isinstance(raw_msg, (dict, list)) else raw_msg

        msg = Message(
            from_agent=self.agent_id,
            to_agent=target,
            msg_type=MessageType.TELL.value,
            payload=message,
            confidence=self._scope_confidence,
        )
        self.message_bus.send(msg)

        return Result(
            value=message,
            confidence=self._scope_confidence,
            source="tell",
            agent=self.agent_id,
            meta={"to": target, "msg_id": msg.id},
        )

    def _eval_ask(self, expr: Expression) -> Result:
        target = expr.get("from", expr.get("from_agent", expr.get("to", "")))
        question = expr.get("question", expr.get("payload", ""))

        msg = Message(
            from_agent=self.agent_id,
            to_agent=target,
            msg_type=MessageType.ASK.value,
            payload=question,
            confidence=self._scope_confidence,
        )
        self.message_bus.send(msg)

        # Look for any prior responses from that agent
        responses = self.message_bus.find_by_reply(msg.id)
        if responses:
            return Result(
                value=responses[-1].payload,
                confidence=responses[-1].confidence,
                source="ask",
                agent=self.agent_id,
            )

        return Result(
            value=None,
            confidence=self._scope_confidence,
            source="ask",
            agent=self.agent_id,
            meta={"target": target, "status": "pending"},
        )

    def _eval_delegate(self, expr: Expression) -> Result:
        target = expr.get("to", expr.get("to_agent", ""))
        task = expr.get("task", {})

        # Send delegation message
        msg = Message(
            from_agent=self.agent_id,
            to_agent=target,
            msg_type=MessageType.DELEGATE.value,
            payload=task,
            confidence=self._scope_confidence,
        )
        self.message_bus.send(msg)

        # Execute the task ourselves (simulated — real impl would await remote)
        if isinstance(task, dict) and "op" in task:
            task_result = self.evaluate(Expression.from_dict(task))
        elif isinstance(task, list):
            task_result = self._eval_body(
                [Expression.from_dict(t) if isinstance(t, dict) else t for t in task]
            )
        else:
            task_result = Result(value=task, confidence=self._scope_confidence, source="delegate", agent=self.agent_id)

        return Result(
            value=task_result.value,
            confidence=task_result.confidence,
            source="delegate",
            agent=self.agent_id,
            meta={"delegated_to": target, "task_result": task_result.to_dict()},
        )

    def _eval_broadcast(self, expr: Expression) -> Result:
        scope = expr.get("scope", "fleet")
        payload = expr.get("message", expr.get("payload", None))

        target_agents = list(self.agents.keys()) if scope == "fleet" else []
        if not target_agents:
            target_agents = [a.id for a in self.agents.values()]

        # Don't broadcast to self
        target_agents = [a for a in target_agents if a != self.agent_id]

        msg = Message(
            from_agent=self.agent_id,
            to_agent="broadcast",
            msg_type=MessageType.BROADCAST.value,
            payload=payload,
            confidence=self._scope_confidence,
            scope=scope,
        )
        self.message_bus.broadcast(msg, target_agents)

        return Result(
            value={"recipients": target_agents, "scope": scope},
            confidence=self._scope_confidence,
            source="broadcast",
            agent=self.agent_id,
        )

    # ------------------------------------------------------------------
    # Agent operations — Branching
    # ------------------------------------------------------------------

    def _eval_branch(self, expr: Expression) -> Result:
        branch_id = expr.get("id", "branch")
        branches_raw = expr.get("branches", [])
        merge_raw = expr.get("merge", {})

        # Parse branches
        branches: list[BranchDef] = []
        for b in branches_raw:
            if isinstance(b, BranchDef):
                branches.append(b)
            elif isinstance(b, dict):
                branches.append(BranchDef.from_dict(b))

        merge_policy = MergePolicy.from_dict(merge_raw) if isinstance(merge_raw, dict) else MergePolicy()

        # Register with branch manager
        bp = self.branch_manager.create_branch_point(branch_id, branches, merge_policy)

        # Execute each branch in sequence (simulated parallelism)
        branch_results: list[tuple[str, Result]] = []
        for branch_def in bp.branches:
            # Fork state for this branch
            saved_state = dict(self.state)
            try:
                result = self._eval_body(branch_def.body)
                branch_results.append((branch_def.label, result))
            finally:
                # Restore state (branches don't mutate parent state directly)
                self.state = saved_state

        # Merge results
        merged = self.branch_manager.merge(branch_id, branch_results)

        return Result(
            value=merged.value,
            confidence=merged.confidence,
            source="branch",
            agent=self.agent_id,
            branch=branch_id,
            children=[r for _, r in branch_results],
            meta={"branches_run": len(branch_results), "merge_strategy": merge_policy.strategy},
        )

    # ------------------------------------------------------------------
    # Agent operations — Forking
    # ------------------------------------------------------------------

    def _eval_fork(self, expr: Expression) -> Result:
        fork_id = expr.get("id", "fork")
        agent_raw = expr.get("agent", {})
        inherit_raw = expr.get("inherit", {})
        body_raw = expr.get("body", [])

        # Parse agent
        if isinstance(agent_raw, Agent):
            child_agent = agent_raw
        elif isinstance(agent_raw, dict):
            child_agent = Agent(**agent_raw)
        else:
            child_agent = Agent(id="child")

        # Parse inheritance
        if isinstance(inherit_raw, dict):
            inherit_state = inherit_raw.get("state", [])
            inherit_context = inherit_raw.get("context", True)
            inherit_trust = inherit_raw.get("trust_graph", False)
        else:
            inherit_state = []
            inherit_context = True
            inherit_trust = False

        # Parse body
        body = [Expression.from_dict(b) if isinstance(b, dict) else b for b in body_raw]

        # Create fork context
        inherited_state = {}
        for key in inherit_state:
            if key in self.state:
                inherited_state[key] = self.state[key]

        fork_ctx = self.fork_manager.create_fork(
            fork_id=fork_id,
            parent_id=self.agent_id,
            child_agent=child_agent,
            inherited_state=inherited_state,
            body=body,
        )

        # Execute fork body with inherited state
        saved_state = dict(self.state)
        self.state = dict(inherited_state)
        if inherit_context and self.agent_id:
            self.state["_fork_parent"] = self.agent_id
            self.state["_fork_id"] = fork_id

        try:
            result = self._eval_body(body)
        finally:
            self.state = saved_state

        # Collect result
        self.fork_manager.complete_fork(fork_id, result)

        return Result(
            value=result.value,
            confidence=result.confidence,
            source="fork",
            agent=self.agent_id,
            meta={
                "fork_id": fork_id,
                "child_agent": child_agent.id,
                "state_keys_inherited": list(inherited_state.keys()),
            },
        )

    # ------------------------------------------------------------------
    # Agent operations — Merge
    # ------------------------------------------------------------------

    def _eval_merge(self, expr: Expression) -> Result:
        strategy = expr.get("strategy", MergePolicyType.WEIGHTED_CONFIDENCE.value)
        results_raw = expr.get("results", expr.get("values", []))

        # If results are expressions, evaluate them first
        results: list[Result] = []
        for r in results_raw:
            if isinstance(r, Result):
                results.append(r)
            elif isinstance(r, dict) and "value" in r:
                results.append(Result(**{k: v for k, v in r.items() if k != "children"}))
            elif isinstance(r, dict) and "op" in r:
                results.append(self.evaluate(Expression.from_dict(r)))
            else:
                results.append(Result(value=r, confidence=self._scope_confidence))

        if strategy == MergePolicyType.BEST_CONFIDENCE.value:
            best = max(results, key=lambda r: r.confidence)
            return best
        elif strategy == MergePolicyType.FIRST_COMPLETE.value:
            return results[0] if results else Result(value=None, source="merge", agent=self.agent_id)
        elif strategy == MergePolicyType.CONSENSUS.value:
            values = [r.value for r in results]
            if all(v == values[0] for v in values):
                return Result(
                    value=values[0],
                    confidence=min(r.confidence for r in results),
                    source="merge:consensus",
                    agent=self.agent_id,
                )
            return Result(
                value=values,
                confidence=0.5,
                source="merge:consensus:disagree",
                agent=self.agent_id,
                meta={"agreement": False},
            )
        elif strategy == MergePolicyType.WEIGHTED_CONFIDENCE.value:
            weights = expr.get("weights", [1.0] * len(results))
            total_w = sum(weights)
            if total_w == 0:
                return Result(value=None, confidence=0.0, source="merge:weighted", agent=self.agent_id)
            weighted_conf = sum(w * r.confidence for w, r in zip(weights, results)) / total_w
            # Return first result's value as representative
            return Result(
                value=results[0].value if results else None,
                confidence=weighted_conf,
                source="merge:weighted_confidence",
                agent=self.agent_id,
                children=results,
            )
        else:
            # Default: last writer wins
            return results[-1] if results else Result(value=None, source="merge", agent=self.agent_id)

    # ------------------------------------------------------------------
    # Agent operations — Co-iteration
    # ------------------------------------------------------------------

    def _eval_co_iterate(self, expr: Expression) -> Result:
        co_id = expr.get("id", "co_iterate")
        program_raw = expr.get("program", {})
        agents_raw = expr.get("agents", [])
        cr_raw = expr.get("conflict_resolution", {})

        # Parse program
        if isinstance(program_raw, dict) and "body" in program_raw:
            body = [Expression.from_dict(e) if isinstance(e, dict) else e
                    for e in program_raw.get("body", [])]
        elif isinstance(program_raw, list):
            body = [Expression.from_dict(e) if isinstance(e, dict) else e for e in program_raw]
        elif isinstance(program_raw, dict) and "op" in program_raw:
            body = [Expression.from_dict(program_raw)]
        else:
            body = []

        # Parse agents
        co_agents = []
        for a in agents_raw:
            if isinstance(a, Agent):
                co_agents.append(a)
            elif isinstance(a, dict):
                co_agents.append(Agent(**a))

        # Create shared program
        shared = SharedProgram(
            id=co_id,
            body=body,
        )

        # Register agent cursors
        for agent in co_agents:
            shared.add_cursor(agent.id, agent)

        # Execute co-iteration
        engine = CoIterationEngine()
        result = engine.execute(shared)

        return Result(
            value=result.value,
            confidence=result.confidence,
            source="co_iterate",
            agent=self.agent_id,
            children=result.children,
            meta={
                "co_iterate_id": co_id,
                "agents": [a.id for a in co_agents],
                "steps": result.meta.get("steps", 0),
            },
        )

    # ------------------------------------------------------------------
    # Signals & Trust
    # ------------------------------------------------------------------

    def _eval_signal(self, expr: Expression) -> Result:
        name = expr.get("name", "")
        payload = self._try_eval(expr.get("payload", None))

        msg = Message(
            from_agent=self.agent_id,
            to_agent="broadcast",
            msg_type=MessageType.SIGNAL.value,
            payload={"signal_name": name, "data": payload},
            confidence=self._scope_confidence,
        )
        self.message_bus.send(msg)

        return Result(
            value={"signal": name, "payload": payload},
            confidence=self._scope_confidence,
            source="signal",
            agent=self.agent_id,
        )

    def _eval_trust(self, expr: Expression) -> Result:
        target = expr.get("agent", "")
        level = float(expr.get("level", 0.5))
        basis = expr.get("basis", "unknown")

        # Record trust in our agent registry
        if target in self.agents:
            self.agents[target].trust = level

        return Result(
            value={"agent": target, "trust": level, "basis": basis},
            confidence=self._scope_confidence,
            source="trust",
            agent=self.agent_id,
        )

    def _eval_confidence_op(self, expr: Expression) -> Result:
        level = float(expr.get("level", expr.get("value", 1.0)))
        self._scope_confidence = max(0.0, min(1.0, level))
        return Result(
            value=self._scope_confidence,
            confidence=self._scope_confidence,
            source="confidence",
            agent=self.agent_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_eval(self, value: Any) -> Any:
        """If value is a dict with 'op', evaluate it.  Otherwise return as-is."""
        if isinstance(value, dict) and "op" in value:
            return self.evaluate(Expression.from_dict(value)).value
        if isinstance(value, str) and value in self.state:
            return self.state[value]
        return value

    def _eval_body(self, body: list[Any]) -> Result:
        """Evaluate a list of expressions sequentially, return the last result."""
        last: Optional[Result] = None
        for step in body:
            if isinstance(step, Expression):
                last = self.evaluate(step)
            elif isinstance(step, dict) and "op" in step:
                last = self.evaluate(Expression.from_dict(step))
            else:
                last = Result(value=step, confidence=self._scope_confidence, source="body", agent=self.agent_id)
        return last or Result(value=None, confidence=self._scope_confidence, source="body", agent=self.agent_id)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def interpret(program: Program) -> Result:
    """Interpret a Signal program and return the final result."""
    interp = Interpreter()
    return interp.interpret(program)


def evaluate(expr: Expression | dict[str, Any]) -> Result:
    """Evaluate a single expression."""
    interp = Interpreter()
    if isinstance(expr, dict):
        expr = Expression.from_dict(expr)
    return interp.evaluate(expr)
