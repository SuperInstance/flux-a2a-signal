"""
Tests for FLUX denotational semantics (Round 10-12).

Tests cover:
  - Superposition (quantum-like register values)
  - FluxState (machine state)
  - FluxContext (execution context)
  - CapSet (capability sets)
  - FluxFunction composition (Kleisli bind, guard, branch)
  - FluxDenotation (expression → FluxFunction mapping)
  - Semantic properties (compositionality, monotonicity, context sensitivity)
  - Effect system mapping
"""

from __future__ import annotations

import math

import pytest

from flux_a2a.semantics import (
    BasisState,
    BranchFunc,
    CapSet,
    ConstFunc,
    ContextDependentFunc,
    DiscussFunc,
    ExtendedCap,
    FluxContext,
    FluxDenotation,
    FluxEffect,
    FluxFunction,
    FluxResult,
    FluxState,
    IdentityFunc,
    PureFunc,
    SemanticProperties,
    Superposition,
    SuperpositionFunc,
    uniform,
    denote,
    denote_and_run,
    pure,
    verify_all_properties,
)


# ═══════════════════════════════════════════════════════════════════════════
# §1  Superposition Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSuperposition:
    def test_pure(self) -> None:
        s = pure(42)
        assert s.is_deterministic
        assert s.deterministic_value() == 42
        assert s.collapse() == 42

    def test_uniform(self) -> None:
        s = uniform([1, 2, 3, 4])
        assert not s.is_deterministic
        assert len(s.states) == 4

    def test_amplitudes_normalized(self) -> None:
        s = uniform([1, 2, 3, 4])
        normed = s.normalize()
        total_prob = sum(basis.probability for basis in normed.states)
        assert abs(total_prob - 1.0) < 1e-9

    def test_entropy_deterministic(self) -> None:
        s = pure(42)
        assert s.entropy() == 0.0

    def test_entropy_uniform(self) -> None:
        s = uniform([1, 2, 3, 4])
        # H(uniform over 4) = log₂(4) = 2.0
        assert abs(s.entropy() - 2.0) < 1e-9

    def test_map(self) -> None:
        s = uniform([1, 2, 3])
        doubled = s.map(lambda x: x * 2)
        values = {basis.value for basis in doubled.states}
        assert values == {2, 4, 6}

    def test_filter(self) -> None:
        s = uniform([1, 2, 3, 4, 5])
        evens = s.filter(lambda x: x % 2 == 0)
        values = {basis.value for basis in evens.states}
        assert values == {2, 4}

    def test_compose_tensor_product(self) -> None:
        a = uniform([0, 1])
        b = uniform([0, 1])
        c = a.compose(b)
        assert len(c.states) == 4  # 2x2 tensor product

    def test_collapse_deterministic(self) -> None:
        s = pure(99)
        assert s.collapse() == 99

    def test_to_dict(self) -> None:
        s = pure(42)
        d = s.to_dict()
        assert d["is_deterministic"] is True
        assert d["entropy"] == 0.0

    def test_repr_deterministic(self) -> None:
        s = pure(42)
        r = repr(s)
        assert "42" in r

    def test_empty_superposition(self) -> None:
        s = Superposition()
        assert s.is_deterministic is False
        assert s.deterministic_value() is None

    def test_add_basis_state(self) -> None:
        s = Superposition()
        s.add("hello", 0.8)
        s.add("world", 0.6)
        assert len(s.states) == 2
        assert abs(s.states[0].amplitude - 0.8) < 1e-9

    def test_basis_state_probability(self) -> None:
        b = BasisState(value=42, amplitude=0.5)
        assert abs(b.probability - 0.25) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# §2  FluxState Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFluxState:
    def test_default_registers(self) -> None:
        state = FluxState()
        assert state.get_register(0).is_deterministic
        assert state.get_register(0).deterministic_value() == 0

    def test_set_register_scalar(self) -> None:
        state = FluxState()
        state.set_register(0, 42)
        assert state.get_register(0).deterministic_value() == 42

    def test_set_register_superposition(self) -> None:
        state = FluxState()
        s = uniform([1, 2, 3])
        state.set_register(0, s)
        reg = state.get_register(0)
        assert len(reg.states) == 3

    def test_fork_isolation(self) -> None:
        state = FluxState()
        state.set_register(0, 10)
        forked = state.fork()
        forked.set_register(0, 99)
        assert state.get_register(0).deterministic_value() == 10
        assert forked.get_register(0).deterministic_value() == 99

    def test_to_dict(self) -> None:
        state = FluxState()
        state.set_register(0, 42)
        d = state.to_dict()
        assert "registers" in d


# ═══════════════════════════════════════════════════════════════════════════
# §3  FluxContext Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFluxContext:
    def test_domain_stack(self) -> None:
        ctx = FluxContext()
        ctx.push_domain("math", 0.8)
        ctx.push_domain("philosophy", 0.9)
        assert ctx.active_domain() == "philosophy"
        ctx.pop_domain()
        assert ctx.active_domain() == "math"
        ctx.pop_domain()
        assert ctx.active_domain() is None

    def test_fork_isolation(self) -> None:
        ctx = FluxContext(language="zho")
        ctx.push_domain("math", 1.0)
        forked = ctx.fork()
        forked.language = "deu"
        assert ctx.language == "zho"
        assert forked.language == "deu"

    def test_temporal_mode(self) -> None:
        ctx = FluxContext(temporal_mode="imperfect")
        assert ctx.temporal_mode == "imperfect"

    def test_sanskrit_scope(self) -> None:
        ctx = FluxContext(scope_depth=5)
        assert ctx.scope_depth == 5

    def test_korean_honorific(self) -> None:
        ctx = FluxContext(honorific_level=1)
        assert ctx.honorific_level == 1


# ═══════════════════════════════════════════════════════════════════════════
# §4  CapSet Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCapSet:
    def test_empty(self) -> None:
        caps = CapSet.empty()
        assert caps.count == 0
        assert not caps.has(ExtendedCap.TELL)

    def test_full(self) -> None:
        caps = CapSet.full()
        assert caps.has(ExtendedCap.TELL)
        assert caps.has(ExtendedCap.FORK)
        assert caps.has(ExtendedCap.META)

    def test_union(self) -> None:
        a = CapSet.from_german("nom")
        b = CapSet.from_german("akk")
        c = a.union(b)
        assert c.has(ExtendedCap.NOM_READ)
        assert c.has(ExtendedCap.AKK_WRITE)
        assert not c.has(ExtendedCap.DAT_SHARE)

    def test_intersection(self) -> None:
        a = CapSet.from_german("nom")
        b = CapSet.from_german("akk")
        c = a.intersection(b)
        assert not c.has(ExtendedCap.NOM_READ)
        assert not c.has(ExtendedCap.AKK_WRITE)

    def test_subset(self) -> None:
        a = CapSet.from_german("nom")
        b = a.union(CapSet.from_german("akk"))
        assert a.is_subset(b)
        assert not b.is_subset(a)

    def test_from_korean_formal(self) -> None:
        caps = CapSet.from_korean(1)
        assert caps.has(ExtendedCap.SPEAK_FORMAL)
        assert caps.has(ExtendedCap.SPEAK_POLITE)
        assert caps.has(ExtendedCap.SPEAK_PLAIN)

    def test_from_korean_plain(self) -> None:
        caps = CapSet.from_korean(7)
        assert caps.has(ExtendedCap.SPEAK_PLAIN)
        assert not caps.has(ExtendedCap.SPEAK_FORMAL)

    def test_from_german_nominativ(self) -> None:
        caps = CapSet.from_german("Nominativ")
        assert caps.has(ExtendedCap.NOM_READ)
        assert not caps.has(ExtendedCap.AKK_WRITE)

    def test_from_german_genitiv(self) -> None:
        caps = CapSet.from_german("Genitiv")
        assert caps.has(ExtendedCap.GEN_OWN)

    def test_from_sanskrit_scope_hierarchy(self) -> None:
        caps = CapSet.from_sanskrit(5)
        assert caps.has(ExtendedCap.SCOPE_1)
        assert caps.has(ExtendedCap.SCOPE_5)
        assert not caps.has(ExtendedCap.SCOPE_6)
        assert not caps.has(ExtendedCap.SCOPE_8)

    def test_from_latin_perfectum(self) -> None:
        caps = CapSet.from_latin("perfect")
        assert caps.has(ExtendedCap.TEMPORAL_CACHE)
        assert not caps.has(ExtendedCap.TEMPORAL_LOOP)

    def test_from_latin_present(self) -> None:
        caps = CapSet.from_latin("present")
        assert len(caps.capabilities) == 0

    def test_requires_all(self) -> None:
        caps = CapSet.from_korean(1)
        assert caps.requires_all([ExtendedCap.SPEAK_FORMAL, ExtendedCap.SPEAK_PLAIN])
        assert not caps.requires_all([ExtendedCap.FORK, ExtendedCap.SPEAK_FORMAL])

    def test_requires_any(self) -> None:
        caps = CapSet.from_german("nom")
        assert caps.requires_any([ExtendedCap.FORK, ExtendedCap.NOM_READ])

    def test_to_dict(self) -> None:
        caps = CapSet.from_korean(1)
        d = caps.to_dict()
        assert d["count"] == 3
        assert "source" in d


# ═══════════════════════════════════════════════════════════════════════════
# §5  FluxFunction Primitive Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFluxFunction:
    def test_identity(self) -> None:
        state = FluxState()
        ctx = FluxContext()
        caps = CapSet.full()
        fn = IdentityFunc()
        result = fn.run(state, ctx, caps)
        assert result.confidence == 1.0
        assert result.is_error is False

    def test_const(self) -> None:
        fn = ConstFunc(42, 0.9)
        result = fn.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == 42
        assert result.confidence == 0.9

    def test_pure_fn(self) -> None:
        fn = PureFunc(lambda s, c, k: FluxResult(value=s.get_register(0).deterministic_value() + 1, new_state=s))
        state = FluxState()
        state.set_register(0, 10)
        result = fn.run(state, FluxContext(), CapSet.full())
        assert result.value == 11

    def test_bind(self) -> None:
        f = ConstFunc(10)
        g = lambda prev: ConstFunc(prev.value + 5)
        composed = f.bind(g)
        result = composed.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == 15

    def test_then(self) -> None:
        f = ConstFunc(10)
        g = ConstFunc(20)
        composed = f.then(g)
        result = composed.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == 20

    def test_guard_permitted(self) -> None:
        fn = ConstFunc(42)
        guarded = fn.guard(ExtendedCap.TELL)
        result = guarded.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == 42
        assert result.is_error is False

    def test_guard_denied(self) -> None:
        fn = ConstFunc(42)
        guarded = fn.guard(ExtendedCap.TELL)
        result = guarded.run(FluxState(), FluxContext(), CapSet.empty())
        assert result.is_error
        assert "tell" in result.error.lower()

    def test_apply_distribution(self) -> None:
        fn = ConstFunc(42)
        results = fn.apply(FluxState(), FluxContext(), CapSet.full())
        assert len(results) == 1
        assert results[0][0].value == 42
        assert results[0][1] == 1.0

    def test_run_empty_result(self) -> None:
        fn = PureFunc(lambda s, c, k: FluxResult(value=None, confidence=0.0, new_state=s, error="fail"))
        result = fn.run(FluxState(), FluxContext(), CapSet.full())
        assert result.is_error


# ═══════════════════════════════════════════════════════════════════════════
# §6  BranchFunc Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBranchFunc:
    def test_simple_branch(self) -> None:
        branch = BranchFunc([
            (ConstFunc("A"), 0.7),
            (ConstFunc("B"), 0.3),
        ])
        results = branch.apply(FluxState(), FluxContext(), CapSet.full())
        values = [r.value for r, p in results]
        assert "A" in values
        assert "B" in values
        assert len(results) == 2

    def test_empty_branch(self) -> None:
        branch = BranchFunc([])
        result = branch.run(FluxState(), FluxContext(), CapSet.full())
        assert result.is_error

    def test_branch_weights_normalized(self) -> None:
        branch = BranchFunc([
            (ConstFunc("A"), 3.0),
            (ConstFunc("B"), 7.0),
        ])
        results = branch.apply(FluxState(), FluxContext(), CapSet.full())
        # Probabilities should be 0.3 and 0.7
        probs = {r.value: p for r, p in results}
        assert abs(probs["A"] - 0.3) < 1e-9
        assert abs(probs["B"] - 0.7) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# §7  DiscussFunc Tests (Game-Theoretic)
# ═══════════════════════════════════════════════════════════════════════════

class TestDiscussFunc:
    def test_brainstorm_cooperative(self) -> None:
        discuss = DiscussFunc(
            agents=[
                ("agent_a", ConstFunc("idea_1")),
                ("agent_b", ConstFunc("idea_2")),
            ],
            game_type=DiscussFunc.GameType.BRAINSTORM,
        )
        results = discuss.apply(FluxState(), FluxContext(), CapSet.full())
        assert len(results) >= 1

    def test_debate_competitive(self) -> None:
        discuss = DiscussFunc(
            agents=[
                ("agent_a", ConstFunc("position_a")),
                ("agent_b", ConstFunc("position_b")),
            ],
            game_type=DiscussFunc.GameType.DEBATE,
        )
        result = discuss.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value in ("position_a", "position_b")

    def test_review_consensus(self) -> None:
        discuss = DiscussFunc(
            agents=[
                ("reviewer_1", ConstFunc(42)),
                ("reviewer_2", ConstFunc(42)),
            ],
            game_type=DiscussFunc.GameType.REVIEW,
        )
        result = discuss.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == 42
        assert not result.is_error

    def test_review_disagreement(self) -> None:
        discuss = DiscussFunc(
            agents=[
                ("reviewer_1", ConstFunc(42)),
                ("reviewer_2", ConstFunc(99)),
            ],
            game_type=DiscussFunc.GameType.REVIEW,
        )
        result = discuss.run(FluxState(), FluxContext(), CapSet.full())
        assert result.is_error
        assert "consensus" in result.error.lower()

    def test_negotiation_compromise(self) -> None:
        discuss = DiscussFunc(
            agents=[
                ("agent_a", ConstFunc("option_a")),
                ("agent_b", ConstFunc("option_b")),
                ("agent_c", ConstFunc("option_a")),  # Most common
            ],
            game_type=DiscussFunc.GameType.NEGOTIATION,
        )
        result = discuss.run(FluxState(), FluxContext(), CapSet.full())
        assert result.value == "option_a"  # Most common wins


# ═══════════════════════════════════════════════════════════════════════════
# §8  ContextDependentFunc Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContextDependentFunc:
    def test_domain_dispatch(self) -> None:
        fn = ContextDependentFunc(
            dispatch={
                "math": ConstFunc(42),
                "philosophy": ConstFunc("道"),
            },
        )
        ctx_math = FluxContext()
        ctx_math.push_domain("math", 1.0)
        ctx_phil = FluxContext()
        ctx_phil.push_domain("philosophy", 1.0)

        result_math = fn.run(FluxState(), ctx_math, CapSet.full())
        result_phil = fn.run(FluxState(), ctx_phil, CapSet.full())

        assert result_math.value == 42
        assert result_phil.value == "道"

    def test_default_dispatch(self) -> None:
        fn = ContextDependentFunc(
            dispatch={"math": ConstFunc(42)},
            default=ConstFunc("unknown"),
        )
        ctx = FluxContext()  # No domain pushed
        result = fn.run(FluxState(), ctx, CapSet.full())
        assert result.value == "unknown"

    def test_custom_key_fn(self) -> None:
        fn = ContextDependentFunc(
            dispatch={
                "zho": ConstFunc("道"),
                "deu": ConstFunc("Weg"),
            },
            key_fn=lambda ctx: ctx.language,
        )
        result_zho = fn.run(FluxState(), FluxContext(language="zho"), CapSet.full())
        result_deu = fn.run(FluxState(), FluxContext(language="deu"), CapSet.full())
        assert result_zho.value == "道"
        assert result_deu.value == "Weg"


# ═══════════════════════════════════════════════════════════════════════════
# §9  SuperpositionFunc Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSuperpositionFunc:
    def test_map_over_superposition(self) -> None:
        state = FluxState()
        state.set_register(0, uniform([1, 2, 3]))

        fn = SuperpositionFunc(0, lambda val: ConstFunc(val * 10))
        results = fn.apply(state, FluxContext(), CapSet.full())

        values = sorted([r.value for r, p in results])
        assert values == [10, 20, 30]

    def test_deterministic_passthrough(self) -> None:
        state = FluxState()
        state.set_register(0, pure(42))

        fn = SuperpositionFunc(0, lambda val: ConstFunc(val + 1))
        results = fn.apply(state, FluxContext(), CapSet.full())
        assert len(results) == 1
        assert results[0][0].value == 43


# ═══════════════════════════════════════════════════════════════════════════
# §10  FluxDenotation Tests (Expression → FluxFunction)
# ═══════════════════════════════════════════════════════════════════════════

class TestFluxDenotation:
    def setup_method(self) -> None:
        self.fd = FluxDenotation()
        self.state = FluxState()
        self.ctx = FluxContext()
        self.caps = CapSet.full()

    def test_literal(self) -> None:
        fn = self.fd.denote_expression({"op": "literal", "value": 42})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == 42

    def test_arithmetic_add(self) -> None:
        fn = self.fd.denote_expression({"op": "add", "args": [3, 4]})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == 7

    def test_arithmetic_div_by_zero(self) -> None:
        fn = self.fd.denote_expression({"op": "div", "args": [10, 0]})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.is_error

    def test_comparison_eq(self) -> None:
        fn = self.fd.denote_expression({"op": "eq", "args": [5, 5]})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value is True

    def test_comparison_gt(self) -> None:
        fn = self.fd.denote_expression({"op": "gt", "args": [10, 5]})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value is True

    def test_seq(self) -> None:
        fn = self.fd.denote_expression({
            "op": "seq",
            "body": [
                {"op": "let", "name": "x", "value": 10},
                {"op": "add", "args": [{"op": "get", "name": "x"}, 5]},
            ],
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == 15

    def test_if_true(self) -> None:
        fn = self.fd.denote_expression({
            "op": "if",
            "condition": True,
            "then": {"op": "literal", "value": "yes"},
            "else": {"op": "literal", "value": "no"},
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == "yes"

    def test_if_false(self) -> None:
        fn = self.fd.denote_expression({
            "op": "if",
            "condition": False,
            "then": {"op": "literal", "value": "yes"},
            "else": {"op": "literal", "value": "no"},
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == "no"

    def test_loop(self) -> None:
        fn = self.fd.denote_expression({
            "op": "loop",
            "times": 5,
            "body": [{"op": "literal", "value": "step"}],
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert len(result.value) == 5

    def test_concat(self) -> None:
        fn = self.fd.denote_expression({
            "op": "concat",
            "args": ["hello", " ", "world"],
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == "hello world"

    def test_confidence(self) -> None:
        fn = self.fd.denote_expression({"op": "confidence", "level": 0.7})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == 0.7

    def test_tell_with_caps(self) -> None:
        fn = self.fd.denote_expression({
            "op": "tell",
            "to": "agent",
            "message": "hello",
        })
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == "hello"
        assert "tell:agent" in result.effects

    def test_tell_without_caps(self) -> None:
        fn = self.fd.denote_expression({
            "op": "tell",
            "to": "agent",
            "message": "hello",
        })
        result = fn.run(self.state, self.ctx, CapSet.empty())
        assert result.is_error
        assert "tell" in result.error.lower()

    def test_unknown_opcode(self) -> None:
        fn = self.fd.denote_expression({"op": "nonexistent"})
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.is_error

    def test_branch(self) -> None:
        fn = self.fd.denote_expression({
            "op": "branch",
            "branches": [
                {"label": "a", "weight": 0.6, "body": [{"op": "literal", "value": "A"}]},
                {"label": "b", "weight": 0.4, "body": [{"op": "literal", "value": "B"}]},
            ],
        })
        results = fn.apply(self.state, self.ctx, self.caps)
        values = {r.value for r, p in results}
        assert "A" in values
        assert "B" in values

    def test_sequence_composition(self) -> None:
        fn = self.fd.denote_sequence([
            {"op": "literal", "value": 1},
            {"op": "literal", "value": 2},
            {"op": "literal", "value": 3},
        ])
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.value == 3

    def test_empty_sequence(self) -> None:
        fn = self.fd.denote_sequence([])
        result = fn.run(self.state, self.ctx, self.caps)
        assert result.confidence == 1.0
        assert not result.is_error


# ═══════════════════════════════════════════════════════════════════════════
# §11  Semantic Properties Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticProperties:
    def test_compositionality(self) -> None:
        fd = FluxDenotation()
        props = SemanticProperties()
        state = FluxState()
        ctx = FluxContext()
        caps = CapSet.full()

        ok, msg = props.verify_compositionality(
            fd,
            {"op": "literal", "value": 10},
            {"op": "literal", "value": 20},
            state, ctx, caps,
        )
        assert ok, msg

    def test_confidence_monotonicity(self) -> None:
        fd = FluxDenotation()
        props = SemanticProperties()
        state = FluxState()
        ctx = FluxContext()
        caps = CapSet.full()

        ok, msg = props.verify_confidence_monotonicity(
            fd,
            [{"op": "confidence", "level": 0.8}],
            state, ctx, caps,
        )
        assert ok, msg

    def test_capability_monotonicity(self) -> None:
        fd = FluxDenotation()
        props = SemanticProperties()
        state = FluxState()
        ctx = FluxContext()

        ok, msg = props.verify_capability_monotonicity(
            fd,
            {"op": "tell", "to": "agent", "message": "hello"},
            state, ctx,
            CapSet.empty(),
            CapSet.full(),
        )
        assert ok, msg

    def test_context_sensitivity(self) -> None:
        fd = FluxDenotation()
        props = SemanticProperties()
        state = FluxState()
        caps = CapSet.full()

        ok, msg = props.verify_context_sensitivity(
            fd,
            {"op": "literal", "value": 42},
            state,
            FluxContext(language="zho"),
            FluxContext(language="deu"),
            caps,
        )
        assert ok, msg

    def test_verify_all(self) -> None:
        results = verify_all_properties()
        for name, (ok, msg) in results.items():
            assert ok, f"Property {name} failed: {msg}"


# ═══════════════════════════════════════════════════════════════════════════
# §12  Effect System Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFluxEffect:
    def test_capability_for_tell(self) -> None:
        cap = FluxEffect.capability_for_effect(FluxEffect.EffectType.TELL)
        assert cap == ExtendedCap.TELL

    def test_capability_for_fork(self) -> None:
        cap = FluxEffect.capability_for_effect(FluxEffect.EffectType.FORK)
        assert cap == ExtendedCap.FORK

    def test_korean_level_effects_formal(self) -> None:
        effects = FluxEffect.effects_for_korean_level(1)
        assert len(effects) == len(FluxEffect.EffectType)

    def test_korean_level_effects_plain(self) -> None:
        effects = FluxEffect.effects_for_korean_level(7)
        assert FluxEffect.EffectType.TELL not in effects


# ═══════════════════════════════════════════════════════════════════════════
# §13  Convenience Function Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConvenience:
    def test_denote(self) -> None:
        fn = denote({"op": "literal", "value": 99})
        assert isinstance(fn, FluxFunction)

    def test_denote_and_run(self) -> None:
        result = denote_and_run({"op": "add", "args": [3, 4]})
        assert result.value == 7

    def test_denote_and_run_with_caps(self) -> None:
        result = denote_and_run(
            {"op": "tell", "to": "agent", "message": "hello"},
            caps=CapSet.empty(),
        )
        assert result.is_error


# ═══════════════════════════════════════════════════════════════════════════
# §14  Integration: Cross-Language Capability Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossLanguageCapabilities:
    """Test capability derivation from all 6 language families."""

    def test_zho_implicit_caps(self) -> None:
        """Chinese: context-dependent meaning, no explicit Kasus."""
        ctx = FluxContext(language="zho")
        caps = CapSet.full()
        assert caps.source == "full"  # ZHO relies on context, not explicit caps

    def test_deu_kasus_chain(self) -> None:
        """German: all 4 Kasus combined."""
        nom = CapSet.from_german("nom")
        akk = CapSet.from_german("akk")
        dat = CapSet.from_german("dat")
        gen = CapSet.from_german("gen")
        all_cases = nom.union(akk).union(dat).union(gen)
        assert all_cases.has(ExtendedCap.NOM_READ)
        assert all_cases.has(ExtendedCap.AKK_WRITE)
        assert all_cases.has(ExtendedCap.DAT_SHARE)
        assert all_cases.has(ExtendedCap.GEN_OWN)

    def test_kor_honorific_levels(self) -> None:
        """Korean: 7 honorific levels map to capability sets."""
        for level in range(1, 8):
            caps = CapSet.from_korean(level)
            assert caps.has(ExtendedCap.SPEAK_PLAIN)
            # Higher levels should NOT have formal
            if level >= 4:
                assert not caps.has(ExtendedCap.SPEAK_FORMAL)

    def test_san_scope_hierarchy_monotonic(self) -> None:
        """Sanskrit: scope capabilities are monotonically increasing."""
        prev_count = 0
        for level in range(1, 9):
            caps = CapSet.from_sanskrit(level)
            assert len(caps.capabilities) >= prev_count
            prev_count = len(caps.capabilities)

    def test_lat_temporal_modes(self) -> None:
        """Latin: each tense produces different temporal capabilities."""
        modes = {
            "imperfect": ExtendedCap.TEMPORAL_LOOP,
            "perfect": ExtendedCap.TEMPORAL_CACHE,
            "pluperfect": ExtendedCap.TEMPORAL_ROLLBACK,
            "future": ExtendedCap.TEMPORAL_DEFER,
            "future_perfect": ExtendedCap.TEMPORAL_EVENTUAL,
        }
        for tense, expected_cap in modes.items():
            caps = CapSet.from_latin(tense)
            assert caps.has(expected_cap), f"Latin {tense} should have {expected_cap.value}"

    def test_cross_language_union(self) -> None:
        """Multi-language program: capabilities from all languages combined."""
        deu_caps = CapSet.from_german("nom")
        kor_caps = CapSet.from_korean(1)
        san_caps = CapSet.from_sanskrit(3)
        lat_caps = CapSet.from_latin("imperfect")

        combined = deu_caps.union(kor_caps).union(san_caps).union(lat_caps)
        assert combined.has(ExtendedCap.NOM_READ)
        assert combined.has(ExtendedCap.SPEAK_FORMAL)
        assert combined.has(ExtendedCap.SCOPE_1)
        assert combined.has(ExtendedCap.TEMPORAL_LOOP)
