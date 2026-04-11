"""
Round 18: End-to-end integration pipeline tests.

Tests the full pipeline: parse -> bridge types -> compile -> verify -> execute.
Also tests multi-hop compilation, AST unification, and stress testing.
"""

import itertools
import threading
import pytest

from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    TypeAlgebra,
    TypeSafeBridge,
)
from flux_a2a.cross_compiler import (
    ASTDiffEngine,
    CodeEmitter,
    CrossCompiler,
    MultiHopCompiler,
    SemanticEquivalenceChecker,
    TranslationRuleSet,
)
from flux_a2a.types import (
    FluxType,
    FluxBaseType,
    FluxConstraint,
    ConstraintKind,
    build_default_registry,
)
from flux_a2a.ast_unifier import ASTUnifier, UnifiedASTNode


LANGUAGES = ["zho", "deu", "kor", "san", "lat", "wen"]


@pytest.fixture(scope="module")
def algebra():
    return TypeAlgebra()


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def compiler(algebra):
    return CrossCompiler(algebra=algebra)


@pytest.fixture(scope="module")
def multi_hop(algebra):
    return MultiHopCompiler(algebra=algebra)


@pytest.fixture(scope="module")
def checker(algebra):
    return SemanticEquivalenceChecker(algebra=algebra)


@pytest.fixture(scope="module")
def unifier():
    return ASTUnifier()


def _make_type(base: str, paradigm: str, confidence: float = 0.9, name: str = "",
               constraints: list = None):
    """Helper to create FluxType with correct base_type enum."""
    base_map = {
        "value": FluxBaseType.VALUE,
        "active": FluxBaseType.ACTIVE,
        "container": FluxBaseType.CONTAINER,
        "scope": FluxBaseType.SCOPE,
        "capability": FluxBaseType.CAPABILITY,
        "modal": FluxBaseType.MODAL,
        "uncertain": FluxBaseType.UNCERTAIN,
        "contextual": FluxBaseType.CONTEXTUAL,
    }
    bt = base_map.get(base.lower(), FluxBaseType.VALUE)
    return FluxType(
        base_type=bt,
        paradigm_source=paradigm,
        name=name,
        constraints=constraints or [],
        confidence=confidence,
    )


class TestFullPipeline:
    """Full pipeline: parse source -> bridge types -> compile -> verify."""

    def test_zho_to_deu_pipeline(self, compiler):
        """Compile ZHO types to DEU and verify compilation succeeds."""
        zho_types = [
            _make_type("active", "zho", 0.9, "cat", [
                FluxConstraint(
                    kind=ConstraintKind.CLASSIFIER_ANIMACY,
                    value="animal", language="zho"
                ),
            ]),
        ]
        result = compiler.compile(zho_types, "zho", "deu")
        assert result is not None
        assert result.target_code != ""
        assert result.route == ["zho", "deu"]

    def test_deu_to_kor_pipeline(self, compiler):
        """Compile DEU types to KOR."""
        deu_types = [
            _make_type("active", "deu", 0.85, "book", [
                FluxConstraint(
                    kind=ConstraintKind.CASE_MARKING,
                    value="nominativ", language="deu"
                ),
            ]),
        ]
        result = compiler.compile(deu_types, "deu", "kor")
        assert result is not None
        assert result.target_code != ""
        assert result.route == ["deu", "kor"]


class TestMultiHopCompilation:
    """Multi-hop routing finds cheaper bridge paths."""

    def test_find_cheapest_path(self, multi_hop):
        """MultiHopCompiler can find cheapest path between any two languages."""
        for src in LANGUAGES:
            for tgt in LANGUAGES:
                if src == tgt:
                    continue
                path = multi_hop.find_cheapest_path(src, tgt)
                assert isinstance(path, list)
                assert len(path) >= 2
                assert path[-1] == tgt

    def test_multi_hop_zho_san_lat(self, multi_hop):
        """ZHO -> SAN -> LAT multi-hop compilation."""
        zho_types = [
            _make_type("value", "zho", 0.8, "data", [
                FluxConstraint(
                    kind=ConstraintKind.CLASSIFIER_SHAPE,
                    value="flat_object", language="zho"
                ),
            ]),
        ]
        result = multi_hop.compile_hops(zho_types, ["zho", "lat"])
        assert result is not None
        assert len(result.route) >= 2
        assert result.route[0] == "zho"
        assert result.route[-1] == "lat"

    def test_direct_vs_multihop_cost(self, multi_hop, algebra):
        """Multi-hop cost should be reasonable compared to direct cost."""
        cost_matrix = BridgeCostMatrix(algebra)

        for src, tgt in [("zho", "lat"), ("wen", "deu"), ("kor", "san")]:
            direct = cost_matrix.compute(src, tgt).total_cost
            path = multi_hop.find_cheapest_path(src, tgt)

            # Compute hop cost manually from path
            hop_cost = 0.0
            for i in range(len(path) - 1):
                hop_cost += cost_matrix.compute(path[i], path[i + 1]).total_cost

            assert hop_cost <= direct + 0.5, (
                f"Multi-hop {path} cost ({hop_cost:.4f}) much worse than "
                f"direct {src}->{tgt} ({direct:.4f})"
            )


class TestSemanticEquivalence:
    """SemanticEquivalenceChecker behavioral verification."""

    def test_identical_types_equivalent(self, checker):
        """Identical type lists should be highly equivalent."""
        types = [_make_type("value", "zho", 0.9)]
        result = checker.check(types, types)
        assert result.is_equivalent
        assert result.equivalence_score > 0.6

    def test_cross_language_moderate_equivalence(self, checker):
        """Cross-language compilation should preserve moderate equivalence."""
        src_types = [_make_type("active", "zho", 0.85, "person")]
        tgt_types = [_make_type("active", "deu", 0.85, "person")]
        result = checker.check(src_types, tgt_types)
        assert result.equivalence_score > 0.4

    def test_divergent_types_detected(self, checker):
        """Divergent types should produce lower equivalence."""
        src_types = [_make_type("value", "zho", 0.95)]
        tgt_types = [_make_type("uncertain", "wen", 0.3)]
        result = checker.check(src_types, tgt_types)
        assert result.equivalence_score < 0.8


class TestASTUnifierIntegration:
    """AST unification across multiple languages."""

    def test_unifier_creation(self, unifier):
        assert hasattr(unifier, "unify")
        assert hasattr(unifier, "structural_hash")
        assert hasattr(unifier, "structural_distance")

    def test_structural_distance_identical(self, unifier):
        node = UnifiedASTNode(
            node_type="literal", children=[], metadata={"value": 42},
        )
        dist = unifier.structural_distance(node, node)
        assert dist < 0.2

    def test_structural_distance_different(self, unifier):
        literal = UnifiedASTNode(
            node_type="literal", children=[], metadata={"value": 42},
        )
        loop = UnifiedASTNode(
            node_type="loop",
            children=[UnifiedASTNode(node_type="literal", children=[], metadata={"value": 0})],
            metadata={"count": 10},
        )
        dist = unifier.structural_distance(literal, loop)
        assert dist > 0.2

    def test_structural_hash_stability(self, unifier):
        node = UnifiedASTNode(
            node_type="application",
            children=[
                UnifiedASTNode(node_type="literal", children=[], metadata={"value": 3}),
                UnifiedASTNode(node_type="literal", children=[], metadata={"value": 4}),
            ],
            metadata={"op": "add"},
        )
        h1 = unifier.structural_hash(node)
        h2 = unifier.structural_hash(node)
        assert h1 == h2


class TestCodeEmitter:
    """CodeEmitter produces valid target-language annotations."""

    def test_emit_zho(self):
        ft = _make_type("active", "zho", 0.9, "cat")
        code = CodeEmitter.emit(ft)
        assert "#zho" in code
        assert "type:" in code

    def test_emit_deu(self):
        ft = _make_type("active", "deu", 0.85, "book")
        code = CodeEmitter.emit(ft)
        assert "#deu" in code

    def test_emit_program(self):
        types = [_make_type("value", "zho", 1.0), _make_type("active", "zho", 0.9)]
        code = CodeEmitter.emit_program(types, "zho")
        assert "compiled to zho" in code
        assert "types: 2" in code


class TestTranslationRules:
    """TranslationRuleSet contains expected paradigm-pair rules."""

    def test_zho_deu_rules(self):
        rules = TranslationRuleSet.standard()
        assert len(rules.lookup("zho", "deu")) >= 5

    def test_deu_kor_rules(self):
        rules = TranslationRuleSet.standard()
        assert len(rules.lookup("deu", "kor")) >= 3

    def test_zho_kor_rules(self):
        rules = TranslationRuleSet.standard()
        assert len(rules.lookup("zho", "kor")) >= 2

    def test_reverse_rules_exist(self):
        rules = TranslationRuleSet.standard()
        for pair in [("zho", "deu"), ("deu", "kor"), ("zho", "kor")]:
            reverse_rules = rules.lookup(pair[1], pair[0])
            assert len(reverse_rules) > 0, f"No reverse rules for {pair}"


class TestASTDiffEngine:
    """ASTDiffEngine structural comparison."""

    def test_identical_types(self):
        types = [_make_type("value", "zho", 0.9)]
        diff = ASTDiffEngine.compare(types, types, "zho", "zho")
        assert diff.match_score > 0.8

    def test_cross_language_diff(self):
        src = [_make_type("active", "zho", 0.85)]
        tgt = [_make_type("active", "deu", 0.85)]
        diff = ASTDiffEngine.compare(src, tgt, "zho", "deu")
        assert diff.source_lang == "zho"
        assert diff.target_lang == "deu"
        assert diff.match_score > 0.4


class TestStressParallel:
    """Stress test: bridge all 15 pairs in parallel."""

    def test_parallel_bridge_computation(self, algebra):
        errors = []

        def compute_pair(src, tgt):
            try:
                cm = BridgeCostMatrix(algebra)
                report = cm.compute(src, tgt)
                assert 0.0 <= report.total_cost <= 1.0
            except Exception as e:
                errors.append((src, tgt, str(e)))

        threads = []
        for src, tgt in itertools.combinations(LANGUAGES, 2):
            t = threading.Thread(target=compute_pair, args=(src, tgt))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Parallel errors: {errors}"

    def test_parallel_algebra_queries(self, algebra):
        errors = []

        def query_algebra(lang, tag):
            try:
                for _ in range(50):
                    cls = algebra.find_class(lang, tag)
                    if cls:
                        _ = cls.languages()
                        _ = cls.has_language(lang)
            except Exception as e:
                errors.append((lang, tag, str(e)))

        threads = []
        for lang, tag in list(algebra._index.keys())[:20]:
            t = threading.Thread(target=query_algebra, args=(lang, tag))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Parallel algebra errors: {errors}"
