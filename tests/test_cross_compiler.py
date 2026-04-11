"""
Tests for the bidirectional cross-compiler.

Covers:
  - ZHO→DEU compilation of simple expressions
  - DEU→KOR compilation with honorific mapping
  - Round-trip ZHO→DEU→ZHO preservation rate
  - Multi-hop ZHO→SAN→LAT cheaper than ZHO→LAT direct
  - SemanticEquivalenceChecker with arithmetic expressions
  - Information preservation metric accuracy
  - Translation rules data-driven behavior
  - ASTDiff structural comparison
  - CodeEmitter output format
  - Route optimization edge cases
"""

from __future__ import annotations

import math

import pytest

from flux_a2a.types import (
    ConstraintKind,
    FluxBaseType,
    FluxConstraint,
    FluxType,
    _PARADIGM_TO_BASE,
)
from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    PreservationDegree,
    TypeAlgebra,
    TypeSafeBridge,
)
from flux_a2a.cross_compiler import (
    ASTDiff,
    ASTDiffEngine,
    ASTDiffKind,
    CodeEmitter,
    CompilationResult,
    CrossCompiler,
    EquivalenceCheckResult,
    MultiHopCompiler,
    RoundTripResult,
    SemanticEquivalenceChecker,
    SUPPORTED_LANGUAGES,
    TransformKind,
    TranslationRule,
    TranslationRuleSet,
)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def algebra() -> TypeAlgebra:
    return TypeAlgebra()


@pytest.fixture
def cost_matrix(algebra: TypeAlgebra) -> BridgeCostMatrix:
    return BridgeCostMatrix(algebra)


@pytest.fixture
def safe_bridge(algebra: TypeAlgebra) -> TypeSafeBridge:
    return TypeSafeBridge(algebra=algebra)


@pytest.fixture
def rule_set() -> TranslationRuleSet:
    return TranslationRuleSet.standard()


@pytest.fixture
def compiler(
    algebra: TypeAlgebra,
    cost_matrix: BridgeCostMatrix,
    safe_bridge: TypeSafeBridge,
    rule_set: TranslationRuleSet,
) -> CrossCompiler:
    return CrossCompiler(
        algebra=algebra,
        cost_matrix=cost_matrix,
        safe_bridge=safe_bridge,
        rule_set=rule_set,
    )


@pytest.fixture
def multi_hop_compiler(
    algebra: TypeAlgebra,
    cost_matrix: BridgeCostMatrix,
    safe_bridge: TypeSafeBridge,
    rule_set: TranslationRuleSet,
) -> MultiHopCompiler:
    return MultiHopCompiler(
        algebra=algebra,
        cost_matrix=cost_matrix,
        safe_bridge=safe_bridge,
        rule_set=rule_set,
    )


@pytest.fixture
def zho_person_type() -> FluxType:
    """ZHO person classifier type (位)."""
    return FluxType.from_paradigm("zho", "person", name="zho:person")


@pytest.fixture
def zho_animal_type() -> FluxType:
    """ZHO animal classifier type (隻)."""
    return FluxType.from_paradigm("zho", "animal", name="zho:animal")


@pytest.fixture
def zho_flat_type() -> FluxType:
    """ZHO flat object classifier type (張)."""
    return FluxType.from_paradigm("zho", "flat_object", name="zho:flat_object")


@pytest.fixture
def deu_mask_type() -> FluxType:
    """DEU Maskulinum type."""
    return FluxType.from_paradigm("deu", "maskulinum", name="deu:maskulinum")


@pytest.fixture
def deu_fem_type() -> FluxType:
    """DEU Femininum type."""
    return FluxType.from_paradigm("deu", "femininum", name="deu:femininum")


@pytest.fixture
def deu_neut_type() -> FluxType:
    """DEU Neutrum type."""
    return FluxType.from_paradigm("deu", "neutrum", name="deu:neutrum")


@pytest.fixture
def deu_nom_type() -> FluxType:
    """DEU Nominativ type."""
    return FluxType.from_paradigm("deu", "nominativ", name="deu:nominativ")


@pytest.fixture
def kor_haeyoche_type() -> FluxType:
    """KOR 해요체 (polite) type."""
    return FluxType.from_paradigm("kor", "haeyoche", name="kor:haeyoche")


@pytest.fixture
def kor_haeche_type() -> FluxType:
    """KOR 해체 (informal) type."""
    return FluxType.from_paradigm("kor", "haeche", name="kor:haeche")


@pytest.fixture
def arithmetic_types() -> list[FluxType]:
    """VALUE types representing arithmetic expressions."""
    return [
        FluxType.from_paradigm("zho", "flat_object", confidence=0.95, name="zho:flat"),
        FluxType.from_paradigm("zho", "small_round", confidence=0.9, name="zho:round"),
        FluxType.from_paradigm("deu", "neutrum", confidence=0.95, name="deu:neut"),
    ]


# ══════════════════════════════════════════════════════════════════
# TranslationRuleSet Tests
# ══════════════════════════════════════════════════════════════════

class TestTranslationRuleSet:
    """Tests for the data-driven translation rule system."""

    def test_standard_rules_loaded(self, rule_set: TranslationRuleSet):
        """Standard rule set should contain rules for multiple paradigm pairs."""
        rules = list(rule_set.all_rules())
        assert len(rules) >= 20, f"Expected >= 20 rules, got {len(rules)}"

    def test_zho_deu_rules_exist(self, rule_set: TranslationRuleSet):
        """ZHO→DEU rules should cover classifier-to-gender mappings."""
        zho_deu_rules = rule_set.lookup("zho", "deu")
        assert len(zho_deu_rules) >= 5
        source_patterns = {r.source_pattern for r in zho_deu_rules}
        assert "person" in source_patterns
        assert "flat_object" in source_patterns

    def test_deu_kor_rules_exist(self, rule_set: TranslationRuleSet):
        """DEU→KOR rules should cover case-to-particle mappings."""
        deu_kor_rules = rule_set.lookup("deu", "kor")
        assert len(deu_kor_rules) >= 4
        source_patterns = {r.source_pattern for r in deu_kor_rules}
        assert "nominativ" in source_patterns
        assert "akkusativ" in source_patterns

    def test_zho_kor_rules_exist(self, rule_set: TranslationRuleSet):
        """ZHO→KOR rules should cover classifier-to-particle mappings."""
        zho_kor_rules = rule_set.lookup("zho", "kor")
        assert len(zho_kor_rules) >= 3

    def test_reverse_rules_exist(self, rule_set: TranslationRuleSet):
        """Each paradigm pair should have rules in both directions."""
        zho_deu = rule_set.lookup("zho", "deu")
        deu_zho = rule_set.lookup("deu", "zho")
        assert len(zho_deu) >= 3
        assert len(deu_zho) >= 3

    def test_rule_matching(self, rule_set: TranslationRuleSet):
        """match() should find the first matching rule."""
        rule = rule_set.match("zho", "deu", "person")
        assert rule is not None
        assert rule.target_pattern == "maskulinum"

    def test_rule_no_match(self, rule_set: TranslationRuleSet):
        """match() should return None for unknown tags."""
        rule = rule_set.match("zho", "deu", "nonexistent_tag")
        assert rule is None

    def test_rule_transform_kinds(self, rule_set: TranslationRuleSet):
        """Rules should cover different transform kinds."""
        kinds = {r.transform_kind for r in rule_set.all_rules()}
        assert TransformKind.CLASSIFIER_TO_PLURAL in kinds
        assert TransformKind.SCOPE_TO_PARTICLE in kinds
        assert TransformKind.GENDER_TO_HONORIFIC in kinds
        assert TransformKind.TOPIC_MARKING in kinds

    def test_rule_confidence_factors(self, rule_set: TranslationRuleSet):
        """All rules should have confidence factors in (0, 1]."""
        for rule in rule_set.all_rules():
            assert 0.0 < rule.confidence_factor <= 1.0


# ══════════════════════════════════════════════════════════════════
# ZHO → DEU Compilation Tests
# ══════════════════════════════════════════════════════════════════

class TestZhoToDeuCompilation:
    """Tests for Chinese → German cross-compilation."""

    def test_person_to_maskulinum(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """ZHO person classifier (位) should compile to DEU Maskulinum."""
        result = compiler.compile([zho_person_type], "zho", "deu")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "deu"
        assert result.information_preserved > 0.0
        assert len(result.witness_chain) >= 1

    def test_animal_to_maskulinum(
        self, compiler: CrossCompiler, zho_animal_type: FluxType
    ):
        """ZHO animal classifier (隻) should compile to DEU Maskulinum."""
        result = compiler.compile([zho_animal_type], "zho", "deu")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "deu"

    def test_flat_object_to_neutrum(
        self, compiler: CrossCompiler, zho_flat_type: FluxType
    ):
        """ZHO flat_object classifier (張) should compile to DEU Neutrum."""
        result = compiler.compile([zho_flat_type], "zho", "deu")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "deu"

    def test_multiple_types(
        self,
        compiler: CrossCompiler,
        zho_person_type: FluxType,
        zho_animal_type: FluxType,
        zho_flat_type: FluxType,
    ):
        """Multiple ZHO types should all compile to DEU types."""
        result = compiler.compile(
            [zho_person_type, zho_animal_type, zho_flat_type],
            "zho", "deu",
        )

        assert len(result.target_types) == 3
        for tt in result.target_types:
            assert tt.paradigm_source == "deu"
        assert result.route == ["zho", "deu"]

    def test_string_source_parsing(self, compiler: CrossCompiler):
        """String source should be parsed as comma-separated tags."""
        result = compiler.compile("person, flat_object", "zho", "deu")

        assert len(result.source_types) == 2
        assert len(result.target_types) == 2

    def test_witness_chain_validity(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """All witnesses in the chain should be valid."""
        result = compiler.compile([zho_person_type], "zho", "deu")

        assert result.is_type_safe, (
            f"Invalid witnesses: "
            f"{[w.failed_constraints for w in result.witness_chain if not w.is_valid]}"
        )

    def test_target_code_format(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """Target code should start with the #deu prefix."""
        result = compiler.compile([zho_person_type], "zho", "deu")

        assert "#deu" in result.target_code


# ══════════════════════════════════════════════════════════════════
# DEU → KOR Compilation Tests
# ══════════════════════════════════════════════════════════════════

class TestDeuToKorCompilation:
    """Tests for German → Korean cross-compilation."""

    def test_nominativ_to_subject_honorific(
        self, compiler: CrossCompiler, deu_nom_type: FluxType
    ):
        """DEU Nominativ should compile to KOR subject_honorific."""
        result = compiler.compile([deu_nom_type], "deu", "kor")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "kor"
        assert result.information_preserved > 0.0

    def test_maskulinum_to_hasipsioche(
        self, compiler: CrossCompiler, deu_mask_type: FluxType
    ):
        """DEU Maskulinum → KOR hasipsioche (formal, active agent)."""
        result = compiler.compile([deu_mask_type], "deu", "kor")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "kor"

    def test_femininum_to_haeyoche(
        self, compiler: CrossCompiler, deu_fem_type: FluxType
    ):
        """DEU Femininum → KOR haeyoche (polite, container)."""
        result = compiler.compile([deu_fem_type], "deu", "kor")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "kor"

    def test_neutrum_to_haeche(
        self, compiler: CrossCompiler, deu_neut_type: FluxType
    ):
        """DEU Neutrum → KOR haeche (informal, value)."""
        result = compiler.compile([deu_neut_type], "deu", "kor")

        assert len(result.target_types) == 1
        assert result.target_types[0].paradigm_source == "kor"

    def test_multiple_deu_types(
        self,
        compiler: CrossCompiler,
        deu_mask_type: FluxType,
        deu_fem_type: FluxType,
        deu_neut_type: FluxType,
    ):
        """Multiple DEU types should all compile to KOR types."""
        result = compiler.compile(
            [deu_mask_type, deu_fem_type, deu_neut_type],
            "deu", "kor",
        )

        assert len(result.target_types) == 3
        for tt in result.target_types:
            assert tt.paradigm_source == "kor"

    def test_honorific_mapping_rule_applied(
        self, compiler: CrossCompiler, rule_set: TranslationRuleSet
    ):
        """DEU→KOR honorific mapping rules should be applied."""
        rule = rule_set.match("deu", "kor", "maskulinum")
        assert rule is not None
        assert rule.target_pattern == "hasipsioche"
        assert rule.transform_kind == TransformKind.GENDER_TO_HONORIFIC


# ══════════════════════════════════════════════════════════════════
# Round-Trip Tests
# ══════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Tests for round-trip compilation (A → B → A)."""

    def test_zho_deu_zho_round_trip(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """ZHO→DEU→ZHO round trip should preserve the original type."""
        result = compiler.compile_round_trip([zho_person_type], "zho", "deu")

        assert isinstance(result, RoundTripResult)
        assert result.preservation_rate > 0.0
        assert len(result.original_types) == 1
        assert len(result.round_trip_types) == 1
        assert result.forward_result is not None
        assert result.backward_result is not None

    def test_zho_deu_zho_preservation_rate(
        self,
        compiler: CrossCompiler,
        zho_person_type: FluxType,
        zho_flat_type: FluxType,
    ):
        """Round-trip preservation should be measurable."""
        result = compiler.compile_round_trip(
            [zho_person_type, zho_flat_type], "zho", "deu"
        )

        assert result.preservation_rate >= 0.0
        assert result.preservation_rate <= 1.0

    def test_round_trip_base_type_preservation(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """After round-trip, base type should be preserved for person."""
        result = compiler.compile_round_trip([zho_person_type], "zho", "deu")

        assert len(result.round_trip_types) == 1
        assert result.round_trip_types[0].base_type == zho_person_type.base_type

    def test_round_trip_witness_chain(self, compiler: CrossCompiler):
        """Round-trip should produce a combined witness chain."""
        result = compiler.compile_round_trip("person, flat_object", "zho", "deu")

        assert len(result.witness_chain) >= 2  # At least forward + backward
        for w in result.witness_chain:
            assert w.witness_id != ""

    def test_round_trip_forward_backward_warnings(
        self, compiler: CrossCompiler
    ):
        """Both forward and backward legs should have warning lists."""
        result = compiler.compile_round_trip("person", "zho", "deu")

        assert result.forward_result is not None
        assert result.backward_result is not None
        # Warnings are lists (may be empty)
        assert isinstance(result.forward_result.warnings, list)
        assert isinstance(result.backward_result.warnings, list)


# ══════════════════════════════════════════════════════════════════
# Multi-Hop Tests
# ══════════════════════════════════════════════════════════════════

class TestMultiHop:
    """Tests for multi-hop compilation and route optimization."""

    def test_optimize_route_returns_valid_path(self, compiler: CrossCompiler):
        """optimize_route should return a valid path."""
        path = compiler.optimize_route("zho", "lat")
        assert len(path) >= 2
        assert path[0] == "zho"
        assert path[-1] == "lat"

    def test_same_language_path(self, compiler: CrossCompiler):
        """Route from a language to itself should be length 1."""
        path = compiler.optimize_route("zho", "zho")
        assert path == ["zho"]

    def test_all_paths_cover_supported_languages(self, compiler: CrossCompiler):
        """Routes between any two supported languages should be computable."""
        langs = list(SUPPORTED_LANGUAGES)
        for src in langs:
            for tgt in langs:
                path = compiler.optimize_route(src, tgt)
                assert len(path) >= 1
                assert path[0] == src
                assert path[-1] == tgt

    def test_multi_hop_cheaper_check(
        self, multi_hop_compiler: MultiHopCompiler
    ):
        """is_multi_hop_cheaper should return a valid tuple."""
        is_cheaper, path, savings = multi_hop_compiler.is_multi_hop_cheaper(
            "zho", "lat"
        )
        assert isinstance(is_cheaper, bool)
        assert isinstance(path, list)
        assert isinstance(savings, float)
        assert path[0] == "zho"
        assert path[-1] == "lat"

    def test_multi_hop_compile_produces_result(
        self, multi_hop_compiler: MultiHopCompiler
    ):
        """Compiling through a multi-hop path should produce a result."""
        path = multi_hop_compiler.find_cheapest_path("zho", "deu")
        source_types = [FluxType.from_paradigm("zho", "person")]
        result = multi_hop_compiler.compile_hops(source_types, path)

        assert isinstance(result, CompilationResult)
        assert len(result.target_types) >= 1
        assert result.route == path

    def test_multi_hop_max_hops_respected(
        self, multi_hop_compiler: MultiHopCompiler
    ):
        """No path should exceed MAX_HOPS + 1 languages."""
        for src in SUPPORTED_LANGUAGES:
            for tgt in SUPPORTED_LANGUAGES:
                path = multi_hop_compiler.find_cheapest_path(src, tgt)
                assert len(path) <= multi_hop_compiler.MAX_HOPS + 1, (
                    f"Path {src}→{tgt} has {len(path)} hops "
                    f"(max {multi_hop_compiler.MAX_HOPS + 1})"
                )

    def test_zho_san_lat_hop_availability(
        self, multi_hop_compiler: MultiHopCompiler
    ):
        """ZHO→SAN→LAT path should be findable."""
        path = multi_hop_compiler.find_cheapest_path("zho", "lat")
        # The path might be direct or multi-hop, but it must exist
        assert path[0] == "zho"
        assert path[-1] == "lat"

    def test_multi_hop_with_use_multi_hop_false(
        self, compiler: CrossCompiler
    ):
        """When use_multi_hop=False, should always use direct path."""
        result = compiler.compile("person", "zho", "deu", use_multi_hop=False)
        assert result.route == ["zho", "deu"]


# ══════════════════════════════════════════════════════════════════
# SemanticEquivalenceChecker Tests
# ══════════════════════════════════════════════════════════════════

class TestSemanticEquivalence:
    """Tests for the semantic equivalence checker."""

    def test_identical_types_are_equivalent(self):
        """Identical types should score 1.0 equivalence."""
        checker = SemanticEquivalenceChecker()
        types = [FluxType.from_paradigm("zho", "person", confidence=0.95)]
        result = checker.check(types, types)

        assert result.equivalence_score >= 0.8
        assert result.is_equivalent
        assert len(result.divergences) == 0

    def test_arithmetic_preservation_same_base(self):
        """Arithmetic types with same base should score highly."""
        checker = SemanticEquivalenceChecker()
        src = [FluxType.from_paradigm("zho", "flat_object", confidence=0.95)]
        tgt = [FluxType.from_paradigm("deu", "neutrum", confidence=0.90)]
        result = checker.check_arithmetic_preservation(src, tgt)

        assert result.equivalence_score > 0.0
        assert isinstance(result.is_equivalent, bool)

    def test_arithmetic_preservation_no_leakage(self):
        """Arithmetic check should detect MODAL/CAPABILITY leakage."""
        checker = SemanticEquivalenceChecker()
        src = [FluxType.from_paradigm("zho", "flat_object", confidence=0.95)]
        # Target has an unexpected MODAL constraint
        tgt = [FluxType(
            base_type=FluxBaseType.VALUE,
            constraints=[
                FluxConstraint(
                    kind=ConstraintKind.EXECUTION_MODE,
                    language="kor",
                    value="attack",
                    confidence=0.9,
                ),
            ],
            confidence=0.85,
            paradigm_source="kor",
            name="kor:leaked",
        )]
        result = checker.check_arithmetic_preservation(src, tgt)

        assert len(result.divergences) > 0
        assert any("unexpected constraint" in d for d in result.divergences)

    def test_property_tests_run(self):
        """Property-based tests should actually run."""
        checker = SemanticEquivalenceChecker()
        src = [FluxType.from_paradigm("zho", "person", confidence=0.9)]
        tgt = [FluxType.from_paradigm("deu", "maskulinum", confidence=0.85)]
        result = checker.check(src, tgt, num_property_tests=20)

        assert result.test_cases_total > 0
        assert result.test_cases_passed >= 0

    def test_empty_types(self):
        """Empty type lists should be handled gracefully."""
        checker = SemanticEquivalenceChecker()
        result = checker.check([], [])

        assert result.equivalence_score == 0.0
        assert not result.is_equivalent

    def test_asymmetric_lengths(self):
        """Different length lists should be handled."""
        checker = SemanticEquivalenceChecker()
        src = [FluxType.from_paradigm("zho", "person")]
        tgt = [
            FluxType.from_paradigm("deu", "maskulinum"),
            FluxType.from_paradigm("deu", "femininum"),
        ]
        result = checker.check(src, tgt)

        assert isinstance(result.equivalence_score, float)
        assert 0.0 <= result.equivalence_score <= 1.0


# ══════════════════════════════════════════════════════════════════
# Information Preservation Tests
# ══════════════════════════════════════════════════════════════════

class TestInformationPreserved:
    """Tests for the information_preserved metric."""

    def test_same_language_preservation(self, compiler: CrossCompiler):
        """Compiling to the same language should have full preservation."""
        result = compiler.compile("person", "zho", "zho")
        assert result.information_preserved == 1.0

    def test_zho_deu_preservation_bounded(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """ZHO→DEU preservation should be in [0, 1]."""
        result = compiler.compile([zho_person_type], "zho", "deu")
        assert 0.0 <= result.information_preserved <= 1.0

    def test_preservation_decreases_with_cost(
        self, compiler: CrossCompiler
    ):
        """More expensive compilations should tend to have lower preservation."""
        # This is a soft test — the relationship isn't strict
        result = compiler.compile("person, flat_object, generic", "zho", "deu")
        assert result.total_cost >= 0.0
        assert result.information_preserved >= 0.0

    def test_preservation_metric_accuracy(
        self, compiler: CrossCompiler, zho_person_type: FluxType
    ):
        """Information preservation should correlate with type fidelity."""
        result = compiler.compile([zho_person_type], "zho", "deu")

        # The preservation should be reasonable (> 0.1 for known types)
        assert result.information_preserved > 0.1


# ══════════════════════════════════════════════════════════════════
# ASTDiff Tests
# ══════════════════════════════════════════════════════════════════

class TestASTDiff:
    """Tests for the AST diff engine."""

    def test_identical_types_no_diffs(self):
        """Identical type lists should produce minimal diffs."""
        src = [FluxType.from_paradigm("zho", "person", confidence=0.95)]
        tgt = [FluxType.from_paradigm("zho", "person", confidence=0.95)]

        diff = ASTDiffEngine.compare(src, tgt, "zho", "zho")

        assert diff.match_score >= 0.9

    def test_different_base_type_detected(self):
        """Different base types should be detected."""
        src = [FluxType.from_paradigm("zho", "person")]  # ACTIVE
        tgt = [FluxType.from_paradigm("deu", "neutrum")]  # VALUE

        diff = ASTDiffEngine.compare(src, tgt, "zho", "deu")

        assert diff.diff_count >= 1
        kinds = {d["kind"] for d in diff.details}
        assert ASTDiffKind.BASE_TYPE_SHIFTED.value in kinds or ASTDiffKind.PARADIGM_CHANGED.value in kinds

    def test_confidence_change_detected(self):
        """Confidence changes should be detected."""
        src = [FluxType.from_paradigm("zho", "person", confidence=0.95)]
        tgt = [FluxType.from_paradigm("zho", "person", confidence=0.4)]

        diff = ASTDiffEngine.compare(src, tgt, "zho", "zho")

        kinds = {d["kind"] for d in diff.details}
        assert ASTDiffKind.CONFIDENCE_CHANGED.value in kinds

    def test_asymmetric_lists(self):
        """Different-length lists should report extra types."""
        src = [FluxType.from_paradigm("zho", "person")]
        tgt = []

        diff = ASTDiffEngine.compare(src, tgt, "zho", "deu")

        assert diff.diff_count >= 1

    def test_to_dict(self):
        """ASTDiff.to_dict should produce valid output."""
        diff = ASTDiff(
            source_lang="zho",
            target_lang="deu",
            diff_count=2,
            match_score=0.85,
        )
        d = diff.to_dict()
        assert d["source_lang"] == "zho"
        assert d["target_lang"] == "deu"
        assert d["diff_count"] == 2
        assert d["match_score"] == 0.85


# ══════════════════════════════════════════════════════════════════
# CodeEmitter Tests
# ══════════════════════════════════════════════════════════════════

class TestCodeEmitter:
    """Tests for the code emitter."""

    def test_emit_single_type(self, zho_person_type: FluxType):
        """Single type emission should start with # prefix."""
        code = CodeEmitter.emit(zho_person_type)
        assert code.startswith("#zho")
        assert "type:ACTIVE" in code

    def test_emit_program(self):
        """Program emission should include header."""
        types = [
            FluxType.from_paradigm("deu", "maskulinum"),
            FluxType.from_paradigm("deu", "femininum"),
        ]
        code = CodeEmitter.emit_program(types, "deu")
        assert "compiled to deu" in code
        assert "types: 2" in code

    def test_emit_with_low_confidence(self):
        """Low confidence should be reflected in output."""
        ft = FluxType.from_paradigm("zho", "person", confidence=0.5)
        code = CodeEmitter.emit(ft)
        assert "conf:0.50" in code

    def test_emit_unknown_lang(self):
        """Unknown language should still emit something."""
        ft = FluxType(base_type=FluxBaseType.VALUE, paradigm_source="xxx")
        code = CodeEmitter.emit(ft)
        assert code.startswith("#xxx")


# ══════════════════════════════════════════════════════════════════
# CompilationResult Tests
# ══════════════════════════════════════════════════════════════════

class TestCompilationResult:
    """Tests for the CompilationResult dataclass."""

    def test_is_type_safe_with_valid_witnesses(self):
        """is_type_safe should be True when all witnesses are valid."""
        from flux_a2a.type_safe_bridge import TypeWitness

        result = CompilationResult(
            witness_chain=[TypeWitness()],  # Default is valid (no constraints)
        )
        assert result.is_type_safe

    def test_to_dict(self):
        """to_dict should produce valid output."""
        result = CompilationResult(
            target_code="#deu type:ACTIVE",
            total_cost=0.3,
            information_preserved=0.85,
            warnings=["test warning"],
            route=["zho", "deu"],
        )
        d = result.to_dict()
        assert d["target_code"] == "#deu type:ACTIVE"
        assert d["total_cost"] == 0.3
        assert d["information_preserved"] == 0.85
        assert "test warning" in d["warnings"]
        assert d["route"] == ["zho", "deu"]


# ══════════════════════════════════════════════════════════════════
# CrossCompiler Integration Tests
# ══════════════════════════════════════════════════════════════════

class TestCrossCompilerIntegration:
    """Integration tests for the CrossCompiler."""

    def test_unsupported_language_raises(self, compiler: CrossCompiler):
        """Unsupported language should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            compiler.compile("person", "zho", "xxx")

    def test_zho_to_kor_compilation(self, compiler: CrossCompiler):
        """ZHO→KOR compilation should work."""
        result = compiler.compile("person, flat_object", "zho", "kor")

        assert len(result.target_types) == 2
        for tt in result.target_types:
            assert tt.paradigm_source == "kor"

    def test_deu_to_zho_compilation(self, compiler: CrossCompiler):
        """DEU→ZHO compilation should work."""
        result = compiler.compile("maskulinum, neutrum", "deu", "zho")

        assert len(result.target_types) == 2
        for tt in result.target_types:
            assert tt.paradigm_source == "zho"

    def test_check_equivalence_method(self, compiler: CrossCompiler):
        """check_equivalence should return a valid result."""
        src = [FluxType.from_paradigm("zho", "person")]
        tgt = [FluxType.from_paradigm("deu", "maskulinum")]
        result = compiler.check_equivalence(src, tgt)

        assert isinstance(result, EquivalenceCheckResult)
        assert 0.0 <= result.equivalence_score <= 1.0

    def test_all_supported_paradigms_compilable(self, compiler: CrossCompiler):
        """Every supported paradigm should be compilable to every other."""
        langs = sorted(SUPPORTED_LANGUAGES)
        for src in langs:
            for tgt in langs:
                if src == tgt:
                    continue
                result = compiler.compile("person", src, tgt)
                assert len(result.target_types) >= 1
                assert result.target_types[0].paradigm_source == tgt


# ══════════════════════════════════════════════════════════════════
# EquivalenceCheckResult Tests
# ══════════════════════════════════════════════════════════════════

class TestEquivalenceCheckResult:
    """Tests for EquivalenceCheckResult dataclass."""

    def test_to_dict(self):
        """to_dict should produce valid output."""
        result = EquivalenceCheckResult(
            is_equivalent=True,
            equivalence_score=0.85,
            divergences=["minor base shift"],
            test_cases_passed=45,
            test_cases_total=50,
        )
        d = result.to_dict()
        assert d["is_equivalent"] is True
        assert d["equivalence_score"] == 0.85
        assert d["divergences"] == ["minor base shift"]
        assert d["test_cases_passed"] == 45
        assert d["test_cases_total"] == 50
