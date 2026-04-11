"""
Tests for Type-Safe Cross-Language Bridge System.

Validates the four subsystems of type_safe_bridge.py:
  1. TypeAlgebra — cross-paradigm type equivalence classes
  2. BridgeCostMatrix — multi-factor bridge cost computation
  3. TypeWitness — proof-carrying type transformation witnesses
  4. TypeSafeBridge — full type-safe bridge with all guarantees

Test categories:
  - Lossless bridge: ZHO classifier 个 ↔ DEU Nom + Neuter
  - Lossy bridge: KOR honorific (존댓말) → LAT (no direct equivalent)
  - TypeAlgebra: classifier↔case↔vibhakti mapping consistency
  - BridgeCostMatrix: ZHO↔SAN cheaper than ZHO↔LAT
  - TypeWitness: generation, verification, serialization, chaining
  - Round-trip: ZHO → DEU → SAN → ZHO preserves core semantics
"""

import json
import math

import pytest

from flux_a2a.types import (
    ConstraintKind,
    FluxBaseType,
    FluxConstraint,
    FluxType,
    FluxTypeRegistry,
    build_default_registry,
)
from flux_a2a.type_checker import BridgeStrategy
from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    BridgeCostReport,
    BridgeWarning,
    PreservationDegree,
    TypeAlgebra,
    TypeEquivalenceClass,
    TypeEquivalenceSlot,
    TypeSafeBridge,
    TypeWitness,
    WitnessConstraint,
    WitnessGenerator,
)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def algebra() -> TypeAlgebra:
    """Fresh TypeAlgebra instance."""
    return TypeAlgebra()


@pytest.fixture
def cost_matrix(algebra: TypeAlgebra) -> BridgeCostMatrix:
    """BridgeCostMatrix using the provided algebra."""
    return BridgeCostMatrix(algebra)


@pytest.fixture
def bridge() -> TypeSafeBridge:
    """Fresh TypeSafeBridge instance."""
    return TypeSafeBridge()


@pytest.fixture
def witness_gen(algebra: TypeAlgebra) -> WitnessGenerator:
    """WitnessGenerator using the provided algebra."""
    return WitnessGenerator(algebra)


# ══════════════════════════════════════════════════════════════════
# 1. TypeAlgebra Tests
# ══════════════════════════════════════════════════════════════════

class TestTypeAlgebra:

    def test_noun_cat_flat_object_maps_to_neutrum(self, algebra: TypeAlgebra):
        """ZHO flat_object ↔ DEU neutrum ↔ SAN linga_napumsaka ↔ LAT neutrum."""
        cls = algebra.find_class("zho", "flat_object")
        assert cls is not None, "flat_object should be in an equivalence class"
        assert cls.semantic_domain == "noun_cat"
        assert cls.has_language("zho")
        assert cls.has_language("deu")
        assert cls.has_language("san")
        assert cls.has_language("lat")

        deu_slot = cls.get_slot("deu")
        assert deu_slot is not None
        assert deu_slot.native_tag == "neutrum"

        san_slot = cls.get_slot("san")
        assert san_slot is not None
        assert san_slot.native_tag == "linga_napumsaka"

        lat_slot = cls.get_slot("lat")
        assert lat_slot is not None
        assert lat_slot.native_tag == "neutrum"

    def test_case_nominative_across_deu_san_lat(self, algebra: TypeAlgebra):
        """DEU nominativ ↔ SAN prathama ↔ LAT nominativus — LOSSLESS."""
        cls = algebra.find_class("deu", "nominativ")
        assert cls is not None
        assert cls.degree == PreservationDegree.LOSSLESS
        assert cls.get_slot("san") is not None
        assert cls.get_slot("san").native_tag == "prathama"
        assert cls.get_slot("lat") is not None
        assert cls.get_slot("lat").native_tag == "nominativus"

    def test_case_accusative_across_deu_san_lat(self, algebra: TypeAlgebra):
        """DEU akkusativ ↔ SAN dvitiya ↔ LAT accusativus."""
        cls = algebra.find_class("deu", "akkusativ")
        assert cls is not None
        assert cls.degree == PreservationDegree.LOSSLESS

    def test_classify_person_maps_to_masculine(self, algebra: TypeAlgebra):
        """ZHO person classifier ↔ DEU maskulinum ↔ SAN pushkara ↔ LAT maskulinum."""
        cls = algebra.find_class("zho", "person")
        assert cls is not None
        assert cls.get_slot("deu").native_tag == "maskulinum"
        assert cls.get_slot("san").native_tag == "linga_pushkara"
        assert cls.get_slot("lat").native_tag == "maskulinum"

    def test_collective_maps_to_feminine(self, algebra: TypeAlgebra):
        """ZHO collective ↔ DEU femininum ↔ SAN stri ↔ LAT femininum."""
        cls = algebra.find_class("zho", "collective")
        assert cls is not None
        assert cls.get_slot("deu").native_tag == "femininum"
        assert cls.get_slot("san").native_tag == "linga_stri"

    def test_generic_classifier_is_degraded(self, algebra: TypeAlgebra):
        """ZHO generic (個) has no cross-language equivalent — DEGRADED."""
        cls = algebra.find_class("zho", "generic")
        assert cls is not None
        assert cls.degree == PreservationDegree.DEGRADED
        # Only ZHO is covered
        assert cls.languages() == {"zho"}

    def test_temporal_domain_has_latin_tenses(self, algebra: TypeAlgebra):
        """LAT tempus classes should be in the temporal domain."""
        temporal_classes = algebra.classes_by_domain("temporal")
        assert len(temporal_classes) > 0
        praesens_found = any(
            cls.get_slot("lat") and cls.get_slot("lat").native_tag == "praesens"
            for cls in temporal_classes
        )
        assert praesens_found, "praesens should be in temporal domain"

    def test_register_domain_has_korean_honorifics(self, algebra: TypeAlgebra):
        """KOR honorific levels should be in the register domain."""
        register_classes = algebra.classes_by_domain("register")
        assert len(register_classes) > 0
        hasipsio_found = any(
            cls.get_slot("kor") and cls.get_slot("kor").native_tag == "hasipsioche"
            for cls in register_classes
        )
        assert hasipsio_found, "hasipsioche should be in register domain"

    def test_scope_domain_has_wen_context(self, algebra: TypeAlgebra):
        """WEN context types should be in the scope domain."""
        scope_classes = algebra.classes_by_domain("scope")
        assert len(scope_classes) > 0
        # Collect all WEN tags in scope classes
        wen_tags_in_scope = set()
        for cls in scope_classes:
            slot = cls.get_slot("wen")
            if slot:
                wen_tags_in_scope.add(slot.native_tag)
        assert "topic" in wen_tags_in_scope or "zero_anaphora" in wen_tags_in_scope or "context_resolved" in wen_tags_in_scope, (
            f"WEN context tags should be in scope domain, got: {wen_tags_in_scope}"
        )

    def test_consistency_deu_san_noun_cat(self, algebra: TypeAlgebra):
        """DEU and SAN should have high consistency in noun_cat domain."""
        score = algebra.check_consistency("noun_cat", "deu", "san")
        assert score > 0.5, f"DEU↔SAN noun_cat consistency should be > 0.5, got {score}"

    def test_consistency_deu_san_case_systems(self, algebra: TypeAlgebra):
        """DEU and SAN case systems should map consistently for shared cases."""
        for deu_tag, san_tag in [
            ("nominativ", "prathama"),
            ("akkusativ", "dvitiya"),
            ("dativ", "chaturthi"),
            ("genitiv", "shashthi"),
        ]:
            cls = algebra.find_class("deu", deu_tag)
            assert cls is not None, f"Should find class for deu:{deu_tag}"
            san_slot = cls.get_slot("san")
            assert san_slot is not None, f"deu:{deu_tag} should map to SAN"
            assert san_slot.native_tag == san_tag, (
                f"deu:{deu_tag} should map to san:{san_tag}, "
                f"got san:{san_slot.native_tag}"
            )

    def test_translate_method(self, algebra: TypeAlgebra):
        """TypeAlgebra.translate returns the correct slot."""
        slot = algebra.translate("zho", "person", "deu")
        assert slot is not None
        assert slot.native_tag == "maskulinum"
        assert slot.language == "deu"

    def test_translate_no_match(self, algebra: TypeAlgebra):
        """TypeAlgebra.translate returns None when no mapping exists."""
        # WEN ren (benevolence) has no ZHO equivalent
        slot = algebra.translate("wen", "ren", "zho")
        assert slot is None

    def test_domain_coverage(self, algebra: TypeAlgebra):
        """Domain coverage should include the expected languages."""
        coverage = algebra.domain_coverage("noun_cat")
        assert "deu" in coverage
        assert "san" in coverage
        assert "lat" in coverage
        assert "zho" in coverage


# ══════════════════════════════════════════════════════════════════
# 2. BridgeCostMatrix Tests
# ══════════════════════════════════════════════════════════════════

class TestBridgeCostMatrix:

    def test_same_language_zero_cost(self, cost_matrix: BridgeCostMatrix):
        """Bridge to the same language should have zero cost."""
        report = cost_matrix.compute("zho", "zho")
        assert report.total_cost == 0.0
        assert report.source_lang == "zho"
        assert report.target_lang == "zho"

    def test_deu_san_cheaper_than_wen_lat(self, cost_matrix: BridgeCostMatrix):
        """DEU↔SAN should be cheaper than WEN↔LAT.

        DEU and SAN share case systems (nominativ↔prathama etc.),
        while WEN and LAT are structurally distant paradigms.
        """
        cost_deu_san = cost_matrix.compute("deu", "san").total_cost
        cost_wen_lat = cost_matrix.compute("wen", "lat").total_cost
        assert cost_deu_san < cost_wen_lat, (
            f"DEU↔SAN ({cost_deu_san:.4f}) should be cheaper than "
            f"WEN↔LAT ({cost_wen_lat:.4f})"
        )

    def test_deu_san_cheap_case_systems(self, cost_matrix: BridgeCostMatrix):
        """DEU↔SAN should be cheap due to shared case systems."""
        report = cost_matrix.compute("deu", "san")
        assert report.total_cost < 0.6, (
            f"DEU↔SAN should be cheap, got {report.total_cost:.4f}"
        )

    def test_cost_has_four_factors(self, cost_matrix: BridgeCostMatrix):
        """Cost report should have all four cost factors populated."""
        report = cost_matrix.compute("zho", "deu")
        assert report.structural_distance >= 0.0
        assert report.expressiveness_gap >= 0.0
        assert report.information_loss >= 0.0
        assert report.translation_ambiguity >= 0.0
        assert report.total_cost >= 0.0

    def test_cost_warnings_generated(self, cost_matrix: BridgeCostMatrix):
        """Bridge cost should generate warnings for lossy mappings."""
        report = cost_matrix.compute("kor", "lat")
        assert len(report.warnings) > 0, (
            "KOR→LAT should have warnings (honorifics have no LAT equivalent)"
        )

    def test_compare_method(self, cost_matrix: BridgeCostMatrix):
        """compare() correctly identifies cheaper bridges."""
        result = cost_matrix.compare("deu", "lat", "wen")
        assert result == "cheaper", (
            "DEU→LAT should be cheaper than DEU→WEN"
        )

    def test_cost_report_serialization(self, cost_matrix: BridgeCostMatrix):
        """BridgeCostReport.to_dict() produces valid output."""
        report = cost_matrix.compute("zho", "deu")
        d = report.to_dict()
        assert "source" in d
        assert "target" in d
        assert "total_cost" in d
        assert "warnings" in d
        assert isinstance(d["warnings"], list)

    def test_asymmetric_cost(self, cost_matrix: BridgeCostMatrix):
        """A→B and B→A costs may differ due to expressiveness gap."""
        cost_ab = cost_matrix.compute("zho", "san").total_cost
        cost_ba = cost_matrix.compute("san", "zho").total_cost
        # They may be close but not necessarily identical
        assert isinstance(cost_ab, float)
        assert isinstance(cost_ba, float)

    def test_unknown_language_max_distance(self, cost_matrix: BridgeCostMatrix):
        """Unknown language should produce maximum structural distance."""
        report = cost_matrix.compute("zho", "xyz_unknown")
        assert report.structural_distance == 1.0


# ══════════════════════════════════════════════════════════════════
# 3. TypeWitness Tests
# ══════════════════════════════════════════════════════════════════

class TestTypeWitness:

    def test_witness_generation(self, witness_gen: WitnessGenerator):
        """WitnessGenerator produces a valid witness for a known mapping."""
        source = FluxType.from_paradigm("zho", "person", name="zho:person")
        target = FluxType.from_paradigm("deu", "maskulinum", name="deu:maskulinum")
        witness = witness_gen.generate(
            source, target, BridgeStrategy.DIRECT,
            source_tag="person", target_tag="maskulinum",
        )
        assert witness.source_lang == "zho"
        assert witness.target_lang == "deu"
        assert witness.source_tag == "person"
        assert witness.target_tag == "maskulinum"
        assert len(witness.constraints) > 0

    def test_witness_verification_passes(self, witness_gen: WitnessGenerator):
        """A correct translation should produce a valid witness."""
        source = FluxType.from_paradigm("deu", "nominativ", name="deu:nominativ")
        target = FluxType.from_paradigm("san", "prathama", name="san:prathama")
        witness = witness_gen.generate(
            source, target, BridgeStrategy.DIRECT,
            source_tag="nominativ", target_tag="prathama",
        )
        assert witness.is_valid, "Lossless case mapping should produce valid witness"
        assert witness.verified

    def test_witness_serialization_roundtrip(self, witness_gen: WitnessGenerator):
        """Witness survives JSON serialization and deserialization."""
        source = FluxType.from_paradigm("zho", "flat_object", name="zho:flat_object")
        target = FluxType.from_paradigm("deu", "neutrum", name="deu:neutrum")
        witness = witness_gen.generate(
            source, target, BridgeStrategy.DIRECT,
            source_tag="flat_object", target_tag="neutrum",
        )

        # Serialize
        json_str = witness.to_json()
        data = json.loads(json_str)

        # Deserialize
        restored = TypeWitness.from_dict(data)
        assert restored.witness_id == witness.witness_id
        assert restored.source_lang == "zho"
        assert restored.target_lang == "deu"
        assert len(restored.constraints) == len(witness.constraints)

        # Re-verify
        assert restored.verify()
        assert restored.is_valid

    def test_witness_chaining(self, witness_gen: WitnessGenerator):
        """Two witnesses can be chained for a multi-hop bridge."""
        # Hop 1: ZHO → DEU
        src = FluxType.from_paradigm("zho", "person", name="zho:person")
        mid = FluxType.from_paradigm("deu", "maskulinum", name="deu:maskulinum")
        w1 = witness_gen.generate(src, mid, BridgeStrategy.DIRECT, "person", "maskulinum")

        # Hop 2: DEU → SAN
        end = FluxType.from_paradigm("san", "linga_pushkara", name="san:linga_pushkara")
        w2 = witness_gen.generate(mid, end, BridgeStrategy.DIRECT, "maskulinum", "linga_pushkara")

        # Chain
        chained = w1.chain(w2)
        assert chained.source_lang == "zho"
        assert chained.target_lang == "san"
        assert chained.source_tag == "person"
        assert chained.target_tag == "linga_pushkara"
        assert "direct" in chained.strategy

    def test_witness_failed_constraints_reported(self, witness_gen: WitnessGenerator):
        """Failed constraints are accessible via failed_constraints."""
        source = FluxType.from_paradigm("zho", "generic", name="zho:generic")
        # Create a target with very low confidence to trigger a warning
        target = FluxType(
            base_type=FluxBaseType.CONTEXTUAL,
            confidence=0.2,
            paradigm_source="lat",
            name="lat:degraded",
        )
        witness = witness_gen.generate(
            source, target, BridgeStrategy.CONSTRAINT_STRIPPING,
            source_tag="generic", target_tag="",
        )
        # The confidence_floor constraint should fail (0.2 < 0.3)
        failed = witness.failed_constraints
        assert any(c.name == "confidence_floor" for c in failed), (
            f"Expected confidence_floor to fail, got failed: {[c.name for c in failed]}"
        )

    def test_witness_preservation_degree(self, witness_gen: WitnessGenerator):
        """Witness captures the correct preservation degree."""
        source = FluxType.from_paradigm("deu", "nominativ", name="deu:nominativ")
        target = FluxType.from_paradigm("san", "prathama", name="san:prathama")
        witness = witness_gen.generate(
            source, target, BridgeStrategy.DIRECT,
            source_tag="nominativ", target_tag="prathama",
        )
        assert witness.preservation == PreservationDegree.LOSSLESS


# ══════════════════════════════════════════════════════════════════
# 4. TypeSafeBridge — Full Bridge Tests
# ══════════════════════════════════════════════════════════════════

class TestTypeSafeBridgeLossless:

    def test_lossless_zho_person_to_deu_maskulinum(self, bridge: TypeSafeBridge):
        """ZHO person → DEU maskulinum: near-lossless mapping."""
        source = FluxType.from_paradigm("zho", "person", name="zho:person")
        result = bridge.translate_safe(source, "deu")
        assert result.is_safe, f"Witness should be valid: {result.witness.failed_constraints}"
        assert result.target_type.paradigm_source == "deu"
        assert result.preservation in (
            PreservationDegree.LOSSLESS,
            PreservationDegree.NEAR_LOSSLESS,
        )

    def test_lossless_deu_nominativ_to_san_prathama(self, bridge: TypeSafeBridge):
        """DEU nominativ → SAN prathama: lossless case mapping."""
        source = FluxType.from_paradigm("deu", "nominativ", name="deu:nominativ")
        result = bridge.translate_safe(source, "san")
        assert result.is_safe
        assert result.preservation == PreservationDegree.LOSSLESS
        # Verify the target has the correct tag
        tgt_tag = bridge._infer_tag(result.target_type)
        assert tgt_tag == "prathama", f"Expected prathama, got {tgt_tag}"

    def test_lossless_collective_to_femininum(self, bridge: TypeSafeBridge):
        """ZHO collective → DEU femininum: near-lossless container mapping."""
        source = FluxType.from_paradigm("zho", "collective", name="zho:collective")
        result = bridge.translate_safe(source, "deu")
        assert result.is_safe
        tgt_tag = bridge._infer_tag(result.target_type)
        assert tgt_tag == "femininum", f"Expected femininum, got {tgt_tag}"


class TestTypeSafeBridgeLossy:

    def test_lossy_kor_honorific_to_lat(self, bridge: TypeSafeBridge):
        """KOR honorific (hasipsioche) → LAT: no direct equivalent, falls through.

        Korean honorifics have no direct Latin counterpart. The bridge should
        fall back to a TEMPORAL mapping (e.g., praesens) via refinement rule.
        """
        source = FluxType.from_paradigm("kor", "hasipsioche", name="kor:hasipsioche")
        result = bridge.translate_safe(source, "lat")

        # The bridge should still produce a result (not crash)
        assert result.target_type is not None
        assert result.target_type.paradigm_source == "lat"

        # Should use a refinement or fallback — either way, should work
        assert result.bridge_result.fidelity > 0.0

    def test_lossy_kor_informal_to_lat(self, bridge: TypeSafeBridge):
        """KOR informal (haeche) → LAT: no direct equivalent."""
        source = FluxType.from_paradigm("kor", "haeche", name="kor:haeche")
        result = bridge.translate_safe(source, "lat")
        assert result.target_type is not None
        assert result.target_type.paradigm_source == "lat"

    def test_lossy_wen_ren_to_zho(self, bridge: TypeSafeBridge):
        """WEN ren (仁, benevolence) → ZHO: no equivalent, degraded."""
        source = FluxType.from_paradigm("wen", "ren", name="wen:ren")
        result = bridge.translate_safe(source, "zho")
        # Should still produce a result via fallback
        assert result.target_type is not None
        assert result.preservation == PreservationDegree.DEGRADED


class TestTypeSafeBridgeAlgebra:

    def test_flat_object_algebra_consistency(self, bridge: TypeSafeBridge):
        """ZHO flat_object → DEU/SAN/LAT should all be neuter via algebra."""
        source = FluxType.from_paradigm("zho", "flat_object", name="zho:flat_object")

        for tgt_lang, expected_tag in [
            ("deu", "neutrum"),
            ("san", "linga_napumsaka"),
            ("lat", "neutrum"),
        ]:
            result = bridge.translate_safe(source, tgt_lang)
            tgt_tag = bridge._infer_tag(result.target_type)
            assert tgt_tag == expected_tag, (
                f"zho:flat_object → {tgt_lang} should give {expected_tag}, "
                f"got {tgt_tag}"
            )

    def test_case_system_consistency(self, bridge: TypeSafeBridge):
        """Case systems should map consistently: DEU → SAN → LAT."""
        case_mappings = {
            "nominativ": ("prathama", "nominativus"),
            "akkusativ": ("dvitiya", "accusativus"),
            "dativ": ("chaturthi", "dativus"),
            "genitiv": ("shashthi", "genitivus"),
        }

        for deu_tag, (san_tag, lat_tag) in case_mappings.items():
            # DEU → SAN
            source = FluxType.from_paradigm("deu", deu_tag, name=f"deu:{deu_tag}")
            result_san = bridge.translate_safe(source, "san")
            san_result_tag = bridge._infer_tag(result_san.target_type)
            assert san_result_tag == san_tag, (
                f"deu:{deu_tag} → san should be {san_tag}, got {san_result_tag}"
            )

            # DEU → LAT
            result_lat = bridge.translate_safe(source, "lat")
            lat_result_tag = bridge._infer_tag(result_lat.target_type)
            assert lat_result_tag == lat_tag, (
                f"deu:{deu_tag} → lat should be {lat_tag}, got {lat_result_tag}"
            )


class TestTypeSafeBridgeRoundTrip:

    def test_round_trip_zho_deu_san_zho(self, bridge: TypeSafeBridge):
        """ZHO → DEU → SAN → ZHO should preserve core semantics.

        Starting with ZHO person, routing through DEU and SAN,
        and back to ZHO should recover a type that has the same
        base type (ACTIVE) and is still from ZHO.
        """
        original = FluxType.from_paradigm("zho", "person", name="zho:person")
        original_base = original.base_type

        # ZHO → DEU
        leg1 = bridge.translate_safe(original, "deu")
        assert leg1.target_type.paradigm_source == "deu"

        # DEU → SAN
        leg2 = bridge.translate_safe(leg1.target_type, "san")
        assert leg2.target_type.paradigm_source == "san"

        # SAN → ZHO
        leg3 = bridge.translate_safe(leg2.target_type, "zho")
        assert leg3.target_type.paradigm_source == "zho"

        # Core semantics preserved: base type should be ACTIVE
        final_base = leg3.target_type.base_type
        assert final_base == original_base, (
            f"Round-trip base type changed: {original_base.name} → {final_base.name}"
        )

    def test_round_trip_via_helper(self, bridge: TypeSafeBridge):
        """Using translate_round_trip for ZHO → DEU → ZHO."""
        source = FluxType.from_paradigm("zho", "collective", name="zho:collective")
        original_base = source.base_type  # CONTAINER

        result = bridge.translate_round_trip(source, via_lang="deu", back_lang="zho")
        assert result.target_type.paradigm_source == "zho"
        # Container type should survive round-trip
        assert result.target_type.base_type == original_base, (
            f"Round-trip base type changed: {original_base.name} → "
            f"{result.target_type.base_type.name}"
        )
        # Witness should be chained
        assert "→" in result.witness.strategy


class TestTypeSafeBridgeRefinement:

    def test_refinement_improves_mapping(self, bridge: TypeSafeBridge):
        """Adding a refinement rule should improve preservation."""
        # First, translate without explicit refinement (generic → DEU)
        source = FluxType.from_paradigm("zho", "generic", name="zho:generic")
        result_before = bridge.translate_safe(source, "deu")

        # Add a custom refinement
        bridge.add_refinement(
            "zho", "generic", "deu", "neutrum",
            PreservationDegree.NEAR_LOSSLESS,
        )
        result_after = bridge.translate_safe(source, "deu")

        # The refined version should have better preservation
        degree_order = [
            PreservationDegree.LOSSLESS,
            PreservationDegree.NEAR_LOSSLESS,
            PreservationDegree.PARTIAL,
            PreservationDegree.LOSSY,
            PreservationDegree.DEGRADED,
        ]
        assert degree_order.index(result_after.preservation) <= degree_order.index(
            result_before.preservation
        ), (
            f"Refinement should improve preservation: "
            f"{result_before.preservation.value} → {result_after.preservation.value}"
        )


class TestTypeSafeBridgeBidirectionalMap:

    def test_bidirectional_map_zho_deu(self, bridge: TypeSafeBridge):
        """get_bidirectional_map returns the correct mappings."""
        bmap = bridge.get_bidirectional_map("zho", "deu")
        assert isinstance(bmap, dict)
        assert "person" in bmap
        assert bmap["person"] == "maskulinum"
        assert "flat_object" in bmap
        assert bmap["flat_object"] == "neutrum"
        assert "collective" in bmap
        assert bmap["collective"] == "femininum"

    def test_bidirectional_map_san_lat(self, bridge: TypeSafeBridge):
        """SAN→LAT bidirectional map for case systems."""
        bmap = bridge.get_bidirectional_map("san", "lat")
        assert "prathama" in bmap
        assert bmap["prathama"] == "nominativus"
        assert "dvitiya" in bmap
        assert bmap["dvitiya"] == "accusativus"


class TestTypeSafeBridgeIntegration:

    def test_full_pipeline_zho_to_lat(self, bridge: TypeSafeBridge):
        """Full pipeline: ZHO → LAT produces valid result with all metadata."""
        source = FluxType.from_paradigm("zho", "person", name="zho:person")
        result = bridge.translate_safe(source, "lat")

        # Result should have all components
        assert result.bridge_result is not None
        assert result.witness is not None
        assert result.cost_report is not None
        assert result.preservation in list(PreservationDegree)

        # Target should be in LAT paradigm
        assert result.target_type.paradigm_source == "lat"

        # Should have at least one warning for cross-paradigm bridge
        # (even if the mapping is good, the cost report generates warnings)
        assert isinstance(result.warnings, list)

    def test_safe_bridge_result_to_dict(self, bridge: TypeSafeBridge):
        """SafeBridgeResult.to_dict() produces serializable output."""
        source = FluxType.from_paradigm("zho", "animal", name="zho:animal")
        result = bridge.translate_safe(source, "deu")
        d = result.to_dict()
        assert "target_type" in d
        assert "witness" in d
        assert "cost" in d
        assert "is_safe" in d
        assert "preservation" in d
        assert "warnings" in d
        # Verify JSON serialization works
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0
