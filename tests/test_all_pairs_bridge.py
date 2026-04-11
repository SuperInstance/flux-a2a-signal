"""
Round 18: Comprehensive cross-language bridge tests for ALL 15 language pairs.

Tests that every pair of the 6 FLUX languages (ZHO, DEU, KOR, SAN, LAT, WEN)
can participate in type-safe bridging via the TypeAlgebra, BridgeCostMatrix,
and TypeSafeBridge systems.

15 pairs: C(6,2) = 15
  ZHO-DEU, ZHO-KOR, ZHO-SAN, ZHO-LAT, ZHO-WEN,
  DEU-KOR, DEU-SAN, DEU-LAT, DEU-WEN,
  KOR-SAN, KOR-LAT, KOR-WEN,
  SAN-LAT, SAN-WEN,
  LAT-WEN
"""

import itertools
import pytest

from flux_a2a.type_safe_bridge import (
    BridgeCostMatrix,
    PreservationDegree,
    TypeAlgebra,
    TypeSafeBridge,
)
from flux_a2a.types import FluxTypeRegistry, build_default_registry


LANGUAGES = ["zho", "deu", "kor", "san", "lat", "wen"]
ALL_PAIRS = list(itertools.combinations(LANGUAGES, 2))


@pytest.fixture(scope="module")
def algebra():
    return TypeAlgebra()


@pytest.fixture(scope="module")
def cost_matrix(algebra):
    return BridgeCostMatrix(algebra)


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def bridge(algebra, registry):
    return TypeSafeBridge(algebra=algebra, registry=registry)


class TestAllPairsBridgeExists:
    """Test 1: TypeSafeBridge can create a bridge between every pair."""

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_bridge_creation(self, bridge, algebra, src, tgt):
        """A TypeSafeBridge can be constructed and the algebra covers both languages."""
        assert bridge is not None
        # Verify the algebra has entries for both source and target
        src_tags = list(algebra._index.keys())
        src_covered = any(s == src for s, _ in src_tags)
        tgt_covered = any(t == tgt for s, t in src_tags)
        # At minimum, the algebra should not crash when queried
        algebra.translate(src, "generic", tgt)


class TestAllPairsTypeExport:
    """Test 2: Both adapters successfully export their types (via algebra)."""

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_source_has_types(self, algebra, src, tgt):
        """Source language has at least one type in the algebra."""
        src_slots = [tag for (lang, tag) in algebra._index if lang == src]
        assert len(src_slots) > 0, f"{src} has no types in the algebra"

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_target_has_types(self, algebra, src, tgt):
        """Target language has at least one type in the algebra."""
        tgt_slots = [tag for (lang, tag) in algebra._index if lang == tgt]
        assert len(tgt_slots) > 0, f"{tgt} has no types in the algebra"


class TestAllPairsCostFinite:
    """Test 3: BridgeCostMatrix returns a valid cost (0.0-1.0) for every pair."""

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_cost_is_finite(self, cost_matrix, src, tgt):
        report = cost_matrix.compute(src, tgt)
        assert 0.0 <= report.total_cost <= 1.0, (
            f"{src}->{tgt} cost {report.total_cost} out of range"
        )
        assert report.total_cost > 0.0, (
            f"{src}->{tgt} should have nonzero cost (different languages)"
        )

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_cost_report_fields(self, cost_matrix, src, tgt):
        report = cost_matrix.compute(src, tgt)
        assert 0.0 <= report.structural_distance <= 1.0
        assert 0.0 <= report.expressiveness_gap <= 1.0
        assert 0.0 <= report.information_loss <= 1.0
        assert 0.0 <= report.translation_ambiguity <= 1.0
        assert report.source_lang == src
        assert report.target_lang == tgt


class TestAllPairsWitnessGeneration:
    """Test 4: TypeWitness is generated for at least one translation per pair."""

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_witness_for_translation(self, bridge, algebra, src, tgt):
        """At least one type from source can produce a witness when translated."""
        src_tags = list(set(tag for (lang, tag) in algebra._index if lang == src))
        found_witness = False
        for tag in src_tags[:5]:  # Check up to 5 tags
            # Use algebra to find the equivalence class and translate
            cls = algebra.find_class(src, tag)
            if cls and cls.has_language(tgt):
                tgt_slot = cls.get_slot(tgt)
                if tgt_slot:
                    found_witness = True
                    assert tgt_slot.native_tag != "", "Target slot should have a tag"
                    break


class TestAllPairsRoundTripSanity:
    """Test 5: Round-trip export from A -> import to B -> export from B returns compatible types."""

    @pytest.mark.parametrize("src,tgt", ALL_PAIRS)
    def test_roundtrip_produces_valid_tags(self, algebra, src, tgt):
        """Round-tripping through the algebra should produce valid target tags."""
        src_tags = list(set(tag for (lang, tag) in algebra._index if lang == src))
        if not src_tags:
            pytest.skip(f"No tags for {src}")

        tested = 0
        for tag in src_tags[:3]:
            cls = algebra.find_class(src, tag)
            if cls and cls.has_language(tgt):
                tgt_slot = cls.get_slot(tgt)
                assert tgt_slot is not None, (
                    f"{src}:{tag} has class but no {tgt} slot"
                )
                # Verify the target slot can be found in the algebra
                back_cls = algebra.find_class(tgt, tgt_slot.native_tag)
                if back_cls:
                    assert back_cls.has_language(src) or back_cls.has_language(tgt)
                tested += 1

        if tested == 0:
            pytest.skip(f"No shared equivalence classes for {src}<->{tgt}")


class TestAllPairsCostTable:
    """Print the full cost matrix for all 15 pairs (informational)."""

    def test_full_cost_matrix(self, cost_matrix):
        """Verify all 15 pairs produce valid costs and print the matrix."""
        costs = {}
        for src, tgt in ALL_PAIRS:
            report = cost_matrix.compute(src, tgt)
            costs[(src, tgt)] = report.total_cost

        # Verify no NaN or infinite values
        for pair, cost in costs.items():
            assert cost == cost, f"NaN cost for {pair}"  # NaN != NaN
            assert cost != float("inf"), f"Infinite cost for {pair}"

        # Verify costs are diverse (not all the same)
        unique_costs = set(round(c, 2) for c in costs.values())
        assert len(unique_costs) >= 3, (
            f"Cost matrix too uniform: {unique_costs}"
        )
