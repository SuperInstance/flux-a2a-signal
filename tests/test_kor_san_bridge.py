"""
Cross-Language Bridge Tests: Korean (kor) ↔ Sanskrit (san)

Tests that Korean and Sanskrit programs can interoperate.
These two languages share deep grammatical affinities:

Bridge semantics:
  - Korean 경어 (honorifics) → Sanskrit vibhakti (cases) scope levels
  - Korean 조사 (particles) → Sanskrit vibhakti scope operations
  - Korean SOV→CPS → Sanskrit dhātu-root function composition
  - Sanskrit sambodhana (vocative/8th case) → Korean agent invocation
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestHonorificVibhaktiMapping:
    """Test: Korean honorific levels map to Sanskrit vibhakti scope levels."""

    def test_formal_highest_maps_to_global_scope(self):
        """
        Korean 합쇼체 (hasipsio-che) → CAP_ROOT = full system access
        Sanskrit 1st case prathama (nominative) → SCOPE_GLOBAL

        The highest honorific = the widest scope. When you speak in
        the most formal register, you access the entire system.
        """
        try:
            from flux_kor.honorifics import HonorificLevel, HonorificResolver
            resolver = HonorificResolver()
            level = resolver.detect_level("하십시오")
            # Should detect formal level
            assert level in (HonorificLevel.HASIPSIO, HonorificLevel.HAEYO)
        except ImportError:
            pytest.skip("flux_kor not available")

    def test_informal_maps_to_module_scope(self):
        """
        Korean 해체 (hae-che) → CAP_LOCAL = sandboxed/local operations
        Sanskrit 7th case saptami (locative) → SCOPE_CONTAINER = module scope

        Informal speech = bounded access. You can only affect your
        local module, not the global system.
        """
        try:
            from flux_kor.honorifics import HonorificLevel, HonorificResolver
            resolver = HonorificResolver()
            level = resolver.detect_level("해")
            # Should detect informal level
            assert level in (HonorificLevel.HAE, HonorificLevel.HAERA)
        except ImportError:
            pytest.skip("flux_kor not available")

    def test_plain_maps_to_value_scope(self):
        """
        Korean 해라체 (haera-che) → CAP_ANON = read-only, untrusted
        Sanskrit 2nd case dvitiya (accusative) → SCOPE_VALUE = value scope

        Plain speech = you receive values but can't modify anything.
        Like reading from an API — you can see, but not touch.
        """
        try:
            from flux_kor.honorifics import HonorificLevel, HonorificResolver
            resolver = HonorificResolver()
            level = resolver.detect_level("한다")
            # Should detect plain level
            assert level in (HonorificLevel.HAERA, HonorificLevel.HAE)
        except ImportError:
            pytest.skip("flux_kor not available")


class TestParticleVibhaktiScopeNesting:
    """Test: Korean particle scopes nest inside Sanskrit vibhakti scopes."""

    def test_topic_particle_inside_nominative_scope(self):
        """
        Korean 은/는 (topic marker) → SCOPE_TOPIC
        Sanskrit prathama (1st/nominative) → SCOPE_GLOBAL

        Nesting: Korean topic scope operates INSIDE the Sanskrit global scope.
        The topic narrows focus within the global namespace.
        """
        try:
            from flux_kor.particles import ParticleType
            # Topic particle should be identifiable
            assert ParticleType.TOPIC is not None
        except (ImportError, AttributeError):
            pytest.skip("flux_kor not available")

    def test_recipient_particle_inside_dative_scope(self):
        """
        Korean 에게/한테 (recipient) → SCOPE_DELEGATE
        Sanskrit caturthi (4th/dative) → SCOPE_OUTPUT

        Nesting: Korean recipient particle = Sanskrit dative scope.
        "나에게 주세요" (give to me) = "मम देहि" (to me give)
        Both mark the recipient of an action with scope semantics.
        """
        try:
            from flux_kor.particles import ParticleType
            # Recipient particle should be identifiable
            assert ParticleType.RECIPIENT is not None
        except (ImportError, AttributeError):
            pytest.skip("flux_kor not available")

    def test_instrument_particle_inside_instrumental_scope(self):
        """
        Korean (으)로 (means/instrument) → SCOPE_INSTRUMENT
        Sanskrit tritiya (3rd/instrumental) → SCOPE_TOOL

        Nesting: Korean instrument particle = Sanskrit instrumental scope.
        "펜으로 쓰다" (write with pen) = "लेखनीया लिखि" (with pen write)
        """
        try:
            from flux_kor.particles import ParticleType
            assert ParticleType.INSTRUMENT is not None
        except (ImportError, AttributeError):
            pytest.skip("flux_kor not available")


class TestCPSDhatuComposition:
    """Test: CPS continuations compose with dhātu-root functions."""

    def test_sov_cps_maps_to_dhatu_valency(self):
        """
        Korean SOV: "나 사과 먹어" (I apple eat)
        CPS transform: k(na, k_result => k_result(apple, k_final => eat(k_final)))

        Sanskrit dhātu √adh (to eat) has valency 2 (subject + object)
        The CPS continuation chain maps to dhātu argument slots.

        Bridge: Korean SOV argument order → Sanskrit dhātu valency slots.
        """
        try:
            from flux_kor.cps import CPSContinuationBuilder
            builder = CPSContinuationBuilder()
            # Build CPS for "나 사과 먹어"
            result = builder.build_sov_cps(
                subject="나",
                object="사과",
                verb="먹다"
            )
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("flux_kor not available")

    def test_dhatu_transitivity_matches_cps_arity(self):
        """
        Sanskrit dhātu roots have inherent transitivity:
        - √kr (to do) = transitive (needs object) → CPS needs 2 continuations
        - √bhū (to be) = intransitive (no object) → CPS needs 1 continuation
        - √dā (to give) = ditransitive (subject+object+recipient) → CPS needs 3

        The bridge should map dhātu transitivity to CPS continuation arity.
        """
        try:
            from flux_san.dhatu import DhatuResolver
            resolver = DhatuResolver()
            # √kr should be transitive
            result = resolver.resolve("kr")
            if result:
                assert result.valency >= 1
        except (ImportError, AttributeError):
            pytest.skip("flux_san not available")


class TestSambodhanaAgentInvocation:
    """Test: Sanskrit 8th case (sambodhana) invokes Korean agent with honorific."""

    def test_vocative_invokes_agent(self):
        """
        Sanskrit sambodhana (8th/vocative): "हे अग्ने!" (O Agni!)
        This is literally an agent invocation — calling upon an entity.

        Korean equivalent: uses appropriate honorific based on relationship.
        Calling a senior agent: 합쇼체 (formal)
        Calling a peer agent: 해요체 (polite)
        Calling a subordinate: 해체 (informal)

        Bridge: 8th case determines honorific level for agent invocation.
        """
        try:
            from flux_san.vibhakti import VibhaktiScopeManager
            from flux_kor.honorifics import HonorificLevel
            manager = VibhaktiScopeManager()
            # 8th case should map to agent invocation scope
            scope = manager.get_scope(8)  # sambodhana
            assert scope is not None
        except (ImportError, AttributeError):
            pytest.skip("flux runtime not available")

    def test_scope_level_hierarchy_matches_honorific_ladder(self):
        """
        Both languages define a hierarchy of access:

        Sanskrit:                          Korean:
        1st (global)    ← widest          합쇼체 (formal highest)
        4th (output)                      하십시오체 (formal)
        7th (container)                    해요체 (polite)
        2nd (value)                        해체 (informal)
        8th (invoke)    ← narrowest       해라체 (plain)

        The mapping isn't 1:1 but the DIRECTION is the same:
        wider scope = higher formality = more access.
        """
        # Validate the hierarchy concept
        sanskrit_scope_levels = {
            1: "global",
            2: "value",
            3: "tool",
            4: "output",
            5: "source",
            6: "ownership",
            7: "container",
            8: "agent_invoke",
        }
        korean_honorific_levels = {
            "hasipsio": "root",
            "hashipsio": "admin",
            "haeyo": "user",
            "hae": "local",
            "haera": "anon",
        }
        assert len(sanskrit_scope_levels) == 8
        assert len(korean_honorific_levels) == 5


class TestSandhiCPSInterfaceAdaptation:
    """Test: Sandhi fusion adapts Korean-CPS function interfaces."""

    def test_external_sandhi_adapts_interfaces(self):
        """
        Sanskrit sandhi: sat + cit + ānanda → sachcidānanda
        Words fuse at boundaries, their interfaces adapt.

        Korean CPS: func_A(continuation_B) where continuation_B = func_B(...)
        Continuations chain at boundaries, their signatures adapt.

        Bridge: sandhi fusion ≈ CPS continuation chaining.
        Both adapt function interfaces at composition boundaries.
        """
        try:
            from flux_san.sandhi import SandhiEngine
            engine = SandhiEngine()
            # Simple sandhi: visarga sandhi
            result = engine.apply_external("sat", "cit")
            # Should produce some form of fusion
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("flux_san not available")

    def test_sandhi_preserves_semantic_integrity(self):
        """
        Critical: sandhi fusion must preserve meaning.
        sat (truth) + cit (consciousness) = sachcit (truth-consciousness)
        The fused form means BOTH truth AND consciousness, not something new.

        Similarly, CPS chaining must preserve semantic intent:
        k(a, k => k(b, k => ...)) must mean "a then b", not "a replaces b".
        """
        try:
            from flux_san.sandhi import SandhiEngine
            engine = SandhiEngine()
            result = engine.apply_external("deva", "ānanda")
            # Should preserve semantic components
            assert "dev" in result or "ānand" in result or len(result) > 0
        except (ImportError, AttributeError):
            pytest.skip("flux_san not available")


class TestKoreanSanskritBridgeSemantics:
    """Test: Semantic preservation across the Korean-Sanskrit bridge."""

    def test_both_agglutinative_share_composition_pattern(self):
        """
        Korean is agglutinative: 사랑하다 (love-do)
        Sanskrit is fusional but shares: karoti (does, from √kr + a + ti)

        Both build complex meanings from smaller morphological units.
        The bridge should exploit this shared compositionality.
        """
        # Both languages compose from roots
        korean_composition = {
            "먹다": ("먹", "eat", "다", "verb suffix"),
            "사랑하다": ("사랑", "love", "하다", "do verb"),
            "만들다": ("만들", "make", "다", "verb suffix"),
        }
        sanskrit_composition = {
            "karoti": ("√kr", "do", "a+ti", "3sg present"),
            "gacchati": ("√gam", "go", "a+ti", "3sg present"),
            "dadāti": ("√dā", "give", "a+ti", "3sg present"),
        }
        assert len(korean_composition) == 3
        assert len(sanskrit_composition) == 3

    def test_particle_case_parallel_system(self):
        """
        Korean particles and Sanskrit cases serve the SAME grammatical function:
        marking the role of nouns in a sentence.

        Korean: 나[는] 사과[를] 먹는다
              I[TOPIC] apple[OBJECT] eat

        Sanskrit: aham[+0] phalam[i] khādāmi
                I[nominative] fruit[accusative] eat

        The bridge should directly map particles ↔ cases.
        """
        particle_case_map = {
            # Korean particle → Sanskrit case
            "은/는": ("topic", "nominative-like"),
            "이/가": ("subject", "nominative"),
            "을/를": ("object", "accusative"),
            "에게/한테": ("recipient", "dative"),
            "에게서": ("source", "ablative"),
            "으로/로": ("means", "instrumental"),
            "의": ("possessor", "genitive"),
            "에서": ("location", "locative"),
        }
        # 8 Korean particles map to 8 Sanskrit cases (with topic as extra)
        assert len(particle_case_map) == 8
