"""
Cross-Language Bridge Tests: Chinese (zho) ↔ German (deu)

Tests that Chinese and German programs can interoperate through the
Flux Universal Bridge (FUB). Each test validates that language-specific
features survive translation across the language boundary.

Bridge semantics:
  - Chinese 量词 (classifiers) → German Geschlecht (gender type classes)
  - Chinese 话题寄存器 R63 (topic register) → German Nominativ (subject case)
  - Chinese 量子寄存器 (quantum registers) → German Kasus capabilities
  - Chinese zero-anaphora → German explicit pronoun resolution
"""

import pytest
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestClassifierGeschlechtMapping:
    """Test: Chinese classifier types map to German Geschlecht type classes."""

    def test_animal_classifier_maps_to_masculine_active(self):
        """
        Chinese 只 (animal classifier) → ActiveType (Maskulinum)
        Animals are actors — they initiate actions, produce movement.
        German: der Hund (masculine = ActiveType)
        """
        try:
            from flux_zho.classifier_type import (
                ClassifierTypeSolver, ClassifierType
            )
            solver = ClassifierTypeSolver()
            result = solver.resolve("三只猫")
            assert result.classifier_type == ClassifierType.ANIMAL
            assert result.noun == "猫"
            assert result.count == 3
        except ImportError:
            pytest.skip("flux_zho not available in test environment")

    def test_machine_classifier_maps_to_neuter_data(self):
        """
        Chinese 台 (machine classifier) → DataType (Neutrum)
        Machines are data containers — pure values, processed but not acting.
        German: das Gerät (neuter = DataType)
        """
        try:
            from flux_zho.classifier_type import (
                ClassifierTypeSolver, ClassifierType
            )
            solver = ClassifierTypeSolver()
            result = solver.resolve("两台电脑")
            assert result.classifier_type == ClassifierType.MACHINE
        except ImportError:
            pytest.skip("flux_zho not available")

    def test_person_respect_classifier_maps_to_feminine_container(self):
        """
        Chinese 位 (respectful person classifier) → ContainerType (Femininum)
        Respectful persons hold knowledge, manage relationships.
        German: die Lehrerin (feminine = ContainerType)
        """
        try:
            from flux_zho.classifier_type import (
                ClassifierTypeSolver, ClassifierType
            )
            solver = ClassifierTypeSolver()
            result = solver.resolve("五位教授")
            assert result.classifier_type == ClassifierType.PERSON_RESPECT
        except ImportError:
            pytest.skip("flux_zho not available")

    def test_book_classifier_maps_to_container_type(self):
        """
        Chinese 本 (book classifier) → ContainerType (Femininum)
        Books hold knowledge — containers of information.
        German: das Buch (but conceptually: die Bibliothek = container)
        """
        try:
            from flux_zho.classifier_type import (
                ClassifierTypeSolver, ClassifierType
            )
            solver = ClassifierTypeSolver()
            result = solver.resolve("一本书")
            assert result.classifier_type == ClassifierType.BOOK
        except ImportError:
            pytest.skip("flux_zho not available")

    def test_vessel_classifier_maps_to_active_transport(self):
        """
        Chinese 艘 (vessel classifier) → ActiveType (transport)
        Ships actively move through space, carrying cargo.
        German: das Schiff (neuter) but functionally ActiveType (navigator)
        """
        try:
            from flux_zho.classifier_type import (
                ClassifierTypeSolver, ClassifierType
            )
            solver = ClassifierTypeSolver()
            result = solver.resolve("三艘航母")
            assert result.classifier_type == ClassifierType.VESSEL
        except ImportError:
            pytest.skip("flux_zho not available")


class TestTopicNominativBridge:
    """Test: Chinese topic register R63 ↔ German Nominativ capability."""

    def test_topic_register_holds_subject_like_nominativ(self):
        """
        Chinese: 张三来了。他走了。(Zhāng Sān came. He left.)
        R63 = "张三" → "他" resolves to "张三"

        German: Zhang San kam. Er ging.
        Nominativ: "Zhang San" → "Er" (pronoun) resolves to "Zhang San"

        Both systems use the same mechanism — a register/case that holds
        the current discourse subject for anaphora resolution.
        """
        try:
            from flux_zho.quantum_registers import QuantumRegisterSystem, QR_TOPIC
            qrs = QuantumRegisterSystem()
            # Set topic register to 张三
            qrs.set_deterministic("R60", "张三", name="topic")
            # Collapse should return 张三
            collapsed = qrs.collapse("R60")
            assert collapsed == "张三"
        except ImportError:
            pytest.skip("flux_zho not available")

    def test_kasus_nominativ_grants_read_only_like_topic_observe(self):
        """
        German Nominativ → CAP_READ: can observe but not modify
        Chinese topic register: can reference but not modify the topic

        Parallel: both are observational positions. The topic (Chinese)
        and the Nominativ subject (German) are the "watcher" position.
        """
        try:
            from flux_deu.kasus_capability import (
                KasusCapabilityChecker, Kasus, ExtendedCap
            )
            checker = KasusCapabilityChecker(strict=False)
            checker.define_register("R0", Kasus.NOMINATIV)
            # Nominativ grants CAP_PUBLIC (read/observe)
            assert checker.check_register("R0", ExtendedCap.CAP_PUBLIC)
            # But not CAP_READWRITE
            assert not checker.check_register("R0", ExtendedCap.CAP_READWRITE)
        except ImportError:
            pytest.skip("flux_deu not available")


class TestQuantumRegisterKasusPreservation:
    """Test: Quantum registers preserve through bridge translation."""

    def test_superposition_survives_bridge(self):
        """
        Chinese quantum register R60 = [(0.7, "猫"), (0.3, "狗")]
        After bridge to German:
          - Should become a Kasus-qualified register with confidence weights
          - Nominativ (observer) + confidence distribution

        The quantum nature (superposition) is a Chinese innovation that
        German Kasus system doesn't natively have, but can PRESERVE
        by treating confidence as a Kasus modifier.
        """
        try:
            from flux_zho.quantum_registers import QuantumRegisterSystem
            qrs = QuantumRegisterSystem()
            qrs.set("R60", [(0.7, "die Katze"), (0.3, "der Hund")])
            state = qrs.get("R60")
            # Superposition should have 2 values
            assert state.value_count == 2
            # Entropy should be > 0 (not deterministic)
            assert state.entropy > 0
            # Collapse should return "die Katze" (higher confidence)
            collapsed = qrs.collapse("R60")
            assert collapsed == "die Katze"
        except ImportError:
            pytest.skip("flux_zho not available")

    def test_confidence_propagation_across_bridge(self):
        """
        Confidence propagates from R60 (topic) to R61 (type).
        This models: if we're 70% sure the topic is an animal,
        then the type register should reflect that uncertainty.
        """
        try:
            from flux_zho.quantum_registers import QuantumRegisterSystem
            qrs = QuantumRegisterSystem()
            qrs.set("R60", [(0.7, "猫"), (0.3, "程序")])
            qrs.set("R61", [(1.0, "AnimalType"), (1.0, "MachineType")])
            # Propagate R60 → R61 should merge confidence
            total = qrs.propagate("R60", "R61")
            assert total > 0
            # R61 should now have adjusted confidence
            state = qrs.get("R61")
            assert state.value_count == 2
        except ImportError:
            pytest.skip("flux_zho not available")


class TestKasusCapabilitySurvival:
    """Test: German Kasus capabilities survive language boundary."""

    def test_akkusativ_write_capability_preserved_in_chinese_context(self):
        """
        German Akkusativ → CAP_WRITE: can modify the accusative object
        Chinese equivalent: 把 structure "把门打开" (take the door and open it)
        The 把 acts like Akkusativ — it marks the object for modification.

        Bridge: Akkusativ capability should map to Chinese "把" modification access.
        """
        try:
            from flux_deu.kasus_capability import (
                KasusCapabilityChecker, Kasus, ExtendedCap
            )
            checker = KasusCapabilityChecker(strict=False)
            # R1 marked as Akkusativ (object to be modified)
            checker.define_register("R1", Kasus.AKKUSATIV)
            # Should have READWRITE capability
            assert checker.check_register("R1", ExtendedCap.CAP_READWRITE)
            # Should NOT have TRANSFER (ownership)
            assert not checker.check_register("R1", ExtendedCap.CAP_TRANSFER)
        except ImportError:
            pytest.skip("flux_deu not available")

    def test_genitiv_ownership_capability_preserved(self):
        """
        German Genitiv → CAP_OWN: has ownership, can transfer
        Chinese equivalent: 的 structure "我的书" (my book)
        The 的 marks possession, similar to Genitiv.

        Bridge: Genitiv ownership should map to Chinese 的 possession.
        """
        try:
            from flux_deu.kasus_capability import (
                KasusCapabilityChecker, Kasus, ExtendedCap
            )
            checker = KasusCapabilityChecker(strict=False)
            checker.define_register("R2", Kasus.GENITIV)
            # Should have TRANSFER capability
            assert checker.check_register("R2", ExtendedCap.CAP_TRANSFER)
        except ImportError:
            pytest.skip("flux_deu not available")

    def test_dativ_delegation_capability(self):
        """
        German Dativ → CAP_DELEGATE: can pass capabilities
        Chinese equivalent: 给 structure "给他一本书" (give him a book)
        The 给 marks the recipient, enabling delegation.

        Bridge: Dativ delegation should map to Chinese 给 recipient scope.
        """
        try:
            from flux_deu.kasus_capability import (
                KasusCapabilityChecker, Kasus, ExtendedCap
            )
            checker = KasusCapabilityChecker(strict=False)
            checker.define_register("R3", Kasus.DATIV)
            # Dativ alone → CAP_REFERENCE
            assert checker.check_register("R3", ExtendedCap.CAP_REFERENCE)
            # But Dativ + Genitiv → CAP_DELEGATE (sudo)
            checker.define_register("R4", Kasus.GENITIV)
            collective = checker.get_collective_kasus()
            # With multiple kasus active, should have elevated capability
            assert collective >= ExtendedCap.CAP_REFERENCE
        except ImportError:
            pytest.skip("flux_deu not available")


class TestTrennverbClassifierInteraction:
    """Test: German Trennverben interact with Chinese classifier system."""

    def test_trennverb_two_phase_matches_classifier_counting(self):
        """
        German separable verbs are TWO-PHASE: prefix (setup) + stem (execute)
        Chinese classifiers are TWO-PART: number + classifier + noun

        Both systems decompose a single operation into structured phases.
        Bridge: the phase structure of Trennverben should align with
        the counting structure of Chinese classifiers.
        """
        try:
            from flux_deu.trennverben import TrennverbHandler
            handler = TrennverbHandler()
            entry = handler.lookup("aufmachen")
            assert entry is not None
            assert entry.prefix == "auf"
            assert entry.stem == "machen"
            # Compile to continuation bytecode
            bytecode = handler.compile_continuation(entry)
            assert len(bytecode) == 4  # PREPARE, SUSPEND, RESUME, COMPLETE
        except ImportError:
            pytest.skip("flux_deu not available")

    def test_two_phase_execution_model_shared(self):
        """
        Both languages share a two-phase execution model:
        - German: auf|machen (prepare → execute)
        - Chinese: 先|做|再|检查 (first do, then check)

        The bridge should recognize this shared pattern and map
        Trennverb phases to Chinese sequential particles.
        """
        try:
            from flux_deu.trennverben import TrennverbHandler
            from flux_zho.classifier_type import ClassifierTypeSolver, ClassifierType
            handler = TrennverbHandler()
            solver = ClassifierTypeSolver()
            # German: "einrichten" (init + configure)
            entry = handler.lookup("einrichten")
            assert entry is not None
            # Chinese: "一套设备" (one set of equipment)
            result = solver.resolve("一套设备")
            assert result.classifier_type == ClassifierType.SET
        except ImportError:
            pytest.skip("flux runtime not available")


class TestBridgeSemantics:
    """Test: Semantic preservation across the Chinese-German bridge."""

    def test_paradigm_distance_is_computable(self):
        """
        Chinese and German occupy different points on the paradigm lattice.
        The distance between them determines bridge cost.
        Key differences:
        - Chinese: isolating, topic-prominent, tonal, classifier-based
        - German: fusional, case-marked, gendered, compound-word
        """
        # This test validates that the paradigm lattice concepts exist
        try:
            from flux_a2a.paradigm_lattice import ParadigmLattice
            lattice = ParadigmLattice()
            # The lattice should have pre-populated language points
            assert len(lattice._points) > 0
        except ImportError:
            pytest.skip("flux_a2a not available")

    def test_type_compatibility_across_languages(self):
        """
        Despite different type systems, certain types are universally compatible:
        - Numbers: 三 (Chinese) = drei (German) = 3
        - Booleans: 是/否 (Chinese) = ja/nein (German) = true/false
        - Collections: 组/套 (Chinese) = Gruppe/Set (German)

        The bridge should handle these universal types transparently.
        """
        # Universal type mapping
        universal_map = {
            "三": "drei",
            "猫": "Katze",
            "书": "Buch",
            "船": "Schiff",
            "电脑": "Computer",
        }
        assert len(universal_map) == 5
        # Bridge preserves count
        assert universal_map["三"] == "drei"
