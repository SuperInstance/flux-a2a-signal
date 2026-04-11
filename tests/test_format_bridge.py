"""Tests for the A2A ↔ FORMAT Bridge module.

Covers:
  - Signal → bytecode compilation (tell, ask, branch, discuss)
  - Bytecode → Signal decompilation (round-trip)
  - Old → new opcode translation for all relocated opcodes
  - New → old opcode translation (reverse)
  - Confidence propagation (CONF_ADD, CONF_MERGE, CONF_THRESHOLD)
  - Trust verification (TRUST_CHECK + STRIPCONF)
  - Bulk bytecode translation
  - Conflict verification (no byte collisions in proposed mapping)
"""

from __future__ import annotations

import struct
import pytest
from flux_a2a.format_bridge import (
    FormatBridge,
    FormatClass,
    MAGIC_FLUX,
    UNIFIED_VERSION,
    HEADER_SIZE,
    A2A_OLD_TO_NEW,
    A2A_NEW_TO_OLD,
    UNIFIED_OPCODES,
    UNIFIED_BY_VALUE,
    SignalOp,
    CompiledInstruction,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bridge() -> FormatBridge:
    """Create a FormatBridge with header emission enabled."""
    return FormatBridge(emit_header=True)


@pytest.fixture
def bridge_no_header() -> FormatBridge:
    """Create a FormatBridge without header emission."""
    return FormatBridge(emit_header=False)


# ══════════════════════════════════════════════════════════════════════════════
# Tell Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTellSignal:
    """Tests for compiling/decompiling 'tell' signals."""

    def test_tell_compiles_to_tell_opcode(self, bridge: FormatBridge) -> None:
        """A tell signal should produce bytecode starting with the TELL opcode (0xD0)."""
        signal = {
            "op": "tell",
            "to": "agent-42",
            "message": "hello world",
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        # Should have FLUX header
        assert bytecode[:2] == MAGIC_FLUX
        assert bytecode[2] == UNIFIED_VERSION

        # After header, first instruction byte should be TELL (0xD0)
        tell_opcode = bytecode[HEADER_SIZE]
        assert tell_opcode == UNIFIED_OPCODES["TELL"], (
            f"Expected TELL (0xD0), got 0x{tell_opcode:02X}"
        )

    def test_tell_includes_message_length(self, bridge: FormatBridge) -> None:
        """A tell signal should encode message length in the bytecode."""
        msg = "hello"
        signal = {"op": "tell", "to": "agent-1", "message": msg}
        bytecode = bridge.compile_signal_to_bytecode(signal)

        # msg_len should be at offset HEADER_SIZE + 2
        msg_len = bytecode[HEADER_SIZE + 2]
        assert msg_len == len(msg), f"Expected msg_len={len(msg)}, got {msg_len}"

    def test_tell_with_confidence(self, bridge: FormatBridge) -> None:
        """A tell signal with confidence should emit CONF_ADD after TELL."""
        signal = {
            "op": "tell",
            "to": "agent-1",
            "message": "hi",
            "confidence": 0.95,
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        # Find CONF_ADD opcode (0x60) in the bytecode
        conf_add_byte = UNIFIED_OPCODES["CONF_ADD"]
        assert conf_add_byte in bytecode, "CONF_ADD (0x60) not found in compiled tell"

    def test_tell_roundtrip(self, bridge: FormatBridge) -> None:
        """Compile tell → decompile should produce equivalent signal."""
        signal = {
            "op": "tell",
            "to": "agent-99",
            "message": "test message",
            "confidence": 0.85,
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)
        result = bridge.decompile_bytecode_to_signal(bytecode)

        assert result["op"] == "tell"
        assert "to" in result
        assert "message" in result
        # Confidence is encoded as u8 (0-254 scale), so roundtrip has ~0.004 precision
        assert "confidence" in result
        assert result["confidence"] == pytest.approx(0.85, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Ask Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAskSignal:
    """Tests for compiling/decompiling 'ask' signals."""

    def test_ask_compiles_to_ask_opcode(self, bridge: FormatBridge) -> None:
        """An ask signal should produce bytecode starting with the ASK opcode (0xD1)."""
        signal = {
            "op": "ask",
            "to": "agent-7",
            "message": "what is 2+2?",
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        ask_opcode = bytecode[HEADER_SIZE]
        assert ask_opcode == UNIFIED_OPCODES["ASK"], (
            f"Expected ASK (0xD1), got 0x{ask_opcode:02X}"
        )

    def test_ask_with_confidence(self, bridge: FormatBridge) -> None:
        """An ask signal with confidence should emit CONF_ADD."""
        signal = {
            "op": "ask",
            "to": "agent-3",
            "message": "?",
            "confidence": 0.5,
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        conf_add_byte = UNIFIED_OPCODES["CONF_ADD"]
        assert conf_add_byte in bytecode, "CONF_ADD not found in compiled ask"

    def test_ask_roundtrip(self, bridge: FormatBridge) -> None:
        """Compile ask → decompile should produce equivalent signal."""
        signal = {
            "op": "ask",
            "to": "agent-5",
            "message": "please compute",
            "confidence": 0.9,
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)
        result = bridge.decompile_bytecode_to_signal(bytecode)

        assert result["op"] == "ask"
        assert "message" in result


# ══════════════════════════════════════════════════════════════════════════════
# Branch Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBranchSignal:
    """Tests for compiling/decompiling 'branch' signals."""

    def test_branch_compiles_to_branch_opcode(self, bridge: FormatBridge) -> None:
        """A branch signal should produce bytecode with OP_BRANCH (0xD6)."""
        signal = {
            "op": "branch",
            "branches": [
                {"label": "path_a", "weight": 0.6, "body": []},
                {"label": "path_b", "weight": 0.4, "body": []},
            ],
            "merge": {"strategy": "weighted_confidence"},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        branch_opcode = bytecode[HEADER_SIZE]
        assert branch_opcode == UNIFIED_OPCODES["OP_BRANCH"], (
            f"Expected OP_BRANCH (0xD6), got 0x{branch_opcode:02X}"
        )

    def test_branch_records_path_count(self, bridge: FormatBridge) -> None:
        """Branch bytecode should encode the number of paths."""
        signal = {
            "op": "branch",
            "branches": [
                {"label": "a", "body": []},
                {"label": "b", "body": []},
                {"label": "c", "body": []},
            ],
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        n_paths = bytecode[HEADER_SIZE + 1]
        assert n_paths == 3, f"Expected 3 paths, got {n_paths}"

    def test_branch_empty_raises(self, bridge: FormatBridge) -> None:
        """An empty branch signal should raise ValueError."""
        signal = {"op": "branch", "branches": []}
        with pytest.raises(ValueError, match="branches"):
            bridge.compile_signal_to_bytecode(signal)

    def test_branch_roundtrip(self, bridge: FormatBridge) -> None:
        """Compile branch → decompile should produce equivalent signal."""
        signal = {
            "op": "branch",
            "branches": [
                {"label": "fast", "weight": 0.7, "body": []},
                {"label": "slow", "weight": 0.3, "body": []},
            ],
            "merge": {"strategy": "weighted_confidence"},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)
        result = bridge.decompile_bytecode_to_signal(bytecode)

        assert result["op"] == "branch"
        assert len(result["branches"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Fork Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestForkSignal:
    """Tests for compiling/decompiling 'fork' signals."""

    def test_fork_compiles_to_delegate(self, bridge: FormatBridge) -> None:
        """A fork signal should compile to DELEGATE (0xD2)."""
        signal = {
            "op": "fork",
            "from": "parent-agent",
            "inherit": {"context": True, "trust_graph": False},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        delegate_opcode = bytecode[HEADER_SIZE]
        assert delegate_opcode == UNIFIED_OPCODES["DELEGATE"], (
            f"Expected DELEGATE (0xD2), got 0x{delegate_opcode:02X}"
        )

    def test_fork_inherit_flags(self, bridge: FormatBridge) -> None:
        """Fork should encode inherit flags correctly."""
        signal = {
            "op": "fork",
            "from": "parent",
            "inherit": {"context": True, "trust_graph": True, "message_history": True},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        flags = bytecode[HEADER_SIZE + 2]
        assert flags & 0x01, "context flag not set"
        assert flags & 0x02, "trust_graph flag not set"
        assert flags & 0x04, "message_history flag not set"

    def test_fork_roundtrip(self, bridge: FormatBridge) -> None:
        """Compile fork → decompile should produce equivalent signal."""
        signal = {
            "op": "fork",
            "from": "agent-parent",
            "inherit": {"context": True, "trust_graph": False, "message_history": True},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)
        result = bridge.decompile_bytecode_to_signal(bytecode)

        assert result["op"] == "fork"
        assert result["inherit"]["context"] is True
        assert result["inherit"]["trust_graph"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Discuss Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscussSignal:
    """Tests for compiling/decompiling 'discuss' signals."""

    def test_discuss_compiles_to_discuss_opcode(self, bridge: FormatBridge) -> None:
        """A discuss signal should produce OP_DISCUSS (0xD8)."""
        signal = {
            "op": "discuss",
            "format": "debate",
            "participants": [
                {"id": "agent-a", "stance": "pro"},
                {"id": "agent-b", "stance": "con"},
            ],
            "until": {"max_rounds": 5},
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        discuss_opcode = bytecode[HEADER_SIZE]
        assert discuss_opcode == UNIFIED_OPCODES["OP_DISCUSS"], (
            f"Expected OP_DISCUSS (0xD8), got 0x{discuss_opcode:02X}"
        )

    def test_discuss_with_topic(self, bridge: FormatBridge) -> None:
        """A discuss with topic should emit SET_TOPIC (0xEF)."""
        signal = {
            "op": "discuss",
            "format": "brainstorm",
            "participants": [{"id": "a"}],
            "topic": "climate policy",
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        set_topic_byte = UNIFIED_OPCODES["SET_TOPIC"]
        assert set_topic_byte in bytecode, "SET_TOPIC not found in compiled discuss"


# ══════════════════════════════════════════════════════════════════════════════
# Co-Iterate Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCoIterateSignal:
    """Tests for compiling 'co_iterate' signals."""

    def test_co_iterate_compiles(self, bridge: FormatBridge) -> None:
        """A co_iterate signal should compile to OP_DISCUSS with agent count."""
        signal = {
            "op": "co_iterate",
            "agents": [
                {"id": "agent-1", "role": "modifier"},
                {"id": "agent-2", "role": "auditor"},
            ],
            "rounds": 10,
        }
        bytecode = bridge.compile_signal_to_bytecode(signal)

        # First opcode should be OP_DISCUSS
        op = bytecode[HEADER_SIZE]
        assert op == UNIFIED_OPCODES["OP_DISCUSS"]
        # Second byte should be agent count
        assert bytecode[HEADER_SIZE + 1] == 2


# ══════════════════════════════════════════════════════════════════════════════
# Confidence Propagation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidencePropagation:
    """Tests for confidence → FORMAT CONF_* opcodes."""

    def test_confidence_merge_compiles(self, bridge: FormatBridge) -> None:
        """Confidence merge should use CONF_MERGE (0x68) + CONF_THRESHOLD (0x69)."""
        bytecode = bridge.compile_confidence_merge(
            weights=[0.8, 0.6, 0.9],
            threshold=0.7,
        )

        # Should have header
        assert bytecode[:2] == MAGIC_FLUX

        # Find CONF_MERGE and CONF_THRESHOLD
        conf_merge = UNIFIED_OPCODES["CONF_MERGE"]
        conf_thresh = UNIFIED_OPCODES["CONF_THRESHOLD"]

        assert conf_merge in bytecode, "CONF_MERGE not found"
        assert conf_thresh in bytecode, "CONF_THRESHOLD not found"

    def test_confidence_values_clamped(self, bridge: FormatBridge) -> None:
        """Confidence values > 1.0 should be clamped to 1.0."""
        bytecode = bridge.compile_confidence_merge(
            weights=[1.5, -0.5, 0.5],
            threshold=0.5,
        )

        # CONF_MERGE byte is 0x68. Weights encoded as u8 (0-254 scale).
        # Weight 1.5 → clamped to 1.0 → 254 (0xFE)
        # Weight -0.5 → clamped to 0.0 → 0
        # Weight 0.5 → 127 (0x7F)
        merge_byte = UNIFIED_OPCODES["CONF_MERGE"]
        positions = [i for i, b in enumerate(bytecode) if b == merge_byte]

        # First CONF_MERGE is the count instruction. Subsequent ones are weights.
        # All weights are u8, so each weight byte follows its CONF_MERGE opcode.
        weight_bytes = []
        for pos in positions[1:]:  # Skip the count instruction
            if pos + 1 < len(bytecode):
                weight_bytes.append(bytecode[pos + 1])

        assert 254 in weight_bytes, "Weight 1.5 should be clamped to 1.0 (254)"
        assert 0 in weight_bytes, "Weight -0.5 should be clamped to 0.0 (0)"
        assert 127 in weight_bytes, "Weight 0.5 should be 127"


# ══════════════════════════════════════════════════════════════════════════════
# Trust Verification Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrustVerification:
    """Tests for trust → TRUST_CHECK + STRIPCONF."""

    def test_trust_verify_compiles(self, bridge: FormatBridge) -> None:
        """Trust verify should emit TRUST_CHECK (0xD4) + STRIPCONF (0x17)."""
        bytecode = bridge.compile_trust_verify("agent-42", required_trust=0.8)

        trust_check = UNIFIED_OPCODES["TRUST_CHECK"]
        stripconf = UNIFIED_OPCODES["STRIPCONF"]

        assert trust_check in bytecode, "TRUST_CHECK not found"
        assert stripconf in bytecode, "STRIPCONF not found"


# ══════════════════════════════════════════════════════════════════════════════
# Opcode Translation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOpcodeTranslation:
    """Tests for old → new and new → old opcode translation."""

    @pytest.mark.parametrize("old_byte,new_byte,name", [
        (0x60, 0xD0, "TELL"),
        (0x61, 0xD1, "ASK"),
        (0x62, 0xD2, "DELEGATE"),
        (0x63, 0xD3, "BROADCAST"),
        (0x64, 0xD4, "TRUST_CHECK"),
        (0x65, 0xD5, "CAP_REQUIRE"),
        (0x70, 0xD6, "OP_BRANCH"),
        (0x71, 0xD7, "OP_MERGE"),
        (0x72, 0xD8, "OP_DISCUSS"),
        (0x73, 0xD9, "OP_DELEGATE"),
        (0x74, 0xDA, "OP_CONFIDENCE"),
        (0x76, 0xDB, "OP_META"),
        (0x80, 0xDC, "IEXP"),
        (0x81, 0xDD, "IROOT"),
        (0x82, 0xDE, "VERIFY_TRUST"),
        (0x83, 0xDF, "CHECK_BOUNDS"),
        (0x84, 0xE0, "OPTIMIZE"),
        (0x85, 0xE1, "ATTACK"),
        (0x86, 0xE2, "DEFEND"),
        (0x87, 0xE3, "ADVANCE"),
        (0x88, 0xE4, "RETREAT"),
        (0x89, 0xE5, "SEQUENCE"),
        (0x8A, 0xE6, "LOOP"),
        (0xA0, 0xE7, "LOOP_START"),
        (0xA1, 0xE8, "LOOP_END"),
        (0xA2, 0xE9, "LAZY_DEFER"),
        (0xA3, 0xEA, "CACHE_LOAD"),
        (0xA4, 0xEB, "CACHE_STORE"),
        (0xA5, 0xEC, "ROLLBACK_SAVE"),
        (0xA6, 0xED, "ROLLBACK_RESTORE"),
        (0xA7, 0xEE, "EVENTUAL_SCHEDULE"),
        (0xB0, 0xEF, "SET_TOPIC"),
        (0xB1, 0xF0, "USE_TOPIC"),
        (0xB2, 0xF1, "CLEAR_TOPIC"),
    ])
    def test_old_to_new_translation(
        self, bridge: FormatBridge, old_byte: int, new_byte: int, name: str
    ) -> None:
        """All relocated opcodes should translate correctly old → new."""
        result_byte, fmt_class = bridge.translate_a2a_to_format(old_byte)
        assert result_byte == new_byte, (
            f"{name}: expected 0x{new_byte:02X}, got 0x{result_byte:02X}"
        )

    @pytest.mark.parametrize("old_byte,new_byte,name", [
        (0x60, 0xD0, "TELL"),
        (0x61, 0xD1, "ASK"),
        (0x62, 0xD2, "DELEGATE"),
        (0x70, 0xD6, "OP_BRANCH"),
        (0x80, 0xDC, "IEXP"),
        (0xA0, 0xE7, "LOOP_START"),
        (0xB0, 0xEF, "SET_TOPIC"),
    ])
    def test_new_to_old_translation(
        self, bridge: FormatBridge, old_byte: int, new_byte: int, name: str
    ) -> None:
        """All relocated opcodes should translate correctly new → old."""
        result_byte, result_name = bridge.translate_format_to_a2a(new_byte)
        assert result_byte == old_byte, (
            f"{name}: expected old 0x{old_byte:02X}, got 0x{result_byte:02X}"
        )
        assert result_name == name

    def test_base_ops_passthrough(self, bridge: FormatBridge) -> None:
        """Base ops (0x00-0x5F) should pass through unchanged."""
        for byte in [0x00, 0x08, 0x20, 0x30, 0x40, 0x50]:
            result, fmt = bridge.translate_a2a_to_format(byte)
            assert result == byte, f"Base op 0x{byte:02X} should passthrough"

    def test_system_ops_passthrough(self, bridge: FormatBridge) -> None:
        """System ops (0xFE-0xFF) should pass through unchanged."""
        for byte in [0xFE, 0xFF]:
            result, fmt = bridge.translate_a2a_to_format(byte)
            assert result == byte

    def test_conflict_zone_raises(self, bridge: FormatBridge) -> None:
        """Opcodes in 0x60-0x69 that are NOT in the old A2A map should raise."""
        # 0x66 is CONF_FMUL in FORMAT — not an old A2A opcode
        with pytest.raises(ValueError, match="CONF.*zone"):
            bridge.translate_a2a_to_format(0x66)

    def test_format_conf_no_old_equivalent(self, bridge: FormatBridge) -> None:
        """FORMAT CONF_* ops (0x60-0x69) should have no old A2A equivalent."""
        for byte in range(0x60, 0x6A):
            if byte in A2A_NEW_TO_OLD:
                # This byte IS a relocated A2A opcode — should work
                old, name = bridge.translate_format_to_a2a(byte)
                assert old in A2A_OLD_TO_NEW
            else:
                # This byte is a pure FORMAT CONF_* op — should raise
                with pytest.raises(ValueError, match="CONF.*no old A2A equivalent"):
                    bridge.translate_format_to_a2a(byte)


# ══════════════════════════════════════════════════════════════════════════════
# Bulk Bytecode Translation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBulkTranslation:
    """Tests for translating entire bytecode streams."""

    def test_tell_bytecode_old_to_new(self, bridge_no_header: FormatBridge) -> None:
        """Old TELL bytecode should translate to new byte range."""
        # Simulate old bytecode: TELL(0x60) agent_id(0x01) msg_len(0x00)
        # Use bytes outside old opcode ranges for operands
        old_bytecode = bytes([0x60, 0x01, 0x00, 0x08])
        new_bytecode = bridge_no_header.translate_bytecode_old_to_new(old_bytecode)

        # TELL should have moved from 0x60 to 0xD0
        assert new_bytecode[0] == 0xD0, f"Expected 0xD0, got 0x{new_bytecode[0]:02X}"
        # Operands unchanged
        assert new_bytecode[1:] == old_bytecode[1:]

    def test_branch_bytecode_old_to_new(self, bridge_no_header: FormatBridge) -> None:
        """Old BRANCH bytecode should translate."""
        old_bytecode = bytes([0x70, 0x02, 0x71, 0x72])
        new_bytecode = bridge_no_header.translate_bytecode_old_to_new(old_bytecode)

        assert new_bytecode[0] == 0xD6  # OP_BRANCH
        assert new_bytecode[2] == 0xD7  # OP_MERGE
        assert new_bytecode[3] == 0xD8  # OP_DISCUSS

    def test_new_to_old_roundtrip(self, bridge_no_header: FormatBridge) -> None:
        """old → new → old should be identity."""
        old_bytecode = bytes([
            0x60, 0x01,  # TELL
            0x70, 0x03,  # OP_BRANCH
            0x80, 0x02,  # IEXP
            0xA0, 0x00,  # LOOP_START
            0xB0, 0x05,  # SET_TOPIC
            0x08, 0x00,  # IADD (passthrough)
            0xFE, 0x00,  # PRINT (passthrough)
        ])

        new_bytecode = bridge_no_header.translate_bytecode_old_to_new(old_bytecode)
        roundtrip = bridge_no_header.translate_bytecode_new_to_old(new_bytecode)

        assert roundtrip == old_bytecode, (
            f"Roundtrip failed:\n  original:  {old_bytecode.hex()}\n"
            f"  roundtrip: {roundtrip.hex()}"
        )

    def test_base_ops_untouched(self, bridge_no_header: FormatBridge) -> None:
        """Base ops should pass through bulk translation unchanged."""
        base_bytecode = bytes(range(0x00, 0x58))
        new_bytecode = bridge_no_header.translate_bytecode_old_to_new(base_bytecode)

        # Only the bytes that are in the relocation table should change
        # But 0x00-0x57 has NO relocated opcodes, so all should be identical
        assert new_bytecode == base_bytecode


# ══════════════════════════════════════════════════════════════════════════════
# Conflict Verification Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConflictVerification:
    """Tests to verify no byte conflicts in the proposed mapping."""

    def test_no_byte_duplicates_in_unified_map(self) -> None:
        """The unified opcode map should have no duplicate byte values."""
        is_clean, conflicts = FormatBridge.check_no_conflicts()
        assert is_clean, f"Found conflicts:\n  " + "\n  ".join(conflicts)

    def test_relocated_ops_outside_conflict_zone(self) -> None:
        """All relocated opcodes must be outside FORMAT's 0x60-0x69 zone."""
        for old_byte, (new_byte, name) in A2A_OLD_TO_NEW.items():
            assert not (0x60 <= new_byte <= 0x69), (
                f"{name} relocated to 0x{new_byte:02X} — inside CONF_* zone!"
            )

    def test_relocated_ops_outside_base_zone(self) -> None:
        """All relocated opcodes must be outside FORMAT's 0x00-0x5F zone."""
        for old_byte, (new_byte, name) in A2A_OLD_TO_NEW.items():
            assert not (0x00 <= new_byte <= 0x5F), (
                f"{name} relocated to 0x{new_byte:02X} — inside base ISA zone!"
            )

    def test_new_ops_do_not_collide_with_format(self) -> None:
        """New unified opcodes must not collide with FORMAT authoritative slots."""
        format_authoritative = set(range(0x00, 0x4F)) | set(range(0x60, 0x6A))
        for _, (new_byte, name) in A2A_OLD_TO_NEW.items():
            assert new_byte not in format_authoritative, (
                f"{name} at 0x{new_byte:02X} collides with FORMAT authoritative!"
            )

    def test_unified_map_values_unique(self) -> None:
        """All values in the unified opcode map must be unique."""
        values = list(UNIFIED_OPCODES.values())
        assert len(values) == len(set(values)), (
            f"Unified opcode map has {len(values) - len(set(values))} duplicate values!"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Unknown Signal / Error Handling Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Tests for error conditions."""

    def test_unknown_signal_op_raises(self, bridge: FormatBridge) -> None:
        """An unknown signal op should raise ValueError."""
        signal = {"op": "teleport", "to": "mars"}
        with pytest.raises(ValueError, match="Unknown signal op"):
            bridge.compile_signal_to_bytecode(signal)

    def test_empty_signal_raises(self, bridge: FormatBridge) -> None:
        """A signal with no 'op' key should raise ValueError."""
        signal = {"to": "nowhere"}
        with pytest.raises(ValueError, match="Unknown signal op"):
            bridge.compile_signal_to_bytecode(signal)

    def test_too_short_bytecode_raises(self, bridge: FormatBridge) -> None:
        """Decompiling very short bytecode should raise ValueError."""
        with pytest.raises(ValueError):
            bridge.decompile_bytecode_to_signal(b'\x00')


# ══════════════════════════════════════════════════════════════════════════════
# Format Classification Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatClassification:
    """Tests for opcode format classification."""

    @pytest.mark.parametrize("byte,expected_cls", [
        (0x00, FormatClass.FORMAT_A),
        (0x03, FormatClass.FORMAT_A),
        (0x08, FormatClass.FORMAT_B),
        (0x10, FormatClass.FORMAT_C),
        (0x18, FormatClass.FORMAT_D),
        (0x20, FormatClass.FORMAT_E),
        (0x30, FormatClass.FORMAT_E),
        (0x40, FormatClass.FORMAT_F),
        (0x48, FormatClass.FORMAT_G),
        (0x60, FormatClass.CONF),
        (0x69, FormatClass.CONF),
        (0xD0, FormatClass.A2A),
        (0xDB, FormatClass.A2A),
        (0xDC, FormatClass.PARADIGM),
        (0xF1, FormatClass.PARADIGM),
        (0xFE, FormatClass.SYSTEM),
        (0xFF, FormatClass.SYSTEM),
        (0x07, FormatClass.BASE),  # gap between A and B
    ])
    def test_classify_opcode(
        self, byte: int, expected_cls: FormatClass
    ) -> None:
        """Opcodes should be classified into the correct format."""
        result = FormatBridge._classify_opcode(byte)
        assert result == expected_cls, (
            f"0x{byte:02X}: expected {expected_cls.value}, got {result.value}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Disassembly Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDisassembly:
    """Tests for bytecode disassembly."""

    def test_disassemble_old_bytecode(self) -> None:
        """Disassembling old bytecode should show RELOCATED markers."""
        old_code = bytes([0x60, 0x01, 0x70, 0x02])
        lines = FormatBridge.disassemble(old_code)

        assert any("RELOCATED" in line for line in lines), (
            "Old opcodes should be marked as RELOCATED in disassembly"
        )

    def test_disassemble_unified_bytecode(self) -> None:
        """Disassembling unified bytecode should show correct names."""
        unified_code = bytes([0xD0, 0x01, 0x60, 0xF4])  # TELL, 1, CONF_ADD, 244
        lines = FormatBridge.disassemble(unified_code)

        # The header-less bytecode won't have FLUX prefix,
        # so all bytes should be disassembled
        assert len(lines) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# CompiledInstruction Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCompiledInstruction:
    """Tests for the CompiledInstruction dataclass."""

    def test_to_bytes_simple(self) -> None:
        """A simple instruction should serialize correctly."""
        instr = CompiledInstruction(opcode=0xD0, operands=[0x01, 0x05])
        result = instr.to_bytes()

        assert result == bytes([0xD0, 0x01, 0x05])

    def test_to_bytes_with_u16(self) -> None:
        """An instruction with u16 operands should use big-endian encoding."""
        instr = CompiledInstruction(opcode=0xD6, operands=[3, 0x1234])
        result = instr.to_bytes()

        assert result[0] == 0xD6
        assert result[1] == 3
        assert result[2:4] == struct.pack('>H', 0x1234)

    def test_repr(self) -> None:
        """Instruction repr should show the opcode name."""
        instr = CompiledInstruction(opcode=0xD0, operands=[42, 5])
        r = repr(instr)
        assert "TELL" in r


# ══════════════════════════════════════════════════════════════════════════════
# Accessor Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAccessors:
    """Tests for static accessor methods."""

    def test_get_relocation_table(self) -> None:
        """get_relocation_table should return a non-empty dict."""
        table = FormatBridge.get_relocation_table()
        assert isinstance(table, dict)
        assert len(table) > 0
        # Should be a copy (not the same reference)
        table["fake"] = 999
        assert "fake" not in FormatBridge.get_relocation_table()

    def test_get_unified_opcode_map(self) -> None:
        """get_unified_opcode_map should return a non-empty dict."""
        omap = FormatBridge.get_unified_opcode_map()
        assert isinstance(omap, dict)
        assert "TELL" in omap
        assert omap["TELL"] == 0xD0

    def test_all_relocated_ops_in_unified_map(self) -> None:
        """Every relocated opcode should be in the unified map."""
        for _, (new_byte, name) in A2A_OLD_TO_NEW.items():
            assert name in UNIFIED_OPCODES, f"{name} not in unified map"
            assert UNIFIED_OPCODES[name] == new_byte
