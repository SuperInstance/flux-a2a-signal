"""
A2A ↔ FORMAT Bridge — Translates Signal JSON opcodes to FLUX FORMAT bytecodes.

This module bridges the A2A Signal protocol's JSON primitives (tell, ask, branch,
fork, co_iterate, discuss) to FLUX FORMAT bytecodes using the unified opcode table.

Handles:
  - Signal → bytecode compilation (compile_signal_to_bytecode)
  - Bytecode → Signal decompilation (decompile_bytecode_to_signal)
  - Old A2A opcode → unified FORMAT translation (translate_a2a_to_format)
  - Unified FORMAT → old A2A opcode translation (translate_format_to_a2a)
  - Confidence propagation via FORMAT CONF_* ops
  - Trust verification via STRIPCONF + runtime checks

Architecture:
  The bridge maintains two parallel opcode maps:
    1. OLD_A2A_MAP:  Original A2A opcode numbering (0x60-0xB2 for A2A/paradigm)
    2. NEW_UNIFIED_MAP: Relocated numbering (0xD0-0xF1 for A2A/paradigm)

  FORMAT spec (Oracle1) is authoritative for 0x00-0x69.
  A2A ops are relocated to 0xD0-0xF1 to avoid conflicts.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Constants & Format Types
# ══════════════════════════════════════════════════════════════════════════════

MAGIC_FLUX = b'\x46\x4C'  # 'FL'
UNIFIED_VERSION = 1
HEADER_SIZE = 4  # 2 magic + 1 version + 1 flags


class FormatClass(Enum):
    """Opcode format classification in the unified scheme."""
    FORMAT_A = "format_a"          # 0x00-0x03: System (HALT, NOP, RET, IRET)
    FORMAT_B = "format_b"          # 0x08-0x0F: Basic ops (INC, DEC, NOT, ...)
    FORMAT_C = "format_c"          # 0x10-0x17: System extensions (SYS, STRIPCONF)
    FORMAT_D = "format_d"          # 0x18-0x1F: Immediate arithmetic
    FORMAT_E = "format_e"          # 0x20-0x3F: Arithmetic, float, memory, control
    FORMAT_F = "format_f"          # 0x40-0x47: Wide immediate
    FORMAT_G = "format_g"          # 0x48-0x4F: Offset addressing
    CONF = "conf"                  # 0x60-0x69: Confidence ops
    A2A = "a2a"                    # 0xD0-0xDB: A2A protocol (relocated)
    PARADIGM = "paradigm"          # 0xDC-0xF1: Paradigm ops (relocated)
    SYSTEM = "system"              # 0xFE-0xFF: PRINT, HALT
    BASE = "base"                  # 0x00-0x5F: Shared base ISA


# ══════════════════════════════════════════════════════════════════════════════
# Unified Opcode Table
# ══════════════════════════════════════════════════════════════════════════════

# Maps opcode name → new unified byte value
UNIFIED_OPCODES: dict[str, int] = {
    # ── FORMAT Authoritative (0x00-0x69) ──
    # Format A
    "HALT": 0x00, "NOP": 0x01, "RET": 0x02, "IRET": 0x03,
    # Format B
    "INC": 0x08, "DEC": 0x09, "NOT": 0x0A, "NEG": 0x0B,
    "PUSH": 0x0C, "POP": 0x0D, "CONF_LOAD": 0x0E, "CONF_STORE": 0x0F,
    # Format C
    "SYS": 0x10, "STRIPCONF": 0x17,
    # Format D
    "FMT_MOVI": 0x18, "FMT_ADDI": 0x19, "FMT_SUBI": 0x1A,
    "FMT_ANDI": 0x1B, "FMT_ORI": 0x1C, "FMT_XORI": 0x1D,
    "FMT_SHLI": 0x1E, "FMT_SHRI": 0x1F,
    # Format E: Arithmetic
    "ADD": 0x20, "SUB": 0x21, "MUL": 0x22, "DIV": 0x23, "MOD": 0x24,
    "AND": 0x25, "OR": 0x26, "XOR": 0x27, "SHL": 0x28, "SHR": 0x29,
    "MIN": 0x2A, "MAX": 0x2B,
    "CMP_EQ": 0x2C, "CMP_LT": 0x2D, "CMP_GT": 0x2E, "CMP_NE": 0x2F,
    # Format E: Float
    "FADD": 0x30, "FSUB": 0x31, "FMUL": 0x32, "FDIV": 0x33,
    "FMIN": 0x34, "FMAX": 0x35, "FTOI": 0x36, "ITOF": 0x37,
    # Format E: Memory/Control
    "FMT_LOAD": 0x38, "FMT_STORE": 0x39, "FMT_MOV": 0x3A, "SWP": 0x3B,
    "FMT_JZ": 0x3C, "FMT_JNZ": 0x3D, "FMT_JLT": 0x3E, "FMT_JGT": 0x3F,
    # Format F
    "MOVI16": 0x40, "ADDI16": 0x41, "SUBI16": 0x42,
    "FMT_JMP": 0x44, "JAL": 0x45,
    # Format G
    "LOADOFF": 0x48, "STOREOFF": 0x49, "LOADI": 0x4A,
    # CONF_* (Oracle1 authoritative)
    "CONF_ADD": 0x60, "CONF_SUB": 0x61, "CONF_MUL": 0x62, "CONF_DIV": 0x63,
    "CONF_FADD": 0x64, "CONF_FSUB": 0x65, "CONF_FMUL": 0x66, "CONF_FDIV": 0x67,
    "CONF_MERGE": 0x68, "CONF_THRESHOLD": 0x69,

    # ── A2A Existing (relocated 0xD0-0xD5) ──
    "TELL": 0xD0, "ASK": 0xD1, "DELEGATE": 0xD2, "BROADCAST": 0xD3,
    "TRUST_CHECK": 0xD4, "CAP_REQUIRE": 0xD5,

    # ── A2A Extended (relocated 0xD6-0xDB) ──
    "OP_BRANCH": 0xD6, "OP_MERGE": 0xD7, "OP_DISCUSS": 0xD8,
    "OP_DELEGATE": 0xD9, "OP_CONFIDENCE": 0xDA, "OP_META": 0xDB,

    # ── WEN Paradigm (relocated 0xDC-0xE6) ──
    "IEXP": 0xDC, "IROOT": 0xDD, "VERIFY_TRUST": 0xDE, "CHECK_BOUNDS": 0xDF,
    "OPTIMIZE": 0xE0, "ATTACK": 0xE1, "DEFEND": 0xE2,
    "ADVANCE": 0xE3, "RETREAT": 0xE4, "SEQUENCE": 0xE5, "LOOP": 0xE6,

    # ── LAT Paradigm (relocated 0xE7-0xEE) ──
    "LOOP_START": 0xE7, "LOOP_END": 0xE8, "LAZY_DEFER": 0xE9,
    "CACHE_LOAD": 0xEA, "CACHE_STORE": 0xEB,
    "ROLLBACK_SAVE": 0xEC, "ROLLBACK_RESTORE": 0xED, "EVENTUAL_SCHEDULE": 0xEE,

    # ── Topic Register (relocated 0xEF-0xF1) ──
    "SET_TOPIC": 0xEF, "USE_TOPIC": 0xF0, "CLEAR_TOPIC": 0xF1,

    # ── System ──
    "PRINT": 0xFE, "FMT_HALT": 0xFF,
}

# Reverse map: unified byte → opcode name
UNIFIED_BY_VALUE: dict[int, str] = {v: k for k, v in UNIFIED_OPCODES.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Old A2A → New Unified Translation Table
# ══════════════════════════════════════════════════════════════════════════════

# Old byte → (new byte, opcode name)
A2A_OLD_TO_NEW: dict[int, tuple[int, str]] = {
    # A2A existing (was 0x60-0x65)
    0x60: (0xD0, "TELL"),
    0x61: (0xD1, "ASK"),
    0x62: (0xD2, "DELEGATE"),
    0x63: (0xD3, "BROADCAST"),
    0x64: (0xD4, "TRUST_CHECK"),
    0x65: (0xD5, "CAP_REQUIRE"),
    # A2A extended (was 0x70-0x76)
    0x70: (0xD6, "OP_BRANCH"),
    0x71: (0xD7, "OP_MERGE"),
    0x72: (0xD8, "OP_DISCUSS"),
    0x73: (0xD9, "OP_DELEGATE"),
    0x74: (0xDA, "OP_CONFIDENCE"),
    0x76: (0xDB, "OP_META"),
    # WEN paradigm (was 0x80-0x8A)
    0x80: (0xDC, "IEXP"),
    0x81: (0xDD, "IROOT"),
    0x82: (0xDE, "VERIFY_TRUST"),
    0x83: (0xDF, "CHECK_BOUNDS"),
    0x84: (0xE0, "OPTIMIZE"),
    0x85: (0xE1, "ATTACK"),
    0x86: (0xE2, "DEFEND"),
    0x87: (0xE3, "ADVANCE"),
    0x88: (0xE4, "RETREAT"),
    0x89: (0xE5, "SEQUENCE"),
    0x8A: (0xE6, "LOOP"),
    # LAT paradigm (was 0xA0-0xA7)
    0xA0: (0xE7, "LOOP_START"),
    0xA1: (0xE8, "LOOP_END"),
    0xA2: (0xE9, "LAZY_DEFER"),
    0xA3: (0xEA, "CACHE_LOAD"),
    0xA4: (0xEB, "CACHE_STORE"),
    0xA5: (0xEC, "ROLLBACK_SAVE"),
    0xA6: (0xED, "ROLLBACK_RESTORE"),
    0xA7: (0xEE, "EVENTUAL_SCHEDULE"),
    # Topic (was 0xB0-0xB2)
    0xB0: (0xEF, "SET_TOPIC"),
    0xB1: (0xF0, "USE_TOPIC"),
    0xB2: (0xF1, "CLEAR_TOPIC"),
}

# Reverse: new byte → old byte
A2A_NEW_TO_OLD: dict[int, tuple[int, str]] = {
    new: (old, name) for old, (new, name) in A2A_OLD_TO_NEW.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# Signal Primitive Types
# ══════════════════════════════════════════════════════════════════════════════

class SignalOp(str, Enum):
    """The six core A2A Signal primitives."""
    TELL = "tell"
    ASK = "ask"
    BRANCH = "branch"
    FORK = "fork"
    CO_ITERATE = "co_iterate"
    DISCUSS = "discuss"


@dataclass
class ConfidenceSpec:
    """Confidence value and threshold for propagation."""
    value: float = 1.0
    threshold: float = 0.5
    merge_strategy: str = "weighted_confidence"


@dataclass
class TrustSpec:
    """Trust verification specification."""
    agent_id: str = ""
    required_trust: float = 0.7
    verify_on_send: bool = True


@dataclass
class CompiledInstruction:
    """A single compiled instruction in the unified bytecode stream."""
    opcode: int
    operands: list[int] = field(default_factory=list)
    comment: str = ""

    def to_bytes(self) -> bytes:
        """Serialize to bytes: opcode followed by operand bytes."""
        buf = bytearray([self.opcode])
        for op in self.operands:
            if 0 <= op <= 0xFF:
                buf.append(op & 0xFF)
            elif 0 <= op <= 0xFFFF:
                buf.extend(struct.pack('>H', op))
            elif -0x8000 <= op <= 0x7FFF:
                buf.extend(struct.pack('>h', op))
            elif -2**31 <= op <= 2**31 - 1:
                buf.extend(struct.pack('>i', op))
            else:
                buf.extend(struct.pack('>q', op))
        return bytes(buf)

    def __repr__(self) -> str:
        name = UNIFIED_BY_VALUE.get(self.opcode, f"0x{self.opcode:02X}")
        ops_str = " ".join(f"{o}" for o in self.operands)
        return f"{name} {ops_str}".strip()


# ══════════════════════════════════════════════════════════════════════════════
# FormatBridge
# ══════════════════════════════════════════════════════════════════════════════

class FormatBridge:
    """Bridge between A2A Signal JSON and FLUX FORMAT bytecodes.

    Translates:
    - Signal's 6 JSON primitives (tell, ask, branch, fork, co_iterate, discuss)
      → FLUX FORMAT bytecodes using the unified opcode table

    Also handles:
    - Confidence propagation (Signal confidence → FORMAT CONF_* ops)
    - Trust verification (Signal trust → STRIPCONF + runtime checks)
    - Branch/merge (Signal fork → FORMAT JMP/JAL + multi-path)

    Usage::

        bridge = FormatBridge()

        # Compile a tell signal
        bytecode = bridge.compile_signal_to_bytecode({
            "op": "tell",
            "to": "agent-42",
            "message": "hello",
            "confidence": 0.95
        })

        # Decompile back
        signal = bridge.decompile_bytecode_to_signal(bytecode)

        # Translate old opcodes to new
        new_byte, fmt = bridge.translate_a2a_to_format(0x60)  # → (0xD0, A2A)
    """

    def __init__(self, emit_header: bool = True) -> None:
        """Initialize the bridge.

        Args:
            emit_header: If True, compiled bytecode includes the FLUX magic header.
        """
        self.emit_header = emit_header
        self._instructions: list[CompiledInstruction] = []
        self._signal_buffer: dict[str, Any] = {}

    # ── Signal → Bytecode Compilation ────────────────────────────────────

    def compile_signal_to_bytecode(self, signal: dict[str, Any]) -> bytes:
        """Compile an A2A Signal JSON primitive to FORMAT bytecode.

        Supported signal ops: tell, ask, branch, fork, co_iterate, discuss.

        Args:
            signal: A dict with at least "op" key specifying the primitive type.
                    Additional keys vary by primitive (e.g., "to", "message" for tell).

        Returns:
            Compiled bytecode with FLUX header (if emit_header=True).

        Raises:
            ValueError: If the signal is malformed or uses an unknown op.
        """
        self._instructions = []
        op = signal.get("op", "").lower()

        if op == SignalOp.TELL.value:
            self._compile_tell(signal)
        elif op == SignalOp.ASK.value:
            self._compile_ask(signal)
        elif op == SignalOp.BRANCH.value:
            self._compile_branch(signal)
        elif op == SignalOp.FORK.value:
            self._compile_fork(signal)
        elif op == SignalOp.CO_ITERATE.value:
            self._compile_co_iterate(signal)
        elif op == SignalOp.DISCUSS.value:
            self._compile_discuss(signal)
        else:
            raise ValueError(f"Unknown signal op: '{op}'. "
                             f"Expected one of: {[s.value for s in SignalOp]}")

        # Emit confidence if present
        confidence = signal.get("confidence")
        if confidence is not None and isinstance(confidence, (int, float)):
            self._emit_confidence(float(confidence))

        return self._assemble()

    def _compile_tell(self, signal: dict[str, Any]) -> None:
        """Compile a 'tell' signal → TELL opcode.

        Encoding: TELL(u8) agent_id(u8) msg_len(u8) [msg_bytes...]
        All operand fields are fixed-width u8 for deterministic layout.
        """
        to_agent = signal.get("to", "")
        msg = signal.get("message", "")
        msg_bytes = msg.encode("utf-8") if isinstance(msg, str) else b""

        # Encode agent ID as u8 (low byte of hash)
        agent_id = self._agent_id_to_u8(to_agent)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["TELL"],
            operands=[agent_id, len(msg_bytes)],
            comment=f"TELL → {to_agent}"
        ))
        # Append message payload as part of the instruction operands
        # Encoding: TELL(u8) agent_id(u8) msg_len(u8) [msg_bytes...]
        self._instructions[-1] = CompiledInstruction(
            opcode=UNIFIED_OPCODES["TELL"],
            operands=[agent_id, len(msg_bytes)] + list(msg_bytes),
            comment=f"TELL → {to_agent}"
        )

    def _compile_ask(self, signal: dict[str, Any]) -> None:
        """Compile an 'ask' signal → ASK opcode.

        Encoding: ASK(u8) agent_id(u8) msg_len(u8) [msg_bytes...]
        """
        to_agent = signal.get("to", "")
        msg = signal.get("message", "")
        msg_bytes = msg.encode("utf-8") if isinstance(msg, str) else b""

        agent_id = self._agent_id_to_u8(to_agent)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["ASK"],
            operands=[agent_id, len(msg_bytes)] + list(msg_bytes),
            comment=f"ASK → {to_agent}"
        ))

    def _compile_branch(self, signal: dict[str, Any]) -> None:
        """Compile a 'branch' signal → OP_BRANCH + path offsets."""
        branches = signal.get("branches", [])
        if not branches:
            raise ValueError("Branch signal requires 'branches' list")

        n_paths = len(branches)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["OP_BRANCH"],
            operands=[n_paths],
            comment=f"BRANCH {n_paths} paths"
        ))

        # Each branch gets a placeholder offset
        for i, branch in enumerate(branches):
            label = branch.get("label", f"path_{i}")
            self._instructions.append(CompiledInstruction(
                opcode=UNIFIED_OPCODES["OP_BRANCH"],
                operands=[i, self._hash_label(label) & 0xFFFF],
                comment=f"  path[{i}]: {label}"
            ))

    def _compile_fork(self, signal: dict[str, Any]) -> None:
        """Compile a 'fork' signal → DELEGATE with inherit flags.

        Encoding: DELEGATE(u8) agent_id(u8) inherit_flags(u8)
        """
        from_agent = signal.get("from", "self")
        inherit = signal.get("inherit", {})
        inherit_flags = 0
        if inherit.get("context", True):
            inherit_flags |= 0x01
        if inherit.get("trust_graph", False):
            inherit_flags |= 0x02
        if inherit.get("message_history", False):
            inherit_flags |= 0x04

        agent_id = self._agent_id_to_u8(from_agent)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["DELEGATE"],
            operands=[agent_id, inherit_flags],
            comment=f"FORK from {from_agent} (flags=0x{inherit_flags:02X})"
        ))

    def _compile_co_iterate(self, signal: dict[str, Any]) -> None:
        """Compile a 'co_iterate' signal → OP_DISCUSS with multi-agent config."""
        agents = signal.get("agents", [])
        rounds = signal.get("rounds", "until_convergence")
        rounds_val = min(rounds, 255) if isinstance(rounds, int) else 0

        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["OP_DISCUSS"],
            operands=[len(agents), rounds_val],
            comment=f"CO_ITERATE {len(agents)} agents, {rounds} rounds"
        ))

        # Trust checks for each agent
        for agent in agents:
            agent_id_str = agent.get("id", "") if isinstance(agent, dict) else str(agent)
            agent_id = self._agent_id_to_u8(agent_id_str)
            self._instructions.append(CompiledInstruction(
                opcode=UNIFIED_OPCODES["TRUST_CHECK"],
                operands=[agent_id],
                comment=f"  TRUST_CHECK agent {agent_id_str}"
            ))

    def _compile_discuss(self, signal: dict[str, Any]) -> None:
        """Compile a 'discuss' signal → OP_DISCUSS with format and rounds."""
        fmt = signal.get("format", "debate")
        participants = signal.get("participants", [])
        until = signal.get("until", {})
        max_rounds = until.get("max_rounds", 10) if isinstance(until, dict) else 10

        fmt_code = self._discuss_format_to_code(fmt)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["OP_DISCUSS"],
            operands=[fmt_code, len(participants), max_rounds],
            comment=f"DISCUSS {fmt}, {len(participants)} participants, {max_rounds} rounds"
        ))

        # Set topic if present
        topic = signal.get("topic", "")
        if topic:
            topic_id = self._hash_label(topic) & 0xFF
            self._instructions.append(CompiledInstruction(
                opcode=UNIFIED_OPCODES["SET_TOPIC"],
                operands=[topic_id],
                comment=f"SET_TOPIC: {topic}"
            ))

    # ── Confidence Propagation ───────────────────────────────────────────

    def _emit_confidence(self, value: float) -> None:
        """Emit CONF_ADD bytecode for a confidence value.

        Encoded as u8: value scaled to 0-254 (254 = 1.0).
        """
        scaled = int(max(0.0, min(1.0, value)) * 254)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["CONF_ADD"],
            operands=[scaled],
            comment=f"CONF_ADD {value:.3f}"
        ))

    def compile_confidence_merge(self, weights: list[float], threshold: float = 0.5) -> bytes:
        """Compile confidence merge operations.

        Uses CONF_MERGE (0x68) followed by CONF_THRESHOLD (0x69).
        Weights are encoded as u8: scaled to 0-254 (254 = 1.0).

        Args:
            weights: List of confidence weights for each source.
            threshold: Minimum confidence threshold.

        Returns:
            Compiled bytecode for the merge operation.
        """
        self._instructions = []

        # CONF_MERGE with weight count
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["CONF_MERGE"],
            operands=[len(weights)],
            comment=f"CONF_MERGE {len(weights)} sources"
        ))

        # Each weight as u8 (scaled to 0-254)
        for w in weights:
            scaled = int(max(0.0, min(1.0, w)) * 254)
            self._instructions.append(CompiledInstruction(
                opcode=UNIFIED_OPCODES["CONF_MERGE"],
                operands=[scaled],
                comment=f"  weight={w:.3f}"
            ))

        # CONF_THRESHOLD
        thr_scaled = int(max(0.0, min(1.0, threshold)) * 254)
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["CONF_THRESHOLD"],
            operands=[thr_scaled],
            comment=f"CONF_THRESHOLD {threshold:.3f}"
        ))

        return self._assemble()

    def compile_trust_verify(self, agent_id: str, required_trust: float = 0.7) -> bytes:
        """Compile trust verification: TRUST_CHECK + STRIPCONF.

        Args:
            agent_id: Agent to verify.
            required_trust: Minimum trust level required.

        Returns:
            Compiled bytecode.
        """
        self._instructions = []

        aid = self._agent_id_to_u8(agent_id)
        trust_scaled = int(max(0.0, min(1.0, required_trust)) * 254)

        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["TRUST_CHECK"],
            operands=[aid, trust_scaled],
            comment=f"TRUST_CHECK {agent_id} (min={required_trust:.2f})"
        ))

        # STRIPCONF: strip confidence if trust fails
        self._instructions.append(CompiledInstruction(
            opcode=UNIFIED_OPCODES["STRIPCONF"],
            operands=[0],  # flag: strip all confidence
            comment="STRIPCONF (on trust failure)"
        ))

        return self._assemble()

    # ── Bytecode → Signal Decompilation ─────────────────────────────────

    def decompile_bytecode_to_signal(self, bytecode: bytes) -> dict[str, Any]:
        """Decompile FORMAT bytecode back to an A2A Signal dict.

        Handles both unified (with FLUX header) and legacy (old A2A) bytecodes.

        Args:
            bytecode: Raw bytecode bytes.

        Returns:
            A dict representing the A2A Signal primitive.

        Raises:
            ValueError: If bytecode is too short or contains unknown opcodes.
        """
        if len(bytecode) < HEADER_SIZE and self.emit_header:
            raise ValueError("Bytecode too short")

        # Check for unified header
        offset = 0
        is_unified = False
        if len(bytecode) >= 2 and bytecode[:2] == MAGIC_FLUX:
            is_unified = True
            offset = HEADER_SIZE

        # Read first instruction
        if offset >= len(bytecode):
            raise ValueError("Bytecode has no instructions")

        opcode = bytecode[offset]

        # Translate old → new if not unified
        if not is_unified and opcode in A2A_OLD_TO_NEW:
            opcode = A2A_OLD_TO_NEW[opcode][0]

        # Decode based on opcode
        if opcode == UNIFIED_OPCODES["TELL"]:
            return self._decompile_tell(bytecode, offset, is_unified)
        elif opcode == UNIFIED_OPCODES["ASK"]:
            return self._decompile_ask(bytecode, offset, is_unified)
        elif opcode == UNIFIED_OPCODES["OP_BRANCH"]:
            return self._decompile_branch(bytecode, offset, is_unified)
        elif opcode == UNIFIED_OPCODES["DELEGATE"]:
            return self._decompile_fork(bytecode, offset, is_unified)
        elif opcode == UNIFIED_OPCODES["OP_DISCUSS"]:
            return self._decompile_discuss(bytecode, offset, is_unified)
        else:
            return {
                "op": "raw",
                "opcode": opcode,
                "name": UNIFIED_BY_VALUE.get(opcode, "UNKNOWN"),
                "bytes": bytecode[offset:].hex(),
            }

    def _decompile_tell(self, bytecode: bytes, offset: int, is_unified: bool) -> dict[str, Any]:
        """Decompile TELL instruction back to signal."""
        # Pattern: TELL <agent_id:u8> <msg_len:u8> [<msg_bytes>]
        agent_id = bytecode[offset + 1] if len(bytecode) > offset + 1 else 0
        msg_len = bytecode[offset + 2] if len(bytecode) > offset + 2 else 0

        msg_start = offset + 3
        msg_end = min(msg_start + msg_len, len(bytecode))
        msg = bytecode[msg_start:msg_end].decode("utf-8", errors="replace")

        signal: dict[str, Any] = {
            "op": "tell",
            "to": f"agent-{agent_id}",
            "message": msg,
        }

        # Check for trailing confidence (u8 scaled: 0-254 → 0.0-1.0)
        if msg_end + 2 <= len(bytecode):
            conf_opcode = bytecode[msg_end]
            if conf_opcode == UNIFIED_OPCODES["CONF_ADD"]:
                conf_raw = bytecode[msg_end + 1]
                signal["confidence"] = conf_raw / 254.0

        return signal

    def _decompile_ask(self, bytecode: bytes, offset: int, is_unified: bool) -> dict[str, Any]:
        """Decompile ASK instruction back to signal."""
        agent_id = bytecode[offset + 1] if len(bytecode) > offset + 1 else 0
        msg_len = bytecode[offset + 2] if len(bytecode) > offset + 2 else 0

        msg_start = offset + 3
        msg_end = min(msg_start + msg_len, len(bytecode))
        msg = bytecode[msg_start:msg_end].decode("utf-8", errors="replace")

        signal: dict[str, Any] = {
            "op": "ask",
            "to": f"agent-{agent_id}",
            "message": msg,
        }

        # Check for trailing confidence (u8 scaled: 0-254 → 0.0-1.0)
        if msg_end + 2 <= len(bytecode):
            conf_opcode = bytecode[msg_end]
            if conf_opcode == UNIFIED_OPCODES["CONF_ADD"]:
                conf_raw = bytecode[msg_end + 1]
                signal["confidence"] = conf_raw / 254.0

        return signal

    def _decompile_branch(
        self, bytecode: bytes, offset: int, is_unified: bool
    ) -> dict[str, Any]:
        """Decompile OP_BRANCH back to signal."""
        n_paths = bytecode[offset + 1] if len(bytecode) > offset + 1 else 0

        branches = []
        for i in range(n_paths):
            path_offset = offset + 2 + (i * 3)
            if path_offset + 2 < len(bytecode):
                idx = bytecode[path_offset]
                label_hash = (bytecode[path_offset + 1] << 8) | bytecode[path_offset + 2]
                branches.append({
                    "label": f"path_{idx}",
                    "weight": 1.0,
                    "body": [],
                })

        return {
            "op": "branch",
            "branches": branches,
            "merge": {"strategy": "weighted_confidence"},
        }

    def _decompile_fork(self, bytecode: bytes, offset: int, is_unified: bool) -> dict[str, Any]:
        """Decompile DELEGATE (fork) back to signal."""
        agent_id = bytecode[offset + 1] if len(bytecode) > offset + 1 else 0
        inherit_flags = bytecode[offset + 2] if len(bytecode) > offset + 2 else 0

        inherit = {
            "context": bool(inherit_flags & 0x01),
            "trust_graph": bool(inherit_flags & 0x02),
            "message_history": bool(inherit_flags & 0x04),
        }

        return {
            "op": "fork",
            "from": f"agent-{agent_id}",
            "inherit": inherit,
        }

    def _decompile_discuss(
        self, bytecode: bytes, offset: int, is_unified: bool
    ) -> dict[str, Any]:
        """Decompile OP_DISCUSS back to signal."""
        fmt_code = bytecode[offset + 1] if len(bytecode) > offset + 1 else 0
        n_parts = bytecode[offset + 2] if len(bytecode) > offset + 2 else 0
        max_rounds = bytecode[offset + 3] if len(bytecode) > offset + 3 else 10

        fmt_name = self._code_to_discuss_format(fmt_code)

        # Collect participants and trust checks
        participants = []
        ptr = offset + 4
        for i in range(n_parts):
            if ptr + 2 < len(bytecode):
                # Check if this is a participant or TRUST_CHECK
                next_op = bytecode[ptr]
                if next_op == UNIFIED_OPCODES["TRUST_CHECK"]:
                    # Trust check — extract agent ID
                    aid = bytecode[ptr + 1]
                    participants.append({"id": f"agent-{aid}", "stance": "neutral"})
                    ptr += 2
                else:
                    participants.append({"id": f"participant-{i}", "stance": "neutral"})
                    ptr += 1

        signal: dict[str, Any] = {
            "op": "discuss",
            "format": fmt_name,
            "participants": participants,
            "until": {"max_rounds": max_rounds},
        }

        # Check for SET_TOPIC
        if ptr < len(bytecode) and bytecode[ptr] == UNIFIED_OPCODES["SET_TOPIC"]:
            topic_id = bytecode[ptr + 1] if ptr + 1 < len(bytecode) else 0
            signal["topic"] = f"topic-{topic_id}"

        return signal

    # ── Opcode Translation ──────────────────────────────────────────────

    def translate_a2a_to_format(self, old_opcode: int) -> tuple[int, FormatClass]:
        """Translate an old A2A opcode byte to the unified FORMAT byte.

        Args:
            old_opcode: The old A2A opcode byte value.

        Returns:
            Tuple of (new_unified_byte, format_class).

        Raises:
            ValueError: If the old opcode is not in the translation table and
                        is in the conflict zone (0x60-0x69).
        """
        if old_opcode in A2A_OLD_TO_NEW:
            new_byte, name = A2A_OLD_TO_NEW[old_opcode]
            fmt = self._classify_opcode(new_byte)
            return new_byte, fmt

        # Check if it's in the FORMAT conflict zone
        if 0x60 <= old_opcode <= 0x69:
            raise ValueError(
                f"Opcode 0x{old_opcode:02X} is in the CONF_* zone (0x60-0x69) "
                f"and cannot be translated. It conflicts with FORMAT authoritative ops."
            )

        # Passthrough for base ops (0x00-0x5F) and system (0xFE-0xFF)
        fmt = self._classify_opcode(old_opcode)
        return old_opcode, fmt

    def translate_format_to_a2a(self, format_opcode: int) -> tuple[int, str]:
        """Translate a unified FORMAT opcode byte back to the old A2A byte.

        Args:
            format_opcode: The unified FORMAT opcode byte value.

        Returns:
            Tuple of (old_a2a_byte, opcode_name).

        Raises:
            ValueError: If the format opcode is not in the reverse table and
                        is in the conflict zone.
        """
        if format_opcode in A2A_NEW_TO_OLD:
            old_byte, name = A2A_NEW_TO_OLD[format_opcode]
            return old_byte, name

        # FORMAT authoritative opcodes (0x60-0x69) — no old A2A equivalent
        if 0x60 <= format_opcode <= 0x69:
            name = UNIFIED_BY_VALUE.get(format_opcode, f"UNKNOWN_0x{format_opcode:02X}")
            raise ValueError(
                f"Opcode 0x{format_opcode:02X} ({name}) is a FORMAT CONF_* op "
                f"with no old A2A equivalent."
            )

        # Passthrough for base ops
        name = UNIFIED_BY_VALUE.get(format_opcode, f"0x{format_opcode:02X}")
        return format_opcode, name

    # ── Bulk Translation ────────────────────────────────────────────────

    def translate_bytecode_old_to_new(self, bytecode: bytes) -> bytes:
        """Translate an entire old-A2A bytecode stream to unified bytes.

        Scans for relocated opcodes and substitutes their new values.
        Base ops (0x00-0x5F, 0xFE-0xFF) pass through unchanged.

        Args:
            bytecode: Old-format bytecode (without header).

        Returns:
            Translated bytecode (without header).
        """
        result = bytearray()
        i = 0
        while i < len(bytecode):
            byte = bytecode[i]
            if byte in A2A_OLD_TO_NEW:
                new_byte, _ = A2A_OLD_TO_NEW[byte]
                result.append(new_byte)
            else:
                result.append(byte)
            i += 1
        return bytes(result)

    def translate_bytecode_new_to_old(self, bytecode: bytes) -> bytes:
        """Translate unified bytecode back to old-A2A numbering.

        Args:
            bytecode: Unified bytecode (without header).

        Returns:
            Old-format bytecode (without header).
        """
        result = bytearray()
        for byte in bytecode:
            if byte in A2A_NEW_TO_OLD:
                old_byte, _ = A2A_NEW_TO_OLD[byte]
                result.append(old_byte)
            else:
                result.append(byte)
        return bytes(result)

    # ── Utility Methods ─────────────────────────────────────────────────

    @staticmethod
    def _classify_opcode(byte: int) -> FormatClass:
        """Classify an opcode byte into its FORMAT class."""
        if byte <= 0x03:
            return FormatClass.FORMAT_A
        if 0x08 <= byte <= 0x0F:
            return FormatClass.FORMAT_B
        if 0x10 <= byte <= 0x17:
            return FormatClass.FORMAT_C
        if 0x18 <= byte <= 0x1F:
            return FormatClass.FORMAT_D
        if 0x20 <= byte <= 0x3F:
            return FormatClass.FORMAT_E
        if 0x40 <= byte <= 0x47:
            return FormatClass.FORMAT_F
        if 0x48 <= byte <= 0x4F:
            return FormatClass.FORMAT_G
        if 0x60 <= byte <= 0x69:
            return FormatClass.CONF
        if 0xD0 <= byte <= 0xDB:
            return FormatClass.A2A
        if 0xDC <= byte <= 0xF1:
            return FormatClass.PARADIGM
        if byte >= 0xFE:
            return FormatClass.SYSTEM
        return FormatClass.BASE

    @staticmethod
    def _agent_id_to_u8(agent_id: str) -> int:
        """Convert an agent ID string to a u8 hash."""
        if not agent_id or agent_id == "self":
            return 0
        h = hash(agent_id) & 0xFF
        return max(1, h)  # 0 is reserved for "self"

    @staticmethod
    def _agent_id_to_u16(agent_id: str) -> int:
        """Convert an agent ID string to a u16 hash."""
        if not agent_id or agent_id == "self":
            return 0
        h = hash(agent_id) & 0xFFFF
        return max(1, h)  # 0 is reserved for "self"

    @staticmethod
    def _hash_label(label: str) -> int:
        """Hash a string label to a u16."""
        return hash(label) & 0xFFFF

    @staticmethod
    def _discuss_format_to_code(fmt: str) -> int:
        """Convert discuss format string to u8 code."""
        codes = {
            "debate": 0, "brainstorm": 1, "review": 2,
            "negotiate": 3, "peer_review": 4,
        }
        return codes.get(fmt.lower(), 0)

    @staticmethod
    def _code_to_discuss_format(code: int) -> str:
        """Convert u8 code back to discuss format string."""
        names = {0: "debate", 1: "brainstorm", 2: "review",
                 3: "negotiate", 4: "peer_review"}
        return names.get(code, "debate")

    def _assemble(self) -> bytes:
        """Assemble compiled instructions into final bytecode."""
        buf = bytearray()
        if self.emit_header:
            buf.extend(MAGIC_FLUX)
            buf.append(UNIFIED_VERSION)
            buf.append(0x00)  # flags

        for instr in self._instructions:
            buf.extend(instr.to_bytes())

        return bytes(buf)

    @staticmethod
    def get_relocation_table() -> dict[int, tuple[int, str]]:
        """Return the complete old→new relocation table (read-only copy).

        Returns:
            Dict mapping old_opcode → (new_opcode, name).
        """
        return dict(A2A_OLD_TO_NEW)

    @staticmethod
    def get_unified_opcode_map() -> dict[str, int]:
        """Return the complete unified opcode map (read-only copy).

        Returns:
            Dict mapping opcode_name → unified_byte.
        """
        return dict(UNIFIED_OPCODES)

    @staticmethod
    def check_no_conflicts() -> tuple[bool, list[str]]:
        """Verify that no byte conflicts exist in the proposed mapping.

        Returns:
            Tuple of (is_clean, conflict_descriptions).
        """
        all_values = list(UNIFIED_OPCODES.values())
        seen: dict[int, str] = {}
        conflicts: list[str] = []

        for name, val in UNIFIED_OPCODES.items():
            if val in seen:
                conflicts.append(
                    f"CONFLICT: 0x{val:02X} claimed by both '{seen[val]}' and '{name}'"
                )
            else:
                seen[val] = name

        # Check that relocated ops don't overlap with FORMAT authoritative zone
        for old_byte, (new_byte, name) in A2A_OLD_TO_NEW.items():
            if 0x60 <= new_byte <= 0x69:
                conflicts.append(
                    f"RELOCATION CONFLICT: {name} (0x{old_byte:02X}) → 0x{new_byte:02X} "
                    f"is in FORMAT CONF_* zone!"
                )

        return (len(conflicts) == 0, conflicts)

    @staticmethod
    def disassemble(bytecode: bytes) -> list[str]:
        """Disassemble bytecode into human-readable instruction strings.

        Args:
            bytecode: Raw bytecode (with or without header).

        Returns:
            List of instruction strings.
        """
        offset = 0
        if len(bytecode) >= 2 and bytecode[:2] == MAGIC_FLUX:
            offset = HEADER_SIZE

        lines: list[str] = []
        while offset < len(bytecode):
            byte = bytecode[offset]
            name = UNIFIED_BY_VALUE.get(byte, f"0x{byte:02X}")
            # Translate old opcodes for display
            if byte in A2A_OLD_TO_NEW:
                new_byte, _ = A2A_OLD_TO_NEW[byte]
                new_name = UNIFIED_BY_VALUE.get(new_byte, f"0x{new_byte:02X}")
                lines.append(f"  0x{offset:04X}: 0x{byte:02X} [{name}] → {new_name} (RELOCATED)")
            else:
                fmt = FormatBridge._classify_opcode(byte)
                lines.append(f"  0x{offset:04X}: 0x{byte:02X} {name} [{fmt.value}]")
            offset += 1

        return lines
