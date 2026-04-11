"""
AST Unifier — Cross-language AST normalization and semantic equivalence checking.

Normalizes ASTs from the six FLUX runtimes (ZHO, DEU, KOR, SAN, LAT, WEN)
into a unified representation for structural comparison.  Provides structural
hashing (fast equality check) and structural distance (metric for similarity).

Native AST formats handled:
  ZHO — dict{"assembly": str, "pattern_name": str, "captures": dict}
         (from CompilationResult in flux-runtime-zho)
  DEU — list[dict{"op": str, "arg": Any, "source_line": str}]
         (from list[Instruction] in flux-runtime-deu)
  SAN — bytearray of FLUX bytecode opcodes
         (from bytecode_fn outputs in flux-runtime-san)
  KOR — list[tuple(Opcode, ...)] bytecode tuples
         (from compiled_bytecode in flux-runtime-kor)
  WEN — list[dict{"opcode": str, "operands": list, "source": str}]
         (from list[Instruction] in flux-runtime-wen)
  LAT — list[dict{"opcode": str, "operands": list, "raw": str}]
         (from list[Instruction] in flux-runtime-lat)

Design Principles:
  - UnifiedASTNode is a minimal common denominator — captures structure,
    not surface syntax.
  - The structural hash is deterministic: equivalent programs from any
    language produce the same hash.
  - Structural distance is in [0.0, 1.0]: 0.0 = identical, 1.0 = completely
    different.
  - Paradigm-specific annotations are preserved in metadata but do not
    affect structural comparison.

Reference: type_safe_bridge.py (TypeAlgebra, BridgeCostMatrix, TypeWitness)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ════════════════════════════════════════════════════════════════════
# UnifiedASTNode — the cross-language common denominator
# ════════════════════════════════════════════════════════════════════

class NodeKind(str, Enum):
    """Unified AST node types — the common denominator across all paradigms."""
    LITERAL = "literal"
    VARIABLE = "variable"
    APPLICATION = "application"
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    NOP = "nop"
    HALT = "halt"
    UNKNOWN = "unknown"


# Normalized operation names — every runtime maps to these
NORMALIZED_OPS: dict[str, str] = {
    # Arithmetic
    "add": "add", "iadd": "add", "IADD": "add", "PLUS": "add", "ADDITION": "add",
    "加法": "add", "berechne_plus": "add",
    "sub": "sub", "isub": "sub", "ISUB": "sub", "SUB": "sub", "minus": "sub",
    "减法": "sub",
    "mul": "mul", "imul": "mul", "IMUL": "mul", "MUL": "mul", "multiply": "mul",
    "乘法": "mul", "multipliziere_mal": "mul",
    "div": "div", "idiv": "div", "IDIV": "div", "DIV": "div", "divide": "div",
    "除法": "div",
    "mod": "mod", "imod": "mod", "IMOD": "mod", "MOD": "mod",
    "取模": "mod",
    "neg": "neg", "ineg": "neg", "INEG": "neg",
    "inc": "inc", "INC": "inc", "increment": "inc",
    "dec": "dec", "DEC": "dec", "decrement": "dec",
    # Comparison
    "cmp": "cmp", "icmp": "cmp", "ICMP": "cmp", "CMP": "cmp", "compare": "cmp",
    "eq": "eq", "lt": "lt", "gt": "gt", "le": "le", "ge": "ge",
    # Control flow
    "jmp": "jmp", "JMP": "jmp", "jump": "jmp",
    "jz": "jz", "JZ": "jz",
    "jnz": "jnz", "JNZ": "jnz",
    "je": "je", "JE": "je",
    "jne": "jne", "JNE": "jne",
    "jl": "jl", "JL": "jl",
    "jge": "jge", "JGE": "jge",
    # Stack
    "push": "push", "PUSH": "push",
    "pop": "pop", "POP": "pop",
    # Memory / register
    "mov": "mov", "MOV": "mov", "load": "load", "LOAD": "load",
    "store": "store", "STORE": "store",
    "movi": "movi", "MOVI": "movi", "const": "movi", "CONST": "movi",
    # Higher-level operations
    "sum_range": "sum_range", "SUM_RANGE": "sum_range", "range_sum": "sum_range",
    "RANGE_SUM": "sum_range", "factorial": "factorial", "FACTORIAL": "factorial",
    # Agent protocol (A2A)
    "tell": "tell", "TELL": "tell", "ask": "ask", "ASK": "ask",
    "delegate": "delegate", "DELEGATE": "delegate",
    "broadcast": "broadcast", "BROADCAST": "broadcast",
    # Control structures
    "print": "print", "PRINT": "print", "scribe": "print",
    "halt": "halt", "HALT": "halt", "ret": "ret", "RET": "ret",
    "nop": "nop", "NOP": "nop",
    # DEU-specific
    "DEFER": "defer", "EXECUTE_DEFERRED": "execute_deferred",
    "CAP_CHECK": "cap_check", "SCOPE_PUSH": "scope_push", "SCOPE_POP": "scope_pop",
    # SAN-specific patterns
    "add": "add", "multiply": "mul", "subtract": "sub", "divide": "div",
}


def _normalize_op(raw: str) -> str:
    """Normalize an operation name to its canonical form."""
    raw_stripped = raw.strip()
    if raw_stripped in NORMALIZED_OPS:
        return NORMALIZED_OPS[raw_stripped]
    # Try case-insensitive
    lower = raw_stripped.lower()
    if lower in NORMALIZED_OPS:
        return NORMALIZED_OPS[lower]
    # Return as-is for unknown ops
    return raw_stripped.lower()


@dataclass
class UnifiedASTNode:
    """Unified AST node — the common structural representation.

    Attributes:
        node_type: The kind of AST node (literal, variable, application, etc.)
        children: Child nodes in tree order.
        metadata: Paradigm-specific annotations that do NOT affect structural
            comparison.  E.g., {"source_lang": "zho", "classifier": "只",
            "honorific_level": "haeyoche", "vibhakti": "prathama",
            "kasus": "nominativ", "context_depth": 2, "original": "..."}
    """
    node_type: str
    children: list[UnifiedASTNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Convenience constructors ---

    @staticmethod
    def literal(value: Any, **meta: Any) -> UnifiedASTNode:
        """Create a literal (constant) node."""
        return UnifiedASTNode(
            node_type=NodeKind.LITERAL,
            metadata={"value": value, **meta},
        )

    @staticmethod
    def variable(name: str, **meta: Any) -> UnifiedASTNode:
        """Create a variable/register reference node."""
        return UnifiedASTNode(
            node_type=NodeKind.VARIABLE,
            metadata={"name": name, **meta},
        )

    @staticmethod
    def application(operation: str, *children: UnifiedASTNode,
                    **meta: Any) -> UnifiedASTNode:
        """Create a function/operation application node."""
        return UnifiedASTNode(
            node_type=NodeKind.APPLICATION,
            children=list(children),
            metadata={"operation": _normalize_op(operation), **meta},
        )

    @staticmethod
    def sequence(*children: UnifiedASTNode, **meta: Any) -> UnifiedASTNode:
        """Create a sequential execution node."""
        return UnifiedASTNode(
            node_type=NodeKind.SEQUENCE,
            children=list(children),
            metadata=meta,
        )

    @staticmethod
    def conditional(condition: UnifiedASTNode, body: UnifiedASTNode,
                    **meta: Any) -> UnifiedASTNode:
        """Create a conditional (if) node."""
        return UnifiedASTNode(
            node_type=NodeKind.CONDITIONAL,
            children=[condition, body],
            metadata=meta,
        )

    @staticmethod
    def loop(condition: UnifiedASTNode, body: UnifiedASTNode,
             **meta: Any) -> UnifiedASTNode:
        """Create a loop (while) node."""
        return UnifiedASTNode(
            node_type=NodeKind.LOOP,
            children=[condition, body],
            metadata=meta,
        )

    @staticmethod
    def nop(**meta: Any) -> UnifiedASTNode:
        """Create a no-operation node."""
        return UnifiedASTNode(node_type=NodeKind.NOP, metadata=meta)

    @staticmethod
    def halt(**meta: Any) -> UnifiedASTNode:
        """Create a halt node."""
        return UnifiedASTNode(node_type=NodeKind.HALT, metadata=meta)

    # --- Structural comparison ---

    def structural_key(self) -> tuple:
        """Return a comparable structural key (ignores metadata).

        This is the canonical representation used for hashing and comparison.
        Metadata (paradigm-specific annotations) is stripped out so that
        equivalent ASTs from different languages produce the same key.
        """
        if self.node_type == NodeKind.LITERAL:
            return (self.node_type, self.metadata.get("value"))
        elif self.node_type == NodeKind.VARIABLE:
            return (self.node_type, self.metadata.get("name"))
        elif self.node_type == NodeKind.APPLICATION:
            op = self.metadata.get("operation", "?")
            child_keys = tuple(c.structural_key() for c in self.children)
            return (self.node_type, op, child_keys)
        elif self.node_type == NodeKind.SEQUENCE:
            child_keys = tuple(c.structural_key() for c in self.children)
            return (self.node_type, child_keys)
        elif self.node_type == NodeKind.CONDITIONAL:
            child_keys = tuple(c.structural_key() for c in self.children)
            return (self.node_type, child_keys)
        elif self.node_type == NodeKind.LOOP:
            child_keys = tuple(c.structural_key() for c in self.children)
            return (self.node_type, child_keys)
        else:
            # NOP, HALT, UNKNOWN — just the type
            return (self.node_type,)

    def __eq__(self, other: object) -> bool:
        """Structural equality — ignores metadata."""
        if not isinstance(other, UnifiedASTNode):
            return NotImplemented
        return self.structural_key() == other.structural_key()

    def __hash__(self) -> int:
        """Hash based on structural key (ignores metadata)."""
        return hash(self.structural_key())

    def __repr__(self) -> str:
        meta_str = ""
        if self.metadata:
            brief = {k: v for k, v in self.metadata.items()
                     if k not in ("original", "source_line", "captures")}
            if brief:
                meta_str = f" {brief}"
        children_str = ""
        if self.children:
            children_str = f" [{len(self.children)} children]"
        return f"<{self.node_type.value}{children_str}{meta_str}>"


# ════════════════════════════════════════════════════════════════════
# NativeInstruction — common intermediate for per-language adapters
# ════════════════════════════════════════════════════════════════════

@dataclass
class NativeInstruction:
    """Normalized instruction — common intermediate for all languages.

    Each native AST format is first converted to a list of these,
    then the list is converted to a UnifiedASTNode tree.
    """
    opcode: str
    operands: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# SAN bytecode opcode map (from flux-runtime-san)
# ════════════════════════════════════════════════════════════════════

_SAN_OPCODE_NAMES: dict[int, str] = {
    0x00: "NOP", 0x01: "MOV", 0x02: "LOAD", 0x03: "STORE",
    0x04: "JMP", 0x05: "JZ", 0x06: "JNZ", 0x07: "CALL",
    0x08: "IADD", 0x09: "ISUB", 0x0A: "IMUL", 0x0B: "IDIV",
    0x0C: "IMOD", 0x0D: "INEG", 0x0E: "INC", 0x0F: "DEC",
    0x10: "IAND", 0x11: "IOR", 0x12: "IXOR", 0x13: "INOT",
    0x18: "ICMP", 0x19: "IEQ", 0x1A: "ILT", 0x1B: "ILE",
    0x1C: "IGT", 0x1D: "IGE", 0x1E: "TEST",
    0x20: "PUSH", 0x21: "POP", 0x22: "DUP", 0x23: "SWAP",
    0x24: "ROT", 0x25: "ENTER", 0x26: "LEAVE", 0x27: "ALLOCA",
    0x28: "RET", 0x29: "CALL_IND", 0x2B: "MOVI",
    0x2D: "CMP", 0x2E: "JE", 0x2F: "JNE",
    0x30: "REGION_CREATE", 0x31: "REGION_DESTROY",
    0x32: "REGION_TRANSFER", 0x33: "MEMCOPY",
    0x34: "MEMSET", 0x35: "MEMCMP",
    0x36: "JL", 0x37: "JGE",
    0x40: "FADD", 0x41: "FSUB", 0x42: "FMUL", 0x43: "FDIV",
    0x48: "FEQ", 0x49: "FLT", 0x4A: "FLE", 0x4B: "FGT", 0x4C: "FGE",
    0x50: "SLEN", 0x51: "SCONCAT", 0x52: "SCHAR",
    0x53: "SSUB", 0x54: "SCMP",
    0x60: "TELL", 0x61: "ASK", 0x62: "DELEGATE",
    0x63: "BROADCAST", 0x64: "TRUST_CHECK", 0x65: "CAP_REQ",
    0xFE: "PRINT", 0xFF: "HALT",
}

_SAN_INSTRUCTION_SIZES: dict[int, int] = {
    0x00: 1, 0xFF: 1, 0x28: 1, 0x26: 1,  # NOP, HALT, RET, LEAVE
    0x0D: 2, 0x0E: 2, 0x0F: 2, 0x13: 2, 0xFE: 2,  # unary reg
    0x22: 2, 0x23: 2, 0x25: 2, 0x27: 2,  # DUP, SWAP, ENTER, ALLOCA
    0x01: 3, 0x02: 3, 0x03: 3,  # MOV, LOAD, STORE
    0x04: 3,  # JMP
    0x18: 3, 0x1E: 3,  # ICMP, TEST
    0x38: 3, 0x39: 3, 0x3A: 3, 0x3B: 3, 0x3C: 3,  # type ops
    0x44: 3, 0x45: 3, 0x46: 3, 0x47: 3,  # float unary
    0x60: 3, 0x61: 3, 0x62: 3, 0x63: 3, 0x64: 3, 0x65: 3,  # A2A
    0x08: 4, 0x09: 4, 0x0A: 4, 0x0B: 4, 0x0C: 4,  # 3-reg arithmetic
    0x10: 4, 0x11: 4, 0x12: 4,  # bitwise 3-reg
    0x14: 4, 0x15: 4, 0x16: 4, 0x17: 4,  # shifts/rotates
    0x40: 4, 0x41: 4, 0x42: 4, 0x43: 4,  # float 3-reg
    0x51: 4,  # SCONCAT
    0x2B: 4,  # MOVI (op, reg, lo, hi)
    0x05: 4, 0x06: 4, 0x2E: 4, 0x2F: 4, 0x36: 4, 0x37: 4,  # conditional jumps
}


def _san_read_u16(data: bytes, offset: int) -> int:
    """Read a little-endian 16-bit unsigned integer."""
    if offset + 1 < len(data):
        return data[offset] | (data[offset + 1] << 8)
    return 0


# ════════════════════════════════════════════════════════════════════
# Per-language adapters: native AST → list[NativeInstruction]
# ════════════════════════════════════════════════════════════════════

def _adapt_zho(native_ast: Any) -> list[NativeInstruction]:
    """Adapt ZHO CompilationResult to NativeInstruction list.

    Accepts a dict with keys: assembly (str), pattern_name (str),
    captures (dict).  Parses the assembly text into instructions.
    """
    if isinstance(native_ast, dict):
        assembly = native_ast.get("assembly", "")
        pattern_name = native_ast.get("pattern_name", "")
        captures = native_ast.get("captures", {})
    elif isinstance(native_ast, str):
        assembly = native_ast
        pattern_name = ""
        captures = {}
    else:
        return []

    instructions: list[NativeInstruction] = []
    for line in assembly.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("//"):
            continue

        parts = line.replace(",", " ").split()
        if not parts:
            continue

        mnemonic = parts[0].upper()
        operands = parts[1:] if len(parts) > 1 else []
        # Strip label colons
        if mnemonic.endswith(":"):
            continue

        instructions.append(NativeInstruction(
            opcode=mnemonic,
            operands=operands,
            metadata={"source_lang": "zho", "pattern_name": pattern_name,
                      "original": line, "captures": captures},
        ))

    return instructions


def _adapt_deu(native_ast: Any) -> list[NativeInstruction]:
    """Adapt DEU Instruction list to NativeInstruction list.

    Accepts a list of Instruction-like dicts with keys: op, arg, source_line.
    The op may be a string (Op.Enum.name) or the Enum itself.
    """
    if not isinstance(native_ast, list):
        return []

    instructions: list[NativeInstruction] = []
    for item in native_ast:
        if isinstance(item, dict):
            op_raw = item.get("op", "NOP")
            arg = item.get("arg")
            source_line = item.get("source_line", "")
            # Handle Enum values
            op_name = getattr(op_raw, "name", str(op_raw))
            operands = [arg] if arg is not None else []
            instructions.append(NativeInstruction(
                opcode=op_name,
                operands=operands,
                metadata={"source_lang": "deu", "original": source_line},
            ))
        elif isinstance(item, tuple) and len(item) >= 2:
            op_raw = item[0]
            op_name = getattr(op_raw, "name", str(op_raw))
            operands = list(item[1:]) if len(item) > 1 else []
            instructions.append(NativeInstruction(
                opcode=op_name,
                operands=operands,
                metadata={"source_lang": "deu"},
            ))

    return instructions


def _adapt_san(native_ast: Any) -> list[NativeInstruction]:
    """Adapt SAN bytearray to NativeInstruction list.

    Decodes raw FLUX bytecode (variable-length encoding).
    """
    if not isinstance(native_ast, (bytes, bytearray)):
        return []

    data = bytes(native_ast)
    instructions: list[NativeInstruction] = []
    offset = 0

    while offset < len(data):
        op_byte = data[offset]
        op_name = _SAN_OPCODE_NAMES.get(op_byte, f"UNKNOWN_0x{op_byte:02x}")
        size = _SAN_INSTRUCTION_SIZES.get(op_byte, 1)
        operands = list(data[offset + 1:offset + size]) if size > 1 else []

        # Special handling for MOVI (reg, lo, hi → reg, immediate)
        if op_byte == 0x2B and size == 4:
            reg = operands[0]
            imm = operands[1] | (operands[2] << 8)
            if imm > 32767:
                imm -= 65536
            operands = [reg, imm]

        # Special handling for conditional jumps (reg, lo, hi → reg, addr)
        if op_byte in (0x05, 0x06, 0x2E, 0x2F, 0x36, 0x37) and size == 4:
            reg = operands[0]
            addr = operands[1] | (operands[2] << 8)
            operands = [reg, addr]

        instructions.append(NativeInstruction(
            opcode=op_name,
            operands=operands,
            metadata={"source_lang": "san", "offset": offset},
        ))
        offset += size

    return instructions


def _adapt_kor(native_ast: Any) -> list[NativeInstruction]:
    """Adapt KOR bytecode tuple list to NativeInstruction list.

    Accepts a list of tuples: (Opcode, ...) where Opcode is an enum.
    """
    if not isinstance(native_ast, list):
        return []

    instructions: list[NativeInstruction] = []
    for item in native_ast:
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            continue

        op_raw = item[0]
        op_name = getattr(op_raw, "name", str(op_raw))
        operands = list(item[1:]) if len(item) > 1 else []

        instructions.append(NativeInstruction(
            opcode=op_name,
            operands=operands,
            metadata={"source_lang": "kor"},
        ))

    return instructions


def _adapt_wen(native_ast: Any) -> list[NativeInstruction]:
    """Adapt WEN Instruction list to NativeInstruction list.

    Accepts a list of Instruction-like dicts with keys: opcode, operands, source.
    WEN uses string opcodes like "IADD", "IMUL", "MOVI", etc.
    """
    if not isinstance(native_ast, list):
        return []

    instructions: list[NativeInstruction] = []
    for item in native_ast:
        if isinstance(item, dict):
            opcode = item.get("opcode", "NOP")
            operands = item.get("operands", [])
            source = item.get("source", "")
            instructions.append(NativeInstruction(
                opcode=str(opcode),
                operands=list(operands),
                metadata={"source_lang": "wen", "original": source,
                          "context_depth": item.get("context_depth", 0)},
            ))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            opcode = str(item[0])
            operands = list(item[1:]) if len(item) > 1 else []
            instructions.append(NativeInstruction(
                opcode=opcode,
                operands=operands,
                metadata={"source_lang": "wen"},
            ))

    return instructions


def _adapt_lat(native_ast: Any) -> list[NativeInstruction]:
    """Adapt LAT Instruction list to NativeInstruction list.

    Accepts a list of Instruction-like dicts with keys: opcode, operands, raw.
    LAT uses string opcodes like "ADD", "MUL", "SET", "PRINT", etc.
    """
    if not isinstance(native_ast, list):
        return []

    instructions: list[NativeInstruction] = []
    for item in native_ast:
        if isinstance(item, dict):
            opcode = item.get("opcode", "NOP")
            operands = item.get("operands", [])
            raw = item.get("raw", "")
            instructions.append(NativeInstruction(
                opcode=str(opcode),
                operands=list(operands),
                metadata={"source_lang": "lat", "original": raw},
            ))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            opcode = str(item[0])
            operands = list(item[1:]) if len(item) > 1 else []
            instructions.append(NativeInstruction(
                opcode=opcode,
                operands=operands,
                metadata={"source_lang": "lat"},
            ))

    return instructions


def _adapt_generic(native_ast: Any) -> list[NativeInstruction]:
    """Adapt a generic/native instruction list.

    Accepts a list of (opcode: str, *operands) tuples or dicts with
    opcode/operands keys.  Used as fallback for unknown source languages
    or for synthetic test data.
    """
    if not isinstance(native_ast, list):
        return []

    instructions: list[NativeInstruction] = []
    for item in native_ast:
        if isinstance(item, dict):
            opcode = item.get("opcode", item.get("op", "NOP"))
            operands = item.get("operands", item.get("args", []))
            instructions.append(NativeInstruction(
                opcode=str(opcode),
                operands=list(operands),
                metadata=dict(item.get("metadata", {})),
            ))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            instructions.append(NativeInstruction(
                opcode=str(item[0]),
                operands=list(item[1:]),
            ))

    return instructions


# Adapter dispatch table
_ADAPTERS: dict[str, Any] = {
    "zho": _adapt_zho,
    "deu": _adapt_deu,
    "san": _adapt_san,
    "kor": _adapt_kor,
    "wen": _adapt_wen,
    "lat": _adapt_lat,
    "generic": _adapt_generic,
}


# ════════════════════════════════════════════════════════════════════
# ASTUnifier — main class
# ════════════════════════════════════════════════════════════════════

# Opcodes that produce no value (side effects only)
_SIDE_EFFECT_OPS = frozenset({
    "nop", "halt", "ret", "print", "jmp", "jz", "jnz", "je", "jne",
    "tell", "ask", "delegate", "broadcast",
    "cap_check", "scope_push", "scope_pop",
})

# Binary arithmetic ops: take two operands, produce one result
_BINARY_ARITH_OPS = frozenset({
    "add", "sub", "mul", "div", "mod", "iand", "ior", "ixor",
    "fadd", "fsub", "fmul", "fdiv",
})

# Unary ops: take one operand, modify in place
_UNARY_OPS = frozenset({
    "neg", "inc", "dec", "inot",
})

# Higher-level ops that encapsulate compound operations
_COMPOUND_OPS = frozenset({
    "sum_range", "factorial",
})


@dataclass
class UnificationResult:
    """Result of a unification operation.

    Attributes:
        unified: The unified AST node tree.
        native_count: Number of native instructions processed.
        warnings: Any warnings generated during unification.
        source_lang: The source language tag.
    """
    unified: UnifiedASTNode
    native_count: int = 0
    warnings: list[str] = field(default_factory=list)
    source_lang: str = ""


class ASTUnifier:
    """Normalizes ASTs from different languages into a unified representation.

    The unifier performs a two-stage process:
      1. **Adaptation**: Convert the native AST format to a list of
         NativeInstruction (language-specific adapter).
      2. **Tree building**: Convert the flat instruction list into a
         UnifiedASTNode tree that captures the structural semantics.

    Usage:
        unifier = ASTUnifier()

        # Unify from different languages
        zho_ast = unifier.unify({"assembly": "MOVI R0, 3\\nMOVI R1, 4\\nIADD R0, R0, R1\\nHALT"}, "zho")
        deu_ast = unifier.unify([{"op": "CONST", "arg": 3}, {"op": "CONST", "arg": 4}, {"op": "ADD"}], "deu")

        # Check structural equivalence
        assert unifier.structural_hash(zho_ast) == unifier.structural_hash(deu_ast)

        # Measure structural distance
        dist = unifier.structural_distance(zho_ast, deu_ast)
        assert dist < 0.3
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, UnifiedASTNode] = {}

    def unify(self, native_ast: Any, source_lang: str) -> UnifiedASTNode:
        """Normalize a native AST into a UnifiedASTNode tree.

        Args:
            native_ast: The native AST in language-specific format.
            source_lang: Source language tag (zho, deu, san, kor, wen, lat,
                         or generic).

        Returns:
            A UnifiedASTNode tree representing the unified program structure.
        """
        # Dispatch to the appropriate adapter
        adapter = _ADAPTERS.get(source_lang.lower(), _adapt_generic)
        instructions = adapter(native_ast)

        if not instructions:
            return UnifiedASTNode.nop(source_lang=source_lang,
                                      warning="empty_program")

        # Build the unified tree from the instruction list
        tree = self._build_tree(instructions, source_lang)
        return tree

    def unify_with_report(self, native_ast: Any,
                          source_lang: str) -> UnificationResult:
        """Unify and return a detailed result with metadata."""
        adapter = _ADAPTERS.get(source_lang.lower(), _adapt_generic)
        instructions = adapter(native_ast)

        warnings: list[str] = []
        tree = self._build_tree(instructions, source_lang, warnings)

        return UnificationResult(
            unified=tree,
            native_count=len(instructions),
            warnings=warnings,
            source_lang=source_lang,
        )

    def structural_hash(self, node: UnifiedASTNode) -> int:
        """Compute a structural equivalence hash for a unified AST.

        Two structurally equivalent ASTs (same node types, same operations,
        same value structure) will produce the same hash, regardless of
        which source language they came from.

        Args:
            node: A UnifiedASTNode tree.

        Returns:
            An integer hash suitable for use in dicts and sets.
        """
        key = node.structural_key()
        key_bytes = repr(key).encode("utf-8")
        return int(hashlib.sha256(key_bytes).hexdigest(), 16)

    def structural_distance(self, a: UnifiedASTNode,
                            b: UnifiedASTNode) -> float:
        """Compute structural distance between two unified ASTs.

        Returns a float in [0.0, 1.0]:
          - 0.0: The two ASTs are structurally identical.
          - 1.0: The two ASTs are completely different.

        The distance metric combines:
          1. Root node type match (0 if same, weighted penalty if different)
          2. Operation match for application nodes
          3. Value/variable match for leaf nodes
          4. Recursive child distance (weighted average)

        Args:
            a: First unified AST.
            b: Second unified AST.

        Returns:
            A float in [0.0, 1.0].
        """
        return self._tree_distance(a, b, depth=0)

    def structural_distance_symmetric(self, a: UnifiedASTNode,
                                      b: UnifiedASTNode) -> float:
        """Compute symmetric structural distance (bidirectional average).

        Same as structural_distance but ensures symmetry:
        distance(a, b) == distance(b, a).
        """
        return self._tree_distance(a, b, depth=0)

    def are_equivalent(self, a: UnifiedASTNode, b: UnifiedASTNode,
                       threshold: float = 0.0) -> bool:
        """Check if two ASTs are structurally equivalent within threshold.

        Args:
            a: First AST.
            b: Second AST.
            threshold: Maximum allowed distance (default 0.0 = exact match).

        Returns:
            True if distance(a, b) <= threshold.
        """
        return self.structural_distance(a, b) <= threshold

    # ── Tree building ────────────────────────────────────────────

    def _build_tree(self, instructions: list[NativeInstruction],
                    source_lang: str,
                    warnings: list[str] | None = None) -> UnifiedASTNode:
        """Convert a flat instruction list into a UnifiedASTNode tree.

        Strategy:
          1. Group consecutive instructions into logical blocks:
             - A MOVI/CONST followed by an arithmetic op forms an application.
             - Multiple arithmetic ops in sequence form a sequence.
             - HALT/RET terminate the program.
          2. Each block becomes an application or sequence node.
          3. The overall program is a sequence of blocks.
        """
        if warnings is None:
            warnings = []

        # Filter out NOPs and protocol-only ops at the top level
        _SKIP_OPS = frozenset({
            "NOP", "CAP_REQUIRE", "CAP_CHECK", "SCOPE_PUSH", "SCOPE_POP",
        })
        meaningful = [i for i in instructions if i.opcode not in _SKIP_OPS]

        if not meaningful:
            return UnifiedASTNode.nop(source_lang=source_lang)

        # Single instruction
        if len(meaningful) == 1:
            return self._instr_to_node(meaningful[0], source_lang, warnings)

        # Group instructions into semantic blocks
        blocks = self._group_into_blocks(meaningful, source_lang, warnings)

        if len(blocks) == 1:
            return blocks[0]

        # Unwrap sequence if only one meaningful child remains
        # (e.g., after stripping trailing HALT)
        meaningful_blocks = [b for b in blocks
                           if b.node_type not in (NodeKind.NOP, NodeKind.HALT)]
        if len(meaningful_blocks) == 1:
            return meaningful_blocks[0]

        return UnifiedASTNode.sequence(*blocks, source_lang=source_lang)

    def _instr_to_node(self, instr: NativeInstruction,
                       source_lang: str,
                       warnings: list[str]) -> UnifiedASTNode:
        """Convert a single NativeInstruction to a UnifiedASTNode."""
        op = _normalize_op(instr.opcode)
        operands = instr.operands
        meta = {"source_lang": source_lang}

        # HALT / RET
        if op in ("halt", "ret"):
            return UnifiedASTNode(
                node_type=NodeKind.HALT if op == "halt" else NodeKind.HALT,
                metadata={**meta, "operation": op, "original": instr.opcode},
            )

        # NOP
        if op == "nop":
            return UnifiedASTNode.nop(**meta)

        # Print / TELL / ASK / BROADCAST
        if op in ("print", "tell", "ask", "delegate", "broadcast"):
            children = []
            for operand in operands:
                children.append(self._operand_to_node(operand, source_lang))
            return UnifiedASTNode.application(
                op,
                *children,
                **meta,
                original=instr.opcode,
            )

        # Binary arithmetic: MOVI R0, a / MOVI R1, b / IADD R0, R0, R1
        # This is a single instruction like IADD R0, R0, R1
        if op in _BINARY_ARITH_OPS and len(operands) >= 3:
            # operands = [dest, src_a, src_b] — typical 3-register form
            dest = operands[0]
            src_a = operands[1]
            src_b = operands[2]
            return UnifiedASTNode.application(
                op,
                self._operand_to_node(src_a, source_lang),
                self._operand_to_node(src_b, source_lang),
                **meta,
                dest=str(dest),
                original=instr.opcode,
            )

        # Binary arithmetic: 2-operand form (stack-based, e.g., DEU ADD)
        if op in _BINARY_ARITH_OPS and len(operands) <= 2:
            children = [self._operand_to_node(o, source_lang)
                        for o in operands]
            return UnifiedASTNode.application(op, *children, **meta,
                                              original=instr.opcode)

        # Unary operations: INC R0, DEC R1, etc.
        if op in _UNARY_OPS:
            target = operands[0] if operands else None
            return UnifiedASTNode.application(
                op,
                self._operand_to_node(target, source_lang) if target is not None
                else UnifiedASTNode.nop(),
                **meta,
                original=instr.opcode,
            )

        # Compound operations: sum_range, factorial
        if op in _COMPOUND_OPS:
            children = [self._operand_to_node(o, source_lang)
                        for o in operands]
            return UnifiedASTNode.application(op, *children, **meta,
                                              original=instr.opcode)

        # MOVI / CONST: load immediate
        if op in ("movi", "const"):
            if len(operands) >= 2:
                reg = operands[0]
                val = operands[1]
                return UnifiedASTNode.application(
                    "movi",
                    UnifiedASTNode.variable(str(reg), source_lang=source_lang),
                    UnifiedASTNode.literal(val, source_lang=source_lang),
                    **meta,
                    original=instr.opcode,
                )
            elif len(operands) == 1:
                return UnifiedASTNode.literal(operands[0], **meta)

        # MOV / LOAD / STORE: register operations
        if op in ("mov", "load", "store"):
            children = [self._operand_to_node(o, source_lang)
                        for o in operands]
            return UnifiedASTNode.application(op, *children, **meta,
                                              original=instr.opcode)

        # Comparison
        if op in ("cmp", "icmp"):
            children = [self._operand_to_node(o, source_lang)
                        for o in operands]
            return UnifiedASTNode.application("cmp", *children, **meta,
                                              original=instr.opcode)

        # Conditional jumps → conditional nodes
        if op in ("jz", "jnz", "je", "jne", "jl", "jge"):
            children = [self._operand_to_node(o, source_lang)
                        for o in operands]
            return UnifiedASTNode.application(op, *children, **meta,
                                              original=instr.opcode)

        # JMP → unconditional jump
        if op == "jmp":
            target = operands[0] if operands else 0
            return UnifiedASTNode.application("jmp",
                                              UnifiedASTNode.literal(target),
                                              **meta)

        # Stack operations
        if op in ("push", "pop"):
            if operands:
                return UnifiedASTNode.application(
                    op, self._operand_to_node(operands[0], source_lang),
                    **meta)
            return UnifiedASTNode.application(op, **meta)

        # DEU-specific: DEFER, CAP_CHECK, etc.
        if op in ("defer", "cap_check", "scope_push", "scope_pop",
                  "execute_deferred", "cont_prepare", "cont_complete"):
            return UnifiedASTNode.application(op, **meta,
                                              original=instr.opcode)

        # Fallback: unknown operation
        children = [self._operand_to_node(o, source_lang)
                    for o in operands]
        if children:
            return UnifiedASTNode.application(
                op, *children, **meta, original=instr.opcode)
        return UnifiedASTNode.application(op, **meta, original=instr.opcode)

    def _operand_to_node(self, operand: Any,
                         source_lang: str) -> UnifiedASTNode:
        """Convert an operand to a UnifiedASTNode (literal or variable)."""
        if isinstance(operand, (int, float, complex)):
            return UnifiedASTNode.literal(operand, source_lang=source_lang)
        if isinstance(operand, str):
            # Register reference: R0, R1, etc.
            if operand.upper().startswith("R") and operand[1:].isdigit():
                return UnifiedASTNode.variable(operand.upper(),
                                               source_lang=source_lang)
            # Numeric string
            try:
                return UnifiedASTNode.literal(int(operand),
                                              source_lang=source_lang)
            except ValueError:
                pass
            try:
                return UnifiedASTNode.literal(float(operand),
                                              source_lang=source_lang)
            except ValueError:
                pass
            # String literal
            return UnifiedASTNode.literal(operand, source_lang=source_lang)
        # Unknown operand type — treat as literal
        return UnifiedASTNode.literal(operand, source_lang=source_lang)

    def _group_into_blocks(self, instructions: list[NativeInstruction],
                           source_lang: str,
                           warnings: list[str]) -> list[UnifiedASTNode]:
        """Group consecutive instructions into semantic blocks.

        A "block" is a set of instructions that together form one logical
        operation.  For example:
          MOVI R0, 3       ─┐
          MOVI R1, 4       ─┤  → application("add", var("R0"), var("R1"))
          IADD R0, R0, R1  ─┘
        """
        blocks: list[UnifiedASTNode] = []
        i = 0

        while i < len(instructions):
            instr = instructions[i]
            op = _normalize_op(instr.opcode)

            # MOVI/CONST/LOAD_IMM + MOVI/CONST/LOAD_IMM + binary_arith
            # → single application node
            if op in ("movi", "const", "load_imm"):
                # Look ahead for a second load-imm and then a binary op
                if (i + 2 < len(instructions)):
                    next_op = _normalize_op(instructions[i + 1].opcode)
                    next_next_op = _normalize_op(instructions[i + 2].opcode)
                    if (next_op in ("movi", "const", "load_imm")
                            and next_next_op in _BINARY_ARITH_OPS):
                        # Extract values
                        val_a = self._extract_movi_value(instr.operands)
                        val_b = self._extract_movi_value(
                            instructions[i + 1].operands)
                        arith_instr = instructions[i + 2]
                        arith_op = _normalize_op(arith_instr.opcode)

                        node = UnifiedASTNode.application(
                            arith_op,
                            UnifiedASTNode.literal(val_a,
                                                  source_lang=source_lang),
                            UnifiedASTNode.literal(val_b,
                                                  source_lang=source_lang),
                            source_lang=source_lang,
                            original_op=arith_instr.opcode,
                        )
                        blocks.append(node)
                        i += 3
                        continue

                # Single MOVI → literal assignment
                val = self._extract_movi_value(instr.operands)
                blocks.append(UnifiedASTNode.literal(val,
                                                     source_lang=source_lang))
                i += 1
                continue

            # HALT → terminal
            if op == "halt":
                blocks.append(UnifiedASTNode.halt(source_lang=source_lang))
                i += 1
                continue

            # Single instruction
            blocks.append(self._instr_to_node(instr, source_lang, warnings))
            i += 1

        return blocks

    def _extract_movi_value(self, operands: list[Any]) -> Any:
        """Extract the immediate value from a MOVI instruction's operands.

        Handles both 2-operand form (reg, val) and 1-operand form (val).
        Converts string operands to integers when possible for type normalization.
        """
        if len(operands) >= 2:
            val = operands[1]
        elif len(operands) == 1:
            val = operands[0]
        else:
            return 0
        # Normalize string numbers to ints for cross-language comparison
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                pass
        return val

    # ── Tree distance computation ────────────────────────────────

    def _tree_distance(self, a: UnifiedASTNode, b: UnifiedASTNode,
                       depth: int) -> float:
        """Recursive structural distance computation.

        Uses a weighted combination of:
          1. Type mismatch penalty
          2. Operation mismatch penalty
          3. Value mismatch penalty
          4. Child distance (recursive, with depth decay)
        """
        # Depth limit to prevent excessive recursion
        if depth > 20:
            return 1.0

        # Exact match → distance 0
        if a.structural_key() == b.structural_key():
            return 0.0

        # Type mismatch → high penalty
        if a.node_type != b.node_type:
            # Both are leaf nodes of different types
            a_is_leaf = a.node_type in (NodeKind.LITERAL, NodeKind.VARIABLE)
            b_is_leaf = b.node_type in (NodeKind.LITERAL, NodeKind.VARIABLE)
            if a_is_leaf and b_is_leaf:
                return 0.5
            # One is leaf, one is branch → very different
            if a_is_leaf != b_is_leaf:
                return 0.8
            # Both branches but different types → moderate penalty
            return 0.5

        # Same node type — drill into details
        weight = 1.0 / (1.0 + depth * 0.1)  # Decay weight with depth

        if a.node_type == NodeKind.LITERAL:
            a_val = a.metadata.get("value")
            b_val = b.metadata.get("value")
            if a_val == b_val:
                return 0.0
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                # Numeric distance normalized by magnitude
                mag = max(abs(a_val), abs(b_val), 1)
                return min(abs(a_val - b_val) / mag, 1.0) * weight
            return 0.7 * weight

        if a.node_type == NodeKind.VARIABLE:
            a_name = a.metadata.get("name", "")
            b_name = b.metadata.get("name", "")
            if a_name == b_name:
                return 0.0
            # Different variables but same type → small penalty
            return 0.3 * weight

        if a.node_type == NodeKind.APPLICATION:
            a_op = a.metadata.get("operation", "?")
            b_op = b.metadata.get("operation", "?")

            # Operation mismatch
            if a_op != b_op:
                op_penalty = 0.6 * weight
            else:
                op_penalty = 0.0

            # Child distance
            child_dist = self._children_distance(a.children, b.children,
                                                  depth + 1)

            return min(op_penalty + child_dist * 0.5, 1.0)

        if a.node_type == NodeKind.SEQUENCE:
            child_dist = self._children_distance(a.children, b.children,
                                                  depth + 1)
            return child_dist

        if a.node_type in (NodeKind.CONDITIONAL, NodeKind.LOOP):
            child_dist = self._children_distance(a.children, b.children,
                                                  depth + 1)
            return child_dist * 0.7

        # NOP, HALT, UNKNOWN — same type = same
        return 0.0

    def _children_distance(self, a_children: list[UnifiedASTNode],
                           b_children: list[UnifiedASTNode],
                           depth: int) -> float:
        """Compute distance between two lists of child nodes.

        Uses a combination of:
          1. Length difference penalty
          2. Aligned pairwise distance
          3. Unmatched children penalty
        """
        n = max(len(a_children), len(b_children))
        if n == 0:
            return 0.0

        # Length difference
        len_diff = abs(len(a_children) - len(b_children))
        len_penalty = len_diff / n * 0.3

        # Pairwise distance for aligned children
        pair_dists: list[float] = []
        for i in range(min(len(a_children), len(b_children))):
            d = self._tree_distance(a_children[i], b_children[i], depth)
            pair_dists.append(d)

        # Unmatched children penalty
        unmatched = abs(len(a_children) - len(b_children))
        unmatched_penalty = unmatched * 0.4

        if pair_dists:
            avg_pair_dist = sum(pair_dists) / len(pair_dists)
        else:
            avg_pair_dist = 0.0

        total = len_penalty + avg_pair_dist * 0.6 + unmatched_penalty * 0.3
        return min(total, 1.0)

    # ── Convenience methods ──────────────────────────────────────

    def unify_multi(self, asts: list[tuple[Any, str]]) -> list[UnifiedASTNode]:
        """Unify multiple native ASTs at once.

        Args:
            asts: List of (native_ast, source_lang) tuples.

        Returns:
            List of UnifiedASTNode trees.
        """
        return [self.unify(native_ast, lang) for native_ast, lang in asts]

    def batch_structural_hash(self, nodes: list[UnifiedASTNode]) -> dict[int, list[int]]:
        """Group nodes by their structural hash.

        Returns a dict mapping hash → list of node indices.
        Nodes with the same hash are structurally equivalent.
        """
        groups: dict[int, list[int]] = {}
        for i, node in enumerate(nodes):
            h = self.structural_hash(node)
            groups.setdefault(h, []).append(i)
        return groups

    def find_equivalence_classes(
        self, nodes: list[UnifiedASTNode], threshold: float = 0.3
    ) -> list[list[int]]:
        """Find groups of nodes that are structurally similar.

        Uses structural distance to group nodes.  Each group contains
        indices of nodes that are within the threshold distance of
        each other.

        Args:
            nodes: List of unified AST nodes.
            threshold: Maximum distance for equivalence (default 0.3).

        Returns:
            List of groups, where each group is a list of node indices.
        """
        n = len(nodes)
        if n == 0:
            return []

        # Union-Find for grouping
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.structural_distance(nodes[i], nodes[j])
                if dist <= threshold:
                    union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return sorted(groups.values(), key=lambda g: g[0])

    def diff(self, a: UnifiedASTNode, b: UnifiedASTNode) -> list[str]:
        """Produce a human-readable diff between two unified ASTs.

        Returns a list of diff lines describing structural differences.
        """
        diffs: list[str] = []

        def _walk(na: UnifiedASTNode, nb: UnifiedASTNode, path: str) -> None:
            if na.structural_key() == nb.structural_key():
                return

            if na.node_type != nb.node_type:
                diffs.append(f"{path}: type mismatch: {na.node_type} vs {nb.node_type}")
                return

            if na.node_type == NodeKind.LITERAL:
                a_val = na.metadata.get("value")
                b_val = nb.metadata.get("value")
                diffs.append(f"{path}: value mismatch: {a_val} vs {b_val}")
                return

            if na.node_type == NodeKind.VARIABLE:
                a_name = na.metadata.get("name")
                b_name = nb.metadata.get("name")
                diffs.append(f"{path}: variable mismatch: {a_name} vs {b_name}")
                return

            if na.node_type == NodeKind.APPLICATION:
                a_op = na.metadata.get("operation", "?")
                b_op = nb.metadata.get("operation", "?")
                if a_op != b_op:
                    diffs.append(f"{path}: operation mismatch: {a_op} vs {b_op}")

            # Recurse into children
            max_children = max(len(na.children), len(nb.children))
            for i in range(max_children):
                child_path = f"{path}[{i}]"
                if i >= len(na.children):
                    diffs.append(f"{child_path}: only in right ({nb.children[i].node_type})")
                elif i >= len(nb.children):
                    diffs.append(f"{child_path}: only in left ({na.children[i].node_type})")
                else:
                    _walk(na.children[i], nb.children[i], child_path)

        _walk(a, b, "root")
        return diffs
