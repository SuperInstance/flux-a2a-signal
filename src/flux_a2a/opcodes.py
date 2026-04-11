"""
Universal Opcode Registry for all FLUX runtimes.

Provides a single source of truth for opcode definitions, cross-runtime
translation tables, version negotiation, and capability-based feature detection.

Architecture:
  - FluxOpcode (IntEnum): Canonical opcode enumeration using the TypeScript ISA hex values
  - OpcodeCategory (Enum): Taxonomic categories (CORE, ARITHMETIC, BITWISE, ...)
  - FluxOpcodeRegistry: Central registry with lookup, translation, and versioning
  - RuntimeID (Enum): Runtime identifiers for cross-runtime translation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Opcode Categories
# ════════════════════════════════════════════════════════════════════

class OpcodeCategory(Enum):
    """Taxonomic classification of opcodes."""
    CORE = "core"              # Fundamental VM control flow (NOP, MOV, JMP, etc.)
    ARITHMETIC = "arithmetic"    # Integer arithmetic
    BITWISE = "bitwise"          # Bit manipulation
    COMPARISON = "comparison"    # Integer and flag-based comparison
    STACK = "stack"              # Stack manipulation
    FUNCTION = "function"        # Call/return mechanism
    MEMORY = "memory"            # Memory regions and raw access
    TYPE = "type"                # Dynamic typing and type checking
    FLOAT = "float"              # Floating-point arithmetic
    FLOAT_CMP = "float_cmp"       # Floating-point comparison
    STRING = "string"            # String operations
    A2A = "a2a"                # Agent-to-agent protocol (existing)
    A2A_EXT = "a2a_extended"     # New A2A agent coordination opcodes
    SYSTEM = "system"            # I/O and halt
    PARADIGM = "paradigm"        # Language-specific paradigm opcodes
    RESERVED = "reserved"        # Reserved for future use


# ══════════════════════════════════════════════════════════════════
# Canonical Opcodes — IntEnum using TypeScript ISA hex values
# ══════════════════════════════════════════════════════════════════

class FluxOpcode(IntEnum):
    """Universal FLUX opcode enumeration.

    Uses the canonical TypeScript ISA hex values as the primary numeric space.
    All runtimes map to this canonical form for cross-runtime bytecode exchange.

    Hex ranges:
        0x00-0x07  Control flow
        0x08-0x0F  Integer arithmetic
        0x10-0x17  Bitwise
        0x18-0x1F  Comparison
        0x20-0x27  Stack operations
        0x28-0x2F  Function operations
        0x30-0x37  Memory management
        0x38-0x3F  Type operations
        0x40-0x47  Float arithmetic
        0x48-0x4F  Float comparison
        0x50-0x57  String operations
        0x58-0x5F  Reserved
        0x60-0x6F  A2A protocol (existing)
        0x70-0x7F  A2A protocol (new — agent coordination)
        0x80-0xFD  Paradigm opcodes (language-specific)
        0xFE-0xFF  System
    """

    # ── Control flow (0x00–0x07) ──────────────────────────────
    NOP = 0x00
    MOV = 0x01
    LOAD = 0x02
    STORE = 0x03
    JMP = 0x04
    JZ = 0x05
    JNZ = 0x06
    CALL = 0x07

    # ── Integer arithmetic (0x08–0x0F) ────────────────────────
    IADD = 0x08
    ISUB = 0x09
    IMUL = 0x0A
    IDIV = 0x0B
    IMOD = 0x0C
    INEG = 0x0D
    INC = 0x0E
    DEC = 0x0F

    # ── Bitwise (0x10–0x17) ────────────────────────────────────
    IAND = 0x10
    IOR = 0x11
    IXOR = 0x12
    INOT = 0x13
    ISHL = 0x14
    ISHR = 0x15
    ROTL = 0x16
    ROTR = 0x17

    # ── Comparison (0x18–0x1F) ─────────────────────────────────
    ICMP = 0x18
    IEQ = 0x19
    ILT = 0x1A
    ILE = 0x1B
    IGT = 0x1C
    IGE = 0x1D
    TEST = 0x1E
    SETCC = 0x1F

    # ── Stack operations (0x20–0x27) ────────────────────────────
    PUSH = 0x20
    POP = 0x21
    DUP = 0x22
    SWAP = 0x23
    ROT = 0x24
    ENTER = 0x25
    LEAVE = 0x26
    ALLOCA = 0x27

    # ── Function operations (0x28–0x2F) ───────────────────────
    RET = 0x28
    CALL_IND = 0x29
    TAILCALL = 0x2A
    MOVI = 0x2B
    IREM = 0x2C
    CMP = 0x2D
    JE = 0x2E
    JNE = 0x2F

    # ── Memory management (0x30–0x37) ──────────────────────────
    REGION_CREATE = 0x30
    REGION_DESTROY = 0x31
    REGION_TRANSFER = 0x32
    MEMCOPY = 0x33
    MEMSET = 0x34
    MEMCMP = 0x35
    JL = 0x36
    JGE = 0x37

    # ── Type operations (0x38–0x3F) ───────────────────────────
    CAST = 0x38
    BOX = 0x39
    UNBOX = 0x3A
    CHECK_TYPE = 0x3B
    CHECK_BOUNDS = 0x3C

    # ── Float arithmetic (0x40–0x47) ───────────────────────────
    FADD = 0x40
    FSUB = 0x41
    FMUL = 0x42
    FDIV = 0x43
    FNEG = 0x44
    FABS = 0x45
    FMIN = 0x46
    FMAX = 0x47

    # ── Float comparison (0x48–0x4F) ──────────────────────────
    FEQ = 0x48
    FLT = 0x49
    FLE = 0x4A
    FGT = 0x4B
    FGE = 0x4C

    # ── String operations (0x50–0x57) ───────────────────────────
    SLEN = 0x50
    SCONCAT = 0x51
    SCHAR = 0x52
    SSUB = 0x53
    SCMP = 0x54

    # ── Reserved (0x58–0x5F) ───────────────────────────────────
    # Reserved for future standardization

    # ── A2A Agent Protocol — existing (0x60–0x6F) ──────────────
    TELL = 0x60
    ASK = 0x61
    DELEGATE = 0x62
    BROADCAST = 0x63
    TRUST_CHECK = 0x64
    CAP_REQUIRE = 0x65

    # ── A2A Agent Protocol — new (0x70–0x7F) ────────────────
    OP_BRANCH = 0x70
    OP_MERGE = 0x71
    OP_DISCUSS = 0x72
    OP_DELEGATE = 0x73
    OP_CONFIDENCE = 0x74
    OP_META = 0x76  # 0x75 intentionally skipped (CAP_REQUIRE promoted to universal)

    # ── Paradigm opcodes (0x80–0xFD) ─────────────────────────
    # 0x80–0x8F: Classical Chinese (wen)
    IEXP = 0x80            # Exponentiation
    IROOT = 0x81            # Square root
    VERIFY_TRUST = 0x82      # 五常: 信 — verify trust/integrity
    CHECK_BOUNDS = 0x83       # 五常: 義 — validate bounds
    OPTIMIZE = 0x84          # 五常: 智 — optimize path
    ATTACK = 0x85            # 兵: 攻 — push data
    DEFEND = 0x86            # 兵: 守 — buffer data
    ADVANCE = 0x87           # 兵: 進 — advance
    RETREAT = 0x88            # 兵: 退 — backoff
    SEQUENCE = 0x89           # 制: 則 — sequential execution
    LOOP = 0x8A              # 制: 循 — loop construct

    # 0xA0–0xA7: Latin temporal markers (lat)
    LOOP_START = 0xA0        # Imperfectum — loop begin
    LOOP_END = 0xA1          # Loop end
    LAZY_DEFER = 0xA2        # Futurum — deferred computation
    CACHE_LOAD = 0xA3         # Perfectum — load cached result
    CACHE_STORE = 0xA4        # Cache store
    ROLLBACK_SAVE = 0xA5      # Plusquamperfectum — save state
    ROLLBACK_RESTORE = 0xA6    # Restore saved state
    EVENTUAL_SCHEDULE = 0xA7  # Futurum Exactum — eventual computation

    # 0xB0–0xB7: Topic register (wen, shared concept)
    SET_TOPIC = 0xB0          # Set topic register (定題)
    USE_TOPIC = 0xB1          # Use topic as implicit operand (用題)
    CLEAR_TOPIC = 0xB2        # Clear topic register (清題)

    # ── System (0xFE–0xFF) ─────────────────────────────────
    PRINT = 0xFE
    HALT = 0xFF


# ══════════════════════════════════════════════════════════════════
# Opcode Metadata Registry
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OpcodeInfo:
    """Metadata for a single opcode."""
    name: str
    hex_value: int
    category: OpcodeCategory
    category_id: int
    operands: int            # Number of expected operands
    description: str
    first_version: int       # Minimum bytecode version that includes this opcode
    required_features: Tuple[str, ...]  # Feature flags needed


# Build metadata table
_OPCODE_INFO: Dict[FluxOpcode, OpcodeInfo] = {}


def _build_info_table() -> None:
    """Build the complete opcode metadata table."""
    table = {
        # (opcode, category, operands, description, first_version, required_features)
        # ── Core ──
        (FluxOpcode.NOP,           OpcodeCategory.CORE,      0, "No operation",                                         1, ()),
        (FluxOpcode.MOV,           OpcodeCategory.CORE,      2, "Copy register to register",                              1, ()),
        (FluxOpcode.LOAD,          OpcodeCategory.CORE,      2, "Load from memory/register",                            1, ()),
        (FluxOpcode.STORE,         OpcodeCategory.CORE,      2, "Store to memory/register",                            1, ()),
        (FluxOpcode.JMP,           OpcodeCategory.CORE,      1, "Unconditional jump",                                  1, ()),
        (FluxOpcode.JZ,            OpcodeCategory.CORE,      2, "Jump if zero",                                       1, ()),
        (FluxOpcode.JNZ,           OpcodeCategory.CORE,      2, "Jump if not zero",                                   1, ()),
        (FluxOpcode.CALL,          OpcodeCategory.CORE,      1, "Call subroutine",                                    1, ()),
        # ── Arithmetic ──
        (FluxOpcode.IADD,          OpcodeCategory.ARITHMETIC, 3, "Integer add",                                        1, ()),
        (FluxOpcode.ISUB,          OpcodeCategory.ARITHMETIC, 3, "Integer subtract",                                     1, ()),
        (FluxOpcode.IMUL,          OpcodeCategory.ARITHMETIC, 3, "Integer multiply",                                    1, ()),
        (FluxOpcode.IDIV,          OpcodeCategory.ARITHMETIC, 3, "Integer divide",                                      1, ()),
        (FluxOpcode.IMOD,          Opcode.ARITHMETIC, 3, "Integer modulo",                                      1, ()),
        (FluxOpcode.INEG,          OpcodeCategory.ARITHMETIC, 1, "Integer negate",                                     1, ()),
        (FluxOpcode.INC,           OpcodeCategory.ARITHMETIC, 1, "Increment by one",                                    1, ()),
        (FluxOpcode.DEC,           Opcode.ARITHMETIC, 1, "Decrement by one",                                    1, ()),
        # ── Bitwise ──
        (FluxOpcode.IAND,          OpcodeCategory.BITWISE,    3, "Bitwise AND",                                        1, ()),
        (FluxOpcode.IOR,           OpcodeCategory.BITWISE,    3, "Bitwise OR",                                         1, ()),
        (FluxOpcode.IXOR,          OpcodeCategory.BITWISE,    3, "Bitwise XOR",                                         1, ()),
        (FluxOpcode.INOT,          OpcodeCategory.BITWISE,    1, "Bitwise NOT",                                         1, ()),
        (FluxOpcode.ISHL,          OpcodeCategory.BITWISE,    3, "Shift left",                                          1, ()),
        (FluxOpcode.ISHR,          OpcodeCategory.BITWISE,    3, "Shift right (arithmetic)",                            1, ()),
        (FluxOpcode.ROTL,          OpcodeCategory.BITWISE,    3, "Rotate left",                                         1, ()),
        (FluxOpcode.ROTR,          OpcodeCategory.BITWISE,    3, "Rotate right",                                        1, ()),
        # ── Comparison ──
        (FluxOpcode.ICMP,          OpcodeCategory.COMPARISON,  2, "Integer compare (sets flags)",                           1, ()),
        (FluxOpcode.IEQ,           OpcodeCategory.COMPARISON,  3, "Is equal",                                            1, ()),
        (FluxOpcode.ILT,           OpcodeCategory.COMPARISON,  3, "Is less than",                                        1, ()),
        (FluxOpcode.ILE,           OpcodeCategory.COMPARISON,  3, "Is less or equal",                                   1, ()),
        (FluxOpcode.IGT,           OpcodeCategory.COMPARISON,  3, "Is greater than",                                     1, ()),
        (FluxOpcode.IGE,           OpcodeCategory.COMPARISON,  3, "Is greater or equal",                                1, ()),
        (FluxOpcode.TEST,          OpcodeCategory.COMPARISON,  1, "Test register value",                                 1, ()),
        (FluxOpcode.SETCC,         OpcodeCategory.COMPARISON,  1, "Set condition code",                                    1, ()),
        # ── Stack ──
        (FluxOpcode.PUSH,          OpcodeCategory.STACK,      1, "Push onto stack",                                     1, ()),
        (FluxOpcode.POP,           OpcodeCategory.STACK,      1, "Pop from stack",                                      1, ()),
        (FluxOpcode.DUP,           OpcodeCategory.STACK,      0, "Duplicate top of stack",                               1, ()),
        (FluxOpcode.SWAP,          OpcodeCategory.STACK,      0, "Swap top two stack elements",                           1, ()),
        (FluxOpcode.ROT,           OpcodeCategory.STACK,      0, "Rotate top three stack elements",                      1, ()),
        (FluxOpcode.ENTER,         OpcodeCategory.STACK,      1, "Enter stack frame (push frame pointer)",                 1, ()),
        (FluxOpcode.LEAVE,         OpcodeCategory.STACK,      1, "Leave stack frame",                                  1, ()),
        (FluxOpcode.ALLOCA,        OpcodeCategory.STACK,      1, "Allocate stack space",                                 1, ()),
        # ── Function ──
        (FluxOpcode.RET,           OpcodeCategory.FUNCTION,   0, "Return from subroutine",                            1, ()),
        (FluxOpcode.CALL_IND,      OpcodeCategory.FUNCTION,   2, "Indirect call via register",                         1, ()),
        (FluxOpcode.TAILCALL,      OpcodeCategory.FUNCTION,   1, "Tail call (optimized recursion)",                   1, ()),
        (FluxOpcode.MOVI,          OpcodeCategory.FUNCTION,   2, "Move immediate (signed 16-bit)",                   1, ()),
        (FluxOpcode.IREM,          OpcodeCategory.FUNCTION,   3, "Integer remainder",                                  1, ()),
        (FluxOpcode.CMP,           OpcodeCategory.FUNCTION,   2, "Compare (sets flags)",                              1, ()),
        (FluxOpcode.JE,            OpcodeCategory.FUNCTION,   2, "Jump if equal (ZF=1)",                             1, ()),
        (FluxOpcode.JNE,           OpcodeCategory.FUNCTION,   2, "Jump if not equal (ZF=0)",                         1, ()),
        # ── Memory ──
        (FluxOpcode.REGION_CREATE,  OpcodeCategory.MEMORY,    1, "Create memory region",                                1, ()),
        (FluxOpcode.REGION_DESTROY, OpcodeCategory.MEMORY,    1, "Destroy memory region",                               1, ()),
        (FluxOpcode.REGION_TRANSFER,OpcodeCategory.MEMORY,    2, "Transfer region ownership",                           1, ()),
        (FluxOpcode.MEMCOPY,       OpcodeCategory.MEMORY,    3, "Copy memory block",                                 1, ()),
        (FluxOpcode.MEMSET,        OpcodeCategory.MEMORY,    3, "Fill memory block",                                  1, ()),
        (FluxOpcode.MEMCMP,        OpcodeCategory.MEMORY,    3, "Compare memory blocks",                              1, ()),
        (FluxOpcode.JL,            OpcodeCategory.MEMORY,    2, "Jump if less than (flag-based)",                    1, ()),
        (FluxOpcode.JGE,           OpcodeCategory.MEMORY,    2, "Jump if greater-or-equal (flag-based)",            1, ()),
        # ── Type ──
        (FluxOpcode.CAST,          OpcodeCategory.TYPE,       2, "Cast to different type",                              1, ()),
        (FluxOpcode.BOX,           OpcodeCategory.TYPE,       1, "Box value into dynamic container",                     1, ()),
        (FluxOpcode.UNBOX,         OpcodeCategory.TYPE,       1, "Unbox dynamic value",                               1, ()),
        (FluxOpcode.CHECK_TYPE,    OpcodeCategory.TYPE,       2, "Check runtime type",                                 1, ()),
        (FluxOpcode.CHECK_BOUNDS,  OpcodeCategory.TYPE,       2, "Check array bounds",                                1, ()),
        # ── Float ──
        (FluxOpcode.FADD,          OpcodeCategory.FLOAT,      3, "Float add",                                         1, ()),
        (FluxOpcode.FSUB,          OpcodeCategory.FLOAT,      3, "Float subtract",                                      1, ()),
        (FluxOpcode.FMUL,          OpcodeCategory.FLOAT,      3, "Float multiply",                                    1, ()),
        (FluxOpcode.FDIV,          OpcodeCategory.FLOAT,      3, "Float divide",                                      1, ()),
        (FluxOpcode.FNEG,          OpcodeCategory.FLOAT,      1, "Float negate",                                     1, ()),
        (FluxOpcode.FABS,          OpcodeCategory.FLOAT,      1, "Float absolute value",                                1, ()),
        (FluxOpcode.FMIN,          OpcodeCategory.FLOAT,      3, "Float minimum",                                     1, ()),
        (FluxOpcode.FMAX,          OpcodeCategory.FLOAT,      3, "Float maximum",                                     1, ()),
        # ── Float comparison ──
        (FluxOpcode.FEQ,           OpcodeCategory.FLOAT_CMP,  3, "Float equal",                                       1, ()),
        (FluxOpcode.FLT,           OpcodeCategory.FLOAT_CMP,  3, "Float less than",                                    1, ()),
        (FluxOpcode.FLE,           OpcodeCategory.FLOAT_CMP,  3, "Float less or equal",                               1, ()),
        (FluxOpcode.FGT,           OpcodeCategory.FLOAT_CMP,  3, "Float greater than",                                 1, ()),
        (FluxOpcode.FGE,           OpcodeCategory.FLOAT_CMP,  3, "Float greater or equal",                              1, ()),
        # ── String ──
        (FluxOpcode.SLEN,          OpcodeCategory.STRING,    2, "String length",                                      1, ()),
        (FluxOpcode.SCONCAT,       OpcodeCategory.STRING,    2, "String concatenation",                                1, ()),
        (FluxOpcode.SCHAR,         OpcodeCategory.STRING,    2, "Character access",                                    1, ()),
        (FluxOpcode.SSUB,          OpcodeCategory.STRING,    3, "Substring",                                          1, ()),
        (FluxOpcode.SCMP,          OpcodeCategory.STRING,    2, "String compare",                                      1, ()),
        # ── A2A existing ──
        (FluxOpcode.TELL,          OpcodeCategory.A2A,        2, "Send one-way message to agent",                         2, ("a2a",)),
        (FluxOpcode.ASK,           OpcodeCategory.A2A,        2, "Question/request to agent",                            2, ("a2a",)),
        (FluxOpcode.DELEGATE,      OpcodeCategory.A2A,        2, "Transfer task to agent",                             2, ("a2a",)),
        (FluxOpcode.BROADCAST,     OpcodeCategory.A2A,        1, "Announce to all agents",                             2, ("a2a",)),
        (FluxOpcode.TRUST_CHECK,    OpcodeCategory.A2A,        2, "Verify trust relationship",                          2, ("a2a",)),
        (FluxOpcode.CAP_REQUIRE,    OpcodeCategory.A2A,        2, "Require capability",                                  2, ("a2a",)),
        # ── A2A new (agent coordination) ──
        (FluxOpcode.OP_BRANCH,       OpcodeCategory.A2A_EXT,   2, "Fork execution into parallel agents",                 3, ("a2a", "branch")),
        (FluxOpcode.OP_MERGE,       OpcodeCategory.A2A_EXT,   2, "Combine results from agents",                        3, ("a2a", "merge")),
        (FluxOpcode.OP_DISCUSS,     OpcodeCategory.A2A_EXT,   3, "Structured multi-round discussion",                    3, ("a2a", "discuss")),
        (FluxOpcode.OP_DELEGATE,    OpcodeCategory.A2A_EXT,   3, "Delegate with wait-for-result",                      3, ("a2a", "delegate")),
        (FluxOpcode.OP_CONFIDENCE,  OpcodeCategory.A2A_EXT,   1, "Set/query confidence level",                         3, ("a2a", "confidence")),
        (FluxOpcode.OP_META,        OpcodeCategory.A2A_EXT,   3, "Self-referential: VM modifies own opcodes",            3, ("a2a", "meta")),
        # ── Paradigm: Classical Chinese (wen) ──
        (FluxOpcode.IEXP,          OpcodeCategory.PARADIGM,   2, "Exponentiation (幂)",                                 2, ("wen_paradigm",)),
        (FluxOpcode.IROOT,         OpcodeCategory.PARADIGM,   2, "Square root (根)",                                    2, ("wen_paradigm",)),
        (FluxOpcode.VERIFY_TRUST,   OpcodeCategory.PARADIGM,   2, "Verify trust — 五常:信",                           2, ("wen_paradigm", "confucian")),
        (FluxOpcode.CHECK_BOUNDS,  OpcodeCategory.PARADIGM,   2, "Check bounds — 五常:義",                            2, ("wen_paradigm", "confucian")),
        (FluxOpcode.OPTIMIZE,     OpcodeCategory.PARADIGM,   2, "Optimize path — 五常:智",                           2, ("wen_paradigm", "confucian")),
        (FluxOpcode.ATTACK,        OpcodeCategory.PARADIGM,   2, "Push data — 兵:攻",                                  2, ("wen_paradigm", "military")),
        (FluxOpcode.DEFEND,        OpcodeCategory.PARADIGM,   2, "Buffer data — 兵:守",                                  2, ("wen_paradigm", "military")),
        (FluxOpcode.ADVANCE,       OpcodeCategory.PARADIGM,   2, "Advance — 兵:進",                                    2, ("wen_paradigm", "military")),
        (FluxOpcode.RETREAT,       OpcodeCategory.PARADIGM,   2, "Back off — 兵:退",                                    2, ("wen_paradigm", "military")),
        (FluxOpcode.SEQUENCE,      OpcodeCategory.PARADIGM,   1, "Sequential — 制:則",                                   2, ("wen_paradigm",)),
        (FluxOpcode.LOOP,          OpcodeCategory.PARADIGM,   1, "Loop construct — 制:循",                               2, ("wen_paradigm",)),
        # ── Paradigm: Latin temporal (lat) ──
        (FluxOpcode.LOOP_START,    OpcodeCategory.PARADIGM,   1, "Loop begin — Imperfectum",                          2, ("lat_paradigm", "temporal")),
        (FluxOpcode.LOOP_END,      OpcodeCategory.PARADIGM,   1, "Loop end",                                          2, ("lat_paradigm", "temporal")),
        (FluxOpcode.LAZY_DEFER,    OpcodeCategory.PARADIGM,   1, "Defer computation — Futurum",                          2, ("lat_paradigm", "temporal")),
        (FluxOpcode.CACHE_LOAD,    OpcodeCategory.PARADIGM,   1, "Load cached result — Perfectum",                        2, ("lat_paradigm", "temporal")),
        (FluxOpcode.CACHE_STORE,   OpcodeCategory.PARADIGM,   1, "Store to cache — Perfectum",                             2, ("lat_paradigm", "temporal")),
        (FluxOpcode.ROLLBACK_SAVE, OpcodeCategory.PARADIGM,   0, "Save state — Plusquamperfectum",                     2, ("lat_paradigm", "temporal")),
        (FluxOpcode.ROLLBACK_RESTORE,OpcodeCategory.PARADIGM,   1, "Restore state — Plusquamperfectum",                  2, ("lat_paradigm", "temporal")),
        (FluxOpcode.EVENTUAL_SCHEDULE,OpcodeCategory.PARADIGM,   1, "Schedule eventual — Futurum Exactum",                2, ("lat_paradigm", "temporal")),
        # ── Paradigm: Topic register ──
        (FluxOpcode.SET_TOPIC,     OpcodeCategory.PARADIGM,   1, "Set topic register — 定題",                             2, ("topic",)),
        (FluxOpcode.USE_TOPIC,     OpcodeCategory.PARADIGM,   1, "Use topic as implicit operand — 用題",                    2, ("topic",)),
        (FluxOpcode.CLEAR_TOPIC,   OpcodeCategory.PARADIGM,   0, "Clear topic register — 清題",                           2, ("topic",)),
        # ── System ──
        (FluxOpcode.PRINT,         OpcodeCategory.SYSTEM,    1, "Print register value",                                 1, ()),
        (FluxOpcode.HALT,          OpcodeCategory.SYSTEM,    0, "Halt execution",                                    1, ()),
    }

    for op, cat, n_ops, desc, ver, feats in table:
        _OPCODE_INFO[op] = OpcodeInfo(
            name=op.name,
            hex_value=int(op),
            category=cat,
            category_id=cat.value,
            operands=n_ops,
            description=desc,
            first_version=ver,
            required_features=feats,
        )


_build_info_table()


# ══════════════════════════════════════════════════════════════════
# Runtime Identifiers
# ══════════════════════════════════════════════════════════════════

class RuntimeID(Enum):
    """Identifiers for each FLUX runtime implementation."""
    CANONICAL = "canonical"    # TypeScript (flux-multilingual)
    DEU = "deu"                # German
    SAN = "san"                # Sanskrit
    WEN = "wen"                # Classical Chinese
    LAT = "lat"                # Latin
    ZHO = "zho"                # Chinese (FIR layer)
    KOR = "kor"                # Korean (FIR layer)
    A2A = "a2a"                # A2A (BcOp string-based)


# ══════════════════════════════════════════════════════════════════
# Cross-Runtime Opcode Mapping Tables
# ══════════════════════════════════════════════════════════════════

# WEN runtime → canonical mapping
# WEN uses different hex values for most opcodes
_WEN_TO_CANONICAL: Dict[int, FluxOpcode] = {
    0x00: FluxOpcode.NOP,           0x01: FluxOpcode.MOV,
    0x02: FluxOpcode.LOAD,          0x03: FluxOpcode.STORE,
    0x04: FluxOpcode.JMP,           0x05: FluxOpcode.JZ,
    0x06: FluxOpcode.JNZ,           0x07: FluxOpcode.CALL,
    0x08: FluxOpcode.RET,           0x09: FluxOpcode.HALT,
    0x0A: FluxOpcode.PRINT,         0x10: FluxOpcode.IADD,
    0x11: FluxOpcode.ISUB,          0x12: FluxOpcode.IMUL,
    0x13: FluxOpcode.IDIV,          0x14: FluxOpcode.IMOD,
    0x15: FluxOpcode.INEG,          0x16: FluxOpcode.INC,
    0x17: FluxOpcode.DEC,           0x18: FluxOpcode.IEQ,
    0x19: FluxOpcode.IGT,           0x1A: FluxOpcode.ILT,
    0x1B: FluxOpcode.ICMP,          0x20: FluxOpcode.PUSH,
    0x21: FluxOpcode.POP,           0x22: FluxOpcode.MOVI,
    0x30: FluxOpcode.JE,            0x31: FluxOpcode.JNE,
    0x40: FluxOpcode.TELL,          0x41: FluxOpcode.ASK,
    0x42: FluxOpcode.DELEGATE,      0x43: FluxOpcode.BROADCAST,
    0x50: FluxOpcode.TRUST_CHECK,    0x51: FluxOpcode.CAP_REQUIRE,
}

# Reverse mapping: canonical → WEN
_CANONICAL_TO_WEN: Dict[int, int] = {
    int(op): wen_hex for wen_hex, op in _WEN_TO_CANONICAL.items()
}

# LAT runtime → canonical mapping
_LAT_TO_CANONICAL: Dict[int, FluxOpcode] = {
    0x00: FluxOpcode.NOP,           0x01: FluxOpcode.HALT,
    0x10: FluxOpcode.MOV,           0x11: FluxOpcode.MOVI,
    0x20: FluxOpcode.LOAD,          0x21: FluxOpcode.STORE,
    0x30: FluxOpcode.JMP,           0x31: FluxOpcode.JZ,
    0x32: FluxOpcode.JNZ,           0x33: FluxOpcode.JE,
    0x34: FluxOpcode.JNE,           0x40: FluxOpcode.CALL,
    0x41: FluxOpcode.RET,           0x50: FluxOpcode.IADD,
    0x51: FluxOpcode.ISUB,          0x52: FluxOpcode.IMUL,
    0x53: FluxOpcode.IDIV,          0x54: FluxOpcode.IMOD,
    0x55: FluxOpcode.INEG,          0x56: FluxOpcode.INC,
    0x57: FluxOpcode.DEC,           0x60: FluxOpcode.PUSH,
    0x61: FluxOpcode.POP,           0x70: FluxOpcode.CMP,
    0x80: FluxOpcode.PRINT,         0x81: FluxOpcode.TELL,
    0x82: FluxOpcode.ASK,           0x83: FluxOpcode.BROADCAST,
    0x90: FluxOpcode.DELEGATE,      0x91: FluxOpcode.TRUST_CHECK,
    0x92: FluxOpcode.CAP_REQUIRE,
}

# Reverse mapping: canonical → LAT
_CANONICAL_TO_LAT: Dict[int, int] = {
    int(op): lat_hex for lat_hex, op in _LAT_TO_CANONICAL.items()
}


def _build_runtime_tables() -> None:
    """Build per-runtime mapping tables (opcode name → hex) for all runtimes."""
    pass  # Tables built inline above


# ══════════════════════════════════════════════════════════════════
# Bytecode Version Constants
# ══════════════════════════════════════════════════════════════════

class BytecodeVersion:
    """Version constants for bytecode format negotiation."""

    V1_CORE = 1           # Core 87 opcodes (control, arithmetic, etc.)
    V2_FLOAT = 2            # + Float arithmetic and comparison
    V2_STRING = 2           # + String operations
    V3_A2A = 3              # + Existing A2A protocol (TELL, ASK, etc.)
    V4_A2A_EXTENDED = 4       # + New A2A agent coordination
    V5_PARADIGM_WEN = 5       # + Classical Chinese paradigm opcodes
    V5_PARADIGM_LAT = 6       # + Latin temporal paradigm opcodes
    V5_PARADIGM_TOPIC = 7     # + Topic register opcodes

    CURRENT = V5_PARADIGM_TOPIC

    # Feature flags (bits in the 16-bit flags field)
    F_HAS_FLOAT = 0x0001
    F_HAS_STRING = 0x0002
    F_HAS_A2A = 0x0004
    F_HAS_A2A_EXT = 0x0008
    F_HAS_WEN_PARADIGM = 0x0010
    F_HAS_LAT_PARADIGM = 0x0020
    F_HAS_TOPIC = 0x0040
    F_HAS_BITWISE = 0x0080
    F_HAS_MEMORY_REGIONS = 0x0100
    F_HAS_BOX_UNBOX = 0x0200


# ══════════════════════════════════════════════════════════════════
# Core Set: Opcodes present in ALL bytecode runtimes
# ════════════════════════════════════════════════════════════════

CORE_OPCODE_NAMES: FrozenSet[str] = frozenset({
    "NOP", "MOV", "LOAD", "STORE", "JMP", "JZ", "JNZ", "CALL",
    "RET", "HALT", "PRINT",
    "IADD", "ISUB", "IMUL", "IDIV", "IMOD", "INEG", "INC", "DEC",
    "CMP", "JE", "JNE", "MOVI", "PUSH", "POP",
    "TELL", "ASK", "DELEGATE", "BROADCAST", "TRUST_CHECK", "CAP_REQUIRE",
})


# ══════════════════════════════════════════════════════════════════
# Universal Opcode Registry
# ══════════════════════════════════════════════════════════════════

class FluxOpcodeRegistry:
    """Universal opcode registry for all FLUX runtimes.

    Single source of truth for:
      - Opcode definitions with metadata
      - Cross-runtime hex translation
      - Version and capability negotiation
      - Category-based opcode lookup

    Usage::

        registry = FluxOpcodeRegistry()

        # Lookup by name
        info = registry.get("IADD")

        # Lookup by canonical hex value
        info = registry.get_by_hex(0x08)

        # Translate WEN bytecode to canonical
        canonical_hex = registry.translate_byte(0x10, from_runtime=RuntimeID.WEN)

        # Version negotiation
        version, flags = registry.negotiate_version(3, {BytecodeVersion.F_HAS_FLOAT})
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, OpcodeInfo] = {}
        self._by_hex: Dict[int, OpcodeInfo] = {}
        self._by_category: Dict[OpcodeCategory, List[OpcodeInfo]] = {}

        # Register all opcodes
        for op, info in _OPCODE_INFO.items():
            self._by_name[info.name] = info
            self._by_hex[info.hex_value] = info
            self._by_category.setdefault(info.category, []).append(info)

    # ── Lookup Methods ────────────────────────────────────────────────────

    def get(self, name_or_hex: str | int) -> Optional[OpcodeInfo]:
        """Look up opcode by name (str) or canonical hex value (int)."""
        if isinstance(name_or_hex, str):
            return self._by_name.get(name_or_hex.upper())
        return self._by_hex.get(name_or_hex)

    def get_by_hex(self, hex_value: int) -> Optional[OpcodeInfo]:
        """Look up opcode by canonical hex value."""
        return self._by_hex.get(hex_value)

    def get_by_name(self, name: str) -> Optional[OpcodeInfo]:
        """Look up opcode by name (case-insensitive)."""
        return self._by_name.get(name.upper())

    def get_category(self, op: FluxOpcode) -> Optional[OpcodeCategory]:
        """Get the category for a known opcode."""
        info = self._by_hex.get(int(op))
        return info.category if info else None

    def list_opcodes(self) -> List[OpcodeInfo]:
        """List all opcodes sorted by hex value."""
        return sorted(self._by_hex.values(), key=lambda i: i.hex_value)

    def list_opcodes_by_category(self, category: OpcodeCategory) -> List[OpcodeInfo]:
        """List all opcodes in a given category."""
        return self._by_category.get(category, [])

    def core_opcodes(self) -> FrozenSet[str]:
        """Return the set of opcode names shared by all bytecode runtimes."""
        return CORE_OPCODE_NAMES

    def is_core(self, name: str) -> bool:
        """Check if an opcode is in the universal core set."""
        return name.upper() in CORE_OPCODE_NAMES

    @property
    def total(self) -> int:
        """Total number of registered opcodes."""
        return len(self._by_hex)

    # ── Cross-Runtime Translation ──────────────────────────────────────────

    def translate_byte(
        self,
        raw_byte: int,
        from_runtime: RuntimeID = RuntimeID.CANONICAL,
        to_runtime: RuntimeID = RuntimeID.CANONICAL,
    ) -> int:
        """Translate a single opcode byte between runtimes.

        Args:
            raw_byte: The opcode byte from the source runtime.
            from_runtime: Source runtime identifier.
            to_runtime: Target runtime identifier.

        Returns:
            The translated opcode byte for the target runtime, or the raw_byte
            if no translation is available.

        Raises:
            ValueError: If the runtime IDs are invalid.
        """
        if from_runtime == to_runtime:
            return raw_byte

        mapping = self._get_translation_table(from_runtime)
        return mapping.get(raw_byte, raw_byte)

    def translate_instruction(
        self,
        raw_byte: int,
        operands: List[int],
        from_runtime: RuntimeID = RuntimeID.CANONICAL,
        to_runtime: RuntimeID = RuntimeID.CANONICAL,
    ) -> Tuple[int, List[int]]:
        """Translate an instruction (opcode + operands) between runtimes.

        Returns:
            Tuple of (translated_opcode, translated_operands).
            Operands are passed through unchanged; only the opcode byte is translated.
        """
        translated = self.translate_byte(raw_byte, from_runtime, to_runtime)
        return translated, list(operands)

    def _get_translation_table(
        self, from_runtime: RuntimeID
    ) -> Dict[int, int]:
        """Get the translation table for a source runtime."""
        tables = {
            RuntimeID.WEN: _WEN_TO_CANONICAL,
            RuntimeID.LAT: _LAT_TO_CANONICAL,
            RuntimeID.DEU: {},  # Already canonical
            RuntimeID.SAN: {},  # Already canonical
            RuntimeID.CANONICAL: {},  # Identity
        }
        table = tables.get(from_runtime)
        if table is None:
            raise ValueError(f"No translation table for runtime: {from_runtime}")
        return table

    def detect_runtime(self, bytecode: bytes) -> Optional[RuntimeID]:
        """Heuristic runtime detection from bytecode content.

        Detection strategy:
        1. Check for 18-byte FLUX header (magic b'FLUX')
        2. Analyze opcode distribution and instruction patterns
        """
        if len(bytecode) >= 4 and bytecode[:4] == b'FLUX':
            return RuntimeID.CANONICAL

        # Heuristic: look for distinctive opcode patterns
        # WEN: RET at 0x08, HALT at 0x09, IADD at 0x10
        # LAT: HALT at 0x01, MOV at 0x10, IADD at 0x50
        if len(bytecode) >= 4:
            b0, b1, b2, b3 = bytecode[0], bytecode[1], bytecode[2], bytecode[3]
            # Check for WEN pattern: HALT at 0x09 (after RET at 0x08)
            if b0 in (0x08, 0x09, 0x0A) and b3 in (0x0A, 0x0B, 0x0C):
                return RuntimeID.WEN
            # Check for LAT pattern: HALT at 0x01
            if b0 == 0x01 and b3 in (0x80, 0x81, 0x82):
                return RuntimeID.LAT

        return None

    # ── Version Negotiation ────────────────────────────────────────────────

    def negotiate_version(
        self,
        bytecode_version: int,
        bytecode_features: Optional[Dict[int, bool]] = None,
        vm_version: Optional[int] = None,
        vm_features: Optional[Dict[int, bool]] = None,
    ) -> Tuple[int, int]:
        """Negotiate the effective bytecode version and feature flags.

        The effective version is min(bytecode_version, vm_version).
        Feature flags are AND'd together — only features supported by both
        sides are activated.

        Args:
            bytecode_version: Minimum version required by the bytecode.
            bytecode_features: Feature flags declared by the bytecode.
            vm_version: Maximum version supported by the VM.
            vm_features: Feature flags supported by the VM.

        Returns:
            Tuple of (effective_version, effective_feature_flags).
        """
        effective_version = bytecode_version
        if vm_version is not None:
            effective_version = min(bytecode_version, vm_version)

        effective_features = 0
        if bytecode_features and vm_features:
            for flag_bit in bytecode_features:
                if bytecode_features[flag_bit] and vm_features.get(flag_bit, False):
                    effective_features |= flag_bit
        elif bytecode_features:
            effective_features = bytecode_features
        elif vm_features:
            effective_features = vm_features

        return effective_version, effective_features

    def check_compatibility(
        self,
        bytecode_version: int,
        bytecode_features: Optional[Dict[int, bool]] = None,
        vm_version: int = BytecodeVersion.CURRENT,
        vm_features: Optional[Dict[int, bool]] = None,
    ) -> Tuple[bool, str]:
        """Check if bytecode is compatible with a VM.

        Returns:
            Tuple of (is_compatible, reason_message).
        """
        if bytecode_version > vm_version:
            return False, f"Bytecode requires version {bytecode_version}, VM supports {vm_version}"

        if bytecode_features and vm_features:
            missing = []
            for flag_bit, needed in bytecode_features.items():
                if needed and not vm_features.get(flag_bit, False):
                    missing.append(f"0x{flag_bit:04X}")
            if missing:
                return False, f"Missing features: {', '.join(missing)}"

        return True, "Compatible"

    # ── Information ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a human-readable summary of the registry."""
        lines = [
            f"FluxOpcodeRegistry — {self.total} opcodes registered",
            f"  Core (all runtimes): {len(CORE_OPCODE_NAMES)} opcodes",
        ]
        for cat in OpcodeCategory:
            ops = self.list_opcodes_by_category(cat)
            if ops:
                lines.append(f"  {cat.value}: {len(ops)} opcodes")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to a dictionary for JSON export."""
        return {
            "total": self.total,
            "core": sorted(CORE_OPCODE_NAMES),
            "opcodes": [
                {
                    "name": info.name,
                    "hex": f"0x{info.hex_value:02X}",
                    "category": info.category.value,
                    "operands": info.operands,
                    "version": info.first_version,
                    "features": list(info.required_features),
                    "description": info.description,
                }
                for info in self.list_opcodes()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FluxOpcodeRegistry:
        """Deserialize registry from a dictionary."""
        return cls()


# ══════════════════════════════════════════════════════════════════
# Convenience exports
# ══════════════════════════════════════════════════════════════════

def get_opcode(name: str) -> Optional[FluxOpcode]:
    """Look up a FluxOpcode by name (case-insensitive)."""
    info = FluxOpcodeRegistry().get(name)
    return FluxOpcode(info.hex_value) if info else None


def get_opcode_info(name: str) -> Optional[OpcodeInfo]:
    """Look up full OpcodeInfo by name."""
    return FluxOpcodeRegistry().get(name)
