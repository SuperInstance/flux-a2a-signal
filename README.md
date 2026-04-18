# FLUX-A2A: Signal Protocol

[![Ci](https://github.com/SuperInstance/flux-a2a-signal/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperInstance/flux-a2a-signal/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
> **"Language is the programming interface for agents."**
> — *Captains Log, FLUX Fleet Doctrine*

---

## Overview

**Signal** is the agent-to-agent (A2A) signaling protocol for the FLUX ecosystem. It enables autonomous AI agents to communicate, coordinate, and co-execute programs across language boundaries — using JSON as both the message format and the executable AST.

Traditional inter-service communication uses REST, gRPC, or message queues. Signal takes a fundamentally different approach: it treats agent communication as **first-class language primitives**. Operations like `tell`, `ask`, `delegate`, `branch`, `fork`, and `co_iterate` are not library calls — they are opcodes in the language itself, compiled to FLUX bytecode (`.fluxb`) and executed on the VM.

A single Signal program can mix expressions written in Chinese, German, Korean, Sanskrit, Classical Chinese, Latin, or direct bytecode — all composing seamlessly because every path converges on shared bytecode through the FLUX universal AST.

### What Is A2A Signaling?

Agent-to-agent signaling is the process by which autonomous agents exchange structured messages to coordinate behavior, delegate tasks, and reach consensus. Unlike request-response APIs, A2A signaling supports:

- **Persistent conversations** — agents maintain context across multiple exchanges
- **Confidence propagation** — every message carries an uncertainty score `[0.0, 1.0]`
- **Trust-weighted coordination** — agent trust scores influence merge priorities and delegation
- **Parallel exploration** — branching lets agents try multiple strategies simultaneously
- **Co-iteration** — multiple agents traverse and modify the same program collaboratively

Signal sits above all six FLUX language runtimes, providing a lingua franca for the fleet:

| Runtime | Human Language | Paradigm |
|---------|---------------|----------|
| `flux_zho` | 中文 (Chinese) | Imperative-ideographic: characters compose operations |
| `flux_deu` | Deutsch (German) | Compound-word composition: grammatical cases as types |
| `flux_kor` | 한국어 (Korean) | Agglutinative: morpheme stacking as program flow |
| `flux_san` | संस्कृतम् (Sanskrit) | Inflectional: declension/conjugation as control flow |
| `flux_wen` | 文言 (Classical Chinese) | Concise-semantic: minimal characters, maximal meaning |
| `flux_lat` | Latina (Latin) | Synthetic-case: word order flexible, case-driven logic |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SuperInstance/flux-a2a-signal.git
cd flux-a2a-signal

# No external dependencies required (Python 3.10+)
python -m pytest tests/    # Run the test suite
```

### Run a Signal Program

```python
from flux_a2a import interpret, evaluate, compile_program
from flux_a2a.schema import Program

# Define a multilingual signal program
program_data = {
    "signal": {
        "id": "hello-fleet",
        "version": "0.1.0",
        "agents": [
            {"id": "captain", "role": "coordinator", "capabilities": ["delegate", "branch"], "trust": 1.0},
            {"id": "navigator", "role": "executor", "capabilities": ["tell", "ask"], "trust": 0.85}
        ],
        "body": [
            {"op": "tell", "to": "navigator", "message": {"op": "struct", "fields": {"heading": 42, "speed": 12}}},
            {"op": "ask", "from": "navigator", "question": "current_position", "timeout_ms": 5000}
        ]
    }
}

program = Program.from_dict(program_data)
result = interpret(program)
print(result)  # ExecutionResult with value, confidence, and metadata
```

### Branching: Explore Multiple Strategies

```python
from flux_a2a.protocol import BranchPrimitive, BranchBody, MergeConfig

branch = BranchPrimitive(
    strategy="parallel",
    branches=[
        BranchBody(label="fast", weight=0.6, body=[{"op": "add", "args": [1, 2]}]),
        BranchBody(label="precise", weight=0.4, body=[{"op": "add", "args": [1, 1, 1]}]),
    ],
    merge=MergeConfig(strategy="weighted_confidence"),
)

# Serialize to JSON for inter-agent transmission
msg = branch.to_dict()   # {"op": "branch", "$schema": "flux.a2a.branch/v1", ...}
reconstituted = BranchPrimitive.from_dict(msg)
```

### Key Exports

```python
from flux_a2a import (
    # Interpret & compile
    interpret, evaluate, compile_program,
    # Schema
    Program, Expression, Agent, Result, Message, MessageBus,
    # Protocol primitives
    BranchPrimitive, ForkPrimitive, CoIteratePrimitive,
    DiscussPrimitive, SynthesizePrimitive, ReflectPrimitive,
    # Workflow pipeline
    AgentWorkflowPipeline, WorkflowSpec,
)
```

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Vision](#vision)
4. [Signal Types](#signal-types)
5. [Protocol](#protocol)
6. [Architecture Overview](#architecture-overview)
7. [Language Design Philosophy](#language-design-philosophy)
8. [JSON as Universal AST](#json-as-universal-ast)
9. [Execution Models](#execution-models)
10. [A2A Opcodes](#a2a-opcodes)
11. [Branching Model](#branching-model)
12. [Forking Model](#forking-model)
13. [Co-iteration Model](#co-iteration-model)
14. [Multilingual Support](#multilingual-support)
15. [Confidence Propagation](#confidence-propagation)
16. [Trust Model](#trust-model)
17. [Schema Reference](#schema-reference)
18. [Example Programs](#example-programs)
19. [Runtime Architecture](#runtime-architecture)
20. [Integration with FLUX VM and Fleet](#integration-with-flux-vm-and-fleet)
21. [Roadmap & Integration](#roadmap--integration)

---

## Vision

FLUX-A2A — codename **Signal** — is the inter-agent communication layer for the FLUX multilingual ecosystem. It is not an API. It is not a protocol wrapper. It is a **first-class programming language** whose primitives are agent actions.

Each of the six FLUX concept-first runtime rebuilds compiles to a shared bytecode (`.fluxb`). Signal sits above all of them — it is the language agents use to **talk to each other**, **coordinate execution**, and **compose programs together**.

### Why "Signal"?

In naval tradition, signals are the primary means of inter-vessel coordination. A signal flag carries meaning regardless of which ship runs it up the mast. Signal is the JSON-based lingua franca that lets any FLUX agent — regardless of its native runtime language — communicate, collaborate, and co-execute.

---

## Signal Types

Signal defines six first-class protocol primitives that form the backbone of agent-to-agent coordination. Each is a Python dataclass with full `to_dict()` / `from_dict()` JSON round-trip support and bytecode encoding.

### 1. Branch — Parallel Exploration

Split execution into parallel paths, each exploring a different strategy, then merge results using a configurable strategy.

| Strategy | Behavior |
|----------|----------|
| `parallel` | All paths execute concurrently |
| `sequential` | Paths execute one after another |
| `competitive` | Paths race; first to complete wins |

```
┌─── optimistic (w=0.6) ───→ result_a (conf=0.7) ──┐
parent ─┤                                             ├── merge → final
└─── conservative (w=0.4) ──→ result_b (conf=0.95) ─┘
```

Merge strategies: `consensus`, `vote`, `best`, `weighted_confidence`, `first_complete`, `last_writer_wins`, `custom`.

### 2. Fork — Agent Inheritance

Create a child agent with fine-grained control over inherited state (variables, context, trust graph, message history). The child executes independently and results merge back on completion.

```
parent agent
    ├── fork (inherit: state, context)
    │       ├── child agent created
    │       ├── body executed in child context
    │       ├── result collected ──→ merge back to parent
    │       └── child terminated
    └── parent continues (or awaits fork result)
```

### 3. Co-Iterate — Shared Program Traversal

Multiple agents simultaneously traverse and modify the same program — like pair-programming, except the "developers" are AI agents and the "code" is a Signal program. Each agent has a cursor tracking its position.

```
Agent A (writer):   ──→ [modify step 3] ──→ [advance to 4] ──→
Agent B (reviewer): ──→ [suggest at step 3] ──→ [approve step 4] ──→
                              │
                              ▼
                     Conflict at step 3
                     Resolution: priority → writer wins
                              │
                              ▼
                     Consensus: both advance to step 4
```

Convergence is detected via agreement metrics, confidence deltas, or value stability thresholds.

### 4. Discuss — Structured Agent Discourse

Multi-round structured discussions between agents with formats including `debate`, `brainstorm`, `review`, `negotiate`, and `peer_review`. Discussions run until consensus, timeout, round limit, or majority.

```json
{
  "op": "discuss",
  "topic": "optimal routing strategy",
  "format": "debate",
  "participants": [
    {"id": "navigator", "stance": "pro", "weight": 0.9},
    {"id": "engineer", "stance": "devil's_advocate", "weight": 0.8}
  ],
  "until": {"condition": "consensus", "max_rounds": 10},
  "output": {"format": "decision", "include_reasoning": true}
}
```

### 5. Synthesize — Result Combination

Combine results from branches, forks, discussions, or external sources using methods like `map_reduce`, `ensemble`, `chain`, `vote`, `weighted_merge`, and `best_effort`.

### 6. Reflect — Meta-Cognition

Agent self-assessment on strategy, progress, uncertainty, or confidence. Reflection can trigger adjustments, spawn branches, ask questions, or emit signals to other agents — enabling agents to reason about their own reasoning.

| Reflect Target | Use Case |
|----------------|----------|
| `strategy` | Evaluate and adjust current approach |
| `progress` | Check task completion status |
| `uncertainty` | Identify low-confidence areas |
| `confidence` | Audit confidence propagation |
| `all` | Full self-assessment |

---

## Protocol

### Message Format

Every inter-agent message in Signal follows the A2A message schema:

```json
{
  "id": "msg-uuid-v4",
  "from": "captain",
  "to": "navigator",
  "type": "tell",
  "payload": { "...": "..." },
  "confidence": 0.95,
  "timestamp": "2025-01-15T10:30:00Z",
  "in_reply_to": null,
  "ttl_ms": 30000
}
```

| Field | Description |
|-------|-------------|
| `type` | One of: `tell`, `ask`, `delegate`, `signal`, `response` |
| `confidence` | Sender's confidence in the payload `[0.0, 1.0]` |
| `ttl_ms` | Time-to-live; expired messages are discarded |
| `in_reply_to` | UUID of the message being replied to |

### Protocol Primitives — Schema Versioning

Each protocol primitive carries a `$schema` field following the AT Protocol Lexicon pattern, enabling forward-compatible evolution:

| Primitive | Schema Version |
|-----------|---------------|
| `branch` | `flux.a2a.branch/v1` |
| `fork` | `flux.a2a.fork/v1` |
| `co_iterate` | `flux.a2a.co_iterate/v1` |
| `discuss` | `flux.a2a.discuss/v1` |
| `synthesize` | `flux.a2a.synthesize/v1` |
| `reflect` | `flux.a2a.reflect/v1` |

Unknown fields are routed into the `meta` dict rather than causing parse errors — this is backward compatibility by design.

### Execution Modes

Signal programs can be executed in three modes, controlled by the runtime invocation:

| Mode | Description | Use Case |
|------|-------------|----------|
| `script` | Evaluated expression-by-expression (REPL) | Interactive sessions, debugging |
| `compile` | Translated to `.fluxb` bytecode, executed on VM | Production execution |
| `meta_compile` | Agent modifies its own compilation strategy | Self-optimizing agents |

### Opcode Bytecode Ranges

The FLUX opcode space is organized by category in contiguous hex ranges:

```
0x00–0x07  Control flow        (NOP, MOV, JMP, CALL, ...)
0x08–0x0F  Integer arithmetic  (IADD, ISUB, IMUL, IDIV, ...)
0x10–0x17  Bitwise             (IAND, IOR, IXOR, ISHL, ...)
0x18–0x1F  Comparison          (IEQ, ILT, ILE, IGT, IGE, ...)
0x20–0x27  Stack               (PUSH, POP, DUP, SWAP, ...)
0x28–0x2F  Function            (RET, TAILCALL, MOVI, ...)
0x30–0x37  Memory              (REGION_CREATE, MEMCOPY, ...)
0x38–0x3F  Type                (CAST, BOX, UNBOX, ...)
0x40–0x4F  Float               (FADD, FSUB, FMUL, FDIV, ...)
0x50–0x57  String              (SLEN, SCONCAT, SSUB, ...)
0x60–0x6F  A2A Protocol        (TELL, ASK, DELEGATE, BROADCAST, ...)
0x70–0x7F  A2A Extended        (BRANCH, MERGE, DISCUSS, CONFIDENCE, ...)
0x80–0xFD  Paradigm            (Language-specific: wen, lat, topic)
0xFE–0xFF  System              (PRINT, HALT)
```

---

## Language Design Philosophy

### 1. JSON Is Not Serialization — It's the AST

In traditional compilers, you write source code in a text language, it gets parsed into an AST, then compiled. In Signal, **the JSON IS the AST**. There is no parse step. A Signal program is a JSON document that is simultaneously:

- A human-readable (with pretty-printing) program
- A machine-executable instruction sequence
- An inter-agent message format
- A compilable unit targeting any FLUX runtime

### 2. Fluid in Language

Every expression in a Signal program can carry a `lang` tag. This means a single program can mix operations expressed in Chinese grammar, German compounds, Korean agglutination, Sanskrit inflection, Classical Chinese concision, Latin cases, or direct bytecode — and they all compose seamlessly.

This is not translation. The `lang` field tells the runtime **which compiler path** to use for that specific expression. The semantic meaning is preserved across language boundaries because all paths converge on shared bytecode.

### 3. Fluid in Execution Model

Signal doesn't force a single execution strategy. The same JSON program can be:

- **Scripted**: Evaluated expression-by-expression in a REPL (interpreted mode)
- **Compiled**: Translated to `.fluxb` bytecode and executed on the VM
- **Interpreted via NL**: Each expression's `lang` tag routes it through the corresponding language-specific natural-language-to-bytecode pipeline

The execution model is a property of the **runtime invocation**, not the program itself.

### 4. First-Class Agent Operations

Traditional languages have `if`, `while`, `function`. Signal has `tell`, `ask`, `delegate`, `broadcast` — and treats them as language primitives, not library functions. Branching, forking, and co-iteration are syntactic forms, not design patterns.

### 5. Confidence as a Native Type

Every expression result carries a confidence score `[0.0, 1.0]`. When an agent isn't sure, it says so. The runtime propagates confidence through the execution tree, and merge strategies can weight results by confidence. This is the "confidence propagation" from the captains-log doctrine made concrete.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FLUX-A2A Signal                         │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  Script   │  │ Compiled  │  │Interpreted│  │  Co-iter  │   │
│  │  (REPL)   │  │ (bytecode)│  │  (via NL) │  │ (multi-   │   │
│  │           │  │           │  │           │  │  agent)   │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │              │           │
│        └──────────────┴──────┬───────┴──────────────┘           │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │   Signal Core     │                        │
│                    │   (interpreter)   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│     ┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐           │
│     │BranchManager │ │ForkManager │ │CoIteration │           │
│     │              │ │            │ │  Engine    │           │
│     └──────────────┘ └────────────┘ └────────────┘           │
│                                                                 │
│                    ┌─────────┬─────────┐                       │
│              ┌─────▼────┐ ┌───▼───┐ ┌──▼───┐                  │
│              │ Compiler │ │ Trust │ │Conf-│                  │
│              │   (→BC)  │ │ Model │ │idnce│                  │
│              └─────┬────┘ └───────┘ └──────┘                  │
│                    │                                            │
│         ┌──────────┼──────────┬──────────┬──────────┐          │
│         │          │          │          │          │          │
│    ┌────▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐       │
│    │ zho    │ │ deu    │ │ kor    │ │ san    │ │ ...   │       │
│    │compiler│ │compiler│ │compiler│ │compiler│ │       │       │
│    └────┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘       │
│         │          │          │          │          │          │
│         └──────────┴──────────┴──────────┴──────────┘          │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │  FLUX Bytecode VM │                        │
│                    │    (.fluxb)       │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## JSON as Universal AST

### Program Structure

Every Signal program is a JSON object with this top-level structure:

```json
{
  "signal": {
    "id": "prog-uuid-v4",
    "version": "0.1.0",
    "meta": {
      "author": "agent:flux-captain",
      "created": "2025-01-15T10:30:00Z",
      "trust_level": 0.9
    },
    "agents": [
      {
        "id": "captain",
        "role": "coordinator",
        "capabilities": ["delegate", "branch", "merge"],
        "trust": 1.0
      },
      {
        "id": "navigator",
        "role": "executor",
        "capabilities": ["tell", "ask"],
        "trust": 0.85
      }
    ],
    "body": [
      { "op": "tell", ... },
      { "op": "branch", ... },
      { "op": "fork", ... }
    ]
  }
}
```

### Expression Forms

Every expression is a JSON object with at minimum an `op` field:

```json
{ "op": "<opcode>", ...params }
```

Expressions can be:

1. **Atomic** — single operation, single result
2. **Compound** — contains nested `body` arrays
3. **Tagged** — carries `lang`, `confidence`, `meta` annotations
4. **Addressed** — carries `from`/`to` agent identifiers

---

## Execution Models

### Script Mode (REPL)

Expressions are evaluated one at a time. No compilation step. Ideal for interactive agent sessions.

```json
{ "op": "eval", "expr": { "op": "add", "args": [2, 3] } }
```

The interpreter walks the JSON structure directly. State is maintained in memory. Results are returned immediately.

### Compiled Mode

The JSON program is compiled to FLUX bytecode (`.fluxb`), which is then executed on the VM. This is the production execution path.

```json
{
  "op": "compile",
  "target": "fluxb",
  "optimizations": ["dead_branch_elim", "cse"],
  "body": [...]
}
```

Compilation preserves all language tags — expressions tagged with `lang: "zho"` go through the `flux_zho` compiler path, `lang: "deu"` through `flux_deu`, etc. Untagged expressions compile through the direct bytecode path.

### Interpreted via Natural Language

Each expression is first converted to its natural-language representation using the tagged language's grammar, then compiled through that runtime's NL→bytecode pipeline. This allows agents to "think" in their preferred language while producing correct bytecode.

```json
{
  "op": "interpret",
  "expr": { "op": "loop", "lang": "kor", "times": 5, "body": [...] },
  "via": "flux_kor"
}
```

---

## A2A Opcodes

### Core Communication

| Opcode | Arity | Description |
|--------|-------|-------------|
| `tell` | binary | Send information to a specific agent |
| `ask` | binary | Request information from a specific agent |
| `delegate` | binary | Assign a task to an agent, await result |
| `broadcast` | unary | Send to all agents in current scope |

### Computation

| Opcode | Arity | Description |
|--------|-------|-------------|
| `add`, `sub`, `mul`, `div`, `mod` | n-ary | Arithmetic |
| `eq`, `neq`, `lt`, `lte`, `gt`, `gte` | binary | Comparison |
| `and`, `or`, `not`, `xor` | n-ary | Logic |
| `concat` | binary | String concatenation |
| `length` | unary | Collection/string length |
| `at` | binary | Index into collection |

### Control Flow

| Opcode | Arity | Description |
|--------|-------|-------------|
| `seq` | n-ary | Sequential execution |
| `if` | ternary | Conditional |
| `loop` | binary | Iteration (count or collection) |
| `while` | binary | Condition + body |
| `match` | n-ary | Pattern matching |
| `yield` | unary | Suspend and return value |
| `await` | unary | Wait for signal/result |

### Agent Operations

| Opcode | Arity | Description |
|--------|-------|-------------|
| `branch` | n-ary | Split execution into parallel paths |
| `fork` | unary | Spawn child agent with inherited state |
| `merge` | n-ary | Join branches with merge strategy |
| `co_iterate` | n-ary | Shared program traversal |
| `signal` | binary | Emit named signal |
| `trust` | binary | Set trust level for agent |
| `confidence` | unary | Set confidence for current scope |

### Data Operations

| Opcode | Arity | Description |
|--------|-------|-------------|
| `let` | binary | Bind name to value |
| `get` | unary | Retrieve named binding |
| `set` | binary | Update named binding |
| `struct` | n-ary | Create structured data |
| `collect` | n-ary | Gather results into collection |
| `reduce` | ternary | Fold over collection |

### Opcode Grammar

```json
// tell — agent communication
{
  "op": "tell",
  "to": "navigator",
  "message": {
    "op": "struct",
    "fields": {
      "heading": 42,
      "speed": 12,
      "confidence": 0.95
    }
  }
}

// ask — request with timeout
{
  "op": "ask",
  "from": "navigator",
  "question": "current_position",
  "timeout_ms": 5000
}

// delegate — task assignment
{
  "op": "delegate",
  "to": "navigator",
  "task": {
    "op": "loop",
    "times": 10,
    "body": [
      { "op": "tell", "to": "captain", "message": { "op": "get", "name": "position" } }
    ]
  },
  "on_complete": "merge"
}

// broadcast — fleet-wide announcement
{
  "op": "broadcast",
  "scope": "fleet",
  "message": {
    "op": "struct",
    "fields": {
      "type": "course_change",
      "new_heading": 270
    }
  }
}
```

---

## Branching Model

Branching is a first-class language construct. An agent can split execution into multiple parallel paths, each exploring a different strategy, and then merge results.

### Branch Syntax

```json
{
  "op": "branch",
  "id": "branch-explore-01",
  "branches": [
    {
      "label": "optimistic",
      "weight": 0.6,
      "body": [
        { "op": "compute", "strategy": "fast_approx", "input": { "op": "get", "name": "data" } }
      ]
    },
    {
      "label": "conservative",
      "weight": 0.4,
      "body": [
        { "op": "compute", "strategy": "precise", "input": { "op": "get", "name": "data" } }
      ]
    }
  ],
  "merge": {
    "strategy": "weighted_confidence",
    "timeout_ms": 10000,
    "fallback": "first_complete"
  }
}
```

### Branch Execution

1. **Spawn**: Each branch gets its own execution context (fork of current state)
2. **Execute**: Branches execute in parallel (or simulated parallelism in single-threaded mode)
3. **Collect**: Results are gathered as they complete
4. **Merge**: The merge strategy determines the final result:
   - `last_writer_wins`: most recent write to shared state
   - `consensus`: all branches must agree (majority vote if n > 2)
   - `weighted_confidence`: weighted average by branch weight × result confidence
   - `first_complete`: take whichever finishes first
   - `best_confidence`: take result with highest confidence score
   - `vote`: democratic majority vote
   - `custom`: user-defined merge function

### Branch Visualization

```
        ┌─── optimistic (w=0.6) ───→ result_a (conf=0.7)
parent ─┤
        └─── conservative (w=0.4) ──→ result_b (conf=0.95)
                        │
                        ▼
              merge: weighted_confidence
              → final = (0.6×0.7 + 0.4×0.95) / (0.6+0.4) = 0.80
```

---

## Forking Model

Forking is distinct from branching. A branch is a parallel execution path within the same agent. A **fork** creates a new agent (or activates an existing one) with inherited state.

### Fork Syntax

```json
{
  "op": "fork",
  "id": "fork-scout-01",
  "agent": {
    "id": "scout",
    "role": "explorer",
    "capabilities": ["tell", "ask", "signal"],
    "trust": 0.75
  },
  "inherit": {
    "state": ["mission_params", "map_data"],
    "context": true,
    "trust_graph": false,
    "message_queue": false
  },
  "body": [
    { "op": "let", "name": "exploration_range", "value": 100 },
    {
      "op": "loop",
      "times": 5,
      "body": [
        { "op": "tell", "to": "captain", "message": { "op": "get", "name": "scan_result" } },
        { "op": "signal", "name": "scanning", "payload": { "op": "get", "name": "position" } }
      ]
    }
  ],
  "on_result": "collect",
  "on_complete": {
    "op": "merge",
    "strategy": "best_confidence"
  }
}
```

### Fork Lifecycle

```
parent agent
    │
    ├── fork (inherit: state, context)
    │       │
    │       ├── child agent created
    │       ├── state copied (shallow for state, deep for context)
    │       ├── body executed in child context
    │       ├── child signals/tells parent periodically
    │       │
    │       ├── result collected ──→ merge back to parent
    │       └── child terminated
    │
    └── parent continues (or awaits fork result)
```

### State Inheritance

| Field | Shallow Copy | Deep Copy | Description |
|-------|-------------|-----------|-------------|
| `state` | ✅ | ✅ | Named variable bindings |
| `context` | — | ✅ | Execution context (agent identity, scope) |
| `trust_graph` | — | ✅ | Trust relationships |
| `message_queue` | ❌ | ❌ | Each agent has its own message queue |

---

## Co-iteration Model

Co-iteration is the most distinctive feature of Signal. It models how **multiple agents can simultaneously traverse and modify the same program** — like two developers pair-programming, except the "developers" are AI agents and the "code" is a Signal program.

### Co-iteration Syntax

```json
{
  "op": "co_iterate",
  "id": "co-iter-collab-01",
  "program": {
    "op": "seq",
    "body": [
      { "op": "let", "name": "shared_counter", "value": 0 },
      { "op": "comment", "text": "Agents will co-iterate from here" },
      { "op": "loop", "id": "shared-loop", "times": 10, "body": [...] },
      { "op": "get", "name": "shared_counter" }
    ]
  },
  "agents": [
    { "id": "writer", "cursor": 0, "role": "modifier", "permissions": ["write", "branch"] },
    { "id": "reviewer", "cursor": 0, "role": "auditor", "permissions": ["read", "suggest"] }
  ],
  "conflict_resolution": {
    "strategy": "priority",
    "priority_order": ["writer", "reviewer"],
    "on_conflict": "notify"
  },
  "merge_strategy": {
    "approach": "sequential_consensus",
    "require_agreement": true
  }
}
```

### AgentCursor

Each agent in a co-iteration has a cursor — a position within the shared program. The cursor tracks:

```json
{
  "agent_id": "writer",
  "position": 3,
  "step_count": 7,
  "local_modifications": 2,
  "suggestions_pending": 1,
  "confidence_at_position": 0.9
}
```

### Conflict Resolution Strategies

When two agents' cursors converge on the same program location:

| Strategy | Behavior |
|----------|----------|
| `priority` | Higher-priority agent's change wins |
| `merge` | Attempt to merge both changes |
| `vote` | All participating agents vote |
| `last_writer` | Most recent write wins |
| `reject` | Block conflicting writes, require explicit resolution |
| `branch` | Automatically branch, let agents work in parallel, merge later |

### Consensus Model

```
Agent A (writer):   ──→ [modify step 3] ──→ [advance to 4] ──→
Agent B (reviewer): ──→ [suggest at step 3] ──→ [approve step 4] ──→
                              │
                              ▼
                     Conflict at step 3
                     Resolution: priority → writer wins
                     Reviewer suggestion logged
                              │
                              ▼
                     Consensus: both advance to step 4
```

---

## Multilingual Support

Every expression can be tagged with a language identifier. This tag determines:

1. **Compilation path**: which FLUX runtime compiler processes this expression
2. **Formatting**: how the expression is displayed to agents who prefer that language
3. **Validation**: language-specific semantic checks

### Language Tags

| Tag | Runtime | Example Expression |
|-----|---------|-------------------|
| `"flux"` | Direct bytecode | `{ "op": "add", "args": [1, 2] }` |
| `"zho"` | flux_zho | `{ "op": "加", "甲": 1, "乙": 2, "lang": "zho" }` |
| `"deu"` | flux_deu | `{ "op": "addieren", "erste": 1, "zweite": 2, "lang": "deu" }` |
| `"kor"` | flux_kor | `{ "op": "더하기", "첫째": 1, "둘째": 2, "lang": "kor" }` |
| `"san"` | flux_san | `{ "op": "योग", "प्रथमः": 1, "द्वितीयः": 2, "lang": "san" }` |
| `"wen"` | flux_wen | `{ "op": "加", "甲": 1, "乙": 2, "lang": "wen" }` |
| `"lat"` | flux_lat | `{ "op": "addere", "primum": 1, "secundum": 2, "lang": "lat" }` |

### Multilingual Program Example

```json
{
  "signal": {
    "id": "multilingual-demo",
    "body": [
      {
        "op": "let", "name": "x",
        "value": { "op": "加", "甲": 10, "乙": 20, "lang": "zho" },
        "lang": "zho"
      },
      {
        "op": "let", "name": "y",
        "value": { "op": "addieren", "erste": 5, "zweite": 3, "lang": "deu" },
        "lang": "deu"
      },
      {
        "op": "let", "name": "result",
        "value": { "op": "mul", "args": [{ "op": "get", "name": "x" }, { "op": "get", "name": "y" }] }
      },
      {
        "op": "tell", "to": "fleet",
        "message": { "op": "get", "name": "result" }
      }
    ]
  }
}
```

The `x` binding is computed via the Chinese runtime, `y` via the German runtime, and `result` via direct bytecode. All three coexist in the same program, sharing state seamlessly.

---

## Confidence Propagation

Every evaluation result carries a confidence score. This is not optional — it is a native part of the type system.

### Confidence Rules

1. **Literal values**: confidence = 1.0 (certain)
2. **Arithmetic on certain inputs**: confidence = 1.0
3. **Arithmetic on uncertain inputs**: confidence = min(input confidences)
4. **Agent operations**: confidence set by the agent (0.0 to 1.0)
5. **Branch merge**: confidence depends on merge strategy
6. **Co-iteration**: confidence = geometric mean of agreeing agents

### Confidence Propagation Example

```json
{
  "op": "mul",
  "args": [
    { "op": "get", "name": "sensor_reading", "confidence": 0.8 },
    { "op": "get", "name": "calibration_factor", "confidence": 0.95 }
  ]
}
// Result confidence = min(0.8, 0.95) = 0.8
```

### Result Type

```json
{
  "value": 42,
  "confidence": 0.8,
  "source": "mul",
  "timestamp": "2025-01-15T10:30:00Z",
  "agent": "sensor_agent"
}
```

---

## Trust Model

Trust is a directed, weighted graph between agents. Trust scores influence:

1. **Merge priority**: higher-trusted agents' results are weighted more heavily
2. **Delegation willingness**: agents prefer delegating to higher-trusted agents
3. **Co-iteration conflict resolution**: trust determines priority order

### Trust Syntax

```json
{
  "op": "trust",
  "agent": "navigator",
  "level": 0.9,
  "basis": "historical_accuracy",
  "decay_rate": 0.01
}
```

### Trust Graph Example

```json
{
  "trust_graph": {
    "captain": {
      "navigator": { "level": 0.9, "basis": "proven" },
      "engineer": { "level": 0.85, "basis": "proven" },
      "scout": { "level": 0.6, "basis": "unproven" }
    },
    "navigator": {
      "captain": { "level": 1.0, "basis": "authority" },
      "engineer": { "level": 0.8, "basis": "collaborative" }
    }
  }
}
```

---

## Schema Reference

### Program

```typescript
interface SignalProgram {
  signal: {
    id: string;              // UUID v4
    version: string;         // semver
    meta: ProgramMeta;
    agents: Agent[];
    trust_graph?: TrustGraph;
    body: Expression[];
  };
}
```

### Expression

```typescript
interface Expression {
  op: string;               // opcode
  lang?: LanguageTag;       // language for compilation path
  confidence?: number;      // 0.0 to 1.0
  meta?: Record<string, any>;
  // opcode-specific fields...
}
```

### Agent

```typescript
interface Agent {
  id: string;
  role: string;
  capabilities: string[];
  trust: number;            // 0.0 to 1.0
  lang?: LanguageTag;       // preferred language
  endpoint?: string;        // communication endpoint
}
```

### Result

```typescript
interface ExecutionResult {
  value: any;
  confidence: number;
  source: string;
  timestamp: string;
  agent: string;
  branch?: string;
  children?: ExecutionResult[];
}
```

### Message

```typescript
interface A2AMessage {
  id: string;
  from: string;
  to: string | "broadcast" | "fleet";
  type: "tell" | "ask" | "delegate" | "signal" | "response";
  payload: any;
  confidence: number;
  timestamp: string;
  in_reply_to?: string;
  ttl_ms?: number;
}
```

---

## Example Programs

### Hello World (Multilingual)

```json
{
  "signal": {
    "id": "hello-world",
    "version": "0.1.0",
    "body": [
      {
        "op": "tell",
        "to": "world",
        "message": "你好世界",
        "lang": "zho",
        "confidence": 1.0
      },
      {
        "op": "tell",
        "to": "world",
        "message": "Hallo Welt",
        "lang": "deu",
        "confidence": 1.0
      },
      {
        "op": "tell",
        "to": "world",
        "message": "안녕하세요 세계",
        "lang": "kor",
        "confidence": 1.0
      }
    ]
  }
}
```

### Branching: Parallel Path Exploration

```json
{
  "signal": {
    "id": "branch-demo",
    "body": [
      { "op": "let", "name": "data", "value": [1, 2, 3, 4, 5] },
      {
        "op": "branch",
        "id": "sort-strategies",
        "branches": [
          {
            "label": "bubble",
            "weight": 0.3,
            "body": [{ "op": "sort", "algorithm": "bubble", "input": { "op": "get", "name": "data" } }]
          },
          {
            "label": "quick",
            "weight": 0.5,
            "body": [{ "op": "sort", "algorithm": "quick", "input": { "op": "get", "name": "data" } }]
          },
          {
            "label": "merge_sort",
            "weight": 0.2,
            "body": [{ "op": "sort", "algorithm": "merge", "input": { "op": "get", "name": "data" } }]
          }
        ],
        "merge": { "strategy": "best_confidence", "timeout_ms": 5000 }
      }
    ]
  }
}
```

### Forking: Agent Delegation

```json
{
  "signal": {
    "id": "fork-demo",
    "agents": [
      { "id": "captain", "role": "coordinator", "capabilities": ["fork", "merge"], "trust": 1.0 }
    ],
    "body": [
      {
        "op": "fork",
        "id": "scout-1",
        "agent": { "id": "scout", "role": "explorer", "capabilities": ["tell"], "trust": 0.7 },
        "inherit": { "state": ["mission"], "context": true },
        "body": [
          { "op": "tell", "to": "captain", "message": "Sector A clear" }
        ]
      },
      {
        "op": "fork",
        "id": "scout-2",
        "agent": { "id": "scout-b", "role": "explorer", "capabilities": ["tell"], "trust": 0.7 },
        "inherit": { "state": ["mission"], "context": true },
        "body": [
          { "op": "tell", "to": "captain", "message": "Sector B has obstacles" }
        ]
      },
      {
        "op": "await",
        "signal": "forks_complete",
        "timeout_ms": 30000
      }
    ]
  }
}
```

---

## Runtime Architecture

### Interpreter Pipeline

```
JSON Program
    │
    ▼
┌──────────────┐
│   Parser     │  (validates JSON structure, resolves lang tags)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Evaluator   │  (walks AST, dispatches opcodes)
└──────┬───────┘
       │
       ├──→ BranchManager  (parallel paths)
       ├──→ ForkManager    (child agents)
       ├──→ CoIterEngine   (shared traversal)
       │
       ▼
┌──────────────┐
│ Result       │  (value + confidence + metadata)
└──────────────┘
```

### Compiler Pipeline

```
JSON Program
    │
    ▼
┌──────────────┐
│   Analyzer   │  (static analysis, dead code detection)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Compiler   │  (lang-tag routing)
└──────┬───────┘
       │
       ├──→ flux_zho compiler
       ├──→ flux_deu compiler
       ├──→ flux_kor compiler
       ├──→ flux_san compiler
       ├──→ flux_wen compiler
       ├──→ flux_lat compiler
       └──→ direct bytecode emitter
       │
       ▼
┌──────────────┐
│   Optimizer  │  (dead branch elim, CSE, constant folding)
└──────┬───────┘
       │
       ▼
  FLUX Bytecode (.fluxb)
```

---

## Integration with FLUX VM and Fleet

### Fleet-Level Signal Flow

Signal operates as the communication backbone across the FLUX fleet, connecting agents running on different language runtimes through a shared bytecode layer:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FLUX FLEET                                   │
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  Agent A     │     │  Agent B     │     │  Agent C     │           │
│  │  (flux_zho)  │     │  (flux_deu)  │     │  (flux_kor)  │           │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘           │
│         │                    │                    │                   │
│         ▼                    ▼                    ▼                   │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │                   SIGNAL PROTOCOL                        │        │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐ │        │
│  │  │ Message  │ │ Branch  │ │ Discuss  │ │ Co-iterate   │ │        │
│  │  │ Bus      │ │ Manager │ │ Protocol │ │ Engine       │ │        │
│  │  └────┬────┘ └────┬────┘ └────┬─────┘ └──────┬───────┘ │        │
│  └───────┼───────────┼──────────┼───────────────┼──────────┘        │
│          │           │          │               │                    │
│          └───────────┴────┬─────┴───────────────┘                    │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │                  FLUX BYTECODE VM                        │        │
│  │              Universal .fluxb execution                   │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │              TRUST GRAPH + CONFIDENCE                    │        │
│  │         Fleet-wide trust network, score propagation      │        │
│  └──────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Architecture

The `flux_a2a` package is organized into composable modules:

| Module | Description |
|--------|-------------|
| `schema.py` | Core dataclasses: Program, Expression, Agent, Result, Message |
| `interpreter.py` | Tree-walking interpreter for script mode |
| `compiler.py` | JSON → `.fluxb` bytecode compiler |
| `protocol.py` | Six protocol primitives (Branch, Fork, CoIterate, Discuss, Synthesize, Reflect) |
| `opcodes.py` | Universal opcode registry with cross-runtime translation |
| `co_iteration.py` | Multi-agent shared program traversal engine |
| `fork_manager.py` | Agent inheritance, branching, and merge management |
| `discussion.py` | Structured agent discourse (debate, brainstorm, review, negotiate) |
| `consensus.py` | Consensus detection and convergence tracking |
| `pipeline.py` | Multi-step agent workflow orchestration |
| `paradigm_lattice.py` | Paradigm similarity space for cross-language bridging |
| `paradigm_flow.py` | Bridge cost simulation between language paradigms |
| `cross_compiler.py` | Cross-runtime bytecode translation |
| `type_safe_bridge.py` | Type-safe inter-language bridging |
| `evolution.py` | Hot-path detection, pattern compilation, self-optimization |
| `partial_eval.py` | Partial evaluation and interpreter specialization |
| `types.py` | FUTS universal type system (FluxType, FluxTypeRegistry) |
| `type_checker.py` | Universal type checker with compatibility reports |
| `optimizer.py` | Bytecode optimization passes |
| `ambiguous.py` | Ambiguity resolution and confidence propagation |
| `causality.py` | Causal ordering of events |
| `temporal.py` | Temporal reasoning and event sequencing |
| `semantics.py` | Semantic analysis |
| `format_bridge.py` | Data format conversion between runtimes |
| `unified_vocabulary.py` | Shared vocabulary across language runtimes |
| `ast_unifier.py` | AST unification across language boundaries |

### Fleet Integration Points

Signal integrates with the FLUX fleet through several mechanisms:

1. **Message-in-a-Bottle Protocol** — Fleet agents communicate via the `message-in-a-bottle/` directory structure. Signal messages can be deposited as bottles for asynchronous fleet coordination.

2. **Git-Agent Standard v2.0** — Signal is compliant with the fleet's agent standard, enabling any fleet vessel to parse and execute Signal programs.

3. **Trust Graph Propagation** — Trust scores established between agents in Signal programs are shared fleet-wide through the trust graph, allowing agents to make informed delegation decisions.

4. **Bytecode Interchange** — The `FluxOpcodeRegistry` handles cross-runtime opcode translation, so bytecode compiled on any FLUX runtime can be executed on any other. Runtime-specific hex values are transparently mapped to canonical opcodes.

5. **Dockside Exam Ready** — Signal passes the fleet's certification checklist with CHARTER.md, README.md, LICENSE, tests, and documentation.

---

## Roadmap & Integration

### Phase 1: Foundation (Current)
- [x] Schema definitions (dataclasses with JSON serialization)
- [x] Interpreter with all core opcodes
- [x] Branching and forking
- [x] Co-iteration engine
- [x] Protocol primitives (Branch, Fork, CoIterate, Discuss, Synthesize, Reflect)
- [x] Paradigm lattice and bridge cost simulation
- [x] Type-safe bridges between all language pairs
- [x] Evolution engine with hot-path detection
- [x] Partial evaluation and interpreter specialization
- [x] Universal type system (FUTS)
- [x] Consensus detection and convergence tracking
- [x] Agent workflow pipeline
- [x] Cross-runtime opcode registry with translation tables
- [x] Example programs and comprehensive test suite

### Phase 2: Runtime Integration
- [ ] Connect to FLUX bytecode VM
- [ ] Implement flux_zho/deu/kor/san/wen/lat compiler paths
- [ ] Real agent communication via message queues
- [ ] Persistent trust graph storage
- [ ] Fleet-wide signal propagation

### Phase 3: Fleet Coordination
- [ ] Multi-program orchestration
- [ ] Fleet-wide confidence-weighted consensus protocols
- [ ] Integration with captains-log nervous system
- [ ] Tender protocol support for edge/offline agents
- [ ] SmartCRDT-backed multi-agent collaboration

### Phase 4: Lucineer Convergence
- [ ] Unified intent-language foundation
- [ ] Cross-fleet co-iteration
- [ ] Shared trust graph between fleets
- [ ] Universal bytecode interchange format

---

## License

MIT — Free as the signal that crosses open waters.

---

*"Every fleet needs a common tongue. Ours is Signal."*

---

<img src="callsign1.jpg" width="128" alt="callsign">
