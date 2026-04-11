# FLUX-A2A: Signal Protocol

> **"Language is the programming interface for agents."**
> — *Captains Log, FLUX Fleet Doctrine*

---

## Table of Contents

1. [Vision](#vision)
2. [Language Design Philosophy](#language-design-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [JSON as Universal AST](#json-as-universal-ast)
5. [Execution Models](#execution-models)
6. [A2A Opcodes](#a2a-opcodes)
7. [Branching Model](#branching-model)
8. [Forking Model](#forking-model)
9. [Co-iteration Model](#co-iteration-model)
10. [Multilingual Support](#multilingual-support)
11. [Confidence Propagation](#confidence-propagation)
12. [Trust Model](#trust-model)
13. [Schema Reference](#schema-reference)
14. [Example Programs](#example-programs)
15. [Runtime Architecture](#runtime-architecture)
16. [Roadmap & Integration](#roadmap--integration)

---

## Vision

FLUX-A2A — codename **Signal** — is the inter-agent communication layer for the FLUX multilingual ecosystem. It is not an API. It is not a protocol wrapper. It is a **first-class programming language** whose primitives are agent actions.

The FLUX ecosystem comprises six concept-first runtime rebuilds, each using a different human language grammar as programming constructs:

| Runtime | Human Language | Paradigm |
|---------|---------------|----------|
| `flux_zho` | 中文 (Chinese) | Imperative-ideographic: characters compose operations |
| `flux_deu` | Deutsch (German) | Compound-word composition: grammatical cases as types |
| `flux_kor` | 한국어 (Korean) | Agglutinative: morpheme stacking as program flow |
| `flux_san` | संस्कृतम् (Sanskrit) | Inflectional: declension/conjugation as control flow |
| `flux_wen` | 文言 (Classical Chinese) | Concise-semantic: minimal characters, maximal meaning |
| `flux_lat` | Latina (Latin) | Synthetic-case: word order flexible, case-driven logic |

Each compiles to a shared bytecode (`.fluxb`). Signal sits above all of them — it is the language agents use to **talk to each other**, **coordinate execution**, and **compose programs together**.

### Why "Signal"?

In naval tradition, signals are the primary means of inter-vessel coordination. A signal flag carries meaning regardless of which ship runs it up the mast. Signal is the JSON-based lingua franca that lets any FLUX agent — regardless of its native runtime language — communicate, collaborate, and co-execute.

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

## Roadmap & Integration

### Phase 1: Foundation (Current)
- [x] Schema definitions (dataclasses with JSON serialization)
- [x] Interpreter with all core opcodes
- [x] Branching and forking
- [x] Co-iteration engine
- [x] Example programs

### Phase 2: Runtime Integration
- [ ] Connect to FLUX bytecode VM
- [ ] Implement flux_zho/deu/kor/san/wen/lat compiler paths
- [ ] Real agent communication via message queues
- [ ] Persistent trust graph storage

### Phase 3: Fleet Coordination
- [ ] Multi-program orchestration
- [ ] Fleet-wide signal propagation
- [ ] Confidence-weighted consensus protocols
- [ ] Integration with captains-log nervous system

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
