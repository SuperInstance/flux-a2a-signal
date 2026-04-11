"""
FLUX-A2A Signal Protocol — Agent-to-Agent First-Class JSON Language.

Signal is the inter-agent communication layer for the FLUX multilingual
ecosystem. JSON is the universal AST, not just a serialization format.
Every expression, branch, fork, and co-iteration is a JSON primitive.

Usage:
    from flux_a2a import interpret, evaluate, compile_program
    from flux_a2a.schema import Program, Expression, Agent, Result

    program = Program.from_dict({...})
    result = interpret(program)
"""

__version__ = "0.1.0"
__all__ = [
    # Schema
    "LanguageTag",
    "ProgramMeta",
    "Agent",
    "TrustEntry",
    "Expression",
    "BranchDef",
    "ForkDef",
    "CoIterateDef",
    "MergePolicy",
    "ConflictResolution",
    "Program",
    "Message",
    "MessageBus",
    "Result",
    "ConfidenceScore",
    # Interpreter
    "Interpreter",
    "interpret",
    "evaluate",
    # Compiler
    "Compiler",
    "compile_program",
    "BytecodeChunk",
    # Co-iteration
    "SharedProgram",
    "AgentCursor",
    "CoIterationEngine",
    "ConflictResolutionStrategy",
    "MergeStrategy",
    # Forking
    "BranchPoint",
    "ForkContext",
    "BranchManager",
    "ForkTree",
    "ForkManager",
    "MergePolicyType",
    # Ambiguous parsing
    "Interpretation",
    "AmbiguousParse",
    "AmbiguityStatus",
    "ConfidencePropagation",
    "EvidenceRecord",
    "ExecutionResult",
    "ExecutionBackend",
    "SimpleBackend",
    "BranchingExecutor",
    "BranchingResult",
    "resolve_ambiguity",
    # Protocol primitives (R1-R3)
    "BranchPrimitive",
    "ForkPrimitive",
    "CoIteratePrimitive",
    "DiscussPrimitive",
    "SynthesizePrimitive",
    "ReflectPrimitive",
    "ProtocolRegistry",
    "ExecutionModeConfig",
    "ModeTransition",
    "NEW_OPCODES",
    "ALL_OPCODES",
    # Paradigm lattice (R4-R6)
    "ParadigmLattice",
    "ParadigmPoint",
    "ParadigmFlow",
    "BridgeCost",
    "BridgeSimulation",
    "DIMENSION_NAMES",
    "DIMENSION_WEIGHTS",
    "generate_simulation_report",
    # Discussion protocol (R4)
    "DiscussionProtocol",
    "DiscussionConfig",
    "DiscussionTurn",
    "DiscussionResult",
    "DiscussionFormat",
    "DiscussionOutcome",
    "Phase",
    "DebateStrategy",
    "BrainstormStrategy",
    "ReviewStrategy",
    "NegotiationStrategy",
    "PeerReviewStrategy",
    "ReviewCriterion",
    "DEFAULT_REVIEW_CHECKLIST",
    # Consensus detection (R5)
    "ConsensusDetector",
    "ConsensusType",
    "ConvergenceTrend",
    "ConvergenceHistory",
    "AgreementMetrics",
    "AgentPosition",
    "Stalemate",
    "ResolutionType",
    "ResolutionStrategy",
    "SimilarityMetric",
    # Pipeline (R6)
    "AgentWorkflowPipeline",
    "WorkflowSpec",
    "WorkflowResult",
    "AgentSpec",
    "BranchResult",
    "WorkflowStatus",
    "SynthesisApproach",
    "BranchingType",
    "OutputType",
    # Evolution engine (R10-12)
    "EvolutionEngine",
    "EvolutionLevel",
    "OptimizationKind",
    "NLObservation",
    "HotPath",
    "NLPattern",
    "CompiledPattern",
    "Optimization",
    "GrammarDelta",
    "FitnessMetrics",
    # Partial evaluator (R10-12)
    "PartialEvaluator",
    "PEResult",
    "StaticKnowledge",
    "PELevel",
    "ReductionResult",
    "partial_evaluate",
    "specialize_interpreter",
    "build_knowledge",
    # FUTS — Universal Type System (R10-12)
    "FluxBaseType",
    "FluxConstraint",
    "ConstraintKind",
    "FluxType",
    "FluxTypeSignature",
    "FluxTypeRegistry",
    "QuantumTypeState",
    "TypeCompatibility",
    "CompatibilityLevel",
    "CompatibilityReport",
    "UniversalTypeChecker",
    "TypeCheckResult",
    "TypeBridge",
    "BridgeResult",
    "BridgeStrategy",
    "build_default_registry",
]

from flux_a2a.schema import (
    Agent,
    BranchDef,
    CoIterateDef,
    ConflictResolution,
    ConfidenceScore,
    Expression,
    ForkDef,
    LanguageTag,
    MergePolicy,
    Message,
    MessageBus,
    Program,
    ProgramMeta,
    Result,
    TrustEntry,
)
from flux_a2a.interpreter import Interpreter, evaluate, interpret
from flux_a2a.compiler import BytecodeChunk, Compiler, compile_program
from flux_a2a.co_iteration import (
    AgentCursor,
    CoIterationEngine,
    ConflictResolutionStrategy,
    MergeStrategy,
    SharedProgram,
)
from flux_a2a.fork_manager import (
    BranchManager,
    BranchPoint,
    ForkContext,
    ForkManager,
    ForkTree,
    MergePolicyType,
)
from flux_a2a.ambiguous import (
    AmbiguityStatus,
    AmbiguousParse,
    BranchingExecutor,
    BranchingResult,
    ConfidencePropagation,
    EvidenceRecord,
    ExecutionBackend,
    ExecutionResult,
    Interpretation,
    SimpleBackend,
    resolve_ambiguity,
)
from flux_a2a.protocol import (
    BranchPrimitive,
    ForkPrimitive,
    CoIteratePrimitive,
    DiscussPrimitive,
    SynthesizePrimitive,
    ReflectPrimitive,
    ProtocolRegistry,
    ExecutionModeConfig,
    ModeTransition,
    NEW_OPCODES,
    ALL_OPCODES,
)
from flux_a2a.paradigm_lattice import (
    DIMENSION_NAMES,
    DIMENSION_WEIGHTS,
    ParadigmLattice,
    ParadigmPoint,
)
from flux_a2a.paradigm_flow import (
    BridgeCost,
    BridgeSimulation,
    ParadigmFlow,
    generate_simulation_report,
)
from flux_a2a.discussion import (
    BrainstormStrategy,
    DebateStrategy,
    DEFAULT_REVIEW_CHECKLIST,
    DiscussionConfig,
    DiscussionFormat,
    DiscussionOutcome,
    DiscussionProtocol,
    DiscussionResult,
    DiscussionTurn,
    NegotiationStrategy,
    PeerReviewStrategy,
    Phase,
    ReviewCriterion,
    ReviewStrategy,
)
from flux_a2a.consensus import (
    AgentPosition,
    AgreementMetrics,
    ConsensusDetector,
    ConsensusType,
    ConvergenceHistory,
    ConvergenceTrend,
    ResolutionStrategy,
    ResolutionType,
    SimilarityMetric,
    Stalemate,
)
from flux_a2a.pipeline import (
    AgentSpec,
    AgentWorkflowPipeline,
    BranchResult,
    BranchingType,
    OutputType,
    SynthesisApproach,
    WorkflowResult,
    WorkflowSpec,
    WorkflowStatus,
)
from flux_a2a.evolution import (
    CompiledPattern,
    EvolutionEngine,
    EvolutionLevel,
    FitnessMetrics,
    GrammarDelta,
    HotPath,
    NLObservation,
    NLPattern,
    Optimization,
    OptimizationKind,
)
from flux_a2a.partial_eval import (
    PELevel,
    PEResult,
    PartialEvaluator,
    ReductionResult,
    StaticKnowledge,
    build_knowledge,
    partial_evaluate,
    specialize_interpreter,
)
from flux_a2a.types import (
    ConstraintKind,
    FluxBaseType,
    FluxConstraint,
    FluxType,
    FluxTypeRegistry,
    FluxTypeSignature,
    QuantumTypeState,
    build_default_registry,
)
from flux_a2a.type_checker import (
    BridgeResult,
    BridgeStrategy,
    CompatibilityLevel,
    CompatibilityReport,
    TypeBridge,
    TypeCheckResult,
    TypeCompatibility,
    UniversalTypeChecker,
)
