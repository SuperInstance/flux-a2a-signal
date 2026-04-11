"""
FLUX-A2A Pipeline — Branch-Merge-Synthesize Agent Workflow (Round 6).

This module implements the AgentWorkflowPipeline, the core workflow that ties
together the discussion protocol and consensus detection:

  1. Parse workflow spec (JSON)
  2. Branch into parallel agents
  3. Each agent explores (script/compile/meta-compile)
  4. Agents discuss findings (using DiscussionProtocol)
  5. Detect consensus (using ConsensusDetector)
  6. If no consensus: re-branch with refined prompts
  7. If consensus: synthesize results
  8. Output: code, spec, decision, or new questions

Design principles:
  - Workflows are declarative JSON specs — the pipeline is the interpreter.
  - The pipeline is async-first — agents run concurrently.
  - Re-branching on stalemate is automatic and configurable.
  - All state is serializable for debugging and replay.
  - Quality thresholds prevent low-confidence outputs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from flux_a2a.consensus import (
    AgentPosition,
    AgreementMetrics,
    ConsensusDetector,
    ConsensusType,
    ConvergenceTrend,
    ResolutionType,
    SimilarityMetric,
    Stalemate,
)
from flux_a2a.discussion import (
    AgentPosition as DiscussionAgentPosition,
    BrainstormStrategy,
    DebateStrategy,
    DiscussionConfig,
    DiscussionFormat,
    DiscussionOutcome,
    DiscussionProtocol,
    DiscussionResult,
    DiscussionStrategy,
    DiscussionTurn,
    NegotiationStrategy,
    PeerReviewStrategy,
    Phase,
    ReviewStrategy,
)


# ===========================================================================
# Enums
# ===========================================================================

class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    BRANCHING = "branching"
    EXPLORING = "exploring"
    DISCUSSING = "discussing"
    DETECTING_CONSENSUS = "detecting_consenting"
    REBRANCHING = "rebranching"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SynthesisApproach(str, Enum):
    """How to synthesize final results."""
    BEST_CONFIDENCE = "best_confidence"
    WEIGHTED_MERGE = "weighted_merge"
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS_ONLY = "unanimous_only"
    ALL_RESULTS = "all_results"
    COMPROMISE = "compromise"


class BranchingType(str, Enum):
    """How to branch agents."""
    PARALLEL = "parallel"
    PERSPECTIVE = "perspective"      # Each agent gets a different perspective
    ADVERSARIAL = "adversarial"      # Pairs of agents argue
    SPECIALIST = "specialist"        # Each agent has a different specialty
    RANDOM = "random"                # Random variation in prompts


class OutputType(str, Enum):
    """What the workflow produces."""
    CODE = "code"
    SPEC = "spec"
    DECISION = "decision"
    QUESTION = "question"
    SUMMARY = "summary"
    MIXED = "mixed"


# ===========================================================================
# Helpers
# ===========================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ===========================================================================
# Data models
# ===========================================================================

@dataclass(slots=True)
class AgentSpec:
    """Specification for an agent in a workflow."""
    id: str = ""
    role: str = ""
    stance: str = "neutral"
    expertise: list[str] = field(default_factory=list)
    weight: float = 1.0
    prompt_template: str = ""
    priorities: list[float] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = max(0.0, float(self.weight))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "stance": self.stance,
            "weight": self.weight,
        }
        if self.expertise:
            d["expertise"] = self.expertise
        if self.prompt_template:
            d["prompt_template"] = self.prompt_template
        if self.priorities:
            d["priorities"] = self.priorities
        if self.goals:
            d["goals"] = self.goals
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        return cls(
            id=data.get("id", ""),
            role=data.get("role", ""),
            stance=data.get("stance", "neutral"),
            expertise=data.get("expertise", []),
            weight=data.get("weight", 1.0),
            prompt_template=data.get("prompt_template", ""),
            priorities=data.get("priorities", []),
            goals=data.get("goals", []),
            meta=data.get("meta", {}),
        )


@dataclass(slots=True)
class WorkflowSpec:
    """
    JSON specification of a multi-agent workflow.

    Example:
        {
            "goal": "Design a caching strategy for the API",
            "agents": [
                {"id": "perf-expert", "role": "performance", "stance": "pro-cache", "weight": 1.0},
                {"id": "simp-expert", "role": "simplicity", "stance": "con-cache", "weight": 0.8},
                {"id": "sec-expert", "role": "security", "stance": "neutral", "weight": 0.9}
            ],
            "branching_strategy": "perspective",
            "discussion_format": "debate",
            "synthesis_method": "weighted_merge",
            "max_rounds": 5,
            "quality_threshold": 0.7,
            "max_rebranches": 2
        }
    """
    id: str = ""
    goal: str = ""
    agents: list[AgentSpec] = field(default_factory=list)
    branching_strategy: str = BranchingType.PERSPECTIVE.value
    discussion_format: str = DiscussionFormat.DEBATE.value
    synthesis_method: str = SynthesisApproach.WEIGHTED_MERGE.value
    max_rounds: int = 5
    quality_threshold: float = 0.7
    max_rebranches: int = 2
    consensus_threshold: float = 0.8
    context: dict[str, Any] = field(default_factory=dict)
    output_type: str = OutputType.DECISION.value
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        self.quality_threshold = _clamp(self.quality_threshold)
        self.consensus_threshold = _clamp(self.consensus_threshold)
        # Auto-convert dict agents to AgentSpec
        self.agents = [
            a if isinstance(a, AgentSpec) else AgentSpec.from_dict(a)
            for a in self.agents
        ]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "goal": self.goal,
            "agents": [a.to_dict() for a in self.agents],
            "branching_strategy": self.branching_strategy,
            "discussion_format": self.discussion_format,
            "synthesis_method": self.synthesis_method,
            "max_rounds": self.max_rounds,
            "quality_threshold": self.quality_threshold,
            "max_rebranches": self.max_rebranches,
            "consensus_threshold": self.consensus_threshold,
            "output_type": self.output_type,
        }
        if self.context:
            d["context"] = self.context
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowSpec:
        agents = [AgentSpec.from_dict(a) for a in data.get("agents", [])]
        return cls(
            id=data.get("id", ""),
            goal=data.get("goal", ""),
            agents=agents,
            branching_strategy=data.get("branching_strategy", BranchingType.PERSPECTIVE.value),
            discussion_format=data.get("discussion_format", DiscussionFormat.DEBATE.value),
            synthesis_method=data.get("synthesis_method", SynthesisApproach.WEIGHTED_MERGE.value),
            max_rounds=data.get("max_rounds", 5),
            quality_threshold=data.get("quality_threshold", 0.7),
            max_rebranches=data.get("max_rebranches", 2),
            consensus_threshold=data.get("consensus_threshold", 0.8),
            context=data.get("context", {}),
            output_type=data.get("output_type", OutputType.DECISION.value),
            meta=data.get("meta", {}),
        )


@dataclass(slots=True)
class BranchResult:
    """Result from a single branch exploration."""
    branch_id: str = ""
    agent_id: str = ""
    content: Any = None
    confidence: float = 0.5
    position: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "branch_id": self.branch_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "confidence": self.confidence,
        }
        if self.position:
            d["position"] = self.position
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchResult:
        return cls(
            branch_id=data.get("branch_id", ""),
            agent_id=data.get("agent_id", ""),
            content=data.get("content"),
            confidence=data.get("confidence", 0.5),
            position=data.get("position"),
            meta=data.get("meta", {}),
        )


@dataclass(slots=True)
class WorkflowResult:
    """The result of a completed workflow."""
    workflow_id: str = ""
    goal: str = ""
    status: str = WorkflowStatus.COMPLETED.value
    output: Any = None
    output_type: str = OutputType.DECISION.value
    confidence: float = 0.0
    quality_score: float = 0.0
    discussion_result: Optional[dict[str, Any]] = None
    consensus_metrics: Optional[dict[str, Any]] = None
    branches_completed: int = 0
    discussion_rounds: int = 0
    rebranches: int = 0
    stalemates_detected: int = 0
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)
        self.quality_score = _clamp(self.quality_score)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "status": self.status,
            "output": self.output,
            "output_type": self.output_type,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "branches_completed": self.branches_completed,
            "discussion_rounds": self.discussion_rounds,
            "rebranches": self.rebranches,
            "stalemates_detected": self.stalemates_detected,
        }
        if self.discussion_result:
            d["discussion_result"] = self.discussion_result
        if self.consensus_metrics:
            d["consensus_metrics"] = self.consensus_metrics
        if self.execution_log:
            d["execution_log"] = self.execution_log
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowResult:
        return cls(**{k: v for k, v in data.items() if k in {
            "workflow_id", "goal", "status", "output", "output_type",
            "confidence", "quality_score", "discussion_result", "consensus_metrics",
            "branches_completed", "discussion_rounds", "rebranches",
            "stalemates_detected", "execution_log", "meta",
        }})


# ===========================================================================
# Pipeline Execution Log
# ===========================================================================

@dataclass(slots=True)
class PipelineStep:
    """A single step in pipeline execution for audit/debug."""
    step_number: int = 0
    phase: str = ""
    description: str = ""
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "phase": self.phase,
            "description": self.description,
            "timestamp": self.timestamp,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }


# ===========================================================================
# AgentWorkflowPipeline — the main orchestrator
# ===========================================================================

class AgentWorkflowPipeline:
    """
    The complete agent workflow: branch → explore → discuss → synthesize.

    This is the core runtime that ties together the discussion protocol
    and consensus detection into a complete multi-agent workflow.

    Usage:
        spec = WorkflowSpec.from_dict({
            "goal": "Design a caching strategy",
            "agents": [
                {"id": "pro", "stance": "pro"},
                {"id": "con", "stance": "con"},
            ],
            "discussion_format": "debate",
            "max_rounds": 3,
        })
        pipeline = AgentWorkflowPipeline(spec)
        result = await pipeline.execute()

    The pipeline can also be used step-by-step:
        pipeline = AgentWorkflowPipeline(spec)
        branches = pipeline.branch()
        branch_results = pipeline.explore(branches)
        discussion = pipeline.discuss(branch_results)
        consensus = pipeline.detect_consensus(discussion)
        result = pipeline.synthesize(discussion, consensus)
    """

    def __init__(
        self,
        spec: WorkflowSpec,
        agent_executor: Optional[Callable] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            spec: The workflow specification.
            agent_executor: Optional async callable(branch_prompt, agent_spec) -> BranchResult.
                If not provided, the pipeline uses a default mock executor.
        """
        self.spec = spec
        self.agent_executor = agent_executor or self._default_executor
        self.consensus_detector = ConsensusDetector(
            threshold=spec.consensus_threshold,
        )
        self._execution_log: list[PipelineStep] = []
        self._step_counter = 0
        self._rebranch_count = 0
        self._stalemate_count = 0
        self._discussion_rounds = 0

    def _log(self, phase: str, description: str, data: dict[str, Any] | None = None) -> None:
        """Log a pipeline step."""
        self._step_counter += 1
        step = PipelineStep(
            step_number=self._step_counter,
            phase=phase,
            description=description,
            timestamp=_now(),
            data=data or {},
        )
        self._execution_log.append(step)

    # -- Default mock executor -----------------------------------------------

    @staticmethod
    async def _default_executor(prompt: str, agent_spec: AgentSpec) -> BranchResult:
        """
        Default mock agent executor for testing.

        In production, this would call actual LLM agents. The mock returns
        a simple response based on the agent's stance and the prompt.
        """
        stance = agent_spec.stance
        content = {
            "prompt": prompt,
            "agent_id": agent_spec.id,
            "role": agent_spec.role,
            "stance": stance,
            "response": f"Agent {agent_spec.id} ({stance}) response to: {prompt[:100]}",
        }
        return BranchResult(
            branch_id=f"branch-{agent_spec.id}",
            agent_id=agent_spec.id,
            content=content,
            confidence=0.7,
            position={
                "approach": [0.8 if stance == "pro" else 0.2],
                "goal": [1.0],
                "priority": agent_spec.priorities[:1] if agent_spec.priorities else [1.0],
                "confidence": 0.7,
            },
        )

    # -- Phase 1: Branch ----------------------------------------------------

    def branch(self) -> list[dict[str, Any]]:
        """
        Create branch prompts for each agent.

        Each agent gets a prompt based on the workflow goal, their role,
        and the branching strategy.
        """
        self._log(WorkflowStatus.BRANCHING.value, "Creating agent branches")

        branches: list[dict[str, Any]] = []
        for agent in self.spec.agents:
            prompt = self._generate_branch_prompt(agent)
            branches.append({
                "agent_id": agent.id,
                "prompt": prompt,
                "stance": agent.stance,
                "role": agent.role,
            })

        self._log(WorkflowStatus.BRANCHING.value,
                   f"Created {len(branches)} branches",
                   {"branches": [{"agent_id": b["agent_id"]} for b in branches]})
        return branches

    def _generate_branch_prompt(self, agent: AgentSpec) -> str:
        """Generate a branch prompt based on the branching strategy."""
        strategy = self.spec.branching_strategy
        goal = self.spec.goal

        if strategy == BranchingType.PERSPECTIVE.value:
            return (
                f"From your perspective as {agent.role} with stance '{agent.stance}', "
                f"analyze: {goal}. "
                f"Expertise: {', '.join(agent.expertise) or 'general'}. "
                f"Provide your position and key arguments."
            )
        elif strategy == BranchingType.ADVERSARIAL.value:
            return (
                f"Challenge or defend the following proposition: {goal}. "
                f"Your stance is '{agent.stance}'. "
                f"Provide strong arguments for your position."
            )
        elif strategy == BranchingType.SPECIALIST.value:
            return (
                f"As a specialist in {', '.join(agent.expertise) or agent.role}, "
                f"evaluate: {goal}. "
                f"Focus on your area of expertise."
            )
        else:  # PARALLEL or RANDOM
            return f"Analyze the following and provide your assessment: {goal}"

    # -- Phase 2: Explore ---------------------------------------------------

    async def explore(self, branches: list[dict[str, Any]]) -> list[BranchResult]:
        """
        Execute each branch and collect results.

        Each agent processes its branch prompt and produces a result.
        """
        self._log(WorkflowStatus.EXPLORING.value, "Exploring branches")

        results: list[BranchResult] = []
        for branch in branches:
            agent_id = branch["agent_id"]
            prompt = branch["prompt"]
            agent_spec = next(
                (a for a in self.spec.agents if a.id == agent_id),
                AgentSpec(id=agent_id),
            )
            result = await self.agent_executor(prompt, agent_spec)
            results.append(result)

        self._log(
            WorkflowStatus.EXPLORING.value,
            f"Completed {len(results)} branch explorations",
            {
                "results": [
                    {"agent_id": r.agent_id, "confidence": r.confidence}
                    for r in results
                ],
            },
        )
        return results

    # -- Phase 3: Discuss ---------------------------------------------------

    def create_discussion(
        self,
        branch_results: list[BranchResult],
    ) -> DiscussionProtocol:
        """Create a discussion protocol from branch results."""
        participants = []
        for agent in self.spec.agents:
            p: dict[str, Any] = {
                "id": agent.id,
                "stance": agent.stance,
                "weight": agent.weight,
            }
            if agent.expertise:
                p["expertise"] = agent.expertise
            if agent.goals:
                p["goals"] = agent.goals
            if agent.priorities:
                p["priorities"] = agent.priorities
            participants.append(p)

        # Build context from branch results
        context = {
            "background": self.spec.context.get("background", ""),
            "constraints": self.spec.context.get("constraints", []),
            "goal": self.spec.goal,
            "branch_results": [r.to_dict() for r in branch_results],
        }

        config = DiscussionConfig(
            format=self.spec.discussion_format,
            topic=self.spec.goal,
            participants=participants,
            context=context,
            max_rounds=self.spec.max_rounds,
            consensus_threshold=self.spec.consensus_threshold,
        )

        return DiscussionProtocol(config)

    def discuss(
        self,
        branch_results: list[BranchResult],
        turn_generator: Optional[Callable] = None,
    ) -> DiscussionResult:
        """
        Run a synchronous discussion based on branch results.

        If no turn_generator is provided, creates a simple mock discussion
        where each agent contributes positions from their branch results.
        """
        self._log(WorkflowStatus.DISCUSSING.value, "Starting agent discussion")

        protocol = self.create_discussion(branch_results)
        protocol.initialize()

        if turn_generator is None:
            turn_generator = self._make_turn_generator(branch_results)

        # Run synchronous discussion (for async, use protocol.run())
        for round_num in range(self.spec.max_rounds):
            is_done, _ = protocol.check_completion()
            if is_done:
                break

            state = protocol.get_state()
            for agent in self.spec.agents:
                turn = turn_generator(state, agent)
                if turn is None:
                    continue
                protocol.process_turn(turn)

            self._discussion_rounds += 1

        result = protocol.conclude()
        self._log(
            WorkflowStatus.DISCUSSING.value,
            f"Discussion completed: {result.outcome}",
            {
                "outcome": result.outcome,
                "confidence": result.confidence,
                "turns": len(result.turns),
            },
        )
        return result

    def _make_turn_generator(
        self,
        branch_results: list[BranchResult],
    ) -> Callable:
        """Create a default turn generator from branch results."""
        # Pre-compute agent positions from branch results
        agent_positions: dict[str, BranchResult] = {
            r.agent_id: r for r in branch_results
        }

        def generator(state: dict[str, Any], agent: AgentSpec) -> Optional[DiscussionTurn]:
            """Generate a discussion turn for an agent."""
            br = agent_positions.get(agent.id)
            if br is None:
                return None

            phase = state.get("current_phase", Phase.SINGLE.value)
            fmt = self.spec.discussion_format
            round_num = state.get("current_round", 0)

            # Adjust confidence based on round (agents may shift)
            confidence = br.confidence
            if round_num > 2:
                # Simulate convergence: shift confidence slightly toward center
                confidence = min(0.95, confidence + 0.05)

            content: dict[str, Any] = {"type": "argument"}
            challenge_to = None

            if fmt == DiscussionFormat.DEBATE.value:
                if agent.stance == "pro":
                    content.update({
                        "type": "argument",
                        "argument": f"Pro argument from {agent.id}: supporting position",
                        "point": "support",
                    })
                elif agent.stance == "con":
                    content.update({
                        "type": "argument",
                        "argument": f"Con argument from {agent.id}: opposing position",
                        "point": "oppose",
                    })
                else:
                    content.update({
                        "type": "synthesis",
                        "argument": f"Synthesis from {agent.id}: integrating perspectives",
                        "point": "synthesize",
                    })

                # After a few rounds, introduce concessions
                if round_num >= 2:
                    content["type"] = "concession"
                    challenge_to = None

            elif fmt == DiscussionFormat.BRAINSTORM.value:
                if phase == Phase.GENERATE.value:
                    content.update({
                        "type": "idea",
                        "content": f"Idea from {agent.id}: novel approach to {self.spec.goal}",
                    })
                else:
                    content.update({
                        "type": "evaluation",
                        "target_idea": "idea-0",
                        "score": confidence,
                        "comment": f"{agent.id} evaluates",
                    })

            elif fmt == DiscussionFormat.REVIEW.value:
                content.update({
                    "type": "criterion_review",
                    "criterion": "correctness" if round_num % 2 == 0 else "efficiency",
                    "score": confidence,
                    "notes": f"Review by {agent.id}",
                })

            elif fmt == DiscussionFormat.NEGOTIATION.value:
                content.update({
                    "type": "proposal" if round_num <= 1 else "compromise",
                    "proposal_vector": [
                        0.8 if agent.stance == "pro" else 0.2,
                        0.5,
                    ],
                    "description": f"Proposal from {agent.id}",
                })

            elif fmt == DiscussionFormat.PEER_REVIEW.value:
                content.update({
                    "type": "independent_review_result",
                    "scores": {
                        "correctness": confidence,
                        "completeness": confidence * 0.9,
                        "clarity": confidence * 0.85,
                    },
                    "summary": f"Review by {agent.id}",
                    "recommendation": "accept" if confidence > 0.7 else "revise",
                })

            return DiscussionTurn(
                agent_id=agent.id,
                content=content,
                confidence=confidence,
                stance=agent.stance,
                challenge_to=challenge_to,
                phase=phase,
                round_number=round_num,
            )

        return generator

    # -- Phase 4: Detect consensus -------------------------------------------

    def detect_consensus(
        self,
        discussion_result: DiscussionResult,
    ) -> tuple[bool, AgreementMetrics, Optional[Stalemate]]:
        """
        Check if the discussion reached consensus.

        Returns (has_consensus, metrics, stalemate_if_any).
        """
        self._log(WorkflowStatus.DETECTING_CONSENSUS.value, "Checking consensus")

        # Convert discussion positions to AgentPosition objects
        positions = [
            AgentPosition.from_dict(p)
            for p in discussion_result.positions
        ]

        if not positions:
            # No positions — extract from turn data
            positions = self._extract_positions_from_turns(discussion_result)

        has_consensus, metrics, stalemate = self.consensus_detector.check_consensus(
            positions,
            ConsensusType(self.spec.consensus_threshold and "majority" or "majority"),
        )

        if stalemate:
            self._stalemate_count += 1

        self._log(
            WorkflowStatus.DETECTING_CONSENSUS.value,
            f"Consensus check: {metrics.consensus_type}",
            {
                "agreement_score": metrics.agreement_score,
                "consensus_type": metrics.consensus_type,
                "stalemate_detected": stalemate is not None,
            },
        )

        return has_consensus, metrics, stalemate

    def _extract_positions_from_turns(
        self,
        discussion_result: DiscussionResult,
    ) -> list[AgentPosition]:
        """Extract agent positions from discussion turns if not explicitly provided."""
        positions: dict[str, AgentPosition] = {}

        for turn_dict in discussion_result.turns:
            agent_id = turn_dict.get("agent_id", "")
            if not agent_id or agent_id in positions:
                continue

            content = turn_dict.get("content", {})
            if isinstance(content, dict) and "position_vector" in content:
                positions[agent_id] = AgentPosition(
                    agent_id=agent_id,
                    approach=content["position_vector"],
                    confidence=turn_dict.get("confidence", 0.5),
                    stance=turn_dict.get("stance", "neutral"),
                )
            else:
                # Create a synthetic position from stance
                stance = turn_dict.get("stance", "neutral")
                positions[agent_id] = AgentPosition(
                    agent_id=agent_id,
                    approach=[0.9 if stance == "pro" else 0.1 if stance == "con" else 0.5],
                    goal=[1.0],
                    confidence=turn_dict.get("confidence", 0.5),
                    label=stance,
                )

        return list(positions.values())

    # -- Phase 5: Re-branch on stalemate -------------------------------------

    def rebranch(
        self,
        stalemate: Stalemate,
        previous_results: list[BranchResult],
    ) -> WorkflowSpec:
        """
        Create a refined workflow spec for re-branching.

        Uses the stalemate information to generate more focused prompts
        that address the disagreements.
        """
        self._log(
            WorkflowStatus.REBRANCHING.value,
            f"Re-branching due to stalemate (severity: {stalemate.severity:.2f})",
            {"stalemate_reason": stalemate.reason},
        )

        self._rebranch_count += 1

        # Refine agent prompts based on stalemate
        refined_agents: list[AgentSpec] = []
        for agent in self.spec.agents:
            refined = AgentSpec(
                id=agent.id,
                role=agent.role,
                stance=agent.stance,
                expertise=agent.expertise,
                weight=agent.weight,
                prompt_template=(
                    f"Previous discussion revealed: {stalemate.reason}. "
                    f"Refine your position considering the other perspectives. "
                    f"Original goal: {self.spec.goal}"
                ),
                priorities=agent.priorities,
                goals=agent.goals,
            )
            refined_agents.append(refined)

        refined_spec = WorkflowSpec(
            goal=f"[Re-branch {self._rebranch_count}] {self.spec.goal}",
            agents=refined_agents,
            branching_strategy=self.spec.branching_strategy,
            discussion_format=self.spec.discussion_format,
            synthesis_method=self.spec.synthesis_method,
            max_rounds=max(2, self.spec.max_rounds - 1),
            quality_threshold=self.spec.quality_threshold * 0.9,  # Slightly lower threshold
            max_rebranches=max(0, self.spec.max_rebranches - 1),
            consensus_threshold=self.spec.consensus_threshold,
            context={
                **self.spec.context,
                "previous_stalemate": stalemate.to_dict(),
            },
            output_type=self.spec.output_type,
            meta={
                "parent_workflow": self.spec.id,
                "rebranch_number": self._rebranch_count,
            },
        )

        return refined_spec

    # -- Phase 6: Synthesize ------------------------------------------------

    def synthesize(
        self,
        discussion_result: DiscussionResult,
        consensus_metrics: AgreementMetrics,
        branch_results: Optional[list[BranchResult]] = None,
    ) -> WorkflowResult:
        """
        Synthesize the final result from discussion and consensus.

        The synthesis method determines how to combine results:
        - best_confidence: return the highest-confidence result
        - weighted_merge: weighted combination based on agent weights
        - majority_vote: return the majority position
        - all_results: return all results with metadata
        """
        self._log(WorkflowStatus.SYNTHESIZING.value, "Synthesizing results")

        method = self.spec.synthesis_method
        confidence = discussion_result.confidence
        quality_score = _clamp(confidence * consensus_metrics.agreement_score)

        output: Any = None

        if method == SynthesisApproach.BEST_CONFIDENCE.value:
            output = self._synthesize_best_confidence(discussion_result, branch_results)
        elif method == SynthesisApproach.WEIGHTED_MERGE.value:
            output = self._synthesize_weighted_merge(discussion_result, branch_results)
        elif method == SynthesisApproach.MAJORITY_VOTE.value:
            output = self._synthesize_majority_vote(discussion_result, consensus_metrics)
        elif method == SynthesisApproach.ALL_RESULTS.value:
            output = self._synthesize_all(discussion_result, consensus_metrics, branch_results)
        elif method == SynthesisApproach.COMPROMISE.value:
            output = self._synthesize_compromise(discussion_result, consensus_metrics)
        else:
            output = self._synthesize_best_confidence(discussion_result, branch_results)

        # Check quality threshold
        meets_quality = quality_score >= self.spec.quality_threshold
        status = WorkflowStatus.COMPLETED.value if meets_quality else WorkflowStatus.COMPLETED.value

        self._log(
            WorkflowStatus.SYNTHESIZING.value,
            f"Synthesis complete: quality={quality_score:.2f}, meets_threshold={meets_quality}",
        )

        return WorkflowResult(
            workflow_id=self.spec.id,
            goal=self.spec.goal,
            status=status,
            output=output,
            output_type=self.spec.output_type,
            confidence=confidence,
            quality_score=quality_score,
            discussion_result=discussion_result.to_dict(),
            consensus_metrics=consensus_metrics.to_dict(),
            branches_completed=len(branch_results) if branch_results else 0,
            discussion_rounds=self._discussion_rounds,
            rebranches=self._rebranch_count,
            stalemates_detected=self._stalemate_count,
            execution_log=[s.to_dict() for s in self._execution_log],
            meta={"meets_quality_threshold": meets_quality},
        )

    def _synthesize_best_confidence(
        self,
        discussion_result: DiscussionResult,
        branch_results: Optional[list[BranchResult]] = None,
    ) -> dict[str, Any]:
        """Return the highest-confidence result."""
        if branch_results:
            best = max(branch_results, key=lambda r: r.confidence)
            return {
                "type": "best_confidence",
                "source": best.agent_id,
                "content": best.content,
                "confidence": best.confidence,
                "discussion_outcome": discussion_result.outcome,
            }
        return {
            "type": "best_confidence",
            "discussion_decision": discussion_result.decision,
            "confidence": discussion_result.confidence,
            "outcome": discussion_result.outcome,
        }

    def _synthesize_weighted_merge(
        self,
        discussion_result: DiscussionResult,
        branch_results: Optional[list[BranchResult]] = None,
    ) -> dict[str, Any]:
        """Weighted combination of results."""
        weights = {a.id: a.weight for a in self.spec.agents}

        if branch_results:
            total_weight = sum(weights.get(r.agent_id, 1.0) for r in branch_results)
            merged_content = []
            for r in branch_results:
                w = weights.get(r.agent_id, 1.0) / max(total_weight, 1.0)
                merged_content.append({
                    "agent_id": r.agent_id,
                    "weight": w,
                    "content": r.content,
                    "confidence": r.confidence,
                })
            return {
                "type": "weighted_merge",
                "items": merged_content,
                "discussion_decision": discussion_result.decision,
                "confidence": discussion_result.confidence,
                "outcome": discussion_result.outcome,
            }
        return {
            "type": "weighted_merge",
            "discussion_decision": discussion_result.decision,
            "confidence": discussion_result.confidence,
            "outcome": discussion_result.outcome,
            "reasoning": discussion_result.reasoning,
        }

    def _synthesize_majority_vote(
        self,
        discussion_result: DiscussionResult,
        consensus_metrics: AgreementMetrics,
    ) -> dict[str, Any]:
        """Return the majority position."""
        return {
            "type": "majority_vote",
            "discussion_decision": discussion_result.decision,
            "outcome": discussion_result.outcome,
            "majority_fraction": consensus_metrics.majority_fraction,
            "confidence": discussion_result.confidence * consensus_metrics.majority_fraction,
            "agreement_score": consensus_metrics.agreement_score,
        }

    def _synthesize_all(
        self,
        discussion_result: DiscussionResult,
        consensus_metrics: AgreementMetrics,
        branch_results: Optional[list[BranchResult]] = None,
    ) -> dict[str, Any]:
        """Return all results with full metadata."""
        return {
            "type": "all_results",
            "discussion": discussion_result.to_dict(),
            "consensus": consensus_metrics.to_dict(),
            "branches": [r.to_dict() for r in branch_results] if branch_results else [],
            "outcome": discussion_result.outcome,
        }

    def _synthesize_compromise(
        self,
        discussion_result: DiscussionResult,
        consensus_metrics: AgreementMetrics,
    ) -> dict[str, Any]:
        """Return a compromise position."""
        return {
            "type": "compromise",
            "discussion_decision": discussion_result.decision,
            "outcome": discussion_result.outcome,
            "agreement_score": consensus_metrics.agreement_score,
            "cluster_count": consensus_metrics.cluster_count,
            "confidence": discussion_result.confidence * 0.9,  # Slightly reduced for compromise
            "reasoning": discussion_result.reasoning,
        }

    # -- Main execution entry point ------------------------------------------

    async def execute(
        self,
        turn_generator: Optional[Callable] = None,
    ) -> WorkflowResult:
        """
        Execute the complete workflow: branch → explore → discuss → detect → synthesize.

        If stalemate is detected and max_rebranches hasn't been reached,
        automatically re-branches with refined prompts.

        Args:
            turn_generator: Optional callable(state, agent) -> DiscussionTurn.
                If not provided, uses default generator from branch results.

        Returns:
            WorkflowResult with the final output and metadata.
        """
        self._log("execute", f"Starting workflow: {self.spec.goal}")

        # Phase 1: Branch
        branches = self.branch()

        # Phase 2: Explore
        branch_results = await self.explore(branches)

        # Phase 3: Discuss
        discussion_result = self.discuss(branch_results, turn_generator)

        # Phase 4: Detect consensus
        has_consensus, consensus_metrics, stalemate = self.detect_consensus(discussion_result)

        # Phase 5: Handle stalemate with re-branching
        if not has_consensus and stalemate is not None:
            if self._rebranch_count < self.spec.max_rebranches:
                refined_spec = self.rebranch(stalemate, branch_results)

                # Create a sub-pipeline for the re-branch
                sub_pipeline = AgentWorkflowPipeline(refined_spec, self.agent_executor)
                sub_pipeline._rebranch_count = self._rebranch_count
                sub_pipeline._stalemate_count = self._stalemate_count
                sub_result = await sub_pipeline.execute(turn_generator)

                # Merge sub-pipeline results
                self._discussion_rounds += sub_result.discussion_rounds
                self._rebranch_count = sub_result.rebranches
                self._stalemate_count = sub_result.stalemates_detected
                self._execution_log.extend(
                    [PipelineStep(**s) if isinstance(s, dict) else s
                     for s in sub_result.execution_log]
                )

                return sub_result

        # Phase 6: Synthesize
        result = self.synthesize(discussion_result, consensus_metrics, branch_results)

        self._log("execute", f"Workflow complete: {result.status}")
        return result

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get the full execution log."""
        return [s.to_dict() for s in self._execution_log]

    def get_state(self) -> dict[str, Any]:
        """Get current pipeline state."""
        return {
            "workflow_id": self.spec.id,
            "goal": self.spec.goal,
            "step": self._step_counter,
            "rebranches": self._rebranch_count,
            "stalemates": self._stalemate_count,
            "discussion_rounds": self._discussion_rounds,
            "consensus_history": self.consensus_detector.get_consensus_summary(),
        }
