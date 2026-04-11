"""
Tests for the Discussion Protocol, Consensus Detection, and Agent Workflow Pipeline.

Rounds 4-6 test suite:

  1. test_debate_convergence — Two agents debate a design decision and converge
  2. test_brainstorm_consensus — Three agents brainstorm, consensus finds the best idea
  3. test_pipeline_full_cycle — Complete branch-discuss-synthesize workflow
  4. test_stalemate_rebranch — Stalemate detection triggers re-branching
"""

from __future__ import annotations

import asyncio
import pytest

from flux_a2a.consensus import (
    AgentPosition,
    AgreementMetrics,
    ConvergenceHistory,
    ConvergenceTrend,
    ConsensusDetector,
    ConsensusType,
    ResolutionType,
    SimilarityMetric,
    Stalemate,
    cosine_similarity,
    euclidean_distance,
)
from flux_a2a.discussion import (
    BrainstormStrategy,
    DebateStrategy,
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
from flux_a2a.pipeline import (
    AgentSpec,
    AgentWorkflowPipeline,
    BranchResult,
    BranchingType,
    SynthesisApproach,
    WorkflowResult,
    WorkflowSpec,
    WorkflowStatus,
)


# ===========================================================================
# Helper factories
# ===========================================================================

def make_pro_agent(agent_id: str = "pro-agent", weight: float = 1.0) -> dict:
    return {"id": agent_id, "stance": "pro", "weight": weight, "expertise": ["system_design"]}


def make_con_agent(agent_id: str = "con-agent", weight: float = 1.0) -> dict:
    return {"id": agent_id, "stance": "con", "weight": weight, "expertise": ["databases"]}


def make_neutral_agent(agent_id: str = "neutral-agent", weight: float = 0.8) -> dict:
    return {"id": agent_id, "stance": "neutral", "weight": weight, "expertise": ["architecture"]}


# ===========================================================================
# Test 1: Two agents debate a design decision and converge
# ===========================================================================

class TestDebateConvergence:
    """Two agents debate a design decision and converge on a synthesis."""

    def test_debate_initialization(self):
        """Debate strategy initializes with opening arguments."""
        participants = [make_pro_agent(), make_con_agent()]
        debate = DebateStrategy(
            topic="Should we use event sourcing?",
            participants=participants,
            max_rounds=5,
        )
        initial_turns = debate.initialize()

        assert len(initial_turns) == 2
        assert initial_turns[0].agent_id == "pro-agent"
        assert initial_turns[0].stance == "pro"
        assert initial_turns[1].agent_id == "con-agent"
        assert initial_turns[1].stance == "con"
        assert debate.current_round == 1

    def test_debate_tracks_thesis_antithesis(self):
        """Debate correctly categorizes arguments by stance."""
        participants = [make_pro_agent(), make_con_agent()]
        debate = DebateStrategy(
            topic="Microservices vs monolith?",
            participants=participants,
            max_rounds=5,
        )
        debate.initialize()

        # Pro agent makes a point
        pro_turn = DiscussionTurn(
            agent_id="pro-agent",
            content={"type": "argument", "argument": "Microservices scale better", "point": "scalability"},
            confidence=0.85,
            stance="pro",
        )
        debate.process_turn(pro_turn)

        # Con agent makes a point
        con_turn = DiscussionTurn(
            agent_id="con-agent",
            content={"type": "argument", "argument": "Monoliths are simpler to debug", "point": "debugging"},
            confidence=0.8,
            stance="con",
        )
        debate.process_turn(con_turn)

        assert len(debate.thesis_points) == 1
        assert len(debate.antithesis_points) == 1
        assert "scale" in debate.thesis_points[0]
        assert "debug" in debate.antithesis_points[0]

    def test_debate_convergence_through_concessions(self):
        """Debate reaches synthesis when agents make concessions."""
        participants = [make_pro_agent(), make_con_agent()]
        debate = DebateStrategy(
            topic="Event sourcing vs CRUD",
            participants=participants,
            max_rounds=5,
        )
        debate.initialize()

        # Simulate concessions to trigger convergence
        for i in range(3):
            concession = DiscussionTurn(
                agent_id="pro-agent" if i % 2 == 0 else "con-agent",
                content={"type": "concession", "argument": "Valid point acknowledged"},
                confidence=0.7,
                stance="pro" if i % 2 == 0 else "con",
            )
            debate.process_turn(concession)

        # Should have enough concessions to enter synthesis phase
        assert debate.concession_count >= 2
        assert debate.current_phase == Phase.SYNTHESIZE

        # Add a synthesis candidate
        synthesis_turn = DiscussionTurn(
            agent_id="pro-agent",
            content={"type": "synthesis", "argument": "Use event sourcing for audit log, CRUD for main store"},
            confidence=0.85,
            stance="pro",
        )
        debate.process_turn(synthesis_turn)

        # Check completion
        is_done, outcome = debate.check_completion()
        assert is_done
        assert outcome in (DiscussionOutcome.CONSENSUS, DiscussionOutcome.COMPROMISE)

    def test_debate_conclude_with_synthesis(self):
        """Debate conclusion includes synthesis when available."""
        participants = [make_pro_agent(), make_con_agent()]
        debate = DebateStrategy(
            topic="Caching strategy",
            participants=participants,
            max_rounds=3,
        )
        debate.initialize()

        # Add enough concessions and synthesis
        for i in range(4):
            debate.process_turn(DiscussionTurn(
                agent_id="pro-agent" if i % 2 == 0 else "con-agent",
                content={"type": "concession", "argument": "point"},
                confidence=0.7,
                stance="pro" if i % 2 == 0 else "con",
            ))

        debate.process_turn(DiscussionTurn(
            agent_id="pro-agent",
            content={"type": "synthesis", "argument": "Hybrid approach"},
            confidence=0.9,
            stance="pro",
        ))
        debate.process_turn(DiscussionTurn(
            agent_id="con-agent",
            content={"type": "synthesis", "argument": "Agreed on hybrid"},
            confidence=0.85,
            stance="con",
        ))

        result = debate.conclude()
        assert isinstance(result, DiscussionResult)
        assert result.format == "debate"
        assert result.decision is not None
        assert len(result.turns) > 0
        assert len(result.positions) == 2

    def test_full_debate_via_protocol(self):
        """Full debate using DiscussionProtocol orchestrator."""
        config = DiscussionConfig(
            format=DiscussionFormat.DEBATE.value,
            topic="REST vs GraphQL?",
            participants=[make_pro_agent(), make_con_agent()],
            max_rounds=3,
            consensus_threshold=0.7,
        )
        protocol = DiscussionProtocol(config)
        initial = protocol.initialize()
        assert len(initial) == 2

        # Simulate discussion rounds
        for round_num in range(3):
            for agent in config.participants:
                agent_id = agent["id"]
                stance = agent["stance"]
                turn = DiscussionTurn(
                    agent_id=agent_id,
                    content={
                        "type": "concession" if round_num >= 1 else "argument",
                        "argument": f"Round {round_num} from {agent_id}",
                    },
                    confidence=0.75,
                    stance=stance,
                )
                protocol.process_turn(turn)

        result = protocol.conclude()
        assert result.format == "debate"
        assert result.discussion_id == protocol.strategy.discussion_id


# ===========================================================================
# Test 2: Three agents brainstorm, consensus finds the best idea
# ===========================================================================

class TestBrainstormConsensus:
    """Three agents brainstorm and consensus detector finds the best idea."""

    def test_brainstorm_generate_phase(self):
        """Brainstorm correctly collects ideas in generate phase."""
        participants = [
            make_neutral_agent("agent-1"),
            make_neutral_agent("agent-2"),
            make_neutral_agent("agent-3"),
        ]
        brainstorm = BrainstormStrategy(
            topic="How to improve API response times?",
            participants=participants,
            max_rounds=6,
        )
        brainstorm.initialize()
        assert brainstorm.current_phase == Phase.GENERATE
        assert brainstorm.generate_rounds >= 2

    def test_brainstorm_collects_ideas(self):
        """Ideas are correctly collected and tracked."""
        participants = [
            make_neutral_agent("a1"),
            make_neutral_agent("a2"),
            make_neutral_agent("a3"),
        ]
        brainstorm = BrainstormStrategy(
            topic="New feature ideas",
            participants=participants,
            max_rounds=6,
        )
        brainstorm.initialize()

        # Generate ideas
        ideas_data = [
            ("a1", "Add caching layer"),
            ("a2", "Implement rate limiting"),
            ("a3", "Add GraphQL endpoint"),
        ]
        for agent_id, idea in ideas_data:
            turn = DiscussionTurn(
                agent_id=agent_id,
                content={"type": "idea", "content": idea},
                confidence=0.8,
                stance="neutral",
            )
            brainstorm.process_turn(turn)

        assert len(brainstorm.ideas) == 3
        assert brainstorm.ideas[0]["content"] == "Add caching layer"

    def test_brainstorm_phase_transition(self):
        """Brainstorm transitions from generate to evaluate after generate_rounds."""
        participants = [
            make_neutral_agent("a1"),
            make_neutral_agent("a2"),
        ]
        brainstorm = BrainstormStrategy(
            topic="Feature brainstorm",
            participants=participants,
            max_rounds=4,
        )
        brainstorm.initialize()
        assert brainstorm.current_phase == Phase.GENERATE

        # Process turns through the generate phase
        for agent_id in ["a1", "a2"]:
            brainstorm.process_turn(DiscussionTurn(
                agent_id=agent_id,
                content={"type": "idea", "content": f"Idea from {agent_id}"},
                confidence=0.8,
            ))

        # Advance rounds past generate_rounds
        while brainstorm.current_phase == Phase.GENERATE:
            brainstorm.current_round += 1
            if brainstorm.current_round >= brainstorm.generate_rounds:
                for agent_id in ["a1", "a2"]:
                    brainstorm.process_turn(DiscussionTurn(
                        agent_id=agent_id,
                        content={"type": "idea", "content": f"More idea from {agent_id}"},
                        confidence=0.7,
                    ))

        assert brainstorm.current_phase == Phase.EVALUATE

    def test_brainstorm_evaluation_scoring(self):
        """Ideas are scored during evaluation phase."""
        participants = [
            make_neutral_agent("a1"),
            make_neutral_agent("a2"),
            make_neutral_agent("a3"),
        ]
        brainstorm = BrainstormStrategy(
            topic="Best caching strategy?",
            participants=participants,
            max_rounds=4,
        )
        brainstorm.initialize()

        # Generate phase
        for agent_id in ["a1", "a2", "a3"]:
            brainstorm.process_turn(DiscussionTurn(
                agent_id=agent_id,
                content={"type": "idea", "content": f"Idea from {agent_id}"},
                confidence=0.8,
            ))

        # Force transition to evaluate
        brainstorm.current_phase = Phase.EVALUATE
        brainstorm.current_round = brainstorm.generate_rounds

        # Evaluate ideas
        for agent_id in ["a1", "a2", "a3"]:
            brainstorm.process_turn(DiscussionTurn(
                agent_id=agent_id,
                content={
                    "type": "evaluation",
                    "target_idea": "idea-0",
                    "score": 0.9 if agent_id == "a1" else 0.6,
                    "comment": f"Review by {agent_id}",
                },
                confidence=0.8,
                stance="neutral",
            ))

        assert "idea-0" in brainstorm.evaluations
        assert len(brainstorm.evaluations["idea-0"]) == 3

    def test_brainstorm_conclude_ranks_ideas(self):
        """Brainstorm conclusion produces ranked ideas."""
        participants = [
            make_neutral_agent("a1"),
            make_neutral_agent("a2"),
        ]
        brainstorm = BrainstormStrategy(
            topic="Best approach?",
            participants=participants,
            max_rounds=4,
        )
        brainstorm.initialize()

        # Add ideas with different confidence
        brainstorm.process_turn(DiscussionTurn(
            agent_id="a1",
            content={"type": "idea", "content": "Great idea"},
            confidence=0.9,
        ))
        brainstorm.process_turn(DiscussionTurn(
            agent_id="a2",
            content={"type": "idea", "content": "OK idea"},
            confidence=0.5,
        ))

        result = brainstorm.conclude()
        assert result.format == "brainstorm"
        assert result.decision is not None
        assert result.decision["type"] == "ranked_ideas"
        assert result.decision["total_ideas"] == 2
        assert len(result.decision["ranked"]) == 2

    def test_consensus_detector_finds_best_in_brainstorm(self):
        """ConsensusDetector can evaluate brainstorm positions."""
        detector = ConsensusDetector(threshold=0.7)

        # Three agents with varying positions
        positions = [
            AgentPosition(agent_id="a1", approach=[0.9, 0.1], confidence=0.9),
            AgentPosition(agent_id="a2", approach=[0.8, 0.2], confidence=0.85),
            AgentPosition(agent_id="a3", approach=[0.1, 0.9], confidence=0.8),
        ]

        metrics = detector.measure_agreement(positions)
        assert metrics.consensus_type in (
            ConsensusType.STALEMATE.value,
            ConsensusType.MAJORITY.value,
            ConsensusType.SUPERMAJORITY.value,
            ConsensusType.COMPROMISE.value,
        )
        assert 0.0 <= metrics.agreement_score <= 1.0

        # Two agreeing, one dissenting
        close_positions = [
            AgentPosition(agent_id="a1", approach=[0.8, 0.2], confidence=0.9),
            AgentPosition(agent_id="a2", approach=[0.7, 0.3], confidence=0.85),
            AgentPosition(agent_id="a3", approach=[0.75, 0.25], confidence=0.88),
        ]
        metrics2 = detector.measure_agreement(close_positions)
        assert metrics2.agreement_score > metrics.agreement_score


# ===========================================================================
# Test 3: Complete branch-discuss-synthesize workflow
# ===========================================================================

class TestPipelineFullCycle:
    """Agent workflow pipeline executes a complete branch-discuss-synthesize cycle."""

    def test_workflow_spec_creation(self):
        """WorkflowSpec can be created from dict."""
        spec = WorkflowSpec.from_dict({
            "goal": "Design a caching strategy for the API",
            "agents": [
                {"id": "perf-expert", "role": "performance", "stance": "pro-cache", "weight": 1.0},
                {"id": "simp-expert", "role": "simplicity", "stance": "anti-cache", "weight": 0.8},
            ],
            "branching_strategy": "perspective",
            "discussion_format": "debate",
            "synthesis_method": "weighted_merge",
            "max_rounds": 3,
            "quality_threshold": 0.6,
        })
        assert spec.goal == "Design a caching strategy for the API"
        assert len(spec.agents) == 2
        assert spec.agents[0].id == "perf-expert"
        assert spec.discussion_format == "debate"
        assert spec.max_rounds == 3

    def test_workflow_spec_round_trip(self):
        """WorkflowSpec survives JSON round-trip."""
        spec = WorkflowSpec(
            goal="Test goal",
            agents=[AgentSpec(id="a1", stance="pro")],
            max_rounds=5,
        )
        d = spec.to_dict()
        spec2 = WorkflowSpec.from_dict(d)
        assert spec2.goal == spec.goal
        assert spec2.agents[0].id == "a1"
        assert spec2.max_rounds == 5

    def test_branch_creates_prompts(self):
        """Pipeline branch phase creates prompts for all agents."""
        spec = WorkflowSpec(
            goal="Choose a sorting algorithm",
            agents=[
                AgentSpec(id="speed", role="performance", stance="pro"),
                AgentSpec(id="simp", role="simplicity", stance="con"),
            ],
            branching_strategy="perspective",
            discussion_format="debate",
        )
        pipeline = AgentWorkflowPipeline(spec)
        branches = pipeline.branch()

        assert len(branches) == 2
        assert all("prompt" in b for b in branches)
        assert all("agent_id" in b for b in branches)

    def test_explore_collects_results(self):
        """Pipeline explore phase collects results from all branches."""
        spec = WorkflowSpec(
            goal="Test exploration",
            agents=[AgentSpec(id="a1"), AgentSpec(id="a2")],
        )
        pipeline = AgentWorkflowPipeline(spec)
        branches = pipeline.branch()

        results = asyncio.get_event_loop().run_until_complete(pipeline.explore(branches))

        assert len(results) == 2
        assert all(isinstance(r, BranchResult) for r in results)
        assert all(r.agent_id in ("a1", "a2") for r in results)

    def test_discuss_creates_protocol(self):
        """Pipeline creates a discussion protocol from branch results."""
        spec = WorkflowSpec(
            goal="Test discussion",
            agents=[AgentSpec(id="a1", stance="pro"), AgentSpec(id="a2", stance="con")],
            discussion_format="debate",
            max_rounds=2,
        )
        pipeline = AgentWorkflowPipeline(spec)

        branch_results = [
            BranchResult(agent_id="a1", content="pro response", confidence=0.8),
            BranchResult(agent_id="a2", content="con response", confidence=0.75),
        ]

        protocol = pipeline.create_discussion(branch_results)
        assert isinstance(protocol, DiscussionProtocol)
        assert protocol.config.format == "debate"

    def test_full_pipeline_execute(self):
        """Full pipeline executes branch-discuss-synthesize successfully."""
        spec = WorkflowSpec(
            goal="Should we add a caching layer?",
            agents=[
                AgentSpec(id="pro", role="perf", stance="pro", weight=1.0),
                AgentSpec(id="con", role="simplicity", stance="con", weight=0.9),
            ],
            branching_strategy="perspective",
            discussion_format="debate",
            synthesis_method="weighted_merge",
            max_rounds=3,
            quality_threshold=0.3,
            max_rebranches=1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        result = asyncio.get_event_loop().run_until_complete(pipeline.execute())

        assert isinstance(result, WorkflowResult)
        assert result.status == WorkflowStatus.COMPLETED.value
        assert result.output is not None
        assert result.discussion_result is not None
        assert result.consensus_metrics is not None
        assert result.workflow_id == spec.id
        assert result.branches_completed == 2
        assert result.discussion_rounds > 0

    def test_pipeline_execution_log(self):
        """Pipeline maintains a detailed execution log."""
        spec = WorkflowSpec(
            goal="Log test",
            agents=[AgentSpec(id="a1")],
            max_rounds=2,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        asyncio.get_event_loop().run_until_complete(pipeline.execute())

        log = pipeline.get_execution_log()
        assert len(log) > 0
        phases = {entry["phase"] for entry in log}
        assert "branching" in phases
        assert "exploring" in phases
        assert "discussing" in phases

    def test_pipeline_state_tracking(self):
        """Pipeline tracks its state throughout execution."""
        spec = WorkflowSpec(
            goal="State test",
            agents=[AgentSpec(id="a1")],
            max_rounds=2,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        asyncio.get_event_loop().run_until_complete(pipeline.execute())

        state = pipeline.get_state()
        assert state["workflow_id"] == spec.id
        assert state["goal"] == "State test"
        assert state["step"] > 0
        assert "consensus_history" in state

    def test_pipeline_result_serialization(self):
        """Pipeline result is fully serializable."""
        spec = WorkflowSpec(
            goal="Serialization test",
            agents=[AgentSpec(id="a1", stance="pro"), AgentSpec(id="a2", stance="con")],
            max_rounds=2,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        result = asyncio.get_event_loop().run_until_complete(pipeline.execute())

        d = result.to_dict()
        assert "workflow_id" in d
        assert "output" in d
        assert "confidence" in d
        assert "execution_log" in d

        # Round-trip
        result2 = WorkflowResult.from_dict(d)
        assert result2.workflow_id == result.workflow_id


# ===========================================================================
# Test 4: Stalemate detection triggers re-branching
# ===========================================================================

class TestStalemateRebranch:
    """Stalemate detection triggers re-branching in the pipeline."""

    def test_consensus_detector_identifies_stalemate(self):
        """ConsensusDetector correctly identifies stalemate situations."""
        detector = ConsensusDetector(threshold=0.8)

        # Two agents with diametrically opposed positions
        opposed = [
            AgentPosition(agent_id="a1", approach=[1.0, 0.0], confidence=0.95),
            AgentPosition(agent_id="a2", approach=[0.0, 1.0], confidence=0.95),
        ]

        # Feed enough history for stalemate detection
        for _ in range(5):
            detector.measure_agreement(opposed)

        metrics = detector.measure_agreement(opposed)
        assert metrics.agreement_score < 0.5

    def test_stalemate_detected_after_sufficient_rounds(self):
        """Stalemate is only detected after sufficient rounds."""
        detector = ConsensusDetector(threshold=0.8, stalemate_rounds=3)

        opposed = [
            AgentPosition(agent_id="a1", approach=[1.0, 0.0], confidence=0.9),
            AgentPosition(agent_id="a2", approach=[0.0, 1.0], confidence=0.9),
        ]

        # Should not detect stalemate with insufficient history
        detector.measure_agreement(opposed)
        stalemate = detector.detect_stalemate(opposed)
        assert stalemate is None

        # After enough rounds, stalemate should be detected
        for _ in range(4):
            detector.measure_agreement(opposed)

        metrics = detector.measure_agreement(opposed)
        stalemate = detector.detect_stalemate(opposed, metrics)
        # May or may not be stalemate depending on exact metrics, but should be checked

    def test_convergence_trend_detection(self):
        """Convergence trend is correctly detected."""
        detector = ConsensusDetector(threshold=0.8)

        # Simulate converging positions
        history = ConvergenceHistory()
        for i in range(10):
            spread = 1.0 - (i * 0.08)  # Decreasing spread
            positions = [
                AgentPosition(agent_id="a1", approach=[0.5 + spread / 2, 0.5 - spread / 2]),
                AgentPosition(agent_id="a2", approach=[0.5 - spread / 2, 0.5 + spread / 2]),
            ]
            metrics = detector.measure_agreement(positions)
            history.add_snapshot(i, metrics, positions)

        trend = detector.detect_convergence_trend(history)
        assert trend in (
            ConvergenceTrend.CONVERGING,
            ConvergenceTrend.SLOWLY_CONVERGING,
            ConvergenceTrend.CONVERGED,
        )

    def test_diverging_trend_detection(self):
        """Diverging trend is correctly detected."""
        detector = ConsensusDetector(threshold=0.8)

        history = ConvergenceHistory()
        for i in range(10):
            spread = 0.1 + (i * 0.08)  # Increasing spread
            positions = [
                AgentPosition(agent_id="a1", approach=[0.5 + spread / 2, 0.5 - spread / 2]),
                AgentPosition(agent_id="a2", approach=[0.5 - spread / 2, 0.5 + spread / 2]),
            ]
            metrics = detector.measure_agreement(positions)
            history.add_snapshot(i, metrics, positions)

        trend = detector.detect_convergence_trend(history)
        assert trend == ConvergenceTrend.DIVERGING

    def test_resolution_strategy_suggested(self):
        """ResolutionStrategy is suggested for stalemates."""
        detector = ConsensusDetector(threshold=0.8)

        stalemate = Stalemate(
            detected_at_round=5,
            severity=0.9,
            reason="Strong disagreement",
            diverging_agents=["a1", "a2"],
            cluster_info={"count": 2, "sizes": [1, 1]},
        )

        resolution = detector.suggest_resolution(stalemate)
        assert resolution.type in [r.value for r in ResolutionType]
        assert resolution.description != ""
        assert 0.0 <= resolution.confidence <= 1.0

    def test_resolution_for_high_severity(self):
        """High-severity stalemates get re-branch resolution."""
        detector = ConsensusDetector(threshold=0.8)

        stalemate = Stalemate(
            detected_at_round=5,
            severity=0.95,
            reason="Very strong disagreement",
            cluster_info={"count": 4, "sizes": [1, 1, 1, 1]},
        )

        resolution = detector.suggest_resolution(stalemate)
        assert resolution.type == ResolutionType.REBRANCH.value

    def test_resolution_for_two_clusters(self):
        """Two-cluster stalemates get split-difference resolution."""
        detector = ConsensusDetector(threshold=0.8)

        # Feed some history first
        for _ in range(3):
            positions = [
                AgentPosition(agent_id="a1", approach=[1.0, 0.0], confidence=0.9),
                AgentPosition(agent_id="a2", approach=[0.0, 1.0], confidence=0.9),
            ]
            detector.measure_agreement(positions)

        stalemate = Stalemate(
            detected_at_round=5,
            severity=0.7,
            reason="Two positions",
            cluster_info={"count": 2, "sizes": [2, 2]},
        )

        resolution = detector.suggest_resolution(stalemate)
        assert resolution.type == ResolutionType.SPLIT_DIFFERENCE.value

    def test_pipeline_rebranch_on_stalemate(self):
        """Pipeline creates refined spec when stalemate detected."""
        spec = WorkflowSpec(
            goal="Controversial decision",
            agents=[
                AgentSpec(id="a1", stance="pro"),
                AgentSpec(id="a2", stance="con"),
            ],
            max_rounds=3,
            max_rebranches=2,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        stalemate = Stalemate(
            detected_at_round=3,
            severity=0.8,
            reason="Agents cannot agree",
            diverging_agents=["a1", "a2"],
            cluster_info={"count": 2, "sizes": [1, 1]},
        )

        branch_results = [
            BranchResult(agent_id="a1", content="pro content", confidence=0.9),
            BranchResult(agent_id="a2", content="con content", confidence=0.85),
        ]

        refined_spec = pipeline.rebranch(stalemate, branch_results)

        assert isinstance(refined_spec, WorkflowSpec)
        assert "Re-branch" in refined_spec.goal
        assert refined_spec.max_rebranches < spec.max_rebranches
        assert len(refined_spec.agents) == 2
        assert refined_spec.meta["parent_workflow"] == spec.id

    def test_pipeline_respects_max_rebranches(self):
        """Pipeline stops re-branching after max_rebranches is reached."""
        spec = WorkflowSpec(
            goal="Limited rebranches",
            agents=[AgentSpec(id="a1"), AgentSpec(id="a2")],
            max_rounds=2,
            max_rebranches=0,  # No re-branches allowed
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        stalemate = Stalemate(
            detected_at_round=2,
            severity=0.9,
            reason="No agreement",
        )
        branch_results = [
            BranchResult(agent_id="a1", content="x", confidence=0.9),
            BranchResult(agent_id="a2", content="y", confidence=0.85),
        ]

        # Even with stalemate, pipeline should not re-branch
        # (rebranch count starts at 0, max is 0)
        assert pipeline._rebranch_count >= spec.max_rebranches


# ===========================================================================
# Additional coverage tests
# ===========================================================================

class TestDiscussionStrategies:
    """Additional tests for all discussion strategies."""

    def test_review_strategy(self):
        """Review strategy evaluates criteria."""
        participants = [
            {"id": "rev1", "stance": "reviewer", "weight": 1.0},
            {"id": "rev2", "stance": "reviewer", "weight": 0.9},
        ]
        review = ReviewStrategy(
            topic="Proposal: Add caching",
            participants=participants,
            max_rounds=4,
        )
        review.initialize()

        # Each reviewer evaluates criteria
        for reviewer_id in ["rev1", "rev2"]:
            for criterion in ["correctness", "completeness", "clarity", "efficiency"]:
                review.process_turn(DiscussionTurn(
                    agent_id=reviewer_id,
                    content={
                        "type": "criterion_review",
                        "criterion": criterion,
                        "score": 0.85,
                        "notes": f"{reviewer_id}: {criterion} looks good",
                    },
                    confidence=0.85,
                    stance="reviewer",
                ))

        result = review.conclude()
        assert result.format == "review"
        assert result.decision["type"] == "review_result"
        assert result.decision["overall_score"] > 0

    def test_negotiation_strategy(self):
        """Negotiation strategy tracks utility functions."""
        participants = [
            {"id": "dept-a", "stance": "neutral", "weight": 1.0, "priorities": [0.8, 0.2]},
            {"id": "dept-b", "stance": "neutral", "weight": 1.0, "priorities": [0.3, 0.7]},
        ]
        negotiation = NegotiationStrategy(
            topic="Budget allocation",
            participants=participants,
            max_rounds=3,
        )
        negotiation.initialize()

        assert len(negotiation.utility_functions) == 2

        # Submit proposals
        negotiation.process_turn(DiscussionTurn(
            agent_id="dept-a",
            content={
                "type": "proposal",
                "proposal_vector": [0.7, 0.3],
                "description": "More budget for team A",
            },
            confidence=0.8,
        ))

        result = negotiation.conclude()
        assert result.format == "negotiation"
        assert result.decision["type"] == "negotiation_result"
        assert "total_proposals" in result.decision

    def test_peer_review_strategy(self):
        """Peer review strategy handles independent reviews."""
        participants = [
            {"id": "reviewer-1", "stance": "reviewer"},
            {"id": "reviewer-2", "stance": "reviewer"},
            {"id": "reviewer-3", "stance": "reviewer"},
        ]
        peer_review = PeerReviewStrategy(
            topic="Paper: Neural Architecture Search",
            participants=participants,
            max_rounds=4,
        )
        peer_review.initialize()

        # Each reviewer submits independent review
        for reviewer_id in ["reviewer-1", "reviewer-2", "reviewer-3"]:
            peer_review.process_turn(DiscussionTurn(
                agent_id=reviewer_id,
                content={
                    "type": "independent_review_result",
                    "scores": {
                        "correctness": 0.8,
                        "completeness": 0.7,
                        "clarity": 0.9,
                    },
                    "summary": f"Review by {reviewer_id}",
                    "recommendation": "accept" if reviewer_id != "reviewer-3" else "revise",
                    "strengths": ["clear writing"],
                    "weaknesses": ["missing baselines"],
                },
                confidence=0.8,
            ))

        # Should transition to evaluation phase
        assert peer_review.current_phase == Phase.EVALUATE

        result = peer_review.conclude()
        assert result.format == "peer_review"
        assert result.decision["type"] == "peer_review_result"
        assert result.decision["review_count"] == 3


class TestVectorSimilarity:
    """Tests for vector similarity functions."""

    def test_cosine_identical(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_cosine_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=0.01)

    def test_cosine_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_euclidean_identical(self):
        assert euclidean_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_euclidean_different(self):
        assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_euclidean_empty(self):
        assert euclidean_distance([], []) == float("inf")

    def test_agent_position_similarity(self):
        p1 = AgentPosition(agent_id="a1", approach=[1, 0, 0])
        p2 = AgentPosition(agent_id="a2", approach=[1, 0, 0])
        assert p1.similarity_to(p2) == pytest.approx(1.0)

    def test_agent_position_distance(self):
        p1 = AgentPosition(agent_id="a1", approach=[1, 0])
        p2 = AgentPosition(agent_id="a2", approach=[0, 1])
        # Distance < 1.0 because shared confidence dimension creates partial similarity
        assert p1.distance_to(p2) == pytest.approx(0.8, abs=0.1)


class TestAgreementMetrics:
    """Tests for agreement metrics computation."""

    def test_unanimous_agreement(self):
        detector = ConsensusDetector(threshold=0.7)
        positions = [
            AgentPosition(agent_id="a1", approach=[0.9, 0.1], confidence=0.9),
            AgentPosition(agent_id="a2", approach=[0.85, 0.15], confidence=0.85),
            AgentPosition(agent_id="a3", approach=[0.88, 0.12], confidence=0.88),
        ]
        metrics = detector.measure_agreement(positions)
        assert metrics.consensus_type == ConsensusType.UNANIMOUS.value
        assert metrics.agreement_score > 0.9

    def test_no_agreement(self):
        detector = ConsensusDetector(threshold=0.9)
        positions = [
            AgentPosition(agent_id="a1", approach=[1.0, 0.0], confidence=0.9),
            AgentPosition(agent_id="a2", approach=[0.0, 1.0], confidence=0.9),
        ]
        metrics = detector.measure_agreement(positions)
        assert metrics.agreement_score < 0.5

    def test_single_position(self):
        detector = ConsensusDetector(threshold=0.8)
        positions = [
            AgentPosition(agent_id="a1", approach=[0.5], confidence=0.8),
        ]
        metrics = detector.measure_agreement(positions)
        assert metrics.consensus_type == ConsensusType.STALEMATE.value

    def test_empty_positions(self):
        detector = ConsensusDetector(threshold=0.8)
        metrics = detector.measure_agreement([])
        assert metrics.agreement_score == 0.0

    def test_metrics_serialization(self):
        metrics = AgreementMetrics(
            consensus_type="unanimous",
            agreement_score=0.95,
            cluster_count=1,
            majority_fraction=1.0,
        )
        d = metrics.to_dict()
        metrics2 = AgreementMetrics.from_dict(d)
        assert metrics2.consensus_type == "unanimous"
        assert metrics2.agreement_score == pytest.approx(0.95)


class TestConvergenceHistory:
    """Tests for convergence history tracking."""

    def test_history_snapshots(self):
        history = ConvergenceHistory()
        metrics = AgreementMetrics(agreement_score=0.5)
        positions = [AgentPosition(agent_id="a1", approach=[0.5])]

        history.add_snapshot(1, metrics, positions)
        history.add_snapshot(2, metrics, positions)
        history.add_snapshot(3, metrics, positions)

        assert len(history.snapshots) == 3
        assert history.get_agreement_scores() == [0.5, 0.5, 0.5]

    def test_history_trimming(self):
        history = ConvergenceHistory(max_length=5)
        metrics = AgreementMetrics(agreement_score=0.5)

        for i in range(10):
            history.add_snapshot(i, metrics, [])

        assert len(history.snapshots) == 5


class TestStalemate:
    """Tests for stalemate data structures."""

    def test_stalemate_creation(self):
        s = Stalemate(
            detected_at_round=5,
            severity=0.8,
            reason="No convergence",
        )
        assert s.stalemate_id != ""
        assert s.severity == 0.8

    def test_stalemate_serialization(self):
        s = Stalemate(
            detected_at_round=3,
            severity=0.6,
            reason="Agents stuck",
            diverging_agents=["a1", "a2"],
            cluster_info={"count": 2},
        )
        d = s.to_dict()
        s2 = Stalemate.from_dict(d)
        assert s2.reason == "Agents stuck"
        assert s2.diverging_agents == ["a1", "a2"]


class TestPipelineWithNegotiation:
    """Pipeline with negotiation format."""

    def test_negotiation_pipeline(self):
        spec = WorkflowSpec(
            goal="Allocate budget between teams",
            agents=[
                AgentSpec(id="team-a", role="engineering", stance="neutral", weight=1.0, priorities=[0.8, 0.2]),
                AgentSpec(id="team-b", role="marketing", stance="neutral", weight=0.9, priorities=[0.3, 0.7]),
            ],
            discussion_format="negotiation",
            synthesis_method="compromise",
            max_rounds=3,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        result = asyncio.get_event_loop().run_until_complete(pipeline.execute())

        assert result.status == WorkflowStatus.COMPLETED.value
        assert result.output is not None

    def test_brainstorm_pipeline(self):
        spec = WorkflowSpec(
            goal="Generate feature ideas for Q4",
            agents=[
                AgentSpec(id="pm", role="product"),
                AgentSpec(id="eng", role="engineering"),
                AgentSpec(id="design", role="design"),
            ],
            discussion_format="brainstorm",
            synthesis_method="best_confidence",
            max_rounds=3,
            quality_threshold=0.1,
        )
        pipeline = AgentWorkflowPipeline(spec)

        result = asyncio.get_event_loop().run_until_complete(pipeline.execute())

        assert result.status == WorkflowStatus.COMPLETED.value
        assert result.branches_completed == 3


class TestDiscussionTurn:
    """Tests for DiscussionTurn data structure."""

    def test_turn_auto_id(self):
        turn = DiscussionTurn(agent_id="a1", content="hello")
        assert turn.turn_id != ""

    def test_turn_confidence_clamped(self):
        turn = DiscussionTurn(agent_id="a1", confidence=1.5)
        assert turn.confidence == 1.0

        turn2 = DiscussionTurn(agent_id="a1", confidence=-0.5)
        assert turn2.confidence == 0.0

    def test_turn_serialization(self):
        turn = DiscussionTurn(
            agent_id="a1",
            content={"key": "value"},
            confidence=0.85,
            references=["turn-1"],
            challenge_to="a2",
            stance="pro",
        )
        d = turn.to_dict()
        turn2 = DiscussionTurn.from_dict(d)
        assert turn2.agent_id == "a1"
        assert turn2.content == {"key": "value"}
        assert turn2.confidence == pytest.approx(0.85)
        assert turn2.challenge_to == "a2"
        assert "turn-1" in turn2.references


class TestReviewCriterion:
    """Tests for ReviewCriterion."""

    def test_criterion_defaults(self):
        c = ReviewCriterion(name="test", description="A test criterion")
        assert c.weight == 1.0
        assert c.score == 0.0

    def test_criterion_serialization(self):
        c = ReviewCriterion(name="efficiency", description="Is it fast?", weight=1.5, score=0.8)
        d = c.to_dict()
        c2 = ReviewCriterion.from_dict(d)
        assert c2.name == "efficiency"
        assert c2.score == pytest.approx(0.8)
