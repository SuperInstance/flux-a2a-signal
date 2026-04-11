"""
FLUX-A2A Agent Causality Model (Rounds 10-11).

When agents branch, fork, co-iterate, and merge, there is a CAUSAL STRUCTURE
to their interactions. Traditional programming has sequential causality (A before B).
Agent systems have CONCURRENT causality (A and B may happen in any order, but the
result must be deterministic).

This module implements:

  - AgentCausalGraph: directed graph tracking data/control dependencies between
    agent actions. Nodes are (agent_id, action_id) pairs; edges carry labels
    describing the type of dependency (data_flow, control, trust, merge_barrier).

  - CausalEdgeType: enumeration of edge semantics — data flow, control dependency,
    trust delegation, merge barrier (synchronisation), and conflict resolution.

  - CausalAnalysisResult: result of analysing a graph for parallelisable groups,
    sequential ordering, cycle detection, and causal distance metrics.

  - CausalLamportClock: a vector-clock-inspired mechanism for assigning partial
    order timestamps to agent events in a concurrent system.

Design principles:
  - The graph is a DAG when well-formed; cycles represent bugs (deadlocks).
  - `parallel_groups()` computes maximal independent sets — agents with no
    causal path between them MAY execute concurrently.
  - `sequential_order()` computes a topological sort — agents with a causal
    path MUST execute in this order.
  - `detect_cycle()` finds the shortest cycle — the deadlock ring.
  - `causal_distance()` measures the length of the shortest causal path.

Mathematical foundations:
  - Concurrent causality is a partial order: ≺ on events where e₁ ≺ e₂ means
    "e₂ depends on e₁". Two events are concurrent (e₁ ∥ e₂) iff neither
    e₁ ≺ e₂ nor e₂ ≺ e₁.
  - This is Lamport's happens-before relation extended to agent systems.
  - The graph can express FORK causality: parent → child (inherit), and
    MERGE causality: child₁, child₂ → parent (barrier).
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ===========================================================================
# Enums
# ===========================================================================

class CausalEdgeType(str, Enum):
    """Type of causal dependency between agent actions."""
    DATA_FLOW = "data_flow"           # Agent B consumes Agent A's output
    CONTROL = "control"               # Agent B executes after Agent A decides
    TRUST_DELEGATION = "trust"        # Agent B acts on Agent A's trust grant
    MERGE_BARRIER = "merge_barrier"   # Merge synchronisation: wait for all
    FORK_INHERIT = "fork_inherit"     # Fork: child inherits parent state
    CONFLICT_RESOLVE = "conflict"     # Conflict resolution dependency
    SESSION_CHANNEL = "session"       # Session-typed channel communication
    TEMPORAL_ORDER = "temporal"       # LTL/CTL temporal ordering constraint


class CausalGraphStatus(str, Enum):
    """Health status of a causal graph."""
    HEALTHY = "healthy"               # DAG — no cycles
    CYCLIC = "cyclic"                 # Contains cycles (potential deadlock)
    DISCONNECTED = "disconnected"     # Has isolated components
    UNKNOWN = "unknown"


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass(slots=True)
class CausalEdge:
    """
    A directed edge in the causal graph: source → target.

    Attributes:
        source: Source (agent_id, action_id) pair.
        target: Target (agent_id, action_id) pair.
        edge_type: Semantic type of the dependency.
        label: Human-readable description.
        weight: Dependency strength (0.0–1.0), for weighted analysis.
        timestamp: When this dependency was created.
    """
    source: tuple[str, str]
    target: tuple[str, str]
    edge_type: str = CausalEdgeType.DATA_FLOW.value
    label: str = ""
    weight: float = 1.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "target": list(self.target),
            "edge_type": self.edge_type,
            "label": self.label,
            "weight": self.weight,
            "timestamp": self.timestamp,
        }


@dataclass
class CausalNode:
    """
    A node in the causal graph: (agent_id, action_id).

    Attributes:
        agent_id: Identity of the agent performing the action.
        action_id: Unique identifier for the action within the agent's execution.
        action_type: What kind of action (branch, fork, co_iterate, discuss, etc.).
        status: Current status of the action.
        metadata: Extensible metadata dict.
    """
    agent_id: str
    action_id: str
    action_type: str = "compute"
    status: str = "pending"  # pending | executing | completed | failed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.agent_id, self.action_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class CausalAnalysisResult:
    """
    Result of analysing a causal graph.

    Attributes:
        status: Overall health of the graph.
        parallel_groups: Maximal independent sets of agents/actions.
        sequential_order: Topological sort of all actions.
        cycle: Shortest detected cycle (None if DAG).
        causal_distances: Pairwise shortest causal distances.
        critical_path: Longest path through the DAG (bottleneck).
        max_parallelism: Theoretical maximum number of concurrent agents.
        total_actions: Total number of actions in the graph.
        total_edges: Total number of causal edges.
    """
    status: str = CausalGraphStatus.UNKNOWN.value
    parallel_groups: list[set[str]] = field(default_factory=list)
    sequential_order: list[str] = field(default_factory=list)
    cycle: Optional[list[str]] = None
    causal_distances: dict[tuple[str, str], int] = field(default_factory=dict)
    critical_path: list[str] = field(default_factory=list)
    max_parallelism: int = 0
    total_actions: int = 0
    total_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "parallel_groups": [list(g) for g in self.parallel_groups],
            "sequential_order": self.sequential_order,
            "cycle": self.cycle,
            "causal_distances": {f"{k[0]}:{k[1]}": v for k, v in self.causal_distances.items()},
            "critical_path": self.critical_path,
            "max_parallelism": self.max_parallelism,
            "total_actions": self.total_actions,
            "total_edges": self.total_edges,
        }


# ===========================================================================
# Vector Clock for Concurrent Causality
# ===========================================================================

@dataclass
class CausalLamportClock:
    """
    Vector clock for assigning partial-order timestamps to agent events.

    Unlike a single Lamport clock (which gives a total order), vector clocks
    capture the partial order: event A happens-before event B iff
    A.clock ≤ B.clock (component-wise) and A.clock ≠ B.clock.

    This is essential for agent systems where agents operate concurrently
    and only synchronize through explicit message passing or merge barriers.
    """
    _clocks: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def tick(self, agent_id: str) -> int:
        """Increment the clock for the given agent; return new value."""
        self._clocks[agent_id] = self._clocks.get(agent_id, 0) + 1
        return self._clocks[agent_id]

    def merge(self, other: CausalLamportClock) -> None:
        """Merge another vector clock into this one (take component-wise max)."""
        for agent_id, value in other._clocks.items():
            self._clocks[agent_id] = max(self._clocks.get(agent_id, 0), value)

    def happens_before(self, other: CausalLamportClock) -> bool:
        """True iff self ≺ other (self happened before other)."""
        all_agents = set(self._clocks.keys()) | set(other._clocks.keys())
        at_least_one_less = False
        for agent_id in all_agents:
            s = self._clocks.get(agent_id, 0)
            o = other._clocks.get(agent_id, 0)
            if s > o:
                return False
            if s < o:
                at_least_one_less = True
        return at_least_one_less

    def is_concurrent(self, other: CausalLamportClock) -> bool:
        """True iff self ∥ other (neither happened before the other)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def get_clock(self, agent_id: str) -> int:
        return self._clocks.get(agent_id, 0)

    def snapshot(self) -> dict[str, int]:
        return dict(self._clocks)

    def to_dict(self) -> dict[str, Any]:
        return {"clocks": dict(self._clocks)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalLamportClock:
        vc = cls()
        vc._clocks = dict(data.get("clocks", {}))
        return vc


# ===========================================================================
# Agent Causal Graph — The Core
# ===========================================================================

class AgentCausalGraph:
    """
    Causal graph tracking dependencies between agent actions.

    When Agent A produces output that Agent B consumes, there's a causal edge.
    When agents run in parallel with no shared state, there's no edge.
    The graph determines what can be parallelized and what must be sequential.

    Usage:
        g = AgentCausalGraph()
        g.add_agent("agent_a")
        g.add_agent("agent_b")
        g.add_agent("agent_c")
        g.add_causal_edge("agent_a", "agent_b", "data_flow", "A feeds B")
        g.add_causal_edge("agent_a", "agent_c", "data_flow", "A feeds C")
        result = g.analyze()
        assert result.parallel_groups == [{"agent_b"}, {"agent_c"}]
        assert result.sequential_order == ["agent_a", "agent_b", "agent_c"]
    """

    def __init__(self) -> None:
        # agent_id -> list of CausalNode (actions for this agent)
        self._agents: dict[str, list[CausalNode]] = defaultdict(list)
        # agent_id -> set of agent_ids this agent depends on
        self._adj: dict[str, set[str]] = defaultdict(set)
        # agent_id -> set of agent_ids that depend on this agent
        self._reverse_adj: dict[str, set[str]] = defaultdict(set)
        # All edges with metadata
        self._edges: list[CausalEdge] = []
        # Vector clock for each agent
        self._clocks: dict[str, CausalLamportClock] = {}

    # -- Agent Management ---------------------------------------------------

    def add_agent(self, agent_id: str) -> None:
        """Register an agent in the causal graph."""
        if agent_id not in self._agents:
            self._agents[agent_id] = []
            self._clocks[agent_id] = CausalLamportClock()

    def add_action(self, agent_id: str, action_id: str,
                   action_type: str = "compute",
                   metadata: Optional[dict[str, Any]] = None) -> CausalNode:
        """
        Add an action for an agent and tick its vector clock.

        Returns the created CausalNode.
        """
        self.add_agent(agent_id)
        node = CausalNode(
            agent_id=agent_id,
            action_id=action_id,
            action_type=action_type,
            status="pending",
            metadata=metadata or {},
        )
        self._agents[agent_id].append(node)
        self._clocks[agent_id].tick(agent_id)
        return node

    def get_actions(self, agent_id: str) -> list[CausalNode]:
        """Get all actions for a given agent."""
        return list(self._agents.get(agent_id, []))

    def get_agents(self) -> list[str]:
        """Get all registered agent IDs."""
        return list(self._agents.keys())

    # -- Edge Management ----------------------------------------------------

    def add_causal_edge(self, source: str, target: str,
                        edge_type: str = CausalEdgeType.DATA_FLOW.value,
                        label: str = "",
                        weight: float = 1.0) -> CausalEdge:
        """
        Add a causal dependency: source must complete before target can begin.

        Args:
            source: Source agent ID (must happen first).
            target: Target agent ID (depends on source).
            edge_type: Type of dependency.
            label: Human-readable description.
            weight: Dependency strength (0.0–1.0).

        Returns:
            The created CausalEdge.

        Raises:
            ValueError: If source == target (self-loop).
        """
        if source == target:
            raise ValueError(f"Self-loop not allowed: {source} → {target}")

        self.add_agent(source)
        self.add_agent(target)

        edge = CausalEdge(
            source=(source, self._clocks[source].get_clock(source)),
            target=(target, self._clocks[target].get_clock(target)),
            edge_type=edge_type,
            label=label or f"{source} → {target}",
            weight=weight,
        )

        self._adj[source].add(target)
        self._reverse_adj[target].add(source)
        self._edges.append(edge)

        # Propagate vector clock: target must have seen source's state
        self._clocks[target].merge(self._clocks[source])

        return edge

    def add_fork_edge(self, parent: str, child: str,
                      label: str = "") -> CausalEdge:
        """Add a fork inheritance edge: child inherits from parent."""
        return self.add_causal_edge(
            parent, child,
            CausalEdgeType.FORK_INHERIT.value,
            label or f"fork: {parent} → {child}",
        )

    def add_merge_barrier(self, children: list[str], parent: str,
                          label: str = "") -> list[CausalEdge]:
        """
        Add merge barrier edges: parent waits for ALL children.

        Returns the list of created edges.
        """
        edges = []
        for child in children:
            e = self.add_causal_edge(
                child, parent,
                CausalEdgeType.MERGE_BARRIER.value,
                label or f"merge: {child} → {parent}",
            )
            edges.append(e)
        return edges

    def add_data_flow(self, source: str, target: str,
                      label: str = "") -> CausalEdge:
        """Add a data flow edge: target consumes source's output."""
        return self.add_causal_edge(
            source, target,
            CausalEdgeType.DATA_FLOW.value,
            label or f"data: {source} → {target}",
        )

    def add_session_channel(self, sender: str, receiver: str,
                            label: str = "") -> CausalEdge:
        """Add a session-typed channel edge."""
        return self.add_causal_edge(
            sender, receiver,
            CausalEdgeType.SESSION_CHANNEL.value,
            label or f"session: {sender} → {receiver}",
        )

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent and all its edges from the graph."""
        # Remove from adjacency lists
        for dep in self._adj.pop(agent_id, set()):
            self._reverse_adj[dep].discard(agent_id)
        for dep in self._reverse_adj.pop(agent_id, set()):
            self._adj[dep].discard(agent_id)
        # Remove edges
        self._edges = [
            e for e in self._edges
            if e.source[0] != agent_id and e.target[0] != agent_id
        ]
        # Remove agent data
        self._agents.pop(agent_id, None)
        self._clocks.pop(agent_id, None)

    # -- Analysis -----------------------------------------------------------

    def detect_cycle(self) -> Optional[list[str]]:
        """
        Detect cycles in the causal graph using Kahn's algorithm.

        Returns:
            A list of agent IDs forming the shortest cycle, or None if the
            graph is a DAG.
        """
        # Build in-degree map
        in_degree: dict[str, int] = defaultdict(int)
        for agent_id in self._agents:
            if agent_id not in in_degree:
                in_degree[agent_id] = 0
            for dep in self._adj.get(agent_id, set()):
                in_degree[dep] += 1

        # Kahn's: remove nodes with in-degree 0
        queue: deque[str] = deque(
            a for a in self._agents if in_degree[a] == 0
        )
        sorted_count = 0
        while queue:
            node = queue.popleft()
            sorted_count += 1
            for dep in self._adj.get(node, set()):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        if sorted_count == len(self._agents):
            return None  # DAG

        # Find shortest cycle using BFS from each unsorted node
        remaining = [a for a in self._agents if in_degree[a] > 0]
        if not remaining:
            return None

        shortest: Optional[list[str]] = None
        for start in remaining:
            # BFS to find shortest path back to start
            visited: dict[str, Optional[str]] = {start: None}
            bfs_queue: deque[str] = deque([start])
            found = False
            while bfs_queue and not found:
                current = bfs_queue.popleft()
                for neighbor in self._adj.get(current, set()):
                    if neighbor == start:
                        # Reconstruct path
                        path = [start]
                        node: Optional[str] = current
                        while node is not None and node != start:
                            path.append(node)
                            node = visited[node]
                        path.reverse()
                        if shortest is None or len(path) < len(shortest):
                            shortest = path
                        found = True
                        break
                    if neighbor not in visited:
                        visited[neighbor] = current
                        bfs_queue.append(neighbor)

        return shortest

    def sequential_order(self) -> list[str]:
        """
        Compute a topological sort of the causal graph.

        Returns:
            Agent IDs in a valid execution order (dependencies first).
            Returns empty list if the graph has cycles.
        """
        cycle = self.detect_cycle()
        if cycle is not None:
            return []

        # Kahn's algorithm
        in_degree: dict[str, int] = {a: 0 for a in self._agents}
        for agent_id in self._agents:
            for dep in self._adj.get(agent_id, set()):
                in_degree[dep] = in_degree.get(dep, 0) + 1

        queue: deque[str] = deque(
            a for a in self._agents if in_degree[a] == 0
        )
        result: list[str] = []
        while queue:
            # Prioritize agents with more dependents (greedy critical path)
            node = max(queue, key=lambda a: len(self._adj.get(a, set())))
            queue.remove(node)
            result.append(node)
            for dep in self._adj.get(node, set()):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        return result

    def parallel_groups(self) -> list[set[str]]:
        """
        Compute maximal sets of agents that can execute concurrently.

        Two agents can run in parallel iff there is NO causal path
        in either direction between them (they are concurrent in the
        partial order sense: A ∥ B).

        Returns:
            List of sets, where each set is a maximal independent set
            of agents that can run concurrently.
        """
        if self.detect_cycle() is not None:
            return []

        topo = self.sequential_order()
        if not topo:
            return []

        # Compute reachability (transitive closure)
        reachable: dict[str, set[str]] = defaultdict(set)
        for agent_id in reversed(topo):
            for dep in self._adj.get(agent_id, set()):
                reachable[agent_id].add(dep)
                reachable[agent_id].update(reachable[dep])

        # Group agents by their "dependency depth" — all agents at the
        # same depth have no mutual causal dependencies
        depth_map: dict[str, int] = {}
        for agent_id in topo:
            deps = self._reverse_adj.get(agent_id, set())
            if not deps:
                depth_map[agent_id] = 0
            else:
                depth_map[agent_id] = max(depth_map.get(d, 0) for d in deps) + 1

        # Group by depth
        depth_groups: dict[int, set[str]] = defaultdict(set)
        for agent_id, depth in depth_map.items():
            depth_groups[depth].add(agent_id)

        return [depth_groups[d] for d in sorted(depth_groups.keys())]

    def causal_distance(self, a: str, b: str) -> int:
        """
        Shortest causal path length from agent A to agent B.

        Returns:
            Number of edges in the shortest path, or -1 if unreachable.
        """
        if a == b:
            return 0

        visited: set[str] = {a}
        queue: deque[tuple[str, int]] = deque([(a, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self._adj.get(current, set()):
                if neighbor == b:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1

    def causal_distance_all_pairs(self) -> dict[tuple[str, str], int]:
        """Compute all-pairs shortest causal distances."""
        agents = self.get_agents()
        distances: dict[tuple[str, str], int] = {}
        for a in agents:
            for b in agents:
                d = self.causal_distance(a, b)
                if d >= 0:
                    distances[(a, b)] = d
        return distances

    def critical_path(self) -> list[str]:
        """
        Find the longest path through the DAG (the bottleneck).

        This is the sequence of agents whose total causal distance is
        maximized — the critical path that determines the minimum
        execution time.
        """
        if self.detect_cycle() is not None:
            return []

        topo = self.sequential_order()
        if not topo:
            return []

        # DP for longest path
        longest: dict[str, int] = {a: 0 for a in topo}
        predecessor: dict[str, Optional[str]] = {a: None for a in topo}

        for agent_id in topo:
            for dep in self._adj.get(agent_id, set()):
                if longest[dep] + 1 > longest[agent_id]:
                    longest[agent_id] = longest[dep] + 1
                    predecessor[agent_id] = dep

        # Find the end of the critical path
        if not longest:
            return []
        end = max(longest, key=longest.get)

        # Reconstruct
        path: list[str] = []
        current: Optional[str] = end
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()
        return path

    def max_parallelism(self) -> int:
        """Theoretical maximum number of concurrently executable agents."""
        groups = self.parallel_groups()
        if not groups:
            return 0
        return max(len(g) for g in groups)

    def predecessors(self, agent_id: str) -> set[str]:
        """Get all agents that agent_id depends on (direct)."""
        return set(self._reverse_adj.get(agent_id, set()))

    def successors(self, agent_id: str) -> set[str]:
        """Get all agents that depend on agent_id (direct)."""
        return set(self._adj.get(agent_id, set()))

    def all_predecessors(self, agent_id: str) -> set[str]:
        """Get ALL agents that agent_id transitively depends on."""
        visited: set[str] = set()
        stack: list[str] = list(self._reverse_adj.get(agent_id, set()))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self._reverse_adj.get(current, set()))
        return visited

    def all_successors(self, agent_id: str) -> set[str]:
        """Get ALL agents that transitively depend on agent_id."""
        visited: set[str] = set()
        stack: list[str] = list(self._adj.get(agent_id, set()))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self._adj.get(current, set()))
        return visited

    def are_concurrent(self, a: str, b: str) -> bool:
        """True iff agents A and B are concurrent (no causal path either way)."""
        return self.causal_distance(a, b) < 0 and self.causal_distance(b, a) < 0

    # -- Full Analysis ------------------------------------------------------

    def analyze(self) -> CausalAnalysisResult:
        """
        Perform full causal analysis of the graph.

        Returns:
            CausalAnalysisResult with parallel groups, sequential order,
            cycle detection, distances, and critical path.
        """
        cycle = self.detect_cycle()

        if cycle is not None:
            return CausalAnalysisResult(
                status=CausalGraphStatus.CYCLIC.value,
                cycle=cycle,
                total_actions=sum(len(actions) for actions in self._agents.values()),
                total_edges=len(self._edges),
            )

        topo = self.sequential_order()
        groups = self.parallel_groups()
        distances = self.causal_distance_all_pairs()
        crit = self.critical_path()

        return CausalAnalysisResult(
            status=CausalGraphStatus.HEALTHY.value,
            parallel_groups=groups,
            sequential_order=topo,
            causal_distances=distances,
            critical_path=crit,
            max_parallelism=self.max_parallelism(),
            total_actions=sum(len(actions) for actions in self._agents.values()),
            total_edges=len(self._edges),
        )

    # -- Visualization / Serialization --------------------------------------

    def to_ascii(self) -> str:
        """Generate ASCII representation of the causal graph."""
        lines = ["AgentCausalGraph"]
        lines.append(f"  Agents: {self.get_agents()}")
        lines.append(f"  Edges: {len(self._edges)}")

        for edge in self._edges:
            src = edge.source[0]
            tgt = edge.target[0]
            arrow = {
                CausalEdgeType.DATA_FLOW: "═>",
                CausalEdgeType.CONTROL: "──>",
                CausalEdgeType.TRUST_DELEGATION: "⇢",
                CausalEdgeType.MERGE_BARRIER: "⟵",
                CausalEdgeType.FORK_INHERIT: "─┬>",
                CausalEdgeType.CONFLICT_RESOLVE: "↔",
                CausalEdgeType.SESSION_CHANNEL: "⟿",
                CausalEdgeType.TEMPORAL_ORDER: "⇢",
            }.get(edge.edge_type, "→")
            lines.append(f"  {src} {arrow} {tgt}  [{edge.edge_type}] {edge.label}")

        result = self.analyze()
        lines.append(f"\n  Status: {result.status}")
        lines.append(f"  Critical path: {' → '.join(result.critical_path)}")
        lines.append(f"  Max parallelism: {result.max_parallelism}")
        for i, group in enumerate(result.parallel_groups):
            lines.append(f"  Wave {i}: {sorted(group)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the causal graph to a dictionary."""
        analysis = self.analyze()
        return {
            "agents": {
                aid: [n.to_dict() for n in nodes]
                for aid, nodes in self._agents.items()
            },
            "edges": [e.to_dict() for e in self._edges],
            "clocks": {
                aid: vc.snapshot() for aid, vc in self._clocks.items()
            },
            "analysis": analysis.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCausalGraph:
        """Deserialize a causal graph from a dictionary."""
        graph = cls()
        for aid in data.get("agents", {}):
            graph.add_agent(aid)
        for edge_data in data.get("edges", {}):
            src = edge_data["source"][0] if isinstance(edge_data["source"], list) else edge_data["source"]
            tgt = edge_data["target"][0] if isinstance(edge_data["target"], list) else edge_data["target"]
            graph.add_causal_edge(
                src, tgt,
                edge_type=edge_data.get("edge_type", CausalEdgeType.DATA_FLOW.value),
                label=edge_data.get("label", ""),
                weight=edge_data.get("weight", 1.0),
            )
        return graph


# ===========================================================================
# Convenience: Build from FLUX primitives
# ===========================================================================

def build_causal_graph_from_branch(
    branch_id: str,
    branch_agents: list[str],
    merge_target: Optional[str] = None,
    strategy: str = "parallel",
) -> AgentCausalGraph:
    """
    Build a causal graph from a BranchPrimitive-like structure.

    Args:
        branch_id: The branch point ID.
        branch_agents: Agent IDs for each branch arm.
        merge_target: Agent ID that receives merged results (if any).
        strategy: "parallel" | "sequential" | "competitive".

    Returns:
        A causal graph encoding the branch dependencies.
    """
    graph = AgentCausalGraph()
    # In a branch, all arms are forked from the branch point
    graph.add_agent(branch_id)

    for agent_id in branch_agents:
        graph.add_fork_edge(branch_id, agent_id)

    if strategy == "sequential":
        # Sequential: each branch depends on the previous
        for i in range(1, len(branch_agents)):
            graph.add_causal_edge(
                branch_agents[i - 1], branch_agents[i],
                CausalEdgeType.CONTROL.value,
                f"sequential: arm {i-1} before arm {i}",
            )

    if merge_target:
        graph.add_merge_barrier(branch_agents, merge_target,
                                f"merge: all branches → {merge_target}")

    return graph


def build_causal_graph_from_co_iterate(
    co_iterate_id: str,
    agents: list[str],
    rounds: Any = "until_convergence",
    shared_state: str = "conflict",
) -> AgentCausalGraph:
    """
    Build a causal graph from a CoIterate-like structure.

    In co-iteration, all agents traverse the same program but conflicts
    may create temporal dependencies.

    Args:
        co_iterate_id: The co-iteration session ID.
        agents: Participating agent IDs.
        rounds: Number of rounds or "until_convergence".
        shared_state: How state is shared ("conflict"|"merge"|"partitioned"|"isolated").
    """
    graph = AgentCausalGraph()
    graph.add_agent(co_iterate_id)

    for agent_id in agents:
        graph.add_fork_edge(co_iterate_id, agent_id)

    # If shared state is "conflict", conflict resolution creates
    # synchronization edges between agents at each round
    if shared_state == "conflict" and len(agents) > 1:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                graph.add_causal_edge(
                    agents[i], agents[j],
                    CausalEdgeType.CONFLICT_RESOLVE.value,
                    f"conflict: {agents[i]} ↔ {agents[j]}",
                )

    # All agents must complete before convergence check
    graph.add_merge_barrier(agents, co_iterate_id,
                            f"convergence_check: all → {co_iterate_id}")

    return graph


def build_causal_graph_from_discussion(
    discussion_id: str,
    participants: list[str],
    format_type: str = "debate",
    turn_order: str = "round_robin",
) -> AgentCausalGraph:
    """
    Build a causal graph from a Discuss-like structure.

    In a discussion, turns are ordered by the turn_order policy.
    Round-robin creates a chain; moderated creates a star from the moderator.
    """
    graph = AgentCausalGraph()
    graph.add_agent(discussion_id)

    if turn_order == "moderated" and participants:
        # Star topology: moderator controls all turns
        moderator = participants[0]
        graph.add_fork_edge(discussion_id, moderator)
        for participant in participants[1:]:
            graph.add_fork_edge(discussion_id, participant)
            graph.add_causal_edge(
                moderator, participant,
                CausalEdgeType.CONTROL.value,
                f"moderator grants turn to {participant}",
            )
            graph.add_causal_edge(
                participant, moderator,
                CausalEdgeType.SESSION_CHANNEL.value,
                f"{participant} responds to moderator",
            )
    elif turn_order == "round_robin":
        # Chain topology: each participant takes a turn after the previous
        for i, participant in enumerate(participants):
            graph.add_fork_edge(discussion_id, participant)
            if i > 0:
                graph.add_causal_edge(
                    participants[i - 1], participant,
                    CausalEdgeType.SESSION_CHANNEL.value,
                    f"turn: {participants[i-1]} → {participant}",
                )

    # All participants must complete before discussion ends
    graph.add_merge_barrier(participants, discussion_id,
                            f"discussion_end: all → {discussion_id}")

    return graph
