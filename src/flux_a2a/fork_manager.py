"""
FLUX-A2A Branching and Forking Manager.

Handles:
  - BranchPoint: a point where execution can split into parallel paths
  - ForkContext: child execution context with inherited state
  - BranchManager: track and manage all active branches
  - ForkManager: track and manage all active forks
  - ForkTree: tree visualization of all branches/forks
  - MergePolicy: how branches rejoin
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from flux_a2a.schema import (
    Agent,
    BranchDef,
    ConfidenceScore,
    Expression,
    MergePolicy,
    MergePolicyType,
    Result,
)


# ---------------------------------------------------------------------------
# Merge policy types
# ---------------------------------------------------------------------------

class MergePolicyTypeExt(str, Enum):
    """Extended merge policy types."""
    LAST_WRITER_WINS = "last_writer_wins"
    CONSENSUS = "consensus"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    FIRST_COMPLETE = "first_complete"
    BEST_CONFIDENCE = "best_confidence"
    VOTE = "vote"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Branch point — a point where execution splits
# ---------------------------------------------------------------------------

@dataclass
class BranchPoint:
    """
    A point in execution where the program splits into multiple parallel
    branches.

    Each branch has a label, weight, and body.  When executed, all branches
    run (in simulated parallelism) and results are merged according to the
    merge policy.
    """
    id: str
    branches: list[BranchDef] = field(default_factory=list)
    merge_policy: MergePolicy = field(default_factory=MergePolicy)
    status: str = "pending"  # pending | executing | merging | completed | failed
    parent_id: str = ""
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            from datetime import datetime, timezone
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def branch_labels(self) -> list[str]:
        return [b.label for b in self.branches]

    def get_branch(self, label: str) -> Optional[BranchDef]:
        for b in self.branches:
            if b.label == label:
                return b
        return None

    def total_weight(self) -> float:
        return sum(b.weight for b in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "branches": [b.to_dict() for b in self.branches],
            "merge_policy": self.merge_policy.to_dict(),
            "status": self.status,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# Branch result — result from a single branch execution
# ---------------------------------------------------------------------------

@dataclass
class BranchResult:
    """The result of executing a single branch."""
    branch_id: str
    label: str
    result: Result
    weight: float = 1.0
    duration_ms: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            from datetime import datetime, timezone
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Branch manager — track and manage all active branches
# ---------------------------------------------------------------------------

@dataclass
class BranchManager:
    """Track and manage all active branches in an execution context."""

    _branches: OrderedDict[str, BranchPoint] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _results: dict[str, list[BranchResult]] = field(
        default_factory=dict, init=False, repr=False
    )

    def create_branch_point(
        self,
        branch_id: str,
        branches: list[BranchDef],
        merge_policy: Optional[MergePolicy] = None,
        parent_id: str = "",
    ) -> BranchPoint:
        """Create a new branch point and register it."""
        bp = BranchPoint(
            id=branch_id,
            branches=branches,
            merge_policy=merge_policy or MergePolicy(),
            parent_id=parent_id,
        )
        self._branches[branch_id] = bp
        self._results[branch_id] = []
        return bp

    def get_branch_point(self, branch_id: str) -> Optional[BranchPoint]:
        """Get a branch point by ID."""
        return self._branches.get(branch_id)

    def record_result(self, branch_id: str, label: str, result: Result, weight: float = 1.0) -> None:
        """Record the result of executing a branch."""
        if branch_id not in self._results:
            self._results[branch_id] = []
        self._results[branch_id].append(BranchResult(
            branch_id=branch_id,
            label=label,
            result=result,
            weight=weight,
        ))

    def merge(
        self,
        branch_id: str,
        branch_results: list[tuple[str, Result]],
    ) -> Result:
        """
        Merge branch results according to the branch point's merge policy.

        Args:
            branch_id: The branch point ID.
            branch_results: List of (label, result) tuples.

        Returns:
            The merged Result.
        """
        bp = self._branches.get(branch_id)
        if bp is None:
            return Result(value=None, confidence=0.0, source="merge", error=f"Unknown branch: {branch_id}")

        bp.status = "merging"

        # Record all results
        for label, result in branch_results:
            branch_def = bp.get_branch(label)
            weight = branch_def.weight if branch_def else 1.0
            self.record_result(branch_id, label, result, weight)

        results = self._results[branch_id]
        if not results:
            bp.status = "failed"
            return Result(value=None, confidence=0.0, source="merge", error="No results")

        strategy = bp.merge_policy.strategy

        # Apply merge strategy
        merged = self._apply_strategy(strategy, results, bp)

        bp.status = "completed"
        from datetime import datetime, timezone
        bp.completed_at = datetime.now(timezone.utc).isoformat()

        return merged

    def _apply_strategy(
        self,
        strategy: str,
        results: list[BranchResult],
        bp: BranchPoint,
    ) -> Result:
        """Apply a merge strategy to a list of branch results."""

        if strategy == MergePolicyType.LAST_WRITER_WINS.value:
            last = results[-1]
            return Result(
                value=last.result.value,
                confidence=last.result.confidence,
                source="merge:last_writer",
                children=[r.result for r in results],
                meta={"winning_branch": last.label},
            )

        elif strategy == MergePolicyType.FIRST_COMPLETE.value:
            first = results[0]
            return Result(
                value=first.result.value,
                confidence=first.result.confidence,
                source="merge:first_complete",
                children=[r.result for r in results],
                meta={"winning_branch": first.label},
            )

        elif strategy == MergePolicyType.BEST_CONFIDENCE.value:
            best = max(results, key=lambda r: r.result.confidence)
            return Result(
                value=best.result.value,
                confidence=best.result.confidence,
                source="merge:best_confidence",
                children=[r.result for r in results],
                meta={"winning_branch": best.label, "best_confidence": best.result.confidence},
            )

        elif strategy == MergePolicyType.CONSENSUS.value:
            values = [r.result.value for r in results]
            # Check if all values agree
            if all(v == values[0] for v in values):
                min_conf = min(r.result.confidence for r in results)
                return Result(
                    value=values[0],
                    confidence=min_conf,
                    source="merge:consensus",
                    children=[r.result for r in results],
                    meta={"agreement": True},
                )
            else:
                # Disagreement — return all values
                return Result(
                    value=values,
                    confidence=0.5,
                    source="merge:consensus:disagree",
                    children=[r.result for r in results],
                    meta={"agreement": False},
                )

        elif strategy == MergePolicyType.WEIGHTED_CONFIDENCE.value:
            total_weight = sum(r.weight for r in results)
            if total_weight == 0:
                total_weight = len(results)
            weighted_conf = sum(r.weight * r.result.confidence for r in results) / total_weight
            return Result(
                value=results[0].result.value,  # Representative value
                confidence=weighted_conf,
                source="merge:weighted_confidence",
                children=[r.result for r in results],
                meta={
                    "weighted_confidence": weighted_conf,
                    "branches": [(r.label, r.weight) for r in results],
                },
            )

        elif strategy == MergePolicyType.VOTE.value:
            from collections import Counter
            values = [r.result.value for r in results]
            counts = Counter(values)
            winner, count = counts.most_common(1)[0]
            return Result(
                value=winner,
                confidence=count / len(values),
                source="merge:vote",
                children=[r.result for r in results],
                meta={"winner": winner, "votes": count, "total": len(values)},
            )

        else:
            # Default: last writer wins
            last = results[-1]
            return Result(
                value=last.result.value,
                confidence=last.result.confidence,
                source="merge:default",
                children=[r.result for r in results],
            )

    def get_all_results(self, branch_id: str) -> list[BranchResult]:
        """Get all results for a branch point."""
        return list(self._results.get(branch_id, []))

    def get_active_branches(self) -> list[BranchPoint]:
        """Get all branches that are not yet completed."""
        return [bp for bp in self._branches.values() if bp.status in ("pending", "executing")]

    def get_completed_branches(self) -> list[BranchPoint]:
        """Get all completed branch points."""
        return [bp for bp in self._branches.values() if bp.status == "completed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": {bid: bp.to_dict() for bid, bp in self._branches.items()},
            "active_count": len(self.get_active_branches()),
            "completed_count": len(self.get_completed_branches()),
        }


# ---------------------------------------------------------------------------
# Fork context — child execution context with inherited state
# ---------------------------------------------------------------------------

@dataclass
class ForkContext:
    """
    A child execution context created by a fork operation.

    The child agent inherits specified state from the parent and executes
    its body independently.  Results are collected and can be merged back.
    """
    id: str
    parent_id: str
    child_agent: Agent
    inherited_state: dict[str, Any] = field(default_factory=dict)
    body: list[Expression] = field(default_factory=list)
    result: Optional[Result] = None
    status: str = "pending"  # pending | executing | completed | failed | collected
    created_at: str = ""
    completed_at: str = ""
    inherit_context: bool = True
    inherit_trust_graph: bool = False
    on_result: str = "collect"  # collect | discard | signal

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            from datetime import datetime, timezone
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds (approximate)."""
        if not self.completed_at or not self.created_at:
            return 0
        try:
            from datetime import datetime, timezone
            start = datetime.fromisoformat(self.created_at)
            end = datetime.fromisoformat(self.completed_at)
            return int((end - start).total_seconds() * 1000)
        except (ValueError, TypeError):
            return 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "parent_id": self.parent_id,
            "child_agent": {
                "id": self.child_agent.id,
                "role": self.child_agent.role,
                "capabilities": self.child_agent.capabilities,
                "trust": self.child_agent.trust,
            },
            "inherited_state_keys": list(self.inherited_state.keys()),
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "inherit_context": self.inherit_context,
            "inherit_trust_graph": self.inherit_trust_graph,
            "on_result": self.on_result,
        }
        if self.result:
            d["result"] = self.result.to_dict()
        return d


# ---------------------------------------------------------------------------
# Fork tree — visualization of all forks
# ---------------------------------------------------------------------------

@dataclass
class ForkTree:
    """
    Tree representation of all forks in an execution context.

    Each node represents either a parent or child context.  Edges
    represent fork relationships.
    """
    _nodes: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _edges: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def add_node(self, node_id: str, agent_id: str, role: str = "", status: str = "") -> None:
        """Add a node to the tree."""
        self._nodes[node_id] = {
            "id": node_id,
            "agent_id": agent_id,
            "role": role,
            "status": status,
            "children": [],
            "parent": None,
        }

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a fork relationship (parent → child)."""
        self._edges.append((parent_id, child_id))
        if parent_id in self._nodes:
            self._nodes[parent_id]["children"].append(child_id)
        if child_id in self._nodes:
            self._nodes[child_id]["parent"] = parent_id

    def get_children(self, node_id: str) -> list[str]:
        """Get all direct children of a node."""
        return self._nodes.get(node_id, {}).get("children", [])

    def get_parent(self, node_id: str) -> Optional[str]:
        """Get the parent of a node."""
        return self._nodes.get(node_id, {}).get("parent")

    def get_root(self) -> Optional[str]:
        """Find the root node (node with no parent)."""
        for nid, node in self._nodes.items():
            if node["parent"] is None:
                return nid
        return None

    def get_depth(self, node_id: str) -> int:
        """Calculate the depth of a node in the tree."""
        depth = 0
        current = node_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            parent = self.get_parent(current)
            if parent is None:
                break
            depth += 1
            current = parent
        return depth

    def to_ascii(self, node_id: Optional[str] = None, prefix: str = "", is_last: bool = True) -> str:
        """Generate an ASCII tree representation."""
        if node_id is None:
            node_id = self.get_root()
            if node_id is None:
                return "(empty tree)"

        node = self._nodes.get(node_id, {})
        agent_id = node.get("agent_id", "?")
        status = node.get("status", "")
        role = node.get("role", "")

        label = f"[{agent_id}]"
        if role:
            label += f" ({role})"
        if status:
            label += f" <{status}>"

        lines = [prefix + ("└── " if is_last else "├── ") + label]

        children = self.get_children(node_id)
        for i, child_id in enumerate(children):
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_is_last = (i == len(children) - 1)
            child_lines = self.to_ascii(child_id, child_prefix, child_is_last)
            lines.append(child_lines)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": dict(self._nodes),
            "edges": [{"parent": p, "child": c} for p, c in self._edges],
            "root": self.get_root(),
        }


# ---------------------------------------------------------------------------
# Fork manager — track and manage all forks
# ---------------------------------------------------------------------------

@dataclass
class ForkManager:
    """Track and manage all active forks in an execution context."""

    _forks: OrderedDict[str, ForkContext] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _tree: ForkTree = field(default_factory=ForkTree, init=False, repr=False)
    _collected_results: list[Result] = field(
        default_factory=list, init=False, repr=False
    )

    def create_fork(
        self,
        fork_id: str,
        parent_id: str,
        child_agent: Agent,
        inherited_state: dict[str, Any],
        body: list[Expression],
        inherit_context: bool = True,
        inherit_trust_graph: bool = False,
        on_result: str = "collect",
    ) -> ForkContext:
        """Create a new fork context."""
        ctx = ForkContext(
            id=fork_id,
            parent_id=parent_id,
            child_agent=child_agent,
            inherited_state=dict(inherited_state),
            body=body,
            inherit_context=inherit_context,
            inherit_trust_graph=inherit_trust_graph,
            on_result=on_result,
        )
        self._forks[fork_id] = ctx

        # Update tree
        self._tree.add_node(parent_id, parent_id, role="parent", status="active")
        self._tree.add_node(fork_id, child_agent.id, role=child_agent.role, status="pending")
        self._tree.add_edge(parent_id, fork_id)

        return ctx

    def get_fork(self, fork_id: str) -> Optional[ForkContext]:
        """Get a fork context by ID."""
        return self._forks.get(fork_id)

    def complete_fork(self, fork_id: str, result: Result) -> None:
        """Mark a fork as completed with its result."""
        ctx = self._forks.get(fork_id)
        if ctx is None:
            return

        ctx.result = result
        ctx.status = "completed"

        from datetime import datetime, timezone
        ctx.completed_at = datetime.now(timezone.utc).isoformat()

        # Update tree node status
        if fork_id in self._tree._nodes:
            self._tree._nodes[fork_id]["status"] = "completed"

        # Collect result if configured
        if ctx.on_result == "collect":
            self._collected_results.append(result)

    def fail_fork(self, fork_id: str, error: str = "") -> None:
        """Mark a fork as failed."""
        ctx = self._forks.get(fork_id)
        if ctx is None:
            return

        ctx.result = Result(value=None, confidence=0.0, source="fork", error=error)
        ctx.status = "failed"

        from datetime import datetime, timezone
        ctx.completed_at = datetime.now(timezone.utc).isoformat()

        if fork_id in self._tree._nodes:
            self._tree._nodes[fork_id]["status"] = "failed"

    def get_active_forks(self) -> list[ForkContext]:
        """Get all forks that are not yet completed or failed."""
        return [ctx for ctx in self._forks.values() if ctx.status in ("pending", "executing")]

    def get_completed_forks(self) -> list[ForkContext]:
        """Get all completed forks."""
        return [ctx for ctx in self._forks.values() if ctx.status == "completed"]

    def get_forks_by_parent(self, parent_id: str) -> list[ForkContext]:
        """Get all forks spawned by a given parent."""
        return [ctx for ctx in self._forks.values() if ctx.parent_id == parent_id]

    def get_collected_results(self) -> list[Result]:
        """Get all collected results from completed forks."""
        return list(self._collected_results)

    def get_tree(self) -> ForkTree:
        """Get the fork tree visualization."""
        return self._tree

    def to_dict(self) -> dict[str, Any]:
        return {
            "forks": {fid: ctx.to_dict() for fid, ctx in self._forks.items()},
            "tree": self._tree.to_dict(),
            "active_count": len(self.get_active_forks()),
            "completed_count": len(self.get_completed_forks()),
            "collected_count": len(self._collected_results),
        }
