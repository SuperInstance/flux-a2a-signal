"""
Paradigm Flow Simulation — what happens when paradigms collide.

For each dimension, bridging from source to target falls into one of three modes:
  - source > target: feature must be "compiled away" (lossy — expressiveness lost)
  - source < target: feature must be "emulated" (overhead — runtime cost)
  - source ≈ target: direct mapping (efficient — natural bridge)

The simulation computes:
  - expressiveness_loss:  what you CAN'T express after bridging
  - performance_overhead: emulation cost for missing features
  - cognitive_load:       how hard it is for a programmer
  - semantic_drift:       how much meaning changes in translation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from flux_a2a.paradigm_lattice import (
    DIMENSION_NAMES,
    DIMENSION_WEIGHTS,
    ParadigmLattice,
    ParadigmPoint,
)


# ══════════════════════════════════════════════════════════════════
# Bridge Simulation Data
# ══════════════════════════════════════════════════════════════════

@dataclass
class DimensionBridge:
    """How a single dimension transforms when bridging two paradigms."""
    dimension: str
    source_value: float
    target_value: float
    delta: float
    mode: str          # "lossy" | "overhead" | "direct" | "gain"
    severity: float    # 0.0 (trivial) to 1.0 (catastrophic)
    description: str


@dataclass
class BridgeCost:
    """Quantified cost of bridging two paradigms."""
    source: str
    target: str
    expressiveness_loss: float   # 0.0 (nothing lost) to 1.0 (total loss)
    performance_overhead: float  # 0.0 (zero overhead) to 1.0 (extreme)
    cognitive_load: float        # 0.0 (trivial) to 1.0 (impossible)
    semantic_drift: float        # 0.0 (no drift) to 1.0 (total transformation)
    total_cost: float            # weighted combination
    dimension_bridges: List[DimensionBridge] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Bridge {self.source} → {self.target}: "
            f"loss={self.expressiveness_loss:.2f} "
            f"overhead={self.performance_overhead:.2f} "
            f"cognitive={self.cognitive_load:.2f} "
            f"drift={self.semantic_drift:.2f} "
            f"total={self.total_cost:.2f}"
        )


@dataclass
class BridgeSimulation:
    """Full bridge simulation between two paradigm points."""
    source: ParadigmPoint
    target: ParadigmPoint
    cost: BridgeCost
    constructs_lost: List[str] = field(default_factory=list)
    constructs_gained: List[str] = field(default_factory=list)
    constructs_transformed: List[str] = field(default_factory=list)
    natural_bridge_quality: float = 0.0  # 0.0 (terrible) to 1.0 (perfect)
    via_intermediate: Optional[str] = None  # Suggested intermediate if direct is bad


# ══════════════════════════════════════════════════════════════════
# Construct Mappings per Dimension
# ══════════════════════════════════════════════════════════════════

# What gets lost/gained when reducing/increasing each dimension
CONSTRUCT_MAP: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "state_magnitude": {
        "reduce": {
            "lost": ["mutable variables", "in-place updates", "pointer arithmetic",
                      " destructive operations", "imperative loops"],
            "gained": ["referential transparency", "pure functions", "memoization",
                        "automatic parallelism", "equational reasoning"],
        },
        "increase": {
            "lost": ["referential transparency", "automatic parallelism",
                      "equational reasoning", "memoization guarantees"],
            "gained": ["mutable variables", "in-place updates", "pointer arithmetic",
                        "destructive operations", "imperative loops"],
        },
    },
    "control_explicitness": {
        "reduce": {
            "lost": ["goto statements", "explicit branching", "loop counters",
                      "manual control flow"],
            "gained": ["dataflow execution", "declarative specification",
                        "automatic scheduling", "lazy evaluation"],
        },
        "increase": {
            "lost": ["dataflow execution", "declarative specification",
                      "automatic scheduling"],
            "gained": ["goto statements", "explicit branching", "loop counters",
                        "fine-grained control flow"],
        },
    },
    "typing_strength": {
        "reduce": {
            "lost": ["compile-time type checking", "type-level computation",
                      "dependent types", "refinement types", "phantom types"],
            "gained": ["rapid prototyping", "duck typing", "dynamic dispatch",
                        "metaprogramming flexibility"],
        },
        "increase": {
            "lost": ["rapid prototyping", "runtime type mutation",
                      "dynamic dispatch flexibility"],
            "gained": ["compile-time guarantees", "type-level computation",
                        "dependent types", "proofs as programs"],
        },
    },
    "composition_style": {
        "reduce": {
            "lost": ["inheritance hierarchies", "polymorphic dispatch",
                      "method overriding", "interface implementation"],
            "gained": ["function composition", "pipeline operators",
                        "point-free style", "concatenative composition"],
        },
        "increase": {
            "lost": ["function pipelines", "point-free composition",
                      "concatenative composition"],
            "gained": ["inheritance hierarchies", "polymorphic dispatch",
                        "method overriding", "design patterns"],
        },
    },
    "concurrency_model": {
        "reduce": {
            "lost": ["actor isolation", "message passing", "distributed computing",
                      "mailbox pattern", "supervision trees"],
            "gained": ["deterministic execution", "simpler reasoning",
                        "no race conditions", "linear execution model"],
        },
        "increase": {
            "lost": ["deterministic execution", "simple reasoning model"],
            "gained": ["parallelism", "actor isolation", "message passing",
                        "distributed computing", "fault tolerance"],
        },
    },
    "effect_tracking": {
        "reduce": {
            "lost": ["effect annotations", "algebraic effects", "effect handlers",
                      "IO tracking", "purity guarantees"],
            "gained": ["simpler code", "no effect annotation burden",
                        "free mixing of effects"],
        },
        "increase": {
            "lost": ["effect-free programming simplicity", "no annotation burden"],
            "gained": ["tracked side effects", "algebraic effects",
                        "effect handlers", "purity by default", "composable effects"],
        },
    },
    "naming_style": {
        "reduce": {
            "lost": ["keyword arguments", "named parameters", "self-documenting calls",
                      "optional named parameters"],
            "gained": ["concise syntax", "arity-based dispatch",
                        "positional composition"],
        },
        "increase": {
            "lost": ["concise positional syntax", "arity-based dispatch"],
            "gained": ["keyword arguments", "named parameters",
                        "self-documenting code", "default values"],
        },
    },
    "abstraction_level": {
        "reduce": {
            "lost": ["high-level abstractions", "declarative interfaces",
                      "domain-specific languages", "automatic optimization"],
            "gained": ["predictable performance", "hardware control",
                        "minimal abstraction cost", "transparent compilation"],
        },
        "increase": {
            "lost": ["predictable low-level performance", "hardware control"],
            "gained": ["high-level abstractions", "declarative interfaces",
                        "domain-specific languages", "intent-based programming"],
        },
    },
}


# ══════════════════════════════════════════════════════════════════
# Natural Language Specific Construct Knowledge
# ══════════════════════════════════════════════════════════════════

NL_CONSTRUCTS: Dict[str, Dict[str, List[str]]] = {
    "zho": {
        "unique": ["classifier types (量词)", "topic-comment (主题-评论)",
                    "zero-anaphora (零形回指)", "context-dependent scope"],
        "bridge_notes": "Classifiers compile to FIR types; topic maps to R63 register.",
    },
    "kor": {
        "unique": ["7-level honorifics (경어체)", "subject/object honorification",
                    "speech level particles", "infix verb conjugation"],
        "bridge_notes": "Honorifics map to CAP_REQUIRE levels; particles to scope markers.",
    },
    "san": {
        "unique": ["8-case vibhakti (विभक्ति)", "3-gender system",
                    "sandhi fusion (सन्धि)", "Paninian grammar rules"],
        "bridge_notes": "Each vibhakti maps to a ScopeLevel; sandhi enables automatic fusion.",
    },
    "deu": {
        "unique": ["4-case Kasus system", "compound noun types",
                    "V2 word order constraint", "separable-prefix verbs"],
        "bridge_notes": "Kasus maps to 4 scope levels; separable verbs = deferred computation.",
    },
    "wen": {
        "unique": ["extreme information density", "Confucian 五常 ethics system",
                    "Sun Tzu military strategy opcodes", "positional composition"],
        "bridge_notes": "五常 maps to trust/capability; military strategy maps to flow control.",
    },
    "lat": {
        "unique": ["6-tense temporal system", "6-case spatial scope",
                    "subjunctive mood = confidence", "word-order freedom"],
        "bridge_notes": "Tenses map to temporal opcodes; subjunctive encodes confidence levels.",
    },
}


# ══════════════════════════════════════════════════════════════════
# Paradigm Flow Engine
# ══════════════════════════════════════════════════════════════════

class ParadigmFlow:
    """Simulate what happens when paradigms interact.

    The engine computes bridge costs, identifies optimal routing paths,
    and discovers paradigm fusion opportunities.
    """

    def __init__(self, lattice: Optional[ParadigmLattice] = None) -> None:
        self.lattice = lattice or ParadigmLattice()

    # ── Bridge Simulation ───────────────────────────────────────

    def simulate_bridge(
        self, source: ParadigmPoint, target: ParadigmPoint
    ) -> BridgeSimulation:
        """What constructs are lost, gained, transformed when bridging paradigms?

        For each dimension:
          - source > target: feature must be "compiled away" (lossy)
          - source < target: feature must be "emulated" (overhead)
          - |delta| < 0.15: direct mapping (efficient)
        """
        dimension_bridges: List[DimensionBridge] = []
        all_lost: List[str] = []
        all_gained: List[str] = []
        all_transformed: List[str] = []

        for dim in DIMENSION_NAMES:
            src_val = source.coordinates[dim]
            tgt_val = target.coordinates[dim]
            delta = src_val - tgt_val
            abs_delta = abs(delta)
            weight = DIMENSION_WEIGHTS.get(dim, 1.0)

            # Determine mode and severity
            if abs_delta < 0.15:
                mode = "direct"
                severity = abs_delta * 2
                desc = f"Near-equal: direct mapping ({src_val:.2f} ≈ {tgt_val:.2f})"
            elif delta > 0:
                mode = "lossy"
                severity = min(delta * weight, 1.0)
                desc = (f"Feature reduction: must compile away "
                        f"({src_val:.2f} → {tgt_val:.2f})")
                # Add constructs lost/gained
                construct_info = CONSTRUCT_MAP.get(dim, {}).get("reduce", {})
                all_lost.extend(construct_info.get("lost", []))
                all_gained.extend(construct_info.get("gained", []))
            else:
                mode = "overhead"
                severity = min(abs_delta * weight, 1.0)
                desc = (f"Feature emulation: must build up "
                        f"({src_val:.2f} → {tgt_val:.2f})")
                construct_info = CONSTRUCT_MAP.get(dim, {}).get("increase", {})
                all_lost.extend(construct_info.get("lost", []))
                all_gained.extend(construct_info.get("gained", []))

            dimension_bridges.append(DimensionBridge(
                dimension=dim,
                source_value=src_val,
                target_value=tgt_val,
                delta=delta,
                mode=mode,
                severity=severity,
                description=desc,
            ))

        # Add NL-specific construct info
        if source.name in NL_CONSTRUCTS:
            all_lost.extend(NL_CONSTRUCTS[source.name]["unique"])
        if target.name in NL_CONSTRUCTS:
            all_gained.extend(NL_CONSTRUCTS[target.name]["unique"])

        # Identify transformed constructs (constructs that exist in both
        # paradigms but with different semantics)
        all_transformed = self._identify_transformed(source, target)

        # Compute cost metrics
        cost = self._compute_bridge_cost(source, target, dimension_bridges)

        # Natural bridge quality (inverse of total cost, normalized)
        natural_bridge_quality = max(0.0, 1.0 - cost.total_cost)

        # Check if intermediate point would help
        via = self._suggest_intermediate(source, target, cost.total_cost)

        return BridgeSimulation(
            source=source,
            target=target,
            cost=cost,
            constructs_lost=list(set(all_lost)),
            constructs_gained=list(set(all_gained)),
            constructs_transformed=all_transformed,
            natural_bridge_quality=natural_bridge_quality,
            via_intermediate=via,
        )

    def _identify_transformed(
        self, source: ParadigmPoint, target: ParadigmPoint
    ) -> List[str]:
        """Find constructs that exist in both paradigms but change meaning."""
        transformed = []
        # Both have state but different magnitude → "mutation semantics differ"
        if (source.coordinates["state_magnitude"] > 0.3 and
                target.coordinates["state_magnitude"] > 0.3 and
                abs(source.coordinates["state_magnitude"] -
                    target.coordinates["state_magnitude"]) > 0.3):
            transformed.append("mutation semantics (different discipline)")
        # Both have typing but different strength → "type system semantics differ"
        if (source.coordinates["typing_strength"] > 0.3 and
                target.coordinates["typing_strength"] > 0.3 and
                abs(source.coordinates["typing_strength"] -
                    target.coordinates["typing_strength"]) > 0.3):
            transformed.append("type system semantics (different guarantees)")
        # Both have effect tracking → "effect model differs"
        if (source.coordinates["effect_tracking"] > 0.2 and
                target.coordinates["effect_tracking"] > 0.2 and
                abs(source.coordinates["effect_tracking"] -
                    target.coordinates["effect_tracking"]) > 0.3):
            transformed.append("effect model (different tracking discipline)")
        # Both have concurrency → "concurrency model differs"
        if (source.coordinates["concurrency_model"] > 0.2 and
                target.coordinates["concurrency_model"] > 0.2 and
                abs(source.coordinates["concurrency_model"] -
                    target.coordinates["concurrency_model"]) > 0.3):
            transformed.append("concurrency model (different isolation strategy)")
        # Both are NL languages → "grammatical viewpoint differs"
        nl_names = {"zho", "kor", "san", "deu", "wen", "lat"}
        if source.name in nl_names and target.name in nl_names:
            transformed.append("grammatical viewpoint (different NL compilation path)")
        return transformed

    def _suggest_intermediate(
        self, source: ParadigmPoint, target: ParadigmPoint,
        direct_cost: float
    ) -> Optional[str]:
        """Suggest an intermediate paradigm if it reduces total bridge cost."""
        if direct_cost < 0.8:
            return None  # Direct bridge is acceptable

        best_intermediate = None
        best_savings = 0.0

        for candidate in self.lattice.all_points():
            if candidate.name in (source.name, target.name):
                continue
            cost_via = (source.distance_to(candidate) +
                        candidate.distance_to(target))
            savings = direct_cost - cost_via
            if savings > best_savings and savings > 0.3:
                best_savings = savings
                best_intermediate = candidate.name

        return best_intermediate

    def _compute_bridge_cost(
        self, source: ParadigmPoint, target: ParadigmPoint,
        bridges: List[DimensionBridge],
    ) -> BridgeCost:
        """Quantify the four cost dimensions of bridging."""
        # Expressiveness loss: severity of lossy dimensions (features you must remove)
        lossy_severities = [b.severity for b in bridges if b.mode == "lossy"]
        overhead_severities = [b.severity for b in bridges if b.mode == "overhead"]

        expressiveness_loss = (
            sum(lossy_severities) / len(DIMENSION_NAMES)
            if lossy_severities else 0.0
        )

        # Performance overhead: severity of overhead dimensions (features you must emulate)
        performance_overhead = (
            sum(overhead_severities) / len(DIMENSION_NAMES)
            if overhead_severities else 0.0
        )

        # Cognitive load: based on number of dimensions that change significantly
        significant_changes = sum(
            1 for b in bridges if abs(b.delta) > 0.3
        )
        cognitive_load = significant_changes / len(DIMENSION_NAMES)

        # Semantic drift: weighted distance (captures meaning change)
        semantic_drift = min(source.distance_to(target, weighted=True) / 3.0, 1.0)

        # Total cost: weighted combination
        total_cost = (
            expressiveness_loss * 0.30 +
            performance_overhead * 0.20 +
            cognitive_load * 0.25 +
            semantic_drift * 0.25
        )
        total_cost = min(total_cost, 1.0)

        return BridgeCost(
            source=source.name,
            target=target.name,
            expressiveness_loss=expressiveness_loss,
            performance_overhead=performance_overhead,
            cognitive_load=cognitive_load,
            semantic_drift=semantic_drift,
            total_cost=total_cost,
            dimension_bridges=bridges,
        )

    # ── All-Pairs Simulation ────────────────────────────────────

    def simulate_all_pairs(
        self, names: Optional[List[str]] = None
    ) -> List[BridgeSimulation]:
        """Run bridge simulations for all pairs of paradigm points."""
        if names is None:
            names = [p.name for p in self.lattice.nl_points()]
        results = []
        for i, a_name in enumerate(names):
            for b_name in names[i+1:]:
                a = self.lattice.get(a_name)
                b = self.lattice.get(b_name)
                results.append(self.simulate_bridge(a, b))
        # Sort by total cost (hardest bridges first)
        results.sort(key=lambda s: s.cost.total_cost, reverse=True)
        return results

    # ── Optimal Routing ─────────────────────────────────────────

    def find_optimal_path(
        self,
        source: str,
        target: str,
        max_hops: int = 2,
        candidates: Optional[List[str]] = None,
    ) -> List[str]:
        """Find the best path through paradigm space.

        Sometimes bridging A→C directly is expensive, but A→B→C is cheaper
        if B is a good intermediate.

        Returns:
            List of paradigm names forming the optimal path.
        """
        if candidates is None:
            candidates = [p.name for p in self.lattice.all_points()]

        direct_cost = self.lattice.distance(source, target, weighted=True)
        best_path = [source, target]
        best_cost = direct_cost

        if max_hops >= 2:
            # Try all single-hop intermediates
            for intermediate in candidates:
                if intermediate in (source, target):
                    continue
                hop1 = self.lattice.distance(source, intermediate, weighted=True)
                hop2 = self.lattice.distance(intermediate, target, weighted=True)
                total = hop1 + hop2
                if total < best_cost:
                    best_cost = total
                    best_path = [source, intermediate, target]

        if max_hops >= 3:
            # Try all two-hop intermediates
            for i1 in candidates:
                if i1 in (source, target):
                    continue
                hop1 = self.lattice.distance(source, i1, weighted=True)
                if hop1 >= best_cost:  # Prune: first hop already worse than direct
                    continue
                for i2 in candidates:
                    if i2 in (source, target, i1):
                        continue
                    hop2 = self.lattice.distance(i1, i2, weighted=True)
                    hop3 = self.lattice.distance(i2, target, weighted=True)
                    total = hop1 + hop2 + hop3
                    if total < best_cost:
                        best_cost = total
                        best_path = [source, i1, i2, target]

        return best_path

    def compute_routing_table(
        self, names: Optional[List[str]] = None, max_hops: int = 2
    ) -> Dict[Tuple[str, str], List[str]]:
        """Compute optimal routing between all pairs."""
        if names is None:
            names = [p.name for p in self.lattice.nl_points()]
        table: Dict[Tuple[str, str], List[str]] = {}
        for i, a in enumerate(names):
            for b in names:
                if a == b:
                    continue
                table[(a, b)] = self.find_optimal_path(a, b, max_hops=max_hops)
        return table

    # ── Fusion Opportunity Detection ────────────────────────────

    def find_fusion_opportunities(
        self, names: Optional[List[str]] = None,
        include_cross_paradigm: bool = True,
    ) -> List[Dict]:
        """Identify pairs of paradigms that naturally complement each other.

        A fusion opportunity exists when:
        1. Two paradigms are distant (high distance = different strengths)
        2. Their strongest dimensions don't overlap (complementary)
        3. Bridging cost is moderate (not catastrophic)
        """
        if names is None:
            if include_cross_paradigm:
                names = [p.name for p in self.lattice.all_points()]
            else:
                names = [p.name for p in self.lattice.nl_points()]

        opportunities = []
        for i, a_name in enumerate(names):
            for b_name in names[i+1:]:
                a = self.lattice.get(a_name)
                b = self.lattice.get(b_name)
                dist = a.distance_to(b, weighted=True)
                sim = self.simulate_bridge(a, b)

                # Find complementary dimensions: one is high, other is low
                complementary_dims = []
                for dim in DIMENSION_NAMES:
                    a_val = a.coordinates[dim]
                    b_val = b.coordinates[dim]
                    # Relaxed thresholds: high > 0.55, low < 0.45, gap > 0.20
                    if (a_val > 0.55 and b_val < 0.45) or (
                        b_val > 0.55 and a_val < 0.45
                    ):
                        if abs(a_val - b_val) > 0.20:
                            complementary_dims.append(dim)

                if len(complementary_dims) >= 2 and sim.cost.total_cost < 0.9:
                    # Score: more complementary dims + moderate cost + NL bonus
                    nl_bonus = 1.5 if (a.name in NL_CONSTRUCTS and b.name in NL_CONSTRUCTS) else 1.0
                    cross_bonus = 1.3 if ((a.name in NL_CONSTRUCTS) != (b.name in NL_CONSTRUCTS)) else 1.0
                    score = len(complementary_dims) * (1.0 - sim.cost.total_cost) * nl_bonus * cross_bonus
                    opportunities.append({
                        "paradigm_a": a_name,
                        "paradigm_b": b_name,
                        "distance": round(dist, 3),
                        "complementary_dimensions": complementary_dims,
                        "bridge_cost": round(sim.cost.total_cost, 3),
                        "fusion_score": round(score, 3),
                        "constructs_a_unique": NL_CONSTRUCTS.get(a_name, {}).get("unique", []),
                        "constructs_b_unique": NL_CONSTRUCTS.get(b_name, {}).get("unique", []),
                        "hypothesis": self._generate_fusion_hypothesis(a, b, complementary_dims),
                    })

        opportunities.sort(key=lambda x: x["fusion_score"], reverse=True)
        return opportunities

    def _generate_fusion_hypothesis(
        self, a: ParadigmPoint, b: ParadigmPoint, dims: List[str]
    ) -> str:
        """Generate a natural-language hypothesis about what fusing two paradigms would yield."""
        parts = []
        nl_info_a = NL_CONSTRUCTS.get(a.name, {})
        nl_info_b = NL_CONSTRUCTS.get(b.name, {})

        if a.name in NL_CONSTRUCTS and b.name in NL_CONSTRUCTS:
            unique_a = nl_info_a.get("unique", [])
            unique_b = nl_info_b.get("unique", [])
            if unique_a and unique_b:
                parts.append(
                    f"Fusing {a.name}'s {unique_a[0]} with {b.name}'s {unique_b[0]}"
                )

        dim_descriptions = {
            "state_magnitude": "state discipline",
            "control_explicitness": "control flow",
            "typing_strength": "type system",
            "composition_style": "composition",
            "concurrency_model": "concurrency",
            "effect_tracking": "effect tracking",
            "naming_style": "argument naming",
            "abstraction_level": "abstraction level",
        }

        if dims:
            dim_names = [dim_descriptions.get(d, d) for d in dims[:3]]
            parts.append(f"Complementary in: {', '.join(dim_names)}")

        return " → ".join(parts) if parts else f"{a.name} + {b.name} hybrid paradigm"


# ══════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════

def generate_simulation_report() -> str:
    """Generate a comprehensive simulation report as markdown."""
    lattice = ParadigmLattice()
    flow = ParadigmFlow(lattice)

    lines: List[str] = []
    lines.append("# Paradigm Simulation Report: Rounds 4-6")
    lines.append("")
    lines.append("**Generated by:** ParadigmFlow simulation engine")
    lines.append(f"**Paradigm points:** {len(lattice.all_points())} "
                 f"({len(lattice.nl_points())} NL + {len(lattice.classical_points())} classical)")
    lines.append("")

    # ── Section 1: Lattice Overview ──
    lines.append("## 1. Paradigm Lattice Overview")
    lines.append("")
    lines.append("### 8-Dimensional Coordinate Table")
    lines.append("")
    lines.append("| Paradigm | state | ctrl | typ | comp | conc | eff | name | abst |")
    lines.append("|----------|-------|------|-----|------|------|-----|------|------|")
    for p in lattice.all_points():
        coords = p.coordinates
        row = (f"| {p.name:10s} | "
               + " | ".join(f"{coords[d]:.2f}" for d in DIMENSION_NAMES)
               + " |")
        lines.append(row)
    lines.append("")

    # ── Section 2: Hub Analysis ──
    lines.append("## 2. Hub Analysis — Which Paradigm Is the Best Bridge?")
    lines.append("")
    hub_name, hub_avg, hub_dists = lattice.best_hub(weighted=True)
    nl_hub, nl_avg, nl_dists = lattice.best_hub(
        candidates=[p.name for p in lattice.nl_points()], weighted=True
    )
    lines.append(f"### Overall Best Hub: **{hub_name}** (avg distance: {hub_avg:.3f})")
    lines.append("")
    lines.append("| From | Distance to hub |")
    lines.append("|------|----------------|")
    for name, dist in sorted(hub_dists.items(), key=lambda x: x[1]):
        lines.append(f"| {name} | {dist:.3f} |")
    lines.append("")
    lines.append(f"### NL-Only Best Hub: **{nl_hub}** (avg distance: {nl_avg:.3f})")
    lines.append("")
    lines.append("| From | Distance to hub |")
    lines.append("|------|----------------|")
    for name, dist in sorted(nl_dists.items(), key=lambda x: x[1]):
        lines.append(f"| {name} | {dist:.3f} |")
    lines.append("")

    # ── Section 3: All-Pairs Bridge Costs ──
    lines.append("## 3. All-Pairs Bridge Cost Matrix (NL Languages)")
    lines.append("")
    nl_names = sorted(p.name for p in lattice.nl_points())

    lines.append("| Source → Target | Loss | Overhead | Cognitive | Drift | **Total** | Via |")
    lines.append("|---------------|------|----------|-----------|-------|----------|-----|")
    simulations = flow.simulate_all_pairs(nl_names)
    for sim in simulations:
        c = sim.cost
        via = sim.via_intermediate or "direct"
        lines.append(
            f"| {c.source} → {c.target} | {c.expressiveness_loss:.2f} | "
            f"{c.performance_overhead:.2f} | {c.cognitive_load:.2f} | "
            f"{c.semantic_drift:.2f} | **{c.total_cost:.2f}** | {via} |"
        )
    lines.append("")

    # ── Section 4: Key Bridge Analyses ──
    lines.append("## 4. Key Bridge Analyses")
    lines.append("")

    # Highlighted bridges
    highlighted = [
        ("zho", "deu"),
        ("kor", "san"),
        ("wen", "lat"),
    ]
    for src_name, tgt_name in highlighted:
        src = lattice.get(src_name)
        tgt = lattice.get(tgt_name)
        sim = flow.simulate_bridge(src, tgt)

        lines.append(f"### {src_name.upper()} → {tgt_name.upper()} Bridge")
        lines.append("")
        lines.append(f"**Bridge quality:** {sim.natural_bridge_quality:.1%}")
        lines.append(f"**Total cost:** {sim.cost.total_cost:.3f}")
        if sim.via_intermediate:
            lines.append(f"**Recommended via:** {sim.via_intermediate}")
        lines.append("")

        lines.append("**Dimension analysis:**")
        lines.append("")
        for db in sim.cost.dimension_bridges:
            indicator = {"lossy": "📉", "overhead": "📈", "direct": "➡️", "gain": "🆙"}.get(
                db.mode, "?")
            lines.append(
                f"- {indicator} **{db.dimension}**: {db.description} "
                f"(severity: {db.severity:.2f})"
            )
        lines.append("")

        if sim.constructs_lost:
            lines.append(f"**Constructs lost:** {', '.join(sim.constructs_lost[:8])}")
            lines.append("")
        if sim.constructs_gained:
            lines.append(f"**Constructs gained:** {', '.join(sim.constructs_gained[:8])}")
            lines.append("")
        if sim.constructs_transformed:
            lines.append(f"**Constructs transformed:** {', '.join(sim.constructs_transformed)}")
            lines.append("")

        # NL-specific notes
        src_info = NL_CONSTRUCTS.get(src_name, {})
        tgt_info = NL_CONSTRUCTS.get(tgt_name, {})
        if src_info.get("bridge_notes"):
            lines.append(f"*{src_name}:* {src_info['bridge_notes']}")
        if tgt_info.get("bridge_notes"):
            lines.append(f"*{tgt_name}:* {tgt_info['bridge_notes']}")
        lines.append("")

    # ── Section 5: Routing Table ──
    lines.append("## 5. Optimal Routing Map")
    lines.append("")
    lines.append("Best path (up to 2 hops) between all NL paradigm points:")
    lines.append("")
    routing = flow.compute_routing_table(nl_names, max_hops=2)
    for (src, tgt), path in sorted(routing.items()):
        path_str = " → ".join(path)
        direct = lattice.distance(src, tgt, weighted=True)
        if len(path) > 2:
            via_cost = sum(
                lattice.distance(path[i], path[i+1], weighted=True)
                for i in range(len(path) - 1)
            )
            savings = direct - via_cost
            lines.append(
                f"- **{src} → {tgt}**: {path_str} "
                f"(saves {savings:.3f} vs direct {direct:.3f})"
            )
        else:
            lines.append(f"- **{src} → {tgt}**: direct ({direct:.3f})")
    lines.append("")

    # ── Section 6: Fusion Opportunities ──
    lines.append("## 6. Paradigm Fusion Opportunities")
    lines.append("")
    lines.append("### 6a. NL-NL Fusion Opportunities")
    lines.append("")
    fusions_nl = flow.find_fusion_opportunities(nl_names, include_cross_paradigm=False)
    if fusions_nl:
        for opp in fusions_nl[:10]:
            lines.append(f"#### {opp['paradigm_a'].upper()} + {opp['paradigm_b'].upper()}")
            lines.append("")
            lines.append(f"- **Fusion score:** {opp['fusion_score']:.3f}")
            lines.append(f"- **Bridge distance:** {opp['distance']:.3f}")
            lines.append(f"- **Bridge cost:** {opp['bridge_cost']:.3f}")
            lines.append(f"- **Complementary dimensions:** {', '.join(opp['complementary_dimensions'])}")
            lines.append(f"- **Hypothesis:** {opp['hypothesis']}")
            lines.append("")
    else:
        lines.append("No significant NL-NL fusion opportunities detected.")
        lines.append("*(This is expected — the FLUX NL paradigms were designed to be interoperable.)*")
        lines.append("")

    lines.append("### 6b. Cross-Paradigm Fusion Opportunities (NL × Classical)")
    lines.append("")
    fusions_all = flow.find_fusion_opportunities(include_cross_paradigm=True)
    # Filter to only cross-paradigm pairs
    nl_set = set(nl_names)
    fusions_cross = [o for o in fusions_all
                     if (o['paradigm_a'] in nl_set) != (o['paradigm_b'] in nl_set)]
    for opp in fusions_cross[:15]:
        lines.append(f"#### {opp['paradigm_a'].upper()} + {opp['paradigm_b'].upper()}")
        lines.append("")
        lines.append(f"- **Fusion score:** {opp['fusion_score']:.3f}")
        lines.append(f"- **Bridge distance:** {opp['distance']:.3f}")
        lines.append(f"- **Bridge cost:** {opp['bridge_cost']:.3f}")
        lines.append(f"- **Complementary dimensions:** {', '.join(opp['complementary_dimensions'])}")
        lines.append(f"- **Hypothesis:** {opp['hypothesis']}")
        lines.append("")

    # ── Section 7: Vacancy Detection ──
    lines.append("## 7. Paradigm Vacancies — Empty Regions of Paradigm Space")
    lines.append("")
    vacancies = lattice.detect_vacancies(resolution=0.25)
    # Keep only large clusters (significant vacancies)
    big_vacancies = [v for v in vacancies if "Cluster of" in v.notes
                     and int(v.notes.split("of ")[-1].split(" ")[0]) >= 20]
    if big_vacancies:
        lines.append(f"Detected {len(big_vacancies)} significant vacancy clusters (≥20 samples):")
        lines.append("")
        for v in big_vacancies[:10]:
            coords_str = ", ".join(
                f"{d[:4]}={v.coordinates[d]:.2f}" for d in DIMENSION_NAMES
            )
            lines.append(f"- **{v.name}**: [{coords_str}]")
            lines.append(f"  {v.notes}")
    else:
        lines.append("No significant vacancies detected at current resolution.")
    lines.append("")

    return "\n".join(lines)
