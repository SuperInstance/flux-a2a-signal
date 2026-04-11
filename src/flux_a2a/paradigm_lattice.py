"""
Paradigm Lattice Model — 8-dimensional paradigm space.

Each paradigm is a POINT in an 8-dimensional coordinate system, not a discrete
category. Distances between points quantify the cost of bridging paradigms.
The topology of this space reveals mountains (hard boundaries), valleys (easy
bridges), and unexplored territories (paradigm vacancies).

Dimensions (each 0.0 to 1.0):
    state_magnitude:    0 = pure functional (immutable), 1 = unrestricted mutation
    control_explicitness: 0 = implicit (dataflow/logic), 1 = explicit (goto/sequence)
    typing_strength:    0 = dynamic/duck, 1 = dependent types
    composition_style:  0 = concatenative/pipeline, 1 = OO inheritance
    concurrency_model:  0 = sequential, 1 = actor model
    effect_tracking:    0 = pure/no tracking, 1 = full effect systems
    naming_style:       0 = positional, 1 = named/keyword
    abstraction_level:  0 = low-level/machine, 1 = very high-level/declarative
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Dimension Definitions
# ══════════════════════════════════════════════════════════════════

DIMENSION_NAMES: Tuple[str, ...] = (
    "state_magnitude",
    "control_explicitness",
    "typing_strength",
    "composition_style",
    "concurrency_model",
    "effect_tracking",
    "naming_style",
    "abstraction_level",
)

DIMENSION_DESCRIPTIONS: Dict[str, str] = {
    "state_magnitude":     "0=pure functional (immutable values), 1=unrestricted mutation",
    "control_explicitness": "0=implicit (dataflow/declarative), 1=explicit (goto/sequence)",
    "typing_strength":     "0=dynamic/duck typing, 1=dependent types/refinement types",
    "composition_style":   "0=concatenative/pipeline, 1=OO inheritance hierarchies",
    "concurrency_model":   "0=sequential/single-threaded, 1=actor model/message passing",
    "effect_tracking":     "0=pure/no effect tracking, 1=full algebraic effect systems",
    "naming_style":        "0=positional (arity-based), 1=named/keyword arguments",
    "abstraction_level":   "0=low-level/machine-close, 1=very high-level/declarative",
}

# Weight of each dimension in bridge cost calculation.
# Higher weight = more expensive to bridge across this axis.
DIMENSION_WEIGHTS: Dict[str, float] = {
    "state_magnitude":     1.5,   # Mutability ↔ immutability is very expensive
    "control_explicitness": 1.0,  # Control flow bridging is moderate
    "typing_strength":     1.2,   # Type system gaps are costly
    "composition_style":   0.8,   # Composition can be emulated more easily
    "concurrency_model":   1.4,   # Sequential ↔ concurrent is hard
    "effect_tracking":     1.3,   # Effect system mismatches are expensive
    "naming_style":        0.5,   # Positional ↔ named is cheap to bridge
    "abstraction_level":   1.0,   # Abstraction level affects all constructs
}


# ══════════════════════════════════════════════════════════════════
# Paradigm Point
# ═══════════════════════════DIMENSION_NAMES════════════════════════════════

@dataclass
class ParadigmPoint:
    """A point in 8-dimensional paradigm space.

    Attributes:
        coordinates:  Mapping from dimension name to value in [0.0, 1.0].
        name:         Human-readable paradigm name.
        examples:     Languages, systems, or natural languages at this point.
        notes:        Optional human-readable notes about the placement.
    """
    coordinates: Dict[str, float]
    name: str
    examples: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        # Ensure all 8 dimensions present, clamped to [0, 1]
        for dim in DIMENSION_NAMES:
            if dim not in self.coordinates:
                self.coordinates[dim] = 0.5  # neutral default
            else:
                self.coordinates[dim] = max(0.0, min(1.0, self.coordinates[dim]))

    def distance_to(self, other: ParadigmPoint, weighted: bool = True) -> float:
        """Euclidean distance in 8-D paradigm space.

        Args:
            other: Target paradigm point.
            weighted: If True, use dimension weights (emphasizes expensive axes).
        """
        total = 0.0
        for dim in DIMENSION_NAMES:
            delta = self.coordinates[dim] - other.coordinates[dim]
            w = DIMENSION_WEIGHTS.get(dim, 1.0) if weighted else 1.0
            total += (delta * w) ** 2
        return math.sqrt(total)

    def dimension_delta(self, other: ParadigmPoint) -> Dict[str, float]:
        """Per-dimension deltas (self - other)."""
        return {dim: self.coordinates[dim] - other.coordinates[dim]
                for dim in DIMENSION_NAMES}

    def centroid_of(self, others: List[ParadigmPoint]) -> ParadigmPoint:
        """Compute centroid of this point with others."""
        all_points = [self] + others
        n = len(all_points)
        coords = {}
        for dim in DIMENSION_NAMES:
            coords[dim] = sum(p.coordinates[dim] for p in all_points) / n
        return ParadigmPoint(
            coordinates=coords,
            name=f"centroid({self.name}+{n-1}others)",
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "coordinates": dict(self.coordinates),
            "examples": list(self.examples),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ParadigmPoint":
        return cls(
            coordinates=d["coordinates"],
            name=d["name"],
            examples=d.get("examples", []),
            notes=d.get("notes", ""),
        )

    def __repr__(self) -> str:
        coords = ", ".join(f"{d[:4]}={v:.2f}" for d, v in self.coordinates.items())
        return f"ParadigmPoint({self.name}, [{coords}])"


# ══════════════════════════════════════════════════════════════════
# Paradigm Point Data Tables
# ══════════════════════════════════════════════════════════════════

# Each entry: (name, coordinates, examples, notes)
_CLASSICAL_LANGUAGE_DATA: List[Tuple[str, Dict[str, float], List[str], str]] = [
    ("haskell",
     {"state_magnitude": 0.05, "control_explicitness": 0.3,
      "typing_strength": 0.95, "composition_style": 0.6,
      "concurrency_model": 0.4, "effect_tracking": 0.9,
      "naming_style": 0.7, "abstraction_level": 0.85},
     ["Haskell", "Idris", "Agda"],
     "Pure functional + type-class driven. Monads reify effects."),

    ("c",
     {"state_magnitude": 0.95, "control_explicitness": 0.95,
      "typing_strength": 0.1, "composition_style": 0.3,
      "concurrency_model": 0.2, "effect_tracking": 0.0,
      "naming_style": 0.2, "abstraction_level": 0.1},
     ["C", "Assembly"],
     "Raw von Neumann. Manual memory. No safety net."),

    ("prolog",
     {"state_magnitude": 0.2, "control_explicitness": 0.3,
      "typing_strength": 0.5, "composition_style": 0.5,
      "concurrency_model": 0.3, "effect_tracking": 0.1,
      "naming_style": 0.8, "abstraction_level": 0.7},
     ["Prolog", "Mercury", "Datalog"],
     "Logic programming via unification/backtracking. Cut = imperative leak."),

    ("rust",
     {"state_magnitude": 0.5, "control_explicitness": 0.8,
      "typing_strength": 0.85, "composition_style": 0.4,
      "concurrency_model": 0.6, "effect_tracking": 0.6,
      "naming_style": 0.4, "abstraction_level": 0.35},
     ["Rust", "Zig"],
     "Ownership = affine types. Borrow checker enforces resource protocol."),

    ("python",
     {"state_magnitude": 0.7, "control_explicitness": 0.6,
      "typing_strength": 0.25, "composition_style": 0.5,
      "concurrency_model": 0.3, "effect_tracking": 0.05,
      "naming_style": 0.8, "abstraction_level": 0.65},
     ["Python", "Ruby", "JavaScript"],
     "Dynamic + OO + multi-paradigm. Batteries-included pragmatism."),

    ("lisp",
     {"state_magnitude": 0.6, "control_explicitness": 0.5,
      "typing_strength": 0.3, "composition_style": 0.2,
      "concurrency_model": 0.5, "effect_tracking": 0.2,
      "naming_style": 0.5, "abstraction_level": 0.8},
     ["Common Lisp", "Scheme", "Racket", "Clojure"],
     "Homoiconic metaprogramming. Macros define paradigms. Code = data."),

    ("apl",
     {"state_magnitude": 0.3, "control_explicitness": 0.2,
      "typing_strength": 0.4, "composition_style": 0.1,
      "concurrency_model": 0.3, "effect_tracking": 0.1,
      "naming_style": 0.1, "abstraction_level": 0.7},
     ["APL", "J", "K", "Q"],
     "Array programming. Extreme information density. Tacit (point-free)."),

    ("forth",
     {"state_magnitude": 0.8, "control_explicitness": 0.9,
      "typing_strength": 0.0, "composition_style": 0.0,
      "concurrency_model": 0.1, "effect_tracking": 0.0,
      "naming_style": 0.0, "abstraction_level": 0.2},
     ["Forth", "Factor", "PostScript", "Joy"],
     "Concatenative. Stack-based. Syntax IS composition."),

    ("smalltalk",
     {"state_magnitude": 0.7, "control_explicitness": 0.5,
      "typing_strength": 0.15, "composition_style": 0.95,
      "concurrency_model": 0.3, "effect_tracking": 0.1,
      "naming_style": 0.7, "abstraction_level": 0.8},
     ["Smalltalk", "Self", "Io"],
     "Pure OO. Everything is an object. Message-passing paradigm."),

    ("java",
     {"state_magnitude": 0.8, "control_explicitness": 0.7,
      "typing_strength": 0.6, "composition_style": 0.9,
      "concurrency_model": 0.5, "effect_tracking": 0.15,
      "naming_style": 0.6, "abstraction_level": 0.5},
     ["Java", "C#", "C++"],
     "Static OO. Verbose. Enterprise-scale. Checked exceptions."),
]

_NL_PARADIGM_DATA: List[Tuple[str, Dict[str, float], List[str], str]] = [
    ("zho",
     {"state_magnitude": 0.5, "control_explicitness": 0.4,
      "typing_strength": 0.55, "composition_style": 0.3,
      "concurrency_model": 0.5, "effect_tracking": 0.35,
      "naming_style": 0.6, "abstraction_level": 0.7},
     ["flux_zho", "Mandarin Chinese"],
     ("Chinese: classifier system = grammatical type system. "
      "Topic-comment structure = zero-anaphora implicit threading. "
      "Context-dependent pronoun resolution = confidence-weighted inference. "
      "Honorific markers → capability annotations.")),

    ("kor",
     {"state_magnitude": 0.4, "control_explicitness": 0.5,
      "typing_strength": 0.5, "composition_style": 0.4,
      "concurrency_model": 0.6, "effect_tracking": 0.65,
      "naming_style": 0.75, "abstraction_level": 0.65},
     ["flux_kor", "Korean"],
     ("Korean: 7-level honorific system = capability/security hierarchy. "
      "Speech levels (하십시오체/해요체/해체 etc.) → CAP_REQUIRE levels. "
      "Subject-honorific + object-honorific → trust-directed communication. "
      "Particle system (은/는, 이/가) → scope/topic markers.")),

    ("san",
     {"state_magnitude": 0.3, "control_explicitness": 0.35,
      "typing_strength": 0.7, "composition_style": 0.3,
      "concurrency_model": 0.45, "effect_tracking": 0.6,
      "naming_style": 0.85, "abstraction_level": 0.8},
     ["flux_san", "Sanskrit"],
     ("Sanskrit: 8-case vibhakti system = 8-level scope hierarchy. "
      "Each case (प्रथमा to संबोधन) maps to a ScopeLevel opcode. "
      "3 genders + 3 numbers = polymorphic type system. "
      "Sandhi rules = automatic word-boundary fusion. "
      "Paninian grammar = world's first formal grammar.")),

    ("deu",
     {"state_magnitude": 0.6, "control_explicitness": 0.7,
      "typing_strength": 0.65, "composition_style": 0.6,
      "concurrency_model": 0.4, "effect_tracking": 0.4,
      "naming_style": 0.7, "abstraction_level": 0.55},
     ["flux_deu", "German"],
     ("German: 4-case Kasus system (Nominativ/Akkusativ/Dativ/Genitiv) = scope. "
      "Compound nouns = type composition (Donaudampfschifffahrt...). "
      "Verb-second (V2) word order = rigid control flow. "
      "Separable-prefix verbs = deferred computation. "
      "Gender system (der/die/das) = nominal type annotations.")),

    ("wen",
     {"state_magnitude": 0.35, "control_explicitness": 0.25,
      "typing_strength": 0.45, "composition_style": 0.15,
      "concurrency_model": 0.35, "effect_tracking": 0.5,
      "naming_style": 0.3, "abstraction_level": 0.85},
     ["flux_wen", "Classical Chinese (文言文)"],
     ("Classical Chinese: extreme information density = minimal syntax. "
      "Topic-comment = zero-anaphora continuation. "
      "Confucian 五常 (五德) → capability/trust system (仁義禮智信). "
      "Sun Tzu 兵法 → attack/defend/advance/retreat opcodes. "
      "No inflection = positional/pipeline composition. "
      "S-V-O flexibility = dataflow rather than strict sequence.")),

    ("lat",
     {"state_magnitude": 0.45, "control_explicitness": 0.6,
      "typing_strength": 0.6, "composition_style": 0.45,
      "concurrency_model": 0.35, "effect_tracking": 0.55,
      "naming_style": 0.8, "abstraction_level": 0.7},
     ["flux_lat", "Latin"],
     ("Latin: 6-case system + 6-tense system = spatial-temporal scope. "
      "Tenses (Present/Imperfect/Perfect/Pluperfect/Future/Fut.Perfect) "
      "= temporal aspect opcodes (LOOP_START/ROLLBACK_SAVE/LAZY_DEFER etc.). "
      "Case system (Nom/Acc/Gen/Dat/Abl/Voc) = scope levels + invocation. "
      "Word-order freedom = named/positional flexibility. "
      "Subjunctive mood = modal/confidence system.")),
]


# ══════════════════════════════════════════════════════════════════
# Paradigm Lattice
# ══════════════════════════════════════════════════════════════════

class ParadigmLattice:
    """8-dimensional model where paradigms are POINTS, not categories.

    Provides:
      - Named access to pre-populated paradigm points (classical PLs + FLUX NLs)
      - Distance matrix computation
      - Nearest-neighbor queries
      - Convex hull / centroid calculations
      - Vacancy detection (empty regions of paradigm space)
    """

    def __init__(self) -> None:
        self._points: Dict[str, ParadigmPoint] = {}
        self._populate()

    # ── Population ───────────────────────────────────────────────

    @property
    def points(self) -> Dict[str, "ParadigmPoint"]:
        return self._points

    def _populate(self) -> None:
        """Register all canonical paradigm points."""
        self._populate_classical_languages()
        self._populate_nl_paradigms()

    def _populate_classical_languages(self) -> None:
        """Register classical programming language paradigm points."""
        for name, coords, examples, notes in _CLASSICAL_LANGUAGE_DATA:
            self.add(ParadigmPoint(
                name=name,
                coordinates=coords,
                examples=examples,
                notes=notes,
            ))

    def _populate_nl_paradigms(self) -> None:
        """Register FLUX natural language paradigm points."""
        for name, coords, examples, notes in _NL_PARADIGM_DATA:
            self.add(ParadigmPoint(
                name=name,
                coordinates=coords,
                examples=examples,
                notes=notes,
            ))

    # ── Point Access ────────────────────────────────────────────

    def add(self, point: ParadigmPoint) -> None:
        """Register a paradigm point by name."""
        self._points[point.name] = point

    def get(self, name: str) -> ParadigmPoint:
        """Retrieve a paradigm point by name."""
        if name not in self._points:
            raise KeyError(f"Unknown paradigm point: {name!r}. "
                           f"Available: {sorted(self._points.keys())}")
        return self._points[name]

    def all_points(self) -> List[ParadigmPoint]:
        """Return all registered paradigm points."""
        return list(self._points.values())

    def nl_points(self) -> List[ParadigmPoint]:
        """Return only the FLUX natural-language paradigm points."""
        nl_names = {"zho", "kor", "san", "deu", "wen", "lat"}
        return [p for p in self.all_points() if p.name in nl_names]

    def classical_points(self) -> List[ParadigmPoint]:
        """Return the classical programming language paradigm points."""
        nl_names = {"zho", "kor", "san", "deu", "wen", "lat"}
        return [p for p in self.all_points() if p.name not in nl_names]

    # ── Distance Queries ────────────────────────────────────────

    def distance(self, a: str, b: str, weighted: bool = True) -> float:
        """Distance between two named paradigm points."""
        return self.get(a).distance_to(self.get(b), weighted=weighted)

    def distance_matrix(
        self, names: Optional[List[str]] = None, weighted: bool = True
    ) -> Dict[Tuple[str, str], float]:
        """Compute all-pairs distance matrix for named points."""
        if names is None:
            names = sorted(self._points.keys())
        matrix: Dict[Tuple[str, str], float] = {}
        for a in names:
            for b in names:
                matrix[(a, b)] = self.distance(a, b, weighted=weighted)
        return matrix

    def nearest_neighbors(
        self, name: str, k: int = 5, weighted: bool = True
    ) -> List[Tuple[str, float]]:
        """Find k nearest paradigm points to the named point."""
        target = self.get(name)
        distances = [
            (p.name, target.distance_to(p, weighted=weighted))
            for p in self.all_points() if p.name != name
        ]
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ── Centroid / Hub Analysis ─────────────────────────────────

    def centroid(self, names: List[str]) -> ParadigmPoint:
        """Compute the centroid of a group of paradigm points."""
        points = [self.get(n) for n in names]
        n = len(points)
        coords = {}
        for dim in DIMENSION_NAMES:
            coords[dim] = sum(p.coordinates[dim] for p in points) / n
        return ParadigmPoint(
            coordinates=coords,
            name=f"centroid({'+'.join(names)})",
        )

    def best_hub(
        self, candidates: Optional[List[str]] = None,
        weighted: bool = True,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Find the paradigm point that minimizes total distance to all others.

        Returns:
            (hub_name, average_distance, distances_to_all)
        """
        if candidates is None:
            candidates = list(self._points.keys())
        best_name = ""
        best_avg = float("inf")
        best_distances: Dict[str, float] = {}
        for candidate in candidates:
            distances = {
                other: self.distance(candidate, other, weighted=weighted)
                for other in candidates if other != candidate
            }
            avg = sum(distances.values()) / len(distances)
            if avg < best_avg:
                best_avg = avg
                best_name = candidate
                best_distances = distances
        return best_name, best_avg, best_distances

    def total_diameter(self, names: Optional[List[str]] = None,
                       weighted: bool = True) -> float:
        """Maximum pairwise distance (diameter) of the point set."""
        if names is None:
            names = sorted(self._points.keys())
        max_dist = 0.0
        for i, a in enumerate(names):
            for b in names[i+1:]:
                d = self.distance(a, b, weighted=weighted)
                if d > max_dist:
                    max_dist = d
        return max_dist

    # ── Vacancy Detection ───────────────────────────────────────

    def detect_vacancies(
        self, resolution: float = 0.2, weighted: bool = True
    ) -> List[ParadigmPoint]:
        """Identify regions of paradigm space with no nearby points.

        Samples the 8-D hypercube on a grid and reports cells whose
        nearest occupied point is farther than a threshold.

        Args:
            resolution: Grid spacing (smaller = finer search, more expensive).
            weighted: Whether to use weighted distance.

        Returns:
            List of ParadigmPoint candidates representing vacant regions.
        """
        threshold = resolution * 1.5  # cells with no point within this range
        vacant = self._sample_vacancy_candidates(resolution, weighted, threshold)
        # Deduplicate: cluster vacancies that are close to each other
        clustered = self._cluster_vacancies(vacant, radius=resolution * 2)
        return clustered

    def _sample_vacancy_candidates(
        self, resolution: float, weighted: bool, threshold: float,
        sample_count: int = 5000,
    ) -> List[ParadigmPoint]:
        """Sample random candidate points and return those far from any occupied point."""
        import random
        random.seed(42)

        occupied = self.all_points()
        vacant = []

        for _ in range(sample_count):
            sample_coords = {
                dim: random.random()
                for dim in DIMENSION_NAMES
            }
            sample = ParadigmPoint(
                coordinates=sample_coords,
                name="vacancy_sample",
            )
            nearest_dist = min(
                sample.distance_to(p, weighted=weighted) for p in occupied
            )

            if nearest_dist > threshold:
                vacant.append(ParadigmPoint(
                    coordinates=dict(sample_coords),
                    name=f"vacancy(nearest={nearest_dist:.2f})",
                    notes=f"Nearest occupied point at distance {nearest_dist:.2f}",
                ))

        return vacant

    def _cluster_vacancies(
        self, vacancies: List[ParadigmPoint], radius: float = 0.4
    ) -> List[ParadigmPoint]:
        """Cluster nearby vacancy candidates into representative points."""
        if not vacancies:
            return []
        clusters: List[List[ParadigmPoint]] = []
        for v in vacancies:
            assigned = False
            for cluster in clusters:
                centroid = cluster[0]
                if v.distance_to(centroid, weighted=False) < radius:
                    cluster.append(v)
                    assigned = True
                    break
            if not assigned:
                clusters.append([v])
        # Return centroid of each cluster
        results = []
        for cluster in clusters:
            n = len(cluster)
            coords = {}
            for dim in DIMENSION_NAMES:
                coords[dim] = sum(p.coordinates[dim] for p in cluster) / n
            results.append(ParadigmPoint(
                coordinates=coords,
                name=f"vacancy_cluster(n={n})",
                notes=f"Cluster of {n} vacant samples",
            ))
        return sorted(results, key=lambda p: -len(p.notes.split("of ")[-1].split(" ")[0]))

    # ── Serialization ───────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "dimensions": {dim: DIMENSION_DESCRIPTIONS[dim] for dim in DIMENSION_NAMES},
            "weights": dict(DIMENSION_WEIGHTS),
            "points": {name: p.to_dict() for name, p in self._points.items()},
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the lattice."""
        lines = [
            "ParadigmLattice — 8-dimensional paradigm space",
            f"  Registered points: {len(self._points)}",
            f"  Dimensions: {', '.join(DIMENSION_NAMES)}",
            "",
            "  Natural Language paradigms:",
        ]
        for p in self.nl_points():
            lines.append(f"    {p.name}: {p.examples}")
        lines.append("")
        lines.append("  Classical PL paradigms:")
        for p in self.classical_points():
            lines.append(f"    {p.name}: {p.examples}")

        hub_name, hub_avg, _ = self.best_hub(weighted=True)
        nl_hub, nl_avg, _ = self.best_hub(
            candidates=[p.name for p in self.nl_points()], weighted=True
        )
        diameter = self.total_diameter(weighted=True)

        lines.extend([
            "",
            f"  Overall hub (lowest avg distance): {hub_name} (avg={hub_avg:.3f})",
            f"  NL-only hub: {nl_hub} (avg={nl_avg:.3f})",
            f"  Total diameter: {diameter:.3f}",
        ])
        return "\n".join(lines)
