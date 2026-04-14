"""
Unified Vocabulary System for FLUX Multilingual Runtime.

Implements a hierarchical concept ontology that maps programming concepts
across all 6 FLUX languages (ZHO, DEU, KOR, SAN, LAT, WEN), enabling
consistent cross-language term resolution, search, and type bridging.

Architecture:
    ConceptOntology
      └── ConceptNode (50+ nodes across 6 semantic domains)
    UnifiedVocabulary
      ├── lookup(term, lang) → [ConceptNode]
      ├── translate(term, src, tgt) → [str]
      ├── find_concept(id) → ConceptNode
      ├── search(query, lang) → [ConceptNode]
      ├── cross_language_synonyms(id) → {lang: [terms]}
      └── concept_distance(a, b) → float
    VocabularyBridge
      ├── concept_to_type(id, lang) → FluxType
      ├── type_to_concept(flux_type) → [ConceptNode]
      └── vocabulary_aware_translate(term, src, tgt) → TranslationResult

Semantic Domains:
    execution  — Agents, functions, control flow (15 concepts)
    data       — Values, references, containers (10 concepts)
    scope      — Accessibility, visibility, environments (8 concepts)
    communication — Messages, events, signals (8 concepts)
    temporal   — Time, duration, scheduling (5 concepts)
    meta       — Types, kinds, classes, traits (4 concepts)

Reference: type_safe_bridge.py (TypeAlgebra), types.py (FUTS)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from flux_a2a.types import (
    FluxBaseType,
    FluxConstraint,
    FluxType,
)


# ══════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════

ALL_LANGUAGES: list[str] = ["zho", "deu", "kor", "san", "lat", "wen"]

ALL_DOMAINS: list[str] = [
    "execution", "data", "scope", "communication", "temporal", "meta",
]


# ══════════════════════════════════════════════════════════════════
# 1. ConceptNode — A single concept in the ontology
# ══════════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    """A hierarchical concept node in the unified vocabulary.

    Each ConceptNode represents a programming concept with native terms
    in all 6 FLUX languages, organized into a semantic domain with
    optional parent-child relationships.

    Attributes:
        concept_id: Unique identifier (e.g., "agent_action", "data_container").
        universal_name: English canonical name.
        description: What this concept represents.
        semantic_domain: One of execution, data, scope, communication,
            temporal, meta.
        children: IDs of sub-concepts (narrower concepts).
        language_terms: Mapping from language tag to list of native terms.
            Each language should have at least one term.
        paradigm_traits: Language-paradigm-specific metadata (e.g., whether
            a concept is realized via inflection in SAN vs compounding in DEU).
        parent: Optional ID of the parent concept (broader concept).
    """
    concept_id: str
    universal_name: str
    description: str
    semantic_domain: str
    children: list[str] = field(default_factory=list)
    language_terms: dict[str, list[str]] = field(default_factory=dict)
    paradigm_traits: dict[str, Any] = field(default_factory=dict)
    parent: str = ""

    def get_term(self, lang: str) -> str:
        """Get the primary (first) term for a language.

        Args:
            lang: Language tag.

        Returns:
            The first native term, or the universal_name as fallback.
        """
        terms = self.language_terms.get(lang, [])
        return terms[0] if terms else self.universal_name

    def has_language(self, lang: str) -> bool:
        """Check whether this concept has terms in the given language."""
        return lang in self.language_terms and len(self.language_terms[lang]) > 0

    def all_terms(self) -> list[str]:
        """Collect all terms across all languages."""
        result: list[str] = []
        for terms in self.language_terms.values():
            result.extend(terms)
        return result

    def covered_languages(self) -> list[str]:
        """Return list of languages that have terms for this concept."""
        return [lang for lang, terms in self.language_terms.items() if terms]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "concept_id": self.concept_id,
            "universal_name": self.universal_name,
            "description": self.description,
            "semantic_domain": self.semantic_domain,
            "children": self.children,
            "language_terms": self.language_terms,
            "paradigm_traits": self.paradigm_traits,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptNode:
        """Deserialize from dictionary."""
        return cls(
            concept_id=data["concept_id"],
            universal_name=data["universal_name"],
            description=data["description"],
            semantic_domain=data["semantic_domain"],
            children=data.get("children", []),
            language_terms=data.get("language_terms", {}),
            paradigm_traits=data.get("paradigm_traits", {}),
            parent=data.get("parent", ""),
        )


# ══════════════════════════════════════════════════════════════════
# 2. ConceptOntology — Hierarchical concept system with 50+ concepts
# ══════════════════════════════════════════════════════════════════

def _build_ontology() -> Dict[str, ConceptNode]:
    """Build the complete ontology from JSON data file."""
    _data_path = Path(__file__).parent / "data" / "ontology.json"
    with open(_data_path, "r", encoding="utf-8") as _f:
        _raw = json.load(_f)
    return {k: ConceptNode.from_dict(v) for k, v in _raw.items()}




# ══════════════════════════════════════════════════════════════════
# 3. UnifiedVocabulary — Main query interface
# ══════════════════════════════════════════════════════════════════

class UnifiedVocabulary:
    """Main query interface for the unified cross-language vocabulary.

    Provides lookup, translation, search, and distance computation
    over the concept ontology.  All queries are case-sensitive for
    native terms (as appropriate for each language) and case-insensitive
    for concept IDs and universal names.

    Usage:
        vocab = UnifiedVocabulary()
        # Look up concepts by native term
        nodes = vocab.lookup("循环", "zho")
        # Translate across languages
        terms = vocab.translate("循环", "zho", "deu")
        # Compute semantic distance
        dist = vocab.concept_distance("loop", "value")
    """

    def __init__(self, concepts: Optional[Dict[str, ConceptNode]] = None) -> None:
        """Initialize the vocabulary with the given concept map.

        Args:
            concepts: Optional pre-built concept dict. If None, builds
                the default 50-concept ontology.
        """
        self._concepts: Dict[str, ConceptNode] = concepts or _build_ontology()
        # Build reverse index: (lang, term) -> [concept_id]
        self._term_index: Dict[tuple[str, str], list[str]] = {}
        # Build description keyword index
        self._desc_index: Dict[str, list[str]] = {}
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild the term and description indices."""
        self._term_index.clear()
        self._desc_index.clear()
        for cid, node in self._concepts.items():
            for lang, terms in node.language_terms.items():
                for term in terms:
                    key = (lang, term)
                    if key not in self._term_index:
                        self._term_index[key] = []
                    self._term_index[key].append(cid)
            # Index description keywords
            for word in node.description.split():
                w = word.lower().strip(".,;:!?()")
                if len(w) > 2:
                    if w not in self._desc_index:
                        self._desc_index[w] = []
                    self._desc_index[w].append(cid)

    # ── Core query methods ────────────────────────────────────────

    def lookup(self, term: str, source_lang: str) -> list[ConceptNode]:
        """Look up concepts by a native term in a specific language.

        Args:
            term: The native term to look up.
            source_lang: Language tag (e.g., "zho", "deu").

        Returns:
            List of ConceptNode instances that have this term.
            Returns an empty list if no match is found.
        """
        concept_ids = self._term_index.get((source_lang, term), [])
        return [self._concepts[cid] for cid in concept_ids if cid in self._concepts]

    def translate(
        self, term: str, source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate a term from one language to another via concept mapping.

        Looks up the term in the source language, finds matching concepts,
        and returns the native terms from those concepts in the target language.

        Args:
            term: The native term in the source language.
            source_lang: Source language tag.
            target_lang: Target language tag.

        Returns:
            List of translated terms. Empty list if no translation found.
        """
        nodes = self.lookup(term, source_lang)
        result: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            for t in node.language_terms.get(target_lang, []):
                if t not in seen:
                    seen.add(t)
                    result.append(t)
        return result

    def find_concept(self, concept_id: str) -> Optional[ConceptNode]:
        """Find a concept by its ID.

        Args:
            concept_id: The unique concept identifier.

        Returns:
            The ConceptNode, or None if not found.
        """
        return self._concepts.get(concept_id)

    def search(self, query: str, lang: str = "") -> list[ConceptNode]:
        """Search concepts by term, concept_id, or description keyword.

        Searches in three modes (union):
        1. Exact term match in the given language
        2. Partial match on concept_id
        3. Keyword match in description

        Args:
            query: Search query string.
            lang: Optional language tag for term search.

        Returns:
            List of matching ConceptNode instances, ranked by relevance.
        """
        scores: Dict[str, float] = {}
        query_lower = query.lower()

        # 1. Exact term match (highest score)
        if lang:
            for (l, t), cids in self._term_index.items():
                if l == lang and t == query:
                    for cid in cids:
                        scores[cid] = scores.get(cid, 0.0) + 10.0

        # 2. Term contains match
        if lang:
            for (l, t), cids in self._term_index.items():
                if l == lang and query in t:
                    for cid in cids:
                        scores[cid] = scores.get(cid, 0.0) + 5.0

        # 3. Concept ID match
        for cid in self._concepts:
            if query_lower in cid.lower():
                scores[cid] = scores.get(cid, 0.0) + 7.0

        # 4. Universal name match
        for cid, node in self._concepts.items():
            if query_lower in node.universal_name.lower():
                scores[cid] = scores.get(cid, 0.0) + 6.0

        # 5. Description keyword match
        for word in query.split():
            w = word.lower().strip(".,;:!?()")
            if len(w) > 2 and w in self._desc_index:
                for cid in self._desc_index[w]:
                    scores[cid] = scores.get(cid, 0.0) + 1.0

        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return [self._concepts[cid] for cid in sorted_ids if cid in self._concepts]

    def cross_language_synonyms(self, concept_id: str) -> dict[str, list[str]]:
        """Get all native terms for a concept across all languages.

        Args:
            concept_id: The concept identifier.

        Returns:
            Dict mapping language tag to list of native terms.
            Returns empty dict if concept not found.
        """
        node = self._concepts.get(concept_id)
        if node is None:
            return {}
        return dict(node.language_terms)

    def concept_distance(self, a: str, b: str) -> float:
        """Compute semantic distance between two concepts.

        Distance is in [0.0, 1.0] where 0.0 = identical and 1.0 = unrelated.

        The distance is computed as a weighted combination of:
        - Same domain bonus (0.0 if same domain, 0.3 if different)
        - Hierarchy distance (number of parent/child hops)
        - Description similarity (keyword overlap)

        Args:
            a: Concept ID of the first concept.
            b: Concept ID of the second concept.

        Returns:
            Float distance in [0.0, 1.0].
        """
        if a == b:
            return 0.0

        node_a = self._concepts.get(a)
        node_b = self._concepts.get(b)

        if node_a is None or node_b is None:
            return 1.0

        # Factor 1: Same domain bonus
        domain_penalty = 0.0 if node_a.semantic_domain == node_b.semantic_domain else 0.3

        # Factor 2: Hierarchy distance
        hierarchy_dist = self._hierarchy_distance(a, b)
        hierarchy_penalty = min(hierarchy_dist * 0.15, 0.5)

        # Factor 3: Description keyword overlap
        desc_penalty = self._description_distance(node_a, node_b) * 0.2

        total = domain_penalty + hierarchy_penalty + desc_penalty
        return min(total, 1.0)

    def _hierarchy_distance(self, a: str, b: str) -> int:
        """Count the minimum number of hops in the parent-child hierarchy."""
        if a == b:
            return 0

        # Build ancestor sets
        ancestors_a = self._get_ancestors(a)
        ancestors_b = self._get_ancestors(b)

        # Check if b is an ancestor of a (or vice versa)
        if b in ancestors_a:
            return ancestors_a[b]
        if a in ancestors_b:
            return ancestors_b[a]

        # Find lowest common ancestor
        common = set(ancestors_a.keys()) & set(ancestors_b.keys())
        if common:
            lca = min(common, key=lambda c: ancestors_a[c] + ancestors_b[c])
            return ancestors_a[lca] + ancestors_b[lca]

        # No common ancestor — unrelated
        return 10

    def _get_ancestors(self, concept_id: str) -> dict[str, int]:
        """Get all ancestors of a concept with their hop distances."""
        ancestors: dict[str, int] = {}
        current = concept_id
        distance = 0
        while current:
            parent_node = self._concepts.get(current)
            if parent_node is None or not parent_node.parent:
                break
            distance += 1
            ancestors[parent_node.parent] = distance
            current = parent_node.parent
        return ancestors

    def _description_distance(self, a: ConceptNode, b: ConceptNode) -> float:
        """Compute keyword-based distance between two concept descriptions."""
        words_a = set(a.description.lower().split())
        words_b = set(b.description.lower().split())
        if not words_a or not words_b:
            return 1.0
        # Remove stop words
        stop_words = {
            "a", "an", "the", "of", "to", "in", "for", "that", "is",
            "or", "and", "by", "its", "their", "a", "from", "with",
        }
        words_a -= stop_words
        words_b -= stop_words
        if not words_a or not words_b:
            return 1.0
        overlap = len(words_a & words_b)
        total = len(words_a | words_b)
        return 1.0 - (overlap / total) if total > 0 else 1.0

    # ── Inspection methods ────────────────────────────────────────

    def all_concepts(self) -> list[ConceptNode]:
        """Return all registered concepts."""
        return list(self._concepts.values())

    def concepts_by_domain(self, domain: str) -> list[ConceptNode]:
        """Return concepts in a specific semantic domain."""
        return [
            n for n in self._concepts.values()
            if n.semantic_domain == domain
        ]

    def domain_counts(self) -> dict[str, int]:
        """Count concepts per domain."""
        counts: dict[str, int] = {}
        for node in self._concepts.values():
            counts[node.semantic_domain] = counts.get(node.semantic_domain, 0) + 1
        return counts

    def total_concepts(self) -> int:
        """Return total number of registered concepts."""
        return len(self._concepts)

    def language_coverage(self) -> dict[str, int]:
        """Count how many concepts have terms in each language."""
        coverage: dict[str, int] = {}
        for node in self._concepts.values():
            for lang in node.covered_languages():
                coverage[lang] = coverage.get(lang, 0) + 1
        return coverage


# ══════════════════════════════════════════════════════════════════
# 4. TranslationResult — Result of vocabulary-aware translation
# ══════════════════════════════════════════════════════════════════

@dataclass
class TranslationResult:
    """Result of a vocabulary-aware translation.

    Attributes:
        source_term: The original term.
        source_lang: Source language tag.
        target_lang: Target language tag.
        target_terms: List of translated terms, ordered by confidence.
        confidence: Overall confidence score [0.0, 1.0].
        concept_id: The concept ID used for translation (if found).
        ambiguity: Number of candidate concepts matched.
        notes: Additional metadata or warnings.
    """
    source_term: str
    source_lang: str
    target_lang: str
    target_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0
    concept_id: str = ""
    ambiguity: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_term": self.source_term,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "target_terms": self.target_terms,
            "confidence": round(self.confidence, 4),
            "concept_id": self.concept_id,
            "ambiguity": self.ambiguity,
            "notes": self.notes,
        }


# ══════════════════════════════════════════════════════════════════
# 5. VocabularyBridge — Bridges vocabulary to the type system
# ══════════════════════════════════════════════════════════════════

# Mapping from semantic domain to the most likely FluxBaseType.
_DOMAIN_TO_BASE_TYPE: Dict[str, FluxBaseType] = {
    "execution": FluxBaseType.ACTIVE,
    "data": FluxBaseType.VALUE,
    "scope": FluxBaseType.SCOPE,
    "communication": FluxBaseType.CAPABILITY,
    "temporal": FluxBaseType.MODAL,
    "meta": FluxBaseType.CONTEXTUAL,
}

# Mapping from specific concept IDs to more precise base types.
_CONCEPT_BASE_TYPE_OVERRIDES: Dict[str, FluxBaseType] = {
    # Execution concepts
    "agent_action": FluxBaseType.ACTIVE,
    "function": FluxBaseType.ACTIVE,
    "application": FluxBaseType.MODAL,
    "abstraction": FluxBaseType.CONTEXTUAL,
    "conditional": FluxBaseType.MODAL,
    "loop": FluxBaseType.MODAL,
    "sequence": FluxBaseType.MODAL,
    "parallel": FluxBaseType.MODAL,
    "fork": FluxBaseType.MODAL,
    "join": FluxBaseType.MODAL,
    "pipeline": FluxBaseType.MODAL,
    "chain": FluxBaseType.MODAL,
    "map": FluxBaseType.ACTIVE,
    "filter": FluxBaseType.ACTIVE,
    "reduce": FluxBaseType.ACTIVE,
    # Data concepts
    "value": FluxBaseType.VALUE,
    "reference": FluxBaseType.SCOPE,
    "container": FluxBaseType.CONTAINER,
    "collection": FluxBaseType.CONTAINER,
    "pair": FluxBaseType.CONTAINER,
    "option": FluxBaseType.UNCERTAIN,
    "result": FluxBaseType.UNCERTAIN,
    "stream": FluxBaseType.VALUE,
    "buffer": FluxBaseType.CONTAINER,
    "channel": FluxBaseType.CAPABILITY,
    # Scope concepts
    "global": FluxBaseType.SCOPE,
    "local": FluxBaseType.SCOPE,
    "closure": FluxBaseType.CONTEXTUAL,
    "environment": FluxBaseType.CONTEXTUAL,
    "stack": FluxBaseType.SCOPE,
    "heap": FluxBaseType.SCOPE,
    "namespace": FluxBaseType.SCOPE,
    "module": FluxBaseType.SCOPE,
    # Communication concepts
    "message": FluxBaseType.CAPABILITY,
    "request": FluxBaseType.CAPABILITY,
    "response": FluxBaseType.CAPABILITY,
    "event": FluxBaseType.MODAL,
    "signal": FluxBaseType.CAPABILITY,
    "broadcast": FluxBaseType.CAPABILITY,
    "subscribe": FluxBaseType.CAPABILITY,
    "publish": FluxBaseType.CAPABILITY,
    # Temporal concepts
    "instant": FluxBaseType.MODAL,
    "duration": FluxBaseType.MODAL,
    "interval": FluxBaseType.MODAL,
    "schedule": FluxBaseType.MODAL,
    "deadline": FluxBaseType.MODAL,
    # Meta concepts
    "type": FluxBaseType.CONTEXTUAL,
    "kind": FluxBaseType.CONTEXTUAL,
    "class": FluxBaseType.CONTEXTUAL,
    "trait": FluxBaseType.CAPABILITY,
}


class VocabularyBridge:
    """Bridge between the unified vocabulary and the FUTS type system.

    Provides bidirectional mapping between vocabulary concepts and FluxTypes,
    enabling vocabulary-aware cross-language translation that respects type
    constraints.

    Usage:
        bridge = VocabularyBridge(vocab)
        flux_type = bridge.concept_to_type("agent_action", "zho")
        concepts = bridge.type_to_concept(flux_type)
        result = bridge.vocabulary_aware_translate("循环", "zho", "deu")
    """

    def __init__(self, vocabulary: Optional[UnifiedVocabulary] = None) -> None:
        """Initialize with a vocabulary instance.

        Args:
            vocabulary: Optional pre-built vocabulary. If None, creates
                a default one.
        """
        self.vocabulary = vocabulary or UnifiedVocabulary()
        # Build reverse index: FluxBaseType -> [concept_id]
        self._type_to_concepts: Dict[FluxBaseType, list[str]] = {}
        self._build_type_index()

    def _build_type_index(self) -> None:
        """Build a reverse mapping from FluxBaseType to concept IDs."""
        self._type_to_concepts.clear()
        for cid, node in self.vocabulary._concepts.items():
            base = _CONCEPT_BASE_TYPE_OVERRIDES.get(
                cid, _DOMAIN_TO_BASE_TYPE.get(node.semantic_domain, FluxBaseType.VALUE)
            )
            if base not in self._type_to_concepts:
                self._type_to_concepts[base] = []
            self._type_to_concepts[base].append(cid)

    def concept_to_type(self, concept_id: str, lang: str = "") -> FluxType:
        """Map a concept to its corresponding FluxType.

        The mapping considers both the semantic domain of the concept
        and any concept-specific overrides.

        Args:
            concept_id: The concept identifier.
            lang: Optional language tag for paradigm-specific constraints.

        Returns:
            A FluxType representing this concept.

        Raises:
            KeyError: If concept_id is not found.
        """
        node = self.vocabulary.find_concept(concept_id)
        if node is None:
            raise KeyError(f"Unknown concept: {concept_id}")

        base_type = _CONCEPT_BASE_TYPE_OVERRIDES.get(
            concept_id,
            _DOMAIN_TO_BASE_TYPE.get(node.semantic_domain, FluxBaseType.VALUE),
        )

        constraints: list[FluxConstraint] = []
        if lang:
            constraints.append(FluxConstraint(
                kind=ConstraintKind.CONTEXT_DOMAIN,
                language=lang,
                value=concept_id,
                confidence=0.9,
            ))

        return FluxType(
            base_type=base_type,
            constraints=constraints,
            confidence=0.85,
            paradigm_source=lang or "flux",
            name=f"vocab:{concept_id}",
            meta={"concept_id": concept_id, "semantic_domain": node.semantic_domain},
        )

    def type_to_concept(self, flux_type: FluxType) -> list[ConceptNode]:
        """Map a FluxType to matching vocabulary concepts.

        Finds concepts whose associated FluxBaseType matches the given
        type's effective base type.

        Args:
            flux_type: The FluxType to map.

        Returns:
            List of matching ConceptNode instances, ordered by relevance.
        """
        effective = flux_type.effective_base_type()
        concept_ids = self._type_to_concepts.get(effective, [])

        result: list[ConceptNode] = []
        for cid in concept_ids:
            node = self.vocabulary.find_concept(cid)
            if node is not None:
                result.append(node)
        return result

    def vocabulary_aware_translate(
        self, term: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """Translate a term with vocabulary and type awareness.

        Performs a concept-mediated translation that:
        1. Looks up the term in the source language
        2. Finds matching concepts
        3. Resolves type information for each concept
        4. Returns translated terms with confidence scores

        Args:
            term: The native term in the source language.
            source_lang: Source language tag.
            target_lang: Target language tag.

        Returns:
            A TranslationResult with terms, confidence, and metadata.
        """
        nodes = self.vocabulary.lookup(term, source_lang)

        if not nodes:
            return TranslationResult(
                source_term=term,
                source_lang=source_lang,
                target_lang=target_lang,
                notes=["No matching concept found"],
            )

        ambiguity = len(nodes)
        all_target_terms: list[tuple[str, float]] = []
        best_concept_id = nodes[0].concept_id

        for node in nodes:
            # Compute confidence based on type compatibility
            source_type = self.concept_to_type(node.concept_id, source_lang)
            target_type = self.concept_to_type(node.concept_id, target_lang)
            type_compat = source_type.is_compatible_base(target_type.base_type)

            # Language coverage bonus
            lang_coverage = 1.0 if node.has_language(target_lang) else 0.3

            # Domain specificity bonus
            domain_bonus = 1.0 if node.semantic_domain in ALL_DOMAINS else 0.5

            confidence = type_compat * lang_coverage * domain_bonus

            for t in node.language_terms.get(target_lang, []):
                all_target_terms.append((t, confidence))

        # Deduplicate, keeping highest confidence
        seen: dict[str, float] = {}
        for t, conf in all_target_terms:
            if t not in seen or conf > seen[t]:
                seen[t] = conf

        # Sort by confidence descending
        sorted_terms = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        target_terms = [t for t, _ in sorted_terms]
        overall_confidence = sorted_terms[0][1] if sorted_terms else 0.0

        notes: list[str] = []
        if ambiguity > 1:
            notes.append(f"Multiple concepts matched ({ambiguity})")

        return TranslationResult(
            source_term=term,
            source_lang=source_lang,
            target_lang=target_lang,
            target_terms=target_terms,
            confidence=round(overall_confidence, 4),
            concept_id=best_concept_id,
            ambiguity=ambiguity,
            notes=notes,
        )


# ══════════════════════════════════════════════════════════════════
# 6. Convenience functions
# ══════════════════════════════════════════════════════════════════

def build_default_vocabulary() -> UnifiedVocabulary:
    """Create a UnifiedVocabulary with the default 50-concept ontology."""
    return UnifiedVocabulary()


def build_default_bridge() -> VocabularyBridge:
    """Create a VocabularyBridge with the default vocabulary."""
    return VocabularyBridge(build_default_vocabulary())
