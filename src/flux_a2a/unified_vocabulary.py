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

import math
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
    """Build the complete ontology with 50 concepts across 6 domains.

    Returns:
        Mapping from concept_id to ConceptNode.
    """
    concepts: Dict[str, ConceptNode] = {}

    # ── Domain 1: execution (15 concepts) ─────────────────────────

    concepts["agent_action"] = ConceptNode(
        concept_id="agent_action",
        universal_name="Agent",
        description="An autonomous entity that can initiate actions and make decisions",
        semantic_domain="execution",
        children=["function", "application"],
        language_terms={
            "zho": ["执行者", "代理"],
            "deu": ["Agent", "Akteur"],
            "kor": ["실행자", "에이전트"],
            "san": ["kartā", "kāraka"],
            "lat": ["Actor", "Agens"],
            "wen": ["執行者", "使"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 3},
            "san": {"morphology": "verbal_noun", "grammar": "kṛt_affix"},
            "wen": {"morphology": "single_char", "compactness": "maximal"},
        },
    )

    concepts["function"] = ConceptNode(
        concept_id="function",
        universal_name="Function",
        description="A reusable computation that maps inputs to outputs",
        semantic_domain="execution",
        children=["abstraction", "conditional", "map", "filter", "reduce"],
        parent="agent_action",
        language_terms={
            "zho": ["函数", "功能"],
            "deu": ["Funktion"],
            "kor": ["함수"],
            "san": ["upakāra", "kārya"],
            "lat": ["Functio", "Ratio"],
            "wen": ["用", "功"],
        },
        paradigm_traits={
            "deu": {"morphology": "compound_root", "gender": "femininum"},
            "san": {"morphology": "compounding", "components": "upa+kāra"},
            "wen": {"morphology": "single_char", "ambiguity": "high"},
        },
    )

    concepts["application"] = ConceptNode(
        concept_id="application",
        universal_name="Application",
        description="Applying a function to arguments, producing a result",
        semantic_domain="execution",
        children=["pipeline", "chain"],
        parent="agent_action",
        language_terms={
            "zho": ["应用", "适用"],
            "deu": ["Anwendung", "Applikation"],
            "kor": ["적용"],
            "san": ["prayoga"],
            "lat": ["Applicatio"],
            "wen": ["行", "施"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "polysemy": "true"},
            "san": {"morphology": "verbal_noun", "grammar": "pra+yuj"},
            "wen": {"morphology": "single_char", "context_dependent": "true"},
        },
    )

    concepts["abstraction"] = ConceptNode(
        concept_id="abstraction",
        universal_name="Abstraction",
        description="Generalizing a computation by parameterizing over values",
        semantic_domain="execution",
        parent="function",
        language_terms={
            "zho": ["抽象"],
            "deu": ["Abstraktion"],
            "kor": ["추상화"],
            "san": ["apahāra", "nirvacana"],
            "lat": ["Abstractio"],
            "wen": ["隱", "約"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 2},
            "san": {"morphology": "compounding", "philosophy": "nyaya_school"},
            "wen": {"morphology": "single_char", "meaning": "conceal_extract"},
        },
    )

    concepts["conditional"] = ConceptNode(
        concept_id="conditional",
        universal_name="Conditional",
        description="Branching execution based on a boolean predicate",
        semantic_domain="execution",
        parent="function",
        language_terms={
            "zho": ["条件"],
            "deu": ["Bedingung"],
            "kor": ["조건"],
            "san": ["paryeṣa", "vicāra"],
            "lat": ["Condicio", "Si"],
            "wen": ["然", "若"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "logic": "classical_chinese_condition"},
            "wen": {"morphology": "single_char", "grammar": "conditional_particle"},
            "san": {"morphology": "compounding", "usage": "logical_inquiry"},
        },
    )

    concepts["loop"] = ConceptNode(
        concept_id="loop",
        universal_name="Loop",
        description="Repeated execution of a computation while a condition holds",
        semantic_domain="execution",
        children=["sequence", "parallel"],
        language_terms={
            "zho": ["循环"],
            "deu": ["Schleife"],
            "kor": ["반복", "루프"],
            "san": ["āvṛtti"],
            "lat": ["Circuitus", "Iteratio"],
            "wen": ["循"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "彳+盾"},
            "deu": {"morphology": "simple", "gender": "femininum"},
            "san": {"morphology": "verbal_noun", "root": "vṛt_turn"},
            "wen": {"morphology": "single_char", "radical": "彳"},
        },
    )

    concepts["sequence"] = ConceptNode(
        concept_id="sequence",
        universal_name="Sequence",
        description="Ordered execution of steps, one after another",
        semantic_domain="execution",
        children=["pipeline"],
        parent="loop",
        language_terms={
            "zho": ["序列", "顺序"],
            "deu": ["Sequenz", "Reihenfolge"],
            "kor": ["순서"],
            "san": ["krama", "anukrama"],
            "lat": ["Series", "Ordo"],
            "wen": ["序"],
        },
        paradigm_traits={
            "san": {"morphology": "verbal_noun", "root": "kram_step"},
            "wen": {"morphology": "single_char", "meaning": "order_sequence"},
        },
    )

    concepts["parallel"] = ConceptNode(
        concept_id="parallel",
        universal_name="Parallel",
        description="Simultaneous execution of multiple computations",
        semantic_domain="execution",
        children=["fork"],
        parent="loop",
        language_terms={
            "zho": ["并行", "平行"],
            "deu": ["Parallel"],
            "kor": ["병렬"],
            "san": ["sāhacarya", "samānakāla"],
            "lat": ["Parallelus", "Concurrentia"],
            "wen": ["並"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "彳+行"},
            "wen": {"morphology": "single_char", "meaning": "side_by_side"},
            "san": {"morphology": "compounding", "components": "sahya+acarya"},
        },
    )

    concepts["fork"] = ConceptNode(
        concept_id="fork",
        universal_name="Fork",
        description="Splitting execution into independent branches",
        semantic_domain="execution",
        children=["join"],
        parent="parallel",
        language_terms={
            "zho": ["分支"],
            "deu": ["Gabelung", "Verzweigung"],
            "kor": ["분기"],
            "san": ["śākhā"],
            "lat": ["Furca", "Divisio"],
            "wen": ["分"],
        },
        paradigm_traits={
            "san": {"morphology": "simple", "meaning": "branch"},
            "wen": {"morphology": "single_char", "meaning": "divide_branch"},
        },
    )

    concepts["join"] = ConceptNode(
        concept_id="join",
        universal_name="Join",
        description="Merging results from parallel branches",
        semantic_domain="execution",
        parent="fork",
        language_terms={
            "zho": ["合并", "汇合"],
            "deu": ["Vereinigung", "Zusammenführung"],
            "kor": ["합병"],
            "san": ["sangati", "samāpti"],
            "lat": ["Iunctio", "Coniunctio"],
            "wen": ["合"],
        },
        paradigm_traits={
            "san": {"morphology": "compounding", "root": "gam_go"},
            "wen": {"morphology": "single_char", "meaning": "unite_join"},
        },
    )

    concepts["pipeline"] = ConceptNode(
        concept_id="pipeline",
        universal_name="Pipeline",
        description="Chained stages where output of one feeds into the next",
        semantic_domain="execution",
        parent="application",
        language_terms={
            "zho": ["流水线", "管道"],
            "deu": ["Pipeline", "Rohrleitung"],
            "kor": ["파이프라인"],
            "san": ["srotas", "pravāha-mārga"],
            "lat": ["Canalis"],
            "wen": ["流"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 3},
            "san": {"morphology": "simple", "root": "sru_flow"},
        },
    )

    concepts["chain"] = ConceptNode(
        concept_id="chain",
        universal_name="Chain",
        description="Linked sequence of operations composed together",
        semantic_domain="execution",
        parent="application",
        language_terms={
            "zho": ["链", "链式"],
            "deu": ["Kette", "Verkettung"],
            "kor": ["체인", "연쇄"],
            "san": ["śreṇī", "samūha"],
            "lat": ["Catena", "Catena"],
            "wen": ["鏈"],
        },
        paradigm_traits={
            "zho": {"morphology": "simple", "metal_literal": "true"},
            "wen": {"morphology": "single_char", "metal_literal": "true"},
        },
    )

    concepts["map"] = ConceptNode(
        concept_id="map",
        universal_name="Map",
        description="Transform each element of a collection by a function",
        semantic_domain="execution",
        parent="function",
        language_terms={
            "zho": ["映射", "变换"],
            "deu": ["Abbildung", "Map"],
            "kor": ["맵"],
            "san": ["citrāṅkura"],
            "lat": ["Mappa", "Imago"],
            "wen": ["映"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "math_context": "true"},
            "deu": {"morphology": "compound", "math_context": "true"},
        },
    )

    concepts["filter"] = ConceptNode(
        concept_id="filter",
        universal_name="Filter",
        description="Select elements from a collection matching a predicate",
        semantic_domain="execution",
        parent="function",
        language_terms={
            "zho": ["过滤", "筛选"],
            "deu": ["Filter"],
            "kor": ["필터"],
            "san": ["chalanam", "parikṣa"],
            "lat": ["Filtrum", "Cribro"],
            "wen": ["濾"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "氵+过"},
            "wen": {"morphology": "single_char", "radical": "氵"},
        },
    )

    concepts["reduce"] = ConceptNode(
        concept_id="reduce",
        universal_name="Reduce",
        description="Combine all elements of a collection into a single value",
        semantic_domain="execution",
        parent="function",
        language_terms={
            "zho": ["归约", "聚合"],
            "deu": ["Reduktion", "Faltung"],
            "kor": ["리덕션"],
            "san": ["saṅkṣepa", "samācaraṇa"],
            "lat": ["Reductio"],
            "wen": ["約"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "math_context": "true"},
            "san": {"morphology": "compounding", "meaning": "drawing_together"},
            "wen": {"morphology": "single_char", "meaning": "reduce_compact"},
        },
    )

    # ── Domain 2: data (10 concepts) ──────────────────────────────

    concepts["value"] = ConceptNode(
        concept_id="value",
        universal_name="Value",
        description="An immutable datum; the most fundamental unit of data",
        semantic_domain="data",
        children=["reference", "option"],
        language_terms={
            "zho": ["值"],
            "deu": ["Wert"],
            "kor": ["값"],
            "san": ["artha"],
            "lat": ["Valor"],
            "wen": ["值"],
        },
        paradigm_traits={
            "zho": {"morphology": "simple", "core_concept": "true"},
            "san": {"morphology": "simple", "philosophy": "meaning_value"},
            "wen": {"morphology": "single_char", "core_concept": "true"},
        },
    )

    concepts["reference"] = ConceptNode(
        concept_id="reference",
        universal_name="Reference",
        description="An indirect pointer to a value or location in memory",
        semantic_domain="data",
        parent="value",
        language_terms={
            "zho": ["引用"],
            "deu": ["Referenz", "Bezug"],
            "kor": ["참조"],
            "san": ["sambandha", "nidarśana"],
            "lat": ["Referentia", "Nomen"],
            "wen": ["指"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "弓+用"},
            "san": {"morphology": "compounding", "meaning": "connection"},
            "wen": {"morphology": "single_char", "meaning": "point_indicate"},
        },
    )

    concepts["container"] = ConceptNode(
        concept_id="container",
        universal_name="Container",
        description="A data structure that holds other values",
        semantic_domain="data",
        children=["collection", "pair", "stream", "buffer", "channel"],
        language_terms={
            "zho": ["容器"],
            "deu": ["Container", "Gefäß"],
            "kor": ["컨테이너"],
            "san": ["pātra", "bhājana"],
            "lat": ["Vascus", "Receptaculum"],
            "wen": ["器"],
        },
        paradigm_traits={
            "deu": {"morphology": "loanword", "gender": "neutrum"},
            "san": {"morphology": "simple", "root": "pā_hold"},
            "wen": {"morphology": "single_char", "meaning": "vessel_receptacle"},
        },
    )

    concepts["collection"] = ConceptNode(
        concept_id="collection",
        universal_name="Collection",
        description="An ordered or unordered group of elements",
        semantic_domain="data",
        parent="container",
        language_terms={
            "zho": ["集合"],
            "deu": ["Sammlung"],
            "kor": ["집합"],
            "san": ["samūha"],
            "lat": ["Collectio"],
            "wen": ["集"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "math_context": "true"},
            "san": {"morphology": "compounding", "meaning": "together+gathered"},
            "wen": {"morphology": "single_char", "meaning": "gather_collect"},
        },
    )

    concepts["pair"] = ConceptNode(
        concept_id="pair",
        universal_name="Pair",
        description="An ordered tuple of exactly two elements",
        semantic_domain="data",
        parent="container",
        language_terms={
            "zho": ["对", "对组"],
            "deu": ["Paar"],
            "kor": ["쌍"],
            "san": ["yugala", "dvandva"],
            "lat": ["Par", "Bini"],
            "wen": ["對"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "neutrum"},
            "san": {"morphology": "simple", "grammar": "dvandva_compound"},
            "wen": {"morphology": "single_char", "meaning": "couple_pair"},
        },
    )

    concepts["option"] = ConceptNode(
        concept_id="option",
        universal_name="Option",
        description="A value that may or may not be present (nullable)",
        semantic_domain="data",
        parent="value",
        language_terms={
            "zho": ["选项", "可选"],
            "deu": ["Option", "Vielleicht"],
            "kor": ["옵션"],
            "san": ["vikalpa"],
            "lat": ["Optio", "Forsitan"],
            "wen": ["擇"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound"},
            "san": {"morphology": "simple", "meaning": "alternative"},
            "deu": {"morphology": "loanword", "gender": "femininum"},
        },
    )

    concepts["result"] = ConceptNode(
        concept_id="result",
        universal_name="Result",
        description="A computation outcome that is either a success value or an error",
        semantic_domain="data",
        parent="value",
        language_terms={
            "zho": ["结果", "成效"],
            "deu": ["Ergebnis", "Resultat"],
            "kor": ["결과"],
            "san": ["phala"],
            "lat": ["Eventus", "Exitus"],
            "wen": ["果"],
        },
        paradigm_traits={
            "san": {"morphology": "simple", "root": "phal_bear_fruit"},
            "wen": {"morphology": "single_char", "meaning": "fruit_result"},
        },
    )

    concepts["stream"] = ConceptNode(
        concept_id="stream",
        universal_name="Stream",
        description="A potentially infinite sequence of values evaluated lazily",
        semantic_domain="data",
        parent="container",
        language_terms={
            "zho": ["流", "数据流"],
            "deu": ["Strom", "Datenstrom"],
            "kor": ["스트림"],
            "san": ["pravāha"],
            "lat": ["Flumen", "Fluxus"],
            "wen": ["川"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "maskulinum"},
            "san": {"morphology": "simple", "root": "vah_carry_flow"},
            "wen": {"morphology": "single_char", "meaning": "river_stream"},
        },
    )

    concepts["buffer"] = ConceptNode(
        concept_id="buffer",
        universal_name="Buffer",
        description="A temporary holding area for data between producer and consumer",
        semantic_domain="data",
        parent="container",
        language_terms={
            "zho": ["缓冲", "缓冲区"],
            "deu": ["Puffer"],
            "kor": ["버퍼"],
            "san": ["parivāra", "avakāśa"],
            "lat": ["Buffer", "Intermedium"],
            "wen": ["蓄"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 2},
            "deu": {"morphology": "loanword", "gender": "maskulinum"},
            "wen": {"morphology": "single_char", "meaning": "store_accumulate"},
        },
    )

    concepts["channel"] = ConceptNode(
        concept_id="channel",
        universal_name="Channel",
        description="A typed conduit for sending values between concurrent agents",
        semantic_domain="data",
        parent="container",
        language_terms={
            "zho": ["通道"],
            "deu": ["Kanal"],
            "kor": ["채널"],
            "san": ["patha", "mārga"],
            "lat": ["Fossa", "Via"],
            "wen": ["道"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "maskulinum"},
            "san": {"morphology": "simple", "meaning": "path_way"},
            "wen": {"morphology": "single_char", "meaning": "way_path_channel"},
        },
    )

    # ── Domain 3: scope (8 concepts) ──────────────────────────────

    concepts["global"] = ConceptNode(
        concept_id="global",
        universal_name="Global",
        description="Accessible from anywhere in the program; top-level visibility",
        semantic_domain="scope",
        children=["local", "namespace"],
        language_terms={
            "zho": ["全局"],
            "deu": ["Global"],
            "kor": ["전역"],
            "san": ["sārvabhauma"],
            "lat": ["Globalis", "Universalis"],
            "wen": ["天"],
        },
        paradigm_traits={
            "san": {"morphology": "compounding", "meaning": "all+pervading"},
            "wen": {"morphology": "single_char", "meaning": "heaven_all_encompassing"},
        },
    )

    concepts["local"] = ConceptNode(
        concept_id="local",
        universal_name="Local",
        description="Accessible only within a specific lexical block or function",
        semantic_domain="scope",
        parent="global",
        language_terms={
            "zho": ["局部"],
            "deu": ["Lokal"],
            "kor": ["지역"],
            "san": ["sāmīpya", "sthānīya"],
            "lat": ["Localis", "Proprius"],
            "wen": ["地"],
        },
        paradigm_traits={
            "san": {"morphology": "simple", "meaning": "nearby_proximity"},
            "wen": {"morphology": "single_char", "meaning": "earth_local"},
        },
    )

    concepts["closure"] = ConceptNode(
        concept_id="closure",
        universal_name="Closure",
        description="A function that captures its lexical environment",
        semantic_domain="scope",
        children=["environment"],
        language_terms={
            "zho": ["闭包"],
            "deu": ["Abschluss"],
            "kor": ["클로저"],
            "san": ["pāriṣadya", "bādha"],
            "lat": ["Clausura"],
            "wen": ["閉"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "translation_origin": "english"},
            "deu": {"morphology": "simple", "gender": "maskulinum"},
            "san": {"morphology": "compounding", "meaning": "surrounding_enclosure"},
        },
    )

    concepts["environment"] = ConceptNode(
        concept_id="environment",
        universal_name="Environment",
        description="The mapping of names to values in a given scope",
        semantic_domain="scope",
        parent="closure",
        language_terms={
            "zho": ["环境", "上下文"],
            "deu": ["Umgebung", "Kontext"],
            "kor": ["환경"],
            "san": ["pariṇāma"],
            "lat": ["Ambiens", "Contextus"],
            "wen": ["境"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound"},
            "wen": {"morphology": "single_char", "meaning": "boundary_realm"},
        },
    )

    concepts["stack"] = ConceptNode(
        concept_id="stack",
        universal_name="Stack",
        description="A last-in-first-out data structure for managing call frames",
        semantic_domain="scope",
        children=["namespace"],
        language_terms={
            "zho": ["栈", "栈帧"],
            "deu": ["Stapel"],
            "kor": ["스택"],
            "san": ["rāśi", "ucchrita"],
            "lat": ["Acervus", "Strues"],
            "wen": ["積"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "maskulinum"},
            "san": {"morphology": "simple", "meaning": "heap_pile"},
            "wen": {"morphology": "single_char", "meaning": "accumulate_stack"},
        },
    )

    concepts["heap"] = ConceptNode(
        concept_id="heap",
        universal_name="Heap",
        description="A region of memory for dynamically allocated, long-lived data",
        semantic_domain="scope",
        language_terms={
            "zho": ["堆", "堆内存"],
            "deu": ["Halde"],
            "kor": ["힙"],
            "san": ["kūṭa", "upari-bhāga"],
            "lat": ["Cumulus", "Acervus"],
            "wen": ["堆"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "femininum"},
            "wen": {"morphology": "single_char", "meaning": "heap_pile"},
        },
    )

    concepts["namespace"] = ConceptNode(
        concept_id="namespace",
        universal_name="Namespace",
        description="A named scope that prevents identifier collisions",
        semantic_domain="scope",
        children=["module"],
        language_terms={
            "zho": ["命名空间"],
            "deu": ["Namensraum"],
            "kor": ["네임스페이스"],
            "san": ["sthāna", "saṃjñā-kṣetra"],
            "lat": ["Nomen", "Spatium"],
            "wen": ["名"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 4},
            "deu": {"morphology": "compound", "gender": "maskulinum"},
            "san": {"morphology": "compounding", "meaning": "name_region"},
            "wen": {"morphology": "single_char", "meaning": "name_identifier"},
        },
    )

    concepts["module"] = ConceptNode(
        concept_id="module",
        universal_name="Module",
        description="A self-contained unit of code with its own namespace",
        semantic_domain="scope",
        parent="namespace",
        language_terms={
            "zho": ["模块"],
            "deu": ["Modul"],
            "kor": ["모듈"],
            "san": ["khaṇḍa", "pustaka"],
            "lat": ["Modulus", "Liber"],
            "wen": ["體"],
        },
        paradigm_traits={
            "deu": {"morphology": "loanword", "gender": "neutrum"},
            "wen": {"morphology": "single_char", "meaning": "body_entity"},
        },
    )

    # ── Domain 4: communication (8 concepts) ──────────────────────

    concepts["message"] = ConceptNode(
        concept_id="message",
        universal_name="Message",
        description="A discrete unit of data sent between agents",
        semantic_domain="communication",
        children=["request", "response", "event"],
        language_terms={
            "zho": ["消息"],
            "deu": ["Nachricht"],
            "kor": ["메시지"],
            "san": ["sandeśa"],
            "lat": ["Nuntius"],
            "wen": ["訊"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "氵"},
            "san": {"morphology": "compounding", "root": "sand_send"},
            "wen": {"morphology": "single_char", "meaning": "tidings_message"},
        },
    )

    concepts["request"] = ConceptNode(
        concept_id="request",
        universal_name="Request",
        description="A message asking an agent to perform an action or return data",
        semantic_domain="communication",
        parent="message",
        language_terms={
            "zho": ["请求"],
            "deu": ["Anfrage", "Bitte"],
            "kor": ["요청"],
            "san": ["yācanā", "prārthanā"],
            "lat": ["Petitio"],
            "wen": ["求"],
        },
        paradigm_traits={
            "deu": {"morphology": "compound", "gender": "femininum"},
            "san": {"morphology": "simple", "root": "yac_ask"},
            "wen": {"morphology": "single_char", "meaning": "seek_request"},
        },
    )

    concepts["response"] = ConceptNode(
        concept_id="response",
        universal_name="Response",
        description="A message returned as a reply to a request",
        semantic_domain="communication",
        parent="message",
        language_terms={
            "zho": ["响应", "回应"],
            "deu": ["Antwort"],
            "kor": ["응답"],
            "san": ["pratikriyā"],
            "lat": ["Responsio"],
            "wen": ["應"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "femininum"},
            "san": {"morphology": "compounding", "meaning": "against+action"},
            "wen": {"morphology": "single_char", "meaning": "reply_respond"},
        },
    )

    concepts["event"] = ConceptNode(
        concept_id="event",
        universal_name="Event",
        description="A notable occurrence that agents can observe and react to",
        semantic_domain="communication",
        parent="message",
        language_terms={
            "zho": ["事件"],
            "deu": ["Ereignis"],
            "kor": ["이벤트", "사건"],
            "san": ["ghaṭanā"],
            "lat": ["Eventus", "Res"],
            "wen": ["事"],
        },
        paradigm_traits={
            "deu": {"morphology": "compound", "gender": "neutrum"},
            "san": {"morphology": "simple", "root": "ghaṭ_happen"},
            "wen": {"morphology": "single_char", "meaning": "affair_event"},
        },
    )

    concepts["signal"] = ConceptNode(
        concept_id="signal",
        universal_name="Signal",
        description="An asynchronous notification or interruption",
        semantic_domain="communication",
        children=["broadcast"],
        language_terms={
            "zho": ["信号"],
            "deu": ["Signal"],
            "kor": ["신호"],
            "san": ["lakṣaṇa", "sūcana"],
            "lat": ["Signum"],
            "wen": ["號"],
        },
        paradigm_traits={
            "deu": {"morphology": "loanword", "gender": "neutrum"},
            "san": {"morphology": "simple", "meaning": "mark_indicator"},
            "wen": {"morphology": "single_char", "meaning": "call_signal"},
        },
    )

    concepts["broadcast"] = ConceptNode(
        concept_id="broadcast",
        universal_name="Broadcast",
        description="Sending a message to all interested agents simultaneously",
        semantic_domain="communication",
        parent="signal",
        children=["publish"],
        language_terms={
            "zho": ["广播"],
            "deu": ["Rundruf"],
            "kor": ["방송"],
            "san": ["saṃpradāya"],
            "lat": ["Divulgatio", "Praedicatio"],
            "wen": ["播"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "radical": "广+播"},
            "san": {"morphology": "compounding", "meaning": "together+handing_down"},
            "wen": {"morphology": "single_char", "meaning": "disseminate_broadcast"},
        },
    )

    concepts["subscribe"] = ConceptNode(
        concept_id="subscribe",
        universal_name="Subscribe",
        description="Registering to receive messages on a topic or channel",
        semantic_domain="communication",
        language_terms={
            "zho": ["订阅"],
            "deu": ["Abonnement"],
            "kor": ["구독"],
            "san": ["upalakṣaṇa"],
            "lat": ["Subscriptio"],
            "wen": ["聽"],
        },
        paradigm_traits={
            "deu": {"morphology": "loanword", "gender": "neutrum"},
            "san": {"morphology": "compounding", "meaning": "toward+observing"},
            "wen": {"morphology": "single_char", "meaning": "listen_hear"},
        },
    )

    concepts["publish"] = ConceptNode(
        concept_id="publish",
        universal_name="Publish",
        description="Emitting a message to all subscribers of a topic",
        semantic_domain="communication",
        parent="broadcast",
        language_terms={
            "zho": ["发布"],
            "deu": ["Veröffentlichung"],
            "kor": ["발행"],
            "san": ["prakāśana"],
            "lat": ["Publicatio"],
            "wen": ["布"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound"},
            "san": {"morphology": "verbal_noun", "root": "prakāś_make_known"},
            "wen": {"morphology": "single_char", "meaning": "declare_publish"},
        },
    )

    # ── Domain 5: temporal (5 concepts) ───────────────────────────

    concepts["instant"] = ConceptNode(
        concept_id="instant",
        universal_name="Instant",
        description="A specific, indivisible point in time",
        semantic_domain="temporal",
        children=["duration"],
        language_terms={
            "zho": ["瞬时", "时刻"],
            "deu": ["Augenblick", "Moment"],
            "kor": ["순간"],
            "san": ["kṣaṇa"],
            "lat": ["Momentum"],
            "wen": ["時"],
        },
        paradigm_traits={
            "san": {"morphology": "simple", "philosophy": "buddhist_instant"},
            "wen": {"morphology": "single_char", "meaning": "time_instant"},
        },
    )

    concepts["duration"] = ConceptNode(
        concept_id="duration",
        universal_name="Duration",
        description="The length of a time interval between two instants",
        semantic_domain="temporal",
        parent="instant",
        children=["interval"],
        language_terms={
            "zho": ["持续", "持续时间"],
            "deu": ["Dauer"],
            "kor": ["지속"],
            "san": ["kāla", "sthiti"],
            "lat": ["Duratio"],
            "wen": ["久"],
        },
        paradigm_traits={
            "san": {"morphology": "simple", "root": "kal_time"},
            "wen": {"morphology": "single_char", "meaning": "long_enduring"},
        },
    )

    concepts["interval"] = ConceptNode(
        concept_id="interval",
        universal_name="Interval",
        description="A continuous range of time between two endpoints",
        semantic_domain="temporal",
        parent="duration",
        language_terms={
            "zho": ["区间", "间隔"],
            "deu": ["Intervall"],
            "kor": ["구간"],
            "san": ["antarāla"],
            "lat": ["Intervallum"],
            "wen": ["間"],
        },
        paradigm_traits={
            "san": {"morphology": "compounding", "meaning": "between_space"},
            "wen": {"morphology": "single_char", "meaning": "between_space"},
        },
    )

    concepts["schedule"] = ConceptNode(
        concept_id="schedule",
        universal_name="Schedule",
        description="A plan for execution at a specific future time",
        semantic_domain="temporal",
        children=["deadline"],
        language_terms={
            "zho": ["调度", "时间表"],
            "deu": ["Zeitplan"],
            "kor": ["스케줄"],
            "san": ["nirṇaya", "krama-kāla"],
            "lat": ["Tempus", "Ordo"],
            "wen": ["期"],
        },
        paradigm_traits={
            "zho": {"morphology": "compound", "character_count": 2},
            "wen": {"morphology": "single_char", "meaning": "appointed_time"},
        },
    )

    concepts["deadline"] = ConceptNode(
        concept_id="deadline",
        universal_name="Deadline",
        description="A point in time by which a computation must complete",
        semantic_domain="temporal",
        parent="schedule",
        language_terms={
            "zho": ["截止", "截止时间"],
            "deu": ["Frist", "Termin"],
            "kor": ["마감"],
            "san": ["sīmā", "anta-kāla"],
            "lat": ["Terminus", "Finis"],
            "wen": ["限"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "femininum"},
            "san": {"morphology": "simple", "meaning": "boundary_limit"},
            "wen": {"morphology": "single_char", "meaning": "limit_boundary"},
        },
    )

    # ── Domain 6: meta (4 concepts) ───────────────────────────────

    concepts["type"] = ConceptNode(
        concept_id="type",
        universal_name="Type",
        description="A classification of values by their structure and behavior",
        semantic_domain="meta",
        children=["kind"],
        language_terms={
            "zho": ["类型"],
            "deu": ["Typ"],
            "kor": ["타입", "형"],
            "san": ["jāti"],
            "lat": ["Genus"],
            "wen": ["類"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "maskulinum"},
            "san": {"morphology": "simple", "philosophy": "nyaya_category"},
            "wen": {"morphology": "single_char", "meaning": "category_class"},
        },
    )

    concepts["kind"] = ConceptNode(
        concept_id="kind",
        universal_name="Kind",
        description="A classification of types themselves (type of types)",
        semantic_domain="meta",
        parent="type",
        language_terms={
            "zho": ["种"],
            "deu": ["Art", "Gattung"],
            "kor": ["종"],
            "san": ["prakāra"],
            "lat": ["Species"],
            "wen": ["種"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "femininum"},
            "san": {"morphology": "compounding", "meaning": "sort_manner"},
            "wen": {"morphology": "single_char", "meaning": "species_kind"},
        },
    )

    concepts["class"] = ConceptNode(
        concept_id="class",
        universal_name="Class",
        description="A blueprint defining the structure and behavior of objects",
        semantic_domain="meta",
        children=["trait"],
        language_terms={
            "zho": ["类", "类定义"],
            "deu": ["Klasse"],
            "kor": ["클래스"],
            "san": ["varga"],
            "lat": ["Classis"],
            "wen": ["屬"],
        },
        paradigm_traits={
            "deu": {"morphology": "simple", "gender": "femininum"},
            "san": {"morphology": "simple", "grammar": "paninian_category"},
            "wen": {"morphology": "single_char", "meaning": "genus_category"},
        },
    )

    concepts["trait"] = ConceptNode(
        concept_id="trait",
        universal_name="Trait",
        description="A named interface that types can implement",
        semantic_domain="meta",
        parent="class",
        language_terms={
            "zho": ["特征", "特性"],
            "deu": ["Eigenschaft", "Merkmal"],
            "kor": ["트레이트", "특성"],
            "san": ["guṇa", "lakṣaṇa"],
            "lat": ["Proprietas", "Indoles"],
            "wen": ["德"],
        },
        paradigm_traits={
            "deu": {"morphology": "compound", "gender": "femininum"},
            "san": {"morphology": "simple", "philosophy": "quality_attribute"},
            "wen": {"morphology": "single_char", "meaning": "virtue_property"},
        },
    )

    return concepts


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
