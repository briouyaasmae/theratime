# ============================================================
# TheraTime v0.6-neurocomputing
# Therapeutic Timing Evaluation for Mental-Health Response Retrieval
#
# Target venue: Neurocomputing software paper
#
# Key fixes in this version:
# 1. Dense retrieval uses all-MiniLM-L6-v2.
# 2. Default timing classifier uses paraphrase-mpnet-base-v2.
# 3. Retrieval and timing encoders are separated to reduce circular encoder bias.
# 4. Self-retrieval exclusion is enabled.
# 5. Stress-test positive control uses pure distress language.
# 6. TTP@2 and Hit@2 are both reported correctly.
# 7. Entropy uses softmax-normalized prototype scores.
#
# Responsible use:
# This is an offline research evaluation framework only.
# It is not a clinical decision-support tool.
# It is not validated for use with real users in therapy or crisis settings.
# ============================================================


import json
import re
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.stats import pointbiserialr, pearsonr
from scipy.stats import chi2 as chi2_dist

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# SECTION 1 - Configuration
# ============================================================

SOFTWARE_NAME = "TheraTime"
SOFTWARE_VERSION = "0.6-neurocomputing"

DATASETS_USED = [
    "thu-coai/esconv",
    "nbertagnolli/counsel-chat",
    "ShenLab/MentalChat16K",
]

OUTPUT_DIR = Path("/kaggle/working/theratime_v06_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_RETRIEVAL = 5
MAX_QUERIES_PER_DATASET = 250

N_BOOTSTRAP = 2000
CONFIDENCE_PERCENTILE = 10.0
RANDOM_SEED = 42

# Timing encoders
DEFAULT_TIMING_ENCODER = "sentence-transformers/paraphrase-mpnet-base-v2"
ABLATION_TIMING_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Dense retrieval encoder separated from default timing encoder
DENSE_RETRIEVAL_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_FALLBACK = 0.03


def compute_adaptive_threshold(
    margins: List[float],
    pct: float = CONFIDENCE_PERCENTILE,
) -> float:
    if not margins:
        return CONFIDENCE_FALLBACK
    return float(np.percentile(margins, pct))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ============================================================
# SECTION 2 - Input quality filter
# ============================================================

MIN_QUERY_WORDS = 6
MAX_QUERY_WORDS = 120
MIN_ANSWER_WORDS = 5

_GREETING_RE = re.compile(
    r"^(hello|hi|hey|good\s+(morning|afternoon|evening|day)|"
    r"thanks|thank\s+you|okay|ok|sure|great|wonderful|"
    r"lol|haha|wow|yes|no|alright|cool|nice)[\s!.,?]*$",
    re.IGNORECASE,
)

_FRAGMENT_RE = re.compile(r"^[\w\s']{1,5}[.!?,]*$")


def is_valid_pair(q: str, a: str) -> bool:
    q = str(q).strip()
    a = str(a).strip()

    q_words = len(q.split())
    a_words = len(a.split())

    if q_words < MIN_QUERY_WORDS or q_words > MAX_QUERY_WORDS:
        return False

    if a_words < MIN_ANSWER_WORDS:
        return False

    if _GREETING_RE.match(q):
        return False

    if _FRAGMENT_RE.match(q):
        return False

    if re.match(r"^\d+\s+\w+", q):
        return False

    return True


# ============================================================
# SECTION 3 - Taxonomies
# ============================================================

SUPPORT_STAGES: Dict[str, str] = {
    "distress_disclosure": (
        "I feel sad, lonely, scared, hurt, or emotionally distressed. "
        "I am sharing my pain for the first time. I need someone to hear me "
        "and acknowledge what I am going through. I have not asked for concrete help yet."
    ),
    "high_emotional_intensity": (
        "I am overwhelmed, panicking, unable to stop crying, spiraling, or falling apart. "
        "My emotional state is very intense and I need immediate calming, grounding, "
        "or stabilizing support before advice."
    ),
    "unclear_need": (
        "Something feels wrong but I cannot identify or explain what it is. "
        "I do not know what kind of support I need. I am confused about my feelings "
        "or about where to start."
    ),
    "advice_seeking": (
        "What should I do? How can I handle this? I am explicitly asking for "
        "practical steps, strategies, coping techniques, or concrete actions."
    ),
    "psychoeducation_seeking": (
        "Why do I feel this way? What causes this? Can you explain what is happening "
        "to me mentally or emotionally? I want to understand my experience."
    ),
    "crisis_safety": (
        "I cannot keep myself safe. I am thinking about suicide or hurting myself. "
        "I need emergency help right now. This is a safety crisis, not just distress."
    ),
    "followup_problem_solving": (
        "I have already received support and now I need the next steps. "
        "I am ready to move from emotional support to planning, action, or follow-up."
    ),
}


SUPPORT_MOVES: Dict[str, str] = {
    "validation": (
        "The response validates the user's feelings and says that the user's emotional "
        "reaction makes sense and is understandable."
    ),
    "empathy": (
        "The response expresses warmth, care, and emotional presence. "
        "It communicates that the supporter is with the user and cares about them."
    ),
    "reflective_listening": (
        "The response reflects or restates what the user is feeling or experiencing, "
        "showing that the supporter has heard and understood."
    ),
    "clarification": (
        "The response asks a gentle, open question to better understand the user's "
        "situation, feelings, or needs before proceeding."
    ),
    "grounding": (
        "The response helps the user calm down, breathe, stabilize emotions, "
        "or focus on the present moment using grounding techniques."
    ),
    "practical_advice": (
        "The response gives concrete coping steps, behavioral strategies, habits, "
        "or action plans the user can try."
    ),
    "psychoeducation": (
        "The response explains mental-health symptoms, emotional processes, "
        "or why certain feelings occur."
    ),
    "encouragement": (
        "The response offers hope, reassurance, motivation, or positive affirmation "
        "of the user's ability to cope or improve."
    ),
    "safety_referral": (
        "The response directs the user toward emergency help, a crisis line, "
        "a trusted person, or professional safety support. It prioritizes immediate safety."
    ),
}


TIMING_LABELS = [
    "well_timed",
    "premature_advice",
    "delayed_safety",
    "over_validation",
    "missing_clarification",
    "stage_mismatch",
]


ALLOWED_MOVES: Dict[str, set] = {
    "distress_disclosure": {
        "validation",
        "empathy",
        "reflective_listening",
        "clarification",
        "encouragement",
    },
    "high_emotional_intensity": {
        "validation",
        "empathy",
        "reflective_listening",
        "grounding",
        "clarification",
    },
    "unclear_need": {
        "validation",
        "empathy",
        "reflective_listening",
        "clarification",
    },
    "advice_seeking": {
        "practical_advice",
        "psychoeducation",
        "encouragement",
        "validation",
    },
    "psychoeducation_seeking": {
        "psychoeducation",
        "validation",
        "clarification",
    },
    "crisis_safety": {
        "safety_referral",
        "grounding",
        "validation",
    },
    "followup_problem_solving": {
        "practical_advice",
        "psychoeducation",
        "encouragement",
    },
}


PRIMARY_MOVE: Dict[str, str] = {
    "distress_disclosure": "validation",
    "high_emotional_intensity": "grounding",
    "unclear_need": "clarification",
    "advice_seeking": "practical_advice",
    "psychoeducation_seeking": "psychoeducation",
    "crisis_safety": "safety_referral",
    "followup_problem_solving": "practical_advice",
}


# ============================================================
# SECTION 4 - Keyword ablation baseline
# ============================================================

_STAGE_KEYWORDS: Dict[str, List[str]] = {
    "crisis_safety": [
        "suicid",
        "self-harm",
        "hurt myself",
        "can't stay safe",
        "can not stay safe",
        "end my life",
        "kill myself",
        "don't want to live",
        "want to die",
        "unsafe tonight",
    ],
    "advice_seeking": [
        "what should i do",
        "how can i",
        "what steps",
        "any advice",
        "what do you suggest",
        "what can i try",
        "what would you recommend",
        "give me some tips",
    ],
    "psychoeducation_seeking": [
        "why do i feel",
        "why does",
        "what causes",
        "can you explain",
        "what is anxiety",
        "why am i",
        "what happens when",
    ],
    "high_emotional_intensity": [
        "overwhelmed",
        "can't cope",
        "can not cope",
        "panicking",
        "cannot stop crying",
        "can not stop crying",
        "spiraling",
        "falling apart",
        "breaking down",
        "cannot breathe",
    ],
    "distress_disclosure": [
        "i feel",
        "i'm so",
        "i have been feeling",
        "i've been feeling",
        "i am sad",
        "i feel lonely",
        "i feel depressed",
        "i am scared",
    ],
    "followup_problem_solving": [
        "next step",
        "what now",
        "i tried that",
        "i've already tried",
        "what else can",
        "following up",
        "after that",
    ],
    "unclear_need": [
        "i don't know",
        "i'm not sure what",
        "something is wrong",
        "i can't explain",
        "i don't understand why",
        "i just feel",
    ],
}


_STAGE_PRIORITY = [
    "crisis_safety",
    "high_emotional_intensity",
    "advice_seeking",
    "psychoeducation_seeking",
    "followup_problem_solving",
    "distress_disclosure",
    "unclear_need",
]


def keyword_stage(text: str) -> str:
    t = str(text).lower()
    for stage in _STAGE_PRIORITY:
        if any(kw in t for kw in _STAGE_KEYWORDS[stage]):
            return stage
    return "unclear_need"


# ============================================================
# SECTION 5 - Data structures
# ============================================================

@dataclass
class QAItem:
    query_id: str
    query: str
    answer: str
    source_dataset: str


@dataclass
class Retrieved:
    response_id: str
    text: str
    score: float
    source_dataset: str


@dataclass
class Example:
    query_id: str
    query: str
    retrieved: List[Retrieved]
    source_dataset: str
    retrieval_method: str
    gold_answer: Optional[str] = None


@dataclass
class Judgment:
    query_id: str
    source_dataset: str
    retrieval_method: str
    response_id: str
    rank: int
    query: str
    response: str
    retrieval_score: float
    predicted_stage: str
    predicted_move: str
    timing_label: str
    is_well_timed: bool
    explanation: str
    stage_confidence: float
    move_confidence: float
    low_confidence: bool
    stage_scores: Dict[str, float] = field(default_factory=dict)
    move_scores: Dict[str, float] = field(default_factory=dict)


# ============================================================
# SECTION 6 - Prototype classifier
# ============================================================

class PrototypeClassifier:
    def __init__(self, descriptions: Dict[str, str], model_name: str):
        self.descriptions = descriptions
        self.labels = list(descriptions.keys())
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        prototype_texts = [
            f"{label}: {description}"
            for label, description in descriptions.items()
        ]

        self.prototype_embeddings = self.model.encode(
            prototype_texts,
            normalize_embeddings=True,
        )

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        emb = self.model.encode([str(text)], normalize_embeddings=True)
        scores = cosine_similarity(emb, self.prototype_embeddings)[0]

        idx = np.argsort(scores)[::-1]
        top_label = self.labels[int(idx[0])]
        margin = float(scores[idx[0]] - scores[idx[1]])

        all_scores = {
            label: float(score)
            for label, score in zip(self.labels, scores)
        }

        return top_label, margin, all_scores


# ============================================================
# SECTION 7 - Timing rule engine
# ============================================================

def judge_timing(stage: str, move: str) -> Dict[str, Any]:
    allowed = ALLOWED_MOVES.get(stage, set())

    if stage == "crisis_safety" and move not in {
        "safety_referral",
        "grounding",
        "validation",
    }:
        return {
            "timing_label": "delayed_safety",
            "is_well_timed": False,
            "explanation": (
                "Safety-related stage detected, but the response does not provide "
                "safety referral, grounding, or validation."
            ),
        }

    if (
        stage in {"distress_disclosure", "high_emotional_intensity", "unclear_need"}
        and move == "practical_advice"
    ):
        return {
            "timing_label": "premature_advice",
            "is_well_timed": False,
            "explanation": (
                "The response gives practical advice before validating, grounding, "
                "or clarifying the user's emotional state."
            ),
        }

    if (
        stage in {"advice_seeking", "followup_problem_solving"}
        and move in {"validation", "empathy", "reflective_listening"}
    ):
        return {
            "timing_label": "over_validation",
            "is_well_timed": False,
            "explanation": (
                "The user is asking for practical help, but the response only "
                "validates, empathizes, or reflects without providing action."
            ),
        }

    if stage == "unclear_need" and move not in allowed:
        return {
            "timing_label": "missing_clarification",
            "is_well_timed": False,
            "explanation": (
                "The user's need is unclear, but the response acts without "
                "clarifying or offering supportive presence."
            ),
        }

    if move in allowed:
        return {
            "timing_label": "well_timed",
            "is_well_timed": True,
            "explanation": "The support move is compatible with the predicted support stage.",
        }

    return {
        "timing_label": "stage_mismatch",
        "is_well_timed": False,
        "explanation": (
            f"The move '{move}' is not compatible with the predicted stage '{stage}'."
        ),
    }


# ============================================================
# SECTION 8 - Evaluation pipelines
# ============================================================

class TheraTimePipeline:
    def __init__(
        self,
        stage_encoder: str = DEFAULT_TIMING_ENCODER,
        move_encoder: str = DEFAULT_TIMING_ENCODER,
        confidence_percentile: float = CONFIDENCE_PERCENTILE,
    ):
        self.confidence_percentile = confidence_percentile
        self.stage_threshold = CONFIDENCE_FALLBACK
        self.move_threshold = CONFIDENCE_FALLBACK

        print(f"Stage classifier: {stage_encoder}")
        self.stage_classifier = PrototypeClassifier(SUPPORT_STAGES, stage_encoder)

        print(f"Move classifier: {move_encoder}")
        self.move_classifier = PrototypeClassifier(SUPPORT_MOVES, move_encoder)

    def calibrate(self, examples: List[Example], sample_n: int = 300) -> None:
        stage_margins = []
        move_margins = []

        for ex in examples[:sample_n]:
            _, stage_margin, _ = self.stage_classifier.predict(ex.query)
            stage_margins.append(stage_margin)

            if ex.retrieved:
                _, move_margin, _ = self.move_classifier.predict(ex.retrieved[0].text)
                move_margins.append(move_margin)

        self.stage_threshold = compute_adaptive_threshold(
            stage_margins,
            self.confidence_percentile,
        )
        self.move_threshold = compute_adaptive_threshold(
            move_margins,
            self.confidence_percentile,
        )

        print(f"Stage threshold p{self.confidence_percentile:.0f}: {self.stage_threshold:.4f}")
        print(f"Move threshold p{self.confidence_percentile:.0f}: {self.move_threshold:.4f}")

    def evaluate_example(self, ex: Example, top_k: int = 5) -> List[Judgment]:
        stage, stage_conf, stage_scores = self.stage_classifier.predict(ex.query)
        judgments = []

        for rank, item in enumerate(ex.retrieved[:top_k], start=1):
            move, move_conf, move_scores = self.move_classifier.predict(item.text)
            timing = judge_timing(stage, move)

            low_confidence = (
                stage_conf < self.stage_threshold
                or move_conf < self.move_threshold
            )

            judgments.append(
                Judgment(
                    query_id=ex.query_id,
                    source_dataset=ex.source_dataset,
                    retrieval_method=ex.retrieval_method,
                    response_id=item.response_id,
                    rank=rank,
                    query=ex.query,
                    response=item.text,
                    retrieval_score=float(item.score),
                    predicted_stage=stage,
                    predicted_move=move,
                    timing_label=timing["timing_label"],
                    is_well_timed=bool(timing["is_well_timed"]),
                    explanation=timing["explanation"],
                    stage_confidence=round(stage_conf, 4),
                    move_confidence=round(move_conf, 4),
                    low_confidence=low_confidence,
                    stage_scores=stage_scores,
                    move_scores=move_scores,
                )
            )

        return judgments

    def run(
        self,
        examples: List[Example],
        top_k: int = 5,
        calibrate: bool = True,
    ) -> List[Judgment]:
        if calibrate and examples:
            self.calibrate(examples)

        all_judgments = []
        for ex in tqdm(examples, desc="TheraTime evaluation"):
            all_judgments.extend(self.evaluate_example(ex, top_k=top_k))

        return all_judgments


class KeywordPipeline:
    def __init__(
        self,
        move_encoder: str = DEFAULT_TIMING_ENCODER,
        confidence_percentile: float = CONFIDENCE_PERCENTILE,
    ):
        self.confidence_percentile = confidence_percentile
        self.move_threshold = CONFIDENCE_FALLBACK

        print(f"Keyword pipeline move classifier: {move_encoder}")
        self.move_classifier = PrototypeClassifier(SUPPORT_MOVES, move_encoder)

    def calibrate(self, examples: List[Example], sample_n: int = 300) -> None:
        move_margins = []

        for ex in examples[:sample_n]:
            if ex.retrieved:
                _, move_margin, _ = self.move_classifier.predict(ex.retrieved[0].text)
                move_margins.append(move_margin)

        self.move_threshold = compute_adaptive_threshold(
            move_margins,
            self.confidence_percentile,
        )

        print(f"Keyword pipeline move threshold p{self.confidence_percentile:.0f}: {self.move_threshold:.4f}")

    def run(
        self,
        examples: List[Example],
        top_k: int = 5,
        calibrate: bool = True,
    ) -> List[Judgment]:
        if calibrate and examples:
            self.calibrate(examples)

        all_judgments = []

        for ex in tqdm(examples, desc="Keyword ablation"):
            stage = keyword_stage(ex.query)

            for rank, item in enumerate(ex.retrieved[:top_k], start=1):
                move, move_conf, move_scores = self.move_classifier.predict(item.text)
                timing = judge_timing(stage, move)

                all_judgments.append(
                    Judgment(
                        query_id=ex.query_id,
                        source_dataset=ex.source_dataset,
                        retrieval_method=ex.retrieval_method,
                        response_id=item.response_id,
                        rank=rank,
                        query=ex.query,
                        response=item.text,
                        retrieval_score=float(item.score),
                        predicted_stage=stage,
                        predicted_move=move,
                        timing_label=timing["timing_label"],
                        is_well_timed=bool(timing["is_well_timed"]),
                        explanation=timing["explanation"],
                        stage_confidence=-1.0,
                        move_confidence=round(move_conf, 4),
                        low_confidence=move_conf < self.move_threshold,
                        stage_scores={},
                        move_scores=move_scores,
                    )
                )

        return all_judgments


# ============================================================
# SECTION 9 - Statistics
# ============================================================

def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return 0.0, 0.0, 0.0

    point = float(stat_fn(values))
    samples = []

    for _ in range(n_bootstrap):
        resampled = rng.choice(values, size=len(values), replace=True)
        samples.append(stat_fn(resampled))

    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))

    return point, lower, upper


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return 0.0

    pooled_var = (
        ((len(a) - 1) * np.var(a, ddof=1))
        + ((len(b) - 1) * np.var(b, ddof=1))
    ) / (len(a) + len(b) - 2)

    return float((np.mean(a) - np.mean(b)) / np.sqrt(max(pooled_var, 1e-9)))


def mcnemar_test(j1: List[Judgment], j2: List[Judgment]) -> Dict[str, Any]:
    d1 = {
        (j.query_id, j.retrieval_method, j.rank): j.is_well_timed
        for j in j1
    }
    d2 = {
        (j.query_id, j.retrieval_method, j.rank): j.is_well_timed
        for j in j2
    }

    keys = sorted(set(d1.keys()) & set(d2.keys()))
    keys = [k for k in keys if k[2] == 1]

    b = sum(1 for k in keys if d1[k] and not d2[k])
    c = sum(1 for k in keys if not d1[k] and d2[k])

    n = b + c

    if n == 0:
        return {
            "b": int(b),
            "c": int(c),
            "chi2": 0.0,
            "p_value": 1.0,
            "significant_p05": False,
            "n_discordant": 0,
        }

    chi2_value = (abs(b - c) - 1.0) ** 2 / n
    p_value = float(1 - chi2_dist.cdf(chi2_value, df=1))

    return {
        "b": int(b),
        "c": int(c),
        "chi2": round(float(chi2_value), 4),
        "p_value": round(p_value, 4),
        "significant_p05": bool(p_value < 0.05),
        "n_discordant": int(n),
    }


def compute_metrics(judgments: List[Judgment], top_k: int = 5) -> Dict[str, Any]:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return {}

    top1 = df[df["rank"] == 1].copy()
    topk = df[df["rank"] <= top_k].copy()
    top1_hc = top1[~top1["low_confidence"]].copy()

    wt = top1["is_well_timed"].astype(float).values
    tta1, tta1_lo, tta1_hi = bootstrap_ci(wt)

    d_vs_05 = cohens_d(wt, np.full(len(wt), 0.5))
    n_queries = max(int(top1["query_id"].nunique()), 1)

    metrics = {
        "n_queries": int(top1["query_id"].nunique()),
        "n_judgments": int(len(df)),
        "TTA@1": round(tta1, 4),
        "TTA@1_CI_lo": round(tta1_lo, 4),
        "TTA@1_CI_hi": round(tta1_hi, 4),
        f"TTP@{top_k}": round(float(topk["is_well_timed"].mean()), 4),
        f"Hit@{top_k}": round(
            float(topk.groupby("query_id")["is_well_timed"].any().mean()),
            4,
        ),
        "cohens_d_vs_0.5": round(d_vs_05, 4),
        "n_high_confidence": int(len(top1_hc)),
        "TTA@1_high_confidence": (
            round(float(top1_hc["is_well_timed"].mean()), 4)
            if len(top1_hc)
            else 0.0
        ),
        "low_confidence_rate": round(float(top1["low_confidence"].mean()), 4),
        "mean_stage_confidence": round(
            float(top1[top1["stage_confidence"] >= 0]["stage_confidence"].mean()),
            4,
        ),
        "mean_move_confidence": round(
            float(top1[top1["move_confidence"] >= 0]["move_confidence"].mean()),
            4,
        ),
    }

    for label in [
        "premature_advice",
        "delayed_safety",
        "over_validation",
        "missing_clarification",
        "stage_mismatch",
    ]:
        metrics[f"{label}_rate"] = round(
            float((top1["timing_label"] == label).sum() / n_queries),
            4,
        )

    return metrics


def timing_hit_at_k(judgments: List[Judgment], k: int = 2) -> float:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return 0.0

    topk = df[df["rank"] <= k].copy()
    return float(topk.groupby("query_id")["is_well_timed"].any().mean())


# ============================================================
# SECTION 10 - Geometry and diagnostics
# ============================================================

def embedding_geometry_analysis(judgments: List[Judgment]) -> pd.DataFrame:
    rows = []

    for j in judgments:
        if j.rank != 1:
            continue

        if not j.stage_scores or not j.move_scores:
            continue

        stage_raw = np.array(list(j.stage_scores.values()), dtype=float)
        move_raw = np.array(list(j.move_scores.values()), dtype=float)

        stage_probs = softmax(stage_raw)
        move_probs = softmax(move_raw)

        stage_entropy = float(-np.sum(stage_probs * np.log(stage_probs + 1e-9)))
        move_entropy = float(-np.sum(move_probs * np.log(move_probs + 1e-9)))

        rows.append(
            {
                "query_id": j.query_id,
                "source_dataset": j.source_dataset,
                "retrieval_method": j.retrieval_method,
                "retrieval_score": j.retrieval_score,
                "is_well_timed": int(j.is_well_timed),
                "timing_label": j.timing_label,
                "predicted_stage": j.predicted_stage,
                "predicted_move": j.predicted_move,
                "max_stage_score": float(stage_raw.max()),
                "max_move_score": float(move_raw.max()),
                "stage_entropy": stage_entropy,
                "move_entropy": move_entropy,
                "stage_margin": j.stage_confidence,
                "move_margin": j.move_confidence,
            }
        )

    return pd.DataFrame(rows)


def geometry_correlation_report(geo_df: pd.DataFrame) -> Dict[str, Any]:
    report = {}

    if geo_df.empty:
        return report

    for feature in [
        "max_stage_score",
        "max_move_score",
        "stage_entropy",
        "move_entropy",
        "stage_margin",
        "move_margin",
        "retrieval_score",
    ]:
        if feature not in geo_df.columns:
            continue

        try:
            r, p = pointbiserialr(
                geo_df[feature].values,
                geo_df["is_well_timed"].astype(int).values,
            )
            report[f"r_{feature}_vs_timing"] = round(float(r), 4)
            report[f"p_{feature}_vs_timing"] = round(float(p), 4)
        except Exception:
            report[f"r_{feature}_vs_timing"] = None
            report[f"p_{feature}_vs_timing"] = None

    if {"retrieval_score", "stage_entropy"}.issubset(geo_df.columns):
        try:
            r2, p2 = pearsonr(
                geo_df["retrieval_score"].values,
                geo_df["stage_entropy"].values,
            )
            report["r_retrieval_vs_stage_entropy"] = round(float(r2), 4)
            report["p_retrieval_vs_stage_entropy"] = round(float(p2), 4)
        except Exception:
            report["r_retrieval_vs_stage_entropy"] = None
            report["p_retrieval_vs_stage_entropy"] = None

    report["interpretation"] = (
        "Geometry correlations are diagnostic only. They describe relationships "
        "between retrieval scores, prototype confidence, entropy, and automatic "
        "timing labels. They should not be interpreted as clinical validation."
    )

    return report


def corpus_move_distribution(judgments: List[Judgment]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return pd.DataFrame()

    dist = (
        df.groupby("predicted_move")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    dist["percentage"] = dist["count"] / dist["count"].sum()
    return dist


def stage_mismatch_decomposition(judgments: List[Judgment]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return pd.DataFrame()

    mismatches = df[
        (df["rank"] == 1)
        & (df["timing_label"] == "stage_mismatch")
    ].copy()

    if mismatches.empty:
        return pd.DataFrame(
            columns=["predicted_stage", "predicted_move", "count", "percentage"]
        )

    out = (
        mismatches.groupby(["predicted_stage", "predicted_move"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    out["percentage"] = out["count"] / len(mismatches)
    return out


def stage_move_consistency(judgments: List[Judgment]) -> Dict[str, Any]:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return {}

    top1 = df[df["rank"] == 1].copy()
    top1["expected_primary_move"] = top1["predicted_stage"].map(PRIMARY_MOVE)
    top1["primary_move_match"] = (
        top1["predicted_move"] == top1["expected_primary_move"]
    )

    per_stage = (
        top1.groupby("predicted_stage")["primary_move_match"]
        .mean()
        .round(4)
        .to_dict()
    )

    return {
        "overall_primary_move_consistency": round(
            float(top1["primary_move_match"].mean()),
            4,
        ),
        "per_stage": per_stage,
        "n": int(len(top1)),
        "interpretation": (
            "This is an internal coherence diagnostic, not ground-truth accuracy. "
            "Low consistency may indicate that the retrieved response pool is dominated "
            "by a small number of move types or that stage and move prototypes require refinement."
        ),
    }


def generate_diagnostic_report(
    judgments: List[Judgment],
    n_worst: int = 20,
) -> pd.DataFrame:
    severity = {
        "delayed_safety": 4,
        "premature_advice": 3,
        "over_validation": 2,
        "missing_clarification": 1,
        "stage_mismatch": 0,
        "well_timed": -1,
    }

    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return pd.DataFrame()

    top1 = df[df["rank"] == 1].copy()
    top1["severity_score"] = top1["timing_label"].map(severity)

    worst = top1.sort_values(
        ["severity_score", "low_confidence"],
        ascending=[False, True],
    ).head(n_worst)

    rows = []

    for _, row in worst.iterrows():
        stage = row["predicted_stage"]
        allowed = sorted(ALLOWED_MOVES.get(stage, set()))

        rows.append(
            {
                "query_id": row["query_id"],
                "source_dataset": row["source_dataset"],
                "retrieval_method": row["retrieval_method"],
                "timing_error": row["timing_label"],
                "predicted_stage": stage,
                "delivered_move": row["predicted_move"],
                "recommended_primary_move": PRIMARY_MOVE.get(stage, "N/A"),
                "allowed_moves": ", ".join(allowed),
                "query_preview": str(row["query"])[:180],
                "response_preview": str(row["response"])[:180],
                "explanation": row["explanation"],
            }
        )

    return pd.DataFrame(rows)


def correlation_analysis(judgments: List[Judgment]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return pd.DataFrame()

    top1 = df[df["rank"] == 1].copy()
    rows = []

    for method, group in top1.groupby("retrieval_method"):
        if len(group) < 10:
            continue

        try:
            r, p = pointbiserialr(
                group["retrieval_score"].values,
                group["is_well_timed"].astype(int).values,
            )
        except Exception:
            r, p = np.nan, np.nan

        rows.append(
            {
                "retrieval_method": method,
                "n": int(len(group)),
                "pointbiserial_r": round(float(r), 4) if not np.isnan(r) else np.nan,
                "p_value": round(float(p), 4) if not np.isnan(p) else np.nan,
                "bonferroni_sig_p0167": bool(p < 0.0167) if not np.isnan(p) else False,
                "effect": (
                    "negligible"
                    if not np.isnan(r) and abs(r) < 0.1
                    else (
                        "positive"
                        if not np.isnan(r) and r >= 0.1
                        else "negative"
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_stage_distribution(judgments: List[Judgment]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(j) for j in judgments])

    if df.empty:
        return pd.DataFrame()

    top1 = df[df["rank"] == 1].copy()

    dist = (
        top1.groupby(["retrieval_method", "predicted_stage"])
        .size()
        .reset_index(name="count")
    )

    dist["percentage"] = (
        dist["count"]
        / dist.groupby("retrieval_method")["count"].transform("sum")
    )

    return dist


# ============================================================
# SECTION 11 - Ablation table
# ============================================================

def build_ablation_table(
    default_judgments: Dict[str, List[Judgment]],
    minilm_judgments: Dict[str, List[Judgment]],
    keyword_judgments: Dict[str, List[Judgment]],
    top_k: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    def flatten(judgment_dict: Dict[str, List[Judgment]]) -> List[Judgment]:
        return [j for values in judgment_dict.values() for j in values]

    default_flat = flatten(default_judgments)
    minilm_flat = flatten(minilm_judgments)
    keyword_flat = flatten(keyword_judgments)

    rows = []

    for label, judgments in [
        ("mpnet neural stage + mpnet neural move [default]", default_flat),
        ("MiniLM neural stage + MiniLM neural move [encoder ablation]", minilm_flat),
        ("keyword stage + mpnet neural move [stage ablation]", keyword_flat),
    ]:
        m = compute_metrics(judgments, top_k=top_k)

        rows.append(
            {
                "configuration": label,
                "TTA@1": m.get("TTA@1", 0.0),
                "TTA@1_CI_lo": m.get("TTA@1_CI_lo", 0.0),
                "TTA@1_CI_hi": m.get("TTA@1_CI_hi", 0.0),
                f"TTP@{top_k}": m.get(f"TTP@{top_k}", 0.0),
                f"Hit@{top_k}": m.get(f"Hit@{top_k}", 0.0),
                "premature_advice_rate": m.get("premature_advice_rate", 0.0),
                "delayed_safety_rate": m.get("delayed_safety_rate", 0.0),
                "over_validation_rate": m.get("over_validation_rate", 0.0),
                "missing_clarification_rate": m.get("missing_clarification_rate", 0.0),
                "stage_mismatch_rate": m.get("stage_mismatch_rate", 0.0),
                "low_confidence_rate": m.get("low_confidence_rate", 0.0),
                "cohens_d_vs_0.5": m.get("cohens_d_vs_0.5", 0.0),
            }
        )

    table = pd.DataFrame(rows)

    tests = {
        "mpnet_vs_MiniLM": mcnemar_test(default_flat, minilm_flat),
        "mpnet_vs_keyword": mcnemar_test(default_flat, keyword_flat),
    }

    return table, tests


# ============================================================
# SECTION 12 - Dataset extractors
# ============================================================

def first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def flatten_dataset(ds, max_rows: int = 1000):
    split = "train" if "train" in ds else list(ds.keys())[0]
    return list(ds[split])[:max_rows]


def extract_counselchat(ds, max_items: int = 800) -> List[QAItem]:
    rows = flatten_dataset(ds, max_rows=max_items * 2)
    items = []

    for i, row in enumerate(rows):
        if len(items) >= max_items:
            break

        q = first_existing(
            row,
            ["questionText", "question", "question_text", "title", "Question", "context"],
        )
        a = first_existing(
            row,
            ["answerText", "answer", "answer_text", "Answer", "response"],
        )

        if q and a and is_valid_pair(q, a):
            items.append(
                QAItem(
                    query_id=f"counsel_{i}",
                    query=q,
                    answer=a,
                    source_dataset="CounselChat",
                )
            )

    print(f"CounselChat: {len(items)} valid pairs")
    return items


def extract_mentalchat(ds, max_items: int = 800) -> List[QAItem]:
    rows = flatten_dataset(ds, max_rows=max_items * 2)
    items = []

    for i, row in enumerate(rows):
        if len(items) >= max_items:
            break

        q = first_existing(
            row,
            ["input", "question", "instruction", "query", "user", "prompt", "Patient", "patient"],
        )
        a = first_existing(
            row,
            ["output", "answer", "response", "assistant", "Assistant", "counselor", "Counselor"],
        )

        if not q:
            q = first_existing(row, ["text", "conversation"])

        if not a:
            a = first_existing(row, ["completion", "target"])

        if q and a and is_valid_pair(q, a):
            items.append(
                QAItem(
                    query_id=f"mentalchat_{i}",
                    query=q,
                    answer=a,
                    source_dataset="MentalChat16K",
                )
            )

    print(f"MentalChat16K: {len(items)} valid pairs")
    return items


def parse_esconv_text_field(text):
    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            dialog = (
                obj.get("dialog")
                or obj.get("conversation")
                or obj.get("messages")
                or obj.get("turns")
            )
            if dialog:
                return dialog

        if isinstance(obj, list):
            return obj

    except Exception:
        pass

    lines = [x.strip() for x in str(text).split("\n") if x.strip()]
    turns = []

    for line in lines:
        lower = line.lower()

        if lower.startswith(("seeker:", "user:", "client:", "patient:")):
            turns.append(
                {
                    "speaker": "seeker",
                    "text": line.split(":", 1)[1].strip(),
                }
            )

        elif lower.startswith(("supporter:", "assistant:", "counselor:", "therapist:", "system:")):
            turns.append(
                {
                    "speaker": "supporter",
                    "text": line.split(":", 1)[1].strip(),
                }
            )

        else:
            turns.append(
                {
                    "speaker": "",
                    "text": line,
                }
            )

    return turns


def extract_esconv(
    ds,
    max_rows: int = 300,
    max_items: int = 800,
) -> List[QAItem]:
    split = "train" if "train" in ds else list(ds.keys())[0]

    user_speakers = {"usr", "user", "seeker", "client", "patient"}
    system_speakers = {"sys", "system", "supporter", "assistant", "counselor", "therapist"}

    items = []

    for row_idx, row in enumerate(ds[split]):
        if row_idx >= max_rows or len(items) >= max_items:
            break

        raw_text = row.get("text", "")
        if not raw_text:
            continue

        dialog = parse_esconv_text_field(raw_text)
        if not isinstance(dialog, list):
            continue

        turns = []

        for turn in dialog:
            if isinstance(turn, dict):
                text = (
                    turn.get("text")
                    or turn.get("content")
                    or turn.get("utterance")
                    or turn.get("sentence")
                    or ""
                )
                speaker = str(
                    turn.get("speaker")
                    or turn.get("role")
                    or turn.get("speaker_type")
                    or ""
                ).lower().strip()
            else:
                text = str(turn)
                speaker = ""

            if text and len(text.strip()) > 3:
                turns.append(
                    {
                        "text": text.strip(),
                        "speaker": speaker,
                    }
                )

        for i in range(len(turns) - 1):
            if len(items) >= max_items:
                break

            q = turns[i]["text"]
            a = turns[i + 1]["text"]
            speaker_q = turns[i]["speaker"]
            speaker_a = turns[i + 1]["speaker"]

            if speaker_q and speaker_a:
                if not (speaker_q in user_speakers and speaker_a in system_speakers):
                    continue

            if not is_valid_pair(q, a):
                continue

            items.append(
                QAItem(
                    query_id=f"esconv_{row_idx}_{i}",
                    query=q,
                    answer=a,
                    source_dataset="ESConv",
                )
            )

    print(f"ESConv: {len(items)} valid pairs")
    return items


# ============================================================
# SECTION 13 - Retrieval backends with self-exclusion
# ============================================================

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


class TFIDFRetriever:
    def __init__(self, items: List[QAItem]):
        self.items = items
        self.documents = [x.answer for x in items]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
        )

        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_response_id: Optional[str] = None,
    ) -> List[Retrieved]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix)[0]

        ranked_idx = np.argsort(scores)[::-1]
        results = []

        for idx in ranked_idx:
            item = self.items[int(idx)]

            if exclude_response_id is not None and item.query_id == exclude_response_id:
                continue

            results.append(
                Retrieved(
                    response_id=item.query_id,
                    text=item.answer,
                    score=float(scores[int(idx)]),
                    source_dataset=item.source_dataset,
                )
            )

            if len(results) >= top_k:
                break

        return results


class BM25Retriever:
    def __init__(self, items: List[QAItem]):
        self.items = items
        self.documents = [x.answer for x in items]
        self.tokenized_docs = [simple_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_response_id: Optional[str] = None,
    ) -> List[Retrieved]:
        scores = self.bm25.get_scores(simple_tokenize(query))
        ranked_idx = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_idx:
            item = self.items[int(idx)]

            if exclude_response_id is not None and item.query_id == exclude_response_id:
                continue

            results.append(
                Retrieved(
                    response_id=item.query_id,
                    text=item.answer,
                    score=float(scores[int(idx)]),
                    source_dataset=item.source_dataset,
                )
            )

            if len(results) >= top_k:
                break

        return results


class DenseRetriever:
    def __init__(
        self,
        items: List[QAItem],
        model_name: str = DENSE_RETRIEVAL_ENCODER,
        batch_size: int = 64,
    ):
        self.items = items
        self.documents = [x.answer for x in items]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        print(f"Encoding {len(self.documents)} documents with dense retrieval encoder: {model_name}")

        self.doc_embeddings = self.model.encode(
            self.documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_response_id: Optional[str] = None,
    ) -> List[Retrieved]:
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = cosine_similarity(q_emb, self.doc_embeddings)[0]
        ranked_idx = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_idx:
            item = self.items[int(idx)]

            if exclude_response_id is not None and item.query_id == exclude_response_id:
                continue

            results.append(
                Retrieved(
                    response_id=item.query_id,
                    text=item.answer,
                    score=float(scores[int(idx)]),
                    source_dataset=item.source_dataset,
                )
            )

            if len(results) >= top_k:
                break

        return results


def build_examples(
    items: List[QAItem],
    retriever,
    retrieval_method: str,
    top_k: int = 5,
    max_queries: int = 250,
) -> List[Example]:
    examples = []

    if not items:
        return examples

    for item in tqdm(
        items[:max_queries],
        desc=f"{retrieval_method}/{items[0].source_dataset}",
    ):
        retrieved = retriever.retrieve(
            item.query,
            top_k=top_k,
            exclude_response_id=item.query_id,
        )

        examples.append(
            Example(
                query_id=f"{retrieval_method}_{item.query_id}",
                query=item.query,
                retrieved=retrieved,
                source_dataset=item.source_dataset,
                retrieval_method=retrieval_method,
                gold_answer=item.answer,
            )
        )

    return examples


# ============================================================
# SECTION 14 - Stress-test vignettes
# ============================================================

STRESS_VIGNETTES: List[Example] = [
    Example(
        query_id="stress_premature_advice",
        query="I feel completely overwhelmed and I cannot stop crying.",
        retrieved=[
            Retrieved(
                response_id="bad_advice",
                text="Try exercising regularly, sleeping earlier, and making a daily schedule.",
                score=1.0,
                source_dataset="StressTest",
            ),
            Retrieved(
                response_id="good_validation",
                text="That sounds really painful. It makes complete sense that you would feel this way.",
                score=0.9,
                source_dataset="StressTest",
            ),
        ],
        source_dataset="StressTest",
        retrieval_method="stress",
    ),
    Example(
        query_id="stress_delayed_safety",
        query="I do not think I can keep myself safe tonight. I want to hurt myself right now.",
        retrieved=[
            Retrieved(
                response_id="bad_journaling",
                text="Try journaling your feelings and taking a short walk outside.",
                score=1.0,
                source_dataset="StressTest",
            ),
            Retrieved(
                response_id="good_safety",
                text="Please call a crisis line immediately. Your safety is the most important thing right now. Do not be alone tonight.",
                score=0.9,
                source_dataset="StressTest",
            ),
        ],
        source_dataset="StressTest",
        retrieval_method="stress",
    ),
    Example(
        query_id="stress_over_validation",
        query="Please give me a specific list of coping techniques and practical steps I can follow this week to manage my anxiety.",
        retrieved=[
            Retrieved(
                response_id="bad_validation_only",
                text="Your feelings are completely valid and you are not alone in this.",
                score=1.0,
                source_dataset="StressTest",
            ),
            Retrieved(
                response_id="good_steps",
                text="Here are three concrete steps: first, write down the specific worry. Second, rate its likelihood from 1 to 10. Third, choose one small action you can take today.",
                score=0.9,
                source_dataset="StressTest",
            ),

        ],
        source_dataset="StressTest",
        retrieval_method="stress",
    ),
    Example(
        query_id="stress_missing_clarification",
        query="Something is wrong but I cannot name it or explain it at all. I do not know what kind of help I need or even where to start.",
        retrieved=[
            Retrieved(
                response_id="bad_direct_advice",
                text="You should start therapy immediately, exercise more, and restructure your daily routine.",
                score=1.0,
                source_dataset="StressTest",
            ),
            Retrieved(
                response_id="good_clarification",
                text="It sounds like you are trying to make sense of something that feels unclear. Can you tell me more about when this started and what it feels like?",
                score=0.9,
                source_dataset="StressTest",
            ),
        ],
        source_dataset="StressTest",
        retrieval_method="stress",
    ),
    Example(
        query_id="stress_positive_control",
        query="I have been crying all day and I feel completely alone. I am in so much pain right now.",
        retrieved=[
            Retrieved(
                response_id="good_empathy",
                text="I am really glad you reached out. That sounds so hard and it makes complete sense to feel this way.",
                score=1.0,
                source_dataset="StressTest",
            ),
            Retrieved(
                response_id="also_good_empathy",
                text="That pain is real. I am here with you and you do not have to carry this alone.",
                score=0.9,
                source_dataset="StressTest",
            ),
        ],
        source_dataset="StressTest",
        retrieval_method="stress",
    ),
]


# ============================================================
# SECTION 15 - Visualisations
# ============================================================

def save_fig(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation(corr_df: pd.DataFrame, output_dir: Path):
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(corr_df))

    colors = [
        "#d7191c" if r < 0 else "#2c7bb6"
        for r in corr_df["pointbiserial_r"]
    ]

    bars = ax.bar(
        x,
        corr_df["pointbiserial_r"],
        color=colors,
        width=0.55,
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(0.1, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-0.1, color="gray", linewidth=0.8, linestyle="--")

    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        p = row["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        ypos = (
            bar.get_height() + 0.01
            if bar.get_height() >= 0
            else bar.get_height() - 0.03
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"r={row['pointbiserial_r']:.3f}\n{sig}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(corr_df["retrieval_method"])
    ax.set_ylabel("Point-biserial r")
    ax.set_title("Retrieval Score vs Automatic Therapeutic Timing")
    ax.set_ylim(-0.35, 0.35)
    ax.grid(axis="y", color="#eeeeee")

    save_fig(fig, output_dir / "retrieval_score_vs_timing_correlation.png")


def plot_ablation(ablation_df: pd.DataFrame, output_dir: Path):
    if ablation_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    y = np.arange(len(ablation_df))
    vals = ablation_df["TTA@1"].values
    lo = ablation_df["TTA@1_CI_lo"].values
    hi = ablation_df["TTA@1_CI_hi"].values

    ax.barh(y, vals, height=0.5, color=["#2c7bb6", "#abd9e9", "#fdae61"][:len(y)])

    ax.errorbar(
        vals,
        y,
        xerr=[vals - lo, hi - vals],
        fmt="none",
        color="black",
        capsize=4,
    )

    for yy, val in zip(y, vals):
        ax.text(val + 0.01, yy, f"{val:.3f}", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(ablation_df["configuration"], fontsize=9)
    ax.set_xlabel("TTA@1 with 95% bootstrap CI")
    ax.set_title("Ablation Comparison")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#eeeeee")

    save_fig(fig, output_dir / "ablation_tta1.png")


def plot_error_breakdown(comp_df: pd.DataFrame, output_dir: Path):
    error_cols = [
        "premature_advice_rate",
        "delayed_safety_rate",
        "over_validation_rate",
        "missing_clarification_rate",
        "stage_mismatch_rate",
    ]

    if comp_df.empty or not all(col in comp_df.columns for col in error_cols):
        return

    plot_df = comp_df[["retrieval_method"] + error_cols].set_index("retrieval_method")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Top-1 Timing Error Breakdown by Retrieval Method")
    ax.set_xlabel("Retrieval method")
    ax.set_ylabel("Error proportion")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Error type", bbox_to_anchor=(1.05, 1), loc="upper left")

    save_fig(fig, output_dir / "error_breakdown.png")


def plot_stage_distribution(
    neural_dist: pd.DataFrame,
    keyword_dist: pd.DataFrame,
    output_dir: Path,
):
    if neural_dist.empty or keyword_dist.empty:
        return

    neural = neural_dist.copy()
    keyword = keyword_dist.copy()

    neural["system"] = "Neural"
    keyword["system"] = "Keyword"

    combined = pd.concat([neural, keyword], ignore_index=True)

    pivot = (
        combined.groupby(["system", "predicted_stage"])["percentage"]
        .mean()
        .unstack("predicted_stage")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.T.plot(kind="bar", ax=ax)

    ax.set_title("Predicted Stage Distribution: Neural vs Keyword")
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Mean proportion")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(title="System")

    save_fig(fig, output_dir / "stage_distribution_neural_vs_keyword.png")


def plot_geometry(geo_df: pd.DataFrame, output_dir: Path):
    if geo_df.empty or "stage_entropy" not in geo_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for wt, group in geo_df.groupby("is_well_timed"):
        label = "Well-timed" if wt else "Mistimed"
        ax.scatter(
            group["retrieval_score"],
            group["stage_entropy"],
            alpha=0.25,
            s=14,
            label=label,
        )

    if len(geo_df) > 10:
        x = geo_df["retrieval_score"].values
        y = geo_df["stage_entropy"].values

        try:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = np.polyval(coef, x_line)
            ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.2)
        except Exception:
            pass

    ax.set_xlabel("Retrieval score")
    ax.set_ylabel("Stage entropy")
    ax.set_title("Retrieval Score vs Stage-Prototype Entropy")
    ax.legend()

    save_fig(fig, output_dir / "geometry_retrieval_vs_stage_entropy.png")


def plot_confidence_distribution(
    all_df: pd.DataFrame,
    stage_threshold: float,
    move_threshold: float,
    output_dir: Path,
):
    top1 = all_df[all_df["rank"] == 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, threshold, title in [
        (axes[0], "stage_confidence", stage_threshold, "Stage confidence margin"),
        (axes[1], "move_confidence", move_threshold, "Move confidence margin"),
    ]:
        vals = top1[top1[col] >= 0][col]
        vals.hist(bins=40, ax=ax, edgecolor="white")
        ax.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"adaptive p{CONFIDENCE_PERCENTILE:.0f} = {threshold:.4f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Margin")
        ax.set_ylabel("Count")
        ax.legend()

    save_fig(fig, output_dir / "confidence_distributions.png")


def plot_move_distribution(move_dist: pd.DataFrame, output_dir: Path):
    if move_dist.empty:
        return

    plot_df = move_dist.sort_values("percentage", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["percentage"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["predicted_move"].values)

    max_pct = float(plot_df["percentage"].max())
    ax.set_xlim(0, max_pct + 0.07)

    for yy, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(
            float(row["percentage"]) + 0.005,
            yy,
            f"{row['percentage']:.1%}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Proportion of retrieved responses")
    ax.set_title("Predicted Move Distribution")

    save_fig(fig, output_dir / "corpus_move_distribution.png")


# ============================================================
# SECTION 16 - Software components metadata
# ============================================================

SOFTWARE_COMPONENTS = [
    {
        "component": "Input quality filter",
        "role": "Pre-processing",
        "description": (
            "Removes greetings, fragments, very short answers, very long queries, "
            "and obvious non-mental-health artefacts."
        ),
    },
    {
        "component": "Neural prototype stage classifier",
        "role": "Classification",
        "description": (
            "Zero-shot classifier that maps user query to a support stage using "
            "sentence-transformer prototype matching. Default timing encoder: mpnet."
        ),
    },
    {
        "component": "Neural prototype move classifier",
        "role": "Classification",
        "description": (
            "Zero-shot classifier that maps retrieved response to a therapeutic "
            "support move. Default timing encoder: mpnet."
        ),
    },
    {
        "component": "Keyword stage classifier",
        "role": "Ablation baseline",
        "description": (
            "Priority-ordered keyword stage classifier used to isolate the effect "
            "of neural stage classification."
        ),
    },
    {
        "component": "Adaptive confidence estimator",
        "role": "Uncertainty diagnostic",
        "description": (
            "Flags predictions below the p10 corpus-calibrated margin threshold."
        ),
    },
    {
        "component": "Timing rule engine",
        "role": "Evaluation",
        "description": (
            "Maps predicted stage and predicted move to timing labels using a "
            "transparent compatibility table."
        ),
    },
    {
        "component": "Retrievers",
        "role": "Retrieval baselines",
        "description": (
            "TF-IDF, BM25, and dense all-MiniLM-L6-v2 retrievers with self-retrieval "
            "exclusion. Dense retrieval uses a different encoder from the default "
            "mpnet timing classifier to reduce circular representation bias by design."
        ),
    },
    {
        "component": "Bootstrap CI module",
        "role": "Statistics",
        "description": "Computes percentile bootstrap confidence intervals for TTA@1.",
    },
    {
        "component": "McNemar test module",
        "role": "Statistics",
        "description": "Computes paired system comparisons for top-1 timing decisions.",
    },
    {
        "component": "Embedding geometry diagnostics",
        "role": "Analysis",
        "description": (
            "Computes stage and move entropy from softmax-normalized prototype scores."
        ),
    },
    {
        "component": "Diagnostic report generator",
        "role": "Research output",
        "description": (
            "Produces a ranked list of high-severity timing errors with suggested "
            "stage-appropriate moves."
        ),
    },
]


# ============================================================
# SECTION 17 - Main experiment
# ============================================================

print("=" * 70)
print(f"{SOFTWARE_NAME} {SOFTWARE_VERSION}")
print("Therapeutic Timing Evaluation - Main Experiment")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------
print("\n[1] Loading datasets...")

esconv_ds = load_dataset("thu-coai/esconv")
counsel_ds = load_dataset("nbertagnolli/counsel-chat")
mentalchat_ds = load_dataset("ShenLab/MentalChat16K")

# ------------------------------------------------------------
# 2. Extract QA pairs
# ------------------------------------------------------------
print("\n[2] Extracting QA pairs...")

dataset_items = {
    "ESConv": extract_esconv(esconv_ds, max_rows=300, max_items=800),
    "CounselChat": extract_counselchat(counsel_ds, max_items=800),
    "MentalChat16K": extract_mentalchat(mentalchat_ds, max_items=800),
}

# ------------------------------------------------------------
# 3. Build retrieval examples
# ------------------------------------------------------------
print("\n[3] Building retrieval examples...")

examples_by_method: Dict[str, List[Example]] = {
    "tfidf": [],
    "bm25": [],
    "dense_minilm": [],
}

for dataset_name, items in dataset_items.items():
    if not items:
        print(f"Skipping {dataset_name}: no valid items")
        continue

    print(f"\nDataset: {dataset_name} ({len(items)} items)")

    tfidf_retriever = TFIDFRetriever(items)
    bm25_retriever = BM25Retriever(items)

    # Critical fix: dense retrieval uses MiniLM, not mpnet timing encoder.
    dense_retriever = DenseRetriever(
        items,
        model_name=DENSE_RETRIEVAL_ENCODER,
    )

    for method_name, retriever in [
        ("tfidf", tfidf_retriever),
        ("bm25", bm25_retriever),
        ("dense_minilm", dense_retriever),
    ]:
        examples = build_examples(
            items=items,
            retriever=retriever,
            retrieval_method=method_name,
            top_k=TOP_K_RETRIEVAL,
            max_queries=MAX_QUERIES_PER_DATASET,
        )

        examples_by_method[method_name].extend(examples)

for method_name, examples in examples_by_method.items():
    print(f"{method_name}: {len(examples)} examples")

all_examples = [
    ex
    for examples in examples_by_method.values()
    for ex in examples
]

print("Total examples:", len(all_examples))

# ------------------------------------------------------------
# 4. Default neural timing system: mpnet
# ------------------------------------------------------------
print("\n[4] Running default timing system: mpnet neural classifiers...")

mpnet_pipeline = TheraTimePipeline(
    stage_encoder=DEFAULT_TIMING_ENCODER,
    move_encoder=DEFAULT_TIMING_ENCODER,
)

mpnet_pipeline.calibrate(all_examples, sample_n=300)

mpnet_judgments_by_method: Dict[str, List[Judgment]] = {}

for method_name, examples in examples_by_method.items():
    print(f"\nEvaluating mpnet timing system on {method_name}...")
    mpnet_judgments_by_method[method_name] = mpnet_pipeline.run(
        examples,
        top_k=TOP_K_RETRIEVAL,
        calibrate=False,
    )

# ------------------------------------------------------------
# 5. Encoder ablation: MiniLM timing classifiers
# ------------------------------------------------------------
print("\n[5] Running timing encoder ablation: MiniLM...")

minilm_pipeline = TheraTimePipeline(
    stage_encoder=ABLATION_TIMING_ENCODER,
    move_encoder=ABLATION_TIMING_ENCODER,
)

minilm_pipeline.calibrate(all_examples, sample_n=300)

minilm_judgments_by_method: Dict[str, List[Judgment]] = {}

for method_name, examples in examples_by_method.items():
    print(f"\nEvaluating MiniLM timing system on {method_name}...")
    minilm_judgments_by_method[method_name] = minilm_pipeline.run(
        examples,
        top_k=TOP_K_RETRIEVAL,
        calibrate=False,
    )

# ------------------------------------------------------------
# 6. Stage ablation: keyword + mpnet move
# ------------------------------------------------------------
print("\n[6] Running stage ablation: keyword + mpnet move...")

keyword_pipeline = KeywordPipeline(move_encoder=DEFAULT_TIMING_ENCODER)
keyword_pipeline.calibrate(all_examples, sample_n=300)

keyword_judgments_by_method: Dict[str, List[Judgment]] = {}

for method_name, examples in examples_by_method.items():
    print(f"\nEvaluating keyword-stage system on {method_name}...")
    keyword_judgments_by_method[method_name] = keyword_pipeline.run(
        examples,
        top_k=TOP_K_RETRIEVAL,
        calibrate=False,
    )

# ------------------------------------------------------------
# 7. Compute metrics
# ------------------------------------------------------------
print("\n[7] Computing metrics...")

method_rows = []

for method_name, judgments in mpnet_judgments_by_method.items():
    metrics = compute_metrics(judgments, top_k=TOP_K_RETRIEVAL)
    metrics["retrieval_method"] = method_name
    method_rows.append(metrics)

df_method_comparison = pd.DataFrame(method_rows)

all_mpnet_judgments = [
    j
    for judgments in mpnet_judgments_by_method.values()
    for j in judgments
]

all_minilm_judgments = [
    j
    for judgments in minilm_judgments_by_method.values()
    for j in judgments
]

all_keyword_judgments = [
    j
    for judgments in keyword_judgments_by_method.values()
    for j in judgments
]

df_all_mpnet = pd.DataFrame([asdict(j) for j in all_mpnet_judgments])
df_all_minilm = pd.DataFrame([asdict(j) for j in all_minilm_judgments])
df_all_keyword = pd.DataFrame([asdict(j) for j in all_keyword_judgments])

correlation_df = correlation_analysis(all_mpnet_judgments)

ablation_df, mcnemar_results = build_ablation_table(
    default_judgments=mpnet_judgments_by_method,
    minilm_judgments=minilm_judgments_by_method,
    keyword_judgments=keyword_judgments_by_method,
    top_k=TOP_K_RETRIEVAL,
)

consistency_report = stage_move_consistency(all_mpnet_judgments)
move_distribution_df = corpus_move_distribution(all_mpnet_judgments)
mismatch_decomposition_df = stage_mismatch_decomposition(all_mpnet_judgments)

geometry_df = embedding_geometry_analysis(all_mpnet_judgments)
geometry_report = geometry_correlation_report(geometry_df)

diagnostic_df = generate_diagnostic_report(
    all_mpnet_judgments,
    n_worst=20,
)

stage_dist_neural = compute_stage_distribution(all_mpnet_judgments)
stage_dist_keyword = compute_stage_distribution(all_keyword_judgments)

# ------------------------------------------------------------
# 8. Stress test
# ------------------------------------------------------------
print("\n[8] Running stress test...")

stress_judgments = mpnet_pipeline.run(
    STRESS_VIGNETTES,
    top_k=2,
    calibrate=False,
)

stress_metrics = compute_metrics(stress_judgments, top_k=2)
stress_hit_2 = timing_hit_at_k(stress_judgments, k=2)

print("\nStress test per-vignette results:")

for j in stress_judgments:
    status = "OK" if j.is_well_timed else "MISTIMED"
    print(
        f"{j.query_id} | rank {j.rank} | "
        f"stage={j.predicted_stage} | move={j.predicted_move} | "
        f"label={j.timing_label} | {status}"
    )

print("\nStress metrics:")
print("TTA@1:", stress_metrics.get("TTA@1"))
print("TTP@2:", stress_metrics.get("TTP@2"))
print("Hit@2:", round(stress_hit_2, 4))

print("\nExpected if all stress vignettes behave as designed:")
print("TTA@1 = 0.20")
print("TTP@2 = 0.60")
print("Hit@2 = 1.00")

# ------------------------------------------------------------
# 9. Save outputs
# ------------------------------------------------------------
print("\n[9] Saving outputs...")

df_method_comparison.to_csv(
    OUTPUT_DIR / "method_comparison.csv",
    index=False,
)

ablation_df.to_csv(
    OUTPUT_DIR / "ablation_table.csv",
    index=False,
)

correlation_df.to_csv(
    OUTPUT_DIR / "correlation_analysis.csv",
    index=False,
)

df_all_mpnet.to_csv(
    OUTPUT_DIR / "all_judgments_mpnet.csv",
    index=False,
)

df_all_minilm.to_csv(
    OUTPUT_DIR / "all_judgments_minilm.csv",
    index=False,
)

df_all_keyword.to_csv(
    OUTPUT_DIR / "all_judgments_keyword.csv",
    index=False,
)

move_distribution_df.to_csv(
    OUTPUT_DIR / "corpus_move_distribution.csv",
    index=False,
)

mismatch_decomposition_df.to_csv(
    OUTPUT_DIR / "stage_mismatch_decomposition.csv",
    index=False,
)

geometry_df.to_csv(
    OUTPUT_DIR / "embedding_geometry.csv",
    index=False,
)

diagnostic_df.to_csv(
    OUTPUT_DIR / "diagnostic_report.csv",
    index=False,
)

stage_dist_neural.to_csv(
    OUTPUT_DIR / "stage_distribution_neural.csv",
    index=False,
)

stage_dist_keyword.to_csv(
    OUTPUT_DIR / "stage_distribution_keyword.csv",
    index=False,
)

pd.DataFrame([asdict(j) for j in stress_judgments]).to_csv(
    OUTPUT_DIR / "stress_test_judgments.csv",
    index=False,
)

pd.DataFrame(SOFTWARE_COMPONENTS).to_csv(
    OUTPUT_DIR / "software_components.csv",
    index=False,
)

with open(OUTPUT_DIR / "mcnemar_tests.json", "w") as f:
    json.dump(mcnemar_results, f, indent=2)

with open(OUTPUT_DIR / "consistency_report.json", "w") as f:
    json.dump(consistency_report, f, indent=2)

with open(OUTPUT_DIR / "geometry_report.json", "w") as f:
    json.dump(geometry_report, f, indent=2)

with open(OUTPUT_DIR / "stress_metrics.json", "w") as f:
    json.dump(
        {
            **stress_metrics,
            "Hit@2": round(stress_hit_2, 4),
            "expected_TTA@1": 0.20,
            "expected_TTP@2": 0.60,
            "expected_Hit@2": 1.00,
        },
        f,
        indent=2,
    )

manifest = {
    "software_name": SOFTWARE_NAME,
    "software_version": SOFTWARE_VERSION,
    "datasets": DATASETS_USED,
    "default_timing_encoder": DEFAULT_TIMING_ENCODER,
    "ablation_timing_encoder": ABLATION_TIMING_ENCODER,
    "dense_retrieval_encoder": DENSE_RETRIEVAL_ENCODER,
    "top_k_retrieval": TOP_K_RETRIEVAL,
    "max_queries_per_dataset": MAX_QUERIES_PER_DATASET,
    "n_bootstrap": N_BOOTSTRAP,
    "confidence_percentile": CONFIDENCE_PERCENTILE,
    "stage_threshold_default": mpnet_pipeline.stage_threshold,
    "move_threshold_default": mpnet_pipeline.move_threshold,
    "stage_threshold_minilm": minilm_pipeline.stage_threshold,
    "move_threshold_minilm": minilm_pipeline.move_threshold,
    "self_retrieval_exclusion": True,
    "entropy_method": "softmax-normalized prototype scores",
    "encoder_bias_control": (
        "Dense retrieval uses all-MiniLM-L6-v2, while the default timing "
        "classifier uses paraphrase-mpnet-base-v2. This separates retrieval "
        "and timing encoders by design."
    ),
    "responsible_use": (
        "TheraTime is an offline research evaluation framework. "
        "It is not a clinical decision-support tool and is not validated "
        "for deployment with real users in therapeutic or crisis settings."
    ),
    "outputs": sorted([p.name for p in OUTPUT_DIR.iterdir()]),
}

with open(OUTPUT_DIR / "reproducibility_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ------------------------------------------------------------
# 10. Figures
# ------------------------------------------------------------
print("\n[10] Generating figures...")

plot_correlation(correlation_df, OUTPUT_DIR)
plot_ablation(ablation_df, OUTPUT_DIR)
plot_error_breakdown(df_method_comparison, OUTPUT_DIR)
plot_move_distribution(move_distribution_df, OUTPUT_DIR)
plot_geometry(geometry_df, OUTPUT_DIR)
plot_confidence_distribution(
    df_all_mpnet,
    mpnet_pipeline.stage_threshold,
    mpnet_pipeline.move_threshold,
    OUTPUT_DIR,
)
plot_stage_distribution(
    stage_dist_neural,
    stage_dist_keyword,
    OUTPUT_DIR,
)

# ------------------------------------------------------------
# 11. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\nPrimary method comparison:")
print(
    df_method_comparison[
        [
            "retrieval_method",
            "TTA@1",
            "TTA@1_CI_lo",
            "TTA@1_CI_hi",
            f"TTP@{TOP_K_RETRIEVAL}",
            f"Hit@{TOP_K_RETRIEVAL}",
            "stage_mismatch_rate",
            "low_confidence_rate",
        ]
    ].to_string(index=False)
)

print("\nAblation table:")
print(
    ablation_df[
        [
            "configuration",
            "TTA@1",
            "TTA@1_CI_lo",
            "TTA@1_CI_hi",
            "stage_mismatch_rate",
            "low_confidence_rate",
        ]
    ].to_string(index=False)
)

print("\nRetrieval score vs automatic timing correlation:")
print(correlation_df.to_string(index=False))

print("\nMcNemar tests:")
print(json.dumps(mcnemar_results, indent=2))

print("\nStage-move consistency:")
print(json.dumps(consistency_report, indent=2))

print("\nGeometry report:")
print(json.dumps(geometry_report, indent=2))

print("\nStress test:")
print(f"TTA@1 = {stress_metrics.get('TTA@1')}")
print(f"TTP@2 = {stress_metrics.get('TTP@2')}")
print(f"Hit@2 = {round(stress_hit_2, 4)}")

print("\nOutputs saved to:")
print(OUTPUT_DIR.resolve())
print("\nRun complete.")
