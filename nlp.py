"""NLP and FAQ Matching Engine for Physiotherapy Clinic Chatbot.

Uses NLTK for preprocessing and tokenization, and sentence-transformers
(all-MiniLM-L6-v2) for semantic vector similarity matching.
Implements emergency keyword detection, contextual boosting (using the last 3
messages and session last_category), confidence thresholding (0.55),
and logging of every exchange.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

import db

# Configure logging
logger = logging.getLogger("physio_nlp")

# SSL fix for NLTK downloads if needed in certain environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure required NLTK packages are downloaded
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning("Could not download NLTK resource %s: %s", resource, e)

# Constants
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.55
CATEGORY_BOOST_WEIGHT = 0.08
CONTEXT_BOOST_WEIGHT = 0.05
PRECISION_KEYWORD_BOOST = 0.08

# Curated keyword patterns for precision boosting and domain term alignment
FAQ_KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "apt_first_visit": [
        "first visit", "initial", "book", "booking", "schedule", "new patient",
        "evaluation", "register", "appointment", "first consultation"
    ],
    "apt_cancellation": [
        "cancel", "cancelling", "cancellation", "reschedule", "rescheduling",
        "penalty", "missed", "no-show", "24 hours notice", "fee"
    ],
    "apt_telehealth": [
        "telehealth", "virtual", "online", "zoom", "video call", "remote",
        "from home", "video consultation", "virtual appointment"
    ],
    "proc_walk_in": [
        "walk in", "walk-in", "walkin", "same day", "same-day", "today urgently",
        "without booking", "emergency opening", "urgent slot"
    ],
    "prep_clothing": [
        "wear", "clothes", "clothing", "dress", "attire", "shorts", "shoes",
        "sneakers", "tank top", "outfit"
    ],
    "prep_documents": [
        "document", "documents", "paperwork", "bring", "photo id", "mri",
        "x-ray", "xray", "imaging", "referral note", "prescription"
    ],
    "ins_coverage": [
        "insurance", "coverage", "direct billing", "bill", "blue cross",
        "sun life", "manulife", "wsib", "wcb", "benefits", "covered"
    ],
    "ins_doctor_referral": [
        "referral", "prescription", "doctor's referral", "physician referral",
        "doctor note", "referral required", "need a doctor"
    ],
    "ins_pricing": [
        "price", "pricing", "cost", "fee", "fees", "rate", "rates",
        "self-pay", "self pay", "out of pocket", "how much does"
    ],
    "ins_payment_methods": [
        "payment methods", "credit card", "debit", "hsa", "fsa", "apple pay",
        "forms of payment", "cash", "financing", "pay for"
    ],
    "treat_back_pain": [
        "back", "lower back", "lumbar", "sciatica", "disc", "herniated",
        "pinched nerve", "spine", "radiating pain"
    ],
    "treat_sports_injuries": [
        "sport", "sports", "athlete", "athletic", "ankle sprain", "acl",
        "mcl", "tennis elbow", "ligament", "runner's knee", "sprain"
    ],
    "treat_shoulder_rotator": [
        "shoulder", "rotator cuff", "frozen shoulder", "impingement",
        "bursitis", "labrum", "arm above head", "adhesive capsulitis"
    ],
    "treat_dry_needling": [
        "dry needling", "needling", "trigger point", "knot", "knots",
        "intramuscular", "ims", "acupuncture", "monofilament"
    ],
    "treat_post_surgery": [
        "surgery", "surgical", "post-op", "post op", "post-operative",
        "joint replacement", "arthroscopy", "rehab after"
    ],
    "treat_neck_headaches": [
        "neck", "headache", "headaches", "cervicogenic", "whiplash",
        "tech neck", "stiff neck", "tension headache", "cervical"
    ],
    "proc_num_sessions": [
        "how many sessions", "how many visits", "how long does it take",
        "how many weeks", "recovery duration", "number of sessions"
    ],
    "proc_pain_during_treatment": [
        "hurt", "painful", "pain during", "soreness", "sore after",
        "discomfort", "pain during treatment", "will it hurt"
    ],
    "proc_home_exercises": [
        "home exercise", "home exercises", "homework", "routine at home",
        "hep", "exercises at home", "daily exercises"
    ],
    "gen_hours_location": [
        "hours", "location", "address", "opening hours", "open", "saturday",
        "weekend", "weekday", "where is", "when are you open"
    ],
    "gen_parking_accessibility": [
        "parking", "park", "wheelchair", "accessible", "accessibility",
        "ramp", "elevator", "handicap", "stalls", "disabled"
    ],
}

FALLBACK_MESSAGE = (
    "I'm sorry, I'm not completely certain about your inquiry. Could you please rephrase, "
    "or are you looking for one of these topics?"
)

EMERGENCY_RESPONSE = (
    "🚨 URGENT MEDICAL ADVISORY: Based on the symptoms described, this may be a critical medical emergency "
    "requiring immediate evaluation. Physiotherapy clinics do not provide emergency medical services. "
    "Please call 911 (or your local emergency number) or proceed to the nearest emergency department immediately."
)

# Curated medical & red-flag emergency keywords/phrases
EMERGENCY_PATTERNS = [
    # Neurological / Cauda Equina red flags
    r"\b(cauda equina)\b",
    r"\b(saddle anesthesia|saddle anaesthesia)\b",
    r"\b(loss of (bowel|bladder) control)\b",
    r"\b(bowel incontinence|urinary incontinence|bladder incontinence)\b",
    r"\b(numbness in (groin|genitals|buttocks|between (my )?legs))\b",
    r"\b(cannot feel (my )?legs|can't feel (my )?legs|paralysis|paralyzed|paralysed)\b",
    r"\b(sudden weakness in (both|my) legs|legs gave out suddenly)\b",
    r"\b(foot drop (sudden|acute))\b",
    # Cardiovascular & Systemic emergencies
    r"\b(chest pain|pressure in chest|crushing chest pain)\b",
    r"\b(heart attack|cardiac arrest)\b",
    r"\b(shortness of breath|difficulty breathing|cannot breathe|can't breathe)\b",
    r"\b(stroke|facial droop|slurred speech)\b",
    r"\b(passed out|fainted|loss of consciousness|unresponsive)\b",
    r"\b(coughing (up )?blood)\b",
    # Acute Trauma / Severe Fractures
    r"\b(bone sticking out|bone protruding|compound fracture|open fracture)\b",
    r"\b(uncontrolled bleeding|profuse bleeding|bleeding heavily)\b",
    r"\b(severe head injury|concussion vomiting|unbearable sudden headache)\b",
    # Direct Emergency Call-outs
    r"\b(call 911|dial 911|emergency room|call (an )?ambulance|urgent care right now)\b",
]

_COMPILED_EMERGENCY_REGEX = [
    re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS
]


class TextPreprocessor:
    """NLTK-based text preprocessing and normalization."""

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words: Set[str] = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text for embedding and matching."""
        if not text:
            return ""
        # Strip trailing/leading spaces and replace excessive whitespace
        cleaned = re.sub(r"\s+", " ", text.strip())
        return cleaned

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize, lowercase, strip punctuation, and lemmatize tokens."""
        if not text:
            return []
        tokens = nltk.word_tokenize(text.lower())
        lemmatized: List[str] = []
        for token in tokens:
            # Keep alphanumeric words only, discard pure punctuation
            if re.match(r"^\w+$", token):
                lemma = self.lemmatizer.lemmatize(token)
                lemmatized.append(lemma)
        return lemmatized

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract significant content keywords excluding common stopwords."""
        tokens = self.tokenize_and_lemmatize(text)
        return {t for t in tokens if t not in self.stop_words and len(t) > 2}


def check_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Check user text against emergency red-flag keywords and phrases.

    Returns (is_emergency, matched_pattern_text).
    """
    if not text:
        return False, None

    normalized_text = text.strip()
    for pattern in _COMPILED_EMERGENCY_REGEX:
        match = pattern.search(normalized_text)
        if match:
            return True, match.group(0)

    return False, None


class FAQMatcher:
    """Semantic vector similarity FAQ matcher with contextual boosting."""

    def __init__(
        self,
        faqs_json_path: Optional[str | Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.preprocessor = TextPreprocessor()
        self.model_name = model_name
        self.faqs_json_path = (
            Path(faqs_json_path) if faqs_json_path else db.DEFAULT_FAQS_PATH
        )

        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)

        # Store loaded FAQs and question variants
        self.faqs: List[Dict[str, Any]] = []
        self.faq_by_id: Dict[str, Dict[str, Any]] = {}
        self.variant_texts: List[str] = []
        self.variant_faq_ids: List[str] = []
        self.variant_embeddings: Optional[np.ndarray] = None

        self.load_and_embed_faqs()

    def load_and_embed_faqs(self) -> None:
        """Load FAQs from JSON (or DB fallback) and precompute embeddings."""
        if self.faqs_json_path.exists():
            with open(self.faqs_json_path, "r", encoding="utf-8") as f:
                self.faqs = json.load(f)
        else:
            self.faqs = db.get_all_faqs()

        self.faq_by_id = {faq["id"]: faq for faq in self.faqs}
        self.variant_texts = []
        self.variant_faq_ids = []

        for faq in self.faqs:
            f_id = faq["id"]
            primary_q = self.preprocessor.clean_text(faq.get("question", ""))
            if primary_q:
                self.variant_texts.append(primary_q)
                self.variant_faq_ids.append(f_id)

            for alt_q in faq.get("alt_questions", []):
                cleaned_alt = self.preprocessor.clean_text(alt_q)
                if cleaned_alt:
                    self.variant_texts.append(cleaned_alt)
                    self.variant_faq_ids.append(f_id)

        if not self.variant_texts:
            logger.warning("No FAQ question variants found to embed!")
            self.variant_embeddings = np.empty((0, 384))
            return

        logger.info(
            "Encoding %d question variants across %d FAQs...",
            len(self.variant_texts),
            len(self.faqs),
        )
        embeddings = self.model.encode(
            self.variant_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.variant_embeddings = np.array(embeddings, dtype=np.float32)

    def _select_answer_variant(
        self,
        faq: Dict[str, Any],
        last_3_messages: Optional[List[Dict[str, Any]]] = None,
        variation_index: Optional[int] = None,
    ) -> str:
        """Select a diverse, clinically accurate answer variation from the FAQ.

        Rotates or alternates variations across conversation turns to ensure
        responses remain dynamic, natural, and never repetitive.
        """
        variants = faq.get("answers")
        if not variants or not isinstance(variants, list) or len(variants) == 0:
            return faq.get("answer", "")

        if variation_index is not None:
            return variants[variation_index % len(variants)]

        # Check if the assistant previously returned any of these variants in the session
        recent_assistant_replies = [
            m.get("text", "").strip()
            for m in (last_3_messages or [])
            if m.get("role") == "assistant"
        ]

        for variant in variants:
            if variant.strip() not in recent_assistant_replies:
                return variant

        # If all were seen recently in this session, return the primary/first variant
        return variants[0]

    def match(
        self,
        query: str,
        last_3_messages: Optional[List[Dict[str, Any]]] = None,
        session_last_category: Optional[str] = None,
        variation_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Match query against FAQs with contextual history, keyword precision, and category boosting.

        Parameters:
        - query: Raw user message
        - last_3_messages: Recent conversation exchanges from the session
        - session_last_category: The session's last discussed category
        - variation_index: Optional explicit answer variation index

        Returns:
        - Dictionary with:
          answer: str
          confidence: float
          matched_faq_id: Optional[str]
          category: Optional[str]
          flag: str
          suggested_questions: Optional[List[str]]
          available_answers: Optional[List[str]]
          debug_info: Dict[str, Any]
        """
        cleaned_query = self.preprocessor.clean_text(query)
        if not cleaned_query or self.variant_embeddings is None or len(self.variant_texts) == 0:
            return self._build_fallback(0.0)

        # 1. Encode user query
        query_emb = self.model.encode(
            cleaned_query, normalize_embeddings=True, show_progress_bar=False
        )

        # 2. Vector cosine similarities (unit-normalized dot product)
        similarities = np.dot(self.variant_embeddings, query_emb)

        # 3. Aggregate candidate FAQ base scores (max score among question variants)
        faq_base_scores: Dict[str, float] = {}
        for idx, f_id in enumerate(self.variant_faq_ids):
            sim = float(similarities[idx])
            if f_id not in faq_base_scores or sim > faq_base_scores[f_id]:
                faq_base_scores[f_id] = sim

        # 4. Contextual Boosting
        # A) Category boost: boost if candidate FAQ belongs to session_last_category
        # B) Recent message context boost: lexical/topic continuity from last 3 messages
        recent_context_text = ""
        recent_keywords: Set[str] = set()
        if last_3_messages:
            recent_texts = [
                m.get("text", "")
                for m in last_3_messages
                if m.get("role") in ("user", "assistant")
            ]
            recent_context_text = " ".join(recent_texts)
            recent_keywords = self.preprocessor.extract_keywords(recent_context_text)

        candidate_evaluations: List[Dict[str, Any]] = []
        lower_query = cleaned_query.lower()

        for faq in self.faqs:
            f_id = faq["id"]
            base_score = faq_base_scores.get(f_id, 0.0)

            # Category boost
            cat_boost = 0.0
            if session_last_category and faq.get("category") == session_last_category:
                cat_boost = CATEGORY_BOOST_WEIGHT

            # Context boost from recent messages
            context_boost = 0.0
            if recent_keywords:
                faq_content = f"{faq.get('question', '')} {faq.get('category', '')}"
                faq_keywords = self.preprocessor.extract_keywords(faq_content)
                overlap = recent_keywords.intersection(faq_keywords)
                if overlap:
                    # Scale boost by overlap (up to CONTEXT_BOOST_WEIGHT)
                    context_boost = min(
                        CONTEXT_BOOST_WEIGHT, len(overlap) * 0.02
                    )

            # Precision boost from clinical/domain keyword patterns
            precision_boost = 0.0
            kw_patterns = FAQ_KEYWORD_PATTERNS.get(f_id, [])
            if kw_patterns:
                matches = sum(1 for kw in kw_patterns if kw in lower_query)
                if matches > 0:
                    precision_boost = min(PRECISION_KEYWORD_BOOST, matches * 0.04)

            final_score = min(1.0, base_score + cat_boost + context_boost + precision_boost)

            candidate_evaluations.append(
                {
                    "faq": faq,
                    "faq_id": f_id,
                    "base_score": round(base_score, 4),
                    "category_boost": round(cat_boost, 4),
                    "context_boost": round(context_boost, 4),
                    "precision_boost": round(precision_boost, 4),
                    "final_score": round(final_score, 4),
                }
            )

        # Sort candidate evaluations descending by final_score
        candidate_evaluations.sort(key=lambda c: c["final_score"], reverse=True)
        best_candidate = candidate_evaluations[0] if candidate_evaluations else None

        best_score = best_candidate["final_score"] if best_candidate else 0.0

        # 5. Check Confidence Threshold (0.55)
        if best_candidate and best_score >= CONFIDENCE_THRESHOLD:
            best_faq = best_candidate["faq"]
            flag = (
                "boosted_match"
                if (
                    best_candidate["category_boost"] > 0
                    or best_candidate["context_boost"] > 0
                )
                and best_candidate["base_score"] < CONFIDENCE_THRESHOLD
                else "matched"
            )

            # Select varied answer from the FAQ's response variations
            selected_answer = self._select_answer_variant(
                best_faq,
                last_3_messages=last_3_messages,
                variation_index=variation_index,
            )

            available_answers = best_faq.get("answers")
            if not available_answers:
                available_answers = [best_faq.get("answer", "")]

            return {
                "answer": selected_answer,
                "confidence": best_score,
                "matched_faq_id": best_faq["id"],
                "category": best_faq["category"],
                "flag": flag,
                "suggested_questions": None,
                "available_answers": available_answers,
                "debug_info": {
                    "base_score": best_candidate["base_score"],
                    "category_boost": best_candidate["category_boost"],
                    "context_boost": best_candidate["context_boost"],
                    "precision_boost": best_candidate["precision_boost"],
                    "final_score": best_candidate["final_score"],
                },
            }
        else:
            # Low confidence fallback with top suggested candidate questions
            top_suggestions = [
                c["faq"]["question"]
                for c in candidate_evaluations[:3]
                if c.get("faq")
            ]
            return self._build_fallback(best_score, top_suggestions)

    def _build_fallback(
        self, confidence: float, suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Construct a standardized fallback response."""
        if not suggestions:
            suggestions = [
                "How do I book an initial physiotherapy appointment?",
                "What should I wear to my physiotherapy appointment?",
                "Do you accept insurance and offer direct billing?",
            ]
        return {
            "answer": FALLBACK_MESSAGE,
            "confidence": round(confidence, 4),
            "matched_faq_id": None,
            "category": None,
            "flag": "low_confidence_fallback",
            "suggested_questions": suggestions,
            "debug_info": {"final_score": round(confidence, 4)},
        }


def process_chat(
    session_id: str,
    message: str,
    matcher: FAQMatcher,
    db_path: Optional[str | Path] = None,
    variation_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end chat exchange logic:

    1. Check emergency keywords first -> return urgent-care advisory if matched.
    2. Load or initialize session.
    3. Retrieve the last 3 messages and session last_category to boost candidate FAQs.
    4. Return best answer with confidence score and response variations.
    5. If confidence < 0.55, return fallback with suggested questions.
    6. Log every exchange into the database.
    """
    # 1. Emergency check
    is_emergency, trigger_keyword = check_emergency(message)
    if is_emergency:
        logger.warning(
            "Emergency keyword '%s' detected for session %s",
            trigger_keyword,
            session_id,
        )
        # Ensure session exists
        db.get_or_create_session(session_id, db_path=db_path)

        # Log emergency exchange
        db.log_message(
            session_id=session_id,
            role="user",
            text=message,
            matched_faq_id=None,
            confidence=1.0,
            flag="emergency_user",
            db_path=db_path,
        )
        db.log_message(
            session_id=session_id,
            role="assistant",
            text=EMERGENCY_RESPONSE,
            matched_faq_id=None,
            confidence=1.0,
            flag="emergency_response",
            db_path=db_path,
        )

        return {
            "session_id": session_id,
            "answer": EMERGENCY_RESPONSE,
            "confidence": 1.0,
            "matched_faq_id": None,
            "category": "Emergency",
            "flag": "emergency",
            "suggested_questions": [],
        }

    # 2. Get or create session
    session = db.get_or_create_session(session_id, db_path=db_path)
    last_category = session.get("last_category")

    # 3. Retrieve last 3 messages from session for contextual boosting
    last_3_messages = db.get_recent_messages(session_id, limit=3, db_path=db_path)

    # 4. Match FAQ with contextual boosting, keyword precision, and variant selection
    match_result = matcher.match(
        query=message,
        last_3_messages=last_3_messages,
        session_last_category=last_category,
        variation_index=variation_index,
    )

    confidence = match_result["confidence"]
    matched_faq_id = match_result.get("matched_faq_id")
    category = match_result.get("category")
    flag = match_result["flag"]
    answer = match_result["answer"]
    suggested_questions = match_result.get("suggested_questions")
    available_answers = match_result.get("available_answers")

    # 5. Update session category if high confidence match
    if confidence >= CONFIDENCE_THRESHOLD and category:
        db.update_session_category(session_id, category, db_path=db_path)

    # 6. Log user message and assistant reply
    db.log_message(
        session_id=session_id,
        role="user",
        text=message,
        matched_faq_id=matched_faq_id,
        confidence=confidence,
        flag="normal" if confidence >= CONFIDENCE_THRESHOLD else "low_confidence",
        db_path=db_path,
    )

    db.log_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        matched_faq_id=matched_faq_id,
        confidence=confidence,
        flag=flag,
        db_path=db_path,
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "confidence": confidence,
        "matched_faq_id": matched_faq_id,
        "category": category,
        "flag": flag,
        "suggested_questions": suggested_questions,
        "available_answers": available_answers,
    }
