"""Comprehensive Test Suite for db.py and nlp.py.

Validates:
1. SQLite schema and operations (faqs, sessions, messages).
2. NLTK preprocessing and tokenization.
3. Emergency keywords detection and fixed urgent-care message.
4. Sentence-transformers semantic matching (question + alt_questions).
5. Contextual boosting (last 3 messages + session last_category).
6. Confidence threshold (< 0.55 fallback with suggested questions).
7. Full exchange logging in the database.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import db
import nlp


class TestPhysiotherapyChatbotComponents(unittest.TestCase):
    """Test suite for SQLite DB and NLP engine."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary test directory and load the FAQ matcher once."""
        cls.test_dir = tempfile.mkdtemp(prefix="physio_test_")
        cls.test_db_path = Path(cls.test_dir) / "test_chatbot.db"
        cls.test_faqs_path = Path(__file__).resolve().parent / "data" / "faqs.json"

        # Initialize SQLite DB
        db.init_db(db_path=cls.test_db_path, faqs_json_path=cls.test_faqs_path)

        # Initialize NLP Matcher
        cls.matcher = nlp.FAQMatcher(faqs_json_path=cls.test_faqs_path)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary test artifacts."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_db_initialization_and_seeding(self) -> None:
        """Verify SQLite tables are created and seeded with FAQs."""
        faqs = db.get_all_faqs(db_path=self.test_db_path)
        self.assertGreater(len(faqs), 10, "Expected at least 10 seeded FAQs")

        # Check structure of a seeded FAQ
        sample_faq = faqs[0]
        self.assertIn("id", sample_faq)
        self.assertIn("category", sample_faq)
        self.assertIn("question", sample_faq)
        self.assertIn("alt_questions", sample_faq)
        self.assertIn("answer", sample_faq)
        self.assertIsInstance(sample_faq["alt_questions"], list)

    def test_02_session_lifecycle(self) -> None:
        """Verify session creation and category updates."""
        session_id = "test_sess_001"
        sess = db.get_or_create_session(session_id, db_path=self.test_db_path)
        self.assertEqual(sess["id"], session_id)
        self.assertIsNone(sess["last_category"])
        self.assertIn("created_at", sess)

        # Update category
        db.update_session_category(
            session_id, "Billing & Insurance", db_path=self.test_db_path
        )
        updated_sess = db.get_session(session_id, db_path=self.test_db_path)
        self.assertIsNotNone(updated_sess)
        self.assertEqual(updated_sess["last_category"], "Billing & Insurance")

    def test_03_nltk_preprocessing(self) -> None:
        """Verify NLTK cleaning, tokenization, and lemmatization."""
        preprocessor = nlp.TextPreprocessor()
        raw_text = "  What clothes should I wear for treatments?!  "
        cleaned = preprocessor.clean_text(raw_text)
        self.assertEqual(cleaned, "What clothes should I wear for treatments?!")

        tokens = preprocessor.tokenize_and_lemmatize(cleaned)
        # Verify 'treatments' is lemmatized to 'treatment'
        self.assertIn("treatment", tokens)
        self.assertIn("wear", tokens)

        keywords = preprocessor.extract_keywords(cleaned)
        self.assertIn("wear", keywords)
        self.assertIn("treatment", keywords)
        self.assertNotIn("what", keywords)  # Stopword removed

    def test_04_emergency_detection(self) -> None:
        """Verify emergency detection triggers urgent care message and logs correctly."""
        emergency_queries = [
            "I have crushing chest pain and shortness of breath",
            "I have numbness in my groin and loss of bowel control",
            "There is a bone sticking out after I fell",
            "I think I need to call 911 right now",
        ]

        session_id = "emergency_test_session"
        for q in emergency_queries:
            is_emerg, trigger = nlp.check_emergency(q)
            self.assertTrue(
                is_emerg, f"Query '{q}' should have been flagged as emergency"
            )
            self.assertIsNotNone(trigger)

            # Test full process_chat for emergency
            res = nlp.process_chat(
                session_id=session_id,
                message=q,
                matcher=self.matcher,
                db_path=self.test_db_path,
            )
            self.assertEqual(res["flag"], "emergency")
            self.assertEqual(res["confidence"], 1.0)
            self.assertIn("URGENT MEDICAL ADVISORY", res["answer"])
            self.assertIn("911", res["answer"])

        # Harmless physiotherapy query should NOT trigger emergency
        safe_query = "What should I wear to my appointment?"
        is_emerg_safe, _ = nlp.check_emergency(safe_query)
        self.assertFalse(is_emerg_safe)

    def test_05_semantic_matching_high_confidence(self) -> None:
        """Verify semantic matching accurately matches user messages against FAQs."""
        test_cases = [
            {
                "query": "What clothes should I wear for physical therapy?",
                "expected_faq_id": "prep_clothing",
                "expected_category": "First Visit & Preparation",
            },
            {
                "query": "Can you bill my insurance company directly?",
                "expected_faq_id": "ins_coverage",
                "expected_category": "Billing & Insurance",
            },
            {
                "query": "Do you treat herniated discs, pinched nerves, and sciatica?",
                "expected_faq_id": "treat_back_pain",
                "expected_category": "Conditions & Treatments",
            },
            {
                "query": "What is the fee if I have to reschedule or cancel?",
                "expected_faq_id": "apt_cancellation",
                "expected_category": "Appointments & Scheduling",
            },
            {
                "query": "Where is the clinic located and what are your opening hours?",
                "expected_faq_id": "gen_hours_location",
                "expected_category": "General Clinic Information",
            },
        ]

        for tc in test_cases:
            match = self.matcher.match(tc["query"])
            self.assertGreaterEqual(
                match["confidence"],
                nlp.CONFIDENCE_THRESHOLD,
                f"Query '{tc['query']}' confidence {match['confidence']} should be >= {nlp.CONFIDENCE_THRESHOLD}",
            )
            self.assertEqual(
                match["matched_faq_id"],
                tc["expected_faq_id"],
                f"Failed matching for query: '{tc['query']}'",
            )
            self.assertEqual(match["category"], tc["expected_category"])
            self.assertIn(match["flag"], ("matched", "boosted_match"))
            self.assertIsNone(match["suggested_questions"])

    def test_06_contextual_boosting_category(self) -> None:
        """Verify category boosting lifts candidate FAQs matching session last_category."""
        # Query that could be ambiguous or borderline
        query = "Do I need a prescription from my physician?"

        # Match without prior category
        match_neutral = self.matcher.match(
            query=query, session_last_category=None
        )

        # Match with relevant category set
        match_boosted = self.matcher.match(
            query=query, session_last_category="Billing & Insurance"
        )

        self.assertEqual(match_boosted["matched_faq_id"], "ins_doctor_referral")
        self.assertGreater(
            match_boosted["confidence"],
            match_neutral["confidence"],
            "Category boost should increase confidence",
        )
        self.assertAlmostEqual(
            match_boosted["confidence"],
            round(
                min(
                    1.0,
                    match_neutral["confidence"] + nlp.CATEGORY_BOOST_WEIGHT,
                ),
                4,
            ),
            places=2,
        )

    def test_07_contextual_boosting_history(self) -> None:
        """Verify recent message history boosts topical candidate FAQs."""
        session_id = "history_boost_session"
        db.get_or_create_session(session_id, db_path=self.test_db_path)

        # Simulate prior exchange about back pain
        last_3 = [
            {
                "role": "user",
                "text": "I injured my lower back and have severe sciatica nerve pain.",
            },
            {
                "role": "assistant",
                "text": "Lower back pain and sciatica are commonly treated with spinal mobilization and decompression.",
            },
            {
                "role": "user",
                "text": "How many sessions does it usually take to recover?",
            },
        ]

        # Match query with history
        match_with_history = self.matcher.match(
            query="Will treatment hurt?",
            last_3_messages=last_3,
            session_last_category="Conditions & Treatments",
        )

        self.assertIsNotNone(match_with_history["matched_faq_id"])
        self.assertGreaterEqual(
            match_with_history["confidence"], nlp.CONFIDENCE_THRESHOLD
        )

    def test_08_low_confidence_fallback(self) -> None:
        """Verify low-confidence queries (< 0.55) return fallback and suggestions."""
        gibberish_queries = [
            "Can I order a pepperoni pizza with extra cheese?",
            "What is the airspeed velocity of an unladen swallow?",
            "asdfkjlasdf 123456 qwerty zxcvbnm",
        ]

        for q in gibberish_queries:
            match = self.matcher.match(q)
            self.assertLess(
                match["confidence"],
                nlp.CONFIDENCE_THRESHOLD,
                f"Query '{q}' had unexpectedly high confidence: {match['confidence']}",
            )
            self.assertEqual(match["flag"], "low_confidence_fallback")
            self.assertIsNone(match["matched_faq_id"])
            self.assertIsNone(match["category"])
            self.assertIn("not completely certain", match["answer"])
            self.assertIsNotNone(match["suggested_questions"])
            self.assertGreater(len(match["suggested_questions"]), 0)

    def test_09_full_chat_process_and_logging(self) -> None:
        """Verify process_chat logs all user and assistant interactions."""
        session_id = "test_audit_session_09"

        # 1. First legitimate exchange
        res1 = nlp.process_chat(
            session_id=session_id,
            message="How do I book an initial physiotherapy session?",
            matcher=self.matcher,
            db_path=self.test_db_path,
        )
        self.assertEqual(res1["matched_faq_id"], "apt_first_visit")
        self.assertGreaterEqual(res1["confidence"], nlp.CONFIDENCE_THRESHOLD)

        # Check that session last_category was updated
        sess = db.get_session(session_id, db_path=self.test_db_path)
        self.assertEqual(sess["last_category"], "Appointments & Scheduling")

        # 2. Low confidence query
        res2 = nlp.process_chat(
            session_id=session_id,
            message="Do you sell airplanes or spaceship parts?",
            matcher=self.matcher,
            db_path=self.test_db_path,
        )
        self.assertEqual(res2["flag"], "low_confidence_fallback")
        self.assertLess(res2["confidence"], nlp.CONFIDENCE_THRESHOLD)

        # 3. Emergency query
        res3 = nlp.process_chat(
            session_id=session_id,
            message="I am having a heart attack and cannot breathe",
            matcher=self.matcher,
            db_path=self.test_db_path,
        )
        self.assertEqual(res3["flag"], "emergency")

        # Verify DB logs for session
        logs = db.get_logs(session_id=session_id, db_path=self.test_db_path)
        # There should be 6 logged messages (3 user + 3 assistant)
        self.assertEqual(len(logs), 6)

        # Verify roles and contents
        roles = [l["role"] for l in logs]
        self.assertEqual(roles.count("user"), 3)
        self.assertEqual(roles.count("assistant"), 3)

        # Verify emergency flags in logs
        flags = [l["flag"] for l in logs]
        self.assertIn("emergency_user", flags)
        self.assertIn("emergency_response", flags)
        self.assertIn("low_confidence_fallback", flags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
