# Physiotherapy Clinic Chatbot

A semantic FAQ chatbot and API for physiotherapy clinics built with FastAPI, Sentence Transformers, NLTK, and SQLite.

## Features

- **Semantic FAQ Matching**: Precomputes embeddings with `sentence-transformers/all-MiniLM-L6-v2` across FAQ questions and variations for high-accuracy intent matching.
- **Urgent Care / Red Flag Detection**: Intercepts potential medical emergencies and directs users to immediate emergency care.
- **Contextual Boosting**: Uses recent session exchanges and category history to disambiguate short or follow-up queries.
- **Confidence Scoring & Fallback**: Handles out-of-scope queries below confidence threshold (0.55) with relevant suggested FAQs.
- **Interactive Web Interface**: Clean frontend for real-time consultation and FAQs exploration.
- **Full Conversation Logging**: Persists sessions, messages, confidence metrics, and flags in SQLite for auditing and analytics.

## Project Structure

```text
├── data/
│   ├── faqs.json          # Curated FAQ dataset with categories and question variations
├── static/
│   └── index.html         # Responsive web UI for the chatbot
├── db.py                  # SQLite database models, initialization, and logging functions
├── main.py                # FastAPI application endpoints (/chat, /logs, /health, /)
├── nlp.py                 # Text preprocessing, semantic matching, and context boosting
├── test_chatbot.py        # Comprehensive test suite for DB and NLP pipelines
├── requirements.txt       # Project dependencies
└── .gitignore             # Git ignore patterns
```

## Getting Started

### 1. Prerequisites

- Python 3.9+

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## API Endpoints

- `POST /chat` — Send query, get answer with confidence score and suggested questions.
- `GET /logs` — Retrieve conversation logs (filterable by `session_id`).
- `GET /health` — Check server status, model readiness, and FAQ embedding counts.
- `GET /` — Serve web UI.

## Running Tests

```bash
python -m unittest test_chatbot.py
```
