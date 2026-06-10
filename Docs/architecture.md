# Architecture — AI-Powered Restaurant Recommendation System

> **Reference:** [context.md](file:///c:/Projects/Zomato%20bot/Docs/context.md)

---

## 1. High-Level Overview

The system is a conversational, AI-powered restaurant recommendation service inspired by Zomato. It combines **structured restaurant data** (from Hugging Face) with a **Large Language Model (LLM)** to deliver personalized, explainable restaurant suggestions.

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│  User      │────▶│  Input       │────▶│  Data Filter   │────▶│  LLM-based   │────▶│  Output /    │
│  Interface │     │  Collector   │     │  & Retrieval   │     │  Recommender │     │  Presenter   │
└────────────┘     └──────────────┘     └────────────────┘     └──────────────┘     └──────────────┘
                                               │
                                        ┌──────┴──────┐
                                        │  Zomato     │
                                        │  Dataset    │
                                        │  (HF)       │
                                        └─────────────┘
```

---

## 2. System Architecture Diagram

```
                          ┌─────────────────────────────────────────────┐
                          │              PRESENTATION LAYER             │
                          │  ┌─────────────────────────────────────┐    │
                          │  │  Web UI / CLI / Chat Interface      │    │
                          │  │  • Preference form / chat input     │    │
                          │  │  • Recommendation cards display     │    │
                          │  └──────────────────┬──────────────────┘    │
                          └─────────────────────┼──────────────────────┘
                                                │  HTTP / WebSocket
                          ┌─────────────────────┼──────────────────────┐
                          │              APPLICATION LAYER              │
                          │                     │                       │
                          │  ┌──────────────────▼──────────────────┐   │
                          │  │         API Gateway / Router         │   │
                          │  └──┬───────────┬───────────┬──────────┘   │
                          │     │           │           │               │
                          │  ┌──▼────┐  ┌──▼────┐  ┌──▼──────────┐   │
                          │  │Input  │  │Filter │  │Recommendation│   │
                          │  │Parser │  │Engine │  │Engine (LLM)  │   │
                          │  └───────┘  └───┬───┘  └──────┬───────┘   │
                          │                 │             │            │
                          └─────────────────┼─────────────┼────────────┘
                                            │             │
                          ┌─────────────────┼─────────────┼────────────┐
                          │              DATA LAYER        │            │
                          │  ┌──────────────▼───┐   ┌─────▼────────┐  │
                          │  │  Zomato Dataset   │   │  LLM API     │  │
                          │  │  (Hugging Face)   │   │  (Groq)      │  │
                          │  │  Preprocessed &   │   │              │  │
                          │  │  Cached Locally   │   │              │  │
                          │  └──────────────────┘   └──────────────┘  │
                          └────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Data Ingestion Module

| Aspect         | Detail                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------ |
| **Source**      | [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) (Hugging Face) |
| **Format**     | CSV / Parquet via `datasets` library                                                       |
| **Fields**     | `restaurant_name`, `location`, `cuisine`, `cost_for_two`, `aggregate_rating`, `votes`, `online_order`, `book_table`, etc. |
| **Processing** | Clean nulls, normalize text (lowercase cities/cuisines), bucket costs into low/medium/high, cast ratings to float |
| **Storage**    | In-memory Pandas DataFrame (or cached Parquet for faster reload)                           |

**Responsibilities:**
- Download and cache the Hugging Face dataset on first run
- Parse and clean raw data (handle missing values, duplicates, encoding)
- Normalize key fields for consistent filtering (e.g., `"new delhi"` → `"Delhi"`)
- Create derived columns: `budget_tier` (low / medium / high), `rating_bucket`
- Expose a clean DataFrame or data-access API to downstream modules

**Suggested file structure:**
```
src/
  data/
    ingestion.py        # Download & cache dataset from HF
    preprocessing.py    # Clean, normalize, derive columns
    schema.py           # Data models / column definitions
```

---

### 3.2 User Input Module

| Aspect         | Detail                                                      |
| -------------- | ----------------------------------------------------------- |
| **Interface**  | Web form (primary) or CLI (secondary)                       |
| **Fields**     | Location, Budget, Cuisine, Min Rating, Additional Prefs     |
| **Validation** | Enum checks for budget, range check for rating (0–5), fuzzy match for location/cuisine |

**Input Schema:**
```json
{
  "location": "Delhi",
  "budget": "medium",
  "cuisine": "Italian",
  "min_rating": 4.0,
  "preferences": ["family-friendly", "quick service"]
}
```

**Responsibilities:**
- Collect and validate user input
- Normalize input values to match dataset conventions
- Provide autocomplete suggestions for location and cuisine (drawn from dataset)
- Return a structured `UserPreferences` object to the Integration Layer

**Suggested file structure:**
```
src/
  input/
    collector.py        # Collect preferences (web form / CLI)
    validator.py        # Validate & normalize inputs
    models.py           # UserPreferences dataclass
```

---

### 3.3 Integration / Filter Layer

This is the bridge between structured data and the LLM. It narrows down the full dataset to a **candidate set** that is small enough to fit within an LLM context window while remaining highly relevant.

**Filtering Pipeline:**

```
Full Dataset (~50k restaurants)
        │
        ▼
┌───────────────────┐
│ Location Filter   │  → Match city/area
└───────┬───────────┘
        ▼
┌───────────────────┐
│ Budget Filter     │  → cost_for_two within budget tier thresholds
└───────┬───────────┘
        ▼
┌───────────────────┐
│ Cuisine Filter    │  → Exact or partial cuisine match
└───────┬───────────┘
        ▼
┌───────────────────┐
│ Rating Filter     │  → aggregate_rating ≥ min_rating
└───────┬───────────┘
        ▼
Candidate Set (10–50 restaurants)
        │
        ▼
┌───────────────────┐
│ Top-K Selection   │  → Sort by rating×votes, take top 15–20
└───────┬───────────┘
        ▼
Shortlist → formatted for LLM prompt
```

**Responsibilities:**
- Apply cascading filters based on `UserPreferences`
- Rank candidates using a weighted score (rating × log(votes))
- Format the shortlist into a structured text block for the LLM prompt
- Handle edge cases: no results (relax filters), too many results (tighten or sample)

**Suggested file structure:**
```
src/
  integration/
    filter.py           # Cascading filter pipeline
    ranker.py           # Weighted scoring & top-K selection
    formatter.py        # Convert shortlist → LLM prompt context
```

---

### 3.4 Recommendation Engine (LLM)

The core intelligence layer. It receives the filtered restaurant shortlist and user preferences, then produces **ranked, explained recommendations**.

**Prompt Design Strategy:**

```
┌─────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                               │
│ You are an expert restaurant advisor. Given the user's      │
│ preferences and a list of matching restaurants, rank the     │
│ top 5 and explain why each is a great fit.                  │
├─────────────────────────────────────────────────────────────┤
│ USER CONTEXT                                                │
│ Preferences: {location, budget, cuisine, min_rating, prefs} │
├─────────────────────────────────────────────────────────────┤
│ RESTAURANT DATA                                             │
│ 1. Name: XYZ | Cuisine: Italian | Rating: 4.5 | Cost: ₹800 │
│ 2. Name: ABC | Cuisine: Italian | Rating: 4.3 | Cost: ₹600 │
│ ... (up to 15–20 candidates)                                │
├─────────────────────────────────────────────────────────────┤
│ INSTRUCTIONS                                                │
│ Return a JSON array of top 5 with: rank, name, cuisine,     │
│ rating, cost, explanation.                                  │
└─────────────────────────────────────────────────────────────┘
```

**LLM Provider:** Groq (Phase 4 uses Groq, not OpenAI)

| Provider   | Model                      | Pros                                              | Cons                 |
| ---------- | -------------------------- | ------------------------------------------------- | -------------------- |
| **Groq**   | `llama-3.3-70b-versatile`  | Extremely fast inference, free tier, high quality  | Rate limits on free  |

**Output Schema (expected from LLM):**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "restaurant_name": "La Piazza",
      "cuisine": "Italian",
      "rating": 4.5,
      "estimated_cost": "₹800 for two",
      "explanation": "Perfect match for your Italian craving in Delhi. Highly rated with a moderate price point, known for its family-friendly ambiance."
    }
  ],
  "summary": "Based on your preference for Italian cuisine in Delhi with a medium budget, here are the top picks..."
}
```

**Responsibilities:**
- Construct the multi-part prompt (system + context + data + instructions)
- Call the LLM API with appropriate parameters (temperature ~0.3 for consistency)
- Parse and validate the structured JSON response
- Handle retries, fallback models, and malformed responses
- Optionally cache responses for identical queries

**Suggested file structure:**
```
src/
  recommendation/
    prompt_builder.py   # Construct system + user + data prompt
    llm_client.py       # LLM API calls (Groq)
    parser.py           # Parse & validate LLM JSON response
    engine.py           # Orchestrate: build prompt → call LLM → parse
```

---

### 3.5 Output / Presentation Layer

Displays the final recommendations in a clear, user-friendly format.

**Display Format (per recommendation card):**

```
┌─────────────────────────────────────────────┐
│  🥇 #1  La Piazza                           │
│  ─────────────────────────────────────────  │
│  🍕 Cuisine:  Italian                       │
│  ⭐ Rating:   4.5 / 5.0                     │
│  💰 Cost:     ₹800 for two (Medium)         │
│  ─────────────────────────────────────────  │
│  🤖 Why this pick:                          │
│  "Perfect match for your Italian craving    │
│   in Delhi. Highly rated with a moderate    │
│   price point, known for family-friendly    │
│   ambiance."                                │
└─────────────────────────────────────────────┘
```

**Responsibilities:**
- Render recommendation cards (web UI or terminal)
- Display a summary paragraph at the top
- Allow the user to refine preferences and re-query
- Provide "More details" or "Similar restaurants" actions

**Suggested file structure:**
```
src/
  presentation/
    renderer.py         # Format output for display
    templates/          # HTML/Jinja templates (if web)
    cli_display.py      # Rich terminal output (if CLI)
```

---

## 4. Technology Stack

| Layer              | Technology                                              |
| ------------------ | ------------------------------------------------------- |
| **Language**        | Python 3.11+                                           |
| **Web Framework**   | Streamlit (MVP) or FastAPI + React (production)        |
| **Data Processing** | Pandas, NumPy                                          |
| **Dataset Source**  | Hugging Face `datasets` library                        |
| **LLM Integration** | Groq Python SDK (`groq`)                               |
| **LLM Provider**    | Groq (`llama-3.3-70b-versatile`)                       |
| **Caching**         | `functools.lru_cache` / Redis (optional)               |
| **Config**          | `.env` file via `python-dotenv`                        |
| **Testing**         | pytest, pytest-asyncio                                 |
| **Linting**         | ruff, mypy                                             |

---

## 5. Project Directory Structure

```
Zomato bot/
├── Docs/
│   ├── ProblemStatement.txt
│   ├── context.md
│   └── architecture.md          ← this file
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point / app runner
│   ├── config.py                # Environment & config management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py         # Download & cache HF dataset
│   │   ├── preprocessing.py     # Clean & normalize data
│   │   └── schema.py            # Column definitions & data models
│   ├── input/
│   │   ├── __init__.py
│   │   ├── collector.py         # Gather user preferences
│   │   ├── validator.py         # Validate & normalize inputs
│   │   └── models.py            # UserPreferences dataclass
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── filter.py            # Cascading filter pipeline
│   │   ├── ranker.py            # Weighted scoring & top-K
│   │   └── formatter.py         # Format shortlist for LLM
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py    # Construct multi-part prompt
│   │   ├── llm_client.py        # Abstract LLM API calls
│   │   ├── parser.py            # Parse & validate LLM output
│   │   └── engine.py            # Orchestration logic
│   └── presentation/
│       ├── __init__.py
│       ├── renderer.py          # Format recommendations
│       ├── cli_display.py       # Terminal output
│       └── templates/           # Web UI templates
├── tests/
│   ├── test_ingestion.py
│   ├── test_filter.py
│   ├── test_prompt.py
│   └── test_engine.py
├── data/                        # Cached dataset files
├── .env                         # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Data Flow (End-to-End)

```
┌──────┐  preferences   ┌──────────┐  validated   ┌──────────┐
│ User │───────────────▶│  Input   │────────────▶│  Filter  │
│      │                │  Module  │              │  Engine  │
└──────┘                └──────────┘              └────┬─────┘
                                                       │
                                              filtered │ candidates
                                                       │
                         ┌─────────────────────────────▼──────┐
                         │        Prompt Builder               │
                         │  • System instructions              │
                         │  • User preferences                 │
                         │  • Candidate restaurant data        │
                         └──────────────────┬──────────────────┘
                                            │ prompt
                                            ▼
                         ┌──────────────────────────────────────┐
                         │           LLM API Call               │
                         │  (Groq — llama-3.3-70b-versatile)   │
                         └──────────────────┬───────────────────┘
                                            │ JSON response
                                            ▼
                         ┌──────────────────────────────────────┐
                         │         Response Parser              │
                         │  • Validate JSON schema              │
                         │  • Extract recommendations           │
                         └──────────────────┬───────────────────┘
                                            │ structured results
                                            ▼
                         ┌──────────────────────────────────────┐
                         │        Presentation Layer            │
                         │  • Recommendation cards              │
                         │  • Summary paragraph                 │
                         │  • Refine / re-query option          │
                         └──────────────────────────────────────┘
```

---

## 7. API Contract (Internal)

### `POST /recommend`

**Request:**
```json
{
  "location": "Delhi",
  "budget": "medium",
  "cuisine": "Italian",
  "min_rating": 4.0,
  "preferences": ["family-friendly"]
}
```

**Response:**
```json
{
  "status": "success",
  "query": { ... },
  "candidates_found": 23,
  "recommendations": [
    {
      "rank": 1,
      "restaurant_name": "La Piazza",
      "cuisine": "Italian",
      "rating": 4.5,
      "estimated_cost": "₹800 for two",
      "explanation": "..."
    }
  ],
  "summary": "...",
  "model_used": "llama-3.3-70b-versatile",
  "response_time_ms": 1200
}
```

---

## 8. Error Handling & Edge Cases

| Scenario                        | Handling Strategy                                              |
| ------------------------------- | -------------------------------------------------------------- |
| No restaurants match filters    | Relax filters progressively (drop cuisine → drop budget → drop location) and inform user |
| LLM API timeout / rate limit    | Retry with exponential backoff (3 attempts), then rule-based fallback |
| LLM returns malformed JSON      | Regex-based extraction, then re-prompt with stricter instructions |
| Dataset download failure        | Use locally cached copy; if none, show clear error message      |
| Ambiguous user input            | Suggest closest matches (fuzzy matching with `fuzzywuzzy`)      |
| Very broad query (1000+ results)| Apply stricter ranking, take top 20 before sending to LLM      |

---

## 9. Future Enhancements

| Enhancement                  | Description                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| **Conversational Memory**    | Multi-turn chat: "Show me something cheaper" remembers context    |
| **User Profiles**            | Store past preferences and build taste profiles                   |
| **Review Summarization**     | Fetch and summarize user reviews per restaurant                   |
| **Map Integration**          | Show restaurant locations on an interactive map                   |
| **Real-time Data**           | Integrate Zomato/Swiggy live APIs for current menus and pricing   |
| **Collaborative Filtering**  | "Users who liked X also liked Y" recommendations                  |
| **Voice Input**              | Accept voice queries via speech-to-text                           |
| **Multi-language Support**   | Recommendations in Hindi, Tamil, etc.                             |

---

## 10. Security & Configuration

- **API keys** stored in `.env`, never committed to version control
- **`.gitignore`** includes: `.env`, `data/`, `__pycache__/`, `*.pyc`
- **Rate limiting** on the recommendation endpoint (if deployed as API)
- **Input sanitization** to prevent prompt injection attacks
- **Logging** via Python `logging` module — structured JSON logs in production

---

*Last updated: 2026-06-04*
