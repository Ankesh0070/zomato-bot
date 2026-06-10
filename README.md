# 🍽️ Zomato AI Restaurant Recommender

An AI-powered restaurant recommendation system that combines structured Zomato restaurant data with Large Language Model (LLM) intelligence to deliver personalized, explainable dining suggestions.

---

## 🎯 Features

- **Smart Filtering** — Narrows 50k+ restaurants by location, budget, cuisine, and rating
- **LLM-Powered Ranking** — Uses Gemini / OpenAI to rank and explain recommendations
- **Natural Language Explanations** — Each suggestion comes with a personalized reason
- **Interactive Web UI** — Streamlit-based interface for easy preference input
- **Fallback & Resilience** — Automatic retry, model fallback, and filter relaxation

---

## 🏗️ Architecture

```
User → Input Collector → Filter Pipeline → LLM Recommender → Presentation
                              ↕
                      Zomato Dataset (HF)
```

See [Docs/architecture.md](Docs/architecture.md) for the full system design.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- A **Gemini API key** (free) or **OpenAI API key**

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd "Zomato bot"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

### 3. Verify Setup

```bash
python src/config.py
```

You should see a configuration check with all green ✅ marks.

### 4. Run the App

```bash
streamlit run src/main.py
```

---

## 📁 Project Structure

```
Zomato bot/
├── Docs/                        # Documentation
│   ├── ProblemStatement.txt     # Original problem statement
│   ├── context.md               # Project context
│   ├── architecture.md          # System architecture
│   ├── implementation_plan.md   # Phase-wise implementation plan
│   └── edge-cases.md            # Edge case handling guide
├── src/                         # Source code
│   ├── main.py                  # Entry point (Streamlit app)
│   ├── config.py                # Configuration management
│   ├── data/                    # Data ingestion & preprocessing
│   ├── input/                   # User input handling
│   ├── integration/             # Filter & ranking pipeline
│   ├── recommendation/          # LLM prompt & response engine
│   └── presentation/            # UI rendering
├── tests/                       # Test suite
├── data/                        # Cached dataset (gitignored)
├── .env.example                 # Environment template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Problem Statement](Docs/ProblemStatement.txt) | Original project requirements |
| [Context](Docs/context.md) | Project context and objectives |
| [Architecture](Docs/architecture.md) | Full system design and data flow |
| [Implementation Plan](Docs/implementation_plan.md) | 6-phase development roadmap |
| [Edge Cases](Docs/edge-cases.md) | 55 edge cases with handling strategies |

---

## 🔧 Configuration

All settings are managed via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key (primary) |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback) |
| `LLM_PROVIDER` | `gemini` | Which LLM provider to use |
| `LLM_MODEL` | `gemini-2.0-flash` | Specific model name |
| `LLM_TEMPERATURE` | `0.3` | LLM creativity (0=deterministic, 1=creative) |
| `TOP_K_CANDIDATES` | `20` | Max restaurants sent to LLM |
| `MAX_RECOMMENDATIONS` | `5` | Number of final recommendations |

---

## 📄 License

This project is for educational purposes.

---

*Built with ❤️ using Python, Streamlit, and Gemini AI*

## ✅ Deployment

A Dockerfile is included for containerized deployment. Example:

```bash
docker build -t zomato-recommender .
docker run -p 8501:8501 zomato-recommender
```

## 🧪 Running Tests

Install dependencies and run the test suite:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## 🧾 Logging

The application emits structured JSON logs by default for easier ingestion into logging systems. Set LOG_FORMAT=text to use the human-readable formatter.

