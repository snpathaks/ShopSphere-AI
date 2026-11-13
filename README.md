# 🛍️ E-Commerce Product Recommendation Engine

An end-to-end recommendation system that personalises product discovery for an
e-commerce platform. The engine compares four approaches — a popularity baseline,
item-item collaborative filtering, content-based filtering, and a hybrid blend —
evaluated with Precision@K, Recall@K, NDCG@K, catalog coverage, and a cold-start
breakdown.

> **Algions AI Labs Private Limited** — AI & Machine Learning Internship Assessment
> (15-day individual project)

---

## Problem Definition

This system is a **decision-support tool** that selects the top-N products to
present to a user. The **prediction unit** is a *(user, item)* pair: for every
candidate product in the catalogue the model estimates the likelihood that the
user will meaningfully interact with (view, add-to-cart, or purchase) that item
in the next evaluation period. **Prediction time** is the moment a user opens
the homepage or a product page and the system must rank the catalogue in
real-time. The **target label** is a binary signal — whether the user actually
interacted with (bought) the item during the held-out test window (the final 30
days of the dataset, selected by a strict chronological split with no future
leakage into training). **Success is measured by four metrics**: NDCG@10
(ranking quality), Recall@10 (relevant-item coverage), Precision@10 (signal-to-
noise in the top slate), and catalog coverage (fraction of the item catalogue
that appears in at least one user's recommendation list). Because this is a
decision-support tool — not an autonomous system — every recommendation is
surfaced alongside its **reason** (e.g. "similar to items you purchased") so a
human can judge its relevance, and the system's limitations (cold-start gaps,
popularity bias, filter-bubble risk) are documented explicitly.

---

## Project Structure

```
E-commerceProductRecommendation/
│
├── data/
│   ├── generate_data.py          # Synthetic data generator (seed=42)
│   ├── raw/                      # Generated CSV files
│   │   ├── users.csv             # 500 users
│   │   ├── products.csv          # 200 products across 10 categories
│   │   ├── interactions.csv      # 15 000+ implicit/explicit events
│   │   └── data_summary.txt      # Dataset statistics
│   └── processed/                # Train/test splits & matrices
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, seeds, hyperparameters
│   ├── preprocessing.py          # Load, clean, split, feature build
│   ├── train.py                  # Full training + evaluation pipeline
│   ├── evaluation.py             # P@K, R@K, NDCG@K, coverage, cold-start
│   └── models/
│       ├── __init__.py
│       ├── base.py               # Abstract BaseRecommender
│       ├── popularity.py         # Popularity baseline
│       ├── collaborative.py      # Item-item collaborative filtering
│       ├── content_based.py      # TF-IDF content-based filtering
│       └── hybrid.py             # Weighted hybrid blend
│
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic request/response models
│   └── model_loader.py           # Singleton model loader
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Streamlit UI (5 pages)
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Model_Evaluation.ipynb # Model comparison & error analysis
│   └── figures/                  # Saved plots
│
├── models/                       # Saved .pkl model artefacts
│   └── .gitkeep
│
├── tests/                        # pytest test suite
│   └── __init__.py
│
├── reports/                      # Technical report & model card
│   └── .gitkeep
│
├── .env.example                  # Environment variable template
├── .gitignore
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # ← You are here
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip (or a virtual-env manager of your choice)

### Installation

```bash
git clone <repo-url>
cd E-commerceProductRecommendation
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 1. Generate Synthetic Data

```bash
python data/generate_data.py
```

Creates `data/raw/users.csv`, `products.csv`, and `interactions.csv` (seed=42,
fully deterministic).

### 2. Train Models

```bash
python src/train.py
```

Runs the full pipeline: load → clean → chronological split → fit four models →
evaluate → save `.pkl` artefacts to `models/` → write
`models/evaluation_results.json`.

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at **http://localhost:8000/docs**.

### 4. Start the UI

```bash
streamlit run app/streamlit_app.py
```

Opens a browser dashboard on **http://localhost:8501**.

### 5. Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## API Reference

| Method | Endpoint                        | Description                                   |
|--------|---------------------------------|-----------------------------------------------|
| GET    | `/`                             | Welcome message + link to docs                |
| GET    | `/health`                       | Health check — lists loaded models             |
| GET    | `/models`                       | Available models + descriptions               |
| GET    | `/recommend/{user_id}`          | Top-N recommendations (query: `n`, `model`, `exclude_seen`) |
| GET    | `/items/{item_id}/similar`      | Similar items (content-based)                 |
| GET    | `/users/{user_id}/history`      | User's interaction history                    |
| POST   | `/feedback`                     | Log user feedback `{user_id, item_id, rating}` |

**Example:**

```bash
curl "http://localhost:8000/recommend/U0001?n=5&model=hybrid"
```

---

## Models

| # | Model               | Approach                       | Cold-Start Handling             |
|---|---------------------|--------------------------------|---------------------------------|
| 1 | **Popularity**      | Weighted interaction count     | Works for all users (baseline)  |
| 2 | **Item-Item CF**    | Cosine similarity on user-item matrix | Falls back to popularity for new users |
| 3 | **Content-Based**   | TF-IDF on product text → cosine similarity | Category popularity for new users |
| 4 | **Hybrid**          | α·CF + β·Content + γ·Popularity | Increases γ for sparse users    |

---

## Evaluation Metrics

Evaluated on a chronological test set (last 30 days, no future leakage):

| Metric        | What It Measures                                    |
|---------------|-----------------------------------------------------|
| Precision@K   | Fraction of top-K recommendations that are relevant |
| Recall@K      | Fraction of relevant items captured in top-K        |
| NDCG@K        | Ranking quality (rewards relevant items at top)     |
| Coverage      | Fraction of catalogue appearing in any user's list  |

Results are printed by `python src/train.py` and saved to
`models/evaluation_results.json`.

---

## Reproducing Results

Every random operation uses `seed=42`. To reproduce from scratch:

```bash
# 1. Clean slate
rm -rf data/raw/*.csv data/processed/ models/*.pkl

# 2. Regenerate data (deterministic)
python data/generate_data.py

# 3. Train + evaluate (deterministic)
python src/train.py

# 4. Verify
pytest tests/ -v
```

The train/test split is chronological (last 30 calendar days → test). No
random sampling is involved in the split, so the same data yields the same
partition every time.

---

## Limitations & Responsible Use

| Concern              | Detail                                                       |
|----------------------|--------------------------------------------------------------|
| **Cold start**       | Users or items with < 5 interactions receive degraded recs; the hybrid model mitigates this by up-weighting the popularity signal. |
| **Popularity bias**  | The baseline (and to a lesser extent the hybrid) over-represents head items. Catalog coverage is reported to quantify this. |
| **Filter bubble**    | Content-based and CF models can reinforce narrow preferences. Diversity-aware re-ranking is a possible extension. |
| **Synthetic data**   | All data is generated — patterns may not reflect real-world behaviour. No personal or sensitive data is used. |
| **Human oversight**  | This is a decision-support tool; every recommendation includes a *reason* string. Final purchase decisions remain with the user. |

---

## Attribution

- **AI tools used:** Portions of this codebase were scaffolded with AI assistance
  (Antigravity / Claude). All code has been reviewed and is understood by the
  author.
- **Key libraries:** NumPy, pandas, scikit-learn, SciPy, FastAPI, Streamlit,
  matplotlib, seaborn, pytest.

---

## License

This project is submitted as part of the Algions AI Labs internship assessment.

