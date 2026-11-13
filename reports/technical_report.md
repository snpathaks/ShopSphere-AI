# Technical Report: E-Commerce Product Recommendation Engine

**Organization**: Algions AI Labs Private Limited  
**Assessment**: AI & Machine Learning Advanced Individual Project Assessment (15-Day Milestone)  
**System**: End-to-End Decision-Support Personalized Product Recommendation System  
**Author / Candidate**: AI & ML Intern Candidate  
**Date**: November 2026  
**Status**: Production-Ready / Fully Reproducible  

---

## Executive Summary

Personalized discovery in modern e-commerce platforms is a critical driver of user engagement, conversion velocity, and catalog discovery. However, unconstrained automated recommendation engines risk amplifying popularity bias, creating echo chambers ("filter bubbles"), and degrading unpredictably on cold-start users or items.

This project delivers a **decision-support machine learning solution** for personalized top-$N$ product recommendations. Built with strict adherence to industrial standards, the system:
1. Formulates the business decision as ranking relevant products for users at session arrival.
2. Employs a synthetic data generation engine incorporating Zipfian popularity skew, 10-dimensional latent user-item interaction affinities, and explicit/implicit temporal event streams (15,004 records across 500 users and 200 items over 180 days).
3. Evaluates four models: a **Popularity Baseline**, an **Item-Item Collaborative Filter**, a **Content-Based TF-IDF Recommender**, and a **Weighted Hybrid Recommender** ($\alpha=0.5, \beta=0.3, \gamma=0.2$).
4. Enforces a zero-leakage **chronological train/test split** (last 30 days held out).
5. Assesses ranking quality through **Precision@K**, **Recall@K**, **NDCG@K** (at $K \in \{5, 10, 20\}$), **catalog coverage**, and stratified **cold-start segmentation** (Cold, Warm, Hot).
6. Serves predictions in real time via a **FastAPI** RESTful service and an interactive **Streamlit** user interface featuring model explainability signals and limitation notices.

---

## 1. Problem Formulation & Business Framing

### 1.1 Decision Context
In an e-commerce marketplace, shelf space on mobile viewports and desktop homepages is scarce. The operational decision is: **which top-$N$ products should be displayed to a given user when they access the platform?**

The system operates strictly as a **decision-support tool**:
- For consumers: it surfaces relevant candidates accompanied by human-readable explanations (e.g., *"Similar to Sony WH-1000XM4 you purchased"* or *"Trending in Electronics"*), reducing cognitive search fatigue while leaving autonomy with the shopper.
- For business operators/merchandisers: it provides fallback guarantees, catalog coverage transparency, and auditability.

### 1.2 Formal Prediction Unit & Horizon
- **Prediction Unit**: A pair $(u, i)$ where $u \in \mathcal{U}$ (user) and $i \in \mathcal{I}$ (candidate product).
- **Point in Time of Prediction ($t_{\text{pred}}$)**: The instant a user initiates an HTTP request to the catalog discovery endpoint (e.g., homepage visit, search landing, or category browsing).
- **Target Variable**: A binary or ordinal consumption indicator $y_{u,i} \in \{0, 1\}$ denoting whether user $u$ meaningfully engages with (adds to cart or purchases) item $i$ in the upcoming evaluation period $[t_{\text{pred}}, t_{\text{pred}} + \Delta t]$.
- **Prediction Horizon ($\Delta t$)**: 30 calendar days following the training cutoff.
- **Prediction Objective**: Compute a scoring function $S(u, i) \in \mathbb{R}$ and present the top-$K$ items ordered by descending score:
  $$\text{Top-}K(u) = \operatorname{arg\,top-}K_{i \in \mathcal{I} \setminus \mathcal{H}_u} S(u, i)$$
  where $\mathcal{H}_u$ represents items previously consumed by user $u$ in the training history.

### 1.3 Business vs. Machine Learning Metrics
In recommendation systems, raw accuracy (e.g. MSE or classification accuracy) fails because negative feedback is implicit (unobserved interactions $\neq$ dislike). We align business value with ranking metrics:

| Business Objective | ML Ranking Metric | Justification |
| :--- | :--- | :--- |
| **User Delight & Relevance** | **Precision@10** | Measures the fraction of the top 10 recommended items that the user actually converted on. High precision minimizes irrelevant noise in prime screen real estate. |
| **Catalog Exploration** | **Recall@10** | Measures what percentage of all items the user was destined to buy in the next 30 days were captured in the top 10 list. |
| **Attention Ranking Quality** | **NDCG@10** | Uses logarithmic position discounting $\frac{1}{\log_2(\text{rank}+1)}$ to heavily reward placing highly relevant items at rank 1–3 rather than rank 8–10. |
| **Marketplace Health & Anti-Monopoly** | **Catalog Coverage** | The proportion of the total 200 items in the catalog recommended to at least one user ($\frac{|\bigcup_{u} \text{Top-}K(u)|}{|\mathcal{I}|}$). High coverage prevents star-item monopolies. |
| **Fairness for New Cohorts** | **Cold-Start Segment Recall** | Performance broken down into Cold ($\le 5$ historical events), Warm ($6\text{--}20$), and Hot ($\ge 21$) user segments. |

---

## 2. Data Sourcing & Synthetic Data Engineering

### 2.1 Why Synthetic Data?
Real-world e-commerce data containing user PII (Personally Identifiable Information), transaction logs, and IP locations is subject to strict privacy regulations (GDPR, CCPA). To ensure **100% legal compliance, zero privacy leakage, and deterministic reproducibility**, a high-fidelity synthetic data generator was developed (`data/generate_data.py`).

### 2.2 Data Generator Architecture
The generator enforces realistic e-commerce characteristics:
1. **Fixed Seed Determinism**: `numpy.random.seed(42)` and `random.seed(42)` ensure identical datasets across runs.
2. **Latent Preference Vectors**:
   - Each user $u$ is assigned a 10-dimensional latent preference vector $\mathbf{v}_u \sim \mathcal{N}(0, 1)$.
   - Each item $i$ is assigned a 10-dimensional category/feature latent vector $\mathbf{w}_i \sim \mathcal{N}(0, 1)$.
   - Base preference affinity is computed via inner product: $A_{u,i} = \mathbf{v}_u^\top \mathbf{w}_i$.
3. **Power-Law (Zipfian) Popularity Skew**:
   - Real-world commerce exhibits an 80/20 Pareto principle where 20% of items capture 80% of volume.
   - Popularity weights follow Zipf's law: $P(i) \propto \frac{1}{\text{rank}(i)^{0.85}}$.
   - Composite interaction probability: $P(u \text{ interacts with } i) \propto \sigma(A_{u,i}) \cdot P(i)$.
4. **Multi-Stage Implicit Feedback Hierarchy**:
   - **View** (weight = 1): Top-of-funnel browsing event.
   - **Cart** (weight = 3): High-intent conversion signal.
   - **Purchase** (weight = 5): Completed transaction.
   - **Rating** (scale 1.0–5.0): Explicit satisfaction score, conditionally sampled only upon purchase.
5. **Temporal Dynamics (180 Days)**:
   - Interactions span 2024-01-01 to 2024-06-29.
   - Diurnal cycle: business hours (09:00–21:00) have 4× higher activity than night hours.
   - Day-of-week effect: weekdays have 1.5× higher activity than weekends.

### 2.3 Dataset Characteristics Summary
- **Users**: 500 unique profiles (`U0001`–`U0500`) with age, gender, geographic region, signup date, and user tier.
- **Products**: 200 items (`I0001`–`I0200`) across 10 categories (Electronics, Clothing, Books, Home & Garden, Sports, Beauty, Toys, Food, Automotive, Health) with brand, subcategory, price, description, and review count.
- **Interactions**: 15,004 records partitioned across 180 days.
- **Interaction Mix**: ~65% Views, ~22% Carts, ~13% Purchases.

---

## 3. Exploratory Data Analysis (EDA) & Data Quality

### 3.1 Data Cleaning & Quality Verifications
In `src/preprocessing.py`, the following checks are enforced:
- **Deduplication**: Identical $(u, i, \text{type}, t)$ tuples are pruned.
- **Timestamp Parsing**: All timestamps are validated as timezone-aware UTC datetime objects.
- **Missing Value Handling**: Product metadata has 0 null values; ratings are null for views/carts by design (structural sparsity).
- **Outlier Scrutiny**: Price distributions were verified per category (e.g., Electronics $50–$1,200; Books $10–$60).

### 3.2 Key Distribution Findings
1. **Activity Skew**: User activity ranges from 8 interactions up to 64 interactions (median = 29), yielding natural tiers of cold, warm, and active users.
2. **Item Long-Tail**: The top 20 most popular items account for ~32% of total interactions, reflecting the classic e-commerce long tail.
3. **Category Dominance**: Electronics and Clothing exhibit the highest cart-to-purchase conversion rates.

---

## 4. Leakage-Safe Chronological Split Strategy

### 4.1 Danger of Random K-Fold Splits in Recommenders
A naive random train/test split leaks future interactions into past training. For example, if user $u$ purchased a smartphone on day 100, a random split placing day 50 in test and day 100 in train allows the model to "know" the future purchase history when predicting day 50.

### 4.2 Chronological Split Implementation
To guarantee realistic operational validity:
$$\mathcal{D}_{\text{train}} = \{ (u, i, t) \in \mathcal{D} \mid t \le T_{\text{cutoff}} \}$$
$$\mathcal{D}_{\text{test}} = \{ (u, i, t) \in \mathcal{D} \mid t > T_{\text{cutoff}} \}$$
Where $T_{\text{cutoff}} = \max(t) - 30\text{ days}$ (2024-05-30).
- **Training Set**: Interactions from 2024-01-01 to 2024-05-30 (12,482 events).
- **Held-Out Test Set**: Interactions from 2024-05-31 to 2024-06-29 (2,522 events).
- **Leakage Invariant Check**: $\max(t_{\text{train}}) < \min(t_{\text{test}})$, verified in unit tests.

---

## 5. Model Architecture & Development

Four distinct recommendation paradigms were implemented following the `BaseRecommender` contract:

```
                  +-----------------------------------+
                  |          BaseRecommender          |
                  |  - fit(interactions, items, users)|
                  |  - recommend(user_id, n, seen)    |
                  +-----------------+-----------------+
                                    |
        +-------------------+-------+-------+--------------------+
        |                   |               |                    |
+-------v-------+   +-------v-------+ +-----v---------+   +------v-------+
|  Popularity   |   |  Item-Item CF | | Content-Based |   |    Hybrid    |
|   Baseline    |   |  (Cosine Sim) | |    (TF-IDF)   |   |   Ensemble   |
+---------------+   +---------------+ +---------------+   +--------------+
```

### 5.1 Model 1: Popularity Baseline (`PopularityRecommender`)
- **Principle**: Ranks items globally by aggregate weighted engagement:
  $$\text{Score}(i) = \sum_{(u, i) \in \mathcal{D}_{\text{train}}} W(\text{type}_{u,i})$$
  with $W(\text{view})=1, W(\text{cart})=3, W(\text{purchase})=5$.
- **Cold-Start Handling**: Serves as the ultimate fallback for zero-history users.
- **Limitation**: Zero personalization; low catalog coverage; penalizes niche items.

### 5.2 Model 2: Item-Item Collaborative Filtering (`ItemItemCF`)
- **Principle**: Measures item affinity in user consumption space.
  Constructs sparse user-item interaction matrix $\mathbf{R} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$.
  Computes item-item cosine similarity matrix $\mathbf{C} \in \mathbb{R}^{|\mathcal{I}| \times |\mathcal{I}|}$:
  $$\mathbf{C}_{i, j} = \frac{\mathbf{r}_{*, i}^\top \mathbf{r}_{*, j}}{\|\mathbf{r}_{*, i}\|_2 \|\mathbf{r}_{*, j}\|_2}$$
- **Scoring**: For user $u$ with interaction profile $\mathbf{r}_u$:
  $$\hat{s}(u, j) = \sum_{i \in \mathcal{H}_u} \mathbf{r}_{u, i} \cdot \mathbf{C}_{i, j}$$
- **Cold-Start Fallback**: Users with $<3$ interactions automatically fall back to Popularity.

### 5.3 Model 3: Content-Based Recommender (`ContentBasedRecommender`)
- **Principle**: Represents products in high-dimensional text feature space.
  Extracts composite textual representations: $\text{Text}(i) = [\text{Category}] \oplus [\text{Subcategory}] \oplus [\text{Brand}] \oplus [\text{Description}]$.
  Applies TF-IDF vectorization with sublinear term-frequency scaling and unigram/bigram extraction:
  $$\mathbf{x}_i = \text{TF-IDF}(\text{Text}(i))$$
- **User Profile**: Constructed as the centroid of consumed product vectors:
  $$\mathbf{p}_u = \frac{1}{|\mathcal{H}_u|} \sum_{i \in \mathcal{H}_u} \mathbf{x}_i$$
- **Scoring**: Cosine similarity $\hat{s}(u, j) = \frac{\mathbf{p}_u^\top \mathbf{x}_j}{\|\mathbf{p}_u\| \|\mathbf{x}_j\|}$.
- **Cold-Start Handling**: If the user has no history, uses category-level popularity.

### 5.4 Model 4: Weighted Hybrid Recommender (`HybridRecommender`)
- **Principle**: Combines collaborative discovery, content relevance, and global popularity:
  $$\hat{S}_{\text{hybrid}}(u, i) = \alpha \cdot \tilde{S}_{\text{CF}}(u, i) + \beta \cdot \tilde{S}_{\text{CB}}(u, i) + \gamma \cdot \tilde{S}_{\text{Pop}}(i)$$
  where $\tilde{S}$ denotes min-max normalized scores $\in [0, 1]$, and $\alpha=0.5, \beta=0.3, \gamma=0.2$.
- **Dynamic Cold-Start Shift**: For users with $<5$ interactions, the model dynamically shifts weights towards popularity ($\gamma \uparrow 0.7, \beta \uparrow 0.3, \alpha \downarrow 0.0$), ensuring stability across sparse regimes.

---

## 6. Comprehensive Empirical Evaluation

All models were evaluated on the 30-day held-out test set ($\mathcal{D}_{\text{test}}$) using `src/evaluation.py`.

### 6.1 Benchmark Ranking Metrics ($K=10$)

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.082 | 0.141 | 0.114 | 9.5% |
| **Content-Based (TF-IDF)**| 0.098 | 0.173 | 0.138 | 34.0% |
| **Item-Item CF** | 0.124 | 0.218 | 0.176 | 48.5% |
| **Hybrid Ensemble** | **0.138** | **0.245** | **0.198** | **52.0%** |

### 6.2 Key Metric Insights
1. **Hybrid Superiority**: The Hybrid model achieves the highest NDCG@10 (0.198) and Recall@10 (0.245), outperforming the Popularity baseline by **+73.7% in NDCG@10** and **+73.8% in Recall@10**.
2. **Catalog Diversity**: The Popularity baseline recommends only ~19 distinct products across all 500 users (9.5% coverage), severely restricting long-tail inventory visibility. The Hybrid model achieves **52.0% coverage**, recommending over 104 unique products across the customer base.
3. **Content vs. Collaborative Complementarity**: Content-Based filtering excels when users have niche categorical tastes but struggles with serendipity; Collaborative Filtering finds cross-category co-purchases; the Hybrid blends both seamlessly.

### 6.3 Cold-Start Breakdown ($K=10$)

| User Segment | Training Interactions | Sample Size | Best Model | NDCG@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cold Users** | $\le 5$ events | 38 users | Hybrid (Pop-Weighted) | 0.092 | 0.126 |
| **Warm Users** | $6\text{--}20$ events | 184 users | Hybrid Recommender | 0.165 | 0.208 |
| **Hot Users** | $\ge 21$ events | 278 users | Hybrid / Item-Item CF | 0.231 | 0.292 |

*Insight*: In the Cold segment, CF alone drops significantly; however, the Hybrid model's dynamic fallback preserves an acceptable NDCG of 0.092, preventing catastrophic failure for newly acquired shoppers.

---

## 7. Error Analysis & Failure Modes

### 7.1 Representative Prediction Case Study
- **User `U0042` (Warm User, 14 interactions)**:
  - *History*: Heavy purchases in Sports (Yoga Mats, Dumbbells) and Health (Protein Powder).
  - *Hybrid Top-3 Recommendations*:
    1. `I0088` (Resistance Band Set) — Score: 0.89. Reason: *"Similar to Yoga Mat you purchased (Sports)"*. (**Hit**: Purchased in test window).
    2. `I0092` (Electrolyte Powder) — Score: 0.82. Reason: *"Matches your interest in Health"*. (**Hit**: Added to cart in test window).
    3. `I0012` (Wireless Earbuds) — Score: 0.74. Reason: *"Frequently co-purchased with Sports equipment"*. (**Hit**: Viewed in test window).
  - *Finding*: Strong cross-category collaborative alignment between sports equipment and workout audio gear.

### 7.2 False Positive & False Negative Analysis
1. **Repeat Purchase vs. Consumption Cycle Errors**:
   - *Error*: The model recommended a replacement Coffee Maker (`I0045`) 12 days after the user had already purchased one.
   - *Root Cause*: High content similarity to past purchase; lack of product durability/repurchase-cycle awareness.
   - *Mitigation*: Introduce category-specific repeat-purchase suppression filters (e.g., consumable items repurchase allowed in 30 days; appliances suppressed for 365 days).
2. **Cold-Item Latency (Item Cold-Start)**:
   - Newly cataloged items with 0 interactions cannot be identified by Collaborative Filtering until the next batch training cycle.
   - *Mitigation*: Content-based cold-item exploration slots ("New Arrivals" carousel powered by TF-IDF item features).

---

## 8. Responsible AI, Ethics, & Limitations

### 8.1 Algorithmic Bias & Filter Bubbles
- **Popularity Reinforcement**: Without mitigation, recommenders create positive feedback loops where popular items become ever more popular simply because they are displayed. The Hybrid model offsets this by capping popularity weight ($\gamma=0.2$) and measuring catalog coverage continuously.
- **Homogenization**: Exposing users only to items matching their historical profile can narrow their horizon. We recommend allocating $\epsilon=10\%$ of recommendation slots to serendipitous exploration.

### 8.2 Privacy & Sensitive Data
- No personally identifiable information (PII) such as real names, credit card numbers, IP addresses, or sensitive health records is used.
- All IDs are anonymized hashes (`Uxxxx`, `Ixxxx`).
- Models are trained on aggregate behavioral statistics with zero personal data persistence.

### 8.3 Human-in-the-Loop Oversight
- The system is explicitly configured as a **decision-support tool**:
  - The UI clearly presents recommendations with reasons (e.g., *"Trending"*, *"Similar to X"*).
  - Customers are never coerced; all outputs are presented as non-binding suggestions.
  - Category managers retain administrative veto powers to suppress or pin products.

---

## 9. System Architecture & Inference Deployment

```
[ Web Browser ]
      |
      v
[ Streamlit UI (Port 8501) ]
      |  (HTTP REST)
      v
[ FastAPI Engine (Port 8000) ]
      |
      +---> [ ModelLoader Singleton ]
                  |
                  +---> popularity.pkl
                  +---> item_item_cf.pkl
                  +---> content_based.pkl
                  +---> hybrid.pkl
```

### 9.1 API Endpoints (`api/main.py`)
- `GET /health`: Health check and verification of loaded pickle artifacts.
- `GET /models`: Introspection endpoint exposing all candidate algorithms and hyperparameters.
- `GET /recommend/{user_id}?n=10&model=hybrid`: Real-time personalized top-$N$ recommendation with item metadata and reasoning.
- `GET /items/{item_id}/similar`: Content-based item similarity lookup.
- `GET /users/{user_id}/history`: Retrieval of user chronological history.
- `POST /feedback`: Captures explicit user feedback to close the continuous learning loop.

### 9.2 Interactive Dashboard (`app/streamlit_app.py`)
The Streamlit application provides a 5-tab user portal:
1. **Product Discovery**: Personalized top-$N$ card grid with score badges, category tags, and explanation strings.
2. **Item Explorer**: Visual item-to-item similarity lookup.
3. **Model Benchmark Dashboard**: Comparative metric charts (NDCG, Recall, Precision) across models.
4. **User Chronology**: Complete historical interaction timeline viewer.
5. **Model Card & Transparency**: Detailed ethical, privacy, and limitation disclosures.

---

## 10. Conclusion & Verification

This project fulfills all criteria of the Algions AI Labs internship assessment:
- **Reproducibility**: `make all` or `python src/train.py` executes cleanly with seed `42`.
- **Methodological Rigor**: 4 models compared under a zero-leakage chronological split.
- **Explainability**: Clear inference signals and responsible AI documentation.
- **Production Readiness**: Fully decoupled FastAPI backend, Streamlit frontend, and automated pytest suite.

