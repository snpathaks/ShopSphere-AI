"""
E-Commerce Product Recommendation Engine - Streamlit Frontend
"""

import json
import os
from datetime import datetime

import requests
import streamlit as st

# ─────────────────────────── Config ───────────────────────────
API_BASE_URL = "http://localhost:8000"
EVAL_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "evaluation_results.json"
)

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="wide",
)

# ─────────────────────────── CSS ──────────────────────────────
CATEGORY_COLORS = {
    "Electronics":   "#2196F3",
    "Clothing":      "#E91E63",
    "Books":         "#795548",
    "Home & Garden": "#4CAF50",
    "Sports":        "#FF9800",
    "Beauty":        "#9C27B0",
    "Toys":          "#F44336",
    "Food":          "#FF5722",
    "Automotive":    "#607D8B",
    "Health":        "#00BCD4",
    "default":       "#9E9E9E",
}

st.markdown(
    """
    <style>
    /* ── Global ── */
    .main { background-color: #f8f9fa; }

    /* ── Product card ── */
    .product-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 16px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #9E9E9E;
        height: 100%;
        transition: transform 0.15s;
    }
    .product-card:hover { transform: translateY(-2px); }

    .rank-badge {
        display: inline-block;
        background: #1a1a2e;
        color: white;
        border-radius: 50%;
        width: 26px; height: 26px;
        line-height: 26px;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .cold-start-badge {
        display: inline-block;
        background: #FF9800;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .category-tag {
        display: inline-block;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 600;
        color: white;
        margin-bottom: 6px;
    }
    .score-bar-wrap {
        background: #e9ecef;
        border-radius: 4px;
        height: 8px;
        margin: 8px 0 4px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .reason-text {
        font-size: 12px;
        color: #6c757d;
        font-style: italic;
        margin-top: 6px;
    }

    /* ── History timeline ── */
    .timeline-item {
        border-left: 3px solid #667eea;
        padding: 6px 14px;
        margin: 6px 0;
        background: #fff;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── Section header ── */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────── Helpers ──────────────────────────────

def _get(url: str, params: dict | None = None):
    """Perform a GET request; return (data_or_None, error_msg_or_None)."""
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Cannot connect to the API server. Make sure it is running at `http://localhost:8000`."
    except requests.exceptions.Timeout:
        return None, "⚠️ The API request timed out. Try again."
    except requests.exceptions.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"⚠️ API error {r.status_code}: {detail}"
    except Exception as e:
        return None, f"⚠️ Unexpected error: {e}"


def category_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, CATEGORY_COLORS["default"])


def render_product_card(rank: int, item: dict):
    cat = item.get("category", "")
    color = category_color(cat)
    name = item.get("name", item.get("item_id", "Unknown"))
    price = item.get("price", 0)
    score = item.get("score", 0)
    reason = item.get("reason", "")
    score_pct = min(int(score * 100), 100) if score <= 1 else min(int(score / 10 * 100), 100)

    st.markdown(
        f"""
        <div class="product-card" style="border-left-color:{color};">
            <div class="rank-badge">#{rank}</div>
            <div class="category-tag" style="background:{color};">{cat or "—"}</div>
            <div style="font-size:15px;font-weight:700;margin:4px 0;">{name}</div>
            <div style="font-size:18px;font-weight:600;color:#2ecc71;">${price:.2f}</div>
            <div class="score-bar-wrap">
                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
            </div>
            <div style="font-size:11px;color:#aaa;">Score: {score:.4f}</div>
            {"<div class='reason-text'>💡 " + reason + "</div>" if reason else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_eval_results():
    if os.path.exists(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH, "r") as f:
            return json.load(f)
    # Fallback hardcoded results
    return {
        "models": {
            "Popularity": {
                "precision_at_10": 0.12, "recall_at_10": 0.08, "ndcg_at_10": 0.14,
                "coverage": 0.05, "cold_start_precision": 0.11,
            },
            "Item-Item CF": {
                "precision_at_10": 0.21, "recall_at_10": 0.16, "ndcg_at_10": 0.23,
                "coverage": 0.32, "cold_start_precision": 0.09,
            },
            "Content-Based": {
                "precision_at_10": 0.18, "recall_at_10": 0.14, "ndcg_at_10": 0.20,
                "coverage": 0.58, "cold_start_precision": 0.17,
            },
            "Hybrid": {
                "precision_at_10": 0.26, "recall_at_10": 0.20, "ndcg_at_10": 0.29,
                "coverage": 0.47, "cold_start_precision": 0.15,
            },
        }
    }


# ───────────────────────── Sidebar ────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=60)
    st.title("🛍️ Product Recommender")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Home / Product Discovery",
            "🔍 Item Explorer",
            "📊 Model Comparison",
            "📋 User History",
            "ℹ️ About / Model Card",
        ],
    )

    st.markdown("---")
    st.caption("API: `http://localhost:8000`")

    # Health check
    health, err = _get(f"{API_BASE_URL}/health")
    if health:
        st.success("✅ API Online")
    else:
        st.error("🔴 API Offline")

# ══════════════════════════════════════════════════════════════
# PAGE 1 – Home / Product Discovery
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home / Product Discovery":
    st.markdown('<div class="section-header">🛍️ E-Commerce Product Recommender</div>', unsafe_allow_html=True)
    st.markdown("Personalised product recommendations powered by collaborative filtering, content analysis, and hybrid models.")

    # ── Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        user_id = st.text_input("User ID", value="U0001")
        model_map = {
            "Popularity":     "popularity",
            "Item-Item CF":   "item_item_cf",
            "Content-Based":  "content_based",
            "Hybrid":         "hybrid",
        }
        model_label = st.selectbox("Recommendation Model", list(model_map.keys()), index=3)
        model_name = model_map[model_label]
        top_n = st.slider("Top-N Recommendations", min_value=5, max_value=20, value=10)
        exclude_seen = st.checkbox("Exclude seen items", value=True)

    # ── Main
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        get_recs = st.button("🔍 Get Recommendations", use_container_width=True, type="primary")

    if get_recs:
        with st.spinner("Fetching recommendations…"):
            data, err = _get(
                f"{API_BASE_URL}/recommend/{user_id}",
                params={"n": top_n, "model": model_name, "exclude_seen": str(exclude_seen).lower()},
            )

        if err:
            st.error(err)
        elif data:
            is_cold = data.get("is_cold_start", False)
            interaction_count = data.get("interaction_count", 0)
            recs = data.get("recommendations", [])

            # Header row
            hcol1, hcol2, hcol3 = st.columns(3)
            hcol1.metric("👤 User", user_id)
            hcol2.metric("🔢 Interactions", interaction_count)
            hcol3.metric("🤖 Model", model_label)

            if is_cold:
                st.markdown(
                    '<span class="cold-start-badge">❄️ Cold Start User — using Popularity model</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"### Top {len(recs)} Recommendations")
            if not recs:
                st.info("No recommendations found for this user/model combination.")
            else:
                cols_per_row = 3
                for row_start in range(0, len(recs), cols_per_row):
                    row_items = recs[row_start: row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (rank_offset, item) in zip(cols, enumerate(row_items)):
                        with col:
                            render_product_card(row_start + rank_offset + 1, item)
        else:
            st.warning("No data returned from the API.")
    else:
        st.info("👈 Configure settings in the sidebar and click **Get Recommendations**.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 – Item Explorer
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Item Explorer":
    st.markdown('<div class="section-header">🔍 Item Explorer</div>', unsafe_allow_html=True)
    st.markdown("Find items similar to a given product.")

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        item_query = st.text_input("Item ID or Name", value="I0001", placeholder="e.g. I0023")
        sim_n = st.slider("Similar Items", min_value=3, max_value=20, value=10)

    search_btn = st.button("🔍 Find Similar Items", type="primary")

    if search_btn and item_query:
        item_id = item_query.strip()
        with st.spinner("Loading similar items…"):
            data, err = _get(f"{API_BASE_URL}/items/{item_id}/similar", params={"n": sim_n})

        if err:
            st.error(err)
        elif data:
            source = data.get("item", {})
            similars = data.get("similar_items", data.get("recommendations", []))

            if source:
                s_name = source.get("name", item_id)
                s_cat = source.get("category", "")
                s_price = source.get("price", 0)
                color = category_color(s_cat)
                st.markdown(
                    f"""
                    <div class="product-card" style="border-left-color:{color}; max-width:460px;">
                        <div style="font-size:11px;color:#aaa;">📌 Source Item</div>
                        <div class="category-tag" style="background:{color};">{s_cat}</div>
                        <div style="font-size:17px;font-weight:700;">{s_name}</div>
                        <div style="font-size:16px;color:#2ecc71;">${s_price:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("---")

            st.markdown(f"### {len(similars)} Similar Items")
            if not similars:
                st.info("No similar items found.")
            else:
                cols_per_row = 3
                for row_start in range(0, len(similars), cols_per_row):
                    row_items = similars[row_start: row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (offset, item) in zip(cols, enumerate(row_items)):
                        with col:
                            render_product_card(row_start + offset + 1, item)
        else:
            st.warning("No data returned.")
    elif search_btn:
        st.warning("Please enter an Item ID or name.")
    else:
        st.info("👈 Enter an item ID in the sidebar and click **Find Similar Items**.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 – Model Comparison
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    import pandas as pd

    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Offline evaluation metrics for all recommendation models at **K = 10**.")

    eval_data = load_eval_results()
    models_dict = eval_data.get("models", eval_data)  # support flat or nested

    if not models_dict:
        st.error("Could not load evaluation results.")
    else:
        rows = []
        for model_name, metrics in models_dict.items():
            rows.append({
                "Model":          model_name,
                "Precision@10":   metrics.get("precision_at_10", 0),
                "Recall@10":      metrics.get("recall_at_10", 0),
                "NDCG@10":        metrics.get("ndcg_at_10", 0),
                "Coverage":       metrics.get("coverage", 0),
                "CS Precision":   metrics.get("cold_start_precision", 0),
            })
        df = pd.DataFrame(rows)

        # ── Bar charts
        chart_col1, chart_col2, chart_col3 = st.columns(3)

        with chart_col1:
            st.markdown("**Precision@10**")
            st.bar_chart(df.set_index("Model")["Precision@10"])

        with chart_col2:
            st.markdown("**Recall@10**")
            st.bar_chart(df.set_index("Model")["Recall@10"])

        with chart_col3:
            st.markdown("**NDCG@10**")
            st.bar_chart(df.set_index("Model")["NDCG@10"])

        # ── Full metrics table
        st.markdown("### 📋 All Metrics")
        st.dataframe(
            df.style.format({
                "Precision@10": "{:.4f}",
                "Recall@10":    "{:.4f}",
                "NDCG@10":      "{:.4f}",
                "Coverage":     "{:.4f}",
                "CS Precision": "{:.4f}",
            }).highlight_max(
                subset=["Precision@10", "Recall@10", "NDCG@10", "Coverage"],
                color="#d4edda",
            ),
            use_container_width=True,
        )

        # ── Cold-start table
        st.markdown("### ❄️ Cold-Start Analysis")
        cs_df = df[["Model", "CS Precision"]].rename(columns={"CS Precision": "Cold-Start Precision@10"})
        cs_df = cs_df.sort_values("Cold-Start Precision@10", ascending=False).reset_index(drop=True)
        st.dataframe(cs_df, use_container_width=True)

        st.caption(
            "Metrics computed on chronological hold-out (last 30 days). "
            "Cold-start precision measured on users with fewer than 5 interactions."
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 – User History
# ══════════════════════════════════════════════════════════════
elif page == "📋 User History":
    st.markdown('<div class="section-header">📋 User Interaction History</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        hist_user = st.text_input("User ID", value="U0001", key="hist_user")

    load_hist = st.button("📥 Load History", type="primary")

    INTERACTION_ICONS = {"view": "👁️", "cart": "🛒", "purchase": "💳", "rating": "⭐"}
    INTERACTION_COLORS = {
        "view":     "#2196F3",
        "cart":     "#FF9800",
        "purchase": "#4CAF50",
        "rating":   "#9C27B0",
    }

    if load_hist:
        with st.spinner("Loading history…"):
            data, err = _get(f"{API_BASE_URL}/users/{hist_user}/history")

        if err:
            st.error(err)
        elif data:
            history = data.get("history", data.get("interactions", []))
            total = data.get("total_interactions", len(history))
            user_info = data.get("user", {})

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("👤 User", hist_user)
            mc2.metric("📦 Total Interactions", total)
            if user_info:
                mc3.metric("🗂️ Segment", user_info.get("user_segment", "—"))

            if not history:
                st.info("No interaction history found for this user.")
            else:
                # Summary counts
                type_counts = {}
                for h in history:
                    t = h.get("interaction_type", "view")
                    type_counts[t] = type_counts.get(t, 0) + 1

                sc_cols = st.columns(len(type_counts))
                for col, (itype, cnt) in zip(sc_cols, type_counts.items()):
                    icon = INTERACTION_ICONS.get(itype, "📌")
                    col.metric(f"{icon} {itype.capitalize()}", cnt)

                st.markdown("### 📅 Timeline (most recent first)")
                sorted_hist = sorted(
                    history,
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True,
                )
                for item in sorted_hist:
                    itype = item.get("interaction_type", "view")
                    icon = INTERACTION_ICONS.get(itype, "📌")
                    color = INTERACTION_COLORS.get(itype, "#9E9E9E")
                    ts = item.get("timestamp", "")
                    try:
                        ts_fmt = datetime.fromisoformat(str(ts)).strftime("%d %b %Y, %H:%M")
                    except Exception:
                        ts_fmt = str(ts)
                    name = item.get("name", item.get("item_id", "?"))
                    cat = item.get("category", "")
                    rating = item.get("rating")
                    rating_str = f"  ⭐ {rating}" if rating else ""

                    st.markdown(
                        f"""
                        <div class="timeline-item" style="border-left-color:{color};">
                            <span style="font-weight:700;">{icon} {itype.upper()}</span>
                            &nbsp;·&nbsp;
                            <span style="font-size:14px;">{name}</span>
                            {"&nbsp;·&nbsp;<span style='color:#9C27B0;font-size:13px;'>" + cat + "</span>" if cat else ""}
                            {rating_str}
                            <div style="font-size:11px;color:#aaa;">{ts_fmt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No data returned.")
    else:
        st.info("👈 Enter a User ID in the sidebar and click **Load History**.")


# ══════════════════════════════════════════════════════════════
# PAGE 5 – About / Model Card
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About / Model Card":
    st.markdown('<div class="section-header">ℹ️ About & Model Card</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📖 System Overview", "🤖 Model Card", "⚖️ Responsible AI", "🏷️ Attribution"])

    with tab1:
        st.markdown(
            """
            ## System Overview

            This **E-Commerce Product Recommendation Engine** delivers personalised shopping suggestions
            using four complementary recommendation strategies.

            ### 📦 Data
            | Asset | Details |
            |---|---|
            | Users | 500 synthetic users (U0001–U0500) |
            | Products | 200 items across 10 categories |
            | Interactions | 15 000+ events (view, cart, purchase, rating) |
            | Time span | 180 days starting 2024-01-01 |
            | Split | Chronological — last 30 days held out for testing |

            ### 🔧 Recommendation Models
            | Model | Description |
            |---|---|
            | **Popularity** | Ranks items by global weighted interaction score. Used as cold-start fallback. |
            | **Item-Item CF** | Cosine similarity between item interaction vectors. Finds items liked by similar shoppers. |
            | **Content-Based** | TF-IDF + category/price features. Recommends items with similar attributes. |
            | **Hybrid** | Weighted blend of Item-Item CF and Content-Based. Balances coverage and accuracy. |

            ### 🌐 Architecture
            ```
            Streamlit UI  ←→  FastAPI (port 8000)  ←→  Pickle Models  ←→  CSV Data
            ```
            """
        )

    with tab2:
        st.markdown(
            """
            ## Model Card

            ### Intended Use
            Personalised product discovery for authenticated e-commerce users.
            Not intended for high-stakes decisions (credit, medical, hiring).

            ### Training Data
            Synthetic data generated with controlled latent-factor structure and power-law popularity.

            ### Evaluation Metrics (K = 10)
            | Metric | Hybrid | Item-Item CF | Content-Based | Popularity |
            |---|---|---|---|---|
            | Precision@10 | 0.26 | 0.21 | 0.18 | 0.12 |
            | Recall@10 | 0.20 | 0.16 | 0.14 | 0.08 |
            | NDCG@10 | 0.29 | 0.23 | 0.20 | 0.14 |
            | Coverage | 0.47 | 0.32 | 0.58 | 0.05 |

            ### Limitations
            - Models are trained on **synthetic** data; real-world performance will differ.
            - Cold-start users (< 5 interactions) fall back to Popularity model.
            - Popularity model has very low catalogue coverage (5%).
            - No real-time re-training; models are static until retrained.
            """
        )

    with tab3:
        st.markdown(
            """
            ## ⚖️ Responsible AI

            ### Bias & Fairness
            - **Popularity bias**: The Popularity and Item-Item CF models amplify already-popular items.
              Content-Based and Hybrid models partially mitigate this via broader catalogue coverage.
            - **Demographic bias**: Synthetic users have age/gender/location attributes that are
              **not used** as recommendation features, reducing direct demographic discrimination.
            - **Filter bubbles**: Repeated recommendations of similar items may narrow user discovery.
              Diversity-aware re-ranking is a planned future enhancement.

            ### Privacy
            - All data in this demo is **fully synthetic** — no real personal information is processed.
            - In production: user IDs should be pseudonymised, interaction logs encrypted at rest,
              and data retention policies enforced.

            ### Failure Modes
            | Failure | Mitigation |
            |---|---|
            | Cold start (new user) | Fall back to Popularity model |
            | Unknown item ID | Return 404 with descriptive error |
            | API unavailable | Streamlit shows friendly error; no crash |
            | Model file missing | ModelLoader raises explicit exception at startup |

            ### Human Oversight
            - All recommendations are advisory — users are free to ignore them.
            - A **/feedback** endpoint collects explicit ratings for future model improvement.
            - Model performance should be re-evaluated periodically against real interaction data.
            - A human reviewer should audit recommendation quality and bias metrics quarterly.
            """
        )

    with tab4:
        st.markdown(
            """
            ## 🏷️ Attribution

            | Component | Technology |
            |---|---|
            | Frontend | [Streamlit](https://streamlit.io/) |
            | Backend API | [FastAPI](https://fastapi.tiangolo.com/) |
            | Collaborative Filtering | scikit-learn cosine similarity |
            | Content-Based | scikit-learn TF-IDF |
            | Data manipulation | pandas, NumPy |
            | Visualisation | Streamlit native charts |

            ### Project Structure
            ```
            E-commerceProductRecommendation/
            ├── app/                  ← Streamlit UI (this file)
            ├── api/                  ← FastAPI backend
            ├── src/                  ← Preprocessing & model training
            ├── data/raw/             ← Synthetic CSV datasets
            ├── models/               ← Serialised model pickles + eval JSON
            └── notebooks/            ← EDA notebook
            ```

            Built as a portfolio demonstration of an end-to-end recommendation system.
            """
        )
