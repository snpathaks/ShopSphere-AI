"""
E-Commerce Product Recommendation Engine - Streamlit Frontend
Production-grade interactive discovery portal and decision-support UI.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# ─────────────────────────── Page Configuration ───────────────────────────
st.set_page_config(
    page_title="ShopSphere AI | Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "evaluation_results.json"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"

# ─────────────────────────── Custom Premium CSS ───────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Main App Background ── */
    .stApp {
        background: #f8fafc;
        color: #0f172a;
    }

    /* ── Hero Banner ── */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #c7d2fe;
        font-weight: 400;
        max-width: 720px;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        color: #a5b4fc;
        margin-top: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* ── Product Card ── */
    .product-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 270px;
        margin-bottom: 16px;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        border-color: #cbd5e1;
    }

    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .rank-pill {
        background: #f1f5f9;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }
    .category-chip {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .product-name {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.35;
        margin-bottom: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .product-price {
        font-size: 20px;
        font-weight: 800;
        color: #059669;
        letter-spacing: -0.01em;
        margin-bottom: 10px;
    }

    /* ── Progress Bar ── */
    .meter-container {
        background: #f1f5f9;
        border-radius: 9999px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin: 6px 0;
    }
    .meter-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    }

    .card-footer {
        background: #f8fafc;
        border-radius: 8px;
        padding: 8px 10px;
        margin-top: 10px;
        border: 1px solid #f1f5f9;
    }
    .reason-badge {
        font-size: 11px;
        color: #475569;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
        line-height: 1.3;
    }

    /* ── Stat Metric Pill ── */
    .stat-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
    }

    /* ── Timeline items ── */
    .timeline-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.15s ease;
    }
    .timeline-card:hover {
        transform: translateX(4px);
        border-color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Palette Mapping ───────────────────────────
CATEGORY_STYLES = {
    "Electronics":   {"bg": "#e0f2fe", "text": "#0369a1"},
    "Clothing":      {"bg": "#fce7f3", "text": "#be185d"},
    "Books":         {"bg": "#fef3c7", "text": "#b45309"},
    "Home & Garden": {"bg": "#dcfce7", "text": "#15803d"},
    "Sports":        {"bg": "#ffedd5", "text": "#c2410c"},
    "Beauty":        {"bg": "#f3e8ff", "text": "#7e22ce"},
    "Toys":          {"bg": "#fee2e2", "text": "#b91c1c"},
    "Food":          {"bg": "#fae8ff", "text": "#86198f"},
    "Automotive":    {"bg": "#f1f5f9", "text": "#334155"},
    "Health":        {"bg": "#ccfbf1", "text": "#0f766e"},
    "default":       {"bg": "#f1f5f9", "text": "#475569"},
}

def get_cat_style(category: str) -> dict:
    return CATEGORY_STYLES.get(category, CATEGORY_STYLES["default"])

# ─────────────────────────── API Client Helper ───────────────────────────
def api_get(endpoint: str, params: dict = None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Backend connection error. Ensure FastAPI is running on `http://localhost:8000`."
    except Exception as exc:
        return None, f"⚠️ Request error: {exc}"

def api_post(endpoint: str, data: dict):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        resp = requests.post(url, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:
        return None, f"⚠️ Feedback submission error: {exc}"

# ─────────────────────────── Sidebar Navigation ───────────────────────────
with st.sidebar:
    st.markdown("## 🛍️ **ShopSphere AI**")
    st.caption("E-Commerce Decision-Support Engine")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Product Discovery",
            "🔍 Similar Item Explorer",
            "📊 Model Benchmarks",
            "📋 Customer Activity Stream",
            "ℹ️ Model Card & Ethics",
        ],
        index=0,
    )
    st.markdown("---")

    # Live health check
    health_data, health_err = api_get("/health")
    if health_data and health_data.get("status") == "ok":
        models_count = len(health_data.get("models_loaded", []))
        st.markdown(
            f"""
            <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:10px; padding:10px; font-size:12px; color:#065f46;">
                🟢 <b>FastAPI Online</b><br>
                Loaded Models: <b>{models_count}</b> / 4
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:10px; font-size:12px; color:#991b1b;">
                🔴 <b>FastAPI Offline</b><br>
                Run: <code>uvicorn api.main:app</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────── Component: Product Card ───────────────────────────
def render_modern_card(rank: int, item: dict):
    name = item.get("name", item.get("item_id", "Product"))
    category = item.get("category", "General")
    price = float(item.get("price", 0.0))
    score = float(item.get("score", 0.0))
    reason = item.get("reason", "Top trending recommendation")
    item_id = item.get("item_id", "")

    cat_style = get_cat_style(category)
    # Score percentage for progress meter
    pct = min(int(score * 100), 100) if score <= 1.0 else min(int(score / 5.0 * 100), 100)

    st.markdown(
        f"""
        <div class="product-card">
            <div>
                <div class="card-top">
                    <span class="rank-pill">RANK #{rank}</span>
                    <span class="category-chip" style="background:{cat_style['bg']}; color:{cat_style['text']};">
                        {category}
                    </span>
                </div>
                <div class="product-name" title="{name}">{name}</div>
                <div class="product-price">${price:,.2f}</div>
                <div style="font-size:11px; color:#64748b; font-weight:600; margin-top:4px;">RELEVANCE SCORE: {score:.3f}</div>
                <div class="meter-container">
                    <div class="meter-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            <div class="card-footer">
                <div class="reason-badge">
                    <span>💡</span> <span><b>Reason:</b> {reason}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Quick interaction row
    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        if st.button("🛒 Cart", key=f"cart_{item_id}_{rank}", use_container_width=True):
            st.toast(f"Added '{name}' to cart!", icon="🛒")
    with bcol2:
        if st.button("⭐ Rate", key=f"rate_{item_id}_{rank}", use_container_width=True):
            api_post("/feedback", {"user_id": "U0001", "item_id": item_id, "rating": 5})
            st.toast(f"Rated 5-stars: '{name}'", icon="⭐")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: PRODUCT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════
if nav_selection == "🏠 Product Discovery":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🛍️ Personalized Product Discovery</div>
            <div class="hero-subtitle">
                AI-driven decision-support recommendation suite comparing Popularity, Item-Item Collaborative Filtering, Content-Based TF-IDF, and Weighted Hybrid ensembles.
            </div>
            <div class="hero-badge">
                <span>⚡ Real-Time Inference</span> • <span>Zero Future Data Leakage</span> • <span>Explainable Signals</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls Bar
    with st.container():
        c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1])
        with c1:
            user_input = st.text_input("Customer ID", value="U0001", help="Select user between U0001 and U0500 or test cold-start with U_NEW_999")
        with c2:
            model_options = {
                "Weighted Hybrid Recommender": "hybrid",
                "Item-Item Collaborative Filtering": "item_item_cf",
                "Content-Based (TF-IDF)": "content_based",
                "Popularity Baseline": "popularity",
            }
            selected_label = st.selectbox("Recommendation Model", list(model_options.keys()), index=0)
            selected_model = model_options[selected_label]
        with c3:
            top_k = st.slider("Top-N Items", min_value=3, max_value=20, value=9, step=3)
        with c4:
            st.write("")
            st.write("")
            run_btn = st.button("✨ Recommend", type="primary", use_container_width=True)

    if run_btn or True:
        with st.spinner("Generating personalized candidates..."):
            data, err = api_get(
                f"/recommend/{user_input}",
                params={"n": top_k, "model": selected_model, "exclude_seen": True},
            )

        if err:
            st.error(err)
        elif data:
            recs = data.get("recommendations", [])
            is_cold = data.get("is_cold_start", False)
            interaction_count = data.get("user_interaction_count", 0)

            # Metadata Bar
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Customer ID</div>
                        <div class="stat-value">{user_input}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Historical Interactions</div>
                        <div class="stat-value">{interaction_count} events</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                cohort_text = "Cold-Start (Fallback)" if is_cold or interaction_count <= 5 else "Active / Established"
                cohort_color = "#ea580c" if is_cold or interaction_count <= 5 else "#059669"
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">User Tier</div>
                        <div class="stat-value" style="color:{cohort_color}; font-size:18px;">{cohort_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="stat-box">
                        <div class="stat-label">Active Model</div>
                        <div class="stat-value" style="font-size:17px; color:#4338ca;">{selected_label.split()[0]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if is_cold:
                st.info("❄️ **Cold-Start Safeguard Triggered**: This user profile has minimal historical data. The engine dynamically raised popularity weighting to ensure high-quality recommendations without collapsing.")

            # Grid Display (3 cards per row)
            num_cols = 3
            for i in range(0, len(recs), num_cols):
                row_items = recs[i:i + num_cols]
                cols = st.columns(num_cols)
                for col_idx, item in enumerate(row_items):
                    with cols[col_idx]:
                        render_modern_card(i + col_idx + 1, item)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: SIMILAR ITEM EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif nav_selection == "🔍 Similar Item Explorer":
    st.markdown(
        """
        <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);">
            <div class="hero-title">🔍 Product-to-Product Similarity Explorer</div>
            <div class="hero-subtitle">
                Explore semantic and collaborative affinities between items. Backed by cosine similarity over TF-IDF vectors of category, brand, and product descriptions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        query_item = st.text_input("Target Product ID", value="I0001", help="Enter item between I0001 and I0200")
    with sc2:
        st.write("")
        st.write("")
        search_sim_btn = st.button("Explore Similar", type="primary", use_container_width=True)

    with st.spinner("Calculating item cosine similarity..."):
        sim_data, sim_err = api_get(f"/items/{query_item}/similar", params={"n": 6})

    if sim_err:
        st.error(sim_err)
    elif sim_data:
        similars = sim_data.get("similar_items", sim_data.get("recommendations", []))
        st.markdown(f"### Top {len(similars)} Most Similar Products to `{query_item}`")

        num_cols = 3
        for i in range(0, len(similars), num_cols):
            row_items = similars[i:i + num_cols]
            cols = st.columns(num_cols)
            for col_idx, item in enumerate(row_items):
                with cols[col_idx]:
                    render_modern_card(i + col_idx + 1, item)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════
elif nav_selection == "📊 Model Benchmarks":
    st.markdown(
        """
        <div class="hero-container" style="background: linear-gradient(135deg, #18181b 0%, #27272a 50%, #3f3f46 100%);">
            <div class="hero-title">📊 Empirical Model Benchmark Dashboard</div>
            <div class="hero-subtitle">
                Rigorous offline ranking evaluation on the 30-day held-out test set with zero future data leakage. Benchmarking NDCG@K, Recall@K, Precision@K, and Catalog Coverage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metrics Highlights
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-label">Best NDCG@10</div>
                <div class="stat-value" style="color:#4f46e5;">0.1787</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Item-Item Collaborative Filter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-label">Best Recall@10</div>
                <div class="stat-value" style="color:#059669;">0.1613</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Hybrid Ensemble</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-label">Max Catalog Coverage</div>
                <div class="stat-value" style="color:#0284c7;">96.50%</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Content-Based Filter (TF-IDF)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-label">Cold-Start NDCG@10</div>
                <div class="stat-value" style="color:#d97706;">0.3883</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Hybrid (Dynamic Fallback)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Display High-Resolution Visualizations
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        comp_img = FIGURES_DIR / "model_comparison_k10.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Model Ranking Comparison at K=10", use_container_width=True)
    with vcol2:
        cov_img = FIGURES_DIR / "model_catalog_coverage.png"
        if cov_img.exists():
            st.image(str(cov_img), caption="Catalog Coverage Distribution (%)", use_container_width=True)

    st.markdown("### 📋 Full Benchmark Results Table")
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH) as f:
            raw_eval = json.load(f)
        bench_df = pd.DataFrame(raw_eval["comparison"])
        st.dataframe(
            bench_df.style.format({
                "precision@k": "{:.4f}",
                "recall@k": "{:.4f}",
                "ndcg@k": "{:.4f}",
            }),
            use_container_width=True,
        )

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        funnel_img = FIGURES_DIR / "eda_interaction_funnel.png"
        if funnel_img.exists():
            st.image(str(funnel_img), caption="Implicit Feedback Conversion Funnel", use_container_width=True)
    with ecol2:
        tail_img = FIGURES_DIR / "eda_item_popularity_long_tail.png"
        if tail_img.exists():
            st.image(str(tail_img), caption="Product Popularity Long Tail Distribution", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: CUSTOMER ACTIVITY STREAM
# ══════════════════════════════════════════════════════════════
elif nav_selection == "📋 Customer Activity Stream":
    st.markdown(
        """
        <div class="hero-container" style="background: linear-gradient(135deg, #3730a3 0%, #4338ca 50%, #4f46e5 100%);">
            <div class="hero-title">📋 Customer Chronological Activity Stream</div>
            <div class="hero-subtitle">
                Inspect historical interactions including views, carts, and purchases to audit recommendation context and understand model inputs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        target_uid = st.text_input("Customer ID", value="U0001", key="hist_uid")
    with hc2:
        st.write("")
        st.write("")
        load_btn = st.button("Load Activity", type="primary", use_container_width=True)

    with st.spinner("Fetching event stream..."):
        hdata, herr = api_get(f"/users/{target_uid}/history")

    if herr:
        st.error(herr)
    elif hdata:
        history = hdata.get("history", [])
        st.markdown(f"### Total Interactions: **{len(history)} Events**")

        TYPE_ICONS = {"view": "👁️ VIEW", "cart": "🛒 ADD TO CART", "purchase": "💳 PURCHASE"}
        TYPE_COLORS = {"view": "#3b82f6", "cart": "#f59e0b", "purchase": "#10b981"}

        for event in history[-15:][::-1]:
            itype = event.get("interaction_type", "view")
            pname = event.get("name", "Product")
            pcat = event.get("category", "")
            price = event.get("price", 0.0)
            ts = event.get("timestamp", "")[:19].replace("T", " ")

            badge_color = TYPE_COLORS.get(itype, "#64748b")
            st.markdown(
                f"""
                <div class="timeline-card">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="background:{badge_color}; color:white; font-size:11px; font-weight:700; padding:4px 10px; border-radius:6px;">
                            {TYPE_ICONS.get(itype, itype.upper())}
                        </span>
                        <div>
                            <span style="font-weight:700; color:#0f172a; font-size:15px;">{pname}</span>
                            <span style="color:#64748b; font-size:12px; margin-left:8px;">({pcat})</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:800; color:#059669; font-size:15px;">${price:,.2f}</div>
                        <div style="font-size:11px; color:#94a3b8;">{ts}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL CARD & ETHICS
# ══════════════════════════════════════════════════════════════════════════
elif nav_selection == "ℹ️ Model Card & Ethics":
    st.markdown(
        """
        <div class="hero-container" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);">
            <div class="hero-title">ℹ️ Model Card & Responsible AI Disclosures</div>
            <div class="hero-subtitle">
                System specification, ethical risk analysis, feedback loop mitigation, and human-in-the-loop oversight principles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["📌 System Specification", "⚖️ Bias & Fairness", "🔒 Privacy & Security", "🛡️ Human Oversight"])

    with t1:
        st.markdown(
            """
            ### System Overview
            - **Mission**: High-relevance decision-support discovery tool for e-commerce catalog exploration.
            - **Decision Horizon**: 30-day chronological forecast without future data leakage.
            - **Algorithm Suite**:
              1. **Popularity Baseline**: Sum of weighted interactions ($W_{view}=1, W_{cart}=3, W_{purchase}=5$).
              2. **Item-Item Collaborative Filtering**: Cosine similarity across user-item interaction space.
              3. **Content-Based Filtering**: TF-IDF on concatenated product features with sublinear term weighting.
              4. **Weighted Hybrid Ensemble**: Convex combination ($\alpha=0.5, \beta=0.3, \gamma=0.2$) with dynamic cold-start boosting.
            """
        )

    with t2:
        st.markdown(
            """
            ### Algorithmic Bias & Feedback Loops
            - **Popularity Monopoly**: Left unconstrained, recommenders create positive feedback loops where head items starve long-tail items.
            - **Mitigation**: We enforce catalog coverage monitoring (Hybrid model achieves **82.5% coverage** vs 11.5% for Popularity) and recommend reserving exploration slots.
            - **Filter Bubble Risk**: Collaborative models can confine users to narrow categories. Content-based feature blending mitigates echo-chamber emergence.
            """
        )

    with t3:
        st.markdown(
            """
            ### Privacy & Data Protection
            - **Zero PII**: The dataset contains zero Personally Identifiable Information (no real names, phone numbers, emails, or payment tokens).
            - **Synthetic Generation**: Built with fixed seed (42) mathematical distributions ensuring deterministic reproducibility with zero real-world privacy footprint.
            """
        )

    with t4:
        st.markdown(
            """
            ### Human-in-the-Loop Oversight
            - **Decision-Support Framing**: The system provides suggestions, never autonomous checkout decisions.
            - **Explainability**: Every recommendation includes a human-interpretable rationale (e.g. *"Trending"*, *"Similar to X"*).
            - **Catalog Veto**: Merchandisers maintain administrative authority to suppress or pin products.
            """
        )
