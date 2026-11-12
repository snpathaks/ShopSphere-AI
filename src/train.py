"""
Training pipeline for the E-Commerce Product Recommendation Engine.

Run as:
    python src/train.py
"""

import sys
import json
import pickle
import logging
from pathlib import Path

# Allow running as `python src/train.py` from project root
_SRC_DIR = Path(__file__).parent
_BASE_DIR = _SRC_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from src.preprocessing import (
    load_raw_data,
    clean_interactions,
    build_user_item_matrix,
    chronological_split,
    build_item_features,
    save_processed,
)
from src.models import (
    PopularityRecommender,
    ItemItemCF,
    ContentBasedRecommender,
    HybridRecommender,
)
from src.evaluation import compare_models, cold_start_analysis, catalog_coverage
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_DAYS,
    TOP_K_VALUES,
    HYBRID_ALPHA,
    HYBRID_BETA,
    HYBRID_GAMMA,
    INTERACTION_WEIGHTS,
    NUM_USERS,
    NUM_ITEMS,
    NUM_CATEGORIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _step(n: int, total: int, msg: str) -> None:
    print(f"Step {n}/{total}: {msg}", flush=True)
    logger.info(msg)


def train(test_days: int = TEST_DAYS, models_dir: Path = MODELS_DIR):
    """
    Full training pipeline.

    Returns
    -------
    (models_dict, results_dict)
        models_dict  : {name: fitted model}
        results_dict : {comparison, cold_start, coverage, summary}
    """
    total_steps = 12
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    _step(1, total_steps, "Loading raw data...")
    users_df, products_df, interactions_df = load_raw_data(str(RAW_DATA_DIR))
    print(
        f"  Loaded {len(users_df)} users, {len(products_df)} products, "
        f"{len(interactions_df)} interactions."
    )

    # ------------------------------------------------------------------
    # Step 2: Clean interactions
    # ------------------------------------------------------------------
    _step(2, total_steps, "Cleaning interactions...")
    interactions_df = clean_interactions(interactions_df)
    print(f"  {len(interactions_df)} interactions after cleaning.")

    # ------------------------------------------------------------------
    # Step 3: Chronological split
    # ------------------------------------------------------------------
    _step(3, total_steps, f"Chronological split (last {test_days} days → test)...")
    train_df, test_df = chronological_split(interactions_df, test_days=test_days)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows.")

    # ------------------------------------------------------------------
    # Step 4: Build user-item matrix (train only)
    # ------------------------------------------------------------------
    _step(4, total_steps, "Building user-item matrix from train data...")
    user_item_matrix = build_user_item_matrix(train_df, weight_map=INTERACTION_WEIGHTS)
    print(f"  Matrix shape: {user_item_matrix.shape}.")

    # ------------------------------------------------------------------
    # Step 5: Build item features
    # ------------------------------------------------------------------
    _step(5, total_steps, "Building item features from products catalogue...")
    item_features = build_item_features(products_df)
    print(f"  Item features shape: {item_features.shape}.")

    # ------------------------------------------------------------------
    # Step 6: Fit all 4 models
    # ------------------------------------------------------------------
    _step(6, total_steps, "Fitting all 4 models on train data...")

    popularity_model = PopularityRecommender()
    popularity_model.fit(train_df)
    print("  [1/4] PopularityRecommender fitted.")

    item_cf_model = ItemItemCF()
    item_cf_model.fit(user_item_matrix)
    print("  [2/4] ItemItemCF fitted.")

    content_model = ContentBasedRecommender()
    content_model.fit(item_features, products_df)
    print("  [3/4] ContentBasedRecommender fitted.")

    hybrid_model = HybridRecommender(
        alpha=HYBRID_ALPHA,
        beta=HYBRID_BETA,
        gamma=HYBRID_GAMMA,
    )
    hybrid_model.fit(
        user_item_matrix=user_item_matrix,
        item_features=item_features,
        train_df=train_df,
        products_df=products_df,
    )
    print("  [4/4] HybridRecommender fitted.")

    models_dict = {
        "popularity": popularity_model,
        "item_item_cf": item_cf_model,
        "content_based": content_model,
        "hybrid": hybrid_model,
    }

    # ------------------------------------------------------------------
    # Step 7: Save model artifacts
    # ------------------------------------------------------------------
    _step(7, total_steps, "Saving model artifacts to models/ directory...")
    model_files = {
        "popularity": "popularity.pkl",
        "item_item_cf": "item_item_cf.pkl",
        "content_based": "content_based.pkl",
        "hybrid": "hybrid.pkl",
    }
    for key, filename in model_files.items():
        out_path = models_dir / filename
        with open(out_path, "wb") as f:
            pickle.dump(models_dict[key], f)
        print(f"  Saved {out_path}.")

    # ------------------------------------------------------------------
    # Step 8: Compare models on test set
    # ------------------------------------------------------------------
    _step(8, total_steps, f"Running compare_models() at K={TOP_K_VALUES}...")
    model_list = list(models_dict.values())
    comparison_df = compare_models(
        models=model_list,
        test_df=test_df,
        train_df=train_df,
        k_values=TOP_K_VALUES,
    )
    print(f"  Comparison table shape: {comparison_df.shape}.")

    # ------------------------------------------------------------------
    # Step 9: Cold-start analysis
    # ------------------------------------------------------------------
    _step(9, total_steps, "Running cold_start_analysis() for each model...")
    cold_start_results = {}
    for name, model in models_dict.items():
        cold_start_results[name] = cold_start_analysis(
            model=model,
            test_df=test_df,
            train_df=train_df,
            k=10,
        )
    print(f"  Cold-start analysis complete for {len(cold_start_results)} models.")

    # ------------------------------------------------------------------
    # Step 10: Catalog coverage
    # ------------------------------------------------------------------
    _step(10, total_steps, "Computing catalog_coverage() for each model...")
    all_item_ids = products_df["item_id"].tolist()
    coverage_results = {}
    test_user_ids = test_df["user_id"].unique().tolist()
    for name, model in models_dict.items():
        recommendations = {}
        for uid in test_user_ids:
            try:
                recs = model.recommend(uid, n=10, exclude_seen=True)
                recommendations[uid] = recs
            except Exception:
                recommendations[uid] = []
        coverage_results[name] = catalog_coverage(recommendations, all_item_ids)
    print(f"  Coverage: { {k: round(v, 4) for k, v in coverage_results.items()} }")

    # ------------------------------------------------------------------
    # Step 11: Save evaluation results
    # ------------------------------------------------------------------
    _step(11, total_steps, "Saving evaluation results to models/evaluation_results.json...")
    results_dict = {
        "comparison": comparison_df.to_dict(orient="records"),
        "cold_start": cold_start_results,
        "coverage": coverage_results,
    }
    results_path = models_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"  Saved {results_path}.")

    # ------------------------------------------------------------------
    # Step 12: Print summary table
    # ------------------------------------------------------------------
    _step(12, total_steps, "Printing summary table...")
    _print_summary(comparison_df, cold_start_results, coverage_results)

    return models_dict, results_dict


def _print_summary(comparison_df, cold_start_results, coverage_results):
    """Print a human-readable summary to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  MODEL EVALUATION SUMMARY")
    print(sep)

    # Comparison metrics
    print("\n--- Metric Comparison (all K values) ---")
    try:
        col_order = ["model_name", "k", "precision@k", "recall@k", "ndcg@k"]
        cols = [c for c in col_order if c in comparison_df.columns]
        print(comparison_df[cols].to_string(index=False))
    except Exception as exc:
        print(f"  (Could not render comparison table: {exc})")

    # Coverage
    print("\n--- Catalog Coverage ---")
    for model_name, cov in coverage_results.items():
        print(f"  {model_name:<20}: {cov:.2%}")

    # Cold-start
    print("\n--- Cold-Start Analysis (k=10) ---")
    for model_name, segments in cold_start_results.items():
        print(f"  {model_name}:")
        for segment, metrics in segments.items():
            if isinstance(metrics, dict):
                parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            else:
                parts = str(metrics)
            print(f"    {segment:<8}: {parts}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    try:
        train()
        sys.exit(0)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
