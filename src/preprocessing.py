"""
preprocessing.py
~~~~~~~~~~~~~~~~
Preprocessing pipeline for the E-Commerce Product Recommendation Engine.

Provides:
  - load_raw_data          : load raw CSVs
  - clean_interactions     : deduplicate and parse timestamps
  - build_user_item_matrix : weighted user-item pivot table
  - chronological_split    : train/test split with no future leakage
  - get_user_history       : items a user interacted with in train set
  - build_item_features    : TF-IDF + normalised price feature matrix
  - save_processed         : persist processed artefacts to disk
  - run_preprocessing_pipeline : end-to-end CLI entry point
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str = "data/raw") -> tuple:
    """
    Load users, products, and interactions CSVs from *data_dir*.

    Returns
    -------
    (users_df, products_df, interactions_df) as pandas DataFrames.
    """
    data_dir = Path(data_dir)
    users_path        = data_dir / "users.csv"
    products_path     = data_dir / "products.csv"
    interactions_path = data_dir / "interactions.csv"

    logger.info("Loading users from %s", users_path)
    users_df = pd.read_csv(users_path)

    logger.info("Loading products from %s", products_path)
    products_df = pd.read_csv(products_path)

    logger.info("Loading interactions from %s", interactions_path)
    interactions_df = pd.read_csv(interactions_path)

    logger.info(
        "Loaded: users=%d, products=%d, interactions=%d",
        len(users_df), len(products_df), len(interactions_df),
    )
    return users_df, products_df, interactions_df


# ---------------------------------------------------------------------------
# 2. Clean interactions
# ---------------------------------------------------------------------------

def clean_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw interactions DataFrame.

    Steps:
    1. Drop exact duplicate rows.
    2. Parse *timestamp* column to ``datetime64[ns, UTC]``.
    3. Sort ascending by timestamp.

    Returns
    -------
    Cleaned DataFrame (new object, original unchanged).
    """
    df = interactions_df.copy()
    n_before = len(df)

    # 1. Remove exact duplicates
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows (%d → %d)", n_before - len(df), n_before, len(df))

    # 2. Parse timestamps (handle various ISO-8601 formats gracefully)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts:
        logger.warning("Dropping %d rows with unparseable timestamps", n_bad_ts)
        df = df.dropna(subset=["timestamp"])

    # 3. Sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Interactions cleaned: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Build user-item matrix
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_MAP = {"view": 1, "cart": 3, "purchase": 5}


def build_user_item_matrix(
    interactions_df: pd.DataFrame,
    weight_map: dict = None,
) -> pd.DataFrame:
    """
    Create a weighted user-item interaction matrix.

    Parameters
    ----------
    interactions_df : cleaned interactions DataFrame
    weight_map      : mapping from interaction_type → weight
                      (default: view=1, cart=3, purchase=5)

    Returns
    -------
    Pivot table with:
      - rows    = user_id
      - columns = item_id
      - values  = sum of weighted scores  (NaN → 0)
    """
    if weight_map is None:
        weight_map = DEFAULT_WEIGHT_MAP

    df = interactions_df.copy()

    # Map weights; unknown types get weight 0
    df["weight"] = df["interaction_type"].map(weight_map).fillna(0).astype(float)

    # Aggregate: sum of weights per (user_id, item_id)
    agg = (
        df.groupby(["user_id", "item_id"], sort=False)["weight"]
        .sum()
        .reset_index(name="weighted_score")
    )

    # Pivot to matrix form
    matrix = agg.pivot(index="user_id", columns="item_id", values="weighted_score").fillna(0)
    matrix.columns.name = None
    matrix.index.name = "user_id"

    logger.info(
        "User-item matrix: %d users × %d items (density=%.4f)",
        matrix.shape[0],
        matrix.shape[1],
        (matrix > 0).values.mean(),
    )
    return matrix


# ---------------------------------------------------------------------------
# 4. Chronological train / test split
# ---------------------------------------------------------------------------

def chronological_split(
    interactions_df: pd.DataFrame,
    test_days: int = 30,
) -> tuple:
    """
    Split interactions into train and test sets chronologically.

    The **last** *test_days* calendar days (relative to the latest timestamp
    in the dataset) become the test set; everything before is training.
    This guarantees no future leakage.

    Parameters
    ----------
    interactions_df : cleaned interactions DataFrame (timestamp must be
                      datetime-like)
    test_days       : number of days at the end of the timeline to hold out

    Returns
    -------
    (train_df, test_df)
    """
    df = interactions_df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    max_ts = df["timestamp"].max()
    cutoff  = max_ts - pd.Timedelta(days=test_days)

    train_df = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test_df  = df[df["timestamp"] >  cutoff].reset_index(drop=True)

    logger.info(
        "Chronological split: train=%d rows (up to %s), test=%d rows (last %d days)",
        len(train_df), cutoff.date(), len(test_df), test_days,
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# 5. Get user history
# ---------------------------------------------------------------------------

def get_user_history(train_df: pd.DataFrame, user_id: str) -> list:
    """
    Return the list of item_ids the given user interacted with in the
    training set.  Order matches the original row order in *train_df*.

    Parameters
    ----------
    train_df : training split of interactions
    user_id  : user identifier string (e.g. "U0001")

    Returns
    -------
    List of item_id strings (may be empty for unseen users).
    """
    mask = train_df["user_id"] == user_id
    return train_df.loc[mask, "item_id"].tolist()


# ---------------------------------------------------------------------------
# 6. Build item features
# ---------------------------------------------------------------------------

def build_item_features(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an item feature matrix for content-based filtering.

    Features
    --------
    * TF-IDF vectors derived from the concatenation of:
        category + " " + subcategory + " " + description + " " + brand
    * Price normalised to [0, 1] using MinMaxScaler.

    Returns
    -------
    DataFrame indexed by *item_id* where each column is a feature.
    TF-IDF columns are named ``tfidf_<term>``; price column is ``price_norm``.
    """
    df = products_df.copy().set_index("item_id")

    # 1. Build text corpus
    text_cols = ["category", "subcategory", "description", "brand"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    corpus = (
        df["category"] + " " + df["subcategory"] + " " +
        df["description"] + " " + df["brand"]
    )

    # 2. TF-IDF
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    tfidf_matrix = tfidf.fit_transform(corpus)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=df.index,
        columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()],
    )

    # 3. Normalise price
    prices = df[["price"]].fillna(df["price"].median())
    scaler = MinMaxScaler()
    price_norm = scaler.fit_transform(prices)
    price_df = pd.DataFrame(price_norm, index=df.index, columns=["price_norm"])

    # 4. Concatenate
    features = pd.concat([tfidf_df, price_df], axis=1)
    logger.info(
        "Item feature matrix: %d items × %d features",
        features.shape[0], features.shape[1],
    )
    return features


# ---------------------------------------------------------------------------
# 7. Save processed artefacts
# ---------------------------------------------------------------------------

def save_processed(
    train_df,
    test_df,
    user_item_matrix,
    output_dir: str = "data/processed",
) -> None:
    """
    Persist processed artefacts to *output_dir*.

    Files written
    -------------
    - train_interactions.csv
    - test_interactions.csv
    - user_item_matrix.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path  = output_dir / "train_interactions.csv"
    test_path   = output_dir / "test_interactions.csv"
    matrix_path = output_dir / "user_item_matrix.csv"

    train_df.to_csv(train_path, index=False)
    logger.info("Saved train interactions → %s (%d rows)", train_path, len(train_df))

    test_df.to_csv(test_path, index=False)
    logger.info("Saved test interactions  → %s (%d rows)", test_path, len(test_df))

    user_item_matrix.to_csv(matrix_path)
    logger.info(
        "Saved user-item matrix   → %s (%s)",
        matrix_path, "×".join(str(d) for d in user_item_matrix.shape),
    )


# ---------------------------------------------------------------------------
# 8. End-to-end pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    test_days: int = 30,
) -> dict:
    """
    Run the full preprocessing pipeline end-to-end.

    Steps
    -----
    1. Load raw data
    2. Clean interactions
    3. Build user-item matrix
    4. Chronological train/test split
    5. Build item features
    6. Save processed artefacts

    Returns a dict with all processed artefacts for downstream use.
    """
    logger.info("=== Starting preprocessing pipeline ===")

    # 1. Load
    users_df, products_df, interactions_df = load_raw_data(data_dir)

    # 2. Clean
    interactions_clean = clean_interactions(interactions_df)

    # 3. User-item matrix (on full cleaned data)
    user_item_matrix = build_user_item_matrix(interactions_clean)

    # 4. Train / test split
    train_df, test_df = chronological_split(interactions_clean, test_days=test_days)

    # 5. Item features
    item_features = build_item_features(products_df)

    # 6. Save
    save_processed(train_df, test_df, user_item_matrix, output_dir)

    logger.info("=== Preprocessing pipeline complete ===")

    return {
        "users_df": users_df,
        "products_df": products_df,
        "interactions_df": interactions_clean,
        "train_df": train_df,
        "test_df": test_df,
        "user_item_matrix": user_item_matrix,
        "item_features": item_features,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
    parser.add_argument("--data-dir",   default="data/raw",       help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed",  help="Output directory")
    parser.add_argument("--test-days",  default=30, type=int,      help="Days in test set")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_days=args.test_days,
    )
