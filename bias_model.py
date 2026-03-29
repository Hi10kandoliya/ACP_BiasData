import pandas as pd
import numpy as np
import difflib

# Expected canonical column names
EXPECTED_COLS = {
    "historical_commission": ["historical_commission", "historical commission"],
    "actual_commission": ["actual_commission", "actual commission"],
    "model_predicted": ["model_predicted", "model predicted"],
    "bias_gap_usd": ["bias_gap_usd", "bias gap usd"],
    "bias_pct": ["bias_pct", "bias percent", "bias percentage"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    clean = (
        df.columns
        .str.strip()
        .str.replace(r"[\n\r\t]+", "", regex=True)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace(" ", "_")
        .str.lower()
    )
    df.columns = clean
    return df


def fuzzy_map_columns(df: pd.DataFrame) -> dict:
    mapped = {}
    for canonical, variants in EXPECTED_COLS.items():
        candidates = df.columns.tolist()
        match = difflib.get_close_matches(canonical, candidates, n=1, cutoff=0.6)
        if match:
            mapped[canonical] = match[0]
            continue

        # Try variant list
        for v in variants:
            match = difflib.get_close_matches(v, candidates, n=1, cutoff=0.6)
            if match:
                mapped[canonical] = match[0]
                break

    return mapped


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    mapped = fuzzy_map_columns(df)

    missing = [c for c in EXPECTED_COLS if c not in mapped]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Detected columns: {df.columns.tolist()}"
        )

    ac = mapped["actual_commission"]
    mp = mapped["model_predicted"]
    hc = mapped["historical_commission"]

    df["delta_actual_vs_model"] = df[ac] - df[mp]
    df["delta_actual_vs_historical"] = df[ac] - df[hc]

    if "bias_gap_usd" not in mapped:
        df["bias_gap_usd"] = df["delta_actual_vs_model"]

    if "bias_pct" not in mapped:
        df["bias_pct"] = 100 * df["delta_actual_vs_model"] / df[mp]

    return df


def group_bias(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature not in df.columns:
        return pd.DataFrame()

    agg = (
        df.groupby(feature)
        .agg(
            count=("advisor_id", "count"),
            avg_actual=("actual_commission", "mean"),
            avg_pred=("model_predicted", "mean"),
            avg_gap=("bias_gap_usd", "mean"),
            avg_pct=("bias_pct", "mean"),
        )
        .reset_index()
    )

    best = agg["avg_pct"].max()
    agg["gap_vs_best"] = agg["avg_pct"] - best
    return agg.sort_values("avg_pct")
