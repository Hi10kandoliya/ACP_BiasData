import pandas as pd
import numpy as np

BIAS_FEATURES = ["Gender", "Client_Segment", "Region", "Education", "Tenure_Years"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Standardize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Commission deltas
    df["Delta_Actual_vs_Model"] = df["Actual_Commission"] - df["Model_Predicted"]
    df["Delta_Actual_vs_Historical"] = df["Actual_Commission"] - df["Historical_Commission"]

    # Normalize by model to get % deviation if not already present
    if "Bias_Pct" not in df.columns:
        df["Bias_Pct"] = np.where(
            df["Model_Predicted"] != 0,
            100 * (df["Actual_Commission"] - df["Model_Predicted"]) / df["Model_Predicted"],
            np.nan,
        )

    if "Bias_Gap_USD" not in df.columns:
        df["Bias_Gap_USD"] = df["Actual_Commission"] - df["Model_Predicted"]

    return df


def group_bias(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Aggregate bias metrics by a single feature (e.g., Gender, Region).
    """
    agg = (
        df.groupby(feature)
        .agg(
            Count=("Advisor_ID", "count"),
            Avg_Actual_Commission=("Actual_Commission", "mean"),
            Avg_Model_Predicted=("Model_Predicted", "mean"),
            Avg_Bias_Gap_USD=("Bias_Gap_USD", "mean"),
            Avg_Bias_Pct=("Bias_Pct", "mean"),
        )
        .reset_index()
    )

    # Relative disadvantage vs best group (higher Avg_Bias_Pct is better)
    max_bias = agg["Avg_Bias_Pct"].max()
    agg["Gap_vs_Best_PctPts"] = agg["Avg_Bias_Pct"] - max_bias

    return agg.sort_values("Avg_Bias_Pct")


def multi_dim_bias(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Look at combinations (e.g., Gender + Client_Segment).
    """
    agg = (
        df.groupby(features)
        .agg(
            Count=("Advisor_ID", "count"),
            Avg_Bias_Gap_USD=("Bias_Gap_USD", "mean"),
            Avg_Bias_Pct=("Bias_Pct", "mean"),
        )
        .reset_index()
    )
    max_bias = agg["Avg_Bias_Pct"].max()
    agg["Gap_vs_Best_PctPts"] = agg["Avg_Bias_Pct"] - max_bias
    return agg.sort_values("Avg_Bias_Pct")


def flag_high_risk_groups(agg_df: pd.DataFrame, pct_threshold: float = -15, min_count: int = 5) -> pd.DataFrame:
    """
    Identify groups with strong negative bias and enough observations.
    """
    mask = (agg_df["Avg_Bias_Pct"] <= pct_threshold) & (agg_df["Count"] >= min_count)
    return agg_df[mask].copy()


def summarize_feedback_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    How 'Flag for Review' is distributed across protected attributes.
    """
    if "Feedback_Flag" not in df.columns:
        return pd.DataFrame()

    flagged = df[df["Feedback_Flag"].str.contains("Flag", case=False, na=False)]
    if flagged.empty:
        return pd.DataFrame()

    summaries = []
    for feature in ["Gender", "Client_Segment", "Region", "Education"]:
        tmp = (
            flagged.groupby(feature)
            .agg(Flagged_Count=("Advisor_ID", "count"))
            .reset_index()
        )
        tmp["Feature"] = feature
        summaries.append(tmp)

    out = pd.concat(summaries, ignore_index=True)
    return out[["Feature", feature, "Flagged_Count"]]
