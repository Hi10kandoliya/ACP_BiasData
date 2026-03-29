import pandas as pd
import numpy as np
import difflib

# Columns we must be able to find
REQUIRED = [
    "Advisor_ID",
    "Actual_Commission",
    "Model_Predicted",
    "Historical_Commission"
]

def load_excel_with_header_detection(uploaded_file):
    raw = pd.read_excel(uploaded_file, header=None, dtype=str)

    header_row = None
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.strip().tolist()
        matches = sum(1 for col in REQUIRED if col in row)
        if matches >= 2:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not detect header row in this file.")

    df = pd.read_excel(uploaded_file, header=header_row)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[\n\r\t]+", "", regex=True)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace(" ", "_")
    )

    return df


def add_derived_columns(df):
    cols = df.columns.tolist()

    def find(col):
        matches = difflib.get_close_matches(col.lower(), [c.lower() for c in cols], n=1, cutoff=0.6)
        if matches:
            idx = [c.lower() for c in cols].index(matches[0])
            return cols[idx]
        return None

    ac = find("actual_commission")
    mp = find("model_predicted")
    hc = find("historical_commission")

    missing = [name for name, val in {
        "Actual_Commission": ac,
        "Model_Predicted": mp,
        "Historical_Commission": hc
    }.items() if val is None]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Delta_Actual_vs_Model"] = df[ac].astype(float) - df[mp].astype(float)
    df["Delta_Actual_vs_Historical"] = df[ac].astype(float) - df[hc].astype(float)

    if "Bias_Gap_USD" not in df.columns:
        df["Bias_Gap_USD"] = df["Delta_Actual_vs_Model"]

    if "Bias_Pct" not in df.columns:
        df["Bias_Pct"] = 100 * df["Delta_Actual_vs_Model"] / df[mp].astype(float)

    return df


def group_bias(df, feature):
    if feature not in df.columns:
        return pd.DataFrame()

    agg = (
        df.groupby(feature)
        .agg(
            Count=("Advisor_ID", "count"),
            Avg_Actual=("Actual_Commission", "mean"),
            Avg_Pred=("Model_Predicted", "mean"),
            Avg_Gap=("Bias_Gap_USD", "mean"),
            Avg_Pct=("Bias_Pct", "mean"),
        )
        .reset_index()
    )

    best = agg["Avg_Pct"].max()
    agg["Gap_vs_Best"] = agg["Avg_Pct"] - best

    return agg.sort_values("Avg_Pct")
