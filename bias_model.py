"""
bias_model.py — Advisor Commission Bias Mitigation Model
Loads data from advisor_commission_bias_datasource.xlsx
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field
import datetime
import warnings

warnings.filterwarnings("ignore")

COLUMN_MAP = {
    "Advisor_ID":            "advisor_id",
    "Gender":                "gender",
    "Age_Group":             "age_group",
    "Education":             "education",
    "Region":                "region",
    "Tenure_Years":          "tenure_years",
    "Client_Segment":        "client_segment",
    "AUM_USD":               "aum",
    "Num_Clients":           "n_clients",
    "Satisfaction_Score":    "satisfaction",
    "Product_Diversity":     "product_diversity",
    "Client_Retention_Rate": "retention_rate",
    "New_Clients_YTD":       "new_clients_ytd",
    "Compliance_Score":      "compliance_score",
    "Quarter":               "quarter",
    "Historical_Commission": "historical_commission",
    "Actual_Commission":     "actual_commission",
    "Model_Predicted":       "model_predicted",
    "Bias_Gap_USD":          "bias_gap",
    "Bias_Pct":              "bias_pct",
    "Feedback_Flag":         "feedback_flag",
}

FEATURES = [
    "gender", "region", "client_segment", "tenure_years",
    "aum", "n_clients", "satisfaction", "product_diversity",
    "retention_rate", "new_clients_ytd", "compliance_score",
]
TARGET = "actual_commission"

NUMERIC_COLS = [
    "tenure_years", "aum", "n_clients", "satisfaction",
    "product_diversity", "retention_rate", "new_clients_ytd",
    "compliance_score", "actual_commission", "historical_commission",
    "model_predicted", "bias_gap", "bias_pct",
]
STRING_COLS = ["gender", "region", "client_segment", "education",
               "age_group", "feedback_flag", "quarter", "advisor_id"]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_datasource(path: str) -> dict:
    """Load all four sheets. Real headers are on Excel row 4 (header=3, 0-indexed)."""
    xl = pd.ExcelFile(path)
    sheets = {}

    df = pd.read_excel(xl, sheet_name="Advisor_Dataset", header=3)
    df = _clean_df(df)

    # Fallback for plain files without banner rows
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        df2 = pd.read_excel(xl, sheet_name="Advisor_Dataset", header=0)
        df2 = _clean_df(df2)
        if len([c for c in FEATURES + [TARGET] if c not in df2.columns]) < len(missing):
            df = df2

    sheets["advisors"] = df

    try:
        fb = pd.read_excel(xl, sheet_name="Feedback_Log", header=3)
        fb = fb.dropna(how="all", axis=0).dropna(how="all", axis=1)
        fb.columns = [str(c).strip() for c in fb.columns]
    except Exception:
        fb = pd.DataFrame()
    sheets["feedback"] = fb

    try:
        sheets["bias_summary_raw"] = pd.read_excel(xl, sheet_name="Bias_Summary", header=3)
    except Exception:
        sheets["bias_summary_raw"] = pd.DataFrame()

    try:
        sheets["assumptions"] = pd.read_excel(xl, sheet_name="Bias_Assumptions", header=3)
    except Exception:
        sheets["assumptions"] = pd.DataFrame()

    return sheets


@dataclass
class BiasAuditReport:
    audit_date: str
    group_metrics: dict
    flagged_segments: list
    recommendations: list


@dataclass
class CommissionPrediction:
    advisor_id: str
    predicted_commission: float
    confidence_interval: tuple
    is_blinded: bool = False
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class BiasAuditor:
    def __init__(self, threshold_mae_ratio: float = 1.20):
        self.threshold_mae_ratio = threshold_mae_ratio
        self.audit_history = []

    def run_audit(self, advisors_df, predictions, actuals, group_columns=None):
        if group_columns is None:
            group_columns = ["gender", "region", "client_segment"]
        # Always work with positional (0-based) indices
        predictions = predictions.reset_index(drop=True)
        actuals     = actuals.reset_index(drop=True)
        advisors_df = advisors_df.reset_index(drop=True)
        errors = (predictions - actuals).abs()
        group_metrics, flagged_segments = {}, []

        for col in group_columns:
            if col not in advisors_df.columns:
                continue
            col_metrics = {}
            for grp, label_idx in advisors_df.groupby(col).groups.items():
                # label_idx are positional after reset_index
                valid = [i for i in label_idx if i < len(errors)]
                if valid:
                    col_metrics[str(grp)] = round(float(errors.iloc[valid].mean()), 2)
            group_metrics[col] = col_metrics

        for col, metrics in group_metrics.items():
            if not metrics:
                continue
            best = min(metrics.values())
            for grp, mae in metrics.items():
                if best > 0 and (mae / best) >= self.threshold_mae_ratio:
                    flagged_segments.append(
                        f"{col}={grp}  (MAE ratio: {mae/best:.2f}x,  MAE=${mae:,.0f})"
                    )

        recs = self._recommendations(flagged_segments)
        report = BiasAuditReport(
            audit_date=datetime.date.today().isoformat(),
            group_metrics=group_metrics,
            flagged_segments=flagged_segments,
            recommendations=recs,
        )
        self.audit_history.append(report)
        return report

    def _recommendations(self, flagged):
        if not flagged:
            return ["No significant bias detected. Continue monitoring next quarter."]
        recs = [
            "Investigate feature distributions for flagged groups.",
            "Collect additional data for underrepresented segments.",
            "Apply sample reweighting or fairness constraints in the next training cycle.",
        ]
        if any("client_segment" in f for f in flagged):
            recs.append("Middle-Income segment flagged — enrich features to capture long-term relationship value beyond raw AUM.")
        if any("gender" in f for f in flagged):
            recs.append("Gender bias detected — audit historical training labels for systematic under-recording of women/non-binary advisor outcomes.")
        return recs


class FairCommissionModel:
    def __init__(self, protected_attributes=None):
        self.protected_attributes = protected_attributes or ["gender", "region", "tenure_bucket"]
        self.base_model = LinearRegression()
        self.scaler = StandardScaler()
        self.group_corrections = {}
        self.is_trained = False
        self._feature_cols = []

    def _tenure_bucket(self, df):
        return pd.cut(df["tenure_years"], bins=[0, 2, 5, 10, np.inf],
                      labels=["0-2yr", "2-5yr", "5-10yr", "10+yr"])

    def fit(self, X, y):
        # CRITICAL: reset index so positional indexing on residuals is safe
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        drop_cols = [c for c in self.protected_attributes + ["tenure_bucket"] if c in df.columns]
        feat_cols = [c for c in df.columns if c not in drop_cols]
        X_enc = pd.get_dummies(df[feat_cols], drop_first=True)
        self._feature_cols = X_enc.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_enc)
        self.base_model.fit(X_scaled, y)
        raw_preds = self.base_model.predict(X_scaled)
        residuals = y.values - raw_preds   # numpy array, positional

        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            for grp, pos_idx in df.groupby(attr).groups.items():
                # pos_idx are positional after reset_index — safe for numpy
                self.group_corrections[f"{attr}={grp}"] = float(
                    residuals[pos_idx].mean()
                )
        self.is_trained = True
        return self

    def predict(self, X):
        assert self.is_trained, "Call .fit() first."
        X = X.reset_index(drop=True)
        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        drop_cols = [c for c in self.protected_attributes + ["tenure_bucket"] if c in df.columns]
        feat_cols = [c for c in df.columns if c not in drop_cols]
        X_enc = pd.get_dummies(df[feat_cols], drop_first=True)
        X_enc = X_enc.reindex(columns=self._feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(X_enc)
        preds = self.base_model.predict(X_scaled).copy()
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            for grp, pos_idx in df.groupby(attr).groups.items():
                key = f"{attr}={grp}"
                if key in self.group_corrections:
                    preds[pos_idx] += self.group_corrections[key]
        return np.clip(preds, 0, None)

    def fairness_report(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        preds = self.predict(X)
        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        residuals = y.values - preds
        report = {}
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            report[attr] = {
                str(grp): round(float(residuals[pos_idx].mean()), 2)
                for grp, pos_idx in df.groupby(attr).groups.items()
            }
        return report

    def corrections_df(self):
        rows = []
        for key, val in self.group_corrections.items():
            attr, grp = key.split("=", 1)
            rows.append({"Attribute": attr, "Group": grp,
                         "Correction ($)": round(val, 2),
                         "Direction": "↑ Boost" if val > 0 else "↓ Reduce"})
        return pd.DataFrame(rows)


class AdvisorFeedbackSystem:
    RETRAIN_THRESHOLD = 5

    def __init__(self, existing_feedback=None):
        self.feedback_log = []
        self.retraining_triggered = False
        if existing_feedback is not None and not existing_feedback.empty:
            self._load_existing(existing_feedback)

    def _load_existing(self, fb):
        for _, row in fb.iterrows():
            try:
                self.feedback_log.append({
                    "advisor_id":          str(row.get("Advisor_ID", "")),
                    "predicted":           float(row.get("Predicted_Commission", 0) or 0),
                    "actual":              float(row.get("Actual_Commission", 0) or 0),
                    "discrepancy_pct":     float(row.get("Bias_Pct", 0) or 0),
                    "client_satisfaction": float(row.get("Satisfaction_Score", 0) or 0),
                    "severity":            str(row.get("Severity", "LOW")),
                    "notes":               str(row.get("Notes", "")),
                    "status":              str(row.get("Status", "Open")),
                    "timestamp":           str(row.get("Quarter", "")),
                })
            except Exception:
                continue
        self._check_trigger()

    def submit_feedback(self, advisor_id, predicted, actual, satisfaction, notes=""):
        disc = abs(actual - predicted) / max(actual, 1) * 100
        severity = "HIGH" if disc > 30 else "MEDIUM" if disc > 15 else "LOW"
        record = {
            "advisor_id": advisor_id, "predicted": round(predicted, 2),
            "actual": round(actual, 2), "discrepancy_pct": round(disc, 2),
            "client_satisfaction": satisfaction, "severity": severity,
            "notes": notes, "status": "Open",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.feedback_log.append(record)
        self._check_trigger()
        return record

    def _check_trigger(self):
        high = sum(1 for r in self.feedback_log if r.get("severity") == "HIGH")
        if high >= self.RETRAIN_THRESHOLD:
            self.retraining_triggered = True

    def summary(self):
        df = pd.DataFrame(self.feedback_log)
        if df.empty:
            return {"total_reports": 0, "retraining_triggered": False,
                    "severity_counts": {}, "avg_discrepancy_pct": 0.0,
                    "high_satisfaction_underpredicted": 0}
        sat = df.get("client_satisfaction", pd.Series(dtype=float))
        return {
            "total_reports": len(df),
            "severity_counts": df["severity"].value_counts().to_dict(),
            "avg_discrepancy_pct": round(float(df["discrepancy_pct"].mean()), 2),
            "high_satisfaction_underpredicted": int(
                ((sat >= 8) & (df["actual"] > df["predicted"])).sum()),
            "retraining_triggered": self.retraining_triggered,
        }

    def as_dataframe(self):
        return pd.DataFrame(self.feedback_log)


class BlindPilotEngine:
    def __init__(self, pilot_active=True, reveal_after_days=90):
        self.pilot_active = pilot_active
        self.reveal_after_days = reveal_after_days
        self._log = []

    def predict(self, advisor_id, model, X_row):
        raw = float(model.predict(X_row)[0])
        ci  = (round(raw * 0.90, 2), round(raw * 1.10, 2))
        pred = CommissionPrediction(
            advisor_id=advisor_id,
            predicted_commission=round(raw, 2),
            confidence_interval=ci,
            is_blinded=self.pilot_active,
        )
        self._log.append(pred)
        return pred

    def end_pilot(self):
        self.pilot_active = False
        return pd.DataFrame([{
            "advisor_id": p.advisor_id, "predicted": p.predicted_commission,
            "ci_lower": p.confidence_interval[0], "ci_upper": p.confidence_interval[1],
            "timestamp": p.timestamp,
        } for p in self._log])

    def log_df(self):
        if not self._log:
            return pd.DataFrame(columns=["advisor_id", "blinded", "predicted", "timestamp"])
        return pd.DataFrame([{
            "advisor_id": p.advisor_id, "blinded": p.is_blinded,
            "predicted": "HIDDEN" if p.is_blinded else p.predicted_commission,
            "timestamp": p.timestamp,
        } for p in self._log])


def build_pipeline(excel_path: str) -> dict:
    sheets = load_datasource(excel_path)
    df     = sheets["advisors"]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns not found: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Capture test set positions BEFORE any reset
    test_pos = X_test.index.tolist()
    df_test  = df.iloc[test_pos].reset_index(drop=True)
    X_test   = X_test.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)
    X_train  = X_train.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)

    model = FairCommissionModel(protected_attributes=["gender", "region", "tenure_bucket"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = float(mean_absolute_error(y_test, preds))

    auditor = BiasAuditor(threshold_mae_ratio=1.20)
    report  = auditor.run_audit(
        advisors_df=df_test,
        predictions=pd.Series(preds),
        actuals=y_test,
        group_columns=["gender", "region", "client_segment"],
    )

    feedback = AdvisorFeedbackSystem(existing_feedback=sheets["feedback"])

    pilot  = BlindPilotEngine(pilot_active=True, reveal_after_days=90)
    sample = df_test.sample(min(10, len(df_test)), random_state=99).reset_index(drop=True)
    for _, row in sample.iterrows():
        pilot.predict(str(row.get("advisor_id", "UNKNOWN")), model,
                      pd.DataFrame([row[FEATURES]]))

    return {
        "df": df, "df_test": df_test, "X_test": X_test,
        "y_test": y_test, "preds": preds, "mae": mae,
        "model": model, "auditor": auditor, "report": report,
        "feedback": feedback, "pilot": pilot, "sheets": sheets,
    }
