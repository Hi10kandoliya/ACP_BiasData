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

# ── Column mapping: Excel sheet → internal names ──────────────────────────
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


# ── Data loader ────────────────────────────────────────────────────────────

def load_datasource(path: str) -> dict[str, pd.DataFrame]:
    """Load all four sheets from the Excel datasource."""
    xl = pd.ExcelFile(path)
    sheets = {}

    # Sheet 1: Main advisor dataset
    df = pd.read_excel(xl, sheet_name="Advisor_Dataset")
    df = df.rename(columns=COLUMN_MAP)
    df.columns = [c.lower() for c in df.columns]
    # Normalise string categories
    for col in ("gender", "region", "client_segment", "education", "age_group", "feedback_flag"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    sheets["advisors"] = df

    # Sheet 3: Feedback log
    fb = pd.read_excel(xl, sheet_name="Feedback_Log")
    fb.columns = [c.strip() for c in fb.columns]
    sheets["feedback"] = fb

    # Sheet 2: Bias summary (pre-computed)
    sheets["bias_summary_raw"] = pd.read_excel(xl, sheet_name="Bias_Summary")

    # Sheet 4: Assumptions reference
    sheets["assumptions"] = pd.read_excel(xl, sheet_name="Bias_Assumptions")

    return sheets


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class BiasAuditReport:
    audit_date: str
    group_metrics: dict
    flagged_segments: list[str]
    recommendations: list[str]


@dataclass
class CommissionPrediction:
    advisor_id: str
    predicted_commission: float
    confidence_interval: tuple[float, float]
    is_blinded: bool = False
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


# ── Strategy 1: Bias Auditor ───────────────────────────────────────────────

class BiasAuditor:
    def __init__(self, threshold_mae_ratio: float = 1.20):
        self.threshold_mae_ratio = threshold_mae_ratio
        self.audit_history: list[BiasAuditReport] = []

    def run_audit(
        self,
        advisors_df: pd.DataFrame,
        predictions: pd.Series,
        actuals: pd.Series,
        group_columns: list[str] = None,
    ) -> BiasAuditReport:
        if group_columns is None:
            group_columns = ["gender", "region", "client_segment"]

        predictions = predictions.reset_index(drop=True)
        actuals     = actuals.reset_index(drop=True)
        advisors_df = advisors_df.reset_index(drop=True)

        errors = (predictions - actuals).abs()
        group_metrics: dict = {}
        flagged_segments: list[str] = []

        for col in group_columns:
            if col not in advisors_df.columns:
                continue
            col_metrics: dict = {}
            for grp, idx in advisors_df.groupby(col).groups.items():
                valid_idx = [i for i in idx if i < len(errors)]
                if valid_idx:
                    col_metrics[str(grp)] = round(errors.iloc[valid_idx].mean(), 2)
            group_metrics[col] = col_metrics

        for col, metrics in group_metrics.items():
            if not metrics:
                continue
            best = min(metrics.values())
            for grp, mae in metrics.items():
                if best > 0 and (mae / best) >= self.threshold_mae_ratio:
                    flagged_segments.append(
                        f"{col}={grp}  (MAE ratio: {mae/best:.2f}×,  MAE=${mae:,.0f})"
                    )

        recommendations = self._recommendations(flagged_segments)
        report = BiasAuditReport(
            audit_date=datetime.date.today().isoformat(),
            group_metrics=group_metrics,
            flagged_segments=flagged_segments,
            recommendations=recommendations,
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
            recs.append(
                "Middle-Income segment flagged — enrich features to capture "
                "long-term relationship value beyond raw AUM."
            )
        if any("gender" in f for f in flagged):
            recs.append(
                "Gender bias detected — audit historical training labels for "
                "systematic under-recording of women/non-binary advisor outcomes."
            )
        return recs


# ── Strategy 2: Fair Commission Model ─────────────────────────────────────

class FairCommissionModel:
    def __init__(self, protected_attributes: list[str] = None):
        self.protected_attributes = protected_attributes or ["gender", "region", "tenure_bucket"]
        self.base_model = LinearRegression()
        self.scaler = StandardScaler()
        self.group_corrections: dict[str, float] = {}
        self.is_trained = False
        self._feature_cols: list[str] = []

    def _tenure_bucket(self, df: pd.DataFrame) -> pd.Series:
        return pd.cut(
            df["tenure_years"],
            bins=[0, 2, 5, 10, np.inf],
            labels=["0-2yr", "2-5yr", "5-10yr", "10+yr"],
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FairCommissionModel":
        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        drop_cols = [c for c in self.protected_attributes + ["tenure_bucket"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_enc = pd.get_dummies(df[feature_cols], drop_first=True)
        self._feature_cols = X_enc.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_enc)
        self.base_model.fit(X_scaled, y)
        raw_preds = self.base_model.predict(X_scaled)
        residuals = y.values - raw_preds
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            for grp, idx in df.groupby(attr).groups.items():
                self.group_corrections[f"{attr}={grp}"] = float(residuals[idx].mean())
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_trained, "Call .fit() first."
        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        drop_cols = [c for c in self.protected_attributes + ["tenure_bucket"] if c in df.columns]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_enc = pd.get_dummies(df[feature_cols], drop_first=True)
        X_enc = X_enc.reindex(columns=self._feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(X_enc)
        preds = self.base_model.predict(X_scaled).copy()
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            for grp, idx in df.groupby(attr).groups.items():
                key = f"{attr}={grp}"
                if key in self.group_corrections:
                    preds[idx] += self.group_corrections[key]
        return np.clip(preds, 0, None)

    def fairness_report(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds = self.predict(X)
        df = X.copy()
        df["tenure_bucket"] = self._tenure_bucket(df)
        residuals = y.values - preds
        report = {}
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
            report[attr] = {
                str(grp): round(residuals[idx].mean(), 2)
                for grp, idx in df.groupby(attr).groups.items()
            }
        return report

    def corrections_df(self) -> pd.DataFrame:
        rows = []
        for key, val in self.group_corrections.items():
            attr, grp = key.split("=", 1)
            rows.append({"Attribute": attr, "Group": grp,
                         "Correction ($)": round(val, 2),
                         "Direction": "↑ Boost" if val > 0 else "↓ Reduce"})
        return pd.DataFrame(rows)


# ── Strategy 3: Feedback System ────────────────────────────────────────────

class AdvisorFeedbackSystem:
    RETRAIN_THRESHOLD = 5

    def __init__(self, existing_feedback: pd.DataFrame = None):
        self.feedback_log: list[dict] = []
        self.retraining_triggered = False
        # Pre-load feedback from Excel sheet if provided
        if existing_feedback is not None and not existing_feedback.empty:
            self._load_existing(existing_feedback)

    def _load_existing(self, fb: pd.DataFrame):
        for _, row in fb.iterrows():
            self.feedback_log.append({
                "advisor_id":   str(row.get("Advisor_ID", "")),
                "predicted":    float(row.get("Predicted_Commission", 0)),
                "actual":       float(row.get("Actual_Commission", 0)),
                "discrepancy_pct": float(row.get("Bias_Pct", 0)),
                "client_satisfaction": float(row.get("Satisfaction_Score", 0)),
                "severity":     str(row.get("Severity", "LOW")),
                "notes":        str(row.get("Notes", "")),
                "status":       str(row.get("Status", "Open")),
                "timestamp":    str(row.get("Quarter", "")),
            })
        self._check_trigger()

    def submit_feedback(self, advisor_id, predicted, actual, satisfaction, notes="") -> dict:
        disc_pct = abs(actual - predicted) / max(actual, 1) * 100
        severity = "HIGH" if disc_pct > 30 else "MEDIUM" if disc_pct > 15 else "LOW"
        record = {
            "advisor_id": advisor_id,
            "predicted":  round(predicted, 2),
            "actual":     round(actual, 2),
            "discrepancy_pct": round(disc_pct, 2),
            "client_satisfaction": satisfaction,
            "severity":   severity,
            "notes":      notes,
            "status":     "Open",
            "timestamp":  datetime.datetime.now().isoformat(),
        }
        self.feedback_log.append(record)
        self._check_trigger()
        return record

    def _check_trigger(self):
        high = sum(1 for r in self.feedback_log if r["severity"] == "HIGH")
        if high >= self.RETRAIN_THRESHOLD:
            self.retraining_triggered = True

    def summary(self) -> dict:
        df = pd.DataFrame(self.feedback_log)
        if df.empty:
            return {"total_reports": 0}
        return {
            "total_reports": len(df),
            "severity_counts": df["severity"].value_counts().to_dict(),
            "avg_discrepancy_pct": round(df["discrepancy_pct"].mean(), 2),
            "high_satisfaction_underpredicted": int(
                ((df.get("client_satisfaction", pd.Series(dtype=float)) >= 8) &
                 (df["actual"] > df["predicted"])).sum()
            ),
            "retraining_triggered": self.retraining_triggered,
        }

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.feedback_log)


# ── Strategy 4: Blind Pilot ────────────────────────────────────────────────

class BlindPilotEngine:
    def __init__(self, pilot_active: bool = True, reveal_after_days: int = 90):
        self.pilot_active = pilot_active
        self.reveal_after_days = reveal_after_days
        self._log: list[CommissionPrediction] = []

    def predict(self, advisor_id: str, model: FairCommissionModel,
                X_row: pd.DataFrame) -> CommissionPrediction:
        raw = model.predict(X_row)[0]
        std = raw * 0.10
        ci  = (round(raw - std, 2), round(raw + std, 2))
        pred = CommissionPrediction(
            advisor_id=advisor_id,
            predicted_commission=round(raw, 2),
            confidence_interval=ci,
            is_blinded=self.pilot_active,
        )
        self._log.append(pred)
        return pred

    def end_pilot(self) -> pd.DataFrame:
        self.pilot_active = False
        rows = [{
            "advisor_id":   p.advisor_id,
            "predicted":    p.predicted_commission,
            "ci_lower":     p.confidence_interval[0],
            "ci_upper":     p.confidence_interval[1],
            "timestamp":    p.timestamp,
        } for p in self._log]
        return pd.DataFrame(rows)

    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "advisor_id": p.advisor_id,
            "blinded":    p.is_blinded,
            "predicted":  p.predicted_commission if not p.is_blinded else "HIDDEN",
            "timestamp":  p.timestamp,
        } for p in self._log])


# ── Pipeline: load data → train → evaluate ────────────────────────────────

def build_pipeline(excel_path: str):
    sheets   = load_datasource(excel_path)
    df       = sheets["advisors"]

    # Ensure numeric columns are correct dtype
    for col in ["tenure_years", "aum", "n_clients", "satisfaction",
                "product_diversity", "retention_rate", "new_clients_ytd",
                "compliance_score", "actual_commission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    df_test = df.iloc[X_test.index].reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    model = FairCommissionModel(
        protected_attributes=["gender", "region", "tenure_bucket"]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    auditor = BiasAuditor(threshold_mae_ratio=1.20)
    report  = auditor.run_audit(
        advisors_df=df_test,
        predictions=pd.Series(preds),
        actuals=y_test,
        group_columns=["gender", "region", "client_segment"],
    )

    feedback = AdvisorFeedbackSystem(existing_feedback=sheets["feedback"])

    pilot = BlindPilotEngine(pilot_active=True, reveal_after_days=90)
    pilot_sample = df_test.sample(min(10, len(df_test)), random_state=99).reset_index(drop=True)
    for _, row in pilot_sample.iterrows():
        pilot.predict(row["advisor_id"], model, pd.DataFrame([row[FEATURES]]))

    return {
        "df":          df,
        "df_test":     df_test,
        "X_test":      X_test,
        "y_test":      y_test,
        "preds":       preds,
        "mae":         mae,
        "model":       model,
        "auditor":     auditor,
        "report":      report,
        "feedback":    feedback,
        "pilot":       pilot,
        "sheets":      sheets,
    }
