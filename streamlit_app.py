"""
streamlit_app.py — Advisor Commission Bias Mitigation Dashboard
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os

from bias_model import (
    build_pipeline, AdvisorFeedbackSystem,
    BlindPilotEngine, FEATURES,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commission Bias Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLOURS
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = "#1B2A4A"
TEAL   = "#0D7377"
GOLD   = "#F2A900"
RED    = "#C0392B"
GREEN  = "#1E8449"
ORANGE = "#E67E22"
PURPLE = "#6C3483"
LGRAY  = "#F5F6FA"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=DM+Mono&display=swap');
  html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
  .main {{ background: {LGRAY}; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

  /* Top banner */
  .top-banner {{
    background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 20px;
  }}
  .top-banner h1 {{ color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }}
  .top-banner p  {{ color: rgba(255,255,255,0.8); margin: 4px 0 0; font-size: 0.9rem; }}

  /* KPI cards */
  .kpi-card {{
    background: white; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center;
    border-top: 4px solid {TEAL};
  }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; color: {NAVY}; line-height: 1; }}
  .kpi-label {{ font-size: 0.78rem; color: #666; margin-top: 6px; text-transform: uppercase; letter-spacing: .5px; }}

  /* Section headers */
  .section-header {{
    background: {NAVY}; color: white; border-radius: 8px;
    padding: 10px 18px; margin: 20px 0 12px; font-weight: 600; font-size: 1rem;
  }}

  /* Flag pills */
  .pill-red    {{ background:#FADBD8; color:{RED};    padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }}
  .pill-orange {{ background:#FDEBD0; color:{ORANGE}; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }}
  .pill-green  {{ background:#D5F5E3; color:{GREEN};  padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{ background: {NAVY}; }}
  section[data-testid="stSidebar"] * {{ color: white !important; }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{ color: rgba(255,255,255,0.8) !important; font-size:.85rem; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(v):  return f"${v:,.0f}"
def fmt_pct(v):  return f"{v:+.1f}%"

def bias_color(val):
    if val < -15: return RED
    if val < -5:  return ORANGE
    if val >  15: return ORANGE
    return GREEN

def kpi(label, value, color=TEAL):
    return f"""
    <div class="kpi-card" style="border-top-color:{color}">
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""

def section(title, icon="▶"):
    st.markdown(f'<div class="section-header">{icon} &nbsp; {title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data & training model…")
def load(path):
    return build_pipeline(path)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Commission Bias\nDashboard")
    st.markdown("---")

    uploaded = st.file_uploader("Upload datasource (.xlsx)", type=["xlsx"])
    default_path = "advisor_commission_bias_datasource.xlsx"

    if uploaded:
        tmp = "/tmp/uploaded_datasource.xlsx"
        with open(tmp, "wb") as f:
            f.write(uploaded.read())
        data_path = tmp
        st.success("Custom file loaded ✓")
    elif os.path.exists(default_path):
        data_path = default_path
        st.info(f"Using: {default_path}")
    else:
        st.error("No datasource found. Upload an .xlsx file.")
        st.stop()

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Overview",
        "🔍 Bias Audit",
        "⚖️ Fair Model",
        "📝 Feedback Loop",
        "🙈 Blind Pilot",
        "📋 Raw Data",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Audit Settings**")
    mae_threshold = st.slider("MAE Ratio Threshold", 1.10, 1.50, 1.20, 0.05)
    pilot_active  = st.toggle("Pilot Mode Active", value=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
pipeline = load(data_path)
df       = pipeline["df"]
df_test  = pipeline["df_test"]
X_test   = pipeline["X_test"]
y_test   = pipeline["y_test"]
preds    = pipeline["preds"]
mae      = pipeline["mae"]
model    = pipeline["model"]
report   = pipeline["report"]
feedback = pipeline["feedback"]
pilot    = pipeline["pilot"]
sheets   = pipeline["sheets"]

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
  <div style="font-size:2.8rem">⚖️</div>
  <div>
    <h1>Advisor Commission Bias Dashboard</h1>
    <p>Four-strategy bias mitigation framework &nbsp;|&nbsp; {len(df):,} advisors &nbsp;|&nbsp;
       Last audit: {report.audit_date}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    flagged_n = int((df["feedback_flag"] == "Flag for Review").sum())
    fb_summary = feedback.summary()
    high_sev   = fb_summary.get("severity_counts", {}).get("HIGH", 0)

    cols = st.columns(5)
    with cols[0]: st.markdown(kpi("Total Advisors",    f"{len(df):,}",           NAVY),   unsafe_allow_html=True)
    with cols[1]: st.markdown(kpi("Model MAE",         fmt_usd(mae),              TEAL),   unsafe_allow_html=True)
    with cols[2]: st.markdown(kpi("Flagged Advisors",  f"{flagged_n}",            RED),    unsafe_allow_html=True)
    with cols[3]: st.markdown(kpi("Feedback Reports",  f"{fb_summary['total_reports']}", ORANGE), unsafe_allow_html=True)
    with cols[4]: st.markdown(kpi("HIGH Severity",     f"{high_sev}",             PURPLE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section("Avg Bias Gap by Gender", "👤")
        g = df.groupby("gender")["bias_pct"].mean().reset_index()
        g.columns = ["Gender", "Avg Bias %"]
        colors = [RED if v < -5 else (ORANGE if v < 0 else GREEN) for v in g["Avg Bias %"]]
        fig = px.bar(g, x="Gender", y="Avg Bias %", color="Gender",
                     color_discrete_sequence=colors,
                     title="Over-prediction (+) / Under-prediction (−) by Gender")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          yaxis_title="Avg Bias %", xaxis_title="")
        fig.add_hline(y=0, line_dash="dash", line_color=NAVY, line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Avg Bias Gap by Client Segment", "💼")
        s = df.groupby("client_segment")["bias_pct"].mean().reset_index()
        s.columns = ["Segment", "Avg Bias %"]
        fig2 = px.bar(s, x="Segment", y="Avg Bias %", color="Segment",
                      color_discrete_sequence=[RED, TEAL, ORANGE])
        fig2.update_layout(showlegend=False, plot_bgcolor="white",
                           yaxis_title="Avg Bias %", xaxis_title="")
        fig2.add_hline(y=0, line_dash="dash", line_color=NAVY, line_width=1.5)
        st.plotly_chart(fig2, use_container_width=True)

    section("Actual vs Predicted Commission (Test Set)", "📈")
    scatter_df = pd.DataFrame({
        "Actual ($)":    y_test.values,
        "Predicted ($)": preds,
        "Gender":        df_test["gender"].values,
        "Segment":       df_test["client_segment"].values,
    })
    fig3 = px.scatter(scatter_df, x="Actual ($)", y="Predicted ($)",
                      color="Gender", symbol="Segment", opacity=0.65,
                      color_discrete_map={"M": TEAL, "F": RED, "Non-binary": GOLD})
    max_v = max(scatter_df["Actual ($)"].max(), scatter_df["Predicted ($)"].max())
    fig3.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v],
                              mode="lines", name="Perfect prediction",
                              line=dict(color=NAVY, dash="dash", width=1.5)))
    fig3.update_layout(plot_bgcolor="white", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    section("Strategy Framework", "🗺️")
    strategies = [
        ("1", "Bias Audit",       "Quarterly MAE comparison across demographics",         TEAL),
        ("2", "Fairness Model",   "Equalized-odds corrections per protected group",        NAVY),
        ("3", "Feedback Loop",    "Advisor-reported unfair predictions → retraining",      ORANGE),
        ("4", "Blind Pilot",      "Hide commission outputs during pilot to reduce gaming", PURPLE),
    ]
    scols = st.columns(4)
    for col, (num, title, desc, color) in zip(scols, strategies):
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:18px;
                    border-top:4px solid {color};box-shadow:0 2px 8px rgba(0,0,0,.07)">
          <div style="font-size:1.8rem;font-weight:700;color:{color}">S{num}</div>
          <div style="font-weight:600;color:{NAVY};margin:6px 0 4px">{title}</div>
          <div style="font-size:.82rem;color:#555">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BIAS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Bias Audit":
    section("Strategy 1 — Quarterly Bias Audit", "🔍")

    st.markdown(f"""
    <div style="background:white;border-radius:8px;padding:16px 20px;margin-bottom:16px;
                box-shadow:0 2px 8px rgba(0,0,0,.06)">
      <b>Audit date:</b> {report.audit_date} &nbsp;|&nbsp;
      <b>MAE threshold ratio:</b> {mae_threshold}× &nbsp;|&nbsp;
      <b>Flagged segments:</b> {len(report.flagged_segments)}
    </div>""", unsafe_allow_html=True)

    # MAE tables per group
    for group_col, metrics in report.group_metrics.items():
        section(f"MAE by {group_col.replace('_',' ').title()}", "📊")
        best = min(metrics.values())
        rows = []
        for grp, mae_val in sorted(metrics.items(), key=lambda x: x[1]):
            ratio = mae_val / best if best > 0 else 1.0
            rows.append({
                "Group":    grp,
                "MAE ($)":  mae_val,
                "vs Best":  f"{ratio:.2f}×",
                "Status":   "⚠️ FLAGGED" if ratio >= mae_threshold else "✅ OK",
            })
        mdf = pd.DataFrame(rows)
        st.dataframe(
            mdf.style
               .map(lambda r: [
                   f"background-color:#FADBD8;color:{RED};font-weight:600"
                   if "FLAGGED" in str(r.get("Status","")) else ""
               ] * len(r), axis=1)
               .format({"MAE ($)": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

        fig = px.bar(mdf, x="Group", y="MAE ($)", color="Status",
                     color_discrete_map={"⚠️ FLAGGED": RED, "✅ OK": TEAL},
                     title=f"MAE by {group_col}")
        fig.update_layout(showlegend=True, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Flagged list
    section("Flagged Segments", "🚨")
    if report.flagged_segments:
        for seg in report.flagged_segments:
            st.markdown(f'<span class="pill-red">⚠ {seg}</span>&nbsp;', unsafe_allow_html=True)
    else:
        st.success("No segments flagged at current threshold.")

    section("Recommendations", "💡")
    for rec in report.recommendations:
        st.markdown(f"→ {rec}")

    # Bias heatmap
    section("Bias % Heatmap — Gender × Client Segment", "🗺️")
    heat = df.pivot_table(values="bias_pct", index="gender",
                          columns="client_segment", aggfunc="mean").round(2)
    fig_h = px.imshow(heat, color_continuous_scale=["#C0392B", "#FFFFFF", "#1E8449"],
                      color_continuous_midpoint=0, aspect="auto",
                      labels=dict(color="Avg Bias %"))
    fig_h.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_h, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FAIR MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ Fair Model":
    section("Strategy 2 — Equalized-Odds Fair Model", "⚖️")

    st.markdown(f"""
    <div style="background:white;border-radius:8px;padding:16px 20px;margin-bottom:16px;
                box-shadow:0 2px 8px rgba(0,0,0,.06)">
      The model applies <b>per-group post-processing corrections</b> to equalize mean
      signed error across protected attributes (gender, region, tenure).
      &nbsp;|&nbsp; <b>Test MAE: {fmt_usd(mae)}</b>
    </div>""", unsafe_allow_html=True)

    section("Group Bias Corrections Applied", "🔧")
    corr_df = model.corrections_df()
    st.dataframe(
        corr_df.style
               .map(lambda v: f"color:{GREEN};font-weight:600" if "Boost" in str(v)
                         else (f"color:{RED};font-weight:600" if "Reduce" in str(v) else ""),
                         subset=["Direction"])
               .format({"Correction ($)": "${:+,.0f}"}),
        use_container_width=True, hide_index=True,
    )

    fig_corr = px.bar(
        corr_df, x="Group", y="Correction ($)", color="Attribute",
        barmode="group", title="Per-Group Correction Values ($)",
        color_discrete_sequence=[TEAL, NAVY, PURPLE, GOLD],
    )
    fig_corr.add_hline(y=0, line_dash="dash", line_color=NAVY)
    fig_corr.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_corr, use_container_width=True)

    section("Residual Bias After Correction (mean signed error)", "📉")
    fairness = model.fairness_report(X_test, y_test)
    for attr, groups in fairness.items():
        st.markdown(f"**{attr.replace('_',' ').title()}**")
        rows = [{"Group": g, "Mean Signed Error ($)": v,
                 "Bias Direction": "Over-predicted" if v < 0 else "Under-predicted"}
                for g, v in groups.items()]
        fdf = pd.DataFrame(rows)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(fdf.style.format({"Mean Signed Error ($)": "${:+,.0f}"}),
                         use_container_width=True, hide_index=True)
        with c2:
            fig_r = px.bar(fdf, x="Group", y="Mean Signed Error ($)",
                           color="Bias Direction",
                           color_discrete_map={"Over-predicted": RED, "Under-predicted": TEAL})
            fig_r.add_hline(y=0, line_dash="dash", line_color=NAVY)
            fig_r.update_layout(plot_bgcolor="white", showlegend=True)
            st.plotly_chart(fig_r, use_container_width=True)

    section("Predict Commission for a Single Advisor", "🧮")
    with st.form("predict_form"):
        pc1, pc2, pc3 = st.columns(3)
        gender   = pc1.selectbox("Gender",         ["M", "F", "Non-binary"])
        region   = pc2.selectbox("Region",          ["North", "South", "East", "West"])
        segment  = pc3.selectbox("Client Segment",  ["High-Net-Worth", "Middle-Income", "Mixed"])
        pc4, pc5, pc6 = st.columns(3)
        tenure   = pc4.slider("Tenure (years)", 0.5, 30.0, 5.0, 0.5)
        aum      = pc5.number_input("AUM ($)", 10_000, 50_000_000, 500_000, 50_000)
        n_cl     = pc6.slider("# Clients", 5, 200, 50)
        pc7, pc8, pc9 = st.columns(3)
        sat      = pc7.slider("Satisfaction Score", 1.0, 10.0, 7.5, 0.1)
        prod_div = pc8.slider("Product Diversity", 1, 10, 4)
        ret_rate = pc9.slider("Retention Rate", 0.50, 1.00, 0.85, 0.01)
        pc10, pc11, _ = st.columns(3)
        new_cl   = pc10.slider("New Clients YTD", 0, 30, 5)
        comp     = pc11.slider("Compliance Score", 70.0, 100.0, 90.0, 0.5)
        submitted = st.form_submit_button("Predict Commission ▶", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "region": region, "client_segment": segment,
            "tenure_years": tenure, "aum": aum, "n_clients": n_cl,
            "satisfaction": sat, "product_diversity": prod_div,
            "retention_rate": ret_rate, "new_clients_ytd": new_cl,
            "compliance_score": comp,
        }])
        pred_val = model.predict(row)[0]
        ci_lo, ci_hi = pred_val * 0.90, pred_val * 1.10
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(kpi("Predicted Commission", fmt_usd(pred_val), TEAL), unsafe_allow_html=True)
        rc2.markdown(kpi("CI Lower (−10%)", fmt_usd(ci_lo), NAVY),        unsafe_allow_html=True)
        rc3.markdown(kpi("CI Upper (+10%)", fmt_usd(ci_hi), GREEN),        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FEEDBACK LOOP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📝 Feedback Loop":
    section("Strategy 3 — Advisor Feedback Loop", "📝")
    fb_sum = feedback.summary()

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1: st.markdown(kpi("Total Reports",    f"{fb_sum['total_reports']:,}",                              NAVY),   unsafe_allow_html=True)
    with fc2: st.markdown(kpi("HIGH Severity",    f"{fb_sum.get('severity_counts',{}).get('HIGH',0)}",         RED),    unsafe_allow_html=True)
    with fc3: st.markdown(kpi("Avg Discrepancy",  f"{fb_sum.get('avg_discrepancy_pct',0):.1f}%",               ORANGE), unsafe_allow_html=True)
    with fc4: st.markdown(kpi("Retrain Triggered", "YES ⚡" if fb_sum["retraining_triggered"] else "No",
                              RED if fb_sum["retraining_triggered"] else GREEN), unsafe_allow_html=True)

    fdf = feedback.as_dataframe()
    if not fdf.empty:
        section("Severity Distribution", "📊")
        sev_counts = fdf["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig_sev = px.pie(sev_counts, names="Severity", values="Count",
                         color="Severity",
                         color_discrete_map={"HIGH": RED, "MEDIUM": ORANGE, "LOW": TEAL},
                         hole=0.45)
        fig_sev.update_layout(legend_title="Severity")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(fig_sev, use_container_width=True)
        with c2:
            section("Discrepancy % Distribution", "📉")
            fig_disc = px.histogram(fdf, x="discrepancy_pct", nbins=30,
                                    color="severity",
                                    color_discrete_map={"HIGH": RED, "MEDIUM": ORANGE, "LOW": TEAL},
                                    barmode="overlay", opacity=0.75)
            fig_disc.update_layout(plot_bgcolor="white", xaxis_title="Discrepancy %",
                                   yaxis_title="Count")
            st.plotly_chart(fig_disc, use_container_width=True)

        section("Feedback Log", "📋")
        display_cols = ["advisor_id", "predicted", "actual", "discrepancy_pct",
                        "client_satisfaction", "severity", "status", "notes"]
        show_cols = [c for c in display_cols if c in fdf.columns]
        st.dataframe(
            fdf[show_cols].style
               .map(lambda v: f"color:{RED};font-weight:600" if v == "HIGH"
                         else (f"color:{ORANGE};font-weight:600" if v == "MEDIUM" else ""),
                         subset=["severity"] if "severity" in show_cols else [])
               .format({c: "${:,.0f}" for c in ["predicted", "actual"] if c in show_cols}),
            use_container_width=True, hide_index=True,
        )

    section("Submit a New Feedback Report", "✉️")
    with st.form("feedback_form"):
        bf1, bf2 = st.columns(2)
        adv_id   = bf1.text_input("Advisor ID", placeholder="ADV0001")
        adv_pred = bf2.number_input("Model Predicted Commission ($)", 0.0, 2_000_000.0, 50000.0, 1000.0)
        bf3, bf4 = st.columns(2)
        adv_act  = bf3.number_input("Actual Commission ($)", 0.0, 2_000_000.0, 65000.0, 1000.0)
        adv_sat  = bf4.slider("Client Satisfaction Score", 1.0, 10.0, 8.0, 0.1)
        adv_note = st.text_area("Notes", placeholder="Describe why the prediction seems unfair…")
        submit_fb = st.form_submit_button("Submit Feedback ▶", use_container_width=True)

    if submit_fb and adv_id:
        rec = feedback.submit_feedback(adv_id, adv_pred, adv_act, adv_sat, adv_note)
        sev = rec["severity"]
        color = RED if sev == "HIGH" else (ORANGE if sev == "MEDIUM" else GREEN)
        st.markdown(
            f'<span class="pill-{"red" if sev=="HIGH" else ("orange" if sev=="MEDIUM" else "green")}">'
            f'{sev} severity — Discrepancy: {rec["discrepancy_pct"]:.1f}%</span>',
            unsafe_allow_html=True
        )
        if feedback.retraining_triggered:
            st.warning("⚡ Retraining threshold reached! Model retraining pipeline should be initiated.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BLIND PILOT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🙈 Blind Pilot":
    section("Strategy 4 — Blind Pilot (Incentive Separation)", "🙈")

    st.markdown(f"""
    <div style="background:white;border-radius:8px;padding:16px 20px;margin-bottom:16px;
                box-shadow:0 2px 8px rgba(0,0,0,.06)">
      When <b>Pilot Mode is active</b>, commission values are generated internally but
      <b>withheld from advisors</b> to prevent them gaming client interactions based on
      predicted payouts. The full log is revealed when the pilot ends.
      <br><br><b>Current status:</b>&nbsp;
      {'<span style="color:'+RED+';font-weight:700">🔴 PILOT ACTIVE — outputs hidden</span>'
       if pilot_active else
       '<span style="color:'+GREEN+';font-weight:700">🟢 PILOT ENDED — outputs revealed</span>'}
    </div>""", unsafe_allow_html=True)

    pilot_log = pilot.log_df()
    section(f"Pilot Log — {len(pilot_log)} entries", "📋")
    st.dataframe(pilot_log, use_container_width=True, hide_index=True)

    section("Run Blind Prediction for Advisor", "🔮")
    with st.form("pilot_form"):
        pp1, pp2, pp3 = st.columns(3)
        p_adv_id = pp1.text_input("Advisor ID", "ADV0099")
        p_gender = pp2.selectbox("Gender",  ["M", "F", "Non-binary"])
        p_region = pp3.selectbox("Region",  ["North", "South", "East", "West"])
        pp4, pp5, pp6 = st.columns(3)
        p_seg    = pp4.selectbox("Segment", ["High-Net-Worth", "Middle-Income", "Mixed"])
        p_ten    = pp5.slider("Tenure", 0.5, 30.0, 6.0, 0.5)
        p_aum    = pp6.number_input("AUM ($)", 10_000, 50_000_000, 400_000, 50_000)
        pp7, pp8, pp9 = st.columns(3)
        p_nc     = pp7.slider("# Clients", 5, 200, 45)
        p_sat    = pp8.slider("Satisfaction", 1.0, 10.0, 8.0, 0.1)
        p_pd     = pp9.slider("Product Diversity", 1, 10, 5)
        pp10, pp11, _ = st.columns(3)
        p_ret    = pp10.slider("Retention Rate", 0.50, 1.00, 0.88, 0.01)
        p_new    = pp11.slider("New Clients YTD", 0, 30, 8)
        p_comp   = st.slider("Compliance Score", 70.0, 100.0, 92.0, 0.5)
        run_pilot = st.form_submit_button("Generate Blind Prediction ▶", use_container_width=True)

    if run_pilot:
        row = pd.DataFrame([{
            "gender": p_gender, "region": p_region, "client_segment": p_seg,
            "tenure_years": p_ten, "aum": p_aum, "n_clients": p_nc,
            "satisfaction": p_sat, "product_diversity": p_pd,
            "retention_rate": p_ret, "new_clients_ytd": p_new,
            "compliance_score": p_comp,
        }])
        prediction = pilot.predict(p_adv_id, model, row)
        if pilot_active:
            st.info(f"🙈 **Prediction for {p_adv_id} is HIDDEN** during the pilot phase.\n\n"
                    f"The value has been stored internally and will be revealed when the pilot ends.")
        else:
            rc1, rc2, rc3 = st.columns(3)
            rc1.markdown(kpi("Predicted Commission", fmt_usd(prediction.predicted_commission), TEAL), unsafe_allow_html=True)
            rc2.markdown(kpi("CI Lower",             fmt_usd(prediction.confidence_interval[0]), NAVY), unsafe_allow_html=True)
            rc3.markdown(kpi("CI Upper",             fmt_usd(prediction.confidence_interval[1]), GREEN), unsafe_allow_html=True)

    if not pilot_active:
        section("Full Pilot Reveal", "👁️")
        revealed = pilot.end_pilot()
        if not revealed.empty:
            st.dataframe(
                revealed.style.format({"predicted": "${:,.0f}", "ci_lower": "${:,.0f}", "ci_upper": "${:,.0f}"}),
                use_container_width=True, hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Raw Data":
    section("Full Advisor Dataset", "📋")

    col_filter, col_seg, col_quarter = st.columns(3)
    gender_filter  = col_filter.multiselect("Gender",  df["gender"].unique().tolist(),  default=df["gender"].unique().tolist())
    seg_filter     = col_seg.multiselect("Segment",    df["client_segment"].unique().tolist(), default=df["client_segment"].unique().tolist())
    qtr_filter     = col_quarter.multiselect("Quarter", df["quarter"].unique().tolist(), default=df["quarter"].unique().tolist())

    filtered = df[
        df["gender"].isin(gender_filter) &
        df["client_segment"].isin(seg_filter) &
        df["quarter"].isin(qtr_filter)
    ]

    st.markdown(f"**{len(filtered):,} rows** (of {len(df):,} total)")

    display = filtered[[
        "advisor_id", "gender", "region", "client_segment", "tenure_years",
        "aum", "n_clients", "satisfaction", "actual_commission",
        "model_predicted", "bias_pct", "feedback_flag",
    ]].copy()

    st.dataframe(
        display.style
               .map(lambda v: f"color:{RED};font-weight:600" if v == "Flag for Review" else "",
                         subset=["feedback_flag"])
               .map(lambda v: f"color:{RED}" if isinstance(v, float) and v < -15
                         else (f"color:{ORANGE}" if isinstance(v, float) and v < -5 else ""),
                         subset=["bias_pct"])
               .format({"aum": "${:,.0f}", "actual_commission": "${:,.0f}",
                        "model_predicted": "${:,.0f}", "bias_pct": "{:+.1f}%",
                        "tenure_years": "{:.1f}"}),
        use_container_width=True, hide_index=True, height=500,
    )

    section("Bias Assumptions Reference", "📖")
    st.dataframe(sheets["assumptions"], use_container_width=True, hide_index=True)