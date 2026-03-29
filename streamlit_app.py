import streamlit as st
import pandas as pd

from bias_analysis import (
    add_derived_columns,
    group_bias,
    multi_dim_bias,
    flag_high_risk_groups,
)

st.set_page_config(page_title="Advisor Commission Bias Review", layout="wide")

st.title("Advisor Commission Bias Review")

st.markdown(
    """
Upload the advisor commission dataset (Excel) to explore potential bias across
gender, segment, region, tenure, and education.
"""
)

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload the `advisor_commission_bias_datasource.xlsx` file to begin.")
    st.stop()

# Load data
df = pd.read_excel(uploaded_file)
df.columns = [c.strip() for c in df.columns]
df = add_derived_columns(df)

st.subheader("Raw data preview")
st.dataframe(df.head(20))

st.markdown("---")

# Sidebar controls
st.sidebar.header("Bias analysis controls")

feature = st.sidebar.selectbox(
    "Single feature to analyze",
    options=["Gender", "Client_Segment", "Region", "Education", "Age_Group"],
    index=0,
)

combo_features = st.sidebar.multiselect(
    "Multi-dimensional combinations",
    options=["Gender", "Client_Segment", "Region", "Education", "Age_Group"],
    default=["Gender", "Client_Segment"],
)

pct_threshold = st.sidebar.slider(
    "High-risk bias threshold (Avg_Bias_Pct ≤)",
    min_value=-60,
    max_value=0,
    value=-15,
    step=1,
)

min_count = st.sidebar.slider(
    "Minimum group size",
    min_value=3,
    max_value=30,
    value=5,
    step=1,
)

# --- Single-feature bias ---
st.subheader(f"Bias by {feature}")
single_agg = group_bias(df, feature)
st.dataframe(single_agg.style.background_gradient(subset=["Avg_Bias_Pct"], cmap="RdYlGn"))

high_risk_single = flag_high_risk_groups(single_agg, pct_threshold=pct_threshold, min_count=min_count)

with st.expander("High-risk groups (single feature)", expanded=True):
    if high_risk_single.empty:
        st.write("No groups meet the high-risk criteria with current thresholds.")
    else:
        st.dataframe(high_risk_single)

# --- Multi-dimensional bias ---
if combo_features:
    st.subheader(f"Bias by combination: {', '.join(combo_features)}")
    multi_agg = multi_dim_bias(df, combo_features)
    st.dataframe(multi_agg.style.background_gradient(subset=["Avg_Bias_Pct"], cmap="RdYlGn"))

    high_risk_multi = flag_high_risk_groups(multi_agg, pct_threshold=pct_threshold, min_count=min_count)

    with st.expander("High-risk groups (combinations)", expanded=False):
        if high_risk_multi.empty:
            st.write("No combinations meet the high-risk criteria with current thresholds.")
        else:
            st.dataframe(high_risk_multi)

# --- Quick visuals ---
st.markdown("---")
st.subheader("Quick visuals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Bias % by Gender**")
    gender_agg = group_bias(df, "Gender")
    st.bar_chart(
        gender_agg.set_index("Gender")["Avg_Bias_Pct"],
        use_container_width=True,
    )

with col2:
    st.markdown("**Average Bias % by Client Segment**")
    seg_agg = group_bias(df, "Client_Segment")
    st.bar_chart(
        seg_agg.set_index("Client_Segment")["Avg_Bias_Pct"],
        use_container_width=True,
    )

st.markdown(
    """
**Interpretation hint:**  
Negative `Avg_Bias_Pct` and large negative `Avg_Bias_Gap_USD` indicate groups whose actual commissions
are systematically below model-predicted levels—potential bias against those advisors.
"""
)
