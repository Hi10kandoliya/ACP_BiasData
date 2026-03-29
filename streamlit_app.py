import streamlit as st
import pandas as pd

from bias_model import load_excel_with_header_detection, add_derived_columns, group_bias

st.set_page_config(page_title="Advisor Bias Review", layout="wide")
st.title("Advisor Commission Bias Review")

uploaded = st.file_uploader("Upload advisor Excel file", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload the dataset to begin.")
    st.stop()

try:
    df = load_excel_with_header_detection(uploaded)
    df = add_derived_columns(df)
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20))

st.markdown("---")

# Identify usable features
numeric_cols = {
    "Actual_Commission",
    "Model_Predicted",
    "Historical_Commission",
    "Bias_Gap_USD",
    "Bias_Pct",
    "Delta_Actual_vs_Model",
    "Delta_Actual_vs_Historical"
}

features = [c for c in df.columns if c not in numeric_cols]

feature = st.selectbox("Analyze bias by feature", features)

agg = group_bias(df, feature)

if agg.empty:
    st.warning("No valid groups found for this feature.")
else:
    st.subheader(f"Bias by {feature}")

    def color_bias(val):
        if val < -30:
            return "background-color: #ff4d4d"
        elif val < -15:
            return "background-color: #ff944d"
        elif val < 0:
            return "background-color: #ffe680"
        else:
            return "background-color: #b3ffb3"

    styled = agg.style.applymap(color_bias, subset=["Avg_Pct"])
    st.dataframe(styled)

    st.bar_chart(
        agg.set_index(feature)["Avg_Pct"],
        use_container_width=True
    )
