import streamlit as st
import pandas as pd

from bias_model import add_derived_columns, group_bias

st.set_page_config(page_title="Advisor Bias Review", layout="wide")
st.title("Advisor Commission Bias Review")

uploaded = st.file_uploader("Upload advisor Excel file", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload the dataset to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded)
    df = add_derived_columns(df)

except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20))

st.markdown("---")

# Feature selection
features = [
    c for c in df.columns
    if c not in ["actual_commission", "model_predicted", "historical_commission",
                 "bias_gap_usd", "bias_pct", "delta_actual_vs_model",
                 "delta_actual_vs_historical"]
]

feature = st.selectbox("Analyze bias by feature", features)

agg = group_bias(df, feature)

if agg.empty:
    st.warning("No valid groups found for this feature.")
else:
    st.subheader(f"Bias by {feature}")
    st.dataframe(agg.style.background_gradient(subset=["avg_pct"], cmap="RdYlGn"))

    st.bar_chart(
        agg.set_index(feature)["avg_pct"],
        use_container_width=True
    )
