# pages/04_Genre_Heatmap.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_agg_year_genre, load_agg_decade_genre, mem_mb

st.set_page_config(page_title="IMDb Explorer — Genre Heatmap", page_icon="🔥", layout="wide")
st.title("🔥 Genre Popularity Over Time (Pre‑aggregated)")

with st.popover("ℹ️ About this view"):
    st.markdown(
        "This heatmap uses **pre‑aggregated data** for speed and low memory. "
        "Pick the time grain and metric, optionally focus on specific genres."
    )

# Controls
c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
time_grain = c1.radio("Time axis", ["Decade", "Year"], horizontal=True)
metric = c2.radio("Metric", ["# Movies", "Avg Rating"], horizontal=True)
norm = c3.toggle("Normalize per row (0–1)", value=False, help="Scale each row by its max to compare shapes over time.")

# Pick base data
if time_grain == "Decade":
    agg = load_agg_decade_genre()
    time_col = "decade"
else:
    agg = load_agg_year_genre()
    time_col = "year"

st.caption(f"RAM after load: {mem_mb()} MB | Rows: {len(agg):,}")

# Basic sanity / column mapping
# Expect at least: time_col, 'genre', and either 'count' or 'avg_rating' (or a generic 'value')
need = {time_col, "genre"}
if not need.issubset(agg.columns):
    st.error(f"Pre‑aggregate missing columns: {', '.join(sorted(need - set(agg.columns)))}")
    st.stop()

# Choose value column
val_col = None
if metric == "# Movies":
    if "count" in agg.columns:
        val_col = "count"
    elif "value" in agg.columns:
        val_col = "value"
else:  # Avg Rating
    if "avg_rating" in agg.columns:
        val_col = "avg_rating"
    elif "value" in agg.columns:
        val_col = "value"

if val_col is None:
    st.error("Couldn't find a suitable value column. Expected 'count', 'avg_rating', or 'value'.")
    st.stop()

# Optional genre focus
all_genres = sorted(agg["genre"].dropna().unique().tolist())
focus = st.multiselect("Focus genres (optional)", all_genres, default=[])
if focus:
    agg = agg[agg["genre"].isin(focus)]

# Pivot for heatmap
if agg.empty:
    pivot = pd.DataFrame()
else:
    # compute desired column order BEFORE creating pivot
    col_order = sorted(agg["genre"].dropna().unique().tolist())
    pivot = (
        agg.pivot_table(index=time_col, columns="genre", values=val_col, fill_value=0)
           .sort_index()
           .reindex(col_order, axis=1)
    )

# Normalize per row if requested
if not pivot.empty and norm:
    denom = pivot.replace(0, pd.NA).max(axis=1)
    pivot = pivot.div(denom, axis=0).fillna(0)

# Render
if pivot.empty:
    st.info("No data for the current settings.")
else:
    title_metric = "# Movies" if metric == "# Movies" else "Average IMDb Rating"
    fig = px.imshow(
        pivot,
        labels=dict(x="Genre", y=time_grain, color=("Normalized" if norm else title_metric)),
        aspect="auto",
        title=f"{title_metric}{' (normalized)' if norm else ''} by {time_grain} × Genre"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show underlying table"):
        st.dataframe(pivot, use_container_width=True)

    st.download_button(
        "⬇️ Download heatmap table (CSV)",
        pivot.to_csv(index=True).encode(),
        file_name=f"genre_heatmap_{time_grain.lower()}_{'norm_' if norm else ''}{'count' if metric=='# Movies' else 'avg'}.csv",
        mime="text/csv",
    )

st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
