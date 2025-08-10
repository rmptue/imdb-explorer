# pages/04_Genre_Heatmap.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_movies, filter_bar

st.set_page_config(page_title="IMDb Explorer — Genre Heatmap", page_icon="🔥", layout="wide")
st.title("🔥 Genre Popularity Over Time")

# ----- Data + global filters -----
df = load_movies()
d, f = filter_bar(df, key="heatmap")

with st.popover("ℹ️ About this view"):
    st.markdown("""
This heatmap shows **how genres evolve over time** under your current filters.
Pick the time grain and metric; use filters above to narrow the slice (years, votes, regions, etc.).
""")

# ----- Controls -----
c1, c2, c3 = st.columns([1.1, 1.1, 2])
time_grain = c1.radio("Time axis", ["Decade", "Year"], horizontal=True)
metric = c2.radio("Metric", ["# Movies", "Avg Rating"], horizontal=True)
focus = c3.multiselect(
    "Focus genres (optional)",
    sorted([g for g in d["genre"].dropna().unique()]),
    default=[]
)

# ----- Prep -----
if time_grain == "Decade":
    d["time"] = (d["startYear"] // 10 * 10).astype("Int64")
else:
    d["time"] = d["startYear"].astype("Int64")

if focus:
    d = d[d["genre"].isin(focus)].copy()

# Aggregate
if metric == "# Movies":
    agg = d.groupby(["time","genre"])["tconst"].nunique().reset_index(name="value")
    title = "Number of Movies"
else:
    agg = d.groupby(["time","genre"])["averageRating"].mean().reset_index(name="value")
    title = "Average IMDb Rating"

# Pivot
pivot = agg.pivot_table(index="time", columns="genre", values="value", fill_value=0)
pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)

# ----- Render -----
if pivot.empty:
    st.info("No data for the current settings.")
else:
    fig = px.imshow(
        pivot,
        labels=dict(x="Genre", y=time_grain, color=title),
        aspect="auto",
        title=f"{title} by {time_grain} × Genre"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show underlying table"):
        st.dataframe(pivot, use_container_width=True)

st.caption(
    f"Filters → Years {f['year_range'][0]}–{f['year_range'][1]} • Min votes ≥ {f['min_votes']} • "
    + (f"Genres: {', '.join(f['genres'])} • " if f['genres'] else "")
    + ("Adult included • " if f['include_adult'] else "Adult excluded • ")
    + (f"Regions: {', '.join(f['regions'])} • " if f['regions'] else "")
    + (f"Languages: {', '.join(f['languages'])} • " if f['languages'] else "")
    + (f"Search: “{f['q']}”" if f['q'] else "")
)
st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
