# overview.py — IMDb Explorer (Overview page, full)
from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils import (
    load_movies,            # full (exploded-by-genre) dataset
    titles_table,
    load_agg_year_genre,    # tiny pre-aggregate
    load_movies_light,      # DuckDB-backed light projection
    mem_mb,
)

# ---------- Page setup & styling ----------
st.set_page_config(page_title="IMDb Explorer — Overview", page_icon="🎬", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.height = 420

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 1.4rem; }
      div[data-testid="stMetric"] > div { padding: 6px 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 IMDb Explorer — Overview")

with st.popover("ℹ️ How to use"):
    st.markdown(
        "This page uses a **DuckDB light loader** for quick KPIs and a **pre-aggregated** Year×Genre table for the heatmap. "
        "Scroll down for the full filterable overview, top list, and browse table."
    )

# ============================================================
# Section A: Fast KPIs (DuckDB light read; no full load)
# ============================================================
try:
    df_light = load_movies_light(min_votes=10_000, start_year=1970)
    st.caption(f"RAM after light load: {mem_mb()} MB | Rows: {len(df_light):,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies (light)", f"{len(df_light):,}")
    c2.metric("Avg Rating (light)", f"{df_light['averageRating'].mean():.2f}" if len(df_light) else "—")
    c3.metric("Median Year (light)", int(df_light['startYear'].median()) if len(df_light) else 0)
except Exception as e:
    st.warning(f"Light KPIs unavailable: {e}")

st.divider()


# ============================================================
# Section C: Full interactive view (loads full exploded dataset)
# ============================================================
# Load full data only once the user reaches this section
with st.spinner("Loading full dataset…"):
    df = load_movies()

titles = titles_table(df)

# ----------------- Global filter bar -----------------
with st.container():
    c1, c2, c3, c4, c5 = st.columns([2, 1.4, 2.2, 1.2, 1.6])

    # Year slider
    year_min = int(df["startYear"].dropna().min()) if len(df) else 1900
    year_max = 2025
    year_range = c1.slider("Year", min_value=year_min, max_value=year_max, value=(max(1990, year_min), year_max))

    # Votes
    min_votes = c2.number_input("Min votes", min_value=0, value=0, step=1000)

    # Genres
    all_genres = sorted(g for g in df["genre"].dropna().unique()) if "genre" in df.columns else []
    sel_genres = c3.multiselect("Genres", all_genres, default=[])

    # Adult toggle
    include_adult = c4.toggle("Include adult", value=False)

    # Title search
    q = c5.text_input("Search title", "")

# Build mask
mask = df["startYear"].between(*year_range) & (df["numVotes"].fillna(0) >= min_votes)
if sel_genres:
    mask &= df["genre"].isin(sel_genres)
if not include_adult and "isAdult" in df.columns:
    mask &= df["isAdult"].fillna(0).astype(int) == 0
if q:
    mask &= df["primaryTitle"].fillna("").str.contains(q, case=False, na=False)

d = df.loc[mask].copy()

# Weighted score for ranking (if not provided by ETL)
if "weighted_score" not in d.columns:
    d["weighted_score"] = d["averageRating"].fillna(0) * np.log10(d["numVotes"].fillna(0) + 1)

# ----------------- Quick nav -----------------
st.caption("Jump to:")
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    st.page_link("pages/02_Recommender.py", label="🎯 Recommender", icon="➡️")
with col_nav2:
    st.page_link("pages/03_Collaborations.py", label="👥 Collaborations", icon="➡️")
with col_nav3:
    st.page_link("pages/04_Genre_Heatmap.py", label="🔥 Genre Heatmap", icon="➡️")

# ----------------- Tabs -----------------
tab_overview, tab_top, tab_table, tab_about = st.tabs([
    "Overview", "Top Movies", "Browse Table", "About"
])

# ================= Tab: Overview =================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies (filtered)", f"{d['tconst'].nunique():,}")
    c2.metric("Avg Rating (filtered)", f"{d['averageRating'].mean():.2f}" if len(d) else "—")
    c3.metric("Total Votes", f"{int(d['numVotes'].sum()):,}" if len(d) else "0")
    busiest = (d.groupby("startYear")["tconst"].nunique().idxmax()
               if len(d) and d["startYear"].notna().any() else "—")
    c4.metric("Busiest Year", f"{busiest}")

    st.caption(
        f"Filters • Years {year_range[0]}–{year_range[1]} • Min votes {min_votes:,}" \
        + (f" • Genres: {', '.join(sel_genres)}" if sel_genres else "") \
        + (" • Adult included" if include_adult else " • Adult excluded") \
        + (f" • Search: “{q}”" if q else "")
    )

    if len(d):
        # Movies per year
        per_year = d.groupby("startYear")["tconst"].nunique().reset_index(name="Movies")
        fig1 = px.line(per_year, x="startYear", y="Movies", title="Movies per Year (filtered)")
        st.plotly_chart(fig1, use_container_width=True)

        # Avg rating by genre (top 20)
        if "genre" in d.columns:
            by_genre = (
                d.dropna(subset=["genre"]).groupby("genre")["averageRating"].mean()
                 .sort_values(ascending=False).reset_index(name="Avg Rating")
            )
            fig2 = px.bar(by_genre.head(20), x="genre", y="Avg Rating",
                          title="Average IMDb Rating by Genre (Top 20, filtered)")
            fig2.update_layout(xaxis_title=None)
            st.plotly_chart(fig2, use_container_width=True)

        # Runtime vs Rating
        st.caption("Scatter auto-excludes implausible runtimes (≤0 or >300) handled in ETL.")
        plot_df = d.dropna(subset=["runtimeMinutes","averageRating"]).copy()
        if not plot_df.empty:
            plot_df = plot_df[plot_df["runtimeMinutes"].between(40, 240, inclusive="both")]
            if not plot_df.empty:
                sample = plot_df.sample(min(6000, len(plot_df)), random_state=1)
                fig3 = px.scatter(
                    sample,
                    x="runtimeMinutes", y="averageRating",
                    size="numVotes",
                    hover_data=["primaryTitle","startYear","director","genre"],
                    title="Runtime vs Rating (bubble size = votes)"
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No titles within the 40–240 min window for current filters.")
        else:
            st.info("No data to plot with the current filters.")
    else:
        st.info("No data with current filters.")

# ================= Tab: Top Movies =================
with tab_top:
    if len(d):
        top = d.sort_values(["weighted_score","numVotes","averageRating"], ascending=False).drop_duplicates("tconst")
        view = top[["primaryTitle","startYear","genre","averageRating","numVotes","director"]].head(100)
        st.dataframe(
            view.rename(columns={
                "primaryTitle":"Title", "startYear":"Year", "genre":"Genre",
                "averageRating":"IMDb", "numVotes":"Votes", "director":"Director(s)"
            }),
            use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "IMDb": st.column_config.NumberColumn(format="%.2f"),
                "Votes": st.column_config.NumberColumn(format=",d"),
            },
            height=520,
        )
        st.download_button(
            "⬇️ Download Top (CSV)",
            view.to_csv(index=False).encode(),
            file_name="top_movies.csv",
            mime="text/csv",
        )
    else:
        st.info("No data with current filters.")

# ================= Tab: Browse Table =================
with tab_table:
    if len(d):
        browse = d[["primaryTitle","startYear","genre","averageRating","numVotes","director","tconst"]].drop_duplicates("tconst")
        st.dataframe(
            browse.rename(columns={
                "primaryTitle":"Title", "startYear":"Year", "genre":"Genre",
                "averageRating":"IMDb", "numVotes":"Votes", "director":"Director(s)"
            }),
            use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "IMDb": st.column_config.NumberColumn(format="%.2f"),
                "Votes": st.column_config.NumberColumn(format=",d"),
                "tconst": st.column_config.TextColumn("IMDb ID"),
            },
            height=640,
        )
        st.download_button(
            "⬇️ Download Filtered Table (CSV)",
            browse.to_csv(index=False).encode(),
            file_name="filtered_table.csv",
            mime="text/csv",
        )
    else:
        st.info("No data with current filters.")

# ================= Tab: About =================
with tab_about:
    st.markdown(
        """
        **How to use**
        - KPIs at the top are from a **light DuckDB read** (fast, low RAM).
        - The **heatmap** uses a **pre-aggregated** Year×Genre table for speed.
        - The lower section loads the **full dataset** for rich filtering and charts.

        **Notes**
        - Rankings use a **weighted score** = rating × log10(votes+1) to balance quality and popularity.
        - Data is pre‑cleaned (implausible runtimes masked; genres standardized) during ETL.
        """
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
