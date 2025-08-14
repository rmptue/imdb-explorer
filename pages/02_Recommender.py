# recommender.py — Light, low‑memory KNN recommender (safe mode)
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from utils import load_movies_light, mem_mb

# ---------- Page config ----------
st.set_page_config(page_title="IMDb Explorer — Recommender (Safe)", page_icon="🎯", layout="wide")
st.title("🎯 Recommender (safe mode)")

with st.popover("ℹ️ How it works"):
    st.markdown(
        "This version avoids loading the full dataset. It builds a small feature matrix "
        "from **genres (one‑hot)** + **average rating**, then uses a cosine **k‑NN** search. "
        "Cap the pool size to keep RAM in check."
    )

# ---------- Controls ----------
with st.expander("Settings", expanded=True):
    min_votes = st.slider("Minimum votes", 1_000, 50_000, 10_000, step=1_000)
    start_year = st.slider("Start year", 1900, 2025, 1970, step=5)
    pool_max = st.slider("Max pool size (caps memory)", 2_000, 20_000, 8_000, step=1_000)
    top_k = st.slider("Recommendations to show", 5, 30, 10)

# ---------- Load (light) ----------
st.caption(f"RAM before load: {mem_mb()} MB")
df = load_movies_light(min_votes=min_votes, start_year=start_year)
if len(df) > pool_max:
    # keep the most popular titles to limit memory
    df = df.nlargest(pool_max, "numVotes")

st.caption(f"RAM after load (capped): {mem_mb()} MB | Pool: {len(df):,}")

if df.empty:
    st.info("No movies match the constraints. Loosen the sliders above.")
    st.stop()

# ---------- Features: genres one‑hot + rating ----------
# Ensure 'genres' exists; fall back to empty string if not
genres_oh = df.get("genres", pd.Series([""] * len(df))).str.get_dummies(sep=",")
X = pd.concat([genres_oh, df[["averageRating"]]], axis=1).astype("float32")

# ---------- Picker ----------
# Order options by popularity for easier selection
pick_from = (
    df.sort_values(["numVotes", "averageRating"], ascending=False)
      .assign(label=lambda x: x["primaryTitle"].fillna("Untitled") +
                              " (" + x["startYear"].astype("Int64").astype(str).str.replace("<NA>", "?", regex=False) + ")")
)
choice = st.selectbox("Pick a movie", pick_from["label"].tolist(), index=0)

# Map choice to index
sel_idx = pick_from.index[pick_from["label"] == choice][0]

# ---------- Run search ----------
if st.button("Find similar", type="primary"):
    st.caption(f"RAM before model: {mem_mb()} MB | Features: {X.shape[0]:,} × {X.shape[1]:,}")
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine", algorithm="brute")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X.iloc[[sel_idx]])

    # skip self (first hit)
    nbr_idx = [i for i in idxs[0] if i != sel_idx][:top_k]
    recs = df.iloc[nbr_idx].copy()
    recs.insert(0, "similarity", (1.0 - pd.Series(dists[0], index=idxs[0]).loc[nbr_idx]).values)

    # tidy view
    view_cols = ["primaryTitle", "startYear", "averageRating", "numVotes"]
    view = recs[view_cols].rename(columns={
        "primaryTitle": "Title",
        "startYear": "Year",
        "averageRating": "IMDb",
        "numVotes": "Votes",
    })
    st.caption(f"RAM after model: {mem_mb()} MB")

    st.dataframe(
        pd.concat([recs[["similarity"]].round(3), view], axis=1).reset_index(drop=True),
        use_container_width=True, height=520,
        column_config={
            "similarity": st.column_config.NumberColumn("Sim", format="%.3f"),
            "Year": st.column_config.NumberColumn(format="%d"),
            "IMDb": st.column_config.NumberColumn(format="%.2f"),
            "Votes": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        recs.assign(similarity=recs["similarity"].round(6))[ ["similarity", "primaryTitle", "startYear", "averageRating", "numVotes"] ]
            .rename(columns={"primaryTitle":"Title","startYear":"Year","averageRating":"IMDb","numVotes":"Votes"})
            .to_csv(index=False).encode(),
        file_name="recommendations_knn.csv",
        mime="text/csv",
    )

st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
