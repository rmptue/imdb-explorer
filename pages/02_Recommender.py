# pages/02_Recommender.py
import streamlit as st
from utils import load_movies, filter_bar, build_rec_base, build_similarity_matrix

st.set_page_config(page_title="IMDb Explorer — Recommender", page_icon="🎯", layout="wide")
st.title("🎯 Recommender")

# ----- Load + global filters (so recs match the same constraints app-wide) -----
df = load_movies()
d, f = filter_bar(df, key="recs")

with st.popover("ℹ️ How it works"):
    st.markdown("""
**Method**
- Build TF‑IDF features for **genres**, **director(s)**, and **decade** bucket.
- Compute cosine similarity per space and combine:  
  `Sim = w₁·Sim(genres) + w₂·Sim(director) + w₃·Sim(decade)`.
- Rank by similarity, breaking ties by IMDb rating and votes.

**Tips**
- Tighten matches by raising **Min rec votes** or upping **Director weight**.
- Keep filters broad for more candidates; narrow years/genres to focus.
""")

# ----- Controls -----
colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.2])
min_rec_votes = colA.number_input("Min rec votes", 0, 2_000_000, max(2000, f["min_votes"]), 1000)
w_genre = colB.slider("Genre weight", 0.0, 3.0, 2.0, 0.1)
w_dir   = colC.slider("Director weight", 0.0, 3.0, 1.5, 0.1)
w_dec   = colD.slider("Decade weight", 0.0, 2.0, 0.5, 0.1)

# ----- Build recommendation base from filtered data -----
rec_base = build_rec_base(d, min_votes=min_rec_votes)
if rec_base.empty:
    st.info("No candidate titles under current filters. Broaden filters or lower Min rec votes.")
    st.stop()

# ----- Similarity (cached) -----
S = build_similarity_matrix(rec_base, w_genre=w_genre, w_dir=w_dir, w_decade=w_dec)

# ----- Pick a seed movie -----
pick_from = rec_base.sort_values(["numVotes","averageRating"], ascending=False)["movie_key"].tolist()
choice = st.selectbox("Pick a movie", pick_from, index=0)

# ----- Compute top-N recommendations -----
sel_idx = rec_base.index[rec_base["movie_key"] == choice][0]
scores = S[sel_idx]

order = scores.argsort()[::-1]
order = [i for i in order if i != sel_idx][:10]
recs = rec_base.iloc[order].copy()
recs["similarity"] = scores[order]
recs = recs.sort_values(["similarity","averageRating","numVotes"], ascending=[False,False,False])

# ----- Show results -----
show_cols = ["primaryTitle","startYear","genres","director","averageRating","numVotes","similarity"]
st.dataframe(
    recs[show_cols].rename(columns={
        "primaryTitle":"Title","startYear":"Year","genres":"Genres",
        "director":"Director(s)","averageRating":"IMDb Rating","numVotes":"Votes"
    }),
    use_container_width=True, height=520
)

st.download_button(
    "⬇️ Download Recommendations (CSV)",
    recs[["primaryTitle","startYear","genres","director","averageRating","numVotes","similarity"]]
        .to_csv(index=False).encode(),
    "recommendations.csv", "text/csv"
)

# ----- Filters caption (safe string join to avoid 'str is not callable') -----
parts = [
    f"Filters → Years {f['year_range'][0]}–{f['year_range'][1]} • ",
    f"Min votes ≥ {f['min_votes']} • ",
]
if f["genres"]:
    parts.append(f"Genres: {', '.join(f['genres'])} • ")
parts.append("Adult included • " if f["include_adult"] else "Adult excluded • ")
if f["regions"]:
    parts.append(f"Regions: {', '.join(f['regions'])} • ")
if f["languages"]:
    parts.append(f"Languages: {', '.join(f['languages'])} • ")
if f["q"]:
    parts.append(f"Search: “{f['q']}”")

st.caption("".join(parts))
st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
