# collaborations.py — Collaborations Explorer (safe mode)
import streamlit as st
import pandas as pd
import networkx as nx

from utils import load_collab_pairs, mem_mb

# ---------- Page config ----------
st.set_page_config(page_title="IMDb Explorer — Collaborations (Safe)", page_icon="👥", layout="wide")
st.title("👥 Collaborations Explorer (safe mode)")

with st.popover("ℹ️ How it works"):
    st.markdown(
        "Builds a compact **Director–Actor pairs** table in DuckDB (via `load_collab_pairs`)\n"
        "and caps the network by **top-degree people** to keep memory safe."
    )

# ---------- Controls ----------
with st.expander("Network settings", expanded=True):
    min_films = st.slider("Min films together", 2, 10, 3)
    max_people = st.slider("Max people (by degree)", 50, 500, 150, step=50)

# ---------- Load pairs (already aggregated in DuckDB) ----------
st.caption(f"RAM before pairs: {mem_mb()} MB")
try:
    pairs = load_collab_pairs(min_films=min_films, max_people=max_people)
except Exception as e:
    st.error(f"Couldn't load collaboration pairs: {e}")
    st.stop()

st.caption(f"RAM after pairs: {mem_mb()} MB | Pairs: {len(pairs):,}")

if pairs.empty:
    st.info("No pairs matched these settings. Lower `Min films together` or raise `Max people`.")
    st.stop()

with st.expander("Preview pairs", expanded=False):
    st.dataframe(pairs.head(50), use_container_width=True)

# ---------- Build network on demand ----------
if st.button("Build network", type="primary"):
    st.caption(f"RAM before graph: {mem_mb()} MB")

    # Create simple undirected graph where nodes are people, edges are collaborations
    G = nx.Graph()
    for _, r in pairs.iterrows():
        d = str(r["director"]) if pd.notna(r["director"]) else None
        a = str(r["actor"]) if pd.notna(r["actor"]) else None
        if not d or not a:
            continue
        G.add_edge(d, a, weight=int(r.get("films", 1)))

    st.write(f"Graph: **{G.number_of_nodes():,}** nodes / **{G.number_of_edges():,}** edges")
    st.caption(f"RAM after graph: {mem_mb()} MB")

    # Keep rendering lightweight: show a ranked table and let users download
    top_pairs = pairs.sort_values("films", ascending=False).head(50).rename(columns={
        "director": "Director", "actor": "Actor", "films": "Films together"
    })
    st.subheader("Top collaborating pairs")
    st.dataframe(
        top_pairs,
        use_container_width=True,
        column_config={"Films together": st.column_config.NumberColumn(format=",d")},
        height=520,
    )

    st.download_button(
        "⬇️ Download pairs (CSV)",
        pairs.rename(columns={"director": "Director", "actor": "Actor", "films": "Films together"})
             .to_csv(index=False).encode(),
        file_name="collaboration_pairs.csv",
        mime="text/csv",
    )

st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
