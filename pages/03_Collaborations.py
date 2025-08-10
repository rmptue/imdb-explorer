# pages/03_Collaborations.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

from utils import load_movies, titles_table, load_collab_edges, filter_bar

st.set_page_config(page_title="IMDb Explorer — Collaborations", page_icon="👥", layout="wide")
st.title("👥 Collaborations Explorer")

with st.popover("ℹ️ How to use"):
    st.markdown("""
Explore frequent **Director–Actor** collaborations under your **current filters**.
- Top partners table shows **films together, average IMDb, and total votes**.
- Use the **Network graph** expander to visualize the ego network.
- Change filters in the bar above to refine results.
""")

# ---------- Data + global filters ----------
df = load_movies()
d, f = filter_bar(df, key="collabs")
titles = titles_table(d)

# Load base edges (tconst, director, actor)
try:
    collab_edges = load_collab_edges()
except Exception as e:
    st.info(str(e)); st.stop()

# Join per-title metrics from the FILTERED titles (avoid director column collision)
titles_no_dir = titles.drop(columns=["director"], errors="ignore")
edges_plus = collab_edges.merge(titles_no_dir, on="tconst", how="inner")  # inner to respect filters

# Guard
required = {"tconst", "director", "actor"}
missing = required - set(edges_plus.columns)
if missing:
    st.error(f"Missing expected columns: {missing}. Re-run prep and refresh.")
    st.stop()

# ---------- Quick search ----------
q = st.text_input("🔎 Search a name (director or actor)")
if q:
    hits = edges_plus[
        edges_plus["director"].str.contains(q, case=False, na=False) |
        edges_plus["actor"].str.contains(q, case=False, na=False)
    ][["director", "actor", "primaryTitle", "startYear", "averageRating"]].head(50)
    st.dataframe(hits.rename(columns={
        "primaryTitle":"Title", "startYear":"Year", "averageRating":"IMDb"
    }), use_container_width=True)

# ---------- Helper: ego network ----------
def show_collab_network(edges_plus: pd.DataFrame, center_name: str, mode="Director", top_n=20, weight_on="films"):
    # Aggregate neighbors
    if mode == "Director":
        agg = (
            edges_plus[edges_plus["director"] == center_name]
            .groupby("actor", as_index=False)
            .agg(films=("tconst", "nunique"),
                 total_votes=("numVotes", "sum"),
                 avg_rating=("averageRating","mean"))
            .rename(columns={"actor": "neighbor"})
        )
    else:
        agg = (
            edges_plus[edges_plus["actor"] == center_name]
            .groupby("director", as_index=False)
            .agg(films=("tconst", "nunique"),
                 total_votes=("numVotes", "sum"),
                 avg_rating=("averageRating","mean"))
            .rename(columns={"director": "neighbor"})
        )

    if agg.empty:
        st.info("No collaborators to draw with current filters.")
        return

    weight_on = weight_on if weight_on in {"films","total_votes"} else "films"
    agg = agg.sort_values([weight_on, "films", "total_votes"], ascending=[False, False, False]).head(top_n)

    # Graph
    G = nx.Graph()
    G.add_node(center_name, role="center")
    for _, r in agg.iterrows():
        G.add_node(r["neighbor"], role="neighbor", films=int(r["films"]))
        G.add_edge(center_name, r["neighbor"], films=int(r["films"]), total_votes=int(r["total_votes"]))

    pos = nx.spring_layout(G, k=0.6, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1), hoverinfo="none", opacity=0.6)

    # Nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]; node_x.append(x); node_y.append(y)
        if n == center_name:
            node_text.append(f"⭐ {n}")
            node_size.append(34)
        else:
            films = int(agg.loc[agg["neighbor"] == n, "films"].iloc[0])
            tvotes = int(agg.loc[agg["neighbor"] == n, "total_votes"].iloc[0])
            node_text.append(f"{n}<br>films: {films} • votes: {tvotes:,}")
            node_size.append(10 + 4 * min(films, 12))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center", hoverinfo="text",
        marker=dict(size=node_size, line=dict(width=1))
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=f"Ego Network — {center_name} ({mode})",
        showlegend=False, hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- Main explorer ----------
mode = st.radio("Explore by:", ["Director", "Actor"], horizontal=True)

if edges_plus.empty:
    st.info("No collaborations under current filters. Broaden the filters above.")
    st.stop()

if mode == "Director":
    dir_options = (
        edges_plus.dropna(subset=["director"])
        .groupby("director")["tconst"].nunique()
        .sort_values(ascending=False).index.tolist()
    )
    sel_dir = st.selectbox("Choose a director", dir_options[:1000])

    df_dir = (
        edges_plus[edges_plus["director"] == sel_dir]
        .dropna(subset=["actor"])
        .groupby("actor", as_index=False)
        .agg(films=("tconst", "nunique"),
             avg_rating=("averageRating", "mean"),
             total_votes=("numVotes", "sum"))
        .sort_values(["films","avg_rating","total_votes"], ascending=[False, False, False])
    )

    st.subheader(f"Top actor partners of {sel_dir}")
    st.dataframe(
        df_dir.rename(columns={"actor":"Actor","films":"Films together",
                               "avg_rating":"Avg IMDb","total_votes":"Total votes"}),
        use_container_width=True, height=480
    )

    st.download_button(
        "⬇️ Download table (CSV)",
        df_dir.to_csv(index=False).encode(),
        f"collabs_{sel_dir.replace(' ','_')}.csv", "text/csv"
    )

    st.plotly_chart(
        px.bar(df_dir.head(20), x="actor", y="films",
               title=f"Most frequent collaborators with {sel_dir}")
          .update_layout(xaxis_title=None),
        use_container_width=True
    )

    with st.expander("Network graph (Top collaborators)"):
        colA, colB = st.columns(2)
        top_n = colA.slider("How many collaborators", 5, 60, 20, step=1)
        weight_on = colB.selectbox("Edge weight", ["films", "total_votes"], index=0)
        show_collab_network(edges_plus, sel_dir, mode="Director", top_n=top_n, weight_on=weight_on)

    with st.expander("See movies they worked on"):
        partner = st.selectbox("Pick an actor partner", df_dir["actor"].tolist())
        films = (
            edges_plus[(edges_plus["director"] == sel_dir) & (edges_plus["actor"] == partner)]
            [["primaryTitle","startYear","averageRating","numVotes"]]
            .dropna(subset=["primaryTitle"]).sort_values("startYear")
            .rename(columns={"primaryTitle":"Title","startYear":"Year","averageRating":"IMDb","numVotes":"Votes"})
        )
        st.dataframe(films, use_container_width=True)

else:
    act_options = (
        edges_plus.dropna(subset=["actor"])
        .groupby("actor")["tconst"].nunique()
        .sort_values(ascending=False).index.tolist()
    )
    sel_act = st.selectbox("Choose an actor", act_options[:1000])

    df_act = (
        edges_plus[edges_plus["actor"] == sel_act]
        .dropna(subset=["director"])
        .groupby("director", as_index=False)
        .agg(films=("tconst", "nunique"),
             avg_rating=("averageRating", "mean"),
             total_votes=("numVotes", "sum"))
        .sort_values(["films","avg_rating","total_votes"], ascending=[False, False, False])
    )

    st.subheader(f"Top director partners of {sel_act}")
    st.dataframe(
        df_act.rename(columns={"director":"Director","films":"Films together",
                               "avg_rating":"Avg IMDb","total_votes":"Total votes"}),
        use_container_width=True, height=480
    )

    st.download_button(
        "⬇️ Download table (CSV)",
        df_act.to_csv(index=False).encode(),
        f"collabs_{sel_act.replace(' ','_')}.csv", "text/csv"
    )

    st.plotly_chart(
        px.bar(df_act.head(20), x="director", y="films",
               title=f"Most frequent collaborators with {sel_act}")
          .update_layout(xaxis_title=None),
        use_container_width=True
    )

    with st.expander("Network graph (Top collaborators)"):
        colA, colB = st.columns(2)
        top_n = colA.slider("How many collaborators", 5, 60, 20, step=1)
        weight_on = colB.selectbox("Edge weight", ["films", "total_votes"], index=0)
        show_collab_network(edges_plus, sel_act, mode="Actor", top_n=top_n, weight_on=weight_on)

    with st.expander("See movies they worked on"):
        partner = st.selectbox("Pick a director partner", df_act["director"].tolist())
        films = (
            edges_plus[(edges_plus["actor"] == sel_act) & (edges_plus["director"] == partner)]
            [["primaryTitle","startYear","averageRating","numVotes"]]
            .dropna(subset=["primaryTitle"]).sort_values("startYear")
            .rename(columns={"primaryTitle":"Title","startYear":"Year","averageRating":"IMDb","numVotes":"Votes"})
        )
        st.dataframe(films, use_container_width=True)

st.caption(
    f"Filters → Years {f['year_range'][0]}–{f['year_range'][1]} • Min votes ≥ {f['min_votes']} • "
    + (f"Genres: {', '.join(f['genres'])} • " if f['genres'] else "")
    + ("Adult included • " if f['include_adult'] else "Adult excluded • ")
    + (f"Regions: {', '.join(f['regions'])} • " if f['regions'] else "")
    + (f"Languages: {', '.join(f['languages'])} • " if f['languages'] else "")
    + (f"Search: “{f['q']}”" if f['q'] else "")
)
st.caption("IMDb Explorer • Built with Streamlit • © Joshua Chua")
