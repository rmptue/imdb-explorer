# utils.py (revised)
from pathlib import Path
import pandas as pd
import streamlit as st

# Optional deps
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # will guard where needed

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent
PARQ = BASE / "out"
DATAFILE = PARQ / "movies_clean.parquet"
EDGESFILE = PARQ / "collabs_edges.parquet"
AGG_YEAR_GENRE = PARQ / "agg_year_genre.parquet"
AGG_DECADE_GENRE = PARQ / "agg_decade_genre.parquet"

# ---------- Plot defaults (nice out of the box) ----------
try:
    import plotly.express as px
    px.defaults.template = "plotly_white"
    px.defaults.height = 440
except Exception:
    pass

# ---------- Memory helper ----------
def mem_mb() -> int:
    """Return current process RSS in MB (best-effort, cross-platform)."""
    try:
        import psutil, os  # type: ignore
        return round(psutil.Process(os.getpid()).memory_info().rss / 1048576)
    except Exception:
        try:
            import resource  # type: ignore
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        except Exception:
            return -1

# ---------- DuckDB connection (cached) ----------
@st.cache_resource(show_spinner=False)
def get_con():
    """Return a cached in-memory DuckDB connection (if duckdb is available)."""
    if duckdb is None:
        raise ImportError("duckdb is not installed. Run `pip install duckdb`.\n"
                          "Or remove DuckDB-dependent calls.")
    con = duckdb.connect(":memory:")
    try:
        con.execute("PRAGMA threads=4;")
    except Exception:
        pass
    return con

# ---------- Core loaders (cached) ----------
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    """
    Exploded-by-genre movies table from prep.py.
    Expected columns include:
    tconst, primaryTitle, startYear, numVotes, averageRating, genre, genres, director,
    (optional) regions, languages
    """
    if not DATAFILE.exists():
        raise FileNotFoundError(
            f"Missing {DATAFILE.as_posix()}. Run prep.py first.")
    return pd.read_parquet(DATAFILE)

@st.cache_data(show_spinner=False)
def titles_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per title (used for consistent metrics everywhere)."""
    cols = ["tconst","primaryTitle","startYear","averageRating","numVotes"]
    for opt in ["regions", "languages", "genres", "director", "decade", "runtimeMinutes", "isAdult"]:
        if opt in df.columns:
            cols.append(opt)
    return df.drop_duplicates("tconst")[cols].copy()

@st.cache_data(show_spinner=False)
def load_collab_edges() -> pd.DataFrame:
    """
    Lightweight edges (tconst, director, actor) produced by prep.py.
    We recompute metrics by merging with titles_table at runtime.
    """
    if not EDGESFILE.exists():
        raise FileNotFoundError(
            f"Missing {EDGESFILE.as_posix()}. Run prep.py again.")
    edges = pd.read_parquet(EDGESFILE)
    keep = [c for c in ["tconst","director","actor"] if c in edges.columns]
    return (edges[keep]
            .dropna(subset=["director","actor"])  # type: ignore[arg-type]
            .drop_duplicates())

# ---------- Light movie view via DuckDB (column-limited, on-demand) ----------
@st.cache_data(show_spinner=False)
def load_movies_light(
    min_votes: int = 10000,
    start_year: int = 1970,
    cols = ("tconst","primaryTitle","startYear","runtimeMinutes","genres","averageRating","numVotes","isAdult"),
) -> pd.DataFrame:
    """
    Fast projection+filter using DuckDB's read_parquet (no need to load full table).
    """
    if not DATAFILE.exists():
        raise FileNotFoundError(
            f"Missing {DATAFILE.as_posix()}. Run prep.py first.")
    con = get_con()
    q = f"""
        SELECT {", ".join(cols)}
        FROM read_parquet('{DATAFILE.as_posix()}')
        WHERE numVotes >= {min_votes} AND startYear >= {start_year}
    """
    return con.execute(q).df()

# ---------- Pre-aggregates (tiny files) ----------
@st.cache_data(show_spinner=False)
def load_agg_year_genre() -> pd.DataFrame:
    if not AGG_YEAR_GENRE.exists():
        raise FileNotFoundError(
            f"Missing {AGG_YEAR_GENRE.as_posix()}.")
    return pd.read_parquet(AGG_YEAR_GENRE)

@st.cache_data(show_spinner=False)
def load_agg_decade_genre() -> pd.DataFrame:
    if not AGG_DECADE_GENRE.exists():
        raise FileNotFoundError(
            f"Missing {AGG_DECADE_GENRE.as_posix()}.")
    return pd.read_parquet(AGG_DECADE_GENRE)

# ---------- Collaborations: aggregate pairs with caps (DuckDB) ----------
@st.cache_data(show_spinner=False)
def load_collab_pairs(min_films: int = 3, max_people: int = 200) -> pd.DataFrame:
    """
    Build a director-actor pairs dataframe with a minimum film count and a cap
    on the top people by total collaborations (degree), to keep graphs small.
    Requires DuckDB.
    """
    if not EDGESFILE.exists():
        raise FileNotFoundError(
            f"Missing {EDGESFILE.as_posix()}. Run prep.py again.")
    con = get_con()
    path = EDGESFILE.as_posix()
    # 1) pair counts meeting min_films
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW pairs AS
        SELECT director, actor, COUNT(*) AS films
        FROM read_parquet('{path}')
        WHERE director IS NOT NULL AND actor IS NOT NULL
        GROUP BY director, actor
        HAVING COUNT(*) >= {min_films};
    """)
    # 2) keep only pairs where both endpoints are within the top `max_people` by degree
    con.execute("""
        WITH deg AS (
            SELECT name, SUM(films) AS score FROM (
                SELECT director AS name, films FROM pairs
                UNION ALL
                SELECT actor    AS name, films FROM pairs
            ) t
            GROUP BY name
            ORDER BY score DESC
            LIMIT ?
        )
        SELECT p.director, p.actor, p.films
        FROM pairs p
        JOIN deg d1 ON p.director = d1.name
        JOIN deg d2 ON p.actor    = d2.name;
    """, [max_people])
    return con.fetch_df()

# ---------- Global filter bar (reusable across pages) ----------
@st.cache_data(show_spinner=False)
def _unique_list(df: pd.DataFrame, col: str):
    return sorted([x for x in df[col].dropna().unique()])

def filter_bar(df: pd.DataFrame, *, key: str = "global"):
    """
    Render a compact filter bar and return (filtered_df, filter_state_dict).
    Filters:
      - Year range
      - Min votes
      - Genres (multi)
      - Include adult
      - Search in title
      - Optional regions / languages if available
    """
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([2, 1.6, 2, 1.4, 1.8])

        # Year range
        y_min = int(df["startYear"].dropna().min()) if len(df) else 1900
        y_max = 2025
        year_range = c1.slider("Year", y_min, y_max, (max(1990, y_min), y_max), key=f"{key}_year")

        # Votes
        min_votes = c2.number_input("Min votes", 0, 2_000_000, 0, 1000, key=f"{key}_votes")

        # Genres
        genres = _unique_list(df.dropna(subset=["genre"]), "genre") if "genre" in df.columns else []
        chosen_genres = c3.multiselect("Genres", genres, default=[], key=f"{key}_genres")

        # Adult toggle
        include_adult = c4.toggle("Include adult", value=False, key=f"{key}_adult")

        # Search
        q = c5.text_input("Search title", "", key=f"{key}_q")

    # Optional region / language filters (if present)
    with st.expander("More filters", expanded=False):
        if "regions" in df.columns:
            all_regions = sorted({r for s in df["regions"].dropna().str.split(",") for r in s})
            chosen_regions = st.multiselect("Regions (akas)", all_regions, default=[], key=f"{key}_regions")
        else:
            chosen_regions = []

        if "languages" in df.columns:
            all_langs = sorted({l for s in df["languages"].dropna().str.split(",") for l in s})
            chosen_langs = st.multiselect("Languages (akas)", all_langs, default=[], key=f"{key}_langs")
        else:
            chosen_langs = []

        hide_outliers = st.checkbox("Hide runtime outliers (<40 or >240 min)", value=True, key=f"{key}_rt_out")

    # Build mask
    m = df["startYear"].between(year_range[0], year_range[1], inclusive="both")
    m &= df["numVotes"].fillna(0) >= min_votes
    if chosen_genres:
        m &= df["genre"].isin(chosen_genres)
    if not include_adult and "isAdult" in df.columns:
        m &= df["isAdult"].fillna(0).astype(int) == 0
    if q:
        m &= df["primaryTitle"].fillna("").str.contains(q, case=False, na=False)
    if chosen_regions and "regions" in df.columns:
        m &= df["regions"].fillna("").str.contains("|".join(map(str, chosen_regions)))
    if chosen_langs and "languages" in df.columns:
        m &= df["languages"].fillna("").str.contains("|".join(map(str, chosen_langs)))

    out = df.loc[m].copy()
    state = dict(
        year_range=year_range, min_votes=min_votes, genres=chosen_genres,
        include_adult=include_adult, q=q, regions=chosen_regions, languages=chosen_langs,
        hide_outliers=hide_outliers
    )
    return out, state

# ---------- Recommender helpers ----------
@st.cache_data(show_spinner=False)
def build_rec_base(df: pd.DataFrame, min_votes: int = 0) -> pd.DataFrame:
    """
    Build a per-title table for recommendations.
    Uses the original 'genres' (comma-separated) and 'director' string.
    The df you pass in should already be filtered by the global bar if desired.
    """
    base = (
        df.groupby("tconst")
          .agg(
              primaryTitle=("primaryTitle","first"),
              startYear=("startYear","first"),
              genres=("genres","first"),        # not exploded
              director=("director","first"),
              averageRating=("averageRating","mean"),
              numVotes=("numVotes","max"),
          ).reset_index()
    )
    base["genres"] = base["genres"].fillna("")
    base["director"] = base["director"].fillna("")
    if min_votes and min_votes > 0:
        base = base[base["numVotes"].fillna(0) >= min_votes].copy()

    # decade bucket (string) to stabilize similarity
    decade = (base["startYear"] // 10 * 10).astype("Int64").astype(str).str.replace("<NA>","", regex=False)
    base["decade_bucket"] = decade

    # movie label for the dropdown
    sy = base["startYear"].astype("Int64").astype(str).str.replace("<NA>","?", regex=False)
    base["movie_key"] = base["primaryTitle"].fillna("Untitled") + " (" + sy + ")"
    return base

@st.cache_resource(show_spinner=False)
def build_similarity_matrix(base: pd.DataFrame, w_genre: float = 2.0, w_dir: float = 1.5, w_decade: float = 0.5):
    """
    Compute a weighted cosine similarity using three TF-IDF spaces:
    - genres (comma-separated tokens)
    - director names
    - decade bucket
    Final similarity = w_genre*sim_genre + w_dir*sim_dir + w_decade*sim_decade
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def tfidf(texts):
        vec = TfidfVectorizer(token_pattern=r"[^,\s]+")
        mat = vec.fit_transform(texts)
        return mat

    genres_text   = base["genres"].fillna("")
    director_text = base["director"].fillna("")
    decade_text   = base["decade_bucket"].fillna("")

    Sg = cosine_similarity(tfidf(genres_text))
    Sd = cosine_similarity(tfidf(director_text))
    St = cosine_similarity(tfidf(decade_text))

    S = w_genre * Sg + w_dir * Sd + w_decade * St
    return S
