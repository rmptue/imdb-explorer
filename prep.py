# prep.py
# Build a clean, analysis-ready movies table from IMDb TSVs
# Works with .tsv or .tsv.gz inside ./data
# Outputs:
#   out/movies_clean.parquet             (exploded by genre)
#   out/collabs_edges.parquet            (tconst, director, actor)
#   out/agg_year_genre.parquet           (year, genre, count, avg_rating, total_votes)
#   out/agg_decade_genre.parquet         (decade, genre, count, avg_rating, total_votes)

from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Config ----------
MIN_VOTES = 0
RUNTIME_MIN = 1
RUNTIME_MAX = 300

GENRE_MAP = {
    "Sci-Fi": "Sci-Fi",
    "Science Fiction": "Sci-Fi",
    "Film-Noir": "Film-Noir",
    "Noir": "Film-Noir",
    "Rom-Com": "Romance",
}

# ---------- Paths ----------
DATA = Path("data")
OUT = Path("out"); OUT.mkdir(exist_ok=True)

# ---------- Helpers ----------
REQUIRED = {
    "title.basics": ["tconst","titleType","primaryTitle","originalTitle","isAdult","startYear","endYear","runtimeMinutes","genres"],
    "title.ratings": ["tconst","averageRating","numVotes"],
    "title.principals": ["tconst","nconst","category"],
    "name.basics": ["nconst","primaryName","primaryProfession","birthYear"],
}

def pick_file(stem: str) -> Path:
    gz = DATA / f"{stem}.tsv.gz"
    tsv = DATA / f"{stem}.tsv"
    if gz.exists(): return gz
    if tsv.exists(): return tsv
    raise FileNotFoundError(f"Missing {gz.name} or {tsv.name} in {DATA.resolve()}")

def read_tsv(stem: str, usecols=None, dtype=None) -> pd.DataFrame:
    path = pick_file(stem)
    print(f"Loading: {path.name}")
    return pd.read_csv(
        path,
        sep="\t",
        na_values="\\N",
        low_memory=False,
        usecols=usecols,
        dtype=dtype
    )

def canonicalize_genres(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip()
    s = s.str.replace(r"\s*,\s*", ",", regex=True)
    def fix_line(line: str) -> str:
        if not line: return ""
        toks = [t.strip() for t in line.split(",") if t.strip()]
        fixed = [GENRE_MAP.get(t, t) for t in toks]
        return ",".join(sorted(set(fixed)))
    return s.map(fix_line)

# ---------- Load ----------
basics = read_tsv(
    "title.basics",
    usecols=REQUIRED["title.basics"],
    dtype={
        "tconst":"string","titleType":"category","primaryTitle":"string","originalTitle":"string",
        "isAdult":"float32","startYear":"string","endYear":"string","runtimeMinutes":"string","genres":"string"
    }
)

movies = basics[basics["titleType"] == "movie"].copy()
movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce").astype("Int64")
movies["endYear"]   = pd.to_numeric(movies["endYear"], errors="coerce").astype("Int64")
movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
movies["runtimeMinutes"] = movies["runtimeMinutes"].mask(
    (movies["runtimeMinutes"] < RUNTIME_MIN) | (movies["runtimeMinutes"] > RUNTIME_MAX)
)
movies["isAdult"] = movies["isAdult"].fillna(0).astype("int8")
movies["genres"] = canonicalize_genres(movies["genres"])

ratings = read_tsv(
    "title.ratings",
    usecols=REQUIRED["title.ratings"],
    dtype={"tconst":"string","averageRating":"float32","numVotes":"int32"}
)
movies = movies.merge(ratings, on="tconst", how="left")

if MIN_VOTES > 0:
    movies = movies[movies["numVotes"].fillna(0) >= MIN_VOTES].copy()
print(f"Movies after votes ≥ {MIN_VOTES}: {len(movies):,}")

principals = read_tsv(
    "title.principals",
    usecols=REQUIRED["title.principals"],
    dtype={"tconst":"string","nconst":"string","category":"category"}
)
dir_link = principals[principals["category"] == "director"][["tconst","nconst"]].copy()

names = read_tsv(
    "name.basics",
    usecols=REQUIRED["name.basics"],
    dtype={"nconst":"string","primaryName":"string","primaryProfession":"string","birthYear":"string"}
)
directors = dir_link.merge(names, on="nconst", how="left").rename(columns={"primaryName":"director"})
directors_agg = (
    directors.dropna(subset=["director"])
             .groupby("tconst", as_index=False)["director"]
             .agg(lambda s: ", ".join(sorted(set(s))))
)
movies = movies.merge(directors_agg, on="tconst", how="left")

# Regions / Languages (optional)
try:
    akas = read_tsv(
        "title.akas",
        usecols=["titleId","region","language"],
        dtype={"titleId":"string","region":"string","language":"string"}
    ).rename(columns={"titleId":"tconst"})
    akas = akas[akas["tconst"].isin(movies["tconst"])]
    agg_regions = (akas.dropna(subset=["region"]).groupby("tconst")["region"]
                      .agg(lambda s: ",".join(sorted(set(s))))).reset_index()
    agg_langs   = (akas.dropna(subset=["language"]).groupby("tconst")["language"]
                      .agg(lambda s: ",".join(sorted(set(s))))).reset_index()
    movies = movies.merge(agg_regions, on="tconst", how="left")
    movies = movies.merge(agg_langs, on="tconst", how="left")
    movies.rename(columns={"region":"regions","language":"languages"}, inplace=True)
except FileNotFoundError:
    movies["regions"] = pd.NA
    movies["languages"] = pd.NA

# ---------- Explode genres & features ----------
movies["genres"] = movies["genres"].replace("", np.nan)
movies["genres_list"] = movies["genres"].str.split(",")
movies_exp = movies.explode("genres_list").rename(columns={"genres_list":"genre"})

movies_exp["decade"] = (movies_exp["startYear"] // 10 * 10).astype("Int64")
movies_exp["weighted_score"] = movies_exp["averageRating"] * np.log10(movies_exp["numVotes"].fillna(0) + 1)

cols = [
    "tconst","primaryTitle","originalTitle","startYear","endYear","decade","runtimeMinutes","isAdult",
    "genre","genres","averageRating","numVotes","weighted_score","director","regions","languages"
]
for c in cols:
    if c not in movies_exp.columns:
        movies_exp[c] = pd.NA
movies_exp = movies_exp[cols]
movies_exp["isAdult"] = movies_exp["isAdult"].astype("int8")

# ---------- Save main table ----------
out_path = OUT / "movies_clean.parquet"
movies_exp.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path.resolve()}")
print(f"Rows: {len(movies_exp):,} | Unique titles: {movies_exp['tconst'].nunique():,}")

# ---------- Pre‑aggregations (year/decade × genre) ----------
print("\nBuilding pre‑aggregations...")

# Year × Genre
agg_year = (
    movies_exp.dropna(subset=["genre"])
      .groupby(["startYear", "genre"])
      .agg(
          count=("tconst", "nunique"),
          avg_rating=("averageRating", "mean"),
          total_votes=("numVotes", "sum"),
      )
      .reset_index()
      .rename(columns={"startYear": "year"})
      .sort_values(["year", "genre"])
)

# Decade × Genre
agg_dec = (
    movies_exp.dropna(subset=["genre"])
      .groupby(["decade", "genre"])
      .agg(
          count=("tconst", "nunique"),
          avg_rating=("averageRating", "mean"),
          total_votes=("numVotes", "sum"),
      )
      .reset_index()
      .sort_values(["decade", "genre"])
)

# Dtypes (optional)
agg_year["year"] = agg_year["year"].astype("Int64")
agg_dec["decade"] = agg_dec["decade"].astype("Int64")
agg_year["count"] = agg_year["count"].astype("int32")
agg_dec["count"] = agg_dec["count"].astype("int32")
agg_year["total_votes"] = agg_year["total_votes"].astype("int64")
agg_dec["total_votes"] = agg_dec["total_votes"].astype("int64")
agg_year["avg_rating"] = agg_year["avg_rating"].astype("float32")
agg_dec["avg_rating"] = agg_dec["avg_rating"].astype("float32")

# Save
agg_year_path = OUT / "agg_year_genre.parquet"
agg_dec_path  = OUT / "agg_decade_genre.parquet"
agg_year.to_parquet(agg_year_path, index=False)
agg_dec.to_parquet(agg_dec_path, index=False)
print(f"Saved: {agg_year_path.resolve()}  ({len(agg_year):,} rows)")
print(f"Saved: {agg_dec_path.resolve()}   ({len(agg_dec):,} rows)")


# ---------- Collaborations (director × actor edges) ----------
print("\nBuilding collaborations (edges only)...")
movie_ids = set(movies["tconst"].unique())
principals_small = principals[principals["tconst"].isin(movie_ids)].copy()

_dir = principals_small[principals_small["category"] == "director"][["tconst", "nconst"]].rename(columns={"nconst": "d_nconst"})
_act = principals_small[principals_small["category"].isin(["actor", "actress"])][["tconst", "nconst"]].rename(columns={"nconst": "a_nconst"})

pairs = _dir.merge(_act, on="tconst", how="inner")

names_small = names[["nconst", "primaryName"]]
pairs = pairs.merge(names_small, left_on="d_nconst", right_on="nconst", how="left") \
             .rename(columns={"primaryName": "director"}).drop(columns=["nconst"])
pairs = pairs.merge(names_small, left_on="a_nconst", right_on="nconst", how="left") \
             .rename(columns={"primaryName": "actor"}).drop(columns=["nconst"])

edges = (
    pairs.dropna(subset=["director","actor"])
         [["tconst","director","actor"]]
         .drop_duplicates(["tconst","director","actor"])
)

edges_path = OUT / "collabs_edges.parquet"
edges.to_parquet(edges_path, index=False)
print(f"Saved: {edges_path.resolve()}")
print(f"Ego edges rows: {len(edges):,} | Unique titles with collabs: {edges['tconst'].nunique():,}")

print("\nDone.")
