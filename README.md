# reciprocal-matching

A small algorithm for matching people by two-sided compatibility: what a person **gives** and what they **need**. Useful for finding a good conversation partner in a local social network, mentor matching, networking events, etc.

Each profile has a `give` and a `need` field (free-text tags). For a viewer V and a candidate B, two directed scores are computed:

- `V<-B`: how useful B's `give` is to V's `need`
- `B<-V`: how useful V's `give` is to B's `need`

The two are combined (geomean / min / sum) into one relevance score, then post-processed with an exposure quota and optional MMR re-ranking for diversity.

## Pipeline

1. **recall** — candidates sharing at least one term with the viewer (with an optional city filter)
2. **ranking** — sparse (TF-IDF over phrases + stemmed words) and, optionally, dense (sentence embeddings via pgvector) directed scores
3. **post-processing** — soft quota to avoid over-showing popular profiles, then MMR to diversify the result list

The scoring itself runs as SQL functions (`sql/functions.sql`) on top of Postgres + pgvector; Python only orchestrates calls and post-processing.

## Usage

```bash
make up           # start Postgres (pgvector) in Docker
make venv         # create a venv and install the package
make init         # apply the schema
make load         # load sample profiles from data/profiles.json
make rec          # recommendations for profile u001
make eval         # compare a few scoring configurations
make batch        # run recommendations for all profiles, honoring quota
```

Or directly via the CLI:

```bash
python -m stand.cli rec u001 -k 10 --combine geomean --mmr 0.7
python -m stand.cli explain u001 u007   # per-term breakdown for a pair
```

Dense embeddings are optional:

```bash
pip install -e ".[dense]"
python -m stand.cli embed
```

## Layout

- `stand/` — CLI, scoring, and post-processing logic
- `sql/` — schema and scoring functions
- `data/profiles.json` — sample profiles for local testing
- `data/recommendations.json` — output of `make batch`, a map of viewer id to a ranked list of candidates:

```json
{
  "u001": [
    {"id": "u007", "name": "...", "score": 0.83, "s_vb": 0.9, "s_bv": 0.76}
  ]
}
```

