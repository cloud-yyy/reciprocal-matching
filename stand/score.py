"""Ranking and post-processing.

Pipeline for a single viewer V:
  1. recall      -- candidates sharing at least one term (+ optional city filter);
  2. ranking     -- two directed scores and their symmetrization;
  3. post-proc   -- exposure quota, then MMR re-ordering.
"""

import math
from dataclasses import dataclass, field


@dataclass
class Config:
    k: int = 10
    pool: int = 50            # how many candidates go into post-processing
    w_phrase: float = 0.6     # weight of exact phrases vs. individual words
    w_word: float = 0.4
    alpha_dense: float = 0.0  # share of the dense vector in the directed score
    combine: str = "geomean"  # geomean | min | sum
    mmr_lambda: float = 1.0   # 1.0 = MMR disabled
    quota: int = 0            # 0 = quota disabled; otherwise soft penalty down to zero at the shows cap
    city: str | None = None


@dataclass
class Candidate:
    id: str
    name: str
    s_vb: float = 0.0   # how useful the candidate is to the viewer
    s_bv: float = 0.0   # how useful the viewer is to the candidate
    parts: dict = field(default_factory=dict)
    rel: float = 0.0    # symmetrized score
    quota_mult: float = 1.0
    final: float = 0.0      # after quota
    mmr_score: float = 0.0  # MMR criterion value, debug only
    shows: int = 0


def combine(a: float, b: float, mode: str) -> float:
    a, b = max(a, 0.0), max(b, 0.0)
    if mode == "geomean":
        return math.sqrt(a * b)
    if mode == "min":
        return min(a, b)
    if mode == "sum":
        return (a + b) / 2
    raise ValueError(f"unknown combine mode: {mode}")


def _directed(conn, viewer: str, self_field: str, other_field: str, city):
    with conn.cursor() as cur:
        cur.execute(
            "select cand, kind, s from directed_scores(%s, %s, %s, %s)",
            (viewer, self_field, other_field, city),
        )
        sparse = cur.fetchall()
        cur.execute("select count(*) from embeddings")
        has_dense = cur.fetchone()[0] > 0
        dense = []
        if has_dense:
            cur.execute(
                "select cand, s from dense_scores(%s, %s, %s, %s)",
                (viewer, self_field, other_field, city),
            )
            dense = cur.fetchall()
    return sparse, dense


def _direction_scores(conn, viewer, self_field, other_field, cfg: Config):
    sparse, dense = _directed(conn, viewer, self_field, other_field, cfg.city)
    parts: dict[str, dict[str, float]] = {}
    for cand, kind, s in sparse:
        parts.setdefault(cand, {})[kind] = s
    for cand, s in dense:
        parts.setdefault(cand, {})["dense"] = s

    total = {}
    for cand, p in parts.items():
        sp = cfg.w_phrase * p.get("phrase", 0.0) + cfg.w_word * p.get("word", 0.0)
        sp /= (cfg.w_phrase + cfg.w_word) or 1.0
        if cfg.alpha_dense > 0 and "dense" in p:
            total[cand] = (1 - cfg.alpha_dense) * sp + cfg.alpha_dense * p["dense"]
        else:
            total[cand] = sp
    return total, parts


def rank(conn, viewer: str, cfg: Config) -> list[Candidate]:
    """recall + ranking, without post-processing."""
    vb, vb_parts = _direction_scores(conn, viewer, "need", "give", cfg)
    bv, bv_parts = _direction_scores(conn, viewer, "give", "need", cfg)

    with conn.cursor() as cur:
        cur.execute(
            "select p.id, p.name, coalesce(e.shows, 0)"
            " from profiles p left join exposure e on e.profile_id = p.id"
        )
        meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    out = []
    for cand in sorted(set(vb) | set(bv)):
        if cand == viewer or cand not in meta:
            continue
        name, shows = meta[cand]
        c = Candidate(id=cand, name=name, shows=shows)
        c.s_vb, c.s_bv = vb.get(cand, 0.0), bv.get(cand, 0.0)
        c.parts = {"vb": vb_parts.get(cand, {}), "bv": bv_parts.get(cand, {})}
        c.rel = combine(c.s_vb, c.s_bv, cfg.combine)
        c.final = c.rel
        out.append(c)

    out.sort(key=lambda c: (-c.rel, c.id))  # id as a secondary key for reproducibility
    return out


def _profile_vectors(conn, ids: list[str]) -> dict[str, dict[int, float]]:
    """IDF-weighted profile vector (give+need, words) for candidate similarity."""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "select pt.profile_id, pt.term_id, max(pt.tf * tr.idf)"
            "  from profile_terms pt join terms tr on tr.id = pt.term_id"
            " where pt.profile_id = any(%s) and tr.kind = 'word'"
            " group by 1, 2",
            (ids,),
        )
        rows = cur.fetchall()
    vecs: dict[str, dict[int, float]] = {i: {} for i in ids}
    for pid, tid, w in rows:
        vecs[pid][tid] = w
    for v in vecs.values():
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        for t in v:
            v[t] /= norm
    return vecs


def _cos(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def post_process(conn, ranked: list[Candidate], cfg: Config) -> list[Candidate]:
    pool = ranked[: cfg.pool]

    if cfg.quota > 0:
        for c in pool:
            c.quota_mult = max(0.0, 1.0 - c.shows / cfg.quota)
            c.final = c.rel * c.quota_mult
        pool = [c for c in pool if c.final > 0]
        pool.sort(key=lambda c: (-c.final, c.id))

    if cfg.mmr_lambda >= 1.0:
        return pool[: cfg.k]

    # Relevance is normalized against the pool max: otherwise the similarity
    # penalty (cosine in [0,1]) dwarfs the score itself and MMR degenerates
    # into sorting by dissimilarity.
    top = max((c.final for c in pool), default=0.0) or 1.0
    vecs = _profile_vectors(conn, [c.id for c in pool])
    selected: list[Candidate] = []
    rest = list(pool)
    while rest and len(selected) < cfg.k:
        best, best_score = None, -1e9
        for c in rest:
            sim = max((_cos(vecs[c.id], vecs[s.id]) for s in selected), default=0.0)
            score = cfg.mmr_lambda * (c.final / top) - (1 - cfg.mmr_lambda) * sim
            if score > best_score:
                best, best_score = c, score
        best.mmr_score = best_score
        selected.append(best)
        rest.remove(best)
    return selected


def recommend(conn, viewer: str, cfg: Config, record: bool = False) -> list[Candidate]:
    result = post_process(conn, rank(conn, viewer, cfg), cfg)
    if record and result:
        with conn.cursor() as cur:
            cur.execute(
                "update exposure set shows = shows + 1 where profile_id = any(%s)",
                ([c.id for c in result],),
            )
    return result
