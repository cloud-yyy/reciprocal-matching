"""Quality metrics and configuration comparison across the whole profile set."""

import json
import math
from dataclasses import replace
from pathlib import Path

from stand.score import Config, _cos, _profile_vectors, post_process, rank


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def run_all(conn, cfg: Config) -> dict[str, list]:
    with conn.cursor() as cur:
        cur.execute("select id from profiles order by id")
        ids = [r[0] for r in cur.fetchall()]
    return {pid: post_process(conn, rank(conn, pid, cfg), cfg) for pid in ids}


def metrics(conn, cfg: Config) -> dict:
    lists = run_all(conn, cfg)
    ids = list(lists)
    shown: dict[str, int] = {}
    top1, asym, intra, empty = [], [], [], 0

    for pid, lst in lists.items():
        if not lst:
            empty += 1
            continue
        top1.append(lst[0].rel)
        for c in lst:
            shown[c.id] = shown.get(c.id, 0) + 1
            hi = max(c.s_vb, c.s_bv)
            asym.append(abs(c.s_vb - c.s_bv) / hi if hi > 0 else 0.0)
        vecs = _profile_vectors(conn, [c.id for c in lst])
        pairs = [
            _cos(vecs[a.id], vecs[b.id])
            for i, a in enumerate(lst)
            for b in lst[i + 1 :]
        ]
        intra.append(sum(pairs) / len(pairs) if pairs else 0.0)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "k": cfg.k,
        "combine": cfg.combine,
        "mmr_lambda": cfg.mmr_lambda,
        "alpha_dense": cfg.alpha_dense,
        "quota": cfg.quota,
        "coverage": len(shown) / len(ids) if ids else 0.0,
        "gini_shows": gini([shown.get(i, 0) for i in ids]),
        "max_shows": max(shown.values(), default=0),
        "mean_top1": mean(top1),
        "mean_asymmetry": mean(asym),
        "intra_list_sim": mean(intra),
        "empty_lists": empty,
        "_lists": lists,
    }


def ground_truth(conn, cfg: Config, gt_path: Path) -> dict:
    """Metrics against the labels from the generation prompt (mutual / one_way / false_friends)."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    lists = run_all(conn, cfg)
    pos = {
        pid: {c.id: i + 1 for i, c in enumerate(lst)} for pid, lst in lists.items()
    }

    def rank_of(a, b):
        return pos.get(a, {}).get(b)

    mutual = [(p["a"], p["b"]) for p in gt.get("mutual", [])]
    hits = sum(
        1 for a, b in mutual if rank_of(a, b) or rank_of(b, a)
    )
    both = sum(1 for a, b in mutual if rank_of(a, b) and rank_of(b, a))
    one_way = [(p["from"], p["to"]) for p in gt.get("one_way", [])]
    ow_hits = sum(1 for a, b in one_way if rank_of(a, b))
    ff = [(p["a"], p["b"]) for p in gt.get("false_friends", [])]
    ff_hits = sum(1 for a, b in ff if rank_of(a, b) or rank_of(b, a))

    return {
        "mutual_recall@k": hits / len(mutual) if mutual else None,
        "mutual_both_directions": both / len(mutual) if mutual else None,
        "one_way_leak@k": ow_hits / len(one_way) if one_way else None,
        "false_friends_leak@k": ff_hits / len(ff) if ff else None,
    }


def compare(conn, base: Config, variants: dict[str, dict]) -> list[dict]:
    out = []
    for label, overrides in variants.items():
        m = metrics(conn, replace(base, **overrides))
        m.pop("_lists")
        out.append({"variant": label, **m})
    return out
