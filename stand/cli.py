"""CLI for the test stand: python -m stand.cli <command>."""

import argparse
import json
from pathlib import Path

from stand import evaluate, load as loader
from stand.db import connect
from stand.score import Config, recommend

ROOT = Path(__file__).resolve().parent.parent


def add_config_args(p):
    p.add_argument("-k", type=int, default=10, help="result list size")
    p.add_argument("--pool", type=int, default=50, help="candidates entering post-processing")
    p.add_argument("--w-phrase", type=float, default=0.6)
    p.add_argument("--w-word", type=float, default=0.4)
    p.add_argument("--alpha-dense", type=float, default=0.0,
                   help="share of the dense vector, 0 = tags only")
    p.add_argument("--combine", choices=["geomean", "min", "sum"], default="geomean")
    p.add_argument("--mmr", type=float, default=1.0, dest="mmr_lambda",
                   help="MMR lambda: 1.0 off, 0.7 moderate diversity")
    p.add_argument("--quota", type=int, default=0, help="cap of shows per profile, 0 = off")
    p.add_argument("--city", default=None)


def cfg_from(a) -> Config:
    return Config(k=a.k, pool=a.pool, w_phrase=a.w_phrase, w_word=a.w_word,
                  alpha_dense=a.alpha_dense, combine=a.combine,
                  mmr_lambda=a.mmr_lambda, quota=a.quota, city=a.city)


def cmd_init(a, conn):
    for f in ("schema.sql", "functions.sql"):
        conn.execute((ROOT / "sql" / f).read_text(encoding="utf-8"))
    print("schema and functions applied")


def cmd_load(a, conn):
    n = loader.load(conn, Path(a.path), fake_cities=a.fake_cities)
    with conn.cursor() as cur:
        cur.execute("select kind, count(*), round(avg(idf)::numeric, 3) from terms group by kind")
        stats = cur.fetchall()
    print(f"profiles loaded: {n}")
    for kind, cnt, idf in stats:
        print(f"  {kind} terms: {cnt}, avg idf {idf}")


def cmd_embed(a, conn):
    n = embed_build(conn, a.model)
    print(f"embeddings written: {n}")


def embed_build(conn, model):
    from stand.embed import build
    return build(conn, model)


def cmd_rec(a, conn):
    cfg = cfg_from(a)
    res = recommend(conn, a.viewer, cfg, record=a.record)
    with conn.cursor() as cur:
        cur.execute("select name, need_raw from profiles where id = %s", (a.viewer,))
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"profile {a.viewer} not found")
    print(f"{a.viewer} {row[0]}\n  need: {row[1][:120]}...\n")
    print(f"{'#':>2} {'id':<6} {'name':<22} {'score':>7} {'V<-B':>7} {'B<-V':>7} {'quota':>5}")
    for i, c in enumerate(res, 1):
        print(f"{i:>2} {c.id:<6} {c.name[:22]:<22} {c.final:>7.4f}"
              f" {c.s_vb:>7.4f} {c.s_bv:>7.4f} {c.quota_mult:>5.2f}")


def cmd_explain(a, conn):
    with conn.cursor() as cur:
        for label, sf, of in (("need(V) x give(B)", "need", "give"),
                              ("give(V) x need(B)", "give", "need")):
            cur.execute(
                "select kind, term, idf, contrib from pair_terms(%s, %s, %s, %s) limit %s",
                (a.viewer, a.cand, sf, of, a.limit),
            )
            print(f"\n{label}")
            for kind, term, idf, contrib in cur.fetchall():
                print(f"  {kind:<6} {term[:45]:<45} idf={idf:5.2f} contrib={contrib:.4f}")


def cmd_eval(a, conn):
    base = cfg_from(a)
    variants = {
        "sum (asymmetric)": {"combine": "sum", "alpha_dense": 0.0},
        "min": {"combine": "min", "alpha_dense": 0.0},
        "geomean": {"combine": "geomean", "alpha_dense": 0.0},
        "geomean + MMR 0.7": {"combine": "geomean", "alpha_dense": 0.0, "mmr_lambda": 0.7},
    }
    if a.alpha_dense > 0:
        variants["geomean + dense"] = {"combine": "geomean", "alpha_dense": a.alpha_dense}
    rows = evaluate.compare(conn, base, variants)
    cols = ["variant", "coverage", "gini_shows", "max_shows", "mean_top1",
            "mean_asymmetry", "intra_list_sim", "empty_lists"]
    print(" | ".join(f"{c:<20}" if c == "variant" else f"{c:>14}" for c in cols))
    for r in rows:
        cells = [f"{r['variant']:<20}"] + [
            f"{r[c]:>14.3f}" if isinstance(r[c], float) else f"{r[c]:>14}"
            for c in cols[1:]
        ]
        print(" | ".join(cells))
    if a.gt:
        print("\nagainst ground truth:", json.dumps(
            evaluate.ground_truth(conn, base, Path(a.gt)), ensure_ascii=False, indent=2))


def cmd_batch(a, conn):
    cfg = cfg_from(a)
    conn.execute("update exposure set shows = 0")
    with conn.cursor() as cur:
        cur.execute("select id from profiles order by id")
        ids = [r[0] for r in cur.fetchall()]
    out = {}
    for pid in ids:
        out[pid] = [
            {"id": c.id, "name": c.name, "score": round(c.final, 4),
             "s_vb": round(c.s_vb, 4), "s_bv": round(c.s_bv, 4)}
            for c in recommend(conn, pid, cfg, record=True)
        ]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results for {len(ids)} profiles written to {a.out}")


def main():
    ap = argparse.ArgumentParser(prog="stand")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the schema").set_defaults(fn=cmd_init)

    p = sub.add_parser("load", help="load profiles from JSON")
    p.add_argument("path", nargs="?", default=str(ROOT / "data" / "profiles.json"))
    p.add_argument("--fake-cities", action="store_true",
                   help="assign random cities to test the city recall filter")
    p.set_defaults(fn=cmd_load)

    p = sub.add_parser("embed", help="build dense vectors")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("rec", help="recommendations for a single profile")
    p.add_argument("viewer")
    p.add_argument("--record", action="store_true", help="count these shows toward the quota")
    add_config_args(p)
    p.set_defaults(fn=cmd_rec)

    p = sub.add_parser("explain", help="per-term breakdown for a pair")
    p.add_argument("viewer")
    p.add_argument("cand")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("eval", help="metrics and configuration comparison")
    p.add_argument("--gt", default=None, help="path to a JSON file with labeled pairs")
    add_config_args(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("batch", help="results for all profiles honoring the quota")
    p.add_argument("--out", default=str(ROOT / "data" / "recommendations.json"))
    add_config_args(p)
    p.set_defaults(fn=cmd_batch)

    a = ap.parse_args()
    with connect() as conn:
        a.fn(a, conn)


if __name__ == "__main__":
    main()
