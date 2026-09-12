"""Load profiles from JSON into Postgres and build sparse vectors."""

import json
import random
from pathlib import Path

from stand import text

CITIES = ["Moscow", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan"]


def load(conn, path: Path, fake_cities: bool = False, seed: int = 42) -> int:
    profiles = json.loads(path.read_text(encoding="utf-8"))
    rnd = random.Random(seed)

    with conn.cursor() as cur:
        cur.execute("truncate profiles cascade")
        cur.execute("truncate terms restart identity cascade")

        cur.executemany(
            "insert into profiles (id, name, city, give_raw, need_raw)"
            " values (%s, %s, %s, %s, %s)",
            [
                (
                    p["id"],
                    p["name"],
                    rnd.choice(CITIES) if fake_cities else None,
                    p["give"],
                    p["need"],
                )
                for p in profiles
            ],
        )

        rows = []  # (profile_id, field, kind, term)
        for p in profiles:
            for field in ("give", "need"):
                for kind, term in text.terms(p[field]):
                    rows.append((p["id"], field, kind, term))

        cur.executemany(
            "insert into terms (kind, term) values (%s, %s) on conflict do nothing",
            sorted({(k, t) for _, _, k, t in rows}),
        )
        cur.execute("select kind, term, id from terms")
        term_id = {(k, t): i for k, t, i in cur.fetchall()}

        cur.executemany(
            "insert into profile_terms (profile_id, field, term_id, tf)"
            " values (%s, %s, %s, 1) on conflict do nothing",
            [(pid, f, term_id[(k, t)]) for pid, f, k, t in rows],
        )
        cur.execute("insert into exposure (profile_id) select id from profiles")
        cur.execute("select refresh_stats()")

    return len(profiles)
