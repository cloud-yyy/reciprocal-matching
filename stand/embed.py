"""Dense vectors (optional): pip install -e '.[dense]'."""

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build(conn, model_name: str = MODEL) -> int:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    with conn.cursor() as cur:
        cur.execute("select id, give_raw, need_raw from profiles order by id")
        rows = cur.fetchall()
        payload = []
        for field_idx, field in ((1, "give"), (2, "need")):
            vectors = model.encode(
                [r[field_idx] for r in rows], normalize_embeddings=True
            )
            payload += [
                (r[0], field, "[" + ",".join(f"{x:.6f}" for x in v) + "]")
                for r, v in zip(rows, vectors)
            ]
        cur.execute("truncate embeddings")
        cur.executemany(
            "insert into embeddings (profile_id, field, emb) values (%s, %s, %s)",
            payload,
        )
    return len(payload)
