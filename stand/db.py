import os

import psycopg

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/reco"
)


def connect():
    return psycopg.connect(DSN, autocommit=True)
