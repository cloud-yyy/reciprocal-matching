"""Parse give/need fields into a sparse vector.

Two kinds of terms:
  phrase -- the whole phrase ("tax breaks for accredited it companies");
            exact match, high precision, low coverage;
  word   -- a single stemmed token ("tax", "break", "it");
            catches paraphrases, which make up most of the dataset.

Both kinds are weighted by their own IDF and combined with tunable weights.

Sample data in this repo is in Russian, so the stemmer/stopwords below
target Russian text -- swap them for your own language if needed.
"""

import re

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("russian")

STOPWORDS = {
    "и", "в", "на", "для", "с", "со", "по", "из", "к", "о", "об", "от", "за",
    "при", "до", "под", "над", "то", "как", "же", "или", "не", "а", "у", "бы",
    "что", "это", "его", "их", "мод", "про",
}

_SPLIT = re.compile(r"[^0-9a-zа-я+#]+")


def normalize(s: str) -> str:
    s = s.lower().replace("ё", "е").replace(" ", " ")
    s = re.sub(r"[«»\"'.;:!?]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -–—")


def phrases(field: str) -> list[str]:
    out, seen = [], set()
    for part in field.split(","):
        p = normalize(part)
        if len(p) >= 2 and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def words(field: str) -> list[str]:
    out, seen = [], set()
    for token in _SPLIT.split(normalize(field)):
        if len(token) < 3 or token in STOPWORDS or token.isdigit():
            continue
        stem = _STEMMER.stemWord(token) if re.search(r"[а-я]", token) else token
        if len(stem) >= 3 and stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def terms(field: str) -> list[tuple[str, str]]:
    """[(kind, term), ...] for a single profile field."""
    return [("phrase", p) for p in phrases(field)] + [("word", w) for w in words(field)]
