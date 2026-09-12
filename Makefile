PY := .venv/bin/python

.PHONY: venv up down init load reset rec eval batch

venv:
	uv venv && uv pip install -e .

up:
	docker compose up -d --build
	@until docker exec reco_db pg_isready -U postgres -d reco >/dev/null 2>&1; do sleep 1; done
	@echo "postgres on localhost:5433"

down:
	docker compose down

init:
	$(PY) -m stand.cli init

load:
	$(PY) -m stand.cli load

reset: up init load

rec:
	$(PY) -m stand.cli rec u001 -k 10

eval:
	$(PY) -m stand.cli eval -k 10

batch:
	$(PY) -m stand.cli batch -k 10 --quota 15 --mmr 0.8
