# FALCON (working name)

**Failure Attribution and Localized Causal Interventions in Federated Learning.**

Localizes *where* an FL pipeline first fails (Selection → Local Training → Compression → Aggregation → Evaluation) by recording stage-level states in matched reference/failure run pairs and performing Restore / Inject / Sham interventions.

Full research plan: [Plan.md](Plan.md). Stage interface contract: [docs/CONTRACTS.md](docs/CONTRACTS.md).

> The name FALCON collides with an ICSE 2025 fault-localization paper and is **internal-only** until renamed (Plan.md §2.2). Do not publish this repo under this name.

## Setup (both machines)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/mac: source .venv/bin/activate
pip install -e ".[dev]"
pytest   # smoke check
```

Exact dependency versions will be pinned (lockfile) once the MVP stabilizes — before any paper-facing experiment.

## Layout

```
configs/        experiment / failure / intervention configs (YAML)
falcon/schema/       typed state & run schemas (pydantic)
falcon/recorder/     stage-boundary state recorder
falcon/matcher/      reference/failure pair validation
falcon/replay/       full / stage / suffix replay
falcon/intervention/ restore / inject / sham engine
falcon/failures/     failure injectors per stage
falcon/attribution/  SRE/SIE/BIS metrics, origin ranking
falcon/reporting/    attribution reports
experiments/    experiment entry scripts
tests/          unit / integration / replay / intervention tests
docs/tasks/     developer task specs (PM → Codex/Kimi)
```

## Workflow

- PM (Claude) writes task specs in `docs/tasks/`, reviews and integrates.
- Developers (Codex, Kimi) implement against `docs/CONTRACTS.md`.
- Never run paper experiments outside a committed config + seed.
