# FALCON (working name)

**Failure Attribution and Localized Causal Interventions in Federated Learning.**

![FALCON overview: matched reference/failure runs with restore and inject interventions](assets/figs/hero.png)

Localizes *where* an FL pipeline first fails (Selection → Local Training → Compression → Aggregation → Evaluation) by recording stage-level states in matched reference/failure run pairs and performing Restore / Inject / Sham interventions. Terminal metrics tell you *that* a run failed; FALCON separates the **originator** stage from downstream **amplifiers**, **suppressors**, and **bystanders**.

Full research plan: [Plan.md](Plan.md). Stage interface contract: [docs/CONTRACTS.md](docs/CONTRACTS.md).

> The name FALCON collides with an ICSE 2025 fault-localization paper and is **internal-only** until renamed (Plan.md §2.2). Do not publish this repo under this name.

## Setup (both machines)

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate falcon
pytest   # smoke check
```

Or plain venv:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/mac: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU note: `environment.yml` installs default (CPU) PyTorch — enough for the numpy-only MVP. Swap in a CUDA build per machine when Tier 1 (CIFAR) experiments start.

## Datasets (per machine)

Dataset location is resolved by `falcon/data_paths.py`: `FALCON_DATA_ROOT` env var if set, else `./data`.

- **Machine with an existing torchvision root** (e.g. `D:\pythondata\torch data`):
  `setx FALCON_DATA_ROOT "D:\pythondata\torch data"` — nothing is re-downloaded.
- **Fresh machine (co-author):** set nothing. `python scripts/prepare_data.py --datasets cifar10,cifar100,mnist,fmnist,svhn` downloads into `./data` and exports standardized pickles to `./data/processed/<name>.pkl` (keys: `x_train,y_train,x_test,y_test`).

The FL pipeline (Tier 1+) reads only the processed pickles, so both machines run identical code.

## Quick run (synthetic MVP)

```bash
# single deterministic FL run, prints per-round accuracy
python experiments/run_synthetic.py

# validate a recorded reference/failure pair
python -m falcon.matcher --reference runs/ref_001 --failure runs/fail_001

# one intervention (restore/inject/sham)
python -m falcon.intervention --runs-root . --target-run fail_001 --source-run ref_001 \
  --round 1 --stage compression --mode restore

# full pipeline: pair validation -> all interventions -> attribution report
python -m falcon.reporting --runs-root . --reference ref_001 --failure fail_001 \
  --metric accuracy --output report.md
```

## Sharing results

After running experiments: `python scripts/collect_output.py` → `tmp/Output_<date_time>.zip`
(metrics, stage hashes, reports, configs, environment snapshot; add `--full` to include raw
recorded tensors). Send that zip back for analysis.

Exact dependency versions will be pinned (lockfile) once the MVP stabilizes — before any paper-facing experiment.

## Architecture

<p align="center">
  <img src="assets/figs/system-architecture.png" alt="FALCON components: federated execution, state recorder, paired run matcher, intervention engine, attribution analyzer, report generator" width="480">
</p>

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
falcon/baselines/    passive-anomaly & terminal-only baselines
experiments/    experiment entry scripts
scripts/        data preparation, result collection
tests/          unit / integration / replay / intervention tests
```
