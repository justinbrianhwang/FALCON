# T28 — Co-author #3 suite: E3 heterogeneity stress + E8 cost + FMNIST/SVHN replication

Owner: Kimi. New file only: `experiments/run_coauthor3_suite.py`. Do NOT modify
anything under `falcon/`. Pattern references: experiments/run_coauthor_cifar.py
(_record/analyze_pair flow), experiments/run_coauthor_suite2.py (scale/cost
profile pattern), falcon/baselines/ (passive + terminal baselines — see how
run_main_matrix.py calls them).

All three parts run from one command with clear [part] prefixes; `--smoke` runs
a tiny variant of each part (~2 min total); `--parts e3,e8,datasets` filter.
End by invoking scripts/collect_output.py like run_coauthor_cifar.py does.

## Part 1 — E3 heterogeneity stress (MNIST)

For alpha in [0.1, 0.5, 1.0, 10.0]: base = configs/cases/mnist_reference.yaml
with dataset.dirichlet_alpha = alpha (rounds 5). Reference + two failures
(selection minority_exclusion sev 3 target_class 5 exclusion_probability 1.0;
local lr_misconfig sev 2 fraction 0.5 lr_multiplier -1.0), active (1, 4).
Per pair: FALCON analyze_pair on accuracy (and class_5_accuracy for selection),
PLUS the passive and terminal baselines' stage predictions. Summary row: alpha,
failure, FALCON outcome/prediction, passive prediction, terminal prediction.
Output table e3_table.md: rows = alpha, cols = failure x method — the H5
comparison (passive degradation vs FALCON stability) must be readable directly.

## Part 2 — E8 cost profile (synthetic)

For num_clients in [10, 25, 50, 100] (synthetic_reference.yaml base,
clients_per_round scaled to 30% rounded, rounds 10): measure wall-clock seconds
for (a) recorded reference run, (b) one windowed restore intervention at the
midpoint stage 'aggregation', and (c) on-disk bytes of the recorded run
directory. Use time.perf_counter. Summary: e8_cost.json + e8_table.md
(clients, record_s, intervention_s, bytes). No attribution needed.

## Part 3 — FMNIST + SVHN stage localization

For name in [fmnist, svhn]: base = mnist_reference.yaml with dataset.name
swapped (rounds 5; the pkl loader handles both — falcon/pipeline/real_data.py).
Reference + the four standard failures exactly as in
experiments/run_coauthor_cifar.py FAILURES but active_rounds (1, 4), and
attribute selection/aggregation on class_5_accuracy, local/compression on
accuracy (import CASE_METRIC/FAILURES from run_coauthor_cifar to avoid
duplication). Summary rows like the CIFAR suite. Data comes from
processed/<name>.pkl (scripts/prepare_data.py --datasets fmnist,svhn).

## Smoke definition

--smoke: E3 only alpha 0.5, selection failure only, 4 rounds; E8 only
clients=10; datasets part SKIPPED unless the pkl files exist (check
falcon.data_paths.processed_path(name).exists(); print SKIP reason).

## Acceptance

- FALCON_DATA_ROOT set: `python experiments/run_coauthor3_suite.py --smoke`
  exits 0 on this machine.
- No edits outside the new script. ASCII-only stdout. Do not commit.
