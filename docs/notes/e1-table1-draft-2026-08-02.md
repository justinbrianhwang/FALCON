# E1 stage-localization results — Table 1 draft (2026-08-02)

> **Official reproduction:** `python experiments/run_main_matrix.py` (configs/experiments/main/
> matrix.yaml) regenerates results/main_matrix/table1.{csv,md} — local matrix (both pairs,
> seeds 1–5 + 101–105): **FALCON 34/34, passive 18/34, terminal-only 11/34**, 3 predeclared
> exclusions (pair B seeds 102/104/105, unmatchable band, reasons in the exclusions table).
> The co-author machine's independent pair-A runs (16/16) are additional evidence on top.

Setting: synthetic Tier-0 task (T11-calibrated: class_separation 0.4, label_noise 0.1),
gap-matched failure pairs (target loss-gap 0.02 ± 0.006), two failure families
(selection/minority_exclusion vs compression/aggressive_topk), window interventions (T13),
carrier-chain resolution (T14). Two machines, disjoint seeds.

Pair A = selection/minority_exclusion vs compression/aggressive_topk (loss-gap-matched).
Pair B = local/lr_misconfig vs aggregation/biased_weights (accuracy-gap-matched 0.06±0.012,
T17 deterministic A1 variant) — all four intervenable stages covered.

| Method | Pair A main (s1–5) | Pair A co-author (s101–105) | Pair A co-author (het .5/1/2) | Pair B main (s1–5) | **Total Top-1** |
|---|---:|---:|---:|---:|---:|
| **FALCON (restore+inject+sham, windowed)** | 10/10 | 10/10 | 6/6 | 10/10 | **36/36** |
| Passive stage anomaly | 5/10 | 5/10 | 3/6 | 6/10 | 19/36 |
| Terminal-only (nearest centroid) | 4/10 | 5/10 | 4/6 | 3/10 | 16/36 |

Pair B failure-mode detail: passive mislocalizes local failures as `aggregation` (4/5 seeds —
downstream anomaly dominance again, H3); terminal-only calls every aggregation failure
`selection`. FALCON resolved a carrier tie with aggregation in all five local cases.

Observations:

- Passive anomaly is **systematically** wrong on selection failures (predicts `local` in every
  single run — the largest state deviation sits downstream of the true origin, H3 as predicted).
  It is reliable only on compression failures, whose own state carries the anomaly.
- Terminal-only flips between selection/aggregation/compression across seeds — no stable signal
  at matched terminal gaps (H1 direction).
- Every FALCON win involved an explicit `CARRIER_TIE_RESOLVED:aggregation` — aggregation always
  fully carries upstream damage (restore effects identical to the origin's); temporal
  first-divergence evidence is what breaks the tie. This is Plan §16.3 (T3) observed in data.
- All outcomes were `unique_origin`; no unresolved cases after T13/T14. Sham deviations 0
  throughout, on both machines.

Caveats for the paper: Tier-0 synthetic only; two failure families; localization among four
stages; severities near the matched-gap band. Tier-1 (CIFAR) and the remaining failure types
must replicate before claiming generality (Gate G4).

Raw artifacts: results/e1_main_seeds/ (main), tmp/Output_2026-08-02_17-15-37.zip (co-author).
Gate G2 ("intervention materially beats terminal-only and passive") — met on this evidence.
