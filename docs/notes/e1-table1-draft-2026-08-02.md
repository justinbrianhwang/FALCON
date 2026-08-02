# E1 stage-localization results — Table 1 draft (2026-08-02)

Setting: synthetic Tier-0 task (T11-calibrated: class_separation 0.4, label_noise 0.1),
gap-matched failure pairs (target loss-gap 0.02 ± 0.006), two failure families
(selection/minority_exclusion vs compression/aggressive_topk), window interventions (T13),
carrier-chain resolution (T14). Two machines, disjoint seeds.

| Method | Main (seeds 1–5) | Co-author (seeds 101–105) | Co-author (heterogeneity .5/1/2) | **Total Top-1** |
|---|---:|---:|---:|---:|
| **FALCON (restore+inject+sham, windowed)** | 10/10 | 10/10 | 6/6 | **26/26** |
| Passive stage anomaly | 5/10 | 5/10 | 3/6 | 13/26 |
| Terminal-only (nearest centroid) | 4/10 | 5/10 | 4/6 | 13/26 |

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
