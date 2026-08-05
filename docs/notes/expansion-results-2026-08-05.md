# Expansion experiment results (T22-T26 full runs, 2026-08-05)

All on the main machine. Artifacts under results/{e5_aggregators, compound,
failure_types, e4_fedprox}. Code: commits a67ba35, 4ba386e, b624627.

## E5 — aggregator-role matrix (H4), synthetic

| failure | weighted_mean | median | trimmed_mean |
|---|---|---|---|
| selection | selection V (+0.018) | insufficient (+0.004) | selection V (+0.008) |
| local | local V (+0.022) | local V (+0.006) | local V (+0.010) |
| compression | compression V (+0.032) | compression V (+0.042) | compression V (+0.038) |

H4 refined: robust rules SUPPRESS upstream damage (median shrinks the selection
gap below the attribution gate and local by ~4x) but are helpless against
compression damage — a robust aggregate cannot recover coordinates the clients
never sent (top-k). Suppression manifests in the failure gap, not in role labels:
whenever a measurable gap survives, aggregation still scores as a faithful
carrier (nSRE/nSIE 1.0, no negative scores). FALCON's prediction is correct in
all 8 sufficient-gap cells.

### T27 extension: 4 rules x 4 failures (adds krum + L5 model poisoning)

| failure | weighted_mean | median | trimmed_mean | krum |
|---|---|---|---|---|
| selection | selection V (+0.018) | insufficient (+0.004) | selection V (+0.008) | selection V (+0.006) |
| local | local V (+0.022) | local V (+0.006) | local V (+0.010) | local V (+0.010) |
| compression | compression V (+0.032) | compression V (+0.042) | compression V (+0.038) | compression V (+0.038) |
| poisoning (L5, gt local) | **local V (+0.108)** | insufficient (**0.000**) | insufficient (-0.002) | local V (+0.006) |

The poisoning row completes the H4 gradient: sign-flip poisoning devastates
weighted_mean (+0.108, correctly localized to local), is FULLY NEUTRALIZED by
median and trimmed_mean (gap exactly 0 / noise-negative — nothing left to
attribute, and the gate correctly refuses), and leaves a small residue under
krum (+0.006, still correctly localized — selecting one honest update forfeits
averaging variance reduction). A "failure" an aggregator genuinely absorbs is
not a failure at the outcome level; FALCON's insufficient-gap gate encodes
exactly that. 12/12 sufficient-gap cells correct across the 4x4.

## Compound failures (S1+C1, S1+A1), synthetic, 10 rounds

Both cases: outcome `unresolved`, origin_set {selection, aggregation},
note COMPOUND_FAILURE_AMBIGUITY. No fabricated unique origins (Plan 15.3
honesty requirement observed in practice). Known limitation recorded: with a
selection failure in the compound, local/compression interventions are
client-mismatch invalid, so a compression co-failure cannot enter the origin
set — the ambiguity set is upstream-biased. Paper-worthy caveat.

## Failure-type broadening (L4/A2/C2), MNIST, 5 rounds

| failure (gt) | accuracy | loss | macro_recall | fairness_disp |
|---|---|---|---|---|
| label_corruption (local) | insufficient (-0.024!) | **local V (+0.143)** | insufficient | insufficient |
| aggressive_clipping (aggregation) | **aggregation V (+0.392)** | aggregation V | aggregation V | aggregation V |
| aggressive_quantization (compression) | **compression V (+0.011)** | compression V | compression V | insufficient |

Label corruption is the second observed metric-DIRECTION disagreement (after the
corrupted-weights case, docs/notes/metric-disagreement-aggregation.md): 50% label
flips on 3/10 clients slightly IMPROVE terminal accuracy while degrading loss by
0.14 — only the loss metric localizes it. The outcome vector is what saved the
case; a single-accuracy tool reports nothing.

Client localization (14.8 demo, label corruption, scoped restores at round 1):
top-2 precision 1.0 — both corrupted clients in the round's cohort rank first
(improvements +0.0074, +0.0013), every clean client improves exactly 0.0.

## E4 — FedProx (mu 0.1 / 0.01), MNIST, 4 stages each

8/8 localized. mu=0.01: 4/4 on accuracy alone. mu=0.1: selection/local/
compression on accuracy; aggregation's accuracy gap is negative (-0.009) but
class_5_accuracy localizes it (+0.391) — the vector again. Carrier ties with
aggregation resolved by temporal evidence throughout, as under FedAvg.
FALCON attribution is not FedAvg-specific. SCAFFOLD deferred (needs
recorder-visible server state).

## Status roll-up

- Failure types now covered: S1, L1, L4, C1, C2, A1 (+biased), A2 across
  synthetic/MNIST/CIFAR; algorithms FedAvg + FedProx; aggregators
  weighted_mean/median/trimmed_mean.
- Every wrong-prediction count across tonight's runs: 0. Every honest
  withholding traced to a measured-zero or negative gap.
- Backlog: full-matrix vector reanalysis needs a recording rerun of E1/E2
  (summaries only on disk); compound upstream-bias limitation; SCAFFOLD.
