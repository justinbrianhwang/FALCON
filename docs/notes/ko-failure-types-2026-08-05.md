# Ko co-author results — failure-type suite (2026-08-05)

Source: tmp/Output_2026-08-05_10-10-09.zip (DESKTOP-2B048I1, torch 2.11+cu126,
numpy 2.4.4 — pipeline forces CPU determinism regardless of the CUDA build).
Suite: experiments/run_failure_types.py (MNIST, L4/A2/C2, scalar + vector
attribution + client localization).

## Stage localization: 3/3 correct

| failure (gt) | accuracy | loss | macro_recall | fairness_disp |
|---|---|---|---|---|
| label_corruption (local) | local V (+0.0072) | local V (+0.206) | local V | insufficient |
| aggressive_clipping (aggregation) | aggregation V (+0.420) | aggregation V (+1.417) | aggregation V | aggregation V |
| aggressive_quantization (compression) | compression V (+0.0397) | compression V (+0.129) | compression V | insufficient |

Client localization (label corruption): top-2 precision 1.0 on their machine too
(corrupted clients rank first with +0.0073/+0.0035; every clean client exactly 0.0).

## Cross-machine observation: Tier-1 gaps are machine-DEPENDENT, attributions are not

Main-machine vs Ko-machine endpoint gaps for the SAME configs/seeds:

| failure | accuracy gap (main) | accuracy gap (Ko) | loss gap (main) | loss gap (Ko) |
|---|---:|---:|---:|---:|
| label_corruption | -0.0243 | +0.0072 | +0.143 | +0.206 |
| aggressive_clipping | +0.392 | +0.420 | +1.359 | +1.417 |
| aggressive_quantization | +0.0112 | +0.0397 | +0.072 | +0.129 |

The synthetic tier is bitwise cross-machine portable (E0, 15/15 hashes); the torch
tier is deterministic only per machine — different torch/BLAS builds change float32
kernels, so borderline metric gaps can flip sign across machines (label corruption's
accuracy: -0.024 here, +0.007 there). Two consequences, both already consistent with
the design:

1. FALCON only ever compares runs recorded on the SAME machine (matched-pair
   protocol) — attributions were correct on both machines independently.
2. Robust vector components (loss here) keep a stable sign across machines while
   accuracy sits in the noise band; borderline single-metric results should never
   be compared across machines. Paper: state Tier-1 determinism as per-machine.

Status: Ko suite complete. Outstanding: Raf bundle results (E5+compound,
synthetic — bitwise-portable tier, so results should match ours exactly),
co-author #3 suite (T28 in progress).
