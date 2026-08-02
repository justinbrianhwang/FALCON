# Aggregation metric disagreement (pair B)

Pair B compares `local/lr_misconfig` with
`aggregation/wrong_sample_weights` in `corrupted` mode. The directional gap
is reference minus failure for accuracy and failure minus reference for loss,
so a positive number means degradation for either metric.

The table below comes from the endpoint entries in the loss severity traces in
`results/e2_main_pairB/` and the accuracy severity traces in
`results/e2_main_pairB_t16/`. Each cell is the gap at corruption intensity
0.25 -> 4.0.

| Seed | Loss gap | Accuracy gap |
|---:|---:|---:|
| 1 | -0.000170 -> -0.003225 | -0.006 -> -0.004 |
| 2 | +0.000779 -> -0.002956 | +0.000 -> +0.002 |
| 3 | -0.004447 -> -0.021235 | -0.004 -> -0.030 |
| 4 | +0.005881 -> +0.030148 | +0.004 -> +0.040 |
| 5 | +0.002369 -> +0.007870 | -0.002 -> -0.004 |

This is not a stable severity response. Increasing corruption improves both
metrics at seeds 1 and 3, worsens both at seed 4, changes the loss direction at
seed 2, and makes loss worse while making accuracy better at seed 5. In other
words, **stage attribution and even failure DIRECTION depend on the outcome
metric**.

This is direct evidence for the outcome-vector requirement in Plan section
14.10 and the metric-disagreement reporting rule in section 21.5. Reporting
only one favorable metric would hide sign reversals, and a bisection matcher
must not treat a declared severity knob as monotone merely because the weight
spread itself is monotone.

## Accuracy retuning result

Accuracy uses 500 evaluation examples and is quantized in steps of 0.002. The
measured local/aggregation endpoint gaps (mild -> severe) were:

| Seed | Local gap | Aggregation gap | Matched aggregation intensity/gap |
|---:|---:|---:|---:|
| 1 | +0.000 -> +0.558 | -0.006 -> -0.004 | 4.0 / -0.004 |
| 2 | +0.000 -> -0.042 | +0.000 -> +0.002 | 0.25 / +0.000 |
| 3 | +0.000 -> +0.002 | -0.004 -> -0.030 | 0.25 / -0.004 |
| 4 | +0.000 -> +0.524 | +0.004 -> +0.040 | 0.25 / +0.004 |
| 5 | +0.000 -> +0.166 | -0.002 -> -0.004 | 0.25 / -0.002 |

There is no shared positive accuracy-gap band across seeds 1-5. The narrowest
shared attainable band is therefore the honest zero-centered interval
[-0.004, +0.004]. All five harness runs match and complete in that band, but
every local case matches the inert endpoint `lr_multiplier: 1.0`; all ten
cases are below the predeclared materiality threshold `min_gap: 0.005` (and
several aggregation gaps are nonpositive). FALCON consequently reports
`insufficient_failure_gap` for all ten cases. These runs validate robust
matching and expose an unusable pair; they are not evidence about localization
accuracy on material failures.

For completeness, the descriptive Top-1 counts are terminal-only 5/10,
passive 5/10, and FALCON 0/10. FALCON made no forced predictions: its ten
outputs were all `unresolved` because the matched pairs had insufficient
failure gaps.
