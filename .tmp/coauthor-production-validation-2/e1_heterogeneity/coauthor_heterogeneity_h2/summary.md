# E1 terminal observational equivalence

Status: **PASS**. Target gap: 0.02 +/- 0.006 (loss).

| Case | Truth | Gap | Terminal-only | Passive | FALCON | FALCON outcome |
|---|---|---:|---|---|---|---|
| selection_minority_exclusion | selection | 0.0172933 | selection | local | unresolved | unresolved |
| compression_aggressive_topk | compression | 0.0234699 | compression | compression | unresolved | unresolved |

Terminal-only training protocol: One labeled run per built-in failure type, all generated from the same reference config and experiment seed; labels are injected stages and NearestCentroidStageClassifier receives terminal_features only.
