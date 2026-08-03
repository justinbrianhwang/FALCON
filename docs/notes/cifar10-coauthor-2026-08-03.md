# CIFAR-10 co-author results, round 1 (exp3 v2, 2026-08-03)

Machine: DESKTOP-2B048I1. Config: committed cifar10_reference (SmallCNN, 50 clients,
Dirichlet 0.1, 5/round, 60 rounds, final global acc 0.159) + four fixed-severity failures
(window rounds 10–49), full FALCON attribution per pair.

| Failure (gt stage) | Global acc gap | Outcome | Prediction |
|---|---:|---|---|
| local (lr sign flip) | +0.059 | unique_origin | **local ✓** (carrier tie resolved) |
| compression (top-k 0.05) | +0.007 | unique_origin | **compression ✓** |
| selection (exclude class-1 clients, p=1.0) | +0.0001 | insufficient_failure_gap | withheld (honest) |
| aggregation (biased weights, class 1) | +0.0009 | insufficient_failure_gap | withheld (honest) |

## Root cause of the two insufficient cases (measured from recorded per-class outcomes)

The clean reference in this weak-model regime learns ONLY two classes:
class 0 → 0.585, class 5 → 0.953, all eight others ≈ 0.000 (global 0.159 is carried almost
entirely by classes 0/5). Both insufficient failures targeted **class 1, which the reference
never learned** — there was literally nothing to damage. Not a FALCON failure: the gap gate
correctly refused to attribute a non-existent failure.

Paper-relevant observations:

1. Severity validity (§17.3) is REGIME-dependent: a failure type that is severe in Tier 0 can be
   null in an undertrained Tier-1 regime. Benchmark cases must verify the reference actually
   learned whatever the failure targets.
2. Global accuracy conceals total class collapse (8/10 classes at zero in a "healthy" run) —
   outcome-vector metrics (§14.10) are needed for any minority-targeted failure. Backlog:
   expose per-class metrics through the intervention engine / analyzer.

## Round 2 (exp3 v3): selection/aggregation retarget class 5; `--cases selection,aggregation`
so the two already-successful cases are not re-run.
