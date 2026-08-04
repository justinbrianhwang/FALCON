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

## Round 2 (exp3 v3, local, 2026-08-04): class-5 retarget alone was NOT enough

Rerun on the main machine (windows still 10-49, global accuracy metric): selection gap
-0.0063, aggregation +0.0010 — both insufficient again. Recorded per-class trajectories
explain it: this regime does not accumulate classes, it whipsaws — the model predicts
1-2 classes at any time and WHICH classes flip round to round (r9: everything class 3,
r30: class 2, r59: classes 0/5). Two consequences:

1. **No memory.** The 10 clean rounds after the window (50-59) draw the same round-keyed
   selections as the reference and fully erase the damage: fail_selection's final
   per-class was identical to ref (class 5: 0.955 vs 0.951).
2. **Global accuracy is mode-invariant.** Every collapse mode scores ~0.10-0.16 globally,
   so even a window reaching the end cannot reliably gap on the global metric.

## Round 3 (2026-08-04, local): windows to the final round + class-5 metric -> 4/4

Changes (commit 943ad90): recorded `per_class` entries exposed as read-time metrics
(`OutcomeState.flat_metrics()`, keys `class_<c>_accuracy`; recorder format and hashes
unchanged — Plan §14.10 outcome vector, first slice); selection/aggregation windows
extended to (10, 59); both attributed on `class_5_accuracy`, min_gap 0.005,
sham_tolerance 1e-9. local/compression cases unchanged (global accuracy).

| Failure (gt stage) | Metric | Gap | Outcome | Prediction |
|---|---|---:|---|---|
| selection (exclude class-5 clients, p=1.0) | class_5_accuracy | **0.951** | unique_origin | **selection ✓** (CARRIER_TIE_RESOLVED:aggregation) |
| aggregation (biased weights, class 5, 0.1) | class_5_accuracy | **0.745** | unique_origin | **aggregation ✓** (selection SRE/SIE exactly 0 — true bystander) |

Sham deviations 0 throughout. local/compression restore/inject were correctly refused
(client_mismatch under the selection failure; lineage_mismatch under the window for
aggregation) — principled refusals, and selection/aggregation evidence sufficed.

**CIFAR-10 stage localization is now 4/4** (local +0.059 and compression +0.007 on the
co-author machine, round 1; selection 0.951 and aggregation 0.745 here). Combined with
synthetic 36/36 and MNIST 12/12, FALCON localizes on two datasets x two model families
(logreg synthetic, SmallCNN on MNIST+CIFAR) — the Gate G4 evidence base, with the
caveat that CIFAR selection/aggregation required the per-class metric (itself a paper
point: minority-targeted failures are invisible to global accuracy, §14.10).

Artifacts: results/coauthor_cifar/ (reports + summary.json), tmp/Output_2026-08-04_19-51-06.zip.
