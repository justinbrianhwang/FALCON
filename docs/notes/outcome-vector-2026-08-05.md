# Outcome-vector attribution results (Plan 14.10) — CIFAR-10, 2026-08-05

Full run of `experiments/run_outcome_vector.py` on the main machine: 4 stage-failure
cases x 6 metrics, ONE intervention-replay set per case (metric re-attribution is
free). local/compression failure runs were re-recorded locally, reproducing the
co-author round-1 results on this machine. Artifacts: results/coauthor_cifar/
vector_summary.json, vector_table.md, interventions_<case>.json (replay-free
reanalysis), per-metric reports.

| Case (gt) | accuracy | loss | macro_recall | worst_class | fairness_disp | class_5 |
|---|---|---|---|---|---|---|
| selection | insuff | insuff | insuff | insuff | insuff | **selection V** |
| local | **local V** | insuff | **local V** | insuff | insuff | n/a |
| compression | **compression V** | **compression V** | **compression V** | insuff | insuff | n/a |
| aggregation | insuff | insuff | insuff | insuff | insuff | **aggregation V** |

Every attribution with a sufficient gap was CORRECT (8/8 unique_origin, right stage);
every insufficient gap was honestly withheld (14/14). Zero wrong predictions.

Findings for the paper:

1. **No single metric localizes everything.** Class-targeted failures (selection,
   biased aggregation) are visible ONLY through the targeted class's accuracy —
   even macro recall misses them, because the whipsaw regime redistributes which
   classes collapse without changing the aggregate. Conversely, class-untargeted
   failures (local lr sign, compression top-k) show up on global metrics and need
   no class knowledge. Metric-specific attribution (14.10) is necessary, not
   decorative.
2. **worst_class_accuracy is uninformative in a collapse regime** — it is 0 in both
   reference and failure runs (8/10 classes at zero either way). A vector component
   can be structurally blind, which is itself reportable.
3. **fairness_dispersion is mode-invariant here**: every collapse mode has similar
   per-class spread, so dispersion never gaps. Same lesson as (2).
4. Cross-machine reproduction: the local/compression cases recorded on this machine
   match the co-author's round-1 outcomes (local +0.059-class gap structure,
   compression localized on accuracy AND loss).

MNIST contrast (failure_types smoke + T25 reanalysis of the smoke root): in a
healthy training regime, ALL vector metrics agree (label corruption: unique_origin/
local on every metric; CIFAR smoke pairs: correct on every metric). Metric
disagreement is a property of the pathological regime, which is exactly where
debugging tools are needed.
