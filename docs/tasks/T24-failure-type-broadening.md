# T24 — Failure-type broadening: L4 label corruption, A2 clipping, C2 quantization

Owner: Codex. Prerequisite: T23 merged (do not touch T23 files beyond what this spec says).

## Goal

Gate G4 requires more than one failure type per stage. Add three types from Plan §10
and a Ko co-author suite that localizes them on MNIST.

## 1. L4 — label corruption (local stage)

Injector API today only transforms configs (candidate_pool, local_cfg,
compression_cfg, weights). Label corruption must transform the affected clients'
TRAINING DATA. Extend `falcon/failures` minimally:

- New injector hook `local_data(client_id, data, round_id)` (default: identity,
  defined on the base injector so existing types are untouched).
- `falcon/pipeline/runner.py`: pass `partition[cid]` through
  `injector.local_data(cid, ..., round_id)` at the local stage (both tiers — the
  torch path receives the same ClientData object; check falcon/pipeline/torch_local.py
  for how data flows and keep dtypes intact).
- New failure type `label_corruption` (stage `local`): parameters
  `{"fraction_clients": f, "flip_probability": p}`. Affected clients = first
  ceil(f * len(pool)) of the SORTED pool (deterministic). Within an affected client
  and active round, each example's label flips to a uniformly random OTHER class with
  probability p, drawn from the `failure.local` RNG stream keyed per
  (client, round) — follow the stream-naming discipline in docs/CONTRACTS.md
  section 3 exactly (order-independence across clients is mandatory: derive a
  substream name like `failure.local.<client_id>.round.<t>`).
- The reference run must remain byte-identical (hook default identity).

## 2. A2 — over-aggressive clipping (aggregation stage)

- `falcon/pipeline/stages.py::aggregate`: support optional
  `parameters={"clip_norm": c}` for every rule — each client update whose L2 norm
  exceeds c is scaled to norm c BEFORE weighting (standard update clipping). No
  parameter -> no behavior change (bitwise).
- New failure type `aggressive_clipping` (stage `aggregation`): overrides the
  aggregation parameters for active rounds with a severity-scaled tiny clip_norm
  {1: 1.0, 2: 0.1, 3: 0.01}. Follow how the existing wrong_sample_weights type is
  wired (falcon/failures/) — this type transforms the AggregationConfig, so extend
  the injector with an `aggregation_cfg(cfg, round_id)` hook (default identity) and
  call it in the runner where cfg.aggregation is consumed.

## 3. C2 — low-bit quantization (compression stage)

- `falcon/pipeline/stages.py::compress`: implement `kind="quantization"`
  (currently NotImplemented): symmetric uniform quantization of the update to
  `bits` levels per tensor (scale = max(|update|), levels = 2^bits - 1,
  dequantize back to float; bytes_transmitted = ceil(n * bits / 8)). Deterministic,
  no RNG.
- New failure type `aggressive_quantization` (stage `compression`): overrides
  compression to quantization with severity-scaled bits {1: 8, 2: 4, 3: 2}
  (mirror how aggressive_topk overrides compression_cfg today).

## 4. Ko suite — `experiments/run_failure_types.py`

Pattern: experiments/run_coauthor_cifar.py. Base config
`configs/cases/mnist_reference.yaml` (10 clients, 5 rounds). Cases (active_rounds
(1,4), i.e. through the final round):

- label_corruption sev 2 (fraction_clients 0.3, flip_probability 0.5)
- aggressive_clipping sev 2
- aggressive_quantization sev 2

Per case: analyze_pair on metric="accuracy" (higher_is_better=True,
min_gap=0.005, sham_tolerance=1e-9) + also run
`falcon.reporting.analyze.analyze_pair_vector` with the default vector used in
experiments/run_outcome_vector.py (import VECTOR from it). `--smoke` = same but
4 rounds max and only label_corruption. Summary json + markdown table as usual.

## 5. Client localization demo (label corruption only)

After the label_corruption attribution, run scoped partial restores: for each
affected-vs-clean client at the first active round, `apply_intervention` with
`scope={"client_ids": [cid]}` (see tests/interventions/test_engine.py
test_partial_scoped_restore_differs_from_both_runs for the call shape) and rank
clients by restored-accuracy improvement. Report top-k client precision against the
known corrupted set in the summary (`client_localization` block). Keep it inside
run_failure_types.py behind `--clients` flag (default on in full mode, off in smoke).

## Tests (required)

- aggregate clip_norm: hand-computed clipping of one oversized update; no-param
  bitwise regression vs current outputs.
- compress quantization: round-trip shape/dtype, monotone error as bits decrease,
  bytes_transmitted formula.
- label_corruption: determinism (same seed twice -> identical corrupted labels),
  client order-independence (CONTRACTS section 3), reference untouched.
- Full pytest green.

## Acceptance

- `python experiments/run_failure_types.py --smoke` exits 0.
- Full suite green; existing recorded-run replay tests still pass (hash stability).
- cp949-safe stdout.
