# T27 — S2 availability bias, Krum aggregator, L5 model poisoning

Owner: Codex. Three small features completing E2 (two types per stage) and
extending E5/E7 toward the poisoning story.

## 1. S2 — availability bias (selection stage)

New failure type `availability_bias` (stage `selection`). Parameters:
`{"biased_fraction": f, "availability": p}` — the first ceil(f * len(pool)) of the
SORTED pool are "low-availability" clients; during active rounds each of them is
independently absent from the candidate pool with probability (1 - p), drawn from
the `failure.selection` stream keyed per round AND per client
(`failure.selection.<client_id>.round.<t>` — CONTRACTS section 3 order-independence;
mirror how label_corruption derives substreams). Severity scales p: {1: 0.7,
2: 0.4, 3: 0.1}. Implement via candidate_pool like minority_exclusion
(falcon/failures/selection/ — read the existing injector first). Unlike S1 this is
STOCHASTIC exclusion, not deterministic — that is the point (different failure
signature at the same stage).

## 2. Krum aggregation rule

`falcon/pipeline/stages.py::aggregate`: new rule `krum` (parameters:
`{"byzantine_f": int (default 1)}`). Standard Krum (Blanchard et al. 2017): for
each candidate update i, score = sum of squared L2 distances to its n - f - 2
nearest other updates; the aggregate IS the single update with the lowest score
(deterministic tie-break: lowest client_id). accepted_ids = [winner],
rejected_ids = the rest, weights = {winner: 1.0}. Guard: n >= f + 3 else
ValueError. Deterministic, client-order independent (sort inputs by client_id
before scoring). Preserve dtype like median/trimmed_mean do.

## 3. L5 — model poisoning (local stage)

New failure type `model_poisoning` (stage `local`). Parameters:
`{"fraction_clients": f, "scale": s}` — affected clients = first
ceil(f * len(pool)) of the sorted pool; during active rounds their local update
is REPLACED by (-s * true_update) (sign-flipped scaled update, a standard
untargeted poisoning model). Severity scales s: {1: 1.0, 2: 5.0, 3: 20.0}.
Implementation: the injector cannot see the update via config hooks — add a
`local_state(client_id, state, round_id)` hook to the injector base (identity
default) applied in the runner AFTER local_fn returns and BEFORE recording, and
implement the type with it. Deterministic (no RNG). Keep the recorded state
consistent: update the state's `update` array only (loss_history etc. stay).

## 4. Experiment glue

Extend `experiments/run_e5_aggregators.py` RULES with
`"krum": {"byzantine_f": 1}` and FAILURES with
`("poisoning", model_poisoning sev 2, fraction 0.2, active like the others)`.
The matrix becomes 4 rules x 4 failures. Keep --smoke as is.

## Tests (required)

- availability_bias: determinism, order-independence, reference untouched,
  exclusion is stochastic (some active rounds keep the client).
- krum: hand-computed winner on 4 crafted updates with one outlier;
  n < f + 3 raises; client-order independence; dtype preserved.
- model_poisoning: replaced update equals -s * original; reference untouched;
  affected set deterministic.
- Full pytest green (replay suite must stay green - all defaults identity).

## Acceptance

- `python experiments/run_e5_aggregators.py --smoke` exits 0.
- Full pytest green. ASCII-only stdout. Do not commit.
