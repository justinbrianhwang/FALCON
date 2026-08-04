# T26 — FedProx local training (E4 algorithm generality)

Owner: Codex. Prerequisites: T23/T24 merged.

## Goal

Plan E4 / RQ5: show FALCON attribution is not FedAvg-specific. Add FedProx as a
local-training variant on BOTH tiers and replicate one localization pair per stage
under it. SCAFFOLD is OUT OF SCOPE for this task (server control variates need
recorder-visible server state — defer, note it in the experiment doc).

## Design

1. `falcon/schema/config.py` — `LocalConfig` gains `algorithm:
   Literal["fedavg", "fedprox"] = "fedavg"` and `prox_mu: float = 0.0`.
   Default keeps every existing config byte-identical.
2. Synthetic tier (`falcon/pipeline/stages.py::local_train`): when algorithm ==
   "fedprox", each local SGD step adds the proximal gradient term
   `prox_mu * (w - w_global)` (w_global = the round's incoming params). Verify
   against the FedProx paper definition (Li et al. 2020, objective
   F_k(w) + mu/2 ||w - w^t||^2).
3. Torch tier (`falcon/pipeline/torch_local.py::local_train`): same term added to
   the loss before backward (`prox_mu / 2 * sum((p - p_global)^2)`), with
   p_global detached constants. Keep CPU determinism (no new RNG draws; same
   stream usage as today). The negative-lr gradient-reversal path must compose
   (prox term uses the true configured objective; reverse AFTER computing the
   full gradient like the current implementation does).
4. `experiments/run_e4_fedprox.py`: base `configs/cases/mnist_reference.yaml`
   with local.algorithm=fedprox, prox_mu=0.1 (also run prox_mu=0.01 if cheap).
   Reference + the four MNIST failure cases used by
   experiments/run_coauthor_cifar.py --smoke (same specs/windows, rounds=5,
   active (1,4)), analyze_pair on accuracy (selection/aggregation additionally on
   class_5_accuracy like the CIFAR suite). `--smoke` = prox_mu=0.1, 4 rounds,
   selection case only. Summary json + table.

## Tests (required)

- fedavg default: bitwise regression (existing recorded hashes still replay —
  run pytest tests/replay).
- fedprox synthetic: hand-check that one step with lr, mu on a 1-param model
  matches the closed-form update.
- fedprox torch: prox_mu=0 equals fedavg output exactly; prox_mu>0 changes the
  update and is deterministic across two runs.
- Full pytest green.

## Acceptance

- `python experiments/run_e4_fedprox.py --smoke` exits 0.
- cp949-safe stdout.
