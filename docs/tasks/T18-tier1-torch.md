# Task T18 — Tier 1: real datasets + small CNN (torch)

**Assignee:** Kimi
**Contract:** docs/CONTRACTS.md stays authoritative; Plan.md §17.2 Tier 1, §18. Schema already
extended by PM: `ModelConfig` (`logistic_regression` | `small_cnn`), `DatasetConfig.name` now
includes mnist/fmnist/cifar10/cifar100/svhn, `dirichlet_alpha`, `RunConfig.model`.

## Design decisions (PM)

- **Boundary contract unchanged:** stage states still carry flat numpy arrays; Tier-1 params are
  float32 (torch-native) — the recorder hashes dtype+bytes, so determinism is well-defined.
  Document the dtype in CONTRACTS-adjacent comments; synthetic stays float64.
- **Torch is confined to `falcon/pipeline/torch_local.py`** (and model defs): local training +
  evaluation forward pass. Selection/compression/aggregation stay pure numpy on flat arrays.
- **Data path:** real datasets load ONLY from `falcon.data_paths.processed_path(name)` pickles
  (prepare_data.py). No torchvision import in the pipeline.
- **Determinism:** CPU only for now; `torch.use_deterministic_algorithms(True)`, single-threaded
  (`torch.set_num_threads(1)`), all randomness (init, shuffling) seeded from the named Rng
  streams (draw seeds from streams, feed to torch generators). Clean duplicate replay must be
  bit-identical (E0-style test on MNIST tiny config).

## Deliverables

1. `falcon/pipeline/models.py`: `flatten(model)->np.ndarray` / `load_flat(model, vec)`;
   `build_model(cfg: ModelConfig, dataset: DatasetConfig)` → logistic (existing numpy path) or
   SmallCNN (2 conv + 2 fc, per Plan Tier-1 "small CNN"; input adapts to 28x28x1 or 32x32x3).
2. `falcon/pipeline/real_data.py`: load processed pkl; Dirichlet(`dirichlet_alpha`) partition
   over `num_clients` (partition seed = `DatasetConfig.seed`, own generator); minority
   concentration params honored like synthetic; eval = full test split (`EvalData`).
3. `falcon/pipeline/torch_local.py`: `local_train` implementation for torch models honoring
   LocalConfig (SGD, lr, local_steps, batch_size), update = flat delta, loss_history,
   rng_state snapshot as today; `evaluate` forward pass → OutcomeState with per-class accuracy.
4. Wire `runner.run` to dispatch on `cfg.model.name` / `cfg.dataset.name`. Reference configs:
   `configs/cases/mnist_reference.yaml` (10 clients, small rounds) and
   `configs/cases/cifar10_reference.yaml` (Tier-1 headline: 50 clients, 5/round, Dirichlet 0.1 —
   rounds chosen so the curve visibly rises; document the measured curve).
5. Tests (`tests/integration/test_tier1.py`): MNIST tiny config — clean duplicate replay
   bit-identical (all stage hashes), loss decreases, per-client stages recorded as lists; one
   T4 failure (aggressive_topk) on MNIST produces a measurable accuracy gap and first divergence
   at compression. Mark CIFAR tests `@pytest.mark.slow` (runnable manually) if they exceed ~60s.
   Torch import must remain absent from falcon/schema and synthetic paths (assert in test).

Rules: you own falcon/pipeline + configs/cases + these tests. torch allowed ONLY in the new
pipeline modules. `.venv` lacks torch — use the falcon conda env python
(~/miniconda3/envs/falcon/python.exe) for testing. Full suite green. No git commit.
