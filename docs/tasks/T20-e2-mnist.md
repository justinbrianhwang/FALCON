# Task T20 — E2 localization on MNIST (Tier-1 first replication)

**Assignee:** Codex
**Goal:** replicate the Tier-0 E2 result (gap-matched stage localization, FALCON vs passive vs
terminal) on MNIST with the SmallCNN pipeline (T18). This is the first Gate-G4 generalization
data point.

## Deliverables

1. `configs/experiments/main/e2_mnist_pairA.yaml` and `e2_mnist_pairB.yaml`: reference block
   derived from configs/cases/mnist_reference.yaml (10 clients, Dirichlet 0.5, 5 rounds,
   small_cnn) with `minority_class: 1` set so selection/biased-weights targeting works.
   Failure families and severity knobs as in the synthetic pair configs (exclusion_probability;
   k_ratio; lr_multiplier [-1,1] hib=false with fraction 1.0; biased weight_multiplier
   [0.05,1] hib=false with target_class 1).
2. **Measure before pinning:** for each family × seeds 1–5, record attainable accuracy-gap
   endpoint ranges (use the harness severity traces), THEN pin a per-pair target band inside the
   shared attainable region. Document the measured ranges in the YAML comments. If a family has
   no meaningful shared band (like corrupted weights in Tier 0), say so in the summary and note
   it — do not force it.
3. Extend `configs/experiments/main/matrix.yaml` with the two MNIST entries (seeds 1–5) and run
   `experiments/run_main_matrix.py` end to end. MNIST bisection runs are slower than synthetic —
   this may take tens of minutes; that is expected.
4. Paste the resulting table1.md (now including MNIST rows) into your summary, with exclusions
   honestly reported. Update `docs/notes/e1-table1-draft-2026-08-02.md` with an MNIST section.
5. CIFAR-10: do NOT run the matrix on it (60-round bisection is too heavy for this pass).
   Add `configs/experiments/main/e2_cifar10_pairA.yaml` as a prepared-but-unrun config with a
   comment stating estimated cost, for a later dedicated run.

Environment: use ~/miniconda3/envs/falcon/python.exe with FALCON_DATA_ROOT="D:\pythondata\torch data"
set inline. mnist.pkl and cifar10.pkl already exist under that root's processed/.

Rules: you own the new configs, matrix.yaml, and the note update; harness changes only if a
Tier-1 incompatibility genuinely blocks (explain in summary). Full suite green. No git commit.
