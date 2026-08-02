# Task T17 — Deterministic biased-weights aggregation failure (A1 variant)

**Assignee:** Kimi
**Context:** `wrong_sample_weights/corrupted` is dominated by its random factor draw — measured
per-seed accuracy-gap endpoints span −0.03…+0.12, sign-unstable, non-monotone in `intensity`
at seeds 1–5. It cannot be gap-matched per seed (E2 pair B blocked). We need a DETERMINISTIC,
monotone aggregation failure: Plan §10.4 A1 "stale/swapped/corrupted counts" includes
systematically wrong weights for specific clients.

## Deliverables

1. New mode in `falcon/failures/aggregation/wrong_sample_weights.py`: `mode: "biased"` with
   parameters `weight_multiplier: float` in (0, 1] and client targeting identical to
   minority_exclusion's "minority-heavy" rule (share of `target_class` above dataset-wide share;
   requires `target_class` parameter). While active, targeted clients' aggregation weights are
   multiplied by `weight_multiplier` BEFORE normalization. No RNG at all. `weight_multiplier=1.0`
   = provable no-op (test bit-identical run). Lower multiplier = more severe, deterministically
   monotone. Reject multiplier outside (0, 1], reject missing target_class, reject `intensity`.
2. Tests: no-op at 1.0 (bit-identical stage hashes); monotone accuracy-gap trend at 1.0/0.5/0.1
   on the case config (measure and assert direction, generous margins); determinism (two runs
   identical); no RNG stream consumed (spy).
3. New case YAML `configs/cases/synthetic_aggregation_biased.yaml` (2 clients/round family,
   minority_class 1) with the measured gap documented.

Rules: you own falcon/failures + tests + the new YAML; nothing else. Full suite green. No git commit.
