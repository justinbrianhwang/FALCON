# Task T15 — Intensity knob for wrong_sample_weights.corrupted

**Assignee:** Kimi
**Motivation:** E2 (full 4-family localization matrix) gap-matches failure severities by
bisection on a continuous parameter. `corrupted` mode draws per-client weight factors
log-uniform in a FIXED [0.1, 10] range — no intensity control, so aggregation failures cannot
be gap-matched (noted in docs/notes/coauthor-results-2026-08-02.md).

## Deliverables

1. `falcon/failures/aggregation/wrong_sample_weights.py`: new parameter
   `intensity: float = 1.0` (valid (0, 4]) for `corrupted` mode — factors drawn log-uniform in
   `[10**-intensity, 10**intensity]`. `intensity=1.0` reproduces today's behavior EXACTLY
   (same stream, same draws — regression-test this bit-for-bit). Reject non-finite/out-of-range.
   `uniform`/`swapped` reject an `intensity` parameter (fail loud).
2. Update `configs/cases/synthetic_aggregation_failure.yaml` to state `intensity: 1.0`
   explicitly and re-document the measured gap.
3. Tests in `tests/unit/test_failures.py`: bit-identical at intensity 1.0; monotone effect
   trend (higher intensity → larger weight spread); validation failures.

Rules: you own falcon/failures + its tests + the case YAML; nothing else. Full suite green
(`.venv/Scripts/python -m pytest tests -q`). No git commit.
