# Task T13 — Window interventions (Plan §13.5)

**Assignee:** Kimi
**Motivation (from co-author E1 results, 2026-08-02):** E1 failures are active over a round
window (e.g. rounds 2–9), but `analyze_pair` intervenes at ONE round (the first divergent one) —
where states are still nearly identical. Measured consequence: nSRE/nSIE ≈ 0 or negative
(restoring one round of an 8-round failure can hurt), so FALCON returned `unresolved` on cases
it should localize. Single-round interventions are near-powerless against windowed failures.

**Schema (PM already added):** `InterventionSpecification.round_window: Optional[tuple[int,int]]`
(inclusive; `round_id` ignored when set).

## Deliverables

1. `falcon/intervention/engine.py`: when `round_window=[t1,t2]` is set, ONE replay of the target
   run in which the stage state is replaced at EVERY round in the window (source = the matched
   run's recorded state for the same round). Validation per round as today; any invalid round →
   whole intervention `valid=False, reason="...:<round>"` (no partial windows). Sham with a
   window = per-round serialization round-trip of live states, same drift rules as T4T5T8-F.
   `outcome_metrics` keeps final metrics + `round_<t>_<metric>` for t1 and t2.
2. CLI: `--round-window t1:t2` (mutually exclusive with `--round`).
3. Tests (`tests/interventions/test_window.py`): windowed restore over a full T4 failure window
   recovers the reference metric substantially better than the single-round restore at the first
   active round (assert both directions on the same pair); windowed inject reproduces degradation;
   windowed sham zero-deviation; invalid round inside window rejects whole spec; window ∩
   recorded rounds validation.

Rules: you own falcon/intervention + falcon/pipeline; don't touch reporting/analyzer (other
developer updates analyze_pair concurrently in T14). Full suite green except possibly
reporting-side files mid-flight. No git commit.
