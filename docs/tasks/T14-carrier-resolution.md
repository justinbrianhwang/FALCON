# Task T14 — Carrier-chain resolution + window-aware analyze_pair

**Assignee:** Codex
**Motivation (from co-author E1 results, 2026-08-02):** two systematic patterns:

1. **Carrier ties:** for a selection failure, selection-restore and aggregation-restore produce
   IDENTICAL effects (aggregation carries all upstream damage — exactly Plan §16.3 / T3). The
   analyzer sees a tie and outputs `unresolved between selection,aggregation`, although §15.2's
   temporal rule exists precisely to resolve this: earliest-divergent stage = origin, strictly
   downstream tied stages = carriers.
2. **Zero-beats-negative ranking:** when all real effects are ≤ 0 (single-round intervention on a
   windowed failure), stages with NO evidence (0.0) outrank the ground-truth stage and the output
   is `unresolved between selection,local` — nonsense ordering.

## Deliverables (falcon/attribution/analyzer.py, falcon/reporting/analyze.py, reporting)

1. **Carrier-chain rule:** when the within-margin tied set consists of the pair's
   `first_divergence_stage` plus ONLY stages strictly downstream of it (pipeline order), resolve
   to `outcome="unique_origin"` with the first-divergent stage as origin; tied downstream stages
   get role `carrier_or_amplifier`; add note `CARRIER_TIE_RESOLVED:<downstream,...>`. Ties
   involving any stage NOT downstream of the first-divergent stage stay `unresolved`.
2. **Evidence gating in ranking:** a stage whose |nSRE| and |nSIE| are both < bystander_threshold
   AND which is not the first-divergent stage is labeled `bystander` and ranked BELOW any stage
   with material (even negative) effects; negative-dominated outcomes (max score ≤ 0 across
   stages with evidence) → `outcome="unresolved"` + note `NO_POSITIVE_EVIDENCE_AT_ROUND` (signals
   the window problem rather than fabricating an ordering).
3. **Window-aware analyze_pair:** when the failure's `active_rounds` window is wider than one
   round (or pair divergence persists across rounds), issue WINDOW interventions
   (`round_window=[t1,t2]`, implemented concurrently by the other developer in T13 — code against
   the schema field, coordinate via the spec) instead of / in addition to the single-round ones;
   record which flavor produced each result in stage_effects (`window: 1.0` marker).
4. Reporting: carrier resolutions and NO_POSITIVE_EVIDENCE render with clear language; benchmark
   verdict counts a carrier-resolved unique origin as a normal prediction.
5. Tests: reconstruct the three co-author patterns as unit fixtures (selection/aggregation
   carrier tie → resolved to selection; selection,local zero-tie with negative compression →
   unresolved + NO_POSITIVE_EVIDENCE; downstream-only tie including a non-downstream stage →
   stays unresolved). End-to-end test updated for window interventions once T13 lands.

Rules: you own attribution/reporting/metrics + their tests; don't touch intervention/pipeline.
Full suite green when both T13/T14 have landed (note cross-team status explicitly). No git commit.
