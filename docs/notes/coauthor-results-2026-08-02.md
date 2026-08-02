# Co-author machine results — 2026-08-02

Machine: DESKTOP-2B048I1, Windows 10.0.26200, Python 3.11.15 (conda env from environment.yml).

## Suite 1 (cross-machine determinism, disjoint seeds, heterogeneity)

- **Cross-machine replay: `bitwise-portable`** — all 15 recorded stage-boundary hashes of the
  calibrated reference config match the main machine's golden fixture exactly
  (verified via `scripts/compare_crossmachine.py` on `Output_2026-08-02_14-45-13.zip`).
- Sham deviation 0.0 in every case on the second machine.
- FALCON attribution never produced a confidently wrong unique origin across seeds 101–105 and
  heterogeneity 0.5/1/2; failures of localization surfaced honestly as `unresolved`
  (root causes fixed in T13/T14: single-round interventions vs windowed failures; carrier ties).
- Passive baseline mislocalized minority_exclusion as `local` in all runs; terminal-only flipped
  between selection/aggregation across seeds — early H2/H5-direction evidence.

## Suite 2 (severity calibration + scale/cost)

Severity-vs-accuracy-gap curves (identical to the main machine's values — metric-level
cross-machine determinism):

| stage | mild | moderate | severe |
|---|---:|---:|---:|
| selection (exclusion_probability .3/.6/.9) | +0.008 | +0.042 | +0.070 |
| local (lr_multiplier .1/.01/−1) | +0.026 | +0.034 | +0.142 |
| compression (k_ratio .5/.2/.05) | +0.012 | +0.046 | +0.126 |
| aggregation (uniform/swapped/corrupted) | 0 | 0 | +0.070 |

Calibration findings:

- selection/local/compression have valid monotone severity axes (class-1 gaps roughly 2× global).
- lr INCREASE (3×/10×) improves this task — excluded from the severity axis (measured on main).
- aggregation `uniform`/`swapped` are provable no-ops under equal per-client sample counts;
  only `corrupted` is a valid severity level. A future intensity knob for `corrupted` (or unequal
  sample counts) is needed for a 3-level aggregation axis.

Scale/cost (their hardware; main machine ≈ 2.5× faster, storage identical):

| clients | wall (s) | recorded |
|---:|---:|---:|
| 10 | 0.31 | 157 KB |
| 25 | 0.53 | 353 KB |
| 50 | 0.94 | 710 KB |

Both time and storage scale ~linearly with client count (RQ6 baseline data point).
