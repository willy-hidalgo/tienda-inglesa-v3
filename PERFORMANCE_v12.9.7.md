# v12.9.7 — Multi-Cutoff Temporal Robustness Audit

## Scope

Diagnostic-only release. The promoted v12.9.6 production policy is unchanged.

## Temporal replay

For Sec23 Value, the audit replays the L4 long-horizon stress-gated policy at historical pseudo-OOS cutoffs. For cutoff `b`, only blocks `b+1..N` are used to construct the policy; block `b` is held out for validation. With 12 materialized blocks and 8 minimum training folds this yields up to four causal replay cutoffs.

The replay applies the same v12.9.6 L4 stress thresholds, high-impact guards and 15% whole-segment volume budget. It records overall gain vs v11, selected volume share and BIAS deterioration per held-out cutoff.

## Acceptance

Section 30 reports replay cutoffs, win rate, median/worst/weighted gain, worst BIAS deterioration, median selected share and the per-fold vectors. The current OOS is excluded from all replay construction and remains the final external evaluation.
