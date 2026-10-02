# Minimal DP kernel audit record

Date: 2026-10-02

## Scope

The v0 kernel is a gamma=1 prototype for bounded add/remove sums. It exposes:

- Gaussian release-all, with one ledger event per candidate and conservative
  per-event Gaussian calibration for `epsilon <= 1`;
- exponential winner-only selection, returning only a candidate ID;
- scalar clipping to `[-1, 1]`, malformed-value mapping to `0`, fixed-plan
  basic composition, atomic budget checks, and persisted ledger state.

The default mechanism RNG is `secrets.SystemRandom`. An injected RNG appears
only in tests and is never passed to candidate code.

## Evidence

`python -m unittest discover -s dprae/kernel -v` passes **5/5** tests:

1. clipping and malformed-value handling;
2. all Gaussian events reserved before any output;
3. atomic refusal after a budget overrun;
4. duplicate-event refusal and no budget reset after reload;
5. winner-only output does not include scores.

The existing calibration, replay, and MT-Ops suites remain green separately.

## Explicit limits

This is not yet the production DP kernel and does not establish G2. It does
not implement or verify Poisson subsampling amplification, SVT, RDP/PLD
accounting, sandbox/network isolation, timing and token padding, provenance
static checks, canary scanning, or adversarial-improver red teaming. The ledger
uses basic composition and requires the caller to declare a fixed plan. These
items must be added and audited before any DP-RAE privacy claim.
