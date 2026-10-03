# Opaque-content procedure-transfer track

This track is a separate estimand from the primary MT-Ops v1.1
private-strategy-minus-public-strategy contrast. It asks whether a shared
improver can transfer the procedure for converting local support evidence into
an action when rule content itself cannot transfer.

## Frozen transformation

For each seed, public, private and test splits receive independent random
bijections from the original feature tuples to opaque five-field tokens. The
split namespaces are disjoint, so a feature-to-procedure table learned from
public or private training data has zero exact feature overlap with test. The
support and query tasks within each test tenant share the test mapping, so a
local support-to-action procedure remains measurable.

The procedure-transfer track freezes `transfer_fraction=0.50` (different from
the primary v1.1 value of 0.75) so that the local support-to-action route has
at least ten percentage points of pooled headroom over the better constant
baseline.

The transformation is implemented in
`benchmarks/mtops/procedure_transfer.py`. It leaves the original v1.1 dataset
and its results unchanged.

## Pre-LLM contract gates

The eight-seed audit in `analysis/procedure_transfer_contract_audit.py` must
pass all of the following:

| Gate | Requirement |
|---|---|
| Public/test disjointness | Zero exact opaque-feature overlap. |
| Private/test disjointness | Zero exact opaque-feature overlap. |
| Public/private disjointness | Zero exact opaque-feature overlap. |
| Local support headroom | Pooled support-only success exceeds the pooled better constant baseline by at least 10 pp. |
| Global table null | Public and private global tables have identical test success under exact lookup. |

These gates establish that global content lookup cannot explain a procedure
transfer result. They do not establish that an LLM can learn the procedure.
The later LLM track must use an instruction-level artifact contract rather than
the row-by-row feature lookup artifact used by the primary v1.1 experiment.

## Reproduction

```bash
python -m analysis.procedure_transfer_contract_audit
python -m unittest discover -s tests -p 'test_procedure_transfer.py' -v
```
