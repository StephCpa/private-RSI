# MT-Ops v0: WP0 G0 evidence

Generated on 2026-10-02 by `python benchmarks/mtops/run_g0.py`.

The benchmark is a deterministic, LLM-free simulator. It has 300 public,
2,000 private, and 500 test tenants; four task families; 40 shared rules; and
one tenant-local rule per tenant. At `overlap=0`, public-only agents receive no
private rule pool. Support tasks expose shared conventions while the first
query task holds out the tenant-local convention, so local-only adaptation has
room below the oracle.

| agent | test query success |
|---|---:|
| public-only | 0.000 |
| local adaptation | 0.482 |
| oracle | 1.000 |

The measured oracle minus public-only gap is **1.000**, exceeding the planned
G0 threshold of 0.15. This is benchmark validity evidence only; it is not a
DP-RAE result and contains no LLM calls.

The overlap sweep is in [`mtops_validity_sweep.json`](mtops_validity_sweep.json):
public-only success increases monotonically from 0.000 at overlap 0 to 0.485
at overlap 1, while the oracle remains 1.000. The full dataset manifest and
SHA-256 are stored in [`mtops_g0.json`](mtops_g0.json).

Validation commands:

```text
python -m unittest discover -s benchmarks/mtops -v
python benchmarks/mtops/run_g0.py
python benchmarks/mtops/run_validity_sweep.py
python -m unittest discover -s tests -v
```
