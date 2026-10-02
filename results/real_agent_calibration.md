# Small real-agent Mode L calibration

Run date: 2026-10-02. Remote host: `gfkd-NF5468M6` (`gpu-gfkd`), GPU0.
Model: `/home/wlwuser/LZN/models/qwen2.5-7b-instruct`.

This is a non-DP pilot using 8 public training tenants, 12 private training
tenants and 12 test tenants. Qwen2.5-7B-Instruct distilled one tenant-neutral
strategy from public support traces and another from private training support
traces. Both strategies were evaluated on the same 12 test tenants, the same
8 support tasks per tenant, the same 4 query tasks, greedy decoding and the
same prompt budget. Rule IDs and canaries were withheld from model prompts.

| condition | successes / queries | query success | parse rate |
|---|---:|---:|---:|
| public strategy | 29 / 48 | 0.6042 | 1.0000 |
| private strategy | 34 / 48 | 0.7083 | 1.0000 |

The paired difference is **+0.1042**. At the tenant level, private strategy
wins on 3 tenants, ties on 7 and loses on 2. A deterministic 200,000-resample
paired bootstrap over the 12 tenants gives a 95% percentile interval of
**[-0.0833, 0.3333]**, so the interval includes zero. This is an E2/G1 pilot
signal, not evidence that the non-DP meta-gain gate has passed.

The raw result and dataset hash are in [`real_agent_calibration.json`](real_agent_calibration.json).
The remote process exited normally and GPU0 was returned to its prior state.
No DP mechanism was applied in this run; the next experiment must repeat the
contrast with more test tenants and fixed pre-registered candidate strategies
before using it to choose the DP mechanism.
