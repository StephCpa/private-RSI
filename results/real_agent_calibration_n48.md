# Expanded real-agent Mode L calibration

Run date: 2026-10-02. Remote host: `gfkd-NF5468M6` (`gpu-gfkd`), GPU0.
Model: `/home/wlwuser/LZN/models/qwen2.5-7b-instruct`.

This is the expanded non-DP pilot with 24 public training tenants, 24 private
training tenants and 48 test tenants. The public and private strategies were
distilled separately, then evaluated on the same test-tenant support and query
tasks. Each condition has 192 query decisions; both parse rates are 1.0.

| condition | successes / queries | query success | parse rate |
|---|---:|---:|---:|
| public strategy | 139 / 192 | 0.7240 | 1.0000 |
| private strategy | 127 / 192 | 0.6615 | 1.0000 |

The private-minus-public difference is **−0.0625**. At the tenant level,
private strategy wins on 2 tenants, ties on 34 and loses on 12. A deterministic
200,000-resample paired bootstrap over the 48 tenants gives a 95% percentile
interval of **[-0.1042, -0.0208]**. This run therefore does not support a
positive private-meta gain; it also shows that the earlier 12-tenant positive
pilot was unstable.

The result is still a calibration, not a G1 decision: the current task uses a
one-shot A/B answer while the generated strategies describe an interactive
"try A, then B if rejected" procedure. The next run must fix candidate
strategies before seeing the test set and evaluate the tool interaction that
the strategy actually specifies. Raw data are in
[`real_agent_calibration_n48.json`](real_agent_calibration_n48.json).
