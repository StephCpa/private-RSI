# Interactive Mode L calibration, seed 20261004

This independent seed repeats the n=48 interactive protocol with Qwen2.5-7B,
24 public training tenants, 24 private training tenants and 48 test tenants.

| condition | successes / queries | query success | attempts | parse rate |
|---|---:|---:|---:|---:|
| public strategy | 145 / 192 | 0.7552 | 210 | 1.0000 |
| private strategy | 144 / 192 | 0.7500 | 212 | 1.0000 |

Private minus public is **−0.0052**. Private strategy wins on 3 tenants, ties
on 40 and loses on 5. The deterministic 200,000-resample paired bootstrap
95% interval is **[-0.0365, 0.0313]**, which includes zero.

Raw output: [`real_agent_calibration_interactive_seed20261004.json`](real_agent_calibration_interactive_seed20261004.json).
