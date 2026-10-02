# Interactive Mode L calibration, seed 20261003

This independent seed repeats the n=48 interactive protocol with Qwen2.5-7B,
24 public training tenants, 24 private training tenants and 48 test tenants.

| condition | successes / queries | query success | attempts | parse rate |
|---|---:|---:|---:|---:|
| public strategy | 137 / 192 | 0.7135 | 213 | 1.0000 |
| private strategy | 133 / 192 | 0.6927 | 213 | 1.0000 |

Private minus public is **−0.0208**. Private strategy wins on 3 tenants, ties
on 39 and loses on 6. The deterministic 200,000-resample paired bootstrap
95% interval is **[-0.0573, 0.0104]**, which includes zero. Together with the
other n=48 seed (+3.125 pp), this shows that the current free-form strategy
distillation protocol is not yet a stable estimate of private-meta gain.

Raw output: [`real_agent_calibration_interactive_seed20261003.json`](real_agent_calibration_interactive_seed20261003.json).
