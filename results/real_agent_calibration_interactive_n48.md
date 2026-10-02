# Expanded interactive real-agent Mode L calibration (n=48)

This run uses Qwen2.5-7B-Instruct with 24 public training tenants, 24 private
training tenants and 48 test tenants. The corrected interaction protocol lets
the model receive `POLICY_CHECK_REQUIRED` and take one corrective second step.
Both strategies use the same test tenants, support/query split and decoding
budget.

| condition | successes / queries | query success | model attempts | parse rate |
|---|---:|---:|---:|---:|
| public strategy | 151 / 192 | 0.7865 | 204 | 1.0000 |
| private strategy | 157 / 192 | 0.8177 | 222 | 1.0000 |

The private-minus-public difference is **+0.03125**. Private strategy wins on
5 tenants, ties on 43 and loses on 0. A deterministic 200,000-resample paired
bootstrap over the 48 tenants gives a 95% percentile interval of
**[0.0052, 0.0625]**. The direction is positive, but the point estimate is
below the research plan's 5 pp G1 threshold; this run does not pass G1.

This is still a non-DP calibration. It supports a multi-seed fixed-strategy
follow-up, not a full DP-RAE experiment. Raw output:
[`real_agent_calibration_interactive_n48.json`](real_agent_calibration_interactive_n48.json).
