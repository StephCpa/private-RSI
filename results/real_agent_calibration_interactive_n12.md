# Interactive real-agent Mode L calibration (n=12)

This run uses the same Qwen2.5-7B strategy-distillation setup as the one-shot
pilot, but fixes the protocol mismatch: when the model chooses direct action A
and the hidden convention requires the extra check, the environment returns
`POLICY_CHECK_REQUIRED` and the model receives one corrective second step.

| condition | successes / queries | query success | model attempts | parse rate |
|---|---:|---:|---:|---:|
| public strategy | 30 / 48 | 0.6250 | 56 | 1.0000 |
| private strategy | 38 / 48 | 0.7917 | 59 | 1.0000 |

The private-minus-public difference is **+0.1667**. Private strategy wins on 3
tenants, ties on 9 and loses on 0. A deterministic 200,000-resample paired
bootstrap over the 12 tenants gives **[0.0000, 0.3750]** as its 95%
percentile interval; the lower endpoint is zero, so this pilot does not pass
the G1 non-DP gate. The expanded n=48 interactive run is the next confirmatory
step. Raw output: [`real_agent_calibration_interactive_n12.json`](real_agent_calibration_interactive_n12.json).
