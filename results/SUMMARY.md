# Results summary

Golden set verified by human: False

| metric | trivial | simple | noretr | agent |
|---|---|---|---|---|
| Intent accuracy (95% CI) | 0.145 [0.100, 0.195] | 0.835 [0.780, 0.885] | 0.985 [0.965, 1.000] | 0.985 [0.965, 1.000] |
| Intent macro-F1 (95% CI) | 0.025 [0.018, 0.033] | 0.856 [0.803, 0.897] | 0.986 [0.968, 1.000] | 0.986 [0.968, 1.000] |
| Escalate precision | 0.430 | 0.423 | 0.854 | 0.786 |
| Escalate recall | 1.000 | 0.128 | 0.407 | 0.384 |
| Escalate F1 | 0.601 | 0.196 | 0.551 | 0.516 |
| Auto-handle safety (95% CI) | 1.000 [1.000, 1.000] | 0.569 [0.494, 0.640] | 0.679 [0.609, 0.753] | 0.665 [0.592, 0.741] |
| Auto-handle rate | 0.000 | 0.870 | 0.795 | 0.790 |
| Reply constraint violations | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge: groundedness | 3.81 | 3.54 | 4.73 | 4.86 |
| Judge: actionability | 3.52 | 3.17 | 4.46 | 4.46 |
| Judge: tone | 4.17 | 3.81 | 4.84 | 4.88 |
| Judge: safety | 5.00 | 4.96 | 5.00 | 5.00 |
| Judge: pass rate (all >=4) | 0.620 | 0.400 | 0.910 | 0.925 |

## Per-class intent F1 (agent)

- feature_howto: 0.951
- praise_feedback: 0.957
- billing_charges: 0.974
- content_availability: 0.977
- account_access: 1.000
- account_security: 1.000
- app_technical: 1.000
- device_integration: 1.000
- other: 1.000
- subscription_management: 1.000
