# Results summary

Golden set verified by human: False

| metric | trivial | simple | noretr | agent |
|---|---|---|---|---|
| Intent accuracy (95% CI) | 0.160 [0.110, 0.210] | 0.795 [0.740, 0.850] | 0.800 [0.745, 0.855] | 0.800 [0.745, 0.855] |
| Intent macro-F1 (95% CI) | 0.028 [0.020, 0.035] | 0.815 [0.756, 0.862] | 0.817 [0.754, 0.867] | 0.817 [0.754, 0.867] |
| Escalate precision | 0.420 | 0.308 | 0.680 | 0.689 |
| Escalate recall | 1.000 | 0.095 | 0.405 | 0.369 |
| Escalate F1 | 0.592 | 0.145 | 0.507 | 0.481 |
| Auto-handle safety (95% CI) | 1.000 [1.000, 1.000] | 0.563 [0.488, 0.635] | 0.667 [0.596, 0.742] | 0.658 [0.582, 0.733] |
| Auto-handle rate | 0.000 | 0.870 | 0.750 | 0.775 |
| Reply constraint violations | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge: groundedness | 3.83 | 3.80 | 4.68 | 4.78 |
| Judge: actionability | 3.52 | 3.46 | 4.38 | 4.43 |
| Judge: tone | 4.18 | 3.97 | 4.80 | 4.84 |
| Judge: safety | 5.00 | 4.96 | 5.00 | 4.99 |
| Judge: pass rate (all >=4) | 0.600 | 0.520 | 0.875 | 0.900 |

## Per-class intent F1 (agent)

- feature_howto: 0.712
- subscription_management: 0.727
- praise_feedback: 0.737
- app_technical: 0.781
- billing_charges: 0.821
- content_availability: 0.826
- other: 0.833
- device_integration: 0.900
- account_access: 0.903
- account_security: 0.929
