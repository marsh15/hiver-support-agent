# Results summary

Golden set verified by human: False

| metric | trivial | simple | noretr | agent |
|---|---|---|---|---|
| Intent accuracy (95% CI) | 0.145 [0.100, 0.195] | 0.835 [0.780, 0.885] | 0.815 [0.760, 0.870] | 0.815 [0.760, 0.870] |
| Intent macro-F1 (95% CI) | 0.025 [0.018, 0.033] | 0.856 [0.803, 0.897] | 0.834 [0.777, 0.881] | 0.834 [0.777, 0.881] |
| Escalate precision | 0.430 | 0.423 | 0.760 | 0.778 |
| Escalate recall | 1.000 | 0.128 | 0.442 | 0.407 |
| Escalate F1 | 0.601 | 0.196 | 0.559 | 0.534 |
| Auto-handle safety (95% CI) | 1.000 [1.000, 1.000] | 0.569 [0.494, 0.640] | 0.680 [0.605, 0.756] | 0.671 [0.595, 0.747] |
| Auto-handle rate | 0.000 | 0.870 | 0.750 | 0.775 |
| Reply constraint violations | 0.000 | 0.000 | 0.000 | 0.000 |
| Judge: groundedness | 3.81 | 3.71 | 4.74 | 4.84 |
| Judge: actionability | 3.52 | 3.50 | 4.42 | 4.50 |
| Judge: tone | 4.17 | 3.96 | 4.82 | 4.87 |
| Judge: safety | 5.00 | 4.96 | 5.00 | 5.00 |
| Judge: pass rate (all >=4) | 0.620 | 0.495 | 0.900 | 0.940 |

## Per-class intent F1 (agent)

- subscription_management: 0.727
- feature_howto: 0.743
- other: 0.773
- app_technical: 0.780
- billing_charges: 0.842
- praise_feedback: 0.870
- account_access: 0.875
- device_integration: 0.900
- content_availability: 0.905
- account_security: 0.929
