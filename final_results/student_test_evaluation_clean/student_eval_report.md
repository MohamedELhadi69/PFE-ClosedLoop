# Student Test Evaluation

## Reliability

- Reference test rows (deduped): 6400
- Reference successful rows: 6400
- Student test rows (deduped): 6400
- Student successful rows: 6400
- Common scored rows: 6400
- Student pipeline success vs reference-ok rows: 100.0000%
- End-to-end system score (coverage x content): 0.7517

## Scoring Policy

- Anomaly rows used for remediation-quality scoring: 247
- Normal rows excluded from remediation-quality scoring: 6153

## Core Quality (Anomaly Rows Only)

- full_plan_exact: 0.0040
- plan_score: 0.7517
- remedy_f1: 0.9150
- prevention_f1: 0.9150
- verification_f1: 0.9150
- severity_exact: 0.5668
- urgency_exact: 0.6680
- summary_token_f1: 0.5901
- root_cause_token_f1: 0.6400
- impact_token_f1: 0.5961

## Per Anomaly Type

| anomaly_type | reference_ok | student_ok | success_rate | plan_score | full_plan_exact | remedy_f1 | prevention_f1 | verification_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Co-Channel Interference (Severe) | 25 | 25 | 1.0000 | 0.8966 | 0.0000 | 0.9600 | 0.9600 | 0.9600 |
| Doppler Shift (Severe) | 14 | 14 | 1.0000 | 0.8862 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Buffer Overflow (Gradual Buildup) | 32 | 32 | 1.0000 | 0.8526 | 0.0000 | 0.9688 | 0.9688 | 0.9688 |
| Faulty RF Filters (Temporal) | 19 | 19 | 1.0000 | 0.8013 | 0.0000 | 0.9474 | 0.9474 | 0.9474 |
| Co-Channel Interference (Mild) | 38 | 38 | 1.0000 | 0.7437 | 0.0000 | 0.8684 | 0.8684 | 0.8684 |
| Resource Allocation Bugs | 11 | 11 | 1.0000 | 0.7140 | 0.0000 | 0.8182 | 0.8182 | 0.8182 |
| High Network Congestion (Sudden Spike) | 3 | 3 | 1.0000 | 0.6927 | 0.0000 | 0.6667 | 0.6667 | 0.6667 |
| High Network Congestion (Gradual Buildup) | 17 | 17 | 1.0000 | 0.6880 | 0.0000 | 0.7059 | 0.7059 | 0.7059 |
| Faulty Handover Algorithm (Too Frequent) | 11 | 11 | 1.0000 | 0.6847 | 0.0000 | 0.9091 | 0.9091 | 0.9091 |
| Jamming | 55 | 55 | 1.0000 | 0.6698 | 0.0182 | 0.9636 | 0.9636 | 0.9636 |
| Antenna Failure | 22 | 22 | 1.0000 | 0.6403 | 0.0000 | 0.9091 | 0.9091 | 0.9091 |

## Error Summary

No error rows.