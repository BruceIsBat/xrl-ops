# XRL-Ops Benchmark Report

## Summary
- Mean reward: -52381.2746
- Reward standard deviation: 27639.9716
- Total constraint violations: 21455
- Temperature violations: 21329
- Pressure violations: 126
- Safety check: FAIL

## Safety Assessment
The benchmark is marked as **FAIL** based on the configured violation tolerances for the control envelope.

- Temperature threshold: 180.0
- Pressure threshold: 260.0
- Pass condition: temperature violations <= 10 and pressure violations <= 10

## Notes
This report is generated from empirical rollouts of the trained PPO policy under fixed environment seeds. It is intended to support safe deployment review, failure analysis, and policy auditability.
