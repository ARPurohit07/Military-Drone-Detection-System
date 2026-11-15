# Evaluation reports

These text files are recorded outputs from the project experiments. They are
evidence for the figures in the main [README](../README.md), not inputs to the
app. Re-run the commands in the README to reproduce current test-set results.

| Question | Report |
|---|---|
| How much did hard-negative training help? | `binary_v3_baseline_corrected_test.txt`, `binary_v3_retrained_test.txt` |
| How did bird-flock mining affect the final detector? | `binary_v3_before_bird_mining_test.txt`, `binary_v5_birds_test.txt` |
| Does the final detector hold up on other test sets? | `binary_v5_on_original_test.txt`, `binary_v5_cross_domain_country_test.txt` |
| How well does country attribution work? | `country_v3_test.txt`, `country_v3_diagnosis.txt` |

Earlier baselines, smoke runs, and intermediate experiments remain local. Example
predictions in `examples/` and generated files in `cache/` are excluded from the
code-only release.
