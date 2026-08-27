# Analysis tools

`analyze_workload.py` converts optional raw routing captures into launch-M,
per-active-expert-M, padding, and configuration summaries.

`analyze_serving.py` takes the small baseline and optimized Serving CSVs,
requires exactly three runs for C8/C16/C32, and reports gains from the median
metric at each concurrency. It never selects a best run.

Raw traces are not included due to size. The checked-in `results/` and `docs/`
files are sufficient to audit every headline number in the README.
