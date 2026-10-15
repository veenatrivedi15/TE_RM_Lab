# CO2 Emission Prediction Using Machine Learning

Reproducible code and reference outputs for one-month-ahead forecasting of
facility-level CO2 emissions from Indian cement plants.

This repository is an analysis package, not a deployed application. It
contains the data extracts, forecasting code, conformal prediction-interval
analyses, validation tests, figures, and generated manuscript materials used
for the current V2 research workflow.

## Project summary

The project evaluates whether time-ordered machine-learning forecasts improve
on transparent forecasting baselines while preserving leakage-safe validation.
It also examines forecast errors by plant size and regime, compares the panel
inventory with an external cement-production index, and evaluates empirical
prediction-interval coverage.

The primary forecasting design is:

- 166 Indian cement facilities.
- Monthly observations from January 2021 through July 2026.
- One-month-ahead targets.
- Features available at the forecast origin only.
- A 12-origin validation window for feature/model selection.
- A later 13-origin hold-out test period.
- Expanding-window rolling-origin evaluation for the extended benchmark.
- Fixed random seed: `42`.

The evaluation period and model-selection protocol are fixed by the existing
research design. Results should not be interpreted as evidence of production
deployment or causal effects.

## Data

### Modeling data

`data/modeling_dataset_step1.csv` is the committed Climate TRACE v5.11.0
extract used by the forecasting pipeline. It contains monthly facility-level
CO2 emissions and the prepared lag/features used by the analysis.

Source: [Climate TRACE](https://climatetrace.org/explore). The final paper
should record the inventory access date and the applicable data terms.

### External comparison data

`data/T13_external_ICI_cement_PIB.csv` is the source-of-truth external series
for the Index of Eight Core Industries cement comparison. It is transcribed
from PIB Release ID 2276691, Annexure II. The source file contains 13 monthly
observations and marks the provisional observation explicitly.

The external-data analysis reads this committed CSV. The observations are not
duplicated in the analysis code.

## Methods

### Forecast models and baselines

The pipeline evaluates persistence and calendar-adjusted persistence baselines,
moving averages, seasonal-naive forecasts, linear regression, random forests,
HistGradientBoosting, and change-prediction variants. The extended rolling
benchmark additionally evaluates SES, Theta, daily-rate models, and fixed
hyperparameter variants.

Metrics include MAE, RMSE, WAPE, bias, MASE, R2 where applicable, and
dependence-aware comparisons using month-block bootstrap and Diebold-Mariano
tests with multiple-testing correction.

### Prediction intervals

The interval analyses include:

- Quantile gradient-boosting intervals.
- Conformalised quantile regression (CQR).
- Plant-size-stratified conformal intervals.
- Rolling static split conformal intervals.
- Adaptive Conformal Inference (ACI).
- Size-conditional Mondrian conformal intervals.

The shared helper in `code/conformal_utils.py` uses the finite-sample
order-statistic rule

```text
rank = ceil((n + 1) * coverage)
```

If the required rank exceeds the calibration sample size, the helper returns
an unbounded correction rather than silently capping the rank. Interval bounds
are clipped and ordered for the nonnegative emissions target.

Reported interval coverage is empirical coverage on the saved test or
rolling-origin rows. It is not a claim of conditional coverage or a universal
guarantee for plant-size groups.

## Current reference results

The checked-in CSV files under `results/tables/` are generated reference
outputs. Selected values from the current 13-origin hold-out include:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Persistence | 12,206.54 | 23,710.03 | 10.66% |
| Calendar-adjusted persistence | 10,192.11 | 22,316.17 | 8.90% |
| Linear Regression | 10,859.82 | 20,041.19 | 9.48% |
| Random Forest | 10,315.48 | 20,781.70 | 9.01% |
| HistGradientBoosting | 10,517.25 | 20,203.75 | 9.18% |

The extended 43-origin benchmark should be read from
`results/tables/T20_benchmark_accuracy.csv`; it preserves comparisons in
which simple baselines can outperform more complex models.

For the main saved test-set intervals:

- Raw 10-90% quantile interval coverage: approximately 68.86%.
- CQR coverage: approximately 82.44%.
- Nominal coverage: 80%.

For the primary Stage 3 configurations:

- Adaptive RF static coverage: approximately 90.12%.
- RF ACI with `gamma=0.01`: approximately 90.21%.
- Mondrian overall coverage: approximately 89.13%.
- Mondrian Q1-small coverage: approximately 84.08%.

These are empirical results from this dataset and evaluation design. The
unfavourable subgroup and zero-target findings remain part of the results.

## Repository layout

```text
data/       Committed modeling and external source data
code/       Forecasting, interval, figure, and manuscript-generation scripts
tests/      Integrity, leakage, conformal, and row-artifact tests
results/    Checked-in tables, figures, and reference predictions
run_all.sh  End-to-end reproducibility workflow
work/       Generated working directory; not a release input
```

Important scripts:

- `code/stage1_pipeline.py`: main audit, model selection, hold-out forecast,
  rolling results, robustness, multi-horizon results, and main intervals.
- `code/stage1b_external_and_intervals.py`: external ICI comparison and
  size-stratified CQR analysis.
- `code/stage2_figures.py`: generated figures.
- `code/stage3_benchmark.py`: extended expanding-window benchmark.
- `code/stage3b_inference.py`: uncertainty and forecast-comparison tests.
- `code/stage3d_adaptive_conformal.py`: static and adaptive conformal rows.
- `code/stage3e_mondrian_conformal.py`: size-conditional conformal rows.
- `code/build_spec.py`: manuscript specification generated from result files.
- `code/render_docx.js`: optional DOCX renderer.

## Installation

Use Python 3.12 or a compatible Python environment with the packages listed
in `requirements.txt`.

```bash
python3 -m pip install -r requirements.txt
```

DOCX generation is optional and requires Node.js plus the locked npm
dependency:

```bash
npm install
```

The Python and Node environments used for a specific release should be
recorded separately from the research outputs.

## Reproduction

Run from the repository root:

```bash
bash run_all.sh
```

If multiple Python installations are available, select the tested interpreter:

```bash
PYTHON=/path/to/python bash run_all.sh
```

On Windows with Git Bash, use the Git Bash path form, for example:

```bash
PYTHON=/c/Users/your-name/AppData/Local/Programs/Python/Python312/python.exe \
bash run_all.sh
```

The workflow:

1. Runs the integrity and no-future-information tests.
2. Creates or reuses the generated `work/` directory.
3. Copies source code and source data into the isolated working directory.
4. Runs all forecasting, interval, benchmark, figure, and manuscript stages.
5. Writes generated outputs to `work/out/`.
6. Builds the DOCX when Node.js and the `docx` package are available.

For a fresh recomputation:

```bash
CLEAN_WORK=1 PYTHON=/path/to/python bash run_all.sh
```

`CLEAN_WORK=1` removes only the generated `work/` directory. Without it, the
extended Stage 3 benchmark remains resumable and may reuse existing
`work/out/preds/s3/origin_*.csv` files.

The workflow does not overwrite the checked-in `results/` directory. To update
reference outputs after a deliberately reviewed clean run, copy only the
validated artifacts from `work/out/` into `results/`.

## Reproducible Stage 3 artifacts

The Stage 3 scripts persist row-level interval calculations:

- `results/preds/stage3_adaptive_intervals.csv`
- `results/preds/stage3_mondrian_intervals.csv`

These files include actual targets, predictions, lower and upper bounds,
widths, coverage indicators, origins, plant identifiers, calibration sample
sizes, correction values, and relevant calibration metadata.

The corresponding aggregate summaries are:

- `results/tables/T30_conformal_summary.csv`
- `results/tables/T31_conformal_coverage_by_group.csv`
- `results/tables/T32_aci_path.csv`
- `results/tables/T33_mondrian_by_size.csv`

The tests independently verify that bounds, widths, coverage indicators, and
aggregate coverage/width summaries agree with the saved rows.

## Validation commands

Run the complete test suite:

```bash
python3 -m pytest -q tests
```

Compile the Python sources:

```bash
python3 -m compileall -q code tests
```

Check Git whitespace:

```bash
git diff --check
```

The clean workflow is the authoritative end-to-end check:

```bash
CLEAN_WORK=1 PYTHON=/path/to/python bash run_all.sh
```

## Limitations and responsible interpretation

- The analysis is retrospective and does not demonstrate operational
  deployment performance.
- Empirical interval coverage is not conditional or guaranteed group-wise
  coverage.
- Coverage is uneven across plant-size and target regimes; zero-target rows
  remain a documented limitation.
- The external ICI series is an aggregate comparison and is not a facility
  level validation set.
- Climate TRACE inventory values and official production-index values have
  distinct definitions, units, revisions, and aggregation procedures.
- Results depend on the committed dataset, evaluation period, fixed feature
  protocol, and fixed model settings.
- Publication claims, authorship, funding, acknowledgements, licensing, and
  DOI information must be completed from legitimate records before
  submission.

## Citation and provenance

Before publication, add:

- The final author list and affiliations.
- The exact Climate TRACE inventory citation and access date.
- The official PIB source citation and access date.
- Data licensing and redistribution permissions.
- Funding and acknowledgement statements.
- A permanent repository DOI, such as a Zenodo record.

Do not treat the current repository metadata as a substitute for those
project-specific records.

## License

No license file is currently included. The repository owner should add an
appropriate software and data-use license after confirming the terms for all
included and derived data.
