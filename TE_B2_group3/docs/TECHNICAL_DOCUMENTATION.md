# Technical Documentation

## 1. Project title

**Machine Learning-Based Prediction of CO₂ Emissions for Indian Cement Facilities**

This project implements a reproducible research workflow for retrospective one-month-ahead forecasting of facility-level CO₂ emissions estimates for Indian cement facilities. The emissions values are modelled/estimated data and must not be described as direct plant-meter readings or independently verified ground-truth measurements.

## 2. Problem statement

Cement production is an important industrial source of carbon dioxide. Analysts need methods to examine facility-level emissions patterns and evaluate whether recent observations and other available information can support short-horizon forecasts. This project develops and evaluates machine-learning and baseline forecasting methods using a monthly facility panel, with time-ordered validation and a separate holdout window. It also examines prediction uncertainty, forecast horizons, robustness, and an external cement-industry indicator.

## 3. Objectives

- Prepare the supplied modelling dataset for analysis.
- Evaluate forecasting approaches using time-ordered validation rather than random train/test shuffling.
- Compare model predictions with simple baseline forecasts.
- Assess model errors across rolling forecast origins and relevant facility or operating regimes.
- Produce figures, tables, prediction files, uncertainty-interval diagnostics, and a generated research manuscript.
- Make the workflow repeatable through version-controlled code, declared dependencies, and automated tests.

## 4. System requirements

### Software

- Windows, Linux, or macOS; on Windows, Git Bash is used to execute `run_all.sh`.
- Python 3.12 or a compatible Python version.
- Git.
- Node.js and npm are required only for the optional DOCX manuscript-generation step.

### Python packages

The project declares the following packages in `requirements.txt`: NumPy, pandas, SciPy, scikit-learn, statsmodels, Matplotlib, Pillow, and pytest. The document-generation step also uses the npm package `docx`, declared through `package.json`/`package-lock.json`.

For more reproducible installations, dependency versions should be pinned after the environment has been validated.

## 5. Technologies used

- **Python:** pipeline implementation and analysis.
- **NumPy and pandas:** numerical operations, tabular data processing, and summaries.
- **SciPy and statsmodels:** statistical calculations and inference.
- **scikit-learn:** machine-learning estimators and evaluation support.
- **Matplotlib and Pillow:** figures and image handling.
- **pytest:** automated tests.
- **Git/GitHub:** source control, version history, and collaboration.
- **Node.js/npm and `docx`:** optional Word manuscript generation.

## 6. Project structure

```text
TE_B2_group3/
├── code/                 # Pipeline and analysis scripts
├── data/                 # Input modelling data and external indicator files
├── results/
│   ├── figures/          # Reference figures
│   ├── preds/            # Reference predictions
│   └── tables/           # Reference result tables
├── tests/                # Automated tests
├── docs/                 # Technical documentation and supporting documents
├── notebooks/            # Reserved for notebooks, if needed
├── src/                  # Reserved for reusable source modules, if needed
├── README.md             # Project overview and quick-start information
├── requirements.txt      # Python dependencies
├── package.json          # Optional DOCX-build dependency declaration
├── package-lock.json     # npm dependency lock file
├── run_all.sh            # End-to-end reproducibility workflow
└── .gitignore            # Excludes transient files from version control
```

The workflow creates a temporary `work/` directory at runtime. It contains working copies, generated intermediate outputs, and the built manuscript; it should remain excluded from version control. Checked-in reference artifacts belong under `results/`.

## 7. Architecture

The project is organised as a staged analysis pipeline:

1. **Data and modelling preparation** — load the modelling panel and prepare the inputs used by the forecasting stages.
2. **Model evaluation** — select a feature set and model using the validation procedure, then evaluate on the designated holdout period.
3. **Rolling-origin analysis** — repeat forecasts over sequential origins to evaluate performance across time.
4. **Inference and error analysis** — compare errors against baselines and investigate heterogeneity by facility size, zero-emission observations, restarts, and calendar period.
5. **Uncertainty analysis** — build and assess prediction intervals, including coverage and interval width across groups and origins.
6. **Forecast-horizon analysis** — evaluate horizons including 1, 2, 3, 6, and 12 months.
7. **Reporting** — generate figures, tables, summaries, and, when dependencies are installed, a DOCX manuscript.
8. **Automated testing** — run the test suite to check key utilities, data integrity, and output expectations.

The main pipeline scripts are in `code/`; `run_all.sh` orchestrates their execution. Runtime outputs are written under `work/` to avoid overwriting the checked-in reference results.

## 8. Methodology

The analysis uses monthly facility-level observations and a time-ordered design. A validation period is used for feature/model selection, and a later holdout period is used for out-of-sample evaluation. Rolling-origin evaluations provide additional evidence across sequential forecast dates. The workflow compares Random Forest and Histogram Gradient Boosting models with baseline forecasting approaches, including persistence and calendar-adjusted persistence, plus other benchmark methods included in the pipeline.

Performance summaries include metrics such as mean absolute error (MAE), root mean squared error (RMSE), weighted absolute percentage error (WAPE), and scaled error measures where available. Statistical comparisons and block-based uncertainty summaries are used in the supplied inference stages. Prediction intervals are assessed using empirical coverage and interval width, including subgroup and origin-level diagnostics.

The pipeline output identifies Random Forest as the model selected on the validation procedure for the latest successful run. This must not be interpreted as proof that Random Forest is best under every metric or evaluation window. For example, in the displayed 13-origin test-window summary, calendar-adjusted persistence had MAE 10,192.107 and WAPE 8.899%, while the daily-rate Random Forest had MAE 10,307.449 and WAPE 9.000%; the daily-rate Random Forest had lower RMSE (20,599.601 versus 22,316.168). These differences illustrate why model conclusions should be metric- and window-specific.

## 9. Installation

Open CMD in the project directory and create an isolated virtual environment outside the repository:

```bat
if not exist "%USERPROFILE%\venvs" mkdir "%USERPROFILE%\venvs"
python -m venv "%USERPROFILE%\venvs\te_b2_group3"
"%USERPROFILE%\venvs\te_b2_group3\Scripts\python.exe" -m pip install --upgrade pip
"%USERPROFILE%\venvs\te_b2_group3\Scripts\python.exe" -m pip install -r requirements.txt
```

Install the optional document-generation dependency from the project directory:

```bat
npm install
```

## 10. Execution

From the project directory, run the tests and syntax check:

```bat
"%USERPROFILE%\venvs\te_b2_group3\Scripts\python.exe" -m pytest -q tests
"%USERPROFILE%\venvs\te_b2_group3\Scripts\python.exe" -m compileall -q code tests
```

Then execute the complete workflow from CMD using Git Bash and explicitly set the Python interpreter so the script uses the virtual environment:

```bat
set "PYTHON=%USERPROFILE%\venvs\te_b2_group3\Scripts\python.exe"
"C:\Program Files\Git\bin\bash.exe" run_all.sh
```

A successful run should show the test summary and finish with `ALL DONE`. The DOCX build is optional; when `npm install` has succeeded, the pipeline can attempt to create `Manuscript_CO2_Cement_Forecasting.docx` under the generated `work/` directory.

## 11. Inputs and outputs

### Inputs

- `data/modeling_dataset_step1.csv` — main modelling dataset.
- `data/external_ici.csv` and `data/T13_external_ICI_cement_PIB.csv` — external industry-indicator inputs used by the external comparison stage.
- Python scripts under `code/` and configuration/build files at the project root.

### Outputs

- Figures, prediction files, and result tables under `results/` provide checked-in reference artifacts.
- The end-to-end run creates intermediate and regenerated outputs under `work/` (including `work/out/`).
- The document-building stage may create `work/Manuscript_CO2_Cement_Forecasting.docx` and a manuscript specification JSON file.
- The console log reports model metrics, diagnostics, external-indicator comparison summaries, and completion status.

Generated working files should not be committed by default. Copy a specific generated artifact to `docs/` or `results/` only when it is intentionally part of the assessed deliverable and its provenance is clear.

## 12. Latest execution check

On the author's Windows environment (Python 3.12.10; Node.js v24.13.0; npm 11.6.2), the recorded end-to-end run completed with `ALL DONE`. The automated test suite reported **10 passed**. The DOCX file was present at `work/Manuscript_CO2_Cement_Forecasting.docx`, with a reported size of 1,133,012 bytes. The test and workflow results document one successful run; they do not, by themselves, prove all scientific assumptions or conclusions are valid.

## 13. Limitations

- The target represents Climate TRACE modelled/estimated facility-level emissions, not direct plant-meter readings or verified ground truth.
- Results depend on the completeness, accuracy, and modelling assumptions of the supplied emissions and external-indicator datasets.
- Forecast performance differs by metric, period, facility size, and operating regime; no single model should be claimed to dominate universally.
- Prediction-interval coverage varies across subgroups, and aggregated coverage can hide poorer coverage for smaller facilities or zero-target observations.
- The declared Python dependencies are currently not pinned to exact versions, which can limit long-term bit-for-bit reproducibility.
- The external indicator comparison is observational and should not be interpreted as causal evidence.
- Successful execution and passing tests confirm software checks for this run, not the validity of every scientific interpretation.

## 14. Future scope

- Pin Python and Node package versions and record operating-system/environment metadata.
- Add continuous integration so tests run automatically on changes.
- Add a concise data dictionary, data provenance notes, and formal checksums for key input/output files.
- Improve subgroup calibration and evaluate interval performance for zero-target observations and smaller facilities.
- Evaluate sensitivity to data revisions, alternative holdout windows, and additional carefully justified baselines.
- Improve command-line configuration and logging while preserving the fixed experimental protocol for any results intended to reproduce the current analysis.

## 15. References

1. **Climate TRACE.** Emissions data and project dataset used as the modelled emissions source. See the source repository and data files distributed with this project for the specific data extract and processing context.
2. **CO2 Emission Prediction using ML — project repository.** https://github.com/Satyam-ptl/CO2-Emission-Prediction-using-ML
3. **scikit-learn documentation.** https://scikit-learn.org/stable/
4. **pandas documentation.** https://pandas.pydata.org/docs/
5. **NumPy documentation.** https://numpy.org/doc/
6. **pytest documentation.** https://docs.pytest.org/

---

**Document maintenance:** Update this file when implementation details, declared dependencies, dataset versions, or the reproducibility workflow change. Keep scientific claims aligned with the checked-in result artifacts and the actual execution logs.
