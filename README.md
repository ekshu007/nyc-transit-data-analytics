# NYC Subway Segment Analytics

A portfolio project exploring NYC subway segment travel times through
**exploratory data analysis, SQL analytics, and machine learning**. The
project analyzes travel time between consecutive stations and compares a
trained Random Forest model with simple baselines.

> **Scope:** The target is the observed mean travel time for a
> station-to-station segment. This project does not predict a full
> end-to-end journey time or directly model subway delays.

## Project Highlights

-   Cleaned and analyzed **296,114 segment-level observations** from
    2025.
-   Explored travel-time distributions, weekday and monthly patterns,
    slow segments, and segment-level variability.
-   Used **DuckDB SQL** for analytical queries and exported reusable CSV
    reports.
-   Evaluated a chronological train/test split to reduce temporal
    leakage.
-   Compared a global-median baseline, a trained Random Forest model,
    and a historical segment-median benchmark.
-   Built an interactive **Streamlit + Plotly** dashboard with filters,
    KPIs, analysis tabs, model evaluation, and CSV downloads.

## Results at a Glance

The model evaluation uses January--September 2025 for training and
October--December 2025 for testing.

  --------------------------------------------------------------------------
  Method                 MAE (seconds)       RMSE (seconds) Interpretation
  --------------- -------------------- -------------------- ----------------
  Global median                  47.76                94.16 Predicts one
  baseline                                                  overall median
                                                            travel time

  Random Forest                  39.53                58.86 Trained ML model
                                                            using segment
                                                            and calendar
                                                            features

  Historical                      6.74                18.44 Benchmark using
  segment median                                            each segment's
                                                            median from the
                                                            training period
  --------------------------------------------------------------------------

**Important interpretation:** The historical segment-median benchmark is
not a trained machine-learning model. It uses historical values for
known station-to-station segments and falls back to the global median
for unseen segments. About **99.14%** of test observations had a
matching segment in the training period. On known segments, its MAE was
**6.13 seconds**; on unseen segments, its MAE was **77.27 seconds**.
These results show how strongly prior segment-specific history helps on
this dataset, and why unseen-segment performance must be reported
separately.

Metrics describe this dataset and split; they are not a guarantee of
performance on future service conditions.

## Dataset

The cleaned dataset used by the notebooks and dashboard is:

`data/subway_ml_dataset.parquet`

It contains 296,114 rows and these columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `service_date`                      Date of the observation

  `a_station`                         Starting station of the segment

  `b_station`                         Ending station of the segment

  `route_ids`                         Route identifier(s) associated with
                                      the segment

  `day_of_week`                       Day of the week

  `month`                             Month number

  `day_of_month`                      Day of the month

  `mean_seconds`                      Mean observed travel time for the
                                      segment, in seconds
  -----------------------------------------------------------------------

The original source data included additional fields such as trip counts
and median travel time; the cleaned ML dataset above contains the eight
columns listed here.

Please consult the original dataset provider's terms and attribution
requirements before redistributing the data.

## Repository Structure

``` text
NYC-Subway-Analytics/
├── data/
│   └── subway_ml_dataset.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_machine_learning.ipynb
│   └── 03_sql_analytics.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── monthly_analysis.csv
│   ├── slowest_segments.csv
│   ├── variable_segments.csv
│   ├── weekday_analysis.csv
│   ├── segment_coverage.csv
│   └── subway_analytics.duckdb
├── sql/
├── src/
├── dashboard/
│   └── app.py
├── visuals/
│   └── model_comparison.png
└── README.md
```

Some generated report files may only appear after running the relevant
notebook cells. Keep large datasets and generated artifacts out of Git
if repository size or dataset licensing requires it.

## Tech Stack

-   **Python** --- data processing and analysis
-   **Pandas / NumPy** --- data preparation and numerical operations
-   **Matplotlib / Plotly** --- visualizations
-   **Scikit-learn** --- Random Forest and evaluation metrics
-   **DuckDB / SQL** --- analytical queries
-   **Streamlit** --- interactive dashboard
-   **Jupyter Notebook** --- reproducible analysis workflow

## Run Locally

### 1. Clone or open the project

If the repository is already on your computer, open PowerShell in the
project root:

``` powershell
cd "C:\Users\Pradeep\OneDrive\Desktop\projects\NYC-Subway-Analytics"
```

### 2. Create and activate a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can either adjust the execution
policy for your user according to your organization's rules, or run the
environment's Python directly:

``` powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy pyarrow scikit-learn matplotlib plotly streamlit duckdb jupyter
```

### 4. Launch the dashboard

From the project root:

``` powershell
python -m py_compile dashboard/app.py
python -m streamlit run dashboard/app.py
```

Streamlit will print a local URL in the terminal; open that URL in your
browser.

The dashboard expects the cleaned Parquet file at
`data/subway_ml_dataset.parquet`. It may also use generated files under
`reports/` and `visuals/` for model-evaluation sections. Run the
notebooks first if those artifacts are missing.

## Analysis Workflow

1.  **`01_eda.ipynb` --- Exploratory Data Analysis**
    -   Inspect schema, missing values, duplicates, distributions, and
        date coverage.
    -   Explore travel times by route, station segment, weekday, and
        month.
    -   Identify slow and highly variable segments.
2.  **`02_machine_learning.ipynb` --- Machine Learning**
    -   Use a chronological split: train on January--September 2025 and
        test on October--December 2025.
    -   Establish a global-median baseline.
    -   Train and evaluate a Random Forest model.
    -   Compare against a historical segment-median benchmark.
    -   Save model metrics and test predictions under `reports/`.
3.  **`03_sql_analytics.ipynb` --- SQL Analytics**
    -   Use DuckDB to query the dataset.
    -   Analyze monthly and weekday trends, slow segments, variability,
        and training/test segment coverage.
    -   Export query results as CSV reports.
4.  **`dashboard/app.py` --- Interactive Dashboard**
    -   Filter the dataset by date and route.
    -   Review headline KPIs, trends, segment and route analyses, and ML
        evaluation.
    -   Explore records and download available CSV outputs.

## Key Analytical Considerations

-   **Chronological evaluation:** Future dates are held out from model
    training to better reflect a time-based evaluation.
-   **Historical benchmark:** Segment medians must be calculated from
    the training period only; using test-period observations would leak
    information.
-   **Unseen segments:** A segment absent from training cannot use its
    own historical median and needs a fallback strategy.
-   **Metric choice:** MAE gives the average absolute error in seconds;
    RMSE penalizes larger errors more strongly.
-   **Interpretation:** Mean segment travel time is not the same as
    total journey duration, platform waiting time, or a direct measure
    of delay.
-   **Reproducibility:** Results may vary if the source data, cleaning
    rules, feature engineering, library versions, or split dates change.

## Future Improvements

-   Add automated data validation and a documented data-cleaning
    pipeline.
-   Add tests for feature generation and train/test leakage.
-   Track metrics by route, segment, month, and seen-versus-unseen
    segment.
-   Add model persistence and a repeatable training script.
-   Improve dashboard screenshots and document deployment steps.
-   Add a clear data-source citation and license details once confirmed
    from the dataset provider.

## Author

Built as a portfolio project demonstrating practical data analytics,
SQL, machine learning evaluation, and dashboard development.
