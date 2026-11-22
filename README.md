# 🚗 US Accidents: Analysis, Severity Prediction & Interactive Dashboard

**Live demo: https://accedent.streamlit.app/**

An end-to-end data science project on the US road accidents dataset: data cleaning, exploratory analysis, a machine-learning model that predicts accident **severity (1 to 4)**, and an interactive dashboard to explore millions of accidents and compare cities.

## Features of the dashboard

| Tab | What you can do |
|---|---|
| **Overview** | Key numbers, severity distribution, top states, US map of accidents by state |
| **City comparison** | Pick 2 to 6 cities and compare accidents, severity mix, peak hour, night share, and road infrastructure. Profiles by hour, weekday, month or year are shown as percentages so a small city and a large city can be compared fairly |
| **Time** | Accidents by hour, weekday and month, plus a weekday × hour heatmap |
| **Weather** | Top weather conditions, severity mix per condition, visibility vs severity |
| **Infrastructure** | Accidents and average severity near traffic signals, junctions, crossings, etc. |
| **Map** | Random sample of accident locations coloured by severity |
| **Model** | Results of the severity prediction models |

All tabs react to the sidebar filters: **years**, **states** and **severity levels**.

## Dataset

- **US Accidents (March 2023 version)**, a countrywide traffic accident dataset covering 2016 to 2023 (originally published by Moosavi et al.).
- After cleaning, the dashboard works on **7,564,472 accident records**, with no sampling.
- The modelling notebook uses a further-prepared version with 4,430,395 rows and 117 features after one-hot encoding.

## Project pipeline

1. **Cleaning** (`cleaning` notebook): remove unusable columns and missing values, build the cleaned dataset.
2. **Exploratory analysis** (`EDA` notebook): severity, hour, day vs night, state, city, weekday, month, weather and road infrastructure.
3. **Modelling** (`Final_Model` notebook):
   - Split: 75% train, 15% validation, 10% test.
   - The target is heavily imbalanced (severity 2 dominates), so the training set is rebalanced with **SMOTE + random undersampling** to 1.6M rows (600k / 600k / 200k / 200k per class).
   - **XGBoost, LightGBM and CatBoost** tuned with `GridSearchCV`, then combined in a **soft-voting ensemble**.

### Results

| Model | Accuracy |
|---|---|
| XGBoost | 75.84% (validation) |
| LightGBM | 75.95% (validation) |
| CatBoost | 72.18% (validation) |
| **Voting ensemble** | **75.10% (test)** |

Ensemble classification report on the test set (443,040 rows):

| Severity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 0.23 | 0.57 | 0.33 | 4,814 |
| 2 | 0.96 | 0.73 | 0.83 | 350,090 |
| 3 | 0.47 | 0.88 | 0.62 | 81,203 |
| 4 | 0.22 | 0.37 | 0.27 | 6,933 |

Macro F1 is 0.51 and weighted F1 is 0.78. Classes 1 and 4 are rare, so their precision stays low even after resampling.

## Main findings from the exploratory analysis

- Severity 2 is by far the most common level; severities 1 and 4 are rare.
- Two daily peaks appear at **7 to 8 AM** and **4 to 5 PM** (rush hours); the quietest period is 2 to 4 AM.
- Friday has the most accidents and Sunday the fewest.
- December, November and January are the worst months; July is the lowest.
- California has the most accidents, followed by Florida and Texas. Miami, Houston and Los Angeles lead among cities.
- Most accidents happen in fair weather, which reflects how often that weather occurs rather than higher danger.
- Traffic signals, crossings and junctions are the most frequent road features near accidents.

## How the dashboard works

- **Streamlit** for the interface and **Plotly** for the charts.
- **DuckDB** queries a **Parquet** file directly instead of loading millions of rows into memory, so the full dataset runs on a small free server.
- The Parquet file is hosted on Hugging Face Datasets and downloaded automatically on the first start, so the repository only needs `app.py` and `requirements.txt`.
- Query results are cached, so repeated filters load instantly.

## Run it locally

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>
pip install -r requirements.txt
streamlit run app.py
```

On the first start the app downloads the dataset (about 120 MB). To skip the download, place your own `accidents.parquet` next to `app.py`.

## Repository structure

```
├── app.py              # Streamlit + DuckDB dashboard
├── requirements.txt    # Python dependencies
└── README.md
```

## Tech stack

Python · pandas · scikit-learn · imbalanced-learn · XGBoost · LightGBM · CatBoost · DuckDB · Streamlit · Plotly

## Author

**Yassine Gouch**: replace this line with your GitHub, LinkedIn or email.
