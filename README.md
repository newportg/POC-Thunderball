# Thunderball Results Viewer and Predictor

A Streamlit application to:

- View past Thunderball draw history
- Explore frequency distributions
- Run multiple heuristic prediction algorithms

## Features

- Data viewer for historical draw records
- CSV upload support
- Three prediction strategies:
  - Frequency Weighted
  - Recency Weighted
  - Hot/Cold Mix
- Optimized 9-ticket portfolio generation targeting a GBP10 total payout threshold
- Side-by-side algorithm output comparison

## Important Note

Lottery outcomes are random. This app provides exploratory statistical heuristics for
entertainment and analysis only, not guaranteed prediction.

## Data Format

The app expects CSV columns:

```text
draw_date,n1,n2,n3,n4,n5,thunderball
```

Rules enforced:

- Main numbers `n1..n5` must be integers in `1..39`
- Main numbers must be unique within each row
- `thunderball` must be in `1..14`
- `draw_date` must be valid ISO date (`YYYY-MM-DD`)

A sample dataset is included at `data/thunderball_results_sample.csv`.

## Run Locally

1. Install dependencies:

```bash
pip3 install -r requirements.txt
```

2. Run the app:

```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

3. Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Project Layout

```text
.
├── data/
│   └── thunderball_results_sample.csv
├── src/
│   └── thunderball_predictor/
│       ├── __init__.py
│       ├── algorithms.py
│       ├── data_models.py
│       └── loader.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```
