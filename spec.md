# Thunderball Predictor - Current State Specification

Last updated: 2026-03-18

## 1. Purpose

This specification defines the current, implemented behavior of the Thunderball project in this repository. It serves as the baseline contract for:
- data ingestion and validation,
- prediction and portfolio optimization,
- rolling timeline evaluation,
- automated result fetching,
- report generation and email delivery.

Unless explicitly marked as planned, statements in this document describe behavior that already exists in code.

## 2. Scope

In scope:
- Streamlit analytics and prediction UI
- Historical draw loading and validation
- Heuristic prediction algorithms and 9-ticket portfolio optimization
- Rolling timeline backtest with optional no-bet threshold
- Automated XML fetch from National Lottery draw history endpoint
- Local report artifact generation and Mailtrap delivery

Out of scope:
- guaranteed prediction accuracy
- real-money betting advice
- account management, user auth, or cloud-native persistent storage

## 3. System Components

- UI application: streamlit_app.py
- Official results fetcher: fetch_lottery_results.py
- Prediction/report pipeline: evaluate_and_predict.py
- Email sender: send_report_via_mailtrap.py
- Core package: src/thunderball_predictor/
  - loader.py
  - algorithms.py
  - data_models.py

## 4. Data Contracts

### 4.1 Accepted Input CSV Schemas

The loader accepts either:

1) Canonical internal schema
- draw_date, n1, n2, n3, n4, n5, thunderball

2) Official export schema (mapped internally)
- DrawDate, Ball 1, Ball 2, Ball 3, Ball 4, Ball 5, Thunderball

Additional columns are ignored after schema normalization.

### 4.2 Validation Rules

- draw_date must parse as a valid date (day-first parsing is allowed for values like 14-Mar-2026)
- main numbers n1..n5 must be numeric integers in range 1..39
- thunderball must be numeric integer in range 1..14
- each row must contain 5 unique main numbers

On validation failure, DataValidationError is raised.

### 4.3 Sort Order

Validated data is returned sorted by draw_date descending.

## 5. Functional Requirements

### FR-1: Draw History Loading

The system shall load draw history from a local CSV path or uploaded CSV bytes, normalize schema, validate values, and expose a typed dataframe for downstream prediction/evaluation.

### FR-2: Prediction Algorithms

The system shall provide these algorithms:
- Frequency Weighted
- Recency Weighted
- Hot/Cold Mix
- Markov Chain

Markov Chain shall fall back to Frequency Weighted if there are fewer than 2 draws.

All main-number prediction paths shall incorporate historical co-occurrence information, and shall exclude any main ball that has never been co-drawn with another main ball in the available history.

Main numbers shall be generated as a sequential co-occurrence chain: the second main ball must come from the first ball's co-occurrence pool, the third from the second ball's co-occurrence pool, and so on through the fifth ball.

### FR-3: Portfolio Optimization

The system shall optimize a multi-ticket portfolio using weighted simulations and a defined prize matrix.

The optimizer shall blend base main-number weights with co-occurrence strength and reject main balls with zero historical co-occurrence support.

Current constants:
- ticket cost: GBP1
- default ticket count: 9
- default target payout: GBP10
- default simulation draws: 2500
- objective modes: balanced, downside_aware

### FR-4: Rolling Timeline Evaluation

The system shall evaluate each actual draw from the ninth draw onward by:
- training on all prior draws,
- generating a 9-ticket portfolio,
- scoring realized payout versus actual draw,
- reporting played/skipped status based on no-bet threshold.

The no-bet threshold decision uses estimated break-even probability.

### FR-5: Streamlit UI

The UI shall provide:
- dataset loading (default local file or uploaded CSV) via in-page controls, without a persistent left sidebar
- dedicated Delta System page for Thunderball with delta-signature analytics, delta-based ticket generation, side-by-side next-draw comparison against the current optimizer, and rolling backtest comparison (Delta vs current optimizer) including winner badge and target-hit likelihood comparison
- historical results table
- frequency charts for main balls and thunderball
- stacked bar chart for main ball frequency by draw position (1st through 5th), with each bar representing a ball number and stack segments representing positional frequency
- interactive main-ball co-occurrence explorer where clicking a ball highlights all other main balls ever drawn with it, with color intensity representing co-occurrence frequency
- prediction chain debug panel showing sequential main-ball edges for generated tickets with historical co-occurrence counts per edge
- explicit ROI target context in prediction and rolling timeline views (GBP10 return from GBP9 stake, about 11.1% ROI), aligned to prize outcomes 3 main balls or 2 main balls plus thunderball
- target-hit probability and target-hit summary indicators for the ROI target
- rolling pre-draw summary table sorted by newest draw date first and including draw number when available from official history
- rolling pre-draw summary status labels distinguish skipped draws from played outcomes; "Break-even" is only used when a played draw has net result 0
- skipped-profitable-draw diagnostics in rolling timeline, including draw-level missed-profit listing, threshold-sweep analysis to guide no-bet threshold tuning, and one-click application of suggested threshold
- machine start-position ball grids for main balls (1..39) and thunderballs (1..14), rendered as colored balls where color intensity maps to historical draw frequency
- prize matrix display
- next-draw prediction generation/regeneration and persistence
- rolling timeline summary and per-ticket drill-down
- CSV downloads for summary and prediction detail
- simplified rolling timeline workflow with automatic recalculation when strategy/threshold changes and one-click threshold application from skipped-profit diagnostics

### FR-6: Official Results Fetch

The fetcher shall:
- request XML from https://www.national-lottery.co.uk/results/thunderball/draw-history/xml
- parse draw records including draw number/date/balls/thunderball/ball set/machine
- transform draw_date to DD-Mon-YYYY
- deduplicate by DrawNumber against local CSV
- prepend newly discovered draws to data/thunderball-draw-history.csv

### FR-7: Prediction Evaluation and Reporting

The evaluate pipeline shall:
- load latest official draw
- compare latest draw against stored prediction when target draw numbers match
- compute ticket-level matches/payouts and portfolio totals
- generate next prediction when source latest draw number changes
- persist report and state artifacts under reports/

### FR-8: Email Delivery

Mailtrap sender shall:
- read subject/body report artifacts
- use MAILTRAP_API_TOKEN, sender, and recipient env vars
- send to sandbox by default unless explicitly disabled
- fail with non-zero exit on non-2xx send response

## 6. Prize Matrix (Implemented)

Payout mapping by (main matches, thunderball match):
- (5, true) -> 500000
- (5, false) -> 5000
- (4, false) -> 250
- (4, true) -> 100
- (3, true) -> 20
- (3, false) -> 10
- (2, true) -> 10
- (1, true) -> 5
- (0, true) -> 3

All unspecified combinations map to 0 payout.

## 7. Persistent Artifacts

### 7.1 Data

- data/thunderball-draw-history.csv
- data/thunderball_results_sample.csv

### 7.2 Reports/State

- reports/current_prediction.json
- reports/latest_prediction_report.txt
- reports/latest_email_subject.txt
- reports/no_bet_threshold.json
- reports/rolling_9_draw_timeline_cache.json
- reports/rolling_9_draw_timeline_summary.csv
- reports/rolling_9_draw_timeline_predictions.csv

## 8. Operational Behavior

- setup_cron.sh installs a cron job for Tue/Wed/Fri/Sat at 21:00
- run_fetch.sh runs fetch_lottery_results.py via local virtual environment Python path
- Streamlit app can be launched with PYTHONPATH=src streamlit run streamlit_app.py

## 9. Known Constraints

- Lottery outcomes are random; predictions are exploratory heuristics only
- SSL verification is disabled in fetch_lottery_results.py (development-oriented behavior)
- Report and cache artifacts are local-file based (no transactional DB guarantees)

## 10. Change Management Policy

This file is the current-state baseline specification.

For every behavior-changing code change, contributors must do one of:
- update this file to reflect the new behavior, or
- explicitly state in PR that the spec is unaffected.

Repository enforcement:
- pull request template includes an explicit Spec Impact checklist
- CI workflow `.github/workflows/spec-sync-check.yml` fails PRs when behavior-impacting files change without updating spec.md

Minimum update requirements when behavior changes:
- update relevant FR section(s)
- update constants/contracts if changed
- update artifact paths/formats if changed
- update Last updated date
