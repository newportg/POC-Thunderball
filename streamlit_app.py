from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from thunderball_predictor.algorithms import (
    TICKET_COST_GBP,
    available_algorithms,
    evaluate_rolling_timeline,
    optimize_ticket_portfolio,
)
from thunderball_predictor.loader import DataValidationError, load_draw_history

DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")
DEFAULT_PORTFOLIO_TICKET_COUNT = 9
DEFAULT_TARGET_PAYOUT = 10
DEFAULT_TIMELINE_LOOKBACK_DRAWS = 9

st.set_page_config(page_title="Thunderball Predictor", page_icon="🎱", layout="wide")

st.title("Thunderball Results Viewer + Predictor")
st.caption(
    "For exploration and entertainment only. Lottery draws are random and predictions are not guaranteed."
)

PRIZE_MATRIX_DF = pd.DataFrame(
    [
        {"Win": "GBP500,000", "1": "X", "2": "X", "3": "X", "4": "X", "5": "X", "B": "X"},
        {"Win": "GBP5,000", "1": "X", "2": "X", "3": "X", "4": "X", "5": "X", "B": ""},
        {"Win": "GBP250", "1": "X", "2": "X", "3": "X", "4": "X", "5": "", "B": ""},
        {"Win": "GBP100", "1": "X", "2": "X", "3": "X", "4": "X", "5": "", "B": "X"},
        {"Win": "GBP20", "1": "X", "2": "X", "3": "X", "4": "", "5": "", "B": "X"},
        {"Win": "GBP10", "1": "X", "2": "X", "3": "X", "4": "", "5": "", "B": ""},
        {"Win": "GBP10", "1": "X", "2": "X", "3": "", "4": "", "5": "", "B": "X"},
        {"Win": "GBP5", "1": "X", "2": "", "3": "", "4": "", "5": "", "B": "X"},
        {"Win": "GBP3", "1": "", "2": "", "3": "", "4": "", "5": "", "B": "X"},
    ]
)


@st.cache_data
def _load_from_path(path: str) -> pd.DataFrame:
    return load_draw_history(path)


@st.cache_data
def _load_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    temp_path = Path("/tmp/thunderball_uploaded.csv")
    temp_path.write_bytes(uploaded_bytes)
    return load_draw_history(temp_path)


@st.cache_data
def _build_rolling_timeline_frames(
    df: pd.DataFrame,
    objective_mode: str,
    no_bet_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = evaluate_rolling_timeline(
        df,
        min_training_draws=DEFAULT_TIMELINE_LOOKBACK_DRAWS,
        ticket_count=DEFAULT_PORTFOLIO_TICKET_COUNT,
        target_payout=DEFAULT_TARGET_PAYOUT,
        seed=42,
        objective_mode=objective_mode,
        no_bet_threshold=no_bet_threshold,
    )

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for outcome in result.outcomes:
        summary_rows.append(
            {
                "Draw Date": outcome.draw_date,
                "Training Window": f"{outcome.training_start_date} to {outcome.training_end_date}",
                "Actual Main Numbers": "-".join(str(number) for number in outcome.actual_draw.main_numbers),
                "Actual Thunderball": outcome.actual_draw.thunderball,
                "Best Main Matches": outcome.best_main_match_count,
                "Thunderball Hits": outcome.thunderball_hit_count,
                "Winning Tickets": outcome.winning_ticket_count,
                "Played": "Yes" if outcome.played else "No",
                "Edge Score": round(outcome.edge_score, 3),
                "Total Payout": outcome.total_payout,
                "Total Cost": outcome.total_cost,
                "Net Result": outcome.net_result,
                "Payout If Played": outcome.payout_if_played,
                "Net If Played": outcome.net_if_played,
                "No-Bet Saved Loss": max(0, -outcome.net_if_played) if not outcome.played else 0,
                "Profitable Draw": "Yes" if outcome.net_result > 0 else "No",
            }
        )

        for ticket_index, ticket_outcome in enumerate(outcome.ticket_outcomes, start=1):
            detail_rows.append(
                {
                    "Draw Date": outcome.draw_date,
                    "Ticket": ticket_index,
                    "Training Start": outcome.training_start_date,
                    "Training End": outcome.training_end_date,
                    "Played": "Yes" if outcome.played else "No",
                    "Predicted Main Numbers": "-".join(
                        str(number) for number in ticket_outcome.ticket.main_numbers
                    ),
                    "Predicted Thunderball": ticket_outcome.ticket.thunderball,
                    "Actual Main Numbers": "-".join(
                        str(number) for number in outcome.actual_draw.main_numbers
                    ),
                    "Actual Thunderball": outcome.actual_draw.thunderball,
                    "Main Matches": ticket_outcome.main_match_count,
                    "Thunderball Match": "Yes" if ticket_outcome.thunderball_match else "No",
                    "Payout": ticket_outcome.payout,
                    "Ticket Net Result": ticket_outcome.payout - TICKET_COST_GBP,
                    "Profitable Ticket": "Yes" if ticket_outcome.payout > TICKET_COST_GBP else "No",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


@st.cache_data
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def _tune_no_bet_threshold(
    df: pd.DataFrame,
    objective_mode: str,
    min_threshold: float = 0.0,
    max_threshold: float = 0.6,
    step: float = 0.02,
) -> tuple[pd.DataFrame, float]:
    if step <= 0:
        raise ValueError("step must be greater than 0.")

    test_points = int(round((max_threshold - min_threshold) / step)) + 1
    thresholds = [round(min_threshold + idx * step, 3) for idx in range(test_points)]

    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        result = evaluate_rolling_timeline(
            df,
            min_training_draws=DEFAULT_TIMELINE_LOOKBACK_DRAWS,
            ticket_count=DEFAULT_PORTFOLIO_TICKET_COUNT,
            target_payout=DEFAULT_TARGET_PAYOUT,
            seed=42,
            objective_mode=objective_mode,
            no_bet_threshold=threshold,
        )

        outcomes = result.outcomes
        played_draws = sum(1 for outcome in outcomes if outcome.played)
        skipped_draws = len(outcomes) - played_draws
        strategy_net = int(sum(outcome.net_result for outcome in outcomes))
        always_play_net = int(sum(outcome.net_if_played for outcome in outcomes))

        rows.append(
            {
                "Threshold": threshold,
                "Played": played_draws,
                "Skipped": skipped_draws,
                "Strategy Net": strategy_net,
                "Always-Play Net": always_play_net,
                "Delta vs Always-Play": strategy_net - always_play_net,
                "Avg Net/Draw": strategy_net / max(len(outcomes), 1),
            }
        )

    tuner_df = pd.DataFrame(rows).sort_values(
        by=["Strategy Net", "Delta vs Always-Play", "Threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_threshold = float(tuner_df.iloc[0]["Threshold"])
    return tuner_df, best_threshold


def _highlight_prediction_cells(value: object, column_name: str) -> str:
    if column_name == "Main Matches" and isinstance(value, (int, float)) and value > 0:
        return "background-color: #d4f5dd"
    if column_name == "Thunderball Match" and value == "Yes":
        return "background-color: #d4f5dd"
    if column_name == "Payout" and isinstance(value, (int, float)) and value > 0:
        return "background-color: #d4f5dd"
    if column_name == "Ticket Net Result" and isinstance(value, (int, float)):
        if value > 0:
            return "background-color: #d4f5dd"
        if value == 0:
            return "background-color: #fff3cd"
    if column_name == "Profitable Ticket" and value == "Yes":
        return "background-color: #d4f5dd"
    return ""


with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Thunderball CSV", type=["csv"])

    st.markdown("CSV columns required:")
    st.code("draw_date,n1,n2,n3,n4,n5,thunderball")

    st.header("Predictions")
    selected_algorithms = st.multiselect(
        "Algorithms",
        list(available_algorithms().keys()),
        default=list(available_algorithms().keys()),
    )
    prediction_count = st.slider("Predictions per algorithm", min_value=1, max_value=10, value=3)

try:
    if uploaded_file is not None:
        df = _load_from_upload(uploaded_file.getvalue())
        source_label = "Uploaded CSV"
    else:
        df = _load_from_path(str(DEFAULT_DATA_PATH))
        source_label = f"Sample dataset ({DEFAULT_DATA_PATH})"
except (DataValidationError, FileNotFoundError) as exc:
    st.error(f"Failed to load draw history: {exc}")
    st.stop()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Draws Loaded", len(df))
col_b.metric("Date Range", f"{df['draw_date'].min().date()} to {df['draw_date'].max().date()}")
col_c.metric("Source", source_label)

st.subheader("Past Results")
st.dataframe(df, use_container_width=True, hide_index=True)

main_number_history = pd.concat([df["n1"], df["n2"], df["n3"], df["n4"], df["n5"]], ignore_index=True)
main_freq = (
    main_number_history.value_counts().sort_index().rename_axis("number").reset_index(name="count")
)
tb_freq = (
    df["thunderball"].value_counts().sort_index().rename_axis("number").reset_index(name="count")
)

left, right = st.columns(2)
with left:
    st.subheader("Main Number Frequency")
    st.bar_chart(main_freq.set_index("number"))

with right:
    st.subheader("Thunderball Frequency")
    st.bar_chart(tb_freq.set_index("number"))

st.subheader("Generate Predictions")
if not selected_algorithms:
    st.warning("Select at least one algorithm in the sidebar.")
else:
    rows: list[dict[str, str | int | float]] = []
    algo_map = available_algorithms()

    if st.button("Predict Next Draw"):
        for algo_name in selected_algorithms:
            predictor = algo_map[algo_name]
            for idx in range(prediction_count):
                result = predictor(df, seed=idx + 1)
                rows.append(
                    {
                        "Algorithm": result.algorithm,
                        "Main Numbers": "-".join(str(n) for n in result.main_numbers),
                        "Thunderball": result.thunderball,
                        "Confidence": round(result.confidence, 3),
                        "Method Note": result.note,
                    }
                )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info(
            "Confidence is a relative heuristic based on distribution concentration, "
            "not a probability of winning."
        )

st.subheader("Prize Matrix")
st.dataframe(PRIZE_MATRIX_DF, use_container_width=True, hide_index=True)

st.subheader("Optimized 9-Ticket Portfolio")
st.caption(
    "Portfolio objective: 9 tickets at GBP1 each, optimized for the chance that total payout reaches GBP10 "
    "or better under a history-weighted simulation of future draws."
)

if "optimized_portfolio" not in st.session_state:
    st.session_state["optimized_portfolio"] = None

if st.button("Optimize 9 Tickets for GBP10"):
    with st.spinner("Searching for the strongest 9-ticket portfolio..."):
        st.session_state["optimized_portfolio"] = optimize_ticket_portfolio(
            df,
            ticket_count=DEFAULT_PORTFOLIO_TICKET_COUNT,
            target_payout=DEFAULT_TARGET_PAYOUT,
            seed=42,
        )

portfolio = st.session_state.get("optimized_portfolio")
if portfolio is not None:
    total_cost = len(portfolio.tickets) * TICKET_COST_GBP
    metric_left, metric_mid, metric_right = st.columns(3)
    metric_left.metric("Portfolio Cost", f"GBP{total_cost}")
    metric_mid.metric(
        "Est. Payout >= GBP10",
        f"{portfolio.estimated_probability_target * 100:.1f}%",
    )
    metric_right.metric(
        "Est. Expected Payout",
        f"GBP{portfolio.estimated_expected_payout:.2f}",
    )

    st.caption(
        f"Estimated break-even or better probability: {portfolio.estimated_probability_break_even * 100:.1f}% | "
        f"Coverage score: {portfolio.coverage_score:.3f}"
    )

    portfolio_rows = []
    for index, ticket in enumerate(portfolio.tickets, start=1):
        portfolio_rows.append(
            {
                "Ticket": index,
                "Main Numbers": "-".join(str(number) for number in ticket.main_numbers),
                "Thunderball": ticket.thunderball,
            }
        )

    st.dataframe(pd.DataFrame(portfolio_rows), use_container_width=True, hide_index=True)
    st.info(portfolio.note)

st.subheader("Rolling 9-Draw Timeline")
st.caption(
    "For each actual draw from draw 9 onward, the app trains on all prior draws, generates a 9-ticket portfolio, "
    "then shows the predicted tickets and the realized payout against the actual result."
)

strategy_col1, strategy_col2 = st.columns(2)
with strategy_col1:
    rolling_objective_mode = st.selectbox(
        "Portfolio Strategy",
        options=["downside_aware", "balanced"],
        index=0,
        format_func=lambda value: "Downside Aware" if value == "downside_aware" else "Balanced",
    )
with strategy_col2:
    # Check if we need to apply recommended threshold from previous rerun
    if st.session_state.get("apply_tuner_recommended", False):
        best_thresh = st.session_state.get("rolling_tuner_best_threshold", 0.20)
        st.session_state["rolling_no_bet_threshold"] = float(best_thresh)
        st.session_state["apply_tuner_recommended"] = False

    if "rolling_no_bet_threshold" not in st.session_state:
        st.session_state["rolling_no_bet_threshold"] = 0.20

    rolling_no_bet_threshold = st.slider(
        "No-Bet Threshold (edge score)",
        min_value=0.0,
        max_value=0.6,
        step=0.01,
        key="rolling_no_bet_threshold",
        help="Draws with estimated break-even probability below this threshold are skipped.",
    )

tuner_col1, tuner_col2 = st.columns([1, 2])
with tuner_col1:
    run_tuner = st.button("Tune No-Bet Threshold")
with tuner_col2:
    st.caption("Tests thresholds from 0.00 to 0.60 and ranks them by strategy net result.")

if run_tuner:
    with st.spinner("Tuning threshold across rolling timeline..."):
        tuner_df, best_threshold = _tune_no_bet_threshold(df, objective_mode=rolling_objective_mode)
        st.session_state["rolling_tuner_df"] = tuner_df
        st.session_state["rolling_tuner_best_threshold"] = best_threshold

tuner_df = st.session_state.get("rolling_tuner_df")
best_threshold = st.session_state.get("rolling_tuner_best_threshold")
if tuner_df is not None and best_threshold is not None:
    st.markdown("**Threshold Tuner Results**")
    best_row = tuner_df.iloc[0]
    st.caption(
        f"Recommended threshold: {best_threshold:.2f} | "
        f"Strategy net: GBP{int(best_row['Strategy Net'])} | "
        f"Delta vs always-play: GBP{int(best_row['Delta vs Always-Play'])}"
    )
    st.dataframe(tuner_df.head(12), use_container_width=True, hide_index=True)
    if st.button("Apply Recommended Threshold"):
        st.session_state["apply_tuner_recommended"] = True
        st.rerun()

if "rolling_timeline_frames" not in st.session_state:
    st.session_state["rolling_timeline_frames"] = None

if st.button("Run Rolling 9-Draw Timeline"):
    with st.spinner("Evaluating each draw from draw 9 onward using all prior draws..."):
        st.session_state["rolling_timeline_frames"] = _build_rolling_timeline_frames(
            df,
            objective_mode=rolling_objective_mode,
            no_bet_threshold=rolling_no_bet_threshold,
        )

rolling_timeline_frames = st.session_state.get("rolling_timeline_frames")
if rolling_timeline_frames is not None:
    summary_df, detail_df = rolling_timeline_frames

    if not summary_df.empty:
        metric_left, metric_mid, metric_right, metric_four = st.columns(4)
        played_draws = int((summary_df["Played"] == "Yes").sum())
        skipped_draws = int((summary_df["Played"] == "No").sum())
        strategy_net = int(summary_df["Net Result"].sum())
        always_play_net = int(summary_df["Net If Played"].sum())
        metric_left.metric("Draws Evaluated", len(summary_df))
        metric_mid.metric("Played / Skipped", f"{played_draws} / {skipped_draws}")
        metric_right.metric("Strategy Net", f"GBP{strategy_net}")
        metric_four.metric("Always-Play Net", f"GBP{always_play_net}")
        st.caption(
            f"Average strategy net per draw: GBP{summary_df['Net Result'].mean():.2f} | "
            f"Total avoided loss from skipped draws: GBP{int(summary_df['No-Bet Saved Loss'].sum())}"
        )

        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button(
                "Download Summary CSV",
                data=_to_csv_bytes(summary_df),
                file_name="rolling_9_draw_timeline_summary.csv",
                mime="text/csv",
            )
        with download_right:
            st.download_button(
                "Download Predictions CSV",
                data=_to_csv_bytes(detail_df),
                file_name="rolling_9_draw_timeline_predictions.csv",
                mime="text/csv",
            )

    summary_table = summary_df.copy()
    summary_table.insert(0, "Status", summary_table["Net Result"].map(lambda value: "Profit" if value > 0 else ("Break-even" if value == 0 else "Loss")))

    st.markdown("**Per-Draw Summary (click a row to open details)**")
    selection = st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="rolling_timeline_summary_table",
    )

    selected_rows = selection.selection.get("rows", []) if selection is not None else []
    if selected_rows:
        selected_idx = int(selected_rows[0])
    else:
        selected_idx = len(summary_table) - 1

    selected_summary = summary_table.iloc[selected_idx]
    selected_draw = str(selected_summary["Draw Date"])

    with st.expander(f"Draw {selected_draw} drill-down", expanded=True):
        drill_col1, drill_col2, drill_col3 = st.columns(3)
        drill_col1.metric("Actual Draw", selected_summary["Actual Main Numbers"])
        drill_col2.metric("Actual Thunderball", int(selected_summary["Actual Thunderball"]))
        drill_col3.metric("Net Result", f"GBP{int(selected_summary['Net Result'])}")
        st.caption(
            f"Training range: {selected_summary['Training Window']} | "
            f"Total payout: GBP{int(selected_summary['Total Payout'])} | "
            f"Winning tickets: {int(selected_summary['Winning Tickets'])}"
        )

        selected_detail = detail_df.loc[detail_df["Draw Date"] == selected_draw].copy()
        styled_detail = selected_detail.style.apply(
            lambda col: [
                _highlight_prediction_cells(value, col.name) for value in col
            ],
            axis=0,
        )
        st.dataframe(styled_detail, use_container_width=True, hide_index=True)
