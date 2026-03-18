from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from thunderball_predictor.algorithms import (
    TICKET_COST_GBP,
    evaluate_rolling_timeline,
    optimize_ticket_portfolio,
)
from thunderball_predictor.loader import DataValidationError, load_draw_history

DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")
DEFAULT_PORTFOLIO_TICKET_COUNT = 9
DEFAULT_TARGET_PAYOUT = 10
DEFAULT_TIMELINE_LOOKBACK_DRAWS = 9
DEFAULT_STAKE_GBP = DEFAULT_PORTFOLIO_TICKET_COUNT * TICKET_COST_GBP
DEFAULT_TARGET_ROI = (DEFAULT_TARGET_PAYOUT / DEFAULT_STAKE_GBP) - 1

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


PREDICTION_STATE_FILE = Path("reports/current_prediction.json")
THRESHOLD_STATE_FILE = Path("reports/no_bet_threshold.json")
ROLLING_TIMELINE_SUMMARY_FILE = Path("reports/rolling_9_draw_timeline_summary.csv")
ROLLING_TIMELINE_DETAIL_FILE = Path("reports/rolling_9_draw_timeline_predictions.csv")
ROLLING_TIMELINE_META_FILE = Path("reports/rolling_9_draw_timeline_cache.json")


def _load_saved_threshold() -> float:
    if THRESHOLD_STATE_FILE.exists():
        try:
            return float(json.loads(THRESHOLD_STATE_FILE.read_text(encoding="utf-8")).get("no_bet_threshold", 0.20))
        except Exception:
            return 0.20
    return 0.20


def _save_threshold(value: float) -> None:
    THRESHOLD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_STATE_FILE.write_text(json.dumps({"no_bet_threshold": value}, indent=2) + "\n", encoding="utf-8")


def _on_threshold_change() -> None:
    _save_threshold(st.session_state["rolling_no_bet_threshold"])


def _build_rolling_timeline_cache_metadata(
    df: pd.DataFrame,
    objective_mode: str,
    no_bet_threshold: float,
) -> dict[str, object]:
    return {
        "objective_mode": objective_mode,
        "no_bet_threshold": round(float(no_bet_threshold), 2),
        "row_count": int(len(df)),
        "data_hash": int(pd.util.hash_pandas_object(df, index=True).sum()),
    }


def _load_rolling_timeline_cache_metadata() -> dict[str, object] | None:
    if ROLLING_TIMELINE_META_FILE.exists():
        try:
            return json.loads(ROLLING_TIMELINE_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _load_saved_rolling_timeline_frames(
    df: pd.DataFrame,
    objective_mode: str,
    no_bet_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    expected_metadata = _build_rolling_timeline_cache_metadata(df, objective_mode, no_bet_threshold)
    saved_metadata = _load_rolling_timeline_cache_metadata()
    if saved_metadata != expected_metadata:
        return None
    if not ROLLING_TIMELINE_SUMMARY_FILE.exists() or not ROLLING_TIMELINE_DETAIL_FILE.exists():
        return None
    try:
        return pd.read_csv(ROLLING_TIMELINE_SUMMARY_FILE), pd.read_csv(ROLLING_TIMELINE_DETAIL_FILE)
    except Exception:
        return None


@st.cache_data
def _build_draw_number_lookup() -> dict[str, int]:
    try:
        raw = pd.read_csv("data/thunderball-draw-history.csv")
    except Exception:
        return {}

    required_columns = {"DrawDate", "DrawNumber"}
    if not required_columns.issubset(set(raw.columns)):
        return {}

    raw["DrawDate"] = pd.to_datetime(raw["DrawDate"], errors="coerce", dayfirst=True)
    raw["DrawNumber"] = pd.to_numeric(raw["DrawNumber"], errors="coerce")
    raw = raw.dropna(subset=["DrawDate", "DrawNumber"]).copy()
    if raw.empty:
        return {}

    raw["DrawDateIso"] = raw["DrawDate"].dt.date.astype(str)
    grouped = raw.sort_values("DrawNumber").groupby("DrawDateIso", as_index=False).last()
    return {str(row["DrawDateIso"]): int(row["DrawNumber"]) for _, row in grouped.iterrows()}


def _save_rolling_timeline_frames(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    df: pd.DataFrame,
    objective_mode: str,
    no_bet_threshold: float,
) -> None:
    ROLLING_TIMELINE_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(ROLLING_TIMELINE_SUMMARY_FILE, index=False)
    detail_df.to_csv(ROLLING_TIMELINE_DETAIL_FILE, index=False)
    metadata = _build_rolling_timeline_cache_metadata(df, objective_mode, no_bet_threshold)
    ROLLING_TIMELINE_META_FILE.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_rolling_timeline_frames(
    df: pd.DataFrame,
    objective_mode: str,
    no_bet_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = _build_rolling_timeline_frames(
        df,
        objective_mode=objective_mode,
        no_bet_threshold=no_bet_threshold,
    )
    _save_rolling_timeline_frames(
        frames[0],
        frames[1],
        df,
        objective_mode=objective_mode,
        no_bet_threshold=no_bet_threshold,
    )
    return frames


def _load_prediction_state() -> dict | None:
    if PREDICTION_STATE_FILE.exists():
        try:
            return json.loads(PREDICTION_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_prediction_state(state: dict) -> None:
    PREDICTION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_prediction_state(df: pd.DataFrame) -> dict:
    portfolio = optimize_ticket_portfolio(
        df,
        ticket_count=DEFAULT_PORTFOLIO_TICKET_COUNT,
        target_payout=DEFAULT_TARGET_PAYOUT,
        seed=42,
        simulation_draws=2500,
        objective_mode="downside_aware",
    )
    latest_draw_number = int(df.sort_values("draw_date", ascending=False).iloc[0].name) if "draw_number" not in df.columns else int(df.sort_values("draw_date", ascending=False).iloc[0]["draw_number"])
    # Try to read the latest draw number from the raw CSV so it matches evaluate_and_predict.py
    try:
        raw = pd.read_csv("data/thunderball-draw-history.csv")
        raw["DrawNumber"] = pd.to_numeric(raw["DrawNumber"], errors="coerce")
        raw["DrawDate"] = pd.to_datetime(raw["DrawDate"], errors="coerce", dayfirst=True)
        latest_draw_number = int(raw.dropna(subset=["DrawNumber"]).sort_values("DrawDate", ascending=False).iloc[0]["DrawNumber"])
    except Exception:
        latest_draw_number = 0

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_latest_draw_number": latest_draw_number,
        "target_draw_number": latest_draw_number + 1,
        "objective_mode": "downside_aware",
        "ticket_count": len(portfolio.tickets),
        "target_payout": portfolio.target_payout,
        "estimated_expected_payout": round(portfolio.estimated_expected_payout, 2),
        "estimated_probability_target": round(portfolio.estimated_probability_target, 4),
        "estimated_probability_break_even": round(portfolio.estimated_probability_break_even, 4),
        "coverage_score": round(portfolio.coverage_score, 4),
        "note": portfolio.note,
        "tickets": [
            {"main_numbers": list(t.main_numbers), "thunderball": t.thunderball}
            for t in portfolio.tickets
        ],
    }


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


def _mix_hex_color(start_hex: str, end_hex: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    start_hex = start_hex.lstrip("#")
    end_hex = end_hex.lstrip("#")

    start_rgb = [int(start_hex[idx : idx + 2], 16) for idx in (0, 2, 4)]
    end_rgb = [int(end_hex[idx : idx + 2], 16) for idx in (0, 2, 4)]
    blended = [int(round(s + (e - s) * ratio)) for s, e in zip(start_rgb, end_rgb)]
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _ball_color(value: int, min_count: int, max_count: int) -> str:
    if max_count <= min_count:
        return "#f59f00"

    normalized = (value - min_count) / (max_count - min_count)
    # Cool-to-hot scale where hotter colors indicate higher draw frequency.
    return _mix_hex_color("#6ec5ff", "#ef476f", normalized)


def _render_ball_grid_html(
    frequency_map: dict[int, int],
    ball_range_end: int,
    column_count: int,
    min_count: int,
    max_count: int,
) -> str:
    cells: list[str] = []
    for number in range(1, ball_range_end + 1):
        count = int(frequency_map.get(number, 0))
        color = _ball_color(count, min_count=min_count, max_count=max_count)
        text_color = "#ffffff" if count >= (min_count + max_count) / 2 else "#0b132b"
        cells.append(
            (
                "<div style='display:flex; flex-direction:column; align-items:center; gap:0.15rem;'>"
                f"<div title='Ball {number}: drawn {count} times' "
                "style='width:2.3rem; height:2.3rem; border-radius:999px; display:flex; align-items:center; "
                f"justify-content:center; background:{color}; color:{text_color}; font-weight:700; font-size:0.85rem; "
                "box-shadow: inset 0 -2px 0 rgba(0,0,0,0.12);'>"
                f"{number:02d}"
                "</div>"
                f"<div style='font-size:0.68rem; color:#4a5568;'>{count}</div>"
                "</div>"
            )
        )

    joined_cells = "".join(cells)
    return (
        "<div style='display:grid; gap:0.65rem; justify-content:start; "
        f"grid-template-columns: repeat({column_count}, minmax(2.3rem, 2.3rem));'>"
        f"{joined_cells}"
        "</div>"
    )


@st.cache_data
def _build_cooccurrence_frequency_map(df: pd.DataFrame, selected_ball: int) -> dict[int, int]:
    position_columns = ["n1", "n2", "n3", "n4", "n5"]
    selected_rows = df[df[position_columns].eq(selected_ball).any(axis=1)]

    counts = {number: 0 for number in range(1, 40)}
    for _, row in selected_rows.iterrows():
        row_numbers = {int(row[column]) for column in position_columns}
        for number in row_numbers:
            if number == selected_ball:
                continue
            counts[number] += 1

    return counts


@st.cache_data
def _build_cooccurrence_chart_frame(
    cooccurrence_map: dict[int, int],
    selected_ball: int,
) -> pd.DataFrame:
    max_count = max(cooccurrence_map.values()) if cooccurrence_map else 0
    rows: list[dict[str, object]] = []

    for number in range(1, 40):
        if number == selected_ball:
            fill_color = "#1d3557"
            text_color = "#ffffff"
            stroke_color = "#f4d35e"
            label = "Selected"
            count = None
            stroke_width = 4
            opacity = 1.0
            count_label = "Selected"
        else:
            count = int(cooccurrence_map.get(number, 0))
            if count > 0:
                fill_color = _ball_color(count, min_count=1, max_count=max_count)
                text_color = "#ffffff" if count >= max(2, max_count / 2) else "#0b132b"
                stroke_color = "#ced4da"
                label = f"Co-drawn {count} times"
                stroke_width = 2.5
                opacity = 1.0
                count_label = str(count)
            else:
                fill_color = "#e9ecef"
                text_color = "#6c757d"
                stroke_color = "#d9dee3"
                label = "Never co-drawn"
                stroke_width = 1.5
                opacity = 0.6
                count_label = "-"

        rows.append(
            {
                "number": number,
                "ball_label": f"{number:02d}",
                "grid_x": ((number - 1) % 8) + 1,
                "grid_y": 5 - ((number - 1) // 8),
                "fill_color": fill_color,
                "stroke_color": stroke_color,
                "text_color": text_color,
                "status": label,
                "cooccurrence_count": count,
                "count_label": count_label,
                "stroke_width": stroke_width,
                "opacity": opacity,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def _build_position_frequency_frame(df: pd.DataFrame) -> pd.DataFrame:
    position_columns = ["n1", "n2", "n3", "n4", "n5"]
    position_labels = {
        "n1": "1st",
        "n2": "2nd",
        "n3": "3rd",
        "n4": "4th",
        "n5": "5th",
    }

    long_df = df[position_columns].melt(var_name="position", value_name="number")
    counts = (
        long_df.groupby(["number", "position"]).size().reset_index(name="count")
        .assign(position_label=lambda frame: frame["position"].map(position_labels))
    )

    all_numbers = pd.Index(range(1, 40), name="number")
    all_positions = pd.Index(position_columns, name="position")
    completed = (
        counts.set_index(["number", "position"])
        .reindex(pd.MultiIndex.from_product([all_numbers, all_positions]))
        .fillna(0)
        .reset_index()
    )
    completed["count"] = completed["count"].astype(int)
    completed["position_label"] = completed["position"].map(position_labels)
    return completed


@st.cache_data
def _build_main_cooccurrence_lookup(df: pd.DataFrame) -> dict[tuple[int, int], int]:
    support: dict[tuple[int, int], int] = {}
    for _, row in df.iterrows():
        numbers = sorted({int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]})
        for left_idx in range(len(numbers)):
            for right_idx in range(left_idx + 1, len(numbers)):
                left = numbers[left_idx]
                right = numbers[right_idx]
                support[(left, right)] = support.get((left, right), 0) + 1
    return support


def _build_prediction_chain_debug_frame(
    tickets: list[dict[str, object]],
    support_lookup: dict[tuple[int, int], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticket_index, ticket in enumerate(tickets, start=1):
        numbers = sorted(int(number) for number in list(ticket["main_numbers"]))
        min_edge_support = None

        for edge_index in range(len(numbers) - 1):
            left = numbers[edge_index]
            right = numbers[edge_index + 1]
            edge_support = int(support_lookup.get((min(left, right), max(left, right)), 0))
            min_edge_support = edge_support if min_edge_support is None else min(min_edge_support, edge_support)
            rows.append(
                {
                    "Ticket": ticket_index,
                    "Edge": f"{edge_index + 1} -> {edge_index + 2}",
                    "From Ball": f"{left:02d}",
                    "To Ball": f"{right:02d}",
                    "Historical Co-Occurrence Count": edge_support,
                }
            )

        if numbers:
            rows.append(
                {
                    "Ticket": ticket_index,
                    "Edge": "Ticket Summary",
                    "From Ball": "-",
                    "To Ball": "-",
                    "Historical Co-Occurrence Count": int(min_edge_support or 0),
                }
            )

    return pd.DataFrame(rows)


def _build_skipped_profitable_analysis(
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | None]]:
    working = summary_df.copy()
    working["Edge Score"] = pd.to_numeric(working["Edge Score"], errors="coerce")
    working["Net If Played"] = pd.to_numeric(working["Net If Played"], errors="coerce")
    working["Payout If Played"] = pd.to_numeric(working["Payout If Played"], errors="coerce")

    skipped_mask = working["Played"].astype(str).str.lower() == "no"
    profitable_if_played_mask = working["Net If Played"] > 0
    skipped_profitable_mask = skipped_mask & profitable_if_played_mask

    missed_profitable_df = (
        working.loc[
            skipped_profitable_mask,
            [
                "Draw Date",
                "Draw Number",
                "Edge Score",
                "Payout If Played",
                "Net If Played",
                "Best Main Matches",
                "Thunderball Hits",
                "Winning Tickets",
            ],
        ]
        .sort_values("Draw Date", ascending=False)
        .reset_index(drop=True)
    )

    threshold_rows: list[dict[str, float | int]] = []
    skipped_profitable_total = int(skipped_profitable_mask.sum())

    for threshold_step in range(0, 61):
        threshold = threshold_step / 100
        add_play_mask = skipped_mask & (working["Edge Score"] >= threshold)

        additional_plays = int(add_play_mask.sum())
        additional_net = float(working.loc[add_play_mask, "Net If Played"].sum())
        additional_profitable = int((add_play_mask & profitable_if_played_mask).sum())
        additional_unprofitable = int(additional_plays - additional_profitable)
        missed_captured = int((add_play_mask & skipped_profitable_mask).sum())
        capture_rate = (
            float(missed_captured / skipped_profitable_total) if skipped_profitable_total > 0 else 0.0
        )
        precision = float(additional_profitable / additional_plays) if additional_plays > 0 else 0.0

        threshold_rows.append(
            {
                "Threshold": round(threshold, 2),
                "Additional Plays": additional_plays,
                "Additional Profitable": additional_profitable,
                "Additional Unprofitable": additional_unprofitable,
                "Missed Profits Captured": missed_captured,
                "Missed Capture Rate": capture_rate,
                "Precision": precision,
                "Additional Net If Played": additional_net,
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    candidate_df = threshold_df.loc[threshold_df["Additional Plays"] > 0].copy()

    recommendation: dict[str, float | int | None] = {
        "recommended_threshold": None,
        "additional_net": None,
        "missed_capture_rate": None,
        "additional_plays": None,
    }
    if not candidate_df.empty:
        positive_net = candidate_df.loc[candidate_df["Additional Net If Played"] > 0]
        selected_pool = positive_net if not positive_net.empty else candidate_df
        best_row = selected_pool.sort_values(
            ["Additional Net If Played", "Missed Capture Rate", "Precision", "Threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        recommendation = {
            "recommended_threshold": float(best_row["Threshold"]),
            "additional_net": float(best_row["Additional Net If Played"]),
            "missed_capture_rate": float(best_row["Missed Capture Rate"]),
            "additional_plays": int(best_row["Additional Plays"]),
        }

    return missed_profitable_df, threshold_df, recommendation


@st.fragment
def _render_cooccurrence_explorer(df: pd.DataFrame) -> None:
    if "selected_cooccurrence_ball" not in st.session_state:
        st.session_state["selected_cooccurrence_ball"] = 1

    selected_cooccurrence_ball = int(st.session_state["selected_cooccurrence_ball"])
    cooccurrence_map = _build_cooccurrence_frequency_map(df, selected_cooccurrence_ball)
    cooccurrence_hits = sum(1 for count in cooccurrence_map.values() if count > 0)
    cooccurrence_peak = max(cooccurrence_map.values()) if cooccurrence_map else 0
    chart_df = _build_cooccurrence_chart_frame(cooccurrence_map, selected_cooccurrence_ball)

    st.subheader("Ball Co-Occurrence Explorer")
    st.caption(
        "Click any ball in the grid to highlight every other main ball that has ever been drawn with it. Only this explorer reruns when you change the selection."
    )

    selector = alt.selection_point(fields=["number"], name="ball_pick", on="click", clear=False)
    circle_layer = (
        alt.Chart(chart_df)
        .mark_circle(size=1300)
        .encode(
            x=alt.X("grid_x:O", axis=None, sort=list(range(1, 9))),
            y=alt.Y("grid_y:O", axis=None, sort=list(range(5, 0, -1))),
            color=alt.Color("fill_color:N", scale=None, legend=None),
            stroke=alt.Color("stroke_color:N", scale=None, legend=None),
            strokeWidth=alt.StrokeWidth("stroke_width:Q", legend=None),
            opacity=alt.Opacity("opacity:Q", legend=None),
            tooltip=[
                alt.Tooltip("ball_label:N", title="Ball"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("cooccurrence_count:Q", title="Co-occurrence count"),
            ],
        )
        .add_params(selector)
    )
    text_layer = alt.Chart(chart_df).mark_text(fontSize=11, fontWeight="bold").encode(
        x=alt.X("grid_x:O", axis=None, sort=list(range(1, 9))),
        y=alt.Y("grid_y:O", axis=None, sort=list(range(5, 0, -1))),
        text="ball_label:N",
        color=alt.Color("text_color:N", scale=None, legend=None),
    )

    count_layer = alt.Chart(chart_df).mark_text(fontSize=9, dy=22, fontWeight="bold").encode(
        x=alt.X("grid_x:O", axis=None, sort=list(range(1, 9))),
        y=alt.Y("grid_y:O", axis=None, sort=list(range(5, 0, -1))),
        text="count_label:N",
        color=alt.value("#52606d"),
        opacity=alt.Opacity("opacity:Q", legend=None),
    )

    chart = (
        alt.layer(circle_layer, text_layer, count_layer)
        .properties(width=320, height=250)
        .configure_view(stroke=None)
        .configure_scale(bandPaddingInner=0.12, bandPaddingOuter=0.08)
    )
    event = st.altair_chart(
        chart,
        width="content",
        key="cooccurrence_ball_chart",
        on_select="rerun",
        selection_mode="ball_pick",
    )

    selected_points = []
    if event is not None:
        if hasattr(event.selection, "ball_pick"):
            selected_points = event.selection.ball_pick
        elif isinstance(event.selection, dict):
            selected_points = event.selection.get("ball_pick", [])

    if selected_points:
        candidate = selected_points[0]
        if isinstance(candidate, dict) and "number" in candidate:
            clicked_ball = int(candidate["number"])
            if clicked_ball != st.session_state["selected_cooccurrence_ball"]:
                st.session_state["selected_cooccurrence_ball"] = clicked_ball
                st.rerun(scope="fragment")

    st.caption(
        f"Ball {selected_cooccurrence_ball:02d} has been co-drawn with {cooccurrence_hits} distinct main balls. Peak co-occurrence frequency: {cooccurrence_peak}."
    )


with st.expander("Data Source", expanded=False):
    uploaded_file = st.file_uploader("Upload Thunderball CSV", type=["csv"])
    st.markdown("CSV columns required:")
    st.code("draw_date,n1,n2,n3,n4,n5,thunderball")

try:
    if uploaded_file is not None:
        df = _load_from_upload(uploaded_file.getvalue())
    else:
        df = _load_from_path(str(DEFAULT_DATA_PATH))
except (DataValidationError, FileNotFoundError) as exc:
    st.error(f"Failed to load draw history: {exc}")
    st.stop()

col_a, col_b = st.columns(2)
col_a.metric("Draws Loaded", len(df))
col_b.markdown("**Date Range**")
col_b.markdown(
    (
        "<div style='font-size:0.9rem; line-height:1.35; white-space:normal;'>"
        f"{df['draw_date'].min().date()} to {df['draw_date'].max().date()}"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.subheader("Past Results")
st.dataframe(df, use_container_width=True, hide_index=True)

main_number_history = pd.concat([df["n1"], df["n2"], df["n3"], df["n4"], df["n5"]], ignore_index=True)
main_freq = (
    main_number_history.value_counts().sort_index().rename_axis("number").reset_index(name="count")
)
tb_freq = (
    df["thunderball"].value_counts().sort_index().rename_axis("number").reset_index(name="count")
)
position_freq_df = _build_position_frequency_frame(df)

main_freq_map = {int(row["number"]): int(row["count"]) for _, row in main_freq.iterrows()}
tb_freq_map = {int(row["number"]): int(row["count"]) for _, row in tb_freq.iterrows()}
main_min_count = min(main_freq_map.values()) if main_freq_map else 0
main_max_count = max(main_freq_map.values()) if main_freq_map else 0
tb_min_count = min(tb_freq_map.values()) if tb_freq_map else 0
tb_max_count = max(tb_freq_map.values()) if tb_freq_map else 0

left, right = st.columns(2)
with left:
    st.subheader("Main Number Frequency")
    st.bar_chart(main_freq.set_index("number"))

with right:
    st.subheader("Thunderball Frequency")
    st.bar_chart(tb_freq.set_index("number"))

st.subheader("Main Ball Frequency by Draw Position")
st.caption(
    "Stacked bars show how often each main ball number was drawn in 1st, 2nd, 3rd, 4th, and 5th position."
)

position_chart = (
    alt.Chart(position_freq_df)
    .mark_bar()
    .encode(
        x=alt.X("number:O", title="Ball Number", sort=list(range(1, 40))),
        y=alt.Y("count:Q", title="Frequency"),
        color=alt.Color(
            "position_label:N",
            title="Draw Position",
            sort=["1st", "2nd", "3rd", "4th", "5th"],
            scale=alt.Scale(
                domain=["1st", "2nd", "3rd", "4th", "5th"],
                range=["#0f4c5c", "#2c7a7b", "#84a98c", "#f6bd60", "#e76f51"],
            ),
        ),
        order=alt.Order(
            "position_label:N",
            sort="ascending",
        ),
        tooltip=[
            alt.Tooltip("number:O", title="Ball"),
            alt.Tooltip("position_label:N", title="Position"),
            alt.Tooltip("count:Q", title="Frequency"),
        ],
    )
    .properties(height=360)
)
st.altair_chart(position_chart, use_container_width=True)

_render_cooccurrence_explorer(df)

st.subheader("Machine Start Position Ball Grid")
st.caption(
    "Balls are laid out by numeric starting position in the machine rack. Color intensity represents historical draw frequency."
)

grid_left, grid_right = st.columns(2)
with grid_left:
    st.markdown("**Main Balls (1-39)**")
    st.markdown(
        _render_ball_grid_html(
            frequency_map=main_freq_map,
            ball_range_end=39,
            column_count=8,
            min_count=main_min_count,
            max_count=main_max_count,
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Frequency range: {main_min_count} to {main_max_count}")

with grid_right:
    st.markdown("**Thunderballs (1-14)**")
    st.markdown(
        _render_ball_grid_html(
            frequency_map=tb_freq_map,
            ball_range_end=14,
            column_count=7,
            min_count=tb_min_count,
            max_count=tb_max_count,
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Frequency range: {tb_min_count} to {tb_max_count}")

st.subheader("Prize Matrix")
st.dataframe(PRIZE_MATRIX_DF, use_container_width=True, hide_index=True)

st.subheader("Next Draw Prediction")
st.caption(
    "Prediction generated from all available historical draws using the downside-aware portfolio optimizer. "
    f"Target is >= GBP{DEFAULT_TARGET_PAYOUT} return from a GBP{DEFAULT_STAKE_GBP} stake "
    f"(about {DEFAULT_TARGET_ROI:.1%} ROI), aligned with prize tiers 3 main balls or 2 main balls + thunderball."
)

if "next_draw_prediction" not in st.session_state:
    st.session_state["next_draw_prediction"] = _load_prediction_state()

pred = st.session_state.get("next_draw_prediction")

regen_col, _ = st.columns([1, 3])
with regen_col:
    if st.button("🔄 Regenerate Prediction", key="regen_next_draw"):
        with st.spinner("Generating new prediction from all available draws..."):
            new_pred = _generate_prediction_state(df)
            _save_prediction_state(new_pred)
            st.session_state["next_draw_prediction"] = new_pred
            pred = new_pred
        st.success("Prediction regenerated.")

if pred:
    nd_col1, nd_col2, nd_col3, nd_col4, nd_col5 = st.columns(5)
    nd_col1.metric("Target Draw", f"#{pred['target_draw_number']}")
    nd_col2.metric("Est. Expected Payout", f"£{pred['estimated_expected_payout']}")
    nd_col3.metric("ROI Target", f"GBP{pred['target_payout']} on GBP{DEFAULT_STAKE_GBP}")
    nd_col4.metric("ROI Target Hit Prob", f"{pred['estimated_probability_target']:.1%}")
    nd_col5.metric("Break-even Probability", f"{pred['estimated_probability_break_even']:.1%}")
    st.caption(
        f"Generated at: {pred['generated_at']} | "
        f"Based on draw #{pred['source_latest_draw_number']} | "
        f"Mode: {pred['objective_mode']} | "
        f"Coverage score: {pred['coverage_score']:.3f}"
    )

    ticket_rows = [
        {
            "Ticket": i + 1,
            "Main Numbers": "  -  ".join(f"{n:02d}" for n in t["main_numbers"]),
            "Thunderball": f"{t['thunderball']:02d}",
        }
        for i, t in enumerate(pred["tickets"])
    ]
    st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True, hide_index=True)

    support_lookup = _build_main_cooccurrence_lookup(df)
    chain_debug_df = _build_prediction_chain_debug_frame(pred["tickets"], support_lookup)
    with st.expander("Show Prediction Chain Debug (Co-Occurrence Edges)", expanded=False):
        st.caption(
            "Shows each ticket's sequential main-ball edges and how many times each edge has appeared together in historical draws."
        )
        st.dataframe(chain_debug_df, use_container_width=True, hide_index=True)

    if pred.get("note"):
        st.info(pred["note"])
else:
    st.info("No prediction found. Click 'Regenerate Prediction' to generate one.")

st.subheader("Rolling 9-Draw Timeline")
st.caption(
    "For each actual draw from draw 9 onward, the app trains on all prior draws, generates a 9-ticket portfolio, "
    "then shows the predicted tickets and the realized payout against the actual result. "
    f"Target performance is >= GBP{DEFAULT_TARGET_PAYOUT} from GBP{DEFAULT_STAKE_GBP} stake "
    f"(about {DEFAULT_TARGET_ROI:.1%} ROI), corresponding to 3 main balls or 2 main balls + thunderball."
)
st.markdown(
    "**Workflow:** 1) Choose strategy and threshold. "
    "2) Timeline recalculates automatically when settings change. "
    "3) Use skipped-profitable analysis to apply the suggested threshold and iterate."
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
    if st.session_state.get("apply_skipped_profit_recommended", False):
        suggested_thresh = st.session_state.get("skipped_profit_suggested_threshold", 0.20)
        st.session_state["rolling_no_bet_threshold"] = float(suggested_thresh)
        st.session_state["apply_skipped_profit_recommended"] = False
        _save_threshold(float(suggested_thresh))

    if "rolling_no_bet_threshold" not in st.session_state:
        st.session_state["rolling_no_bet_threshold"] = _load_saved_threshold()

    rolling_no_bet_threshold = st.slider(
        "No-Bet Threshold (edge score)",
        min_value=0.0,
        max_value=0.6,
        step=0.01,
        key="rolling_no_bet_threshold",
        on_change=_on_threshold_change,
        help="Draws with estimated break-even probability below this threshold are skipped.",
    )

if "rolling_timeline_frames" not in st.session_state:
    st.session_state["rolling_timeline_frames"] = None
if "rolling_timeline_cache_metadata" not in st.session_state:
    st.session_state["rolling_timeline_cache_metadata"] = None

current_rolling_timeline_metadata = _build_rolling_timeline_cache_metadata(
    df,
    objective_mode=rolling_objective_mode,
    no_bet_threshold=rolling_no_bet_threshold,
)

if st.session_state["rolling_timeline_frames"] is None:
    cached_frames = _load_saved_rolling_timeline_frames(
        df,
        objective_mode=rolling_objective_mode,
        no_bet_threshold=rolling_no_bet_threshold,
    )
    if cached_frames is not None:
        st.session_state["rolling_timeline_frames"] = cached_frames
        st.session_state["rolling_timeline_cache_metadata"] = current_rolling_timeline_metadata

refresh_rolling_timeline = (
    st.session_state["rolling_timeline_frames"] is None
    or st.session_state["rolling_timeline_cache_metadata"] != current_rolling_timeline_metadata
)

if refresh_rolling_timeline:
    with st.spinner("Evaluating each draw from draw 9 onward using all prior draws..."):
        st.session_state["rolling_timeline_frames"] = _refresh_rolling_timeline_frames(
            df,
            objective_mode=rolling_objective_mode,
            no_bet_threshold=rolling_no_bet_threshold,
        )
        st.session_state["rolling_timeline_cache_metadata"] = current_rolling_timeline_metadata

rolling_timeline_frames = st.session_state.get("rolling_timeline_frames")
if rolling_timeline_frames is not None:
    summary_df, detail_df = rolling_timeline_frames
    draw_number_lookup = _build_draw_number_lookup()
    if "Draw Number" not in summary_df.columns:
        summary_df = summary_df.copy()
        summary_df.insert(
            1,
            "Draw Number",
            summary_df["Draw Date"].map(lambda value: draw_number_lookup.get(str(value), pd.NA)),
        )

    if not summary_df.empty:
        metric_left, metric_mid, metric_right, metric_four = st.columns(4)
        played_draws = int((summary_df["Played"] == "Yes").sum())
        skipped_draws = int((summary_df["Played"] == "No").sum())
        strategy_net = int(summary_df["Net Result"].sum())
        always_play_net = int(summary_df["Net If Played"].sum())
        target_hits_if_played = int((summary_df["Payout If Played"] >= DEFAULT_TARGET_PAYOUT).sum())
        metric_left.metric("Draws Evaluated", len(summary_df))
        metric_mid.metric("Played / Skipped", f"{played_draws} / {skipped_draws}")
        metric_right.metric("Strategy Net", f"GBP{strategy_net}")
        metric_four.metric("Always-Play Net", f"GBP{always_play_net}")
        st.caption(
            f"Average strategy net per draw: GBP{summary_df['Net Result'].mean():.2f} | "
            f"Total avoided loss from skipped draws: GBP{int(summary_df['No-Bet Saved Loss'].sum())} | "
            f"ROI target hits if always played: {target_hits_if_played}"
        )

        missed_profitable_df, threshold_sweep_df, threshold_reco = _build_skipped_profitable_analysis(summary_df)

        st.markdown("**Skipped Profitable Draw Analysis**")
        if missed_profitable_df.empty:
            st.info("No skipped draws would have produced a positive net result in the current evaluation window.")
        else:
            missed_col1, missed_col2, missed_col3 = st.columns(3)
            missed_col1.metric("Skipped Profitable Draws", len(missed_profitable_df))
            missed_col2.metric("Missed Net Potential", f"GBP{int(missed_profitable_df['Net If Played'].sum())}")
            missed_col3.metric("Avg Missed Edge Score", f"{missed_profitable_df['Edge Score'].mean():.3f}")

            reco_threshold = threshold_reco.get("recommended_threshold")
            if reco_threshold is not None:
                st.caption(
                    "Threshold sweep suggestion: "
                    f"try no-bet threshold {float(reco_threshold):.2f} "
                    f"(additional net GBP{float(threshold_reco['additional_net']):.0f}, "
                    f"captures {float(threshold_reco['missed_capture_rate']):.1%} of skipped profitable draws, "
                    f"adds {int(threshold_reco['additional_plays'])} plays)."
                )
                if st.button("Apply Suggested Threshold", key="apply_skipped_profit_threshold"):
                    suggested = float(reco_threshold)
                    st.session_state["skipped_profit_suggested_threshold"] = suggested
                    st.session_state["apply_skipped_profit_recommended"] = True
                    st.rerun()

            with st.expander("Show skipped profitable draws", expanded=False):
                st.dataframe(missed_profitable_df, use_container_width=True, hide_index=True)

            sweep_chart = threshold_sweep_df[["Threshold", "Additional Net If Played", "Missed Capture Rate"]].copy()
            sweep_chart["Missed Capture Rate"] = sweep_chart["Missed Capture Rate"] * 100
            with st.expander("Show threshold sweep diagnostics", expanded=False):
                st.caption(
                    "Additional Net If Played estimates how much net result would change by lowering the no-bet threshold to each value."
                )
                st.line_chart(
                    sweep_chart.set_index("Threshold")[["Additional Net If Played", "Missed Capture Rate"]],
                    use_container_width=True,
                )
                st.dataframe(threshold_sweep_df, use_container_width=True, hide_index=True)

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
    summary_table["__draw_date_sort"] = pd.to_datetime(summary_table["Draw Date"], errors="coerce")
    summary_table = summary_table.sort_values("__draw_date_sort", ascending=False).drop(columns=["__draw_date_sort"])
    def _status_label(row: pd.Series) -> str:
        if str(row.get("Played", "")).strip().lower() != "yes":
            return "Skipped"
        net_result = float(row.get("Net Result", 0))
        if net_result > 0:
            return "Profit"
        if net_result == 0:
            return "Break-even"
        return "Loss"

    summary_table.insert(0, "Status", summary_table.apply(_status_label, axis=1))
    summary_display_table = summary_table.drop(columns=["Training Window", "No-Bet Saved Loss"], errors="ignore")

    st.markdown("**Per-Draw Summary (click a row to open details)**")
    selection = st.dataframe(
        summary_display_table,
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
