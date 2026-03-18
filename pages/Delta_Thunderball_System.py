from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from thunderball_predictor.algorithms import (
    PRIZE_MATRIX,
    TICKET_COST_GBP,
    optimize_ticket_portfolio,
)
from thunderball_predictor.loader import DataValidationError, load_draw_history

MAX_MAIN_NUMBER = 39
THUNDERBALL_MAX = 14
DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")
DEFAULT_TICKET_COUNT = 9
DEFAULT_TARGET_PAYOUT = 10
DEFAULT_LOOKBACK_DRAWS = 9


@dataclass(frozen=True)
class DeltaTicket:
    main_numbers: tuple[int, int, int, int, int]
    thunderball: int
    delta_signature: tuple[int, int, int, int, int]


def _compute_delta_signature(main_numbers: list[int] | tuple[int, ...]) -> tuple[int, int, int, int, int]:
    ordered = sorted(int(number) for number in main_numbers)
    deltas = [ordered[idx + 1] - ordered[idx] for idx in range(4)]
    deltas.append(MAX_MAIN_NUMBER - ordered[-1] + ordered[0])
    return tuple(deltas)


def _numbers_from_start_and_signature(start: int, signature: tuple[int, int, int, int, int]) -> tuple[int, ...] | None:
    current = int(start)
    numbers = [current]
    for delta in signature[:-1]:
        current = ((current - 1 + int(delta)) % MAX_MAIN_NUMBER) + 1
        numbers.append(current)

    if len(set(numbers)) != 5:
        return None

    return tuple(sorted(numbers))


@st.cache_data
def _load_from_path(path: str) -> pd.DataFrame:
    return load_draw_history(path)


@st.cache_data
def _load_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    temp_path = Path("/tmp/thunderball_uploaded_delta.csv")
    temp_path.write_bytes(uploaded_bytes)
    return load_draw_history(temp_path)


@st.cache_data
def _build_delta_frame(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in df.sort_values("draw_date", ascending=False).iterrows():
        main_numbers = [int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]]
        signature = _compute_delta_signature(main_numbers)
        records.append(
            {
                "Draw Date": row["draw_date"].date().isoformat(),
                "Main Numbers": "-".join(f"{number:02d}" for number in sorted(main_numbers)),
                "Thunderball": int(row["thunderball"]),
                "Delta 1": signature[0],
                "Delta 2": signature[1],
                "Delta 3": signature[2],
                "Delta 4": signature[3],
                "Delta 5": signature[4],
                "Signature": "-".join(str(value) for value in signature),
            }
        )

    return pd.DataFrame(records)


@st.cache_data
def _build_signature_scores(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    score_by_signature: dict[tuple[int, int, int, int, int], float] = {}
    count_by_signature: dict[tuple[int, int, int, int, int], int] = {}

    for idx, (_, row) in enumerate(ordered.iterrows()):
        signature = _compute_delta_signature([int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]])
        weight = 0.985**idx
        score_by_signature[signature] = score_by_signature.get(signature, 0.0) + weight
        count_by_signature[signature] = count_by_signature.get(signature, 0) + 1

    rows = []
    for signature, score in score_by_signature.items():
        rows.append(
            {
                "Signature": "-".join(str(value) for value in signature),
                "Weighted Score": score,
                "Occurrences": count_by_signature[signature],
                "SignatureTuple": signature,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Signature", "Weighted Score", "Occurrences", "SignatureTuple"])

    return pd.DataFrame(rows).sort_values(
        ["Weighted Score", "Occurrences", "Signature"], ascending=[False, False, True]
    ).reset_index(drop=True)


@st.cache_data
def _build_thunderball_weights(df: pd.DataFrame) -> np.ndarray:
    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    weights = np.ones(THUNDERBALL_MAX, dtype=float)

    for idx, (_, row) in enumerate(ordered.iterrows()):
        thunderball = int(row["thunderball"])
        weights[thunderball - 1] += 0.985**idx

    return weights / weights.sum()


def _generate_delta_tickets(
    signature_df: pd.DataFrame,
    thunderball_weights: np.ndarray,
    ticket_count: int,
    top_signatures: int,
    seed: int,
) -> list[DeltaTicket]:
    rng = np.random.default_rng(seed)

    usable = signature_df.head(max(1, top_signatures)).copy()
    if usable.empty:
        return []

    signatures = usable["SignatureTuple"].tolist()
    signature_weights = usable["Weighted Score"].to_numpy(dtype=float)
    signature_weights = signature_weights / signature_weights.sum()

    tickets: list[DeltaTicket] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    max_attempts = ticket_count * 60
    for _ in range(max_attempts):
        if len(tickets) >= ticket_count:
            break

        signature = signatures[int(rng.choice(np.arange(len(signatures)), p=signature_weights))]
        start = int(rng.integers(1, MAX_MAIN_NUMBER + 1))
        main_numbers = _numbers_from_start_and_signature(start, signature)
        if main_numbers is None:
            continue

        thunderball = int(rng.choice(np.arange(1, THUNDERBALL_MAX + 1), p=thunderball_weights))
        key = (main_numbers, thunderball)
        if key in seen:
            continue

        seen.add(key)
        tickets.append(
            DeltaTicket(
                main_numbers=(
                    int(main_numbers[0]),
                    int(main_numbers[1]),
                    int(main_numbers[2]),
                    int(main_numbers[3]),
                    int(main_numbers[4]),
                ),
                thunderball=thunderball,
                delta_signature=(
                    int(signature[0]),
                    int(signature[1]),
                    int(signature[2]),
                    int(signature[3]),
                    int(signature[4]),
                ),
            )
        )

    return tickets


def _ticket_payout(main_numbers: tuple[int, ...], thunderball: int, actual_main: tuple[int, ...], actual_tb: int) -> int:
    main_matches = len(set(main_numbers) & set(actual_main))
    thunder_match = thunderball == actual_tb
    return int(PRIZE_MATRIX.get((main_matches, thunder_match), 0))


@st.cache_data
def _run_delta_vs_current_backtest(
    df: pd.DataFrame,
    ticket_count: int,
    top_signatures: int,
    lookback_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    if len(ordered) <= lookback_draws:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows: list[dict[str, object]] = []

    for actual_idx in range(lookback_draws, len(ordered)):
        training = ordered.iloc[:actual_idx].copy()
        actual_row = ordered.iloc[actual_idx]
        actual_main = tuple(int(actual_row[col]) for col in ["n1", "n2", "n3", "n4", "n5"])
        actual_tb = int(actual_row["thunderball"])
        draw_date = actual_row["draw_date"].date().isoformat()

        training_signatures = _build_signature_scores(training)
        training_tb_weights = _build_thunderball_weights(training)
        delta_tickets = _generate_delta_tickets(
            signature_df=training_signatures,
            thunderball_weights=training_tb_weights,
            ticket_count=ticket_count,
            top_signatures=top_signatures,
            seed=seed,
        )

        current_portfolio = optimize_ticket_portfolio(
            training,
            ticket_count=ticket_count,
            target_payout=DEFAULT_TARGET_PAYOUT,
            seed=seed,
            objective_mode="downside_aware",
        )

        delta_payout = sum(
            _ticket_payout(ticket.main_numbers, ticket.thunderball, actual_main, actual_tb)
            for ticket in delta_tickets
        )
        current_payout = sum(
            _ticket_payout(ticket.main_numbers, int(ticket.thunderball), actual_main, actual_tb)
            for ticket in current_portfolio.tickets
        )

        delta_cost = len(delta_tickets) * TICKET_COST_GBP
        current_cost = len(current_portfolio.tickets) * TICKET_COST_GBP
        delta_net = delta_payout - delta_cost
        current_net = current_payout - current_cost

        detail_rows.append(
            {
                "Draw Date": draw_date,
                "Delta Payout": int(delta_payout),
                "Delta Cost": int(delta_cost),
                "Delta Net": int(delta_net),
                "Delta Target Hit": "Yes" if int(delta_payout) >= DEFAULT_TARGET_PAYOUT else "No",
                "Current Payout": int(current_payout),
                "Current Cost": int(current_cost),
                "Current Net": int(current_net),
                "Current Target Hit": "Yes" if int(current_payout) >= DEFAULT_TARGET_PAYOUT else "No",
                "Net Delta (Delta-Current)": int(delta_net - current_net),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = pd.DataFrame(
        [
            {
                "Model": "Delta System",
                "Draws Evaluated": int(len(detail_df)),
                "Total Payout": int(detail_df["Delta Payout"].sum()),
                "Total Cost": int(detail_df["Delta Cost"].sum()),
                "Net Result": int(detail_df["Delta Net"].sum()),
                "Target Hits": int((detail_df["Delta Target Hit"] == "Yes").sum()),
                "Avg Net/Draw": float(detail_df["Delta Net"].mean()),
            },
            {
                "Model": "Current Optimizer",
                "Draws Evaluated": int(len(detail_df)),
                "Total Payout": int(detail_df["Current Payout"].sum()),
                "Total Cost": int(detail_df["Current Cost"].sum()),
                "Net Result": int(detail_df["Current Net"].sum()),
                "Target Hits": int((detail_df["Current Target Hit"] == "Yes").sum()),
                "Avg Net/Draw": float(detail_df["Current Net"].mean()),
            },
        ]
    )

    summary_df["Avg Net/Draw"] = summary_df["Avg Net/Draw"].round(2)
    summary_df["Target Hit Rate"] = (
        summary_df["Target Hits"] / summary_df["Draws Evaluated"].replace(0, np.nan)
    ).fillna(0.0)
    detail_df = detail_df.sort_values("Draw Date", ascending=False).reset_index(drop=True)
    return summary_df, detail_df


st.set_page_config(page_title="Thunderball Delta System", page_icon="📐", layout="wide")
st.title("Thunderball Delta System")
st.caption(
    "Experimental page: analyze historical delta signatures and generate delta-based Thunderball tickets. "
    "For exploration and entertainment only."
)

with st.expander("Data Source", expanded=False):
    uploaded_file = st.file_uploader("Upload Thunderball CSV", type=["csv"], key="delta_upload")
    st.markdown("CSV columns required:")
    st.code("draw_date,n1,n2,n3,n4,n5,thunderball")

try:
    if uploaded_file is not None:
        history_df = _load_from_upload(uploaded_file.getvalue())
        source_label = "Uploaded CSV"
    else:
        history_df = _load_from_path(str(DEFAULT_DATA_PATH))
        source_label = f"Default ({DEFAULT_DATA_PATH})"
except (DataValidationError, FileNotFoundError) as exc:
    st.error(f"Failed to load draw history: {exc}")
    st.stop()

delta_df = _build_delta_frame(history_df)
signature_df = _build_signature_scores(history_df)

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.metric("Draws Loaded", len(history_df))
meta_col2.metric("Unique Delta Signatures", signature_df["Signature"].nunique())
meta_col3.metric("Data Source", source_label)

if not signature_df.empty:
    top_row = signature_df.iloc[0]
    st.caption(
        f"Most frequent weighted signature: {top_row['Signature']} "
        f"(occurrences: {int(top_row['Occurrences'])}, weighted score: {float(top_row['Weighted Score']):.2f})"
    )

st.subheader("Delta Signature Frequency")
if signature_df.empty:
    st.info("No signatures available to display.")
else:
    chart_df = signature_df.copy()
    chart_df = chart_df.head(20)
    signature_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Signature:N", sort="-y", title="Delta Signature"),
            y=alt.Y("Occurrences:Q", title="Occurrences"),
            tooltip=[
                alt.Tooltip("Signature:N"),
                alt.Tooltip("Occurrences:Q"),
                alt.Tooltip("Weighted Score:Q", format=".2f"),
            ],
            color=alt.Color("Weighted Score:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        )
        .properties(height=320)
    )
    st.altair_chart(signature_chart, use_container_width=True)

st.subheader("Delta Values by Position")
if delta_df.empty:
    st.info("No delta values available.")
else:
    long_delta = delta_df.melt(
        id_vars=["Draw Date", "Signature"],
        value_vars=["Delta 1", "Delta 2", "Delta 3", "Delta 4", "Delta 5"],
        var_name="Position",
        value_name="Delta Value",
    )
    position_chart = (
        alt.Chart(long_delta)
        .mark_bar()
        .encode(
            x=alt.X("Delta Value:O", title="Delta Value"),
            y=alt.Y("count():Q", title="Frequency"),
            color=alt.Color(
                "Position:N",
                scale=alt.Scale(range=["#0f4c5c", "#2c7a7b", "#84a98c", "#f6bd60", "#e76f51"]),
            ),
            tooltip=[alt.Tooltip("Position:N"), alt.Tooltip("Delta Value:Q"), alt.Tooltip("count():Q")],
        )
        .properties(height=300)
    )
    st.altair_chart(position_chart, use_container_width=True)

with st.expander("Show Delta Signature Table", expanded=False):
    show_df = signature_df[["Signature", "Occurrences", "Weighted Score"]].copy()
    show_df["Weighted Score"] = show_df["Weighted Score"].round(3)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

st.subheader("Delta-Based Ticket Generator")
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
with ctrl_col1:
    delta_ticket_count = st.slider("Ticket Count", min_value=1, max_value=20, value=DEFAULT_TICKET_COUNT, step=1)
with ctrl_col2:
    top_signature_count = st.slider("Top Signatures Considered", min_value=1, max_value=30, value=10, step=1)
with ctrl_col3:
    delta_seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42, step=1)

generated_tickets = _generate_delta_tickets(
    signature_df=signature_df,
    thunderball_weights=_build_thunderball_weights(history_df),
    ticket_count=int(delta_ticket_count),
    top_signatures=int(top_signature_count),
    seed=int(delta_seed),
)

if not generated_tickets:
    st.warning("No tickets could be generated from the available signature pool.")
else:
    st.caption(
        "Tickets are generated from weighted historical delta signatures. "
        "Main numbers are reconstructed from a random start point and signature; thunderball uses weighted historical sampling."
    )

    ticket_rows = []
    for index, ticket in enumerate(generated_tickets, start=1):
        ticket_rows.append(
            {
                "Ticket": index,
                "Main Numbers": "-".join(f"{number:02d}" for number in ticket.main_numbers),
                "Thunderball": f"{ticket.thunderball:02d}",
                "Delta Signature": "-".join(str(value) for value in ticket.delta_signature),
            }
        )

    st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True, hide_index=True)

st.subheader("Side-by-Side Next Draw Comparison")
st.caption(
    "Compares tickets generated by the Delta System and the current downside-aware optimizer on the same latest history window."
)

current_portfolio = optimize_ticket_portfolio(
    history_df,
    ticket_count=int(delta_ticket_count),
    target_payout=DEFAULT_TARGET_PAYOUT,
    seed=int(delta_seed),
    objective_mode="downside_aware",
)

compare_left, compare_right = st.columns(2)
with compare_left:
    st.markdown("**Delta System Tickets**")
    delta_rows = [
        {
            "Ticket": index + 1,
            "Main Numbers": "-".join(f"{number:02d}" for number in ticket.main_numbers),
            "Thunderball": f"{ticket.thunderball:02d}",
            "Delta Signature": "-".join(str(value) for value in ticket.delta_signature),
        }
        for index, ticket in enumerate(generated_tickets)
    ]
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

with compare_right:
    st.markdown("**Current Optimizer Tickets**")
    current_rows = [
        {
            "Ticket": index + 1,
            "Main Numbers": "-".join(f"{number:02d}" for number in ticket.main_numbers),
            "Thunderball": f"{int(ticket.thunderball):02d}",
        }
        for index, ticket in enumerate(current_portfolio.tickets)
    ]
    st.dataframe(pd.DataFrame(current_rows), use_container_width=True, hide_index=True)

st.subheader("Rolling Backtest: Delta vs Current Optimizer")
st.caption(
    "Evaluates both models draw-by-draw from the selected lookback onward using only prior-history training windows."
)

backtest_col1, backtest_col2 = st.columns(2)
with backtest_col1:
    lookback_draws = st.slider(
        "Minimum Training Draws",
        min_value=5,
        max_value=30,
        value=DEFAULT_LOOKBACK_DRAWS,
        step=1,
        key="delta_backtest_lookback",
    )
with backtest_col2:
    run_backtest = st.button("Run Delta vs Current Backtest", key="run_delta_compare_backtest")

if run_backtest:
    with st.spinner("Running rolling comparison backtest..."):
        backtest_summary_df, backtest_detail_df = _run_delta_vs_current_backtest(
            history_df,
            ticket_count=int(delta_ticket_count),
            top_signatures=int(top_signature_count),
            lookback_draws=int(lookback_draws),
            seed=int(delta_seed),
        )

        st.session_state["delta_backtest_summary"] = backtest_summary_df
        st.session_state["delta_backtest_detail"] = backtest_detail_df

backtest_summary_df = st.session_state.get("delta_backtest_summary")
backtest_detail_df = st.session_state.get("delta_backtest_detail")

if backtest_summary_df is not None and backtest_detail_df is not None and not backtest_summary_df.empty:
    display_summary_df = backtest_summary_df.copy()
    display_summary_df["Target Hit Rate"] = display_summary_df["Target Hit Rate"].map(
        lambda value: f"{float(value):.1%}"
    )
    st.dataframe(display_summary_df, use_container_width=True, hide_index=True)

    if not backtest_detail_df.empty:
        delta_net_total = int(backtest_detail_df["Delta Net"].sum())
        current_net_total = int(backtest_detail_df["Current Net"].sum())
        net_advantage = delta_net_total - current_net_total
        delta_hits = int((backtest_detail_df["Delta Target Hit"] == "Yes").sum())
        current_hits = int((backtest_detail_df["Current Target Hit"] == "Yes").sum())
        draws_evaluated = int(len(backtest_detail_df))
        delta_hit_rate = (delta_hits / draws_evaluated) if draws_evaluated > 0 else 0.0
        current_hit_rate = (current_hits / draws_evaluated) if draws_evaluated > 0 else 0.0
        hit_rate_advantage = delta_hit_rate - current_hit_rate
        hit_advantage = int(
            (backtest_detail_df["Delta Target Hit"] == "Yes").sum()
            - (backtest_detail_df["Current Target Hit"] == "Yes").sum()
        )

        if delta_net_total > current_net_total:
            winner = "Delta System"
        elif current_net_total > delta_net_total:
            winner = "Current Optimizer"
        elif delta_hits > current_hits:
            winner = "Delta System"
        elif current_hits > delta_hits:
            winner = "Current Optimizer"
        elif float(backtest_summary_df.loc[backtest_summary_df["Model"] == "Delta System", "Avg Net/Draw"].iloc[0]) > float(
            backtest_summary_df.loc[backtest_summary_df["Model"] == "Current Optimizer", "Avg Net/Draw"].iloc[0]
        ):
            winner = "Delta System"
        elif float(backtest_summary_df.loc[backtest_summary_df["Model"] == "Current Optimizer", "Avg Net/Draw"].iloc[0]) > float(
            backtest_summary_df.loc[backtest_summary_df["Model"] == "Delta System", "Avg Net/Draw"].iloc[0]
        ):
            winner = "Current Optimizer"
        else:
            winner = "Tie"

        if winner == "Tie":
            st.info("Winner: Tie after net, target hits, and average net per draw tie-breakers.")
        else:
            st.success(f"Winner: {winner}")

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Net Advantage (Delta - Current)", f"GBP{net_advantage}")
        metric_b.metric("Target Hit Advantage", hit_advantage)
        metric_c.metric("Delta Target Hit Rate", f"{delta_hit_rate:.1%}")
        metric_d.metric("Current Target Hit Rate", f"{current_hit_rate:.1%}")
        st.caption(
            f"Likelihood edge for target hit (Delta - Current): {hit_rate_advantage:+.1%} over {draws_evaluated} evaluated draws."
        )

        hit_compare_df = pd.DataFrame(
            [
                {"Model": "Delta System", "Target Hit Rate": delta_hit_rate * 100},
                {"Model": "Current Optimizer", "Target Hit Rate": current_hit_rate * 100},
            ]
        )
        hit_chart = (
            alt.Chart(hit_compare_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title=None),
                y=alt.Y("Target Hit Rate:Q", title="Target Hit Rate (%)"),
                color=alt.Color("Model:N", scale=alt.Scale(range=["#1b4965", "#5fa8d3"]), legend=None),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Target Hit Rate:Q", format=".2f", title="Target Hit Rate (%)"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(hit_chart, use_container_width=True)

        with st.expander("Show draw-by-draw comparison", expanded=False):
            st.dataframe(backtest_detail_df, use_container_width=True, hide_index=True)
