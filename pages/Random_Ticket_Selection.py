from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from thunderball_predictor.loader import DataValidationError, load_draw_history

DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")

st.set_page_config(page_title="Random Ticket Selection", page_icon="🎲", layout="wide")
st.title("Random Ticket Selection")
st.caption(
    "Baseline method: pure random ticket generation with no pattern analysis, pattern frequency weighting, or range signals. "
    "This serves as a control to see how random play compares to structured prediction methods."
)
st.markdown(
    "This method generates valid Thunderball tickets by uniformly sampling random main ball combinations and Thunderballs. "
    "No historical data is analyzed—each draw from the ticket pool has equal probability. "
    "Use this page to establish a statistical baseline for lottery prediction."
)

# ── Data loading ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload draw history CSV (optional)", type="csv")
    st.divider()
    st.header("Prediction settings")
    ticket_count = st.slider(
        "Ticket Count",
        min_value=1,
        max_value=20,
        value=9,
        step=1,
    )
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
    )


@st.cache_data
def _load_from_path(path: str) -> pd.DataFrame:
    return load_draw_history(path)


@st.cache_data
def _load_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    temp_path = Path("/tmp/thunderball_random.csv")
    temp_path.write_bytes(uploaded_bytes)
    return load_draw_history(temp_path)


try:
    if uploaded is not None:
        df = _load_from_upload(uploaded.read())
    else:
        df = _load_from_path(str(DEFAULT_DATA_PATH))
except DataValidationError as exc:
    st.error(f"Data validation error: {exc}")
    st.stop()
except FileNotFoundError:
    st.error(f"Default data file not found: {DEFAULT_DATA_PATH}")
    st.stop()

# ── Load data metadata ────────────────────────────────────────────────────────

meta_col1, meta_col2 = st.columns(2)
meta_col1.metric("Draws in history", len(df))
meta_col2.metric("Data source", "Uploaded CSV" if uploaded is not None else f"Default ({DEFAULT_DATA_PATH})")

# ── Method explanation ────────────────────────────────────────────────────────

st.subheader("How Random Ticket Selection Works")
st.markdown(
    """
The random method is conceptually simple:
1. Generate uniformly random main-ball combinations (5 different numbers from 1–39)
2. Generate uniformly random Thunderballs (1–14)
3. Ensure all tickets are unique
4. Evaluate against actual draw history to measure performance

This is a **null hypothesis** method—if pattern-based methods claim to have predictive power, they should outperform random 
selection by a margin larger than would be expected from luck. If they underperform random, the pattern is misleading.
"""
)

# ── Random ticket generation ──────────────────────────────────────────────────

st.subheader("Generate Random 9-Ticket Portfolio")

@st.cache_data
def _generate_random_tickets(ticket_count: int, seed: int) -> list[tuple[int, int, int, int, int, int]]:
    import random as _rnd
    rng = _rnd.Random(seed)
    tickets: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(ticket_count * 500):
        if len(tickets) >= ticket_count:
            break
        # Generate 5 unique random main balls in sorted order
        main_balls = tuple(sorted(rng.sample(range(1, 40), 5)))
        # Generate random thunderball
        thunderball = rng.randint(1, 14)
        ticket = (*main_balls, thunderball)
        if ticket not in seen:
            seen.add(ticket)
            tickets.append(ticket)

    return tickets


generated_tickets = _generate_random_tickets(int(ticket_count), int(seed))

if generated_tickets:
    ticket_rows = [
        {
            "Ticket": idx + 1,
            "Main Numbers": "-".join(f"{n:02d}" for n in ticket[:5]),
            "Thunderball": f"{ticket[5]:02d}",
        }
        for idx, ticket in enumerate(generated_tickets)
    ]
    st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True, hide_index=True)
    st.caption(f"Generated {len(generated_tickets)} random ticket(s) using seed {int(seed)}.")
else:
    st.warning("Could not generate random tickets.")

# ── Historical back-test P&L ──────────────────────────────────────────────────

st.divider()
st.header("Historical Back-Test P&L")
st.caption(
    "For every draw in history (after a warm-up period), 9 random tickets are generated and evaluated against the actual result using the standard Thunderball prize matrix. Cost = £9 per draw."
)

PRIZE_MATRIX: dict[tuple[int, bool], int] = {
    (5, True): 500_000,
    (5, False): 5_000,
    (4, True): 100,
    (4, False): 250,
    (3, True): 20,
    (3, False): 10,
    (2, True): 10,
    (1, True): 5,
    (0, True): 3,
}
TICKET_COST = 1  # £ per ticket
TARGET_PAYOUT = 10


def _ticket_payout(ticket: tuple[int, ...], actual_main: set[int], actual_tb: int) -> int:
    main_hits = len(set(ticket[:5]) & actual_main)
    tb_hit = ticket[5] == actual_tb
    return PRIZE_MATRIX.get((main_hits, tb_hit), 0)


@st.cache_data
def _run_random_backtest(df: pd.DataFrame, ticket_count: int, seed: int, warmup: int = 10) -> pd.DataFrame:
    import random as _r
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for actual_idx in range(warmup, len(ordered)):
        rng = _r.Random(seed + actual_idx)
        actual_row = ordered.iloc[actual_idx]
        actual_main = {int(actual_row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]}
        actual_tb = int(actual_row["thunderball"])

        # Generate random tickets
        tickets: list[tuple[int, int, int, int, int, int]] = []
        seen: set[tuple[int, int, int, int, int, int]] = set()
        for _ in range(ticket_count * 100):
            if len(tickets) >= ticket_count:
                break
            main_balls = tuple(sorted(rng.sample(range(1, 40), 5)))
            thunderball = rng.randint(1, 14)
            ticket = (*main_balls, thunderball)
            if ticket not in seen:
                seen.add(ticket)
                tickets.append(ticket)

        if not tickets:
            continue

        total_payout = sum(_ticket_payout(t, actual_main, actual_tb) for t in tickets)
        cost = len(tickets) * TICKET_COST
        best_main = max(len(set(t[:5]) & actual_main) for t in tickets) if tickets else 0
        tb_hits = sum(1 for t in tickets if t[5] == actual_tb)

        rows.append({
            "Draw Date": actual_row["draw_date"].date().isoformat(),
            "Actual": "-".join(str(b) for b in sorted(actual_main)) + f" | TB {actual_tb}",
            "Cost": cost,
            "Payout": total_payout,
            "Net": total_payout - cost,
            "Best Main Matches": best_main,
            "TB Hits": tb_hits,
            "Winning Tickets": sum(1 for t in tickets if _ticket_payout(t, actual_main, actual_tb) > 0),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Cumulative Net"] = result["Net"].cumsum()
    return result


with st.spinner("Running back-test over draw history…"):
    bt_df = _run_random_backtest(df, ticket_count=int(ticket_count), seed=int(seed))

if bt_df.empty:
    st.warning("Not enough draw history to back-test.")
else:
    total_cost = int(bt_df["Cost"].sum())
    total_payout = int(bt_df["Payout"].sum())
    total_net = total_payout - total_cost
    draws_played = len(bt_df)
    profitable_draws = int((bt_df["Net"] > 0).sum())
    best_draw_net = int(bt_df["Net"].max())

    kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
    kc1.metric("Draws back-tested", draws_played)
    kc2.metric("Total staked", f"£{total_cost:,}")
    kc3.metric("Total returned", f"£{total_payout:,}")
    kc4.metric("Net P&L", f"£{total_net:,}", delta=f"£{total_net:,}")
    kc5.metric("Profitable draws", f"{profitable_draws} / {draws_played}")
    kc6.metric("Best single draw", f"£{best_draw_net:,} net")

    # Cumulative P&L chart
    cum_chart = (
        alt.Chart(bt_df)
        .mark_line(color="#95a5a6", strokeWidth=2)
        .encode(
            x=alt.X("Draw Date:O", axis=alt.Axis(labels=False, ticks=False, title="Draw (oldest → newest)"), sort=None),
            y=alt.Y("Cumulative Net:Q", title="Cumulative net (£)"),
            tooltip=[
                alt.Tooltip("Draw Date:O", title="Draw date"),
                alt.Tooltip("Cumulative Net:Q", title="Cumulative net £"),
                alt.Tooltip("Net:Q", title="Draw net £"),
                alt.Tooltip("Payout:Q", title="Payout £"),
                alt.Tooltip("Best Main Matches:Q"),
            ],
        )
        .properties(height=260, width="container", title="Cumulative net P&L over draw history")
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )

    # Zero line
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="grey", strokeDash=[3, 3], strokeWidth=1)
        .encode(y="y:Q")
    )

    st.altair_chart(cum_chart, use_container_width=True)

    # Per-draw net bar chart
    bar_colour_expr = alt.condition(
        alt.datum["Net"] > 0,
        alt.value("#2ecc71"),
        alt.value("#e74c3c"),
    )
    bar_chart = (
        alt.Chart(bt_df)
        .mark_bar(size=3)
        .encode(
            x=alt.X("Draw Date:O", axis=alt.Axis(labels=False, ticks=False, title="Draw (oldest → newest)"), sort=None),
            y=alt.Y("Net:Q", title="Net per draw (£)"),
            color=bar_colour_expr,
            tooltip=[
                alt.Tooltip("Draw Date:O", title="Draw date"),
                alt.Tooltip("Net:Q", title="Net £"),
                alt.Tooltip("Payout:Q", title="Payout £"),
                alt.Tooltip("Cost:Q", title="Cost £"),
                alt.Tooltip("Best Main Matches:Q"),
                alt.Tooltip("TB Hits:Q"),
                alt.Tooltip("Winning Tickets:Q"),
            ],
        )
        .properties(height=200, width="container", title="Net per draw (green = profit, red = loss)")
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    with st.expander("Full back-test results table"):
        display_bt = bt_df.copy()
        display_bt["Cost"] = display_bt["Cost"].apply(lambda v: f"£{v}")
        display_bt["Payout"] = display_bt["Payout"].apply(lambda v: f"£{v}")
        display_bt["Net"] = display_bt["Net"].apply(lambda v: f"£{v:+d}")
        display_bt["Cumulative Net"] = display_bt["Cumulative Net"].apply(lambda v: f"£{int(v):+d}")
        st.dataframe(display_bt, use_container_width=True, hide_index=True)

    st.subheader("Method Verdict")
    average_net = float(bt_df["Net"].mean()) if not bt_df.empty else 0.0
    target_like_draws = int((bt_df["Payout"] >= TARGET_PAYOUT).sum()) if not bt_df.empty else 0

    if total_net > 0:
        st.info(
            "Random selection produced a positive net result in this historical window by chance. "
            "This shows that lottery outcomes are highly random and that beating this baseline would require genuine predictive insight."
        )
    elif total_net == 0:
        st.info(
            "Random selection broke even in this window. Any betting method should substantially outperform random selection to be worth considering."
        )
    else:
        st.warning(
            "Random selection lost money (as expected from a negative-EV game). "
            "This is the statistical baseline against which all other methods should be compared."
        )

    st.caption(
        f"Average net per draw: £{average_net:.2f} | Profitable draws: {profitable_draws}/{draws_played} | "
        f"Draws returning at least £{TARGET_PAYOUT}: {target_like_draws}/{draws_played}"
    )

st.divider()
st.header("Final Next 9-Ticket Prediction")
st.caption(
    "This is the random method's current next-draw portfolio using the active seed."
)
final_ticket_rows = [
    {
        "Ticket": idx + 1,
        "Main Numbers": "-".join(f"{n:02d}" for n in ticket[:5]),
        "Thunderball": f"{ticket[5]:02d}",
    }
    for idx, ticket in enumerate(generated_tickets)
]
if final_ticket_rows:
    st.dataframe(pd.DataFrame(final_ticket_rows), use_container_width=True, hide_index=True)
else:
    st.info("No random ticket set is currently available.")
