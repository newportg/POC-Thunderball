from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from thunderball_predictor.loader import DataValidationError, load_draw_history

DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")

st.set_page_config(page_title="Main Ball Sum Filter", page_icon="🔢", layout="wide")
st.title("Main Ball Sum Filter")
st.caption(
    "The sum of the five main balls in each draw forms a near-normal distribution. "
    "This method constrains every predicted ticket so its main-ball sum falls within the "
    "historically likely bell-curve range — filtering out combinations that are statistically rare."
)
st.markdown(
    """
**Theory:** When five numbers are drawn uniformly from 1–39 their sum is approximately normally distributed 
with a theoretical mean near 100. Sums in the extreme tails (very high or very low) have occurred far less 
frequently than central sums. By restricting tickets to the central band we avoid generating combinations 
that history suggests are unlikely.

The *sigma multiplier* on this page lets you choose how tight the filter is:
- **±1σ** captures ~68% of historical draws — a tight central band
- **±1.5σ** captures ~87%
- **±2σ** captures ~95% — a wide band that still excludes rare extremes
"""
)

# ── Data loading ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload draw history CSV (optional)", type="csv")
    st.divider()
    st.header("Filter settings")
    sigma_multiplier = st.slider(
        "Bell-curve sigma multiplier (±σ)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.25,
        help="Predicted tickets must have a main-ball sum within mean ± (σ × this value).",
    )
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        step=1,
    )
    ticket_count = st.slider(
        "Ticket Count",
        min_value=1,
        max_value=20,
        value=9,
        step=1,
    )


@st.cache_data
def _load_from_path(path: str) -> pd.DataFrame:
    return load_draw_history(path)


@st.cache_data
def _load_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    temp_path = Path("/tmp/thunderball_sum_filter.csv")
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

# ── Compute sum distribution ──────────────────────────────────────────────────

@st.cache_data
def _compute_sums(df: pd.DataFrame) -> pd.Series:
    return df[["n1", "n2", "n3", "n4", "n5"]].apply(
        lambda row: int(row["n1"]) + int(row["n2"]) + int(row["n3"]) + int(row["n4"]) + int(row["n5"]),
        axis=1,
    )


sums = _compute_sums(df)
sum_mean = float(sums.mean())
sum_std = float(sums.std())
sum_min = int(sums.min())
sum_max = int(sums.max())
sum_p10 = float(sums.quantile(0.10))
sum_p90 = float(sums.quantile(0.90))

# Active filter bounds from selected sigma
filter_low = sum_mean - sigma_multiplier * sum_std
filter_high = sum_mean + sigma_multiplier * sum_std
# Clamp to realistic range
filter_low = max(15.0, filter_low)   # min possible sum: 1+2+3+4+5
filter_high = min(185.0, filter_high)  # max possible sum: 35+36+37+38+39

# Coverage = fraction of historical draws captured by this filter
draws_in_range = int(((sums >= filter_low) & (sums <= filter_high)).sum())
coverage_pct = draws_in_range / len(sums) * 100

# ── Analysis section ──────────────────────────────────────────────────────────

st.subheader("Historical Sum Distribution")

kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
kc1.metric("Draws", len(sums))
kc2.metric("Mean sum", f"{sum_mean:.1f}")
kc3.metric("Std dev", f"{sum_std:.1f}")
kc4.metric("Min / Max", f"{sum_min} / {sum_max}")
kc5.metric("Filter range", f"{int(filter_low)}–{int(filter_high)}")
kc6.metric("Draws in range", f"{draws_in_range} ({coverage_pct:.0f}%)")

# Histogram
sum_df = pd.DataFrame({"Sum": sums})
hist_base = (
    alt.Chart(sum_df)
    .mark_bar(color="#4c9be8", opacity=0.75)
    .encode(
        x=alt.X(
            "Sum:Q",
            bin=alt.Bin(step=5),
            title="Main ball sum",
        ),
        y=alt.Y("count():Q", title="Frequency"),
        tooltip=[
            alt.Tooltip("Sum:Q", bin=alt.Bin(step=5), title="Sum bucket"),
            alt.Tooltip("count():Q", title="Draws"),
        ],
    )
    .properties(height=300, width="container", title="Distribution of main-ball sums across all historical draws")
)
st.altair_chart(
    hist_base.configure_view(strokeOpacity=0).configure_axis(grid=True, gridOpacity=0.25),
    use_container_width=True,
)

# Bell-curve overlay using normal PDF sampled over the sum range
x_range = np.linspace(sum_min - 10, sum_max + 10, 300)
pdf_values = (
    (1 / (sum_std * math.sqrt(2 * math.pi)))
    * np.exp(-0.5 * ((x_range - sum_mean) / sum_std) ** 2)
)
# Scale PDF to match histogram counts (density → count)
bin_width = 5
pdf_counts = pdf_values * len(sums) * bin_width
normal_df = pd.DataFrame({"x": x_range, "y": pdf_counts})

normal_chart = (
    alt.Chart(normal_df)
    .mark_line(color="#e74c3c", strokeWidth=2)
    .encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
    )
)

# Shaded filter band
band_df = pd.DataFrame({"x1": [filter_low], "x2": [filter_high]})
band_chart = (
    alt.Chart(band_df)
    .mark_rect(opacity=0.12, color="#2ecc71")
    .encode(
        x=alt.X("x1:Q"),
        x2=alt.X2("x2"),
    )
)

combined_chart = (
    (hist_base + normal_chart + band_chart)
    .properties(
        height=300,
        width="container",
        title=f"Sum distribution with fitted normal curve and ±{sigma_multiplier}σ filter band (green shading)",
    )
    .configure_view(strokeOpacity=0)
    .configure_axis(grid=True, gridOpacity=0.25)
)
st.altair_chart(combined_chart, use_container_width=True)
st.caption(
    f"Red curve = fitted normal distribution (μ={sum_mean:.1f}, σ={sum_std:.1f}). "
    f"Green band = active filter range [{int(filter_low)}, {int(filter_high)}] at ±{sigma_multiplier}σ."
)

# ── Ticket generation ─────────────────────────────────────────────────────────

st.subheader("Bell-Curve Constrained Ticket Generation")


@st.cache_data
def _generate_sum_filtered_tickets(
    sum_low: int,
    sum_high: int,
    ticket_count: int,
    seed: int,
) -> list[tuple[int, int, int, int, int, int]]:
    rng = np.random.default_rng(seed)
    tickets: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(ticket_count * 5000):
        if len(tickets) >= ticket_count:
            break
        main_balls = tuple(sorted(rng.choice(np.arange(1, 40), size=5, replace=False).tolist()))
        total = sum(main_balls)
        if not (sum_low <= total <= sum_high):
            continue
        thunderball = int(rng.integers(1, 15))
        ticket = (*main_balls, thunderball)
        if ticket not in seen:
            seen.add(ticket)
            tickets.append(ticket)

    return tickets


sum_low_int = int(round(filter_low))
sum_high_int = int(round(filter_high))

generated_tickets = _generate_sum_filtered_tickets(
    sum_low=sum_low_int,
    sum_high=sum_high_int,
    ticket_count=int(ticket_count),
    seed=int(seed),
)

if generated_tickets:
    st.caption(
        f"Each ticket's main-ball sum is constrained to [{sum_low_int}, {sum_high_int}] "
        f"(±{sigma_multiplier}σ of the historical mean {sum_mean:.1f})."
    )
    ticket_rows = [
        {
            "Ticket": idx + 1,
            "Main Numbers": "-".join(f"{n:02d}" for n in ticket[:5]),
            "Main Sum": sum(ticket[:5]),
            "Thunderball": f"{ticket[5]:02d}",
        }
        for idx, ticket in enumerate(generated_tickets)
    ]
    st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True, hide_index=True)
else:
    st.warning(
        f"Could not generate {ticket_count} unique tickets within sum range [{sum_low_int}, {sum_high_int}]. "
        "Try widening the sigma multiplier."
    )

# ── Historical back-test ──────────────────────────────────────────────────────

st.divider()
st.header("Historical Back-Test P&L")
st.caption(
    "For every draw in history (after a warm-up period), the sum-filter range is derived from all prior draws, "
    "and 9 tickets constrained to that range are evaluated against the actual result. Cost = £9 per draw."
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
def _run_sum_filter_backtest(
    df: pd.DataFrame,
    sigma_mult: float,
    n_tickets: int,
    seed: int,
    warmup: int = 15,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for actual_idx in range(warmup, len(ordered)):
        training = ordered.iloc[:actual_idx]
        actual_row = ordered.iloc[actual_idx]
        actual_main = {int(actual_row[c]) for c in ["n1", "n2", "n3", "n4", "n5"]}
        actual_tb = int(actual_row["thunderball"])

        # Compute sum stats from training window
        train_sums = training[["n1", "n2", "n3", "n4", "n5"]].sum(axis=1).astype(int)
        t_mean = float(train_sums.mean())
        t_std = float(train_sums.std())
        if t_std == 0:
            continue
        low = max(15, int(round(t_mean - sigma_mult * t_std)))
        high = min(185, int(round(t_mean + sigma_mult * t_std)))

        tickets: list[tuple[int, int, int, int, int, int]] = []
        seen: set[tuple[int, int, int, int, int, int]] = set()

        for _ in range(n_tickets * 5000):
            if len(tickets) >= n_tickets:
                break
            main_balls = tuple(sorted(int(v) for v in rng.choice(np.arange(1, 40), size=5, replace=False)))
            if not (low <= sum(main_balls) <= high):
                continue
            thunderball = int(rng.integers(1, 15))
            ticket = (*main_balls, thunderball)
            if ticket not in seen:
                seen.add(ticket)  # type: ignore[arg-type]
                tickets.append(ticket)

        if not tickets:
            continue

        total_payout = sum(_ticket_payout(t, actual_main, actual_tb) for t in tickets)
        cost = len(tickets) * TICKET_COST
        best_main = max(len(set(t[:5]) & actual_main) for t in tickets)
        tb_hits = sum(1 for t in tickets if t[5] == actual_tb)

        rows.append({
            "Draw Date": actual_row["draw_date"].date().isoformat(),
            "Actual": "-".join(str(b) for b in sorted(actual_main)) + f" | TB {actual_tb}",
            "Filter Range": f"{low}–{high}",
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
    bt_df = _run_sum_filter_backtest(df, sigma_mult=sigma_multiplier, n_tickets=int(ticket_count), seed=int(seed))

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
        .mark_line(color="#8e44ad", strokeWidth=2)
        .encode(
            x=alt.X("Draw Date:O", axis=alt.Axis(labels=False, ticks=False, title="Draw (oldest → newest)"), sort=None),
            y=alt.Y("Cumulative Net:Q", title="Cumulative net (£)"),
            tooltip=[
                alt.Tooltip("Draw Date:O", title="Draw date"),
                alt.Tooltip("Cumulative Net:Q", title="Cumulative net £"),
                alt.Tooltip("Net:Q", title="Draw net £"),
                alt.Tooltip("Filter Range:N", title="Sum filter range"),
                alt.Tooltip("Best Main Matches:Q"),
            ],
        )
        .properties(height=260, width="container", title="Cumulative net P&L over draw history")
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )
    st.altair_chart(cum_chart, use_container_width=True)

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
                alt.Tooltip("Filter Range:N", title="Sum filter range"),
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

    # Sum of actual draws vs filter range over time
    sums_over_time = df.sort_values("draw_date", ascending=True).copy()
    sums_over_time["draw_date_str"] = sums_over_time["draw_date"].dt.date.astype(str)
    sums_over_time["actual_sum"] = (
        sums_over_time["n1"].astype(int)
        + sums_over_time["n2"].astype(int)
        + sums_over_time["n3"].astype(int)
        + sums_over_time["n4"].astype(int)
        + sums_over_time["n5"].astype(int)
    )
    in_range_col = sums_over_time["actual_sum"].apply(
        lambda s: "In range" if filter_low <= s <= filter_high else "Out of range"
    )
    sums_over_time["Status"] = in_range_col

    dot_colour = alt.condition(
        alt.datum["Status"] == "In range",
        alt.value("#2ecc71"),
        alt.value("#e74c3c"),
    )
    dot_chart = (
        alt.Chart(sums_over_time)
        .mark_circle(size=20, opacity=0.6)
        .encode(
            x=alt.X("draw_date_str:O", axis=alt.Axis(labels=False, ticks=False, title="Draw (oldest → newest)"), sort=None),
            y=alt.Y("actual_sum:Q", title="Main ball sum"),
            color=dot_colour,
            tooltip=[
                alt.Tooltip("draw_date_str:O", title="Draw date"),
                alt.Tooltip("actual_sum:Q", title="Main sum"),
                alt.Tooltip("Status:N"),
            ],
        )
        .properties(
            height=220,
            width="container",
            title=f"Actual main-ball sums per draw — green = within ±{sigma_multiplier}σ filter, red = outside",
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )
    st.altair_chart(dot_chart, use_container_width=True)

    with st.expander("Full back-test results table"):
        display_bt = bt_df.copy()
        display_bt["Cost"] = display_bt["Cost"].apply(lambda v: f"£{v}")
        display_bt["Payout"] = display_bt["Payout"].apply(lambda v: f"£{v}")
        display_bt["Net"] = display_bt["Net"].apply(lambda v: f"£{v:+d}")
        display_bt["Cumulative Net"] = display_bt["Cumulative Net"].apply(lambda v: f"£{int(v):+d}")
        st.dataframe(display_bt, use_container_width=True, hide_index=True)

    st.subheader("Method Verdict")
    average_net = float(bt_df["Net"].mean())
    target_like_draws = int((bt_df["Payout"] >= TARGET_PAYOUT).sum())

    if total_net > 0:
        st.success(
            "Over the tested history the sum-filter method produced a positive overall net result. "
            "The bell-curve constraint is capturing combinations that align with historically observed sums, "
            "though this does not demonstrate forward predictive power."
        )
    elif total_net == 0:
        st.info(
            "The sum-filter method roughly broke even over the tested history. The filter is effective at eliminating "
            "rare-sum tickets but has not produced a meaningful edge against the baseline."
        )
    else:
        st.warning(
            "The sum-filter method lost money over the tested history. While the filter removes statistically rare "
            "combinations, lottery draws are independent events and past sum distributions do not predict future sums. "
            "Use this as an analytical tool rather than a betting strategy."
        )
    st.caption(
        f"Average net per draw: £{average_net:.2f} | Profitable draws: {profitable_draws}/{draws_played} | "
        f"Draws returning at least £{TARGET_PAYOUT}: {target_like_draws}/{draws_played} | "
        f"Filter coverage: {coverage_pct:.0f}% of all historical draws"
    )

# ── Final prediction ──────────────────────────────────────────────────────────

st.divider()
st.header("Final Next 9-Ticket Prediction")
st.caption(
    f"Each ticket's main-ball sum is constrained to [{sum_low_int}, {sum_high_int}] "
    f"(±{sigma_multiplier}σ of the full historical mean {sum_mean:.1f}). "
    "Thunderball is selected uniformly at random."
)
final_ticket_rows = [
    {
        "Ticket": idx + 1,
        "Main Numbers": "-".join(f"{n:02d}" for n in ticket[:5]),
        "Main Sum": sum(ticket[:5]),
        "Thunderball": f"{ticket[5]:02d}",
    }
    for idx, ticket in enumerate(generated_tickets)
]
if final_ticket_rows:
    st.dataframe(pd.DataFrame(final_ticket_rows), use_container_width=True, hide_index=True)
else:
    st.info("No sum-filtered ticket set is currently available. Try widening the sigma multiplier.")
