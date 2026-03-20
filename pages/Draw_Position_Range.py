from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from thunderball_predictor.loader import DataValidationError, load_draw_history

DEFAULT_DATA_PATH = Path("data/thunderball-draw-history.csv")

st.set_page_config(page_title="Draw Position Range", page_icon="📊", layout="wide")
st.title("Draw Position Range Analysis")
st.caption(
    "Each draw's main balls are sorted lowest to highest. "
    "For every sorted position (1st smallest through 5th smallest) the chart shows the "
    "value that appeared in that position across all historical draws, together with the "
    "running min and max envelope."
)

# ── Data loading ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload draw history CSV (optional)", type="csv")
    st.divider()
    st.header("Prediction settings")
    lookback = st.slider(
        "Momentum lookback (draws)",
        min_value=3,
        max_value=20,
        value=5,
        help="Number of recent draws used for the trend momentum signal.",
    )


@st.cache_data
def _load_from_path(path: str) -> pd.DataFrame:
    return load_draw_history(path)


@st.cache_data
def _load_from_upload(uploaded_bytes: bytes) -> pd.DataFrame:
    temp_path = Path("/tmp/thunderball_position_range.csv")
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

# ── Build sorted-position frame ───────────────────────────────────────────────

@st.cache_data
def _build_position_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame with columns: draw_date, position (1-5), value."""
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        sorted_balls = sorted(int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"])
        date_str = row["draw_date"].date().isoformat()
        for pos, ball in enumerate(sorted_balls, start=1):
            rows.append({"draw_date": date_str, "position": pos, "value": ball})
    return pd.DataFrame(rows)


pos_df = _build_position_frame(df)

# Chronological order for x-axis
pos_df["draw_date"] = pd.Categorical(
    pos_df["draw_date"],
    categories=sorted(pos_df["draw_date"].unique()),
    ordered=True,
)

# Per-position cumulative min/max  (chronologically)
@st.cache_data
def _build_envelope(pos_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pos in range(1, 6):
        sub = pos_df[pos_df["position"] == pos].sort_values("draw_date").reset_index(drop=True)
        sub = sub.copy()
        sub["running_min"] = sub["value"].cummin()
        sub["running_max"] = sub["value"].cummax()
        # Overall (final) min and max for reference lines
        sub["overall_min"] = int(sub["value"].min())
        sub["overall_max"] = int(sub["value"].max())
        records.append(sub)
    return pd.concat(records, ignore_index=True)


envelope_df = _build_envelope(pos_df)


# ── Prediction signals ────────────────────────────────────────────────────────

@st.cache_data
def _compute_position_predictions(pos_df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    For each sorted position predict Higher / Lower / Neutral for the next draw
    using three weighted heuristic signals:
      1. Range position  – where the last value sits in the all-time min/max range (weight ×2)
      2. Mean reversion  – whether the last value is above or below the historical mean (weight ×2)
      3. Trend momentum  – linear-regression slope over the last `lookback` draws (weight ×1)
    """
    results = []
    for pos in range(1, 6):
        sub = pos_df[pos_df["position"] == pos].sort_values("draw_date").reset_index(drop=True)
        values = sub["value"].tolist()

        overall_min = int(min(values))
        overall_max = int(max(values))
        overall_mean = sum(values) / len(values)
        last_value = int(values[-1])

        range_span = overall_max - overall_min
        range_pos = (last_value - overall_min) / range_span if range_span > 0 else 0.5

        # Signal 1: range position (upper ≥65 % → lower, lower ≤35 % → higher)
        if range_pos >= 0.65:
            range_signal = -1
        elif range_pos <= 0.35:
            range_signal = 1
        else:
            range_signal = 0

        # Signal 2: mean reversion
        if last_value > overall_mean:
            reversion_signal = -1
        elif last_value < overall_mean:
            reversion_signal = 1
        else:
            reversion_signal = 0

        # Signal 3: trend momentum — linear regression slope over last N draws
        recent = values[-lookback:] if len(values) >= lookback else values[:]
        trend_signal = 0
        if len(recent) >= 2:
            n = len(recent)
            x = list(range(n))
            xm = sum(x) / n
            ym = sum(recent) / n
            num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, recent))
            den = sum((xi - xm) ** 2 for xi in x)
            slope = num / den if den != 0 else 0.0
            if slope > 0.3:
                trend_signal = 1
            elif slope < -0.3:
                trend_signal = -1

        combined = range_signal * 2 + reversion_signal * 2 + trend_signal
        if combined > 0:
            prediction = "Higher"
        elif combined < 0:
            prediction = "Lower"
        else:
            prediction = "Neutral"

        results.append({
            "position": pos,
            "last_value": last_value,
            "overall_min": overall_min,
            "overall_max": overall_max,
            "overall_mean": round(overall_mean, 2),
            "range_position_pct": round(range_pos * 100, 1),
            "range_signal": range_signal,
            "reversion_signal": reversion_signal,
            "trend_signal": trend_signal,
            "combined_score": combined,
            "prediction": prediction,
            "confidence": round(abs(combined) / 5.0, 2),
        })

    return pd.DataFrame(results)


predictions_df = _compute_position_predictions(pos_df, lookback=lookback)

_PRED_ARROW = {"Higher": "↑", "Lower": "↓", "Neutral": "→"}
_PRED_COLOUR = {"Higher": "green", "Lower": "red", "Neutral": "grey"}
_POS_SHORT = {1: "Pos 1 · Smallest", 2: "Pos 2", 3: "Pos 3 · Middle", 4: "Pos 4", 5: "Pos 5 · Largest"}

# ── Summary table ─────────────────────────────────────────────────────────────

st.subheader("Position summary")
summary = (
    envelope_df.groupby("position")["value"]
    .agg(Min="min", Max="max", Mean="mean", Median="median", Std="std")
    .round(2)
    .reset_index()
)
summary.columns = ["Position", "Min", "Max", "Mean", "Median", "Std Dev"]
summary["Position"] = summary["Position"].map(
    {1: "1st (smallest)", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th (largest)"}
)
st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Next draw prediction overview ─────────────────────────────────────────────

st.subheader("Next draw predictions")
st.caption(
    "Three heuristic signals combined: range position, mean reversion, and trend momentum. "
    "Lottery draws are independent events — treat these as analytical observations only."
)

pred_overview_cols = st.columns(5)
for _pos in range(1, 6):
    _pr = predictions_df[predictions_df["position"] == _pos].iloc[0]
    _arrow = _PRED_ARROW[_pr["prediction"]]
    _colour = _PRED_COLOUR[_pr["prediction"]]
    _conf = round(float(_pr["confidence"]) * 100)
    with pred_overview_cols[_pos - 1]:
        st.markdown(f"**{_POS_SHORT[_pos]}**")
        st.markdown(f":{_colour}[**{_arrow} {_pr['prediction']}**]")
        st.caption(f"Last: {_pr['last_value']} · {_pr['range_position_pct']}% of range · {_conf}% confidence")

# ── Range-direction 9-ticket prediction ──────────────────────────────────────

@st.cache_data
def _generate_range_tickets(
    pos_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    tb_pred: dict,
    n_tickets: int = 9,
    seed: int = 42,
) -> tuple[list[tuple[int, int, int, int, int, int]], dict[int, tuple[int, int, int, int, str]], tuple[int, int, str]]:
    """
    Build n_tickets whose main-ball values lie in the direction-biased zone for each
    sorted position, and whose Thunderball lies in the direction-biased TB zone.
    Zone boundaries:
      Higher  -> [last_value, historical_max]   (last ball is the lower fence)
      Lower   -> [historical_min, last_value]   (last ball is the upper fence)
      Neutral -> [historical_min, historical_max]
    Strict ordering (pos1 < pos2 < .. < pos5) is enforced with a forward/backward pass.
    Returns (tickets, main_zones, tb_zone) where each ticket is (b1,b2,b3,b4,b5,thunderball).
    """
    import random as _rnd
    rng = _rnd.Random(seed)

    # Main-ball zones
    zones: dict[int, tuple[int, int, int, int, str]] = {}
    for pos in range(1, 6):
        sub = pos_df[pos_df["position"] == pos]["value"]
        p_min = int(sub.min())
        p_max = int(sub.max())
        pred_row = predictions_df[predictions_df["position"] == pos].iloc[0]
        last_val = int(pred_row["last_value"])
        direction = str(pred_row["prediction"])
        if direction == "Higher":
            lo, hi = last_val, p_max
        elif direction == "Lower":
            lo, hi = p_min, last_val
        else:
            lo, hi = p_min, p_max
        lo = max(1, lo)
        hi = min(39, max(hi, lo + 1))
        zones[pos] = (lo, hi, p_min, p_max, direction)

    # Thunderball zone
    tb_last = int(tb_pred["last_value"])
    tb_direction = str(tb_pred["prediction"])
    if tb_direction == "Higher":
        tb_lo, tb_hi = tb_last, int(tb_pred["tb_max"])
    elif tb_direction == "Lower":
        tb_lo, tb_hi = int(tb_pred["tb_min"]), tb_last
    else:
        tb_lo, tb_hi = int(tb_pred["tb_min"]), int(tb_pred["tb_max"])
    tb_lo = max(1, tb_lo)
    tb_hi = min(14, max(tb_hi, tb_lo + 1))
    tb_zone = (tb_lo, tb_hi, tb_direction)

    tickets: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(n_tickets * 500):
        if len(tickets) >= n_tickets:
            break
        raw = [rng.randint(zones[p][0], zones[p][1]) for p in range(1, 6)]

        # Forward pass: enforce strict ascending order
        for i in range(1, 5):
            if raw[i] <= raw[i - 1]:
                raw[i] = raw[i - 1] + 1

        # Backward pass: clamp if we overshot 39
        if raw[4] > 39:
            raw[4] = 39
            for i in range(3, -1, -1):
                if raw[i] >= raw[i + 1]:
                    raw[i] = raw[i + 1] - 1

        tb_val = rng.randint(tb_lo, tb_hi)
        ticket = (raw[0], raw[1], raw[2], raw[3], raw[4], tb_val)
        if (
            ticket not in seen
            and raw[0] >= 1
            and raw[4] <= 39
            and len(set(raw)) == 5
            and all(raw[i] < raw[i + 1] for i in range(4))
        ):
            seen.add(ticket)
            tickets.append(ticket)

    return tickets, zones, tb_zone


# ── Thunderball helpers (needed before ticket generation) ────────────────────

@st.cache_data
def _build_tb_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.sort_values("draw_date", ascending=True).iterrows():
        rows.append({"draw_date": row["draw_date"].date().isoformat(), "value": int(row["thunderball"])})
    return pd.DataFrame(rows)


@st.cache_data
def _compute_tb_prediction(tb_df: pd.DataFrame, lookback: int = 5) -> dict:
    values = tb_df["value"].tolist()
    tb_min = int(min(values))
    tb_max = int(max(values))
    tb_mean = sum(values) / len(values)
    last_val = int(values[-1])
    range_span = tb_max - tb_min
    range_pos = (last_val - tb_min) / range_span if range_span > 0 else 0.5
    range_signal = -1 if range_pos >= 0.65 else (1 if range_pos <= 0.35 else 0)
    reversion_signal = -1 if last_val > tb_mean else (1 if last_val < tb_mean else 0)
    recent = values[-lookback:] if len(values) >= lookback else values[:]
    trend_signal = 0
    if len(recent) >= 2:
        n = len(recent)
        x = list(range(n))
        xm = sum(x) / n
        ym = sum(recent) / n
        num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, recent))
        den = sum((xi - xm) ** 2 for xi in x)
        slope = num / den if den != 0 else 0.0
        trend_signal = 1 if slope > 0.3 else (-1 if slope < -0.3 else 0)
    combined = range_signal * 2 + reversion_signal * 2 + trend_signal
    prediction = "Higher" if combined > 0 else ("Lower" if combined < 0 else "Neutral")
    return {
        "last_value": last_val, "tb_min": tb_min, "tb_max": tb_max, "tb_mean": round(tb_mean, 2),
        "range_position_pct": round(range_pos * 100, 1), "range_signal": range_signal,
        "reversion_signal": reversion_signal, "trend_signal": trend_signal,
        "combined_score": combined, "prediction": prediction, "confidence": round(abs(combined) / 5.0, 2),
    }


@st.cache_data
def _build_tb_envelope(tb_df: pd.DataFrame) -> pd.DataFrame:
    tb = tb_df.copy().sort_values("draw_date").reset_index(drop=True)
    tb["running_min"] = tb["value"].cummin()
    tb["running_max"] = tb["value"].cummax()
    tb["overall_min"] = int(tb["value"].min())
    tb["overall_max"] = int(tb["value"].max())
    return tb


_tb_df_raw = _build_tb_frame(df)
_tb_df_raw["draw_date"] = pd.Categorical(
    _tb_df_raw["draw_date"],
    categories=sorted(_tb_df_raw["draw_date"].unique()),
    ordered=True,
)
tb_pred = _compute_tb_prediction(_tb_df_raw, lookback=lookback)

# ── Generate range tickets (main balls + Thunderball) ─────────────────────────

range_tickets, zone_info, tb_zone = _generate_range_tickets(
    pos_df, predictions_df, tb_pred, n_tickets=9, seed=42
)

st.subheader("9-ticket prediction from position ranges")
st.caption(
    "Each ticket's ball at sorted position p is drawn from the direction-biased zone: "
    "**Higher** \u2192 between last drawn value and historical max \u00b7 "
    "**Lower** \u2192 between historical min and last drawn value \u00b7 "
    "**Neutral** \u2192 full historical range. "
    "Strict ordering is enforced across positions. "
    "The Thunderball applies the same zone logic (range 1\u201314)."
)

# Zone reference table
zone_rows = []
for pos in range(1, 6):
    lo, hi, p_min, p_max, direction = zone_info[pos]
    pred_row = predictions_df[predictions_df["position"] == pos].iloc[0]
    zone_rows.append({
        "Position": f"Pos {pos}",
        "Direction": f"{_PRED_ARROW[direction]} {direction}",
        "Last drawn": int(pred_row["last_value"]),
        "Zone low": lo, "Zone high": hi,
        "All-time min": p_min, "All-time max": p_max,
    })
tb_lo_z, tb_hi_z, tb_dir_z = tb_zone
zone_rows.append({
    "Position": "Thunderball",
    "Direction": f"{_PRED_ARROW[tb_dir_z]} {tb_dir_z}",
    "Last drawn": tb_pred["last_value"],
    "Zone low": tb_lo_z, "Zone high": tb_hi_z,
    "All-time min": tb_pred["tb_min"], "All-time max": tb_pred["tb_max"],
})
st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

# Ticket grid — 3 columns × 3 rows
if range_tickets:
    ticket_cols = st.columns(3)
    for t_idx, ticket in enumerate(range_tickets):
        with ticket_cols[t_idx % 3]:
            main_part = " &nbsp;\u00b7&nbsp; ".join(f"<b>{b}</b>" for b in ticket[:5])
            tb_part = f"<b style='color:#e67e22'>TB {ticket[5]}</b>"
            st.markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:10px 14px;"
                f"margin-bottom:8px;background:#f8faff'>"
                f"<span style='color:#888;font-size:0.78rem'>Ticket {t_idx + 1}</span><br>"
                f"<span style='font-size:1.05rem'>{main_part} &nbsp;|&nbsp; {tb_part}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
else:
    st.warning("Could not generate 9 valid tickets from the current zone constraints. Try adjusting the lookback slider.")

# ── Back-test P&L ─────────────────────────────────────────────────────────────

st.divider()
st.header("Historical back-test P&L")
st.caption(
    "For every draw in history (after a warm-up period) the method is replayed using only data "
    "available *before* that draw. 9 tickets are generated and evaluated against the actual result "
    "using the standard Thunderball prize matrix. Cost = £9 per draw."
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


def _signals_from_values(
    values: list[int],
    lookback: int,
    val_min: int,
    val_max: int,
) -> str:
    """Return 'Higher' / 'Lower' / 'Neutral' for a value series."""
    if len(values) < 2:
        return "Neutral"
    last = values[-1]
    val_mean = sum(values) / len(values)
    range_span = val_max - val_min
    range_pos = (last - val_min) / range_span if range_span > 0 else 0.5
    rs = -1 if range_pos >= 0.65 else (1 if range_pos <= 0.35 else 0)
    rv = -1 if last > val_mean else (1 if last < val_mean else 0)
    recent = values[-lookback:] if len(values) >= lookback else values[:]
    ts = 0
    if len(recent) >= 2:
        n = len(recent)
        x = list(range(n))
        xm = sum(x) / n
        ym = sum(recent) / n
        num = sum((a - xm) * (b - ym) for a, b in zip(x, recent))
        den = sum((a - xm) ** 2 for a in x)
        slope = num / den if den else 0.0
        ts = 1 if slope > 0.3 else (-1 if slope < -0.3 else 0)
    c = rs * 2 + rv * 2 + ts
    return "Higher" if c > 0 else ("Lower" if c < 0 else "Neutral")


def _zone(direction: str, last: int, hist_min: int, hist_max: int, cap: int) -> tuple[int, int]:
    if direction == "Higher":
        lo, hi = last, hist_max
    elif direction == "Lower":
        lo, hi = hist_min, last
    else:
        lo, hi = hist_min, hist_max
    return max(1, lo), min(cap, max(hi, lo + 1))


def _make_tickets(
    pos_series: dict[int, list[int]],
    tb_series: list[int],
    lookback: int,
    n: int = 9,
    seed: int = 42,
) -> list[tuple[int, int, int, int, int, int]]:
    import random as _r
    rng = _r.Random(seed)
    zones: dict[int, tuple[int, int]] = {}
    for pos in range(1, 6):
        vals = pos_series[pos]
        hist_min, hist_max = min(vals), max(vals)
        d = _signals_from_values(vals, lookback, hist_min, hist_max)
        zones[pos] = _zone(d, vals[-1], hist_min, hist_max, 39)
    tb_min, tb_max = min(tb_series), max(tb_series)
    tb_dir = _signals_from_values(tb_series, lookback, tb_min, tb_max)
    tb_lo, tb_hi = _zone(tb_dir, tb_series[-1], tb_min, tb_max, 14)
    tickets: list[tuple[int, int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for _ in range(n * 500):
        if len(tickets) >= n:
            break
        raw = [rng.randint(zones[p][0], zones[p][1]) for p in range(1, 6)]
        for i in range(1, 5):
            if raw[i] <= raw[i - 1]:
                raw[i] = raw[i - 1] + 1
        if raw[4] > 39:
            raw[4] = 39
            for i in range(3, -1, -1):
                if raw[i] >= raw[i + 1]:
                    raw[i] = raw[i + 1] - 1
        tb_val = rng.randint(tb_lo, tb_hi)
        t = (raw[0], raw[1], raw[2], raw[3], raw[4], tb_val)
        if (
            t not in seen
            and raw[0] >= 1
            and raw[4] <= 39
            and len(set(raw)) == 5
            and all(raw[i] < raw[i + 1] for i in range(4))
        ):
            seen.add(t)
            tickets.append(t)
    return tickets


def _ticket_payout(ticket: tuple[int, ...], actual_main: set[int], actual_tb: int) -> int:
    main_hits = len(set(ticket[:5]) & actual_main)
    tb_hit = ticket[5] == actual_tb
    return PRIZE_MATRIX.get((main_hits, tb_hit), 0)


@st.cache_data
def _run_backtest(df: pd.DataFrame, lookback: int, warmup: int = 10) -> pd.DataFrame:
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for actual_idx in range(warmup, len(ordered)):
        training = ordered.iloc[:actual_idx]
        actual_row = ordered.iloc[actual_idx]
        actual_main = {int(actual_row[c]) for c in ["n1", "n2", "n3", "n4", "n5"]}
        actual_tb = int(actual_row["thunderball"])

        # Build per-position sorted-value series from training window
        pos_series: dict[int, list[int]] = {p: [] for p in range(1, 6)}
        for _, tr in training.iterrows():
            sorted_balls = sorted(int(tr[c]) for c in ["n1", "n2", "n3", "n4", "n5"])
            for pos, val in enumerate(sorted_balls, 1):
                pos_series[pos].append(val)
        tb_series = [int(r["thunderball"]) for _, r in training.iterrows()]

        tickets = _make_tickets(pos_series, tb_series, lookback=lookback, n=9, seed=42)
        if not tickets:
            continue

        total_payout = sum(_ticket_payout(t, actual_main, actual_tb) for t in tickets)
        cost = len(tickets) * TICKET_COST
        best_main = max(len(set(t[:5]) & actual_main) for t in tickets)
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
    bt_df = _run_backtest(df, lookback=lookback)

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
        .mark_line(color="#4c9be8", strokeWidth=2)
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
    target_like_draws = int((bt_df["Payout"] >= 10).sum()) if not bt_df.empty else 0
    if total_net > 0:
        st.success(
            "Over the tested history this method produced a positive overall net result, so it has been directionally useful "
            "in this sample. That still does not imply predictive power going forward."
        )
    elif total_net == 0:
        st.info(
            "Over the tested history this method roughly broke even, so the evidence here is neutral rather than strong."
        )
    else:
        st.warning(
            "Over the tested history this method lost money overall, so it is better treated as a descriptive analysis tool "
            "than a strong betting method."
        )
    st.caption(
        f"Average net per draw: £{average_net:.2f} | Profitable draws: {profitable_draws}/{draws_played} | "
        f"Draws returning at least £10: {target_like_draws}/{draws_played}"
    )

# ── One chart per position ────────────────────────────────────────────────────

POSITION_LABELS = {
    1: "Position 1 — Smallest ball",
    2: "Position 2",
    3: "Position 3 — Middle ball",
    4: "Position 4",
    5: "Position 5 — Largest ball",
}

BALL_COLOUR = "#4c9be8"
MIN_COLOUR = "#2ecc71"
MAX_COLOUR = "#e74c3c"
BAND_COLOUR = "#d6eaff"

total_draws = pos_df["draw_date"].nunique()

for pos in range(1, 6):
    st.subheader(POSITION_LABELS[pos])

    sub = envelope_df[envelope_df["position"] == pos].copy()
    overall_min = int(sub["overall_min"].iloc[0])
    overall_max = int(sub["overall_max"].iloc[0])

    pred_row = predictions_df[predictions_df["position"] == pos].iloc[0]
    pred_arrow = _PRED_ARROW[pred_row["prediction"]]
    pred_colour = _PRED_COLOUR[pred_row["prediction"]]
    conf_pct = round(float(pred_row["confidence"]) * 100)

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("All-time min", overall_min)
    col_m2.metric("All-time max", overall_max)
    col_m3.metric("Range", overall_max - overall_min)
    col_m4.metric("Last drawn", int(pred_row["last_value"]))
    col_m5.metric("Range position", f"{pred_row['range_position_pct']}%")

    st.markdown(
        f"**Next draw: :{pred_colour}[{pred_arrow} {pred_row['prediction']}]** "
        f"\u2014 confidence **{conf_pct}%**"
    )
    with st.expander("Signal breakdown"):
        _sig_map = {-1: "↓ Lower", 0: "→ Neutral", 1: "↑ Higher"}
        _sig_col = {-1: "red", 0: "grey", 1: "green"}
        _rs = int(pred_row["range_signal"])
        _rvs = int(pred_row["reversion_signal"])
        _ts = int(pred_row["trend_signal"])
        sc1, sc2, sc3 = st.columns(3)
        sc1.markdown(
            f"**Range position** (weight ×2)\n\n"
            f":{_sig_col[_rs]}[{_sig_map[_rs]}]\n\n"
            f"Last value at {pred_row['range_position_pct']}% of range"
        )
        sc2.markdown(
            f"**Mean reversion** (weight ×2)\n\n"
            f":{_sig_col[_rvs]}[{_sig_map[_rvs]}]\n\n"
            f"Last {pred_row['last_value']} vs mean {pred_row['overall_mean']}"
        )
        sc3.markdown(
            f"**Trend momentum** (weight ×1)\n\n"
            f":{_sig_col[_ts]}[{_sig_map[_ts]}]\n\n"
            f"Slope over last {lookback} draws"
        )

    # Altair base — shared x encoding
    base = alt.Chart(sub).encode(
        x=alt.X(
            "draw_date:O",
            title="Draw date",
            axis=alt.Axis(labels=False, ticks=False, title="Draw date (oldest → newest)"),
            sort=None,
        )
    )

    # Min/max band (area between running_min and running_max)
    band = base.mark_area(opacity=0.20, color=BAND_COLOUR).encode(
        y=alt.Y("running_min:Q", title="Ball value", scale=alt.Scale(domain=[0, 40])),
        y2=alt.Y2("running_max:Q"),
        tooltip=[
            alt.Tooltip("draw_date:O", title="Draw date"),
            alt.Tooltip("running_min:Q", title="Running min"),
            alt.Tooltip("running_max:Q", title="Running max"),
        ],
    )

    # Running min line
    min_line = base.mark_line(color=MIN_COLOUR, strokeWidth=1.5, strokeDash=[4, 2]).encode(
        y=alt.Y("running_min:Q"),
        tooltip=[
            alt.Tooltip("draw_date:O", title="Draw date"),
            alt.Tooltip("running_min:Q", title="Running min"),
        ],
    )

    # Running max line
    max_line = base.mark_line(color=MAX_COLOUR, strokeWidth=1.5, strokeDash=[4, 2]).encode(
        y=alt.Y("running_max:Q"),
        tooltip=[
            alt.Tooltip("draw_date:O", title="Draw date"),
            alt.Tooltip("running_max:Q", title="Running max"),
        ],
    )

    # Actual ball values as dots
    dots = base.mark_circle(size=30, color=BALL_COLOUR, opacity=0.7).encode(
        y=alt.Y("value:Q", title="Ball value"),
        tooltip=[
            alt.Tooltip("draw_date:O", title="Draw date"),
            alt.Tooltip("value:Q", title="Ball value"),
        ],
    )

    chart = (
        (band + min_line + max_line + dots)
        .properties(
            height=260,
            width="container",
            title=f"{POSITION_LABELS[pos]}  |  min={overall_min}, max={overall_max}  |  {total_draws} draws",
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )

    st.altair_chart(chart, use_container_width=True)

st.divider()
st.caption(
    "Blue dots = actual ball value drawn. "
    "Green dashed = running historical minimum. "
    "Red dashed = running historical maximum. "
    "Shaded band = running min\u2013max envelope."
)

# ── Thunderball (bonus ball) section ────────────────────────────────────────

st.header("Thunderball (bonus ball) range analysis")
st.caption("Thunderball values range from 1 to 14. The same range-position, mean-reversion and trend-momentum signals are applied to predict whether the next Thunderball will be higher or lower.")

# tb_pred already computed above; build the envelope here for the chart
tb_df = _tb_df_raw
tb_env = _build_tb_envelope(tb_df)

tb_overall_min = int(tb_env["overall_min"].iloc[0])
tb_overall_max = int(tb_env["overall_max"].iloc[0])
tb_arrow = _PRED_ARROW[tb_pred["prediction"]]
tb_colour = _PRED_COLOUR[tb_pred["prediction"]]
tb_conf_pct = round(float(tb_pred["confidence"]) * 100)

tb_c1, tb_c2, tb_c3, tb_c4, tb_c5 = st.columns(5)
tb_c1.metric("All-time min", tb_overall_min)
tb_c2.metric("All-time max", tb_overall_max)
tb_c3.metric("Range", tb_overall_max - tb_overall_min)
tb_c4.metric("Last drawn", tb_pred["last_value"])
tb_c5.metric("Range position", f"{tb_pred['range_position_pct']}%")

st.markdown(
    f"**Next draw: :{tb_colour}[{tb_arrow} {tb_pred['prediction']}]** "
    f"\u2014 confidence **{tb_conf_pct}%**"
)
with st.expander("Signal breakdown"):
    _sig_map = {-1: "\u2193 Lower", 0: "\u2192 Neutral", 1: "\u2191 Higher"}
    _sig_col = {-1: "red", 0: "grey", 1: "green"}
    _r = int(tb_pred["range_signal"])
    _v = int(tb_pred["reversion_signal"])
    _t = int(tb_pred["trend_signal"])
    tb_sc1, tb_sc2, tb_sc3 = st.columns(3)
    tb_sc1.markdown(
        f"**Range position** (weight \u00d72)\n\n"
        f":{_sig_col[_r]}[{_sig_map[_r]}]\n\n"
        f"Last value at {tb_pred['range_position_pct']}% of range"
    )
    tb_sc2.markdown(
        f"**Mean reversion** (weight \u00d72)\n\n"
        f":{_sig_col[_v]}[{_sig_map[_v]}]\n\n"
        f"Last {tb_pred['last_value']} vs mean {tb_pred['tb_mean']}"
    )
    tb_sc3.markdown(
        f"**Trend momentum** (weight \u00d71)\n\n"
        f":{_sig_col[_t]}[{_sig_map[_t]}]\n\n"
        f"Slope over last {lookback} draws"
    )

# Thunderball chart
tb_base = alt.Chart(tb_env).encode(
    x=alt.X(
        "draw_date:O",
        axis=alt.Axis(labels=False, ticks=False, title="Draw date (oldest \u2192 newest)"),
        sort=None,
    )
)
tb_band = tb_base.mark_area(opacity=0.20, color=BAND_COLOUR).encode(
    y=alt.Y("running_min:Q", title="Ball value", scale=alt.Scale(domain=[0, 15])),
    y2=alt.Y2("running_max:Q"),
    tooltip=[
        alt.Tooltip("draw_date:O", title="Draw date"),
        alt.Tooltip("running_min:Q", title="Running min"),
        alt.Tooltip("running_max:Q", title="Running max"),
    ],
)
tb_min_line = tb_base.mark_line(color=MIN_COLOUR, strokeWidth=1.5, strokeDash=[4, 2]).encode(
    y=alt.Y("running_min:Q"),
)
tb_max_line = tb_base.mark_line(color=MAX_COLOUR, strokeWidth=1.5, strokeDash=[4, 2]).encode(
    y=alt.Y("running_max:Q"),
)
tb_dots = tb_base.mark_circle(size=30, color="#f59f00", opacity=0.75).encode(
    y=alt.Y("value:Q", title="Thunderball value"),
    tooltip=[
        alt.Tooltip("draw_date:O", title="Draw date"),
        alt.Tooltip("value:Q", title="Thunderball"),
    ],
)
tb_chart = (
    (tb_band + tb_min_line + tb_max_line + tb_dots)
    .properties(
        height=260,
        width="container",
        title=f"Thunderball (1\u201314)  |  min={tb_overall_min}, max={tb_overall_max}  |  {tb_df['draw_date'].nunique()} draws",
    )
    .configure_view(strokeOpacity=0)
    .configure_axis(grid=True, gridOpacity=0.25)
)
st.altair_chart(tb_chart, use_container_width=True)
st.caption("Orange dots = Thunderball value drawn.")

st.divider()
st.header("Final Next 9-Ticket Prediction")
st.caption(
    "This is the current Draw Position Range portfolio using the active lookback and the direction-biased zones shown above."
)
final_ticket_rows = [
    {
        "Ticket": idx + 1,
        "Main Numbers": "-".join(f"{value:02d}" for value in ticket[:5]),
        "Thunderball": f"{ticket[5]:02d}",
    }
    for idx, ticket in enumerate(range_tickets)
]
if final_ticket_rows:
    st.dataframe(pd.DataFrame(final_ticket_rows), use_container_width=True, hide_index=True)
else:
    st.info("No position-range ticket set is currently available from the active settings.")
