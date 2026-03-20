from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thunderball_predictor.algorithms import optimize_ticket_portfolio
from thunderball_predictor.data_models import TicketPrediction

CURRENT_OPTIMIZER_METHOD = "current_optimizer"
DELTA_SYSTEM_METHOD = "delta_system"
POSITION_RANGE_METHOD = "position_range"
RANDOM_METHOD = "random"
MAIN_SUM_METHOD = "main_sum"

METHOD_LABELS = {
    CURRENT_OPTIMIZER_METHOD: "Current Optimizer",
    DELTA_SYSTEM_METHOD: "Delta System",
    POSITION_RANGE_METHOD: "Draw Position Range",
    RANDOM_METHOD: "Random Ticket Selection",
    MAIN_SUM_METHOD: "Main Ball Sum Filter",
}

AUTOMATION_CONFIG_DEFAULTS = {
    "method": CURRENT_OPTIMIZER_METHOD,
    "ticket_count": 9,
    "seed": 42,
    "optimizer_objective_mode": "downside_aware",
    "delta_top_signatures": 12,
    "range_lookback_draws": 8,
    "sum_sigma_multiplier": 1.5,
}


@dataclass(frozen=True)
class MethodPrediction:
    method_id: str
    method_label: str
    tickets: tuple[TicketPrediction, ...]
    note: str
    objective_mode: str | None = None
    estimated_expected_payout: float | None = None
    estimated_probability_target: float | None = None
    estimated_probability_break_even: float | None = None
    coverage_score: float | None = None


def normalize_automation_config(config: dict[str, object] | None) -> dict[str, object]:
    normalized = dict(AUTOMATION_CONFIG_DEFAULTS)
    if config:
        normalized.update(config)

    method = str(normalized.get("method", CURRENT_OPTIMIZER_METHOD))
    if method not in METHOD_LABELS:
        method = CURRENT_OPTIMIZER_METHOD
    normalized["method"] = method

    objective_mode = str(normalized.get("optimizer_objective_mode", "downside_aware"))
    if objective_mode not in {"downside_aware", "balanced", "main_hit_focused"}:
        objective_mode = "downside_aware"
    normalized["optimizer_objective_mode"] = objective_mode

    normalized["ticket_count"] = max(1, min(20, int(normalized.get("ticket_count", 9))))
    normalized["seed"] = max(0, int(normalized.get("seed", 42)))
    normalized["delta_top_signatures"] = max(
        1,
        min(30, int(normalized.get("delta_top_signatures", AUTOMATION_CONFIG_DEFAULTS["delta_top_signatures"]))),
    )
    normalized["range_lookback_draws"] = max(
        3,
        min(20, int(normalized.get("range_lookback_draws", AUTOMATION_CONFIG_DEFAULTS["range_lookback_draws"]))),
    )
    normalized["sum_sigma_multiplier"] = max(
        0.5,
        min(3.0, float(normalized.get("sum_sigma_multiplier", AUTOMATION_CONFIG_DEFAULTS["sum_sigma_multiplier"]))),
    )
    return normalized


def _compute_delta_signature(main_numbers: list[int] | tuple[int, ...]) -> tuple[int, int, int, int, int]:
    ordered = sorted(int(number) for number in main_numbers)
    deltas = [ordered[idx + 1] - ordered[idx] for idx in range(4)]
    deltas.append(39 - ordered[-1] + ordered[0])
    return tuple(deltas)


def _numbers_from_start_and_signature(
    start: int,
    signature: tuple[int, int, int, int, int],
) -> tuple[int, ...] | None:
    current = int(start)
    numbers = [current]
    for delta in signature[:-1]:
        current = ((current - 1 + int(delta)) % 39) + 1
        numbers.append(current)

    if len(set(numbers)) != 5:
        return None

    return tuple(sorted(numbers))


def _build_signature_scores(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    score_by_signature: dict[tuple[int, int, int, int, int], float] = {}
    count_by_signature: dict[tuple[int, int, int, int, int], int] = {}

    for idx, (_, row) in enumerate(ordered.iterrows()):
        signature = _compute_delta_signature([int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]])
        weight = 0.985**idx
        score_by_signature[signature] = score_by_signature.get(signature, 0.0) + weight
        count_by_signature[signature] = count_by_signature.get(signature, 0) + 1

    rows: list[dict[str, object]] = []
    for signature, score in score_by_signature.items():
        rows.append(
            {
                "SignatureTuple": signature,
                "Weighted Score": score,
                "Occurrences": count_by_signature[signature],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["SignatureTuple", "Weighted Score", "Occurrences"])

    return pd.DataFrame(rows).sort_values(
        ["Weighted Score", "Occurrences"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _build_thunderball_weights(df: pd.DataFrame) -> np.ndarray:
    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    weights = np.ones(14, dtype=float)

    for idx, (_, row) in enumerate(ordered.iterrows()):
        thunderball = int(row["thunderball"])
        weights[thunderball - 1] += 0.985**idx

    return weights / weights.sum()


def _generate_delta_tickets(
    df: pd.DataFrame,
    ticket_count: int,
    top_signatures: int,
    seed: int,
) -> tuple[TicketPrediction, ...]:
    signature_df = _build_signature_scores(df)
    if signature_df.empty:
        return tuple()

    usable = signature_df.head(max(1, top_signatures)).copy()
    weighted_scores = usable["Weighted Score"].to_numpy(dtype=float)
    if weighted_scores.sum() <= 0:
        weighted_scores = np.ones(len(usable), dtype=float)
    weighted_scores = weighted_scores / weighted_scores.sum()

    thunderball_weights = _build_thunderball_weights(df)
    rng = np.random.default_rng(seed)
    tickets: list[TicketPrediction] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    for _ in range(ticket_count * 300):
        if len(tickets) >= ticket_count:
            break

        row_index = int(rng.choice(np.arange(len(usable)), p=weighted_scores))
        signature = tuple(int(value) for value in usable.iloc[row_index]["SignatureTuple"])
        start = int(rng.integers(1, 40))
        main_numbers = _numbers_from_start_and_signature(start, signature)
        if main_numbers is None:
            continue

        thunderball = int(rng.choice(np.arange(1, 15), p=thunderball_weights))
        seen_key = (main_numbers, thunderball)
        if seen_key in seen:
            continue

        tickets.append(TicketPrediction(main_numbers=main_numbers, thunderball=thunderball))
        seen.add(seen_key)

    return tuple(tickets)


def _build_position_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        sorted_balls = sorted(int(row[col]) for col in ["n1", "n2", "n3", "n4", "n5"])
        for pos, ball in enumerate(sorted_balls, start=1):
            rows.append({"position": pos, "value": ball})
    return pd.DataFrame(rows)


def _compute_position_predictions(pos_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    for pos in range(1, 6):
        sub = pos_df[pos_df["position"] == pos].reset_index(drop=True)
        values = sub["value"].tolist()

        overall_min = int(min(values))
        overall_max = int(max(values))
        overall_mean = sum(values) / len(values)
        last_value = int(values[-1])
        range_span = overall_max - overall_min
        range_pos = (last_value - overall_min) / range_span if range_span > 0 else 0.5

        range_signal = -1 if range_pos >= 0.65 else (1 if range_pos <= 0.35 else 0)
        reversion_signal = -1 if last_value > overall_mean else (1 if last_value < overall_mean else 0)

        recent = values[-lookback:] if len(values) >= lookback else values[:]
        trend_signal = 0
        if len(recent) >= 2:
            count = len(recent)
            x_values = list(range(count))
            x_mean = sum(x_values) / count
            y_mean = sum(recent) / count
            numerator = sum((x_val - x_mean) * (y_val - y_mean) for x_val, y_val in zip(x_values, recent))
            denominator = sum((x_val - x_mean) ** 2 for x_val in x_values)
            slope = numerator / denominator if denominator != 0 else 0.0
            trend_signal = 1 if slope > 0.3 else (-1 if slope < -0.3 else 0)

        combined = range_signal * 2 + reversion_signal * 2 + trend_signal
        prediction = "Higher" if combined > 0 else ("Lower" if combined < 0 else "Neutral")
        results.append(
            {
                "position": pos,
                "last_value": last_value,
                "overall_min": overall_min,
                "overall_max": overall_max,
                "prediction": prediction,
            }
        )

    return pd.DataFrame(results)


def _build_tb_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"value": [int(value) for value in df.sort_values("draw_date", ascending=True)["thunderball"]]})


def _compute_tb_prediction(tb_df: pd.DataFrame, lookback: int) -> dict[str, object]:
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
        count = len(recent)
        x_values = list(range(count))
        x_mean = sum(x_values) / count
        y_mean = sum(recent) / count
        numerator = sum((x_val - x_mean) * (y_val - y_mean) for x_val, y_val in zip(x_values, recent))
        denominator = sum((x_val - x_mean) ** 2 for x_val in x_values)
        slope = numerator / denominator if denominator != 0 else 0.0
        trend_signal = 1 if slope > 0.3 else (-1 if slope < -0.3 else 0)

    combined = range_signal * 2 + reversion_signal * 2 + trend_signal
    prediction = "Higher" if combined > 0 else ("Lower" if combined < 0 else "Neutral")
    return {
        "last_value": last_val,
        "tb_min": tb_min,
        "tb_max": tb_max,
        "prediction": prediction,
    }


def _generate_random_tickets(
    ticket_count: int,
    seed: int,
) -> tuple[TicketPrediction, ...]:
    """Generate completely random valid Thunderball tickets."""
    rng = np.random.default_rng(seed)
    tickets: list[TicketPrediction] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(ticket_count * 500):
        if len(tickets) >= ticket_count:
            break
        # Generate 5 unique random main balls from 1-39, sorted
        main_balls = tuple(sorted(rng.choice(np.arange(1, 40), size=5, replace=False)))
        # Generate random thunderball from 1-14
        thunderball = int(rng.integers(1, 15))
        seen_key = (*main_balls, thunderball)
        if seen_key not in seen:
            seen.add(seen_key)
            tickets.append(TicketPrediction(main_numbers=main_balls, thunderball=thunderball))

    return tuple(tickets)


def _generate_sum_filtered_tickets(
    df: pd.DataFrame,
    ticket_count: int,
    sigma_multiplier: float,
    seed: int,
) -> tuple[TicketPrediction, ...]:
    """Generate tickets whose main-ball sum falls within mean ± sigma_multiplier * std."""
    sums = df[["n1", "n2", "n3", "n4", "n5"]].astype(int).sum(axis=1)
    mean = float(sums.mean())
    std = float(sums.std())
    if std == 0:
        return tuple()
    low = max(15, int(round(mean - sigma_multiplier * std)))
    high = min(185, int(round(mean + sigma_multiplier * std)))

    rng = np.random.default_rng(seed)
    tickets: list[TicketPrediction] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for _ in range(ticket_count * 5000):
        if len(tickets) >= ticket_count:
            break
        main_balls = tuple(sorted(int(v) for v in rng.choice(np.arange(1, 40), size=5, replace=False)))
        if not (low <= sum(main_balls) <= high):
            continue
        thunderball = int(rng.integers(1, 15))
        seen_key = (*main_balls, thunderball)
        if seen_key not in seen:
            seen.add(seen_key)
            tickets.append(TicketPrediction(main_numbers=main_balls, thunderball=thunderball))

    return tuple(tickets)


def _generate_range_tickets(
    df: pd.DataFrame,
    ticket_count: int,
    lookback: int,
    seed: int,
) -> tuple[TicketPrediction, ...]:
    rng = np.random.default_rng(seed)
    pos_df = _build_position_frame(df.sort_values("draw_date", ascending=True).reset_index(drop=True))
    predictions_df = _compute_position_predictions(pos_df, lookback)
    tb_pred = _compute_tb_prediction(_build_tb_frame(df), lookback)

    zones: dict[int, tuple[int, int]] = {}
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
        zones[pos] = (max(1, lo), min(39, max(hi, lo + 1)))

    if str(tb_pred["prediction"]) == "Higher":
        tb_lo, tb_hi = int(tb_pred["last_value"]), int(tb_pred["tb_max"])
    elif str(tb_pred["prediction"]) == "Lower":
        tb_lo, tb_hi = int(tb_pred["tb_min"]), int(tb_pred["last_value"])
    else:
        tb_lo, tb_hi = int(tb_pred["tb_min"]), int(tb_pred["tb_max"])
    tb_lo = max(1, tb_lo)
    tb_hi = min(14, max(tb_hi, tb_lo + 1))

    tickets: list[TicketPrediction] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for _ in range(ticket_count * 500):
        if len(tickets) >= ticket_count:
            break
        raw = [int(rng.integers(zones[pos][0], zones[pos][1] + 1)) for pos in range(1, 6)]
        for idx in range(1, 5):
            if raw[idx] <= raw[idx - 1]:
                raw[idx] = raw[idx - 1] + 1
        if raw[4] > 39:
            raw[4] = 39
            for idx in range(3, -1, -1):
                if raw[idx] >= raw[idx + 1]:
                    raw[idx] = raw[idx + 1] - 1

        if raw[0] < 1 or raw[4] > 39 or len(set(raw)) != 5 or not all(raw[idx] < raw[idx + 1] for idx in range(4)):
            continue

        thunderball = int(rng.integers(tb_lo, tb_hi + 1))
        main_numbers = tuple(int(value) for value in raw)
        seen_key = (main_numbers, thunderball)
        if seen_key in seen:
            continue

        tickets.append(TicketPrediction(main_numbers=main_numbers, thunderball=thunderball))
        seen.add(seen_key)

    return tuple(tickets)


def generate_method_prediction(
    df: pd.DataFrame,
    config: dict[str, object] | None = None,
    target_payout: int = 10,
    simulation_draws: int = 2500,
) -> MethodPrediction:
    normalized = normalize_automation_config(config)
    method = str(normalized["method"])
    ticket_count = int(normalized["ticket_count"])
    seed = int(normalized["seed"])

    if method == CURRENT_OPTIMIZER_METHOD:
        objective_mode = str(normalized["optimizer_objective_mode"])
        portfolio = optimize_ticket_portfolio(
            df,
            ticket_count=ticket_count,
            target_payout=target_payout,
            seed=seed,
            simulation_draws=simulation_draws,
            objective_mode=objective_mode,
        )
        return MethodPrediction(
            method_id=method,
            method_label=METHOD_LABELS[method],
            tickets=tuple(portfolio.tickets),
            note=portfolio.note,
            objective_mode=objective_mode,
            estimated_expected_payout=portfolio.estimated_expected_payout,
            estimated_probability_target=portfolio.estimated_probability_target,
            estimated_probability_break_even=portfolio.estimated_probability_break_even,
            coverage_score=portfolio.coverage_score,
        )

    if method == DELTA_SYSTEM_METHOD:
        tickets = _generate_delta_tickets(
            df,
            ticket_count=ticket_count,
            top_signatures=int(normalized["delta_top_signatures"]),
            seed=seed,
        )
        return MethodPrediction(
            method_id=method,
            method_label=METHOD_LABELS[method],
            tickets=tickets,
            note=(
                "Generated from recent weighted delta signatures, where each signature captures the gaps between "
                "sorted main balls in historical draws."
            ),
        )

    if method == RANDOM_METHOD:
        tickets = _generate_random_tickets(
            ticket_count=ticket_count,
            seed=seed,
        )
        return MethodPrediction(
            method_id=method,
            method_label=METHOD_LABELS[method],
            tickets=tickets,
            note=(
                "Generated by uniformly random selection of main-ball combinations and Thunderballs. "
                "No historical data analysis is performed. This serves as a null hypothesis baseline."
            ),
        )

    if method == MAIN_SUM_METHOD:
        tickets = _generate_sum_filtered_tickets(
            df,
            ticket_count=ticket_count,
            sigma_multiplier=float(normalized["sum_sigma_multiplier"]),
            seed=seed,
        )
        return MethodPrediction(
            method_id=method,
            method_label=METHOD_LABELS[method],
            tickets=tickets,
            note=(
                "Generated by constraining main-ball sums to within the bell-curve range of historical sums "
                f"(±{float(normalized['sum_sigma_multiplier'])}σ of the historical mean)."
            ),
        )

    tickets = _generate_range_tickets(
        df,
        ticket_count=ticket_count,
        lookback=int(normalized["range_lookback_draws"]),
        seed=seed,
    )
    return MethodPrediction(
        method_id=method,
        method_label=METHOD_LABELS[method],
        tickets=tickets,
        note=(
            "Generated from draw-position range signals using range position, mean reversion, and momentum to bias "
            "each sorted ball slot and the Thunderball."
        ),
    )
