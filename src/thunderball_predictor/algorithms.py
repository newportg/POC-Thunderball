from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from thunderball_predictor.data_models import (
    DrawOutcomeResult,
    PortfolioOptimizationResult,
    PredictionResult,
    RollingTimelineResult,
    TicketPrediction,
    TicketOutcomeResult,
)

MAIN_RANGE = np.arange(1, 40)
THUNDERBALL_RANGE = np.arange(1, 15)
TICKET_COST_GBP = 1
PRIZE_MATRIX = {
    (5, True): 500000,
    (5, False): 5000,
    (4, False): 250,
    (4, True): 100,
    (3, True): 20,
    (3, False): 10,
    (2, True): 10,
    (1, True): 5,
    (0, True): 3,
}


def _extract_frequency(df: pd.DataFrame) -> tuple[Counter[int], Counter[int]]:
    main_counter: Counter[int] = Counter()
    thunder_counter: Counter[int] = Counter()

    for col in ["n1", "n2", "n3", "n4", "n5"]:
        main_counter.update(df[col].tolist())
    thunder_counter.update(df["thunderball"].tolist())

    return main_counter, thunder_counter


def _weighted_pick_without_replacement(
    rng: np.random.Generator, universe: np.ndarray, weights: np.ndarray, k: int
) -> list[int]:
    selected: list[int] = []
    candidates = universe.copy()
    current_weights = weights.astype(float).copy()

    for _ in range(k):
        probs = current_weights / current_weights.sum()
        pick = int(rng.choice(candidates, p=probs))
        selected.append(pick)

        idx = int(np.where(candidates == pick)[0][0])
        candidates = np.delete(candidates, idx)
        current_weights = np.delete(current_weights, idx)

    return sorted(selected)


def _confidence_from_entropy(main_weights: np.ndarray, thunder_weights: np.ndarray) -> float:
    main_probs = main_weights / main_weights.sum()
    tb_probs = thunder_weights / thunder_weights.sum()

    main_entropy = -np.sum(main_probs * np.log(main_probs + 1e-12))
    tb_entropy = -np.sum(tb_probs * np.log(tb_probs + 1e-12))

    max_main_entropy = np.log(len(main_probs))
    max_tb_entropy = np.log(len(tb_probs))

    concentration = 1 - ((main_entropy / max_main_entropy) * 0.7 + (tb_entropy / max_tb_entropy) * 0.3)
    return float(np.clip(concentration, 0.01, 0.99))


def _normalize(weights: np.ndarray) -> np.ndarray:
    adjusted = weights.astype(float) + 1e-12
    return adjusted / adjusted.sum()


def _build_blended_weights(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    main_counter, thunder_counter = _extract_frequency(df)

    freq_main = np.array([main_counter.get(int(n), 0) + 1 for n in MAIN_RANGE], dtype=float)
    freq_tb = np.array([thunder_counter.get(int(n), 0) + 1 for n in THUNDERBALL_RANGE], dtype=float)

    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    recency_main = Counter({int(n): 1.0 for n in MAIN_RANGE})
    recency_tb = Counter({int(n): 1.0 for n in THUNDERBALL_RANGE})

    for idx, row in ordered.iterrows():
        weight = 0.985**idx
        for col in ["n1", "n2", "n3", "n4", "n5"]:
            recency_main[int(row[col])] += weight
        recency_tb[int(row["thunderball"])] += weight

    recency_main_weights = np.array([recency_main[int(n)] for n in MAIN_RANGE], dtype=float)
    recency_tb_weights = np.array([recency_tb[int(n)] for n in THUNDERBALL_RANGE], dtype=float)

    main_weights = 0.55 * _normalize(freq_main) + 0.45 * _normalize(recency_main_weights)
    thunder_weights = 0.50 * _normalize(freq_tb) + 0.50 * _normalize(recency_tb_weights)

    return _normalize(main_weights), _normalize(thunder_weights)


def _generate_diverse_ticket(
    rng: np.random.Generator,
    main_weights: np.ndarray,
    thunder_weights: np.ndarray,
    main_usage: Counter[int],
    thunder_usage: Counter[int],
) -> TicketPrediction:
    adjusted_main = main_weights.copy()
    adjusted_thunder = thunder_weights.copy()

    for idx, number in enumerate(MAIN_RANGE):
        adjusted_main[idx] = adjusted_main[idx] / (1.0 + 0.12 * main_usage.get(int(number), 0))

    for idx, number in enumerate(THUNDERBALL_RANGE):
        adjusted_thunder[idx] = adjusted_thunder[idx] / (1.0 + 0.75 * thunder_usage.get(int(number), 0))

    main_numbers = tuple(
        _weighted_pick_without_replacement(rng, MAIN_RANGE, _normalize(adjusted_main), 5)
    )
    thunderball = int(rng.choice(THUNDERBALL_RANGE, p=_normalize(adjusted_thunder)))
    return TicketPrediction(main_numbers=main_numbers, thunderball=thunderball)


def _sample_weighted_draws(
    rng: np.random.Generator, main_weights: np.ndarray, thunder_weights: np.ndarray, sample_size: int
) -> list[TicketPrediction]:
    draws: list[TicketPrediction] = []
    normalized_main = _normalize(main_weights)
    normalized_thunder = _normalize(thunder_weights)

    for _ in range(sample_size):
        main_numbers = tuple(
            _weighted_pick_without_replacement(rng, MAIN_RANGE, normalized_main, 5)
        )
        thunderball = int(rng.choice(THUNDERBALL_RANGE, p=normalized_thunder))
        draws.append(TicketPrediction(main_numbers=main_numbers, thunderball=thunderball))

    return draws


def _ticket_payout(ticket: TicketPrediction, draw: TicketPrediction) -> int:
    main_matches = len(set(ticket.main_numbers) & set(draw.main_numbers))
    thunderball_match = ticket.thunderball == draw.thunderball
    return PRIZE_MATRIX.get((main_matches, thunderball_match), 0)


def _count_matches(ticket: TicketPrediction, draw: TicketPrediction) -> tuple[int, bool]:
    main_matches = len(set(ticket.main_numbers) & set(draw.main_numbers))
    thunderball_match = ticket.thunderball == draw.thunderball
    return main_matches, thunderball_match


def _row_to_ticket(row: pd.Series) -> TicketPrediction:
    return TicketPrediction(
        main_numbers=(int(row["n1"]), int(row["n2"]), int(row["n3"]), int(row["n4"]), int(row["n5"])),
        thunderball=int(row["thunderball"]),
    )


def _build_candidate_pool(
    df: pd.DataFrame,
    main_weights: np.ndarray,
    thunder_weights: np.ndarray,
    rng: np.random.Generator,
    pool_size: int,
) -> list[TicketPrediction]:
    pool: list[TicketPrediction] = []
    seen: set[TicketPrediction] = set()
    main_usage: Counter[int] = Counter()
    thunder_usage: Counter[int] = Counter()

    seeded_predictors = [predict_frequency_weighted, predict_recency_weighted, predict_hot_cold_mix]
    for seed in range(1, 13):
        for predictor in seeded_predictors:
            result = predictor(df, seed=seed)
            ticket = TicketPrediction(result.main_numbers, result.thunderball)
            if ticket not in seen:
                pool.append(ticket)
                seen.add(ticket)
                main_usage.update(ticket.main_numbers)
                thunder_usage.update([ticket.thunderball])
            if len(pool) >= pool_size:
                return pool

    for _ in range(pool_size * 6):
        ticket = _generate_diverse_ticket(rng, main_weights, thunder_weights, main_usage, thunder_usage)
        if ticket in seen:
            continue

        pool.append(ticket)
        seen.add(ticket)
        main_usage.update(ticket.main_numbers)
        thunder_usage.update([ticket.thunderball])

        if len(pool) >= pool_size:
            break

    return pool


def _portfolio_coverage_score(tickets: list[TicketPrediction]) -> float:
    if not tickets:
        return 0.0

    unique_mains = {number for ticket in tickets for number in ticket.main_numbers}
    unique_thunderballs = {ticket.thunderball for ticket in tickets}
    main_score = len(unique_mains) / min(39, len(tickets) * 5)
    thunder_score = len(unique_thunderballs) / min(14, len(tickets))
    return 0.7 * main_score + 0.3 * thunder_score


def _portfolio_overlap_penalty(tickets: list[TicketPrediction]) -> float:
    if len(tickets) < 2:
        return 0.0

    main_overlap_total = 0.0
    thunder_overlap_total = 0.0
    pair_count = 0

    for left_idx in range(len(tickets)):
        for right_idx in range(left_idx + 1, len(tickets)):
            left_ticket = tickets[left_idx]
            right_ticket = tickets[right_idx]
            main_overlap_total += len(set(left_ticket.main_numbers) & set(right_ticket.main_numbers)) / 5.0
            thunder_overlap_total += 1.0 if left_ticket.thunderball == right_ticket.thunderball else 0.0
            pair_count += 1

    return 0.85 * (main_overlap_total / pair_count) + 0.15 * (thunder_overlap_total / pair_count)


def _portfolio_objective(
    tickets: list[TicketPrediction],
    payout_vector: np.ndarray,
    ticket_cost: int,
    target_payout: int,
    objective_mode: str,
) -> tuple[float, float, float, float]:
    expected_payout = float(np.mean(payout_vector))
    probability_target = float(np.mean(payout_vector >= target_payout))
    break_even_value = len(tickets) * ticket_cost
    probability_break_even = float(np.mean(payout_vector >= break_even_value))
    probability_profit = float(np.mean(payout_vector > break_even_value))
    full_loss_probability = float(np.mean(payout_vector == 0))
    downside_shortfall = float(np.mean(np.maximum(0, break_even_value - payout_vector))) / max(
        break_even_value, 1
    )
    coverage_score = _portfolio_coverage_score(tickets)
    overlap_penalty = _portfolio_overlap_penalty(tickets)

    if objective_mode == "downside_aware":
        objective = (
            probability_profit * 13.0
            + probability_break_even * 7.0
            + probability_target * 5.0
            + (expected_payout / max(target_payout, 1))
            + coverage_score * 0.60
            - overlap_penalty * 0.45
            - full_loss_probability * 8.0
            - downside_shortfall * 4.0
        )
    else:
        objective = (
            probability_target * 12.0
            + probability_break_even * 4.0
            + (expected_payout / max(target_payout, 1))
            + coverage_score * 0.75
            - overlap_penalty * 0.5
        )
    return objective, expected_payout, probability_target, probability_break_even


def optimize_ticket_portfolio(
    df: pd.DataFrame,
    ticket_count: int = 9,
    target_payout: int = 10,
    seed: int | None = None,
    simulation_draws: int = 2500,
    objective_mode: str = "balanced",
) -> PortfolioOptimizationResult:
    if objective_mode not in {"balanced", "downside_aware"}:
        raise ValueError("objective_mode must be 'balanced' or 'downside_aware'.")

    rng = np.random.default_rng(seed)
    main_weights, thunder_weights = _build_blended_weights(df)

    candidate_pool = _build_candidate_pool(
        df,
        main_weights,
        thunder_weights,
        rng,
        pool_size=max(72, ticket_count * 10),
    )
    sampled_draws = _sample_weighted_draws(rng, main_weights, thunder_weights, simulation_draws)

    payout_matrix = np.zeros((len(candidate_pool), len(sampled_draws)), dtype=int)
    for candidate_idx, ticket in enumerate(candidate_pool):
        payout_matrix[candidate_idx] = np.array(
            [_ticket_payout(ticket, draw) for draw in sampled_draws], dtype=int
        )

    selected_indices: list[int] = []
    current_payouts = np.zeros(len(sampled_draws), dtype=int)

    for _ in range(ticket_count):
        best_idx = -1
        best_objective = -float("inf")

        for candidate_idx in range(len(candidate_pool)):
            if candidate_idx in selected_indices:
                continue

            candidate_tickets = [candidate_pool[idx] for idx in [*selected_indices, candidate_idx]]
            candidate_payouts = current_payouts + payout_matrix[candidate_idx]
            objective, _, _, _ = _portfolio_objective(
                candidate_tickets,
                candidate_payouts,
                ticket_cost=TICKET_COST_GBP,
                target_payout=target_payout,
                objective_mode=objective_mode,
            )

            if objective > best_objective:
                best_objective = objective
                best_idx = candidate_idx

        selected_indices.append(best_idx)
        current_payouts = current_payouts + payout_matrix[best_idx]

    selected_tickets = [candidate_pool[idx] for idx in selected_indices]
    _, expected_payout, probability_target, probability_break_even = _portfolio_objective(
        selected_tickets,
        current_payouts,
        ticket_cost=TICKET_COST_GBP,
        target_payout=target_payout,
        objective_mode=objective_mode,
    )

    coverage_score = _portfolio_coverage_score(selected_tickets)
    if objective_mode == "downside_aware":
        note = (
            "Optimized with downside-aware scoring to reduce complete-loss scenarios while keeping target-payout "
            "coverage under a history-weighted simulation model."
        )
    else:
        note = (
            "Optimized for the probability that a 9-ticket portfolio returns at least the target payout "
            "under a history-weighted simulation model using the current Thunderball prize matrix."
        )

    return PortfolioOptimizationResult(
        tickets=tuple(selected_tickets),
        ticket_cost=TICKET_COST_GBP,
        target_payout=target_payout,
        estimated_expected_payout=expected_payout,
        estimated_probability_target=probability_target,
        estimated_probability_break_even=probability_break_even,
        coverage_score=coverage_score,
        note=note,
    )


def evaluate_rolling_timeline(
    df: pd.DataFrame,
    min_training_draws: int = 9,
    ticket_count: int = 9,
    target_payout: int = 10,
    seed: int = 42,
    objective_mode: str = "balanced",
    no_bet_threshold: float = 0.0,
) -> RollingTimelineResult:
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    if len(ordered) <= min_training_draws:
        raise ValueError(
            f"At least {min_training_draws + 1} draws are required for the timeline evaluation."
        )

    outcomes: list[DrawOutcomeResult] = []

    for actual_idx in range(min_training_draws, len(ordered)):
        training_window = ordered.iloc[:actual_idx].copy()
        actual_row = ordered.iloc[actual_idx]
        actual_draw = _row_to_ticket(actual_row)

        portfolio = optimize_ticket_portfolio(
            training_window,
            ticket_count=ticket_count,
            target_payout=target_payout,
            seed=seed,
            objective_mode=objective_mode,
        )

        payouts: list[int] = []
        main_match_counts: list[int] = []
        thunderball_hits = 0
        ticket_outcomes: list[TicketOutcomeResult] = []

        for ticket in portfolio.tickets:
            main_matches, thunderball_match = _count_matches(ticket, actual_draw)
            payout = _ticket_payout(ticket, actual_draw)
            payouts.append(payout)
            main_match_counts.append(main_matches)
            thunderball_hits += int(thunderball_match)
            ticket_outcomes.append(
                TicketOutcomeResult(
                    ticket=ticket,
                    main_match_count=main_matches,
                    thunderball_match=thunderball_match,
                    payout=payout,
                )
            )

        payout_if_played = sum(payouts)
        cost_if_played = len(portfolio.tickets) * TICKET_COST_GBP
        net_if_played = payout_if_played - cost_if_played
        edge_score = portfolio.estimated_probability_break_even
        played = edge_score >= no_bet_threshold

        total_payout = payout_if_played if played else 0
        total_cost = cost_if_played if played else 0
        winning_ticket_count = sum(1 for payout in payouts if payout > 0)

        outcomes.append(
            DrawOutcomeResult(
                draw_date=actual_row["draw_date"].date().isoformat(),
                actual_draw=actual_draw,
                training_start_date=training_window["draw_date"].min().date().isoformat(),
                training_end_date=training_window["draw_date"].max().date().isoformat(),
                total_payout=total_payout,
                total_cost=total_cost,
                net_result=total_payout - total_cost,
                best_main_match_count=max(main_match_counts, default=0),
                thunderball_hit_count=thunderball_hits,
                winning_ticket_count=winning_ticket_count,
                played=played,
                edge_score=edge_score,
                payout_if_played=payout_if_played,
                net_if_played=net_if_played,
                ticket_outcomes=tuple(ticket_outcomes),
            )
        )

    return RollingTimelineResult(min_training_draws=min_training_draws, outcomes=tuple(outcomes))


def predict_frequency_weighted(df: pd.DataFrame, seed: int | None = None) -> PredictionResult:
    rng = np.random.default_rng(seed)
    main_counter, thunder_counter = _extract_frequency(df)

    main_weights = np.array([main_counter.get(n, 0) + 1 for n in MAIN_RANGE], dtype=float)
    tb_weights = np.array([thunder_counter.get(n, 0) + 1 for n in THUNDERBALL_RANGE], dtype=float)

    main_numbers = tuple(_weighted_pick_without_replacement(rng, MAIN_RANGE, main_weights, 5))
    thunderball = int(rng.choice(THUNDERBALL_RANGE, p=tb_weights / tb_weights.sum()))

    return PredictionResult(
        algorithm="Frequency Weighted",
        main_numbers=main_numbers,
        thunderball=thunderball,
        confidence=_confidence_from_entropy(main_weights, tb_weights),
        note="Biases toward numbers that appear more often in historical draws.",
    )


def predict_recency_weighted(
    df: pd.DataFrame, decay: float = 0.97, seed: int | None = None
) -> PredictionResult:
    rng = np.random.default_rng(seed)

    main_scores = Counter({n: 1.0 for n in MAIN_RANGE})
    tb_scores = Counter({n: 1.0 for n in THUNDERBALL_RANGE})

    ordered = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
    for idx, row in ordered.iterrows():
        weight = decay**idx
        for col in ["n1", "n2", "n3", "n4", "n5"]:
            main_scores[int(row[col])] += weight
        tb_scores[int(row["thunderball"])] += weight

    main_weights = np.array([main_scores[n] for n in MAIN_RANGE], dtype=float)
    tb_weights = np.array([tb_scores[n] for n in THUNDERBALL_RANGE], dtype=float)

    main_numbers = tuple(_weighted_pick_without_replacement(rng, MAIN_RANGE, main_weights, 5))
    thunderball = int(rng.choice(THUNDERBALL_RANGE, p=tb_weights / tb_weights.sum()))

    return PredictionResult(
        algorithm="Recency Weighted",
        main_numbers=main_numbers,
        thunderball=thunderball,
        confidence=_confidence_from_entropy(main_weights, tb_weights),
        note="Gives more weight to recent draws using exponential decay.",
    )


def predict_hot_cold_mix(df: pd.DataFrame, seed: int | None = None) -> PredictionResult:
    rng = np.random.default_rng(seed)
    main_counter, thunder_counter = _extract_frequency(df)

    ranked_main = sorted(MAIN_RANGE, key=lambda n: main_counter.get(int(n), 0), reverse=True)
    hot = ranked_main[:20]
    cold = ranked_main[-19:]

    chosen_hot = rng.choice(np.array(hot), size=3, replace=False)
    chosen_cold = rng.choice(np.array(cold), size=2, replace=False)
    main_numbers = tuple(sorted([int(n) for n in np.concatenate([chosen_hot, chosen_cold])]))

    ranked_tb = sorted(THUNDERBALL_RANGE, key=lambda n: thunder_counter.get(int(n), 0), reverse=True)
    thunderball = int(rng.choice(np.array(ranked_tb[:7])))

    main_weights = np.array([main_counter.get(int(n), 0) + 1 for n in MAIN_RANGE], dtype=float)
    tb_weights = np.array([thunder_counter.get(int(n), 0) + 1 for n in THUNDERBALL_RANGE], dtype=float)

    return PredictionResult(
        algorithm="Hot/Cold Mix",
        main_numbers=main_numbers,
        thunderball=thunderball,
        confidence=_confidence_from_entropy(main_weights, tb_weights),
        note="Combines frequently drawn numbers with underrepresented numbers.",
    )


def _build_markov_transition_matrices(
    df: pd.DataFrame, smoothing: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    main_transition = np.full((len(MAIN_RANGE), len(MAIN_RANGE)), smoothing, dtype=float)
    thunder_transition = np.full(
        (len(THUNDERBALL_RANGE), len(THUNDERBALL_RANGE)), smoothing, dtype=float
    )

    for idx in range(1, len(ordered)):
        prev_main = [int(ordered.loc[idx - 1, col]) for col in ["n1", "n2", "n3", "n4", "n5"]]
        next_main = [int(ordered.loc[idx, col]) for col in ["n1", "n2", "n3", "n4", "n5"]]

        for prev_number in prev_main:
            for next_number in next_main:
                main_transition[prev_number - 1, next_number - 1] += 1.0

        prev_tb = int(ordered.loc[idx - 1, "thunderball"])
        next_tb = int(ordered.loc[idx, "thunderball"])
        thunder_transition[prev_tb - 1, next_tb - 1] += 1.0

    return main_transition, thunder_transition


def predict_markov_chain(df: pd.DataFrame, seed: int | None = None) -> PredictionResult:
    if len(df) < 2:
        fallback = predict_frequency_weighted(df, seed=seed)
        return PredictionResult(
            algorithm="Markov Chain",
            main_numbers=fallback.main_numbers,
            thunderball=fallback.thunderball,
            confidence=max(0.05, fallback.confidence * 0.8),
            note="Insufficient sequential history for transitions; fell back to frequency weighting.",
        )

    rng = np.random.default_rng(seed)
    ordered = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    latest_row = ordered.iloc[-1]

    main_transition, thunder_transition = _build_markov_transition_matrices(df, smoothing=1.0)
    main_counter, thunder_counter = _extract_frequency(df)

    latest_main = [int(latest_row[col]) for col in ["n1", "n2", "n3", "n4", "n5"]]
    main_markov_scores = np.zeros(len(MAIN_RANGE), dtype=float)
    for prev_number in latest_main:
        main_markov_scores += main_transition[prev_number - 1]

    latest_tb = int(latest_row["thunderball"])
    thunder_markov_scores = thunder_transition[latest_tb - 1]

    main_freq_scores = np.array([main_counter.get(int(n), 0) + 1 for n in MAIN_RANGE], dtype=float)
    thunder_freq_scores = np.array(
        [thunder_counter.get(int(n), 0) + 1 for n in THUNDERBALL_RANGE], dtype=float
    )

    main_weights = 0.75 * _normalize(main_markov_scores) + 0.25 * _normalize(main_freq_scores)
    thunder_weights = 0.75 * _normalize(thunder_markov_scores) + 0.25 * _normalize(thunder_freq_scores)

    main_numbers = tuple(_weighted_pick_without_replacement(rng, MAIN_RANGE, main_weights, 5))
    thunderball = int(rng.choice(THUNDERBALL_RANGE, p=thunder_weights))

    return PredictionResult(
        algorithm="Markov Chain",
        main_numbers=main_numbers,
        thunderball=thunderball,
        confidence=_confidence_from_entropy(main_weights, thunder_weights),
        note=(
            "Uses transition probabilities between consecutive draws and anchors on the most recent draw, "
            "with a small frequency blend for stability."
        ),
    )


def available_algorithms() -> dict[str, Callable[..., PredictionResult]]:
    return {
        "Frequency Weighted": predict_frequency_weighted,
        "Recency Weighted": predict_recency_weighted,
        "Hot/Cold Mix": predict_hot_cold_mix,
        "Markov Chain": predict_markov_chain,
    }
