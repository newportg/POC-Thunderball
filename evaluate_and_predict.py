#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thunderball_predictor.algorithms import (
    PRIZE_MATRIX,
    TICKET_COST_GBP,
)
from thunderball_predictor.loader import load_draw_history
from thunderball_predictor.methods import (
    AUTOMATION_CONFIG_DEFAULTS,
    CURRENT_OPTIMIZER_METHOD,
    generate_method_prediction,
    normalize_automation_config,
)

DATA_FILE = Path("data/thunderball-draw-history.csv")
REPORTS_DIR = Path("reports")
STATE_FILE = REPORTS_DIR / "current_prediction.json"
REPORT_FILE = REPORTS_DIR / "latest_prediction_report.txt"
SUBJECT_FILE = REPORTS_DIR / "latest_email_subject.txt"
AUTOMATION_CONFIG_FILE = REPORTS_DIR / "future_prediction_config.json"

DEFAULT_TICKET_COUNT = 9
DEFAULT_TARGET_PAYOUT = 10
DEFAULT_OBJECTIVE_MODE = "downside_aware"
DEFAULT_SEED = 42
DEFAULT_SIMULATION_DRAWS = 2500


def _load_official_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["DrawNumber"] = pd.to_numeric(df["DrawNumber"], errors="coerce").astype("Int64")
    df["DrawDate"] = pd.to_datetime(df["DrawDate"], errors="coerce", dayfirst=True)
    valid = df.dropna(subset=["DrawNumber", "DrawDate"]).copy()
    return valid.sort_values(["DrawDate", "DrawNumber"], ascending=[False, False]).reset_index(drop=True)


def _read_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None

    with STATE_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_automation_config() -> dict[str, Any]:
    if not AUTOMATION_CONFIG_FILE.exists():
        return normalize_automation_config(dict(AUTOMATION_CONFIG_DEFAULTS))

    with AUTOMATION_CONFIG_FILE.open("r", encoding="utf-8") as handle:
        return normalize_automation_config(json.load(handle))


def _write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing != content:
        path.write_text(content, encoding="utf-8")


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text_if_changed(path, serialized)


def _format_ticket(main_numbers: list[int] | tuple[int, ...], thunderball: int) -> str:
    main_display = "-".join(f"{number:02d}" for number in main_numbers)
    return f"{main_display} | TB {thunderball:02d}"


def _count_matches(ticket: dict[str, Any], actual_main: set[int], actual_thunderball: int) -> tuple[int, bool, int]:
    main_matches = len(set(ticket["main_numbers"]) & actual_main)
    thunderball_match = int(ticket["thunderball"]) == actual_thunderball
    payout = PRIZE_MATRIX.get((main_matches, thunderball_match), 0)
    return main_matches, thunderball_match, payout


def _evaluate_prediction(
    state: dict[str, Any] | None,
    latest_draw_number: int,
    latest_draw_row: pd.Series,
) -> dict[str, Any] | None:
    if state is None:
        return None

    if int(state.get("target_draw_number", -1)) != latest_draw_number:
        return None

    actual_main = {int(latest_draw_row[column]) for column in ["Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5"]}
    actual_thunderball = int(latest_draw_row["Thunderball"])

    ticket_results: list[dict[str, Any]] = []
    total_payout = 0

    for index, ticket in enumerate(state.get("tickets", []), start=1):
        main_matches, thunderball_match, payout = _count_matches(ticket, actual_main, actual_thunderball)
        total_payout += payout
        ticket_results.append(
            {
                "index": index,
                "main_numbers": ticket["main_numbers"],
                "thunderball": int(ticket["thunderball"]),
                "main_matches": main_matches,
                "thunderball_match": thunderball_match,
                "payout": payout,
            }
        )

    total_cost = len(ticket_results) * TICKET_COST_GBP
    return {
        "predicted_for_draw_number": latest_draw_number,
        "generated_at": state.get("generated_at"),
        "objective_mode": state.get("objective_mode"),
        "source_latest_draw_number": state.get("source_latest_draw_number"),
        "ticket_results": ticket_results,
        "total_payout": total_payout,
        "total_cost": total_cost,
        "net_result": total_payout - total_cost,
        "winning_ticket_count": sum(1 for item in ticket_results if item["payout"] > 0),
        "best_main_match_count": max((item["main_matches"] for item in ticket_results), default=0),
        "thunderball_hit_count": sum(1 for item in ticket_results if item["thunderball_match"]),
    }


def _generate_next_prediction(latest_draw_number: int) -> dict[str, Any]:
    history = load_draw_history(DATA_FILE)
    automation_config = _read_automation_config()
    method_prediction = generate_method_prediction(
        history,
        config=automation_config,
        target_payout=DEFAULT_TARGET_PAYOUT,
        simulation_draws=DEFAULT_SIMULATION_DRAWS,
    )

    objective_mode = method_prediction.objective_mode
    if objective_mode is None and str(automation_config.get("method")) == CURRENT_OPTIMIZER_METHOD:
        objective_mode = DEFAULT_OBJECTIVE_MODE

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_latest_draw_number": latest_draw_number,
        "target_draw_number": latest_draw_number + 1,
        "prediction_method": method_prediction.method_id,
        "prediction_method_label": method_prediction.method_label,
        "objective_mode": objective_mode,
        "ticket_count": len(method_prediction.tickets),
        "target_payout": DEFAULT_TARGET_PAYOUT,
        "estimated_expected_payout": round(method_prediction.estimated_expected_payout, 2)
        if method_prediction.estimated_expected_payout is not None
        else None,
        "estimated_probability_target": round(method_prediction.estimated_probability_target, 4)
        if method_prediction.estimated_probability_target is not None
        else None,
        "estimated_probability_break_even": round(method_prediction.estimated_probability_break_even, 4)
        if method_prediction.estimated_probability_break_even is not None
        else None,
        "coverage_score": round(method_prediction.coverage_score, 4)
        if method_prediction.coverage_score is not None
        else None,
        "automation_config": automation_config,
        "note": method_prediction.note,
        "tickets": [
            {
                "main_numbers": list(ticket.main_numbers),
                "thunderball": ticket.thunderball,
            }
            for ticket in method_prediction.tickets
        ],
    }


def _should_refresh_prediction_state(
    previous_state: dict[str, Any] | None,
    latest_draw_number: int,
) -> bool:
    if previous_state is None:
        return True

    if int(previous_state.get("source_latest_draw_number", -1)) != latest_draw_number:
        return True

    expected_config = _read_automation_config()
    previous_config = normalize_automation_config(previous_state.get("automation_config"))
    if previous_config != expected_config:
        return True

    expected_method = str(expected_config.get("method", CURRENT_OPTIMIZER_METHOD))
    if str(previous_state.get("prediction_method", "")) != expected_method:
        return True

    required_fields = {
        "prediction_method_label",
        "automation_config",
        "estimated_expected_payout",
        "estimated_probability_target",
        "estimated_probability_break_even",
        "coverage_score",
    }
    return any(field not in previous_state for field in required_fields)


def _build_report(
    latest_draw_row: pd.Series,
    evaluation: dict[str, Any] | None,
    next_prediction: dict[str, Any],
    previous_state: dict[str, Any] | None,
) -> tuple[str, str]:
    def _format_optional_percent(value: Any) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.2%}"

    def _format_optional_currency(value: Any) -> str:
        if value is None:
            return "N/A"
        return f"GBP{value}"

    def _format_optional_score(value: Any) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.4f}"

    latest_draw_number = int(latest_draw_row["DrawNumber"])
    actual_main = [int(latest_draw_row[column]) for column in ["Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5"]]
    actual_thunderball = int(latest_draw_row["Thunderball"])
    actual_display = _format_ticket(actual_main, actual_thunderball)

    subject = (
        f"Thunderball draw {latest_draw_number} result and next prediction {next_prediction['target_draw_number']}"
    )

    lines = [
        "Thunderball Automated Update",
        "",
        f"Latest draw: #{latest_draw_number}",
        f"Draw date: {pd.Timestamp(latest_draw_row['DrawDate']).date().isoformat()}",
        f"Draw result: {actual_display}",
        f"Machine: {latest_draw_row.get('Machine', '')}",
        f"Ball set: {latest_draw_row.get('Ball Set', '')}",
        "",
        "Evaluation against previous prediction",
        "--------------------------------------",
    ]

    if evaluation is None:
        if previous_state is None:
            lines.append("No stored prior prediction was available to evaluate.")
        else:
            lines.append(
                "Stored prediction target did not match the latest available draw, so no direct evaluation was produced."
            )
            lines.append(
                f"Stored target draw: #{int(previous_state.get('target_draw_number', 0))}"
            )
    else:
        lines.extend(
            [
                f"Prediction generated at: {evaluation['generated_at']}",
                f"Winning tickets: {evaluation['winning_ticket_count']} of {len(evaluation['ticket_results'])}",
                f"Best main match count: {evaluation['best_main_match_count']}",
                f"Thunderball hits: {evaluation['thunderball_hit_count']}",
                f"Total payout: GBP{evaluation['total_payout']}",
                f"Total cost: GBP{evaluation['total_cost']}",
                f"Net result: GBP{evaluation['net_result']}",
                "",
                "Per-ticket results:",
            ]
        )
        for item in evaluation["ticket_results"]:
            ticket_display = _format_ticket(item["main_numbers"], item["thunderball"])
            tb_label = "Y" if item["thunderball_match"] else "N"
            lines.append(
                f"{item['index']:02d}. {ticket_display} | main matches {item['main_matches']} | thunderball {tb_label} | payout GBP{item['payout']}"
            )

    lines.extend(
        [
            "",
            f"Next prediction for draw #{next_prediction['target_draw_number']}",
            "--------------------------------",
            f"Generated at: {next_prediction['generated_at']}",
            f"Prediction method: {next_prediction.get('prediction_method_label', 'Current Optimizer')}",
            f"Objective mode: {next_prediction['objective_mode']}",
            f"Estimated expected payout: {_format_optional_currency(next_prediction['estimated_expected_payout'])}",
            f"Estimated probability of payout >= GBP{next_prediction['target_payout']}: {_format_optional_percent(next_prediction['estimated_probability_target'])}",
            f"Estimated probability of break-even: {_format_optional_percent(next_prediction['estimated_probability_break_even'])}",
            f"Coverage score: {_format_optional_score(next_prediction['coverage_score'])}",
            "",
            "Predicted tickets:",
        ]
    )

    for index, ticket in enumerate(next_prediction["tickets"], start=1):
        lines.append(f"{index:02d}. {_format_ticket(ticket['main_numbers'], int(ticket['thunderball']))}")

    lines.extend(["", f"Model note: {next_prediction['note']}"])
    return subject, "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    official_history = _load_official_history()
    if official_history.empty:
        raise ValueError("No official draw history found in data file.")

    latest_draw_row = official_history.iloc[0]
    latest_draw_number = int(latest_draw_row["DrawNumber"])
    previous_state = _read_state()
    evaluation = _evaluate_prediction(previous_state, latest_draw_number, latest_draw_row)

    if _should_refresh_prediction_state(previous_state, latest_draw_number):
        next_prediction = _generate_next_prediction(latest_draw_number)
    else:
        next_prediction = previous_state

    subject, report = _build_report(latest_draw_row, evaluation, next_prediction, previous_state)

    _write_json_if_changed(STATE_FILE, next_prediction)
    _write_text_if_changed(REPORT_FILE, report)
    _write_text_if_changed(SUBJECT_FILE, subject + "\n")

    print(subject)
    if evaluation is None:
        print("No matching previous prediction was available for evaluation.")
    else:
        print(
            f"Evaluated previous prediction against draw #{latest_draw_number}: net GBP{evaluation['net_result']}"
        )
    print(f"Prepared next prediction for draw #{next_prediction['target_draw_number']}.")


if __name__ == "__main__":
    main()