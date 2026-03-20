from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from thunderball_predictor.methods import (
    AUTOMATION_CONFIG_DEFAULTS,
    CURRENT_OPTIMIZER_METHOD,
    DELTA_SYSTEM_METHOD,
    MAIN_SUM_METHOD,
    METHOD_LABELS,
    POSITION_RANGE_METHOD,
    RANDOM_METHOD,
    normalize_automation_config,
)

CONFIG_FILE = Path("reports/future_prediction_config.json")
OBJECTIVE_MODE_LABELS = {
    "downside_aware": "Downside Aware",
    "balanced": "Balanced",
    "main_hit_focused": "Main-Hit Focused (3+ Mains)",
}


def _load_config() -> dict[str, object]:
    if not CONFIG_FILE.exists():
        return normalize_automation_config(dict(AUTOMATION_CONFIG_DEFAULTS))

    try:
        return normalize_automation_config(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except Exception:
        return normalize_automation_config(dict(AUTOMATION_CONFIG_DEFAULTS))


def _save_config(config: dict[str, object]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(normalize_automation_config(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


st.set_page_config(page_title="Prediction Configuration", page_icon="⚙️", layout="wide")
st.title("Prediction Configuration")
st.caption(
    "Choose which prediction method the automated GitHub Action should use when it generates the next saved portfolio."
)
st.markdown(
    "This page controls `evaluate_and_predict.py`. The selected configuration is written to "
    "`reports/future_prediction_config.json`, which the scheduled workflow reads before creating the next prediction."
)

config = _load_config()
method_options = [CURRENT_OPTIMIZER_METHOD, DELTA_SYSTEM_METHOD, POSITION_RANGE_METHOD, RANDOM_METHOD, MAIN_SUM_METHOD]

left_col, right_col = st.columns([1.4, 1])
with left_col:
    selected_method = st.selectbox(
        "Automation Prediction Method",
        options=method_options,
        index=method_options.index(str(config["method"])),
        format_func=lambda value: METHOD_LABELS[value],
    )

    ticket_count = st.slider(
        "Ticket Count",
        min_value=1,
        max_value=20,
        value=int(config["ticket_count"]),
        step=1,
    )
    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=int(config["seed"]),
        step=1,
    )

    optimizer_objective_mode = str(config["optimizer_objective_mode"])
    delta_top_signatures = int(config["delta_top_signatures"])
    range_lookback_draws = int(config["range_lookback_draws"])
    sum_sigma_multiplier = float(config["sum_sigma_multiplier"])

    if selected_method == CURRENT_OPTIMIZER_METHOD:
        optimizer_objective_mode = st.selectbox(
            "Optimizer Strategy",
            options=list(OBJECTIVE_MODE_LABELS.keys()),
            index=list(OBJECTIVE_MODE_LABELS.keys()).index(str(config["optimizer_objective_mode"])),
            format_func=lambda value: OBJECTIVE_MODE_LABELS[value],
        )
    elif selected_method == DELTA_SYSTEM_METHOD:
        delta_top_signatures = st.slider(
            "Top Delta Signatures Considered",
            min_value=1,
            max_value=30,
            value=int(config["delta_top_signatures"]),
            step=1,
        )
    elif selected_method == POSITION_RANGE_METHOD:
        range_lookback_draws = st.slider(
            "Position Range Lookback",
            min_value=3,
            max_value=20,
            value=int(config["range_lookback_draws"]),
            step=1,
        )
    elif selected_method == RANDOM_METHOD:
        st.info("Random method has no additional parameters. Only ticket count and seed apply.")
    elif selected_method == MAIN_SUM_METHOD:
        sum_sigma_multiplier = st.slider(
            "Sum Filter Width (σ multiplier)",
            min_value=0.5,
            max_value=3.0,
            value=float(config["sum_sigma_multiplier"]),
            step=0.25,
        )

    updated_config = normalize_automation_config(
        {
            "method": selected_method,
            "ticket_count": ticket_count,
            "seed": int(seed),
            "optimizer_objective_mode": optimizer_objective_mode,
            "delta_top_signatures": delta_top_signatures,
            "range_lookback_draws": range_lookback_draws,
            "sum_sigma_multiplier": sum_sigma_multiplier,
        }
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Save Automation Config", use_container_width=True):
            _save_config(updated_config)
            st.success("Automation configuration saved.")
    with action_col2:
        if st.button("Reset To Defaults", use_container_width=True):
            _save_config(dict(AUTOMATION_CONFIG_DEFAULTS))
            st.success("Automation configuration reset to defaults.")
            st.rerun()

with right_col:
    st.markdown("**What the action will use**")
    st.markdown(f"Method: **{METHOD_LABELS[str(updated_config['method'])]}**")
    st.markdown(f"Ticket count: **{int(updated_config['ticket_count'])}**")
    st.markdown(f"Seed: **{int(updated_config['seed'])}**")

    if str(updated_config["method"]) == CURRENT_OPTIMIZER_METHOD:
        st.markdown(
            f"Strategy: **{OBJECTIVE_MODE_LABELS[str(updated_config['optimizer_objective_mode'])]}**"
        )
        st.info(
            "Uses the simulation-based current optimizer. This is the only method that produces the simulated payout, "
            "break-even probability, and coverage metrics in the saved automated report."
        )
    elif str(updated_config["method"]) == DELTA_SYSTEM_METHOD:
        st.markdown(f"Top signatures: **{int(updated_config['delta_top_signatures'])}**")
        st.info(
            "Uses recurring delta signatures from historical sorted-ball gap patterns to generate the next 9-ticket portfolio."
        )
    elif str(updated_config["method"]) == POSITION_RANGE_METHOD:
        st.markdown(f"Lookback draws: **{int(updated_config['range_lookback_draws'])}**")
        st.info(
            "Uses draw-position range, mean-reversion, and momentum signals to bias each sorted ball slot and the Thunderball."
        )
    elif str(updated_config["method"]) == RANDOM_METHOD:
        st.info(
            "Generates completely random valid tickets with no historical data analysis. "
            "This serves as a statistical baseline—all other methods should substantially outperform random selection to be viable."
        )
    elif str(updated_config["method"]) == MAIN_SUM_METHOD:
        st.markdown(f"Sum filter: **±{float(updated_config['sum_sigma_multiplier'])}σ**")
        st.info(
            "Constrains generated tickets so that the main-ball sum falls within the historical mean ± σ multiplier. "
            "Narrower bands (lower σ) favour historically central sums; wider bands allow more variation."
        )

st.markdown("**Saved Config Preview**")
st.code(json.dumps(updated_config, indent=2, sort_keys=True), language="json")
st.caption(
    "If you want the scheduled GitHub Action to use a new selection in the repository, make sure the updated "
    "`reports/future_prediction_config.json` file is committed."
)
