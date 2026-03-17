from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ThunderballDraw:
    draw_date: date
    main_numbers: tuple[int, int, int, int, int]
    thunderball: int


@dataclass(frozen=True)
class PredictionResult:
    algorithm: str
    main_numbers: tuple[int, int, int, int, int]
    thunderball: int
    confidence: float
    note: str


@dataclass(frozen=True)
class TicketPrediction:
    main_numbers: tuple[int, int, int, int, int]
    thunderball: int


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    tickets: tuple[TicketPrediction, ...]
    ticket_cost: int
    target_payout: int
    estimated_expected_payout: float
    estimated_probability_target: float
    estimated_probability_break_even: float
    coverage_score: float
    note: str


@dataclass(frozen=True)
class TicketOutcomeResult:
    ticket: TicketPrediction
    main_match_count: int
    thunderball_match: bool
    payout: int


@dataclass(frozen=True)
class DrawOutcomeResult:
    draw_date: str
    actual_draw: TicketPrediction
    training_start_date: str
    training_end_date: str
    total_payout: int
    total_cost: int
    net_result: int
    best_main_match_count: int
    thunderball_hit_count: int
    winning_ticket_count: int
    played: bool
    edge_score: float
    payout_if_played: int
    net_if_played: int
    ticket_outcomes: tuple[TicketOutcomeResult, ...]


@dataclass(frozen=True)
class RollingTimelineResult:
    min_training_draws: int
    outcomes: tuple[DrawOutcomeResult, ...]
