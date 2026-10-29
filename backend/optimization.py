"""Filter shipment actions against budget and delivery constraints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionOption:
    id: int
    type: str
    extra_cost: float
    delivery_days: int


def filter_feasible_actions(
    options: list[ActionOption],
    budget: float,
    max_delivery_days: int,
) -> list[ActionOption]:
    """Return actions that fit both the budget and delivery deadline."""
    return [
        option
        for option in options
        if option.extra_cost <= budget
        and option.delivery_days <= max_delivery_days
    ]


def filter_pareto_optimal_actions(
    options: list[ActionOption],
) -> list[ActionOption]:
    """Return actions not strictly dominated on cost and delivery time."""
    return [
        option
        for option in options
        if not any(
            other.extra_cost <= option.extra_cost
            and other.delivery_days <= option.delivery_days
            and (
                other.extra_cost < option.extra_cost
                or other.delivery_days < option.delivery_days
            )
            for other in options
        )
    ]
