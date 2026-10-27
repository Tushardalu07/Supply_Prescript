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
