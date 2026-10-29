from backend.optimization import (
    ActionOption,
    filter_feasible_actions,
    filter_pareto_optimal_actions,
)


def test_keeps_action_within_budget_and_deadline():
    options = [
        ActionOption(1, "Air Freight", 15000, 2),
        ActionOption(2, "Secondary Supplier", 12000, 5),
    ]

    result = filter_feasible_actions(options, budget=15000, max_delivery_days=3)

    assert result == [options[0]]


def test_excludes_action_over_budget():
    option = ActionOption(1, "Air Freight", 15001, 2)

    assert filter_feasible_actions([option], 15000, 3) == []


def test_excludes_action_over_delivery_deadline():
    option = ActionOption(1, "Secondary Supplier", 10000, 5)

    assert filter_feasible_actions([option], 15000, 4) == []


def test_pareto_filter_excludes_dominated_actions():
    options = [
        ActionOption(1, "Air Freight", 15000, 2),
        ActionOption(2, "Secondary Supplier", 12000, 5),
        ActionOption(3, "Delay Launch", 0, 14),
        ActionOption(4, "Slower and more expensive", 16000, 8),
    ]

    assert filter_pareto_optimal_actions(options) == options[:3]


def test_pareto_filter_keeps_cost_speed_tradeoffs():
    faster = ActionOption(1, "Air Freight", 15000, 2)
    cheaper = ActionOption(2, "Secondary Supplier", 12000, 5)

    assert filter_pareto_optimal_actions([faster, cheaper]) == [faster, cheaper]


def test_pareto_filter_keeps_equivalent_options():
    first = ActionOption(1, "Supplier A", 10000, 4)
    second = ActionOption(2, "Supplier B", 10000, 4)

    assert filter_pareto_optimal_actions([first, second]) == [first, second]
