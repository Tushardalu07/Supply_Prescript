from backend.optimization import ActionOption, filter_feasible_actions


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
