from fastapi.testclient import TestClient

from backend import main


class FakeCursor:
    def __init__(self, shipment):
        self.shipment = shipment

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, parameters):
        pass

    def fetchone(self):
        return self.shipment


class FakeConnection:
    def __init__(self, shipment=("SH-TEST", 14, 14)):
        self.shipment = shipment

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return FakeCursor(self.shipment)


def test_recommendations_endpoint_returns_optimized_options(monkeypatch):
    monkeypatch.setattr(main, "get_db_connection", FakeConnection)

    response = TestClient(main.app).get(
        "/shipments/SH-TEST/recommendations"
    )

    assert response.status_code == 200
    assert [option["type"] for option in response.json()["options"]] == [
        "Air Freight",
        "Secondary Supplier",
        "Delay Launch",
    ]


def test_execute_rejects_an_action_excluded_by_constraints(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_db_connection",
        lambda: FakeConnection(("SH-TEST", 20, 20)),
    )

    response = TestClient(main.app).post(
        "/shipments/SH-TEST/execute",
        json={"decision_id": 3},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Recommendation not found"
