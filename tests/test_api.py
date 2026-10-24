"""Run with:  cd app && python -m pytest ../tests -q     (needs: pip install pytest httpx)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import numpy as np
from fastapi.testclient import TestClient
from main import app, E_
c = TestClient(app)

def test_health_and_meta():
    assert c.get("/api/health").json()["models_loaded"] == 5
    m = c.get("/api/meta").json(); assert len(m["sectors"]) == 8 and len(m["decision_dates"]) >= 30

def test_allocate_valid_weights():
    d = c.get("/api/meta").json()["decision_dates"][10]
    r = c.post("/api/allocate", json={"date": d}).json()
    w = np.array(list(r["weights"].values()))
    assert abs(w.sum() - 1) < 1e-4 and (w >= 0).all()
    # tilt is 50%, so each weight must stay within [0.5*anchor, 0.5*anchor+0.5] of the anchor
    a = np.array(list(r["anchor_inverse_garch_vol"].values())); assert (w >= 0.5 * a - 1e-6).all()
    assert r["in_test_window"] is True

def test_allocate_depends_on_current_weights():
    d = c.get("/api/meta").json()["decision_dates"][10]
    a = c.post("/api/allocate", json={"date": d, "current_weights": [1/8]*8 + [0]}).json()
    b = c.post("/api/allocate", json={"date": d, "current_weights": [0]*8 + [1]}).json()
    assert a["weights"] != b["weights"] and b["turnover"] > a["turnover"]

def test_bad_inputs():
    assert c.post("/api/allocate", json={"date": "2024-06-01"}).status_code == 404       # Saturday
    assert c.post("/api/allocate", json={"date": "2024-06-03", "current_weights": [1, 1]}).status_code == 422
    assert c.get("/api/forecast?series=Nifty%20Foo").status_code == 404

def test_forecast():
    r = c.get("/api/forecast?series=Nifty%20Bank&date=2024-06-03&horizon=21").json()
    assert r["band80"][0] < r["expected_return"] < r["band80"][1] and 0 < r["persistence"] < 1.01 and len(r["daily_vol_path_pct"]) == 21

def test_backtest_matches_research_notebook():
    b = c.get("/api/backtest").json()
    assert abs(b["agent"]["CAGR"] - 0.1913) < 0.003 and abs(b["equal_weight"]["CAGR"] - 0.1797) < 0.002   # numbers in the report

def test_frontend_served():
    r = c.get("/"); assert r.status_code == 200 and "SectorRL-India" in r.text
