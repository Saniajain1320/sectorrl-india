import os, time
from functools import lru_cache
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
from core import Engine

HERE = os.path.dirname(__file__)
app = FastAPI(title="SectorRL-India API", version="1.0",
              description="Sector allocation over 8 NSE sector indices + cash. PPO agents (anchored at inverse-GARCH-volatility weights, 50% tilt) and a live AR(1)-GARCH(1,1)-t forecaster. Educational project, not investment advice.")
E_ = Engine()
STARTED = time.time()

class AllocReq(BaseModel):
    date: str = Field(..., examples=["2024-06-03"], description="Decision date (a trading day in the dataset)")
    current_weights: Optional[List[float]] = Field(None, description="Current portfolio, 9 numbers (8 sectors + cash) summing to 1. Omit to use the weights the agent would hold on that date.")

def L(w): return {c: round(float(x), 5) for c, x in zip(E_.cols, w)}

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": len(E_.seeds), "uptime_s": round(time.time() - STARTED, 1)}

@app.get("/api/meta")
def meta():
    return {"sectors": E_.sectors, "series": E_.series, "decision_dates": E_.reb_dates, "tilt": E_.tilt, "cost_bps": E_.cost,
            "rebalance_every_days": E_.H, "anchor": E_.meta["anchor"], "seeds": E_.seeds, "data_range": [E_.date_str[0], E_.date_str[-1]],
            "trained_on": "2014-01 to 2022-12", "policy": "PPO MlpPolicy 2x64 tanh, 5 seeds averaged"}

@app.post("/api/allocate")
def allocate(req: AllocReq):
    if req.current_weights is not None:
        cw = np.asarray(req.current_weights, float)
        if cw.shape != (9,) or (cw < -1e-9).any() or abs(cw.sum() - 1) > 1e-3:
            raise HTTPException(422, "current_weights must be 9 non-negative numbers summing to 1")
        cw = cw / cw.sum()
    else: cw = None
    try: r = E_.allocate(req.date, cw)
    except KeyError: raise HTTPException(404, f"{req.date} is not a trading day in the dataset ({E_.date_str[0]} to {E_.date_str[-1]})")
    return {"date": r["date"], "trained_on": r["trained_on"], "in_test_window": r["in_test_window"],
            "weights": L(r["ensemble"]), "anchor_inverse_garch_vol": L(r["anchor"]), "equal_weight": L(r["equal"]),
            "starting_weights": L(r["from"]), "turnover": round(r["turnover"], 4), "estimated_cost_bps": round(r["est_cost_bps"], 2),
            "per_seed": {f"seed_{s}": L(w) for s, w in r["per_seed"].items()},
            "note": "Weights apply for the next 21 trading days. Educational model; not investment advice."}

@app.get("/api/forecast")
def forecast(series: str = Query("Nifty IT"), date: str = Query("2024-06-03"), horizon: int = Query(21, ge=1, le=63)):
    if series not in E_.series: raise HTTPException(404, f"series must be one of {E_.series}")
    if date not in E_.date_str: raise HTTPException(404, f"{date} is not a trading day in the dataset")
    return _fc(series, date, horizon)

@lru_cache(maxsize=512)
def _fc(series, date, horizon): return E_.forecast(series, date, horizon)

@app.get("/api/backtest")
def backtest():
    return _bt()

@lru_cache(maxsize=1)
def _bt(): return E_.backtest_summary()

app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "static", "index.html"))
