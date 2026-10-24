# SectorRL-India

Monthly allocation across 8 NSE sector indices plus cash, using time-series analysis (ARMA, ARCH/GARCH) and a PPO reinforcement-learning agent.

This repository is the deployed service: a FastAPI app that serves the interactive front end and a JSON API.

| Endpoint | What it does |
|---|---|
| `GET /` | The web app (pipeline, time-series lab, RL lab, game, risk outlook, live model) |
| `POST /api/allocate` | Runs the 5 trained PPO agents on a date's features and returns the ensemble weights |
| `GET /api/forecast` | Fits AR(1)-GARCH(1,1)-t on the history up to a date and returns a 21-day forecast range |
| `GET /api/backtest` | Test-years (2023-25) scorecard of the deployed agents against equal weight and Nifty 50 |
| `GET /api/health`, `GET /api/meta` | Status and metadata |
| `GET /docs` | Interactive API documentation |

Educational project, not investment advice. Agents were trained on 2014-2022 and are evaluated once on 2023-2025.

## Deploying (free)

Render, free web service, no Docker needed: build `pip install -r requirements.txt`, start `cd app && uvicorn main:app --host 0.0.0.0 --port $PORT`, environment variable `PYTHON_VERSION=3.11.9`. `render.yaml` holds the same settings. The free tier sleeps after 15 idle minutes; the first request after that takes about a minute.
