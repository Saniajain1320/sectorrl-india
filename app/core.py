"""SectorRL-India service core: policy inference (numpy), portfolio accounting, live GARCH forecasts."""
import json, os, warnings
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
warnings.filterwarnings("ignore")

class Engine:
    def __init__(self, data_dir=os.path.join(HERE, "data")):
        self.meta = json.load(open(f"{data_dir}/meta.json"))
        b = np.load(f"{data_dir}/bundle.npz", allow_pickle=False)
        self.dates = pd.to_datetime(b["dates"]); self.date_str = [d.strftime("%Y-%m-%d") for d in self.dates]
        self.Xs, self.Ms, self.X, self.Gv = b["Xs"], b["Ms"], b["X"], b["G"]
        self.logcum, self.R_next, self.bm_next, self.split, self.lr = b["logcum"], b["R_next"], b["bm_next"], b["split"], b["lr"]
        p = np.load(f"{data_dir}/policy.npz")
        self.seeds = self.meta["seeds"]
        self.nets = {s: [p[f"s{s}_{k}"] for k in ("W0", "b0", "W1", "b1", "Wa", "ba")] for s in self.seeds}
        self.sectors, self.series = self.meta["sectors"], self.meta["series"]
        self.cols = self.sectors + ["CASH"]
        self.tilt, self.temp, self.H, self.cost = self.meta["tilt"], self.meta["temperature"], self.meta["rebalance_every"], self.meta["cost_bps"]
        idx = np.where(self.split == "test")[0][:-1]
        self.test_idx = idx
        self.reb_t = list(idx[::self.H])                       # decision days of the test years (every 21 trading days)
        self.reb_dates = [self.date_str[t] for t in self.reb_t]
        self._path = None

    # ---- policy -------------------------------------------------------------------------------------
    def anchor(self, t):
        inv = 1.0 / self.Gv[t]; return np.append(inv / inv.sum(), 0.0)

    def obs(self, t, w):
        return np.concatenate([self.Xs[t].ravel(), self.Ms[t], w]).astype(np.float32)

    def act(self, seed, o):
        W0, b0, W1, b1, Wa, ba = self.nets[seed]
        h = np.tanh(W0 @ o + b0); h = np.tanh(W1 @ h + b1)
        return np.clip(Wa @ h + ba, -1, 1)

    def target(self, seed, t, w_prev):
        a = self.act(seed, self.obs(t, w_prev)); z = self.temp * a; e = np.exp(z - z.max()); sm = e / e.sum()
        return (1 - self.tilt) * self.anchor(t) + self.tilt * sm, a

    # ---- rolled-forward path of every seed through the test years (what the agent actually held) --------
    def path(self):
        if self._path is None:
            out = {}
            for s in self.seeds:
                w = self.anchor(self.reb_t[0]); rows = {}
                for k, t in enumerate(self.reb_t):
                    tgt, _ = self.target(s, t, w); rows[t] = (w.copy(), tgt)
                    h = len(self.test_idx[k * self.H:(k + 1) * self.H])
                    P = np.append(np.exp(self.logcum[t + h] - self.logcum[t]), 1.0)
                    w = tgt * P / (tgt @ P)
                out[s] = rows
            self._path = out
        return self._path

    def allocate(self, date, current=None):
        t = self.date_str.index(date) if date in self.date_str else None
        if t is None: raise KeyError(date)
        per, wprev = {}, {}
        for s in self.seeds:
            if current is not None: w0 = np.asarray(current, float)
            elif t in self.path()[self.seeds[0]]: w0 = self.path()[s][t][0]
            else: w0 = self.anchor(t)
            tgt, a = self.target(s, t, w0); per[s] = tgt; wprev[s] = w0
        ens = np.mean([per[s] for s in self.seeds], 0)
        w_start = np.mean([wprev[s] for s in self.seeds], 0)
        turnover = float(np.abs(ens - w_start).sum())
        return {"date": date, "ensemble": ens, "anchor": self.anchor(t), "equal": np.array([1 / 8] * 8 + [0.0]),
                "per_seed": per, "from": w_start, "turnover": turnover, "est_cost_bps": turnover * self.cost,
                "trained_on": "2014-01 to 2022-12", "in_test_window": bool(self.split[t] == "test")}

    # ---- accounting ---------------------------------------------------------------------------------------
    def backtest(self, W):
        """W: dict {t: weights}. Same rules as the research notebook (15 bps on turnover, cash earns 0)."""
        w = W[self.test_idx[0]]; rets = []
        for t in self.test_idx:
            tgt = W.get(t, w); to = np.abs(tgt - w).sum()
            r = np.append(self.R_next[t], 0.0); gross = tgt @ r
            rets.append((1 - self.cost / 1e4 * to) * (1 + gross) - 1); w = tgt * (1 + r) / (1 + gross)
        return np.array(rets)

    @staticmethod
    def stats(r):
        eq = np.cumprod(1 + r); yrs = len(r) / 252
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return {"CAGR": float(eq[-1] ** (1 / yrs) - 1), "Volatility": float(r.std(ddof=1) * np.sqrt(252)),
                "Sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)), "MaxDrawdown": float(dd)}

    def backtest_summary(self):
        ens = {t: np.mean([self.path()[s][t][1] for s in self.seeds], 0) for t in self.reb_t}
        ew = {t: np.array([1 / 8] * 8 + [0.0]) for t in self.reb_t}
        iv = {t: self.anchor(t) for t in self.reb_t}
        r_a, r_e, r_i = self.backtest(ens), self.backtest(ew), self.backtest(iv)
        bm = self.bm_next[self.test_idx]
        curve = lambda r: [round(float(x), 4) for x in np.cumprod(1 + r)[::5]]
        return {"period": [self.date_str[self.test_idx[0]], self.date_str[self.test_idx[-1]]],
                "agent": self.stats(r_a), "equal_weight": self.stats(r_e), "inverse_vol_anchor": self.stats(r_i), "nifty50": self.stats(bm),
                "dates": [self.date_str[t] for t in self.test_idx[::5]],
                "curves": {"agent": curve(r_a), "equal_weight": curve(r_e), "nifty50": curve(bm)}}

    # ---- live GARCH ----------------------------------------------------------------------------------------
    def forecast(self, series, date, horizon=21):
        from arch import arch_model
        j = self.series.index(series); t = self.date_str.index(date)
        t0 = self.date_str.index(self.meta["train_start"]) if self.meta["train_start"] in self.date_str else int(np.argmax(~np.isnan(self.lr[:, j])))
        hist = pd.Series(self.lr[t0:t + 1, j] * 100).dropna()
        res = arch_model(hist, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
        f = res.forecast(horizon=horizon, reindex=False)
        m, v = f.mean.values[-1] / 100, f.variance.values[-1] / 1e4
        mu, sd = float(m.sum()), float(np.sqrt(v.sum()))
        pr = res.params; persist = float(pr["alpha[1]"] + pr["beta[1]"])
        out = {"series": series, "date": date, "horizon_days": horizon, "model": "AR(1)-GARCH(1,1)-t", "n_obs": int(len(hist)),
               "expected_return": mu, "sd": sd, "band80": [mu - 1.2815516 * sd, mu + 1.2815516 * sd],
               "daily_vol_now_pct": float(np.sqrt(res.conditional_volatility.iloc[-1] ** 2)),
               "params": {k: float(x) for k, x in pr.items()}, "persistence": persist,
               "vol_half_life_days": float(np.log(0.5) / np.log(persist)) if 0 < persist < 1 else None,
               "daily_vol_path_pct": [float(np.sqrt(x) * 100) for x in v]}
        fut = self.lr[t + 1:t + 1 + horizon, j]
        if len(fut) == horizon and not np.isnan(fut).any(): out["realised_return"] = float(fut.sum())
        return out
