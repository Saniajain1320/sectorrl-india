"""Export everything the web service needs from the research notebook state (run once, offline).
   - data/bundle.npz : features + volatility + returns used by the policy and the GARCH endpoint
   - data/policy.npz : the 5 trained PPO actors as plain numpy arrays (no torch needed in production)
   - data/meta.json  : names, splits, constants
"""
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, '/home/claude/fe')
from common import G
from stable_baselines3 import PPO
OUT = '/home/claude/deploy/app/data'
data, cfg, panel, TR = G["data"], G["cfg"], G["panel"], G["TRADABLE"]
dates = data["dates"]; SER = TR + [cfg["benchmark"]]
lr = np.stack([panel[panel.SYMBOL == s].set_index("TIMESTAMP").sort_index().reindex(dates)["log_return"].to_numpy(float) for s in SER], 1)
np.savez_compressed(f"{OUT}/bundle.npz", dates=np.array([d.strftime("%Y-%m-%d") for d in dates]), Xs=data["Xs"].astype(np.float32),
    Ms=data["Ms"].astype(np.float32), X=data["X"].astype(np.float32), G=data["G"].astype(np.float64), logcum=data["logcum"],
    R_next=data["R_next"], bm_next=data["bm_next"], split=data["split"].astype(str), lr=lr)
arr = {}
for s in G["SEEDS"]:
    m = PPO.load(f"/home/claude/fe/models/ppo_s{s}", device="cpu")
    sd = {k: v.cpu().numpy() for k, v in m.policy.state_dict().items()}
    arr[f"s{s}_W0"] = sd["mlp_extractor.policy_net.0.weight"]; arr[f"s{s}_b0"] = sd["mlp_extractor.policy_net.0.bias"]
    arr[f"s{s}_W1"] = sd["mlp_extractor.policy_net.2.weight"]; arr[f"s{s}_b1"] = sd["mlp_extractor.policy_net.2.bias"]
    arr[f"s{s}_Wa"] = sd["action_net.weight"]; arr[f"s{s}_ba"] = sd["action_net.bias"]
np.savez_compressed(f"{OUT}/policy.npz", **arr)
json.dump({"sectors": TR, "series": SER, "benchmark": cfg["benchmark"], "seeds": G["SEEDS"], "tilt": float(G["TILT"]) if "TILT" in G else 0.5,
           "temperature": 3.0, "rebalance_every": 21, "cost_bps": 15, "anchor": "INVVOL",
           "splits": cfg["splits"], "train_start": cfg["splits"]["train"][0]}, open(f"{OUT}/meta.json", "w"), indent=1)
print("exported", {k: v.shape for k, v in arr.items() if k.startswith("s0")})
