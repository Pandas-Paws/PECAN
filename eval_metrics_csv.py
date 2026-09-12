#!/usr/bin/env python3
"""Compute NSE, KGE, CC, PBIAS, FHV, FLV for G1 and G2 from a test_preds CSV."""
import sys, numpy as np, pandas as pd

def nse(obs, sim):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

def kge(obs, sim):
    r  = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def pbias(obs, sim):
    return 100 * (np.sum(sim - obs) / np.sum(obs))

def fhv(obs, sim, pct=0.02):
    n = max(1, int(len(obs) * pct))
    idx = np.argsort(obs)[-n:]
    return 100 * (np.sum(sim[idx] - obs[idx]) / np.sum(obs[idx]))

def flv(obs, sim, pct=0.30):
    n = max(1, int(len(obs) * pct))
    idx = np.argsort(obs)[:n]
    lo, ls = np.log(obs[idx] + 1e-6), np.log(sim[idx] + 1e-6)
    return 100 * (np.sum(ls - lo) / np.sum(lo))

csv = sys.argv[1]
df = pd.read_csv(csv)
for g, obs_col, sim_col in [("G1 (outlet)", "Observed_1", "Predicted_1"),
                              ("G2 (headwater)", "Observed_2", "Predicted_2")]:
    obs = df[obs_col].values.astype(float)
    sim = df[sim_col].values.astype(float)
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0)
    obs, sim = obs[mask], sim[mask]
    r = np.corrcoef(obs, sim)[0, 1]
    print(f"{g}:")
    print(f"  NSE={nse(obs,sim):.4f}  KGE={kge(obs,sim):.4f}  CC={r:.4f}  "
          f"PBIAS={pbias(obs,sim):.2f}%  FHV={fhv(obs,sim):.2f}%  FLV={flv(obs,sim):.2f}%")
