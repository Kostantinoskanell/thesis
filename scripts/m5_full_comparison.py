"""M5 Full Comparison (Navigation Layer)
Full 6-controller comparison over 10 seeds with the sensor-dropout shift.
Computes success rate, recovery time, SPL, and pairwise significance tests
(Holm-Bonferroni corrected).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch
import scipy.stats
from statsmodels.stats.multitest import multipletests

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.controllers.mlp import MLPPolicy, OnlineMLP
from nmc.controllers.tmnorm import TMNormController
from nmc.plasticity.stdp import STDPConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig

OUT = ROOT / "archive" / "M5_full_comparison"
SNN_CKPT = ROOT / "assets" / "snn_seeds" / "snn_seed0.pt"
MLP_CKPT = ROOT / "assets" / "mlp_frozen_go2.pt"


def load_snn_net():
    ckpt = torch.load(SNN_CKPT, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["tsteps"]

def readout_bounds(net, margin_scale=2.0):
    w = net.fc[-1].weight.detach().cpu().numpy()
    b = margin_scale * float(np.abs(w).max())
    return -b, b

def make_rstdp(seed):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=0.05, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=T, plasticity_enabled=True,
                            stdp_cfg=cfg, seed=seed, reward_mode="td",
                            gate_threshold=0.0, plastic_layers=[0, -1], anchor=0.005)

def make_pure_stdp(seed):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=False, eta=0.05, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=T, plasticity_enabled=True,
                            stdp_cfg=cfg, seed=seed, reward_mode="raw",
                            gate_threshold=0.0, plastic_layers=[0, -1], anchor=0.005)

def make_frozen_snn(seed):
    net, T = load_snn_net()
    return SNNNavController(net, n_steps=T, plasticity_enabled=False, seed=seed)

def make_tmnorm(seed):
    net, T = load_snn_net()
    return TMNormController(net, n_steps=T, seed=seed, ema_alpha=0.01)

def _load_mlp():
    ckpt = torch.load(MLP_CKPT, weights_only=True)
    pol = MLPPolicy(ckpt["obs_dim"], ckpt["n_actions"], hidden=tuple(ckpt.get("hidden", (512, 512))))
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol

def make_online_mlp():
    return OnlineMLP(_load_mlp(), frozen=False, lam=0.9)

def make_frozen_mlp():
    return OnlineMLP(_load_mlp(), frozen=True)

import concurrent.futures

def evaluate_seed(name, seed, shift_cfg, num_episodes):
    env = Go2NavEnv(shift_cfg)
    
    # We must instantiate the controller inside the worker to avoid pickling issues
    # But make_fn isn't passed directly. We use a factory.
    if name == "Frozen MLP": ctrl = make_frozen_mlp()
    elif name == "Online MLP": ctrl = make_online_mlp()
    elif name == "Frozen SNN": ctrl = make_frozen_snn(seed)
    elif name == "Pure-STDP SNN": ctrl = make_pure_stdp(seed)
    elif name == "R-STDP SNN": ctrl = make_rstdp(seed)
    elif name == "TM-NORM SNN": ctrl = make_tmnorm(seed)
    else: raise ValueError(f"Unknown controller {name}")
    
    succ_block = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ_block.append(int(info["reached"]))
    
    env.close()
    succ_arr = np.array(succ_block)
    succ_mean = succ_arr.mean()
    
    window = 5
    smoothed = np.convolve(succ_arr, np.ones(window)/window, mode='valid')
    rec_idx = np.where(smoothed >= 0.30)[0]
    rec_time = rec_idx[0] if len(rec_idx) > 0 else num_episodes
    
    return name, seed, succ_mean, rec_time

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout", type=float, default=0.30,
                    help="sensor dropout fraction (0.30 was M4's pick, later found "
                         "to already be floored -- see severity_screen.md)")
    args = ap.parse_args()

    out_dir = OUT if args.dropout == 0.30 else OUT.parent / f"M5_full_comparison_dropout{args.dropout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(6000, 6010))  # 10 seeds

    controllers = [
        "Frozen MLP",
        "Online MLP",
        "Frozen SNN",
        "Pure-STDP SNN",
        "R-STDP SNN",
        "TM-NORM SNN",
    ]

    print(f"Running M5 comparison at dropout={args.dropout} (Parallel over seeds)...", flush=True)
    # sensor_dropout_start pinned to 8 (matches pilot_m4.py's default) so the dead
    # beams are the SAME block every episode. Left at the class default (-1 =
    # random location per episode) there is no stable corrupted mapping for R-STDP
    # to learn, which silently defeats the whole point of this comparison.
    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=args.dropout, sensor_dropout_start=8,
                             episode_len_s=60.0)
    
    results = {name: {"success": [], "recovery": []} for name in controllers}
    
    # Generate all tasks
    tasks = []
    for name in controllers:
        for s in seeds:
            tasks.append((name, s, shift_cfg, 30))
            
    # Capped at 4 workers: each spawns a full MuJoCo + PyTorch process, and the
    # default (one per CPU core -- 20 on this machine) exhausted the page file.
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_seed, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            name, seed, succ_mean, rec_time = future.result()
            results[name]["success"].append(succ_mean)
            results[name]["recovery"].append(rec_time)
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{len(tasks)} tasks...", flush=True)

    # Convert lists to numpy arrays
    final_results = []
    for name in controllers:
        final_results.append({
            "name": name,
            "success": np.array(results[name]["success"]),
            "recovery": np.array(results[name]["recovery"])
        })
    results = final_results
    
    # Statistical testing vs R-STDP SNN
    rstdp_res = next(r for r in results if r["name"] == "R-STDP SNN")
    
    print("\n# M5 Results")
    print("| Controller | Success Rate (Mean ± SD) | Mean Recovery Time (Episodes) | p-value (Success vs R-STDP) | p-value (Recovery vs R-STDP) |")
    print("|---|---|---|---|---|")
    
    pvals_succ = []
    pvals_rec = []
    names = []
    for r in results:
        if r["name"] == "R-STDP SNN": continue
        _, p_succ = scipy.stats.ttest_ind(r["success"], rstdp_res["success"], equal_var=False)
        _, p_rec = scipy.stats.ttest_ind(r["recovery"], rstdp_res["recovery"], equal_var=False)
        pvals_succ.append(p_succ)
        pvals_rec.append(p_rec)
        names.append(r["name"])
        
    _, pvals_succ_corr, _, _ = multipletests(pvals_succ, method='holm')
    _, pvals_rec_corr, _, _ = multipletests(pvals_rec, method='holm')
    
    pval_map_succ = dict(zip(names, pvals_succ_corr))
    pval_map_rec = dict(zip(names, pvals_rec_corr))
    
    for r in results:
        name = r["name"]
        succ_mean = r["success"].mean()
        succ_std = r["success"].std()
        rec_mean = r["recovery"].mean()
        
        if name == "R-STDP SNN":
            p_s_str = "-"
            p_r_str = "-"
        else:
            p_s = pval_map_succ[name]
            p_r = pval_map_rec[name]
            p_s_str = f"{p_s:.3e}" + (" *" if p_s < 0.05 else "")
            p_r_str = f"{p_r:.3e}" + (" *" if p_r < 0.05 else "")
            
        print(f"| {name} | {succ_mean:.1%} ± {succ_std:.1%} | {rec_mean:.1f} | {p_s_str} | {p_r_str} |")
        
    with open(out_dir / "README.md", "w") as f:
        f.write("# M5 Full Comparison Results\n\n")
        f.write("| Controller | Success Rate (Mean ± SD) | Mean Recovery Time (Eps) | p-value (Success vs R-STDP) | p-value (Recovery vs R-STDP) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            name = r["name"]
            succ_mean = r["success"].mean()
            succ_std = r["success"].std()
            rec_mean = r["recovery"].mean()
            if name == "R-STDP SNN":
                p_s_str, p_r_str = "-", "-"
            else:
                p_s, p_r = pval_map_succ[name], pval_map_rec[name]
                p_s_str = f"{p_s:.3e}" + (" *" if p_s < 0.05 else "")
                p_r_str = f"{p_r:.3e}" + (" *" if p_r < 0.05 else "")
            f.write(f"| {name} | {succ_mean:.1%} ± {succ_std:.1%} | {rec_mean:.1f} | {p_s_str} | {p_r_str} |\n")

if __name__ == "__main__":
    main()
