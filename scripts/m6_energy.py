"""M6 — energy of the NAVIGATION-layer controllers (H2), plus the firing-rate
distribution-shift figure (M4b item 6).

Ports L5's SynOps/Horowitz method from the locomotion layer to the nav layer,
with three honest extensions the plain literature model omits (see
`nmc.eval.energy`): neuron-state updates, the spike encoder, and the cost of
ONLINE LEARNING itself -- R-STDP's local update vs the online-MLP's backprop,
with R-STDP reported both as the dense reference implementation runs and as the
event-driven lower bound a crossbar/FPGA would reach (the quantified M7/M8 target).

Firing rates are MEASURED on real closed-loop rollouts (clean and shifted), not
assumed, because the whole point is that the rate is what sets the cost -- and
because R-STDP changes the weights, so its firing statistics are not the frozen
network's.

Run:  conda run -n nmc python scripts/m6_energy.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.energy import (Breakdown, OpCount, snn_inference, mlp_inference,
                             rstdp_update_dense, rstdp_update_event_driven,
                             backprop_update, tmnorm_update, E_AC, E_MAC, E_MUL)
from m5_full_comparison import (make_frozen_mlp, make_online_mlp, make_frozen_snn,
                                make_pure_stdp, make_rstdp, make_tmnorm)

import concurrent.futures

OUT = ROOT / "archive" / "M6_energy"

# nav-layer architecture (assets/snn_seeds/snn_seed0.pt, assets/mlp_frozen_go2.pt)
T = 20
SNN_FC = [(73, 512), (512, 512), (512, 64)]          # (in, out)
SNN_LAYERS = [512, 512, 64]
MLP_FC = [(37, 512), (512, 512), (512, 4), (512, 1)]  # incl. the value head
MLP_PARAMS = sum(i * o + o for i, o in MLP_FC)
PLASTIC_DIMS = [(73, 512), (512, 64)]                 # input + readout (M4/M5 recipe)

SNN_CONTROLLERS = ["Frozen SNN", "Pure-STDP SNN", "R-STDP SNN", "TM-NORM SNN"]
MLP_CONTROLLERS = ["Frozen MLP", "Online MLP"]
# success rates measured at dropout=0.20 (archive/M5_full_comparison_dropout0.2)
SUCCESS_020 = {"Frozen MLP": 0.340, "Online MLP": 0.240, "Frozen SNN": 0.213,
               "Pure-STDP SNN": 0.087, "R-STDP SNN": 0.193, "TM-NORM SNN": 0.017}


def build(name, seed):
    return {"Frozen MLP": lambda: make_frozen_mlp(),
            "Online MLP": lambda: make_online_mlp(),
            "Frozen SNN": lambda: make_frozen_snn(seed),
            "Pure-STDP SNN": lambda: make_pure_stdp(seed),
            "R-STDP SNN": lambda: make_rstdp(seed),
            "TM-NORM SNN": lambda: make_tmnorm(seed)}[name]()


def measure(name, seed, shifted, num_episodes):
    """Closed-loop rollout; returns per-episode firing statistics + step counts."""
    if shifted:
        cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                           sensor_dropout_frac=0.20, sensor_dropout_start=8,
                           episode_len_s=60.0)
    else:
        cfg = Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0)
    env = Go2NavEnv(cfg)
    ctrl = build(name, seed)
    is_snn = hasattr(ctrl, "spike_stats")

    episodes = []
    prev = {"spikes": 0.0, "slots": 0, "enc_s": 0.0, "enc_n": 0,
            "layer_s": [0.0] * len(SNN_FC), "layer_n": [0] * len(SNN_FC), "nd": 0}
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        steps = 0
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            steps += 1
            if term or trunc:
                break
        rec = {"steps": steps, "reached": int(info["reached"])}
        if is_snn:
            rec["layer_rates"] = [
                (ctrl.layer_spikes[i] - prev["layer_s"][i]) /
                max(ctrl.layer_slots[i] - prev["layer_n"][i], 1)
                for i in range(len(SNN_FC))]
            rec["enc_rate"] = ((ctrl.enc_spikes - prev["enc_s"]) /
                               max(ctrl.enc_slots - prev["enc_n"], 1))
            rec["mean_rate"] = ((ctrl.total_spikes - prev["spikes"]) /
                                max(ctrl.total_neuron_steps - prev["slots"], 1))
            rec["decision_rates"] = list(ctrl.decision_rates[prev["nd"]:])
            prev = {"spikes": ctrl.total_spikes, "slots": ctrl.total_neuron_steps,
                    "enc_s": ctrl.enc_spikes, "enc_n": ctrl.enc_slots,
                    "layer_s": list(ctrl.layer_spikes),
                    "layer_n": list(ctrl.layer_slots), "nd": ctrl.n_decisions}
        episodes.append(rec)
    env.close()
    return name, seed, shifted, episodes


def energy_for(name, enc_rate, layer_rates):
    """Per-decision energy breakdown for one controller at its measured rates."""
    if name in MLP_CONTROLLERS:
        bd = mlp_inference(MLP_FC, label=name)
        if name == "Online MLP":
            bd.learning = backprop_update(MLP_PARAMS, sum(i * o for i, o in MLP_FC))
        return bd, None

    pre_rates = [enc_rate, layer_rates[0], layer_rates[1]]
    neuron = "alif"
    bd = snn_inference(SNN_FC, pre_rates, SNN_LAYERS, T, neuron=neuron, label=name)
    bd_event = None
    if name in ("R-STDP SNN", "Pure-STDP SNN"):
        bd.learning = rstdp_update_dense(PLASTIC_DIMS, T)
        # event-driven variant: same inference, cheaper update
        ev = rstdp_update_event_driven(
            PLASTIC_DIMS,
            pre_rates=[enc_rate, layer_rates[1]],      # pre of fc[0], pre of fc[2]
            post_rates=[layer_rates[0], layer_rates[2]],
            T=T)
        bd_event = Breakdown(label=name + " (event-driven)", inference=bd.inference,
                             neurons=bd.neurons, encoder=bd.encoder, learning=ev)
    elif name == "TM-NORM SNN":
        bd.learning = tmnorm_update(SNN_LAYERS, T)
    return bd, bd_event


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controllers = MLP_CONTROLLERS + SNN_CONTROLLERS
    seeds = [6000, 6001, 6002]
    n_eps = 10

    tasks = [(n, s, sh, n_eps) for n in controllers for s in seeds for sh in (False, True)]
    print(f"Measuring firing rates: {len(tasks)} tasks x {n_eps} eps...", flush=True)

    raw = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(measure, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, shifted, episodes = f.result()
            raw.setdefault((name, shifted), []).extend(episodes)
            if (i + 1) % 6 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    # ---- energy table (shifted condition = the deployment case M5 measured) ----
    lines = ["# M6 — nav-layer energy (H2), 45 nm Horowitz / SynOps method\n",
             f"Measured on closed-loop rollouts, {len(seeds)} seeds x {n_eps} eps, "
             f"sensor-dropout=0.20 (the corrected severity from M5).\n",
             f"Per-op: AC {E_AC} pJ, MUL {E_MUL} pJ, MAC {E_MAC} pJ. SNN T={T}.\n",
             "\n## Per-decision energy breakdown (nJ)\n",
             "| controller | firing rate | inference | neurons | encoder | learning | **total** | vs frozen MLP |",
             "|---|---|---|---|---|---|---|---|"]

    results = {}
    frozen_mlp_total = None
    for name in controllers:
        eps = raw[(name, True)]
        if name in SNN_CONTROLLERS:
            enc = float(np.mean([e["enc_rate"] for e in eps]))
            lr = [float(np.mean([e["layer_rates"][i] for e in eps])) for i in range(3)]
            fr = float(np.mean([e["mean_rate"] for e in eps]))
        else:
            enc, lr, fr = 0.0, [0, 0, 0], float("nan")
        bd, bd_ev = energy_for(name, enc, lr)
        results[name] = {"bd": bd, "bd_ev": bd_ev, "enc": enc, "layer_rates": lr,
                         "firing": fr, "steps": float(np.mean([e["steps"] for e in eps]))}
        if name == "Frozen MLP":
            frozen_mlp_total = bd.total_pj

    for name in controllers:
        r = results[name]
        nj = r["bd"].as_nj()
        fr = "—" if np.isnan(r["firing"]) else f"{r['firing']:.2%}"
        ratio = frozen_mlp_total / r["bd"].total_pj
        lines.append(f"| {name} | {fr} | {nj['inference']:.1f} | {nj['neurons']:.1f} | "
                     f"{nj['encoder']:.1f} | {nj['learning']:.1f} | **{nj['total']:.1f}** | "
                     f"{ratio:.2f}x {'cheaper' if ratio > 1 else 'COSTLIER'} |")
        if r["bd_ev"] is not None:
            nje = r["bd_ev"].as_nj()
            re_ = frozen_mlp_total / r["bd_ev"].total_pj
            lines.append(f"| ↳ {name}, event-driven update (FPGA/crossbar target) | {fr} | "
                         f"{nje['inference']:.1f} | {nje['neurons']:.1f} | {nje['encoder']:.1f} | "
                         f"{nje['learning']:.1f} | **{nje['total']:.1f}** | "
                         f"{re_:.2f}x {'cheaper' if re_ > 1 else 'COSTLIER'} |")

    # ---- SynOps-only view (what the literature usually reports) ----
    lines += ["\n## Literature-standard view (SynOps vs MACs only, no neuron/encoder/learning)\n",
              "| controller | SynOps or MACs | energy (nJ) | vs frozen MLP |", "|---|---|---|---|"]
    mlp_macs = sum(i * o for i, o in MLP_FC)
    mlp_only_pj = mlp_macs * E_MAC
    lines.append(f"| Frozen MLP | {mlp_macs:,.0f} MACs | {mlp_only_pj/1e3:.1f} | 1.00x |")
    for name in SNN_CONTROLLERS:
        r = results[name]
        synops = sum(i * o * p * T for (i, o), p in
                     zip(SNN_FC, [r["enc"], r["layer_rates"][0], r["layer_rates"][1]]))
        pj = synops * E_AC
        lines.append(f"| {name} | {synops:,.0f} SynOps | {pj/1e3:.1f} | "
                     f"{mlp_only_pj/pj:.1f}x cheaper |")

    # ---- energy-to-success ----
    lines += ["\n## Energy per SUCCESSFUL navigation (deployment-relevant)\n",
              "energy/decision x decisions/episode / success-rate — a controller that is "
              "cheap but fails is not efficient.\n",
              "| controller | nJ/decision | decisions/ep | success (M5, dropout 0.20) | **µJ per success** |",
              "|---|---|---|---|---|"]
    for name in controllers:
        r = results[name]
        per_dec_nj = r["bd"].total_pj / 1e3
        sr = SUCCESS_020[name]
        uj = (per_dec_nj * r["steps"] / max(sr, 1e-9)) / 1e3
        lines.append(f"| {name} | {per_dec_nj:.1f} | {r['steps']:.0f} | {sr:.1%} | "
                     f"**{uj:.1f}** |")

    # ---- firing-rate distribution shift (M4b item 6) ----
    clean = np.array([r for e in raw[("Frozen SNN", False)] for r in e["decision_rates"]])
    shifted_frozen = np.array([r for e in raw[("Frozen SNN", True)] for r in e["decision_rates"]])
    rstdp_late = np.array([r for e in raw[("R-STDP SNN", True)][-len(seeds) * 3:]
                           for r in e["decision_rates"]])
    w_frozen = wasserstein_distance(clean, shifted_frozen)
    w_rstdp = wasserstein_distance(clean, rstdp_late)
    lines += ["\n## Firing-rate distribution shift (M4b item 6)\n",
              "Wasserstein distance of the per-decision firing-rate distribution from the "
              "clean (unshifted) frozen-SNN reference — does the shift move spike statistics, "
              "and does R-STDP move them back?\n",
              "| distribution | mean rate | Wasserstein vs clean |", "|---|---|---|",
              f"| frozen SNN, clean | {clean.mean():.2%} | 0 (reference) |",
              f"| frozen SNN, shifted | {shifted_frozen.mean():.2%} | {w_frozen:.2e} |",
              f"| R-STDP, shifted (late episodes) | {rstdp_late.mean():.2%} | {w_rstdp:.2e} |",
              "",
              f"**Renormalization: {'YES' if w_rstdp < w_frozen else 'NO'}** — R-STDP's firing "
              f"statistics are {'closer to' if w_rstdp < w_frozen else 'further from'} the clean "
              f"reference than the frozen network's ({w_rstdp:.2e} vs {w_frozen:.2e})."]

    # ---- figures ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = controllers
    bottoms = np.zeros(len(names))
    for key, colour in [("inference", "#1950a0"), ("neurons", "#2e86c1"),
                        ("encoder", "#8e44ad"), ("learning", "#c0392b")]:
        vals = np.array([results[n]["bd"].as_nj()[key] for n in names])
        a1.bar(range(len(names)), vals, bottom=bottoms, label=key, color=colour)
        bottoms += vals
    a1.set_xticks(range(len(names)))
    a1.set_xticklabels([n.replace(" SNN", "\nSNN").replace(" MLP", "\nMLP") for n in names],
                       fontsize=8)
    a1.set_ylabel("nJ per decision"); a1.set_yscale("log")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("per-decision energy breakdown (45 nm)")

    a2.hist(clean, bins=40, alpha=0.55, label=f"frozen, clean ({clean.mean():.2%})",
            color="#1f9d3a", density=True)
    a2.hist(shifted_frozen, bins=40, alpha=0.55,
            label=f"frozen, shifted ({shifted_frozen.mean():.2%})", color="#7f8c8d", density=True)
    a2.hist(rstdp_late, bins=40, alpha=0.55,
            label=f"R-STDP, shifted ({rstdp_late.mean():.2%})", color="#c0392b", density=True)
    a2.set_xlabel("per-decision firing rate"); a2.set_ylabel("density")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("firing-rate distribution under shift (M4b-6)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_m6_energy.png", bbox_inches="tight", dpi=150)

    # README.md is the hand-written narrative; this script owns results.md only.
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'results.md'} and {OUT/'fig_m6_energy.png'}")


if __name__ == "__main__":
    main()
