# Neuromorphic Robotics: Navigation & Online Adaptation

## Most directly relevant to our thesis (adaptation under change)
- **Rapid-Adapting & Continual-Learning SNN Path Planning for Mobile Robots (2024)**,
  arXiv:2404.15524 — SNN that adapts/continual-learns for mobile-robot path planning.
  **Nearest neighbour to our "recover after a mid-episode shift" claim** — read for the
  adaptation protocol and metrics. https://arxiv.org/abs/2404.15524
- **Lobov et al. (2020)**, *Front. Neuroscience* — STDP-controlled mobile robot with
  memristive devices. (In proposal — nearest predecessor.)
- **Juarez-Lora et al. (2022)** — R-STDP arm control under changing friction. (In proposal.)

## SNN + RL and continuous control (the L-track core — spiking locomotion policy)
- **PopSAN — Tang et al. (2020)**, CoRL, arXiv:2010.09635 — **the L-track foundation.**
  Population-coded spiking actor + deep critic; learnable Gaussian input populations +
  learnable population output decoder; integrates with PPO/TD3/SAC/DDPG; 140× less energy on
  Loihi. Its RateSAN ablation = why we can't use a plain rate readout. Code:
  https://github.com/combra-lab/pop-spiking-deep-rl · https://arxiv.org/abs/2010.09635
- **Fully Spiking NN for Legged Robots — Wang/Wu et al. (2023)**, arXiv:2310.05022 — PopSAN on
  A1/Cassie/MIT-Humanoid in Isaac Gym (RMA+AMP); nearest prior work to the L-track; SNN ≈ ANN
  across terrains. https://arxiv.org/abs/2310.05022
- **MDC-SAN (AAAI 2022)** — population coding + 2nd-order dynamic neurons; SOTA upgrade option
  (L6). https://ojs.aaai.org/index.php/AAAI/article/view/19879
- **ILC-SAN — Chen et al. (2024)** — first *fully* spiking actor (membrane-voltage action decode
  + intralayer connections) matching deep RL; SOTA upgrade option (L6).
- **Noisy Spiking Actor Network (2024)**, arXiv:2403.04162 — spiking exploration; option if PPO
  exploration is weak. https://arxiv.org/abs/2403.04162
- **Exploring SNNs for deep RL in robotic tasks (2024)**, *Scientific Reports* — framing.
  https://www.nature.com/articles/s41598-024-77779-8
- **SNNs for Continuous Control via end-to-end training (2025)**, arXiv:2509.05356.
  https://arxiv.org/abs/2509.05356

## LiDAR-specific (our exact sensor)
- **On the Importance of Neural Membrane Potential Leakage for LiDAR-Based Robot
  Obstacle Avoidance Using SNNs (2025)** — directly about LIF leak (our τ_m / H4 low-pass
  robustness claim) with LiDAR input. https://elmi.hbku.edu.qa/en/publications/on-the-importance-of-neural-membrane-potential-leakage-for-lidar-/

## Dynamic obstacles / event sensing (future extension, not core)
- **Event-Enhanced Multi-Modal SNN for Dynamic Obstacle Avoidance**, arXiv:2310.02361 —
  if a physical demo with an event camera is ever added. https://arxiv.org/abs/2310.02361
