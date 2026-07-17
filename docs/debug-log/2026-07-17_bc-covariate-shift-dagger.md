# Behavior cloning: 86.5% step accuracy but 23% closed-loop success (covariate shift)

**Milestone:** M2 (MLP baselines on the dynamic Go2 env)
**Symptom:** the frozen MLP student trained on D3's A* demos reached **86.5% validation
accuracy** (per-class 0.79–0.93, no majority collapse), yet in closed-loop evaluation on
held-out seeds it scored only **23% success (7/30)** with 21 collisions — versus the
teacher's 62%.

## Cause
Textbook **imitation-learning covariate shift** (the DAgger paper's motivating problem):
step accuracy is measured on the *teacher's* state distribution, but at execution time
the student's small errors compound and drift it into states the teacher never visited
(too close to obstacles, wrong approach angles). There the policy has no training signal
and behaves arbitrarily → collision. High offline accuracy, poor online performance.

Full dynamics make it worse than the kinematic env: turn lag and gait sway mean even a
correct-in-hindsight action executes slightly differently than in the demos.

## Fix
**DAgger** (`scripts/dagger_go2.py`): roll out the *student* (visiting its own state
distribution), query the A* teacher for the correct action in every visited state,
aggregate into the dataset, retrain. Our privileged teacher is callable per-state and
the env runs ~20× realtime, so iterations are cheap (~5 min each).

## Lessons
- Never accept offline imitation accuracy as evidence a policy works: **only closed-loop
  evaluation counts.** (Same lesson as the CPG "metric said PASS, GIF said fell" entry —
  different milestone, same failure shape.)
- Keep teachers *queryable* (state → action), not just demo generators; it makes DAgger
  and any future corrective labeling nearly free.
- The M3 SNN student should be trained on the DAgger-aggregated dataset
  (`data/imitation_go2_dagger.npz`), not the raw demos, or it will inherit this problem.
