# A* teacher freezes permanently after the shift (dense field disconnects free space)

**Milestone:** D3 (Go2 MuJoCo nav env)
**Symptom:** in the first shift-enabled demo episode (seed 7, shift at 8 s), the
distance-to-goal plot **flatlined for 38 s** right at shift time; action histogram was
~90 % brake; the standing robot was eventually hit by a drifting dynamic obstacle
(`COLLIDED(dynamic)` at t=46 s). Caught **visually** — the tracking figure made the
freeze obvious at a glance (this is why every milestone renders graphs).

## Diagnosis
Two suspects: (a) `PrivilegedExpert._dynamic_threat` latching on and yielding forever,
or (b) A* finding no path. A scratch diagnostic at the frozen state settled it:

```
dynamic threat: False
inflate=0.9:  occupied 82% of grid, path found: False
inflate=0.75: occupied 75% of grid, path found: False
inflate=0.6:  occupied 64% of grid, path found: False
inflate=0.45: occupied 52% of grid, path found: True (len 29)
```

Post-shift (22 obstacles, r=0.4) with the Go2-sized inflation (0.9 m), the inflated
occupancy blocks 82 % of the 10 m arena and **disconnects the free space** — A* returns
no path, `act()` returns brake, and since the static field never changes, it brakes
*every* subsequent step. The kinematic env never hit this because its expert only ran
pre-shift (11 obstacles) with a smaller robot (inflate 0.75).

## Fix
Progressive inflation fallback in `PrivilegedExpert` (`fallback_inflations_m` config):
plan with the safe margin first, retry with tighter margins only when the field is too
dense, brake only if even the minimum fails. Applied ladder for Go2: **(0.9, 0.7, 0.55)**.

**Second finding while tuning:** a 0.45 floor (= exactly the Go2's bounding circle)
produced `COLLIDED(static)` while squeezing corridors — the trotting base sways ~5 cm
and turns lag under dynamics, so the planner's margin must exceed the bounding circle.
**0.55 floor works**: seed-7 episode then REACHED through the shifted field at t=28.5 s.

## Lessons
- A planner that treats "no path" as "brake and wait" must have a densification
  fallback if the world can *permanently* densify mid-episode — otherwise it deadlocks.
- Under real dynamics, planner clearance must budget for gait sway + turn lag
  (bounding circle + ~0.1 m), not just the static footprint.
- The failure was invisible in scalar success metrics of earlier runs and obvious in
  the trajectory/goal-distance figure: keep rendering per-episode diagnostics.
