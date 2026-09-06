# Five-candidate waypoint MADDPG pretraining

This directory is a standalone training path. It does not import or modify the
existing continuous-action MADDPG environments or Gazebo controllers.

## Fixed task interface

- Agents: `go2`, `go3`; go1 is the moving formation reference.
- Default slots in the go1 frame: go2 `(2,+2)`, go3 `(2,-2)` metres.
- Actions: go1-frame lateral offsets `(+2,+1,0,-1,-2)` metres.
- Action transitions are adjacent-only at each decision: an agent may hold its
  current candidate or move one level left/right.  For example, `0` m may
  transition to `+1`, `0`, or `-1` m, but never directly to `+2` or `-2` m.
- This freedom is per-agent: a follower whose own default corridor is clear is
  held at its default candidate.  Only a blocked follower may detour; after
  three consecutive clear observations it returns one adjacent level per
  second toward its default slot.
- High-level decision period: 1 s.
- Nav2 proxy update period: 0.1 s.
- Follower maximum speed: 0.15 m/s.
- Leader pretraining speed: 0.10 m/s, providing recovery margin after a detour.
- Episode layout: a clear straight approach, obstacle avoidance, then a
  straight recovery in the default triangular formation.
- Obstacle: one axis-aligned 1.5 x 1.5 m box, with centre x randomly sampled
  from `[8,10]` m and placement randomized between the two follower lanes.
- An episode allows 130 decisions and succeeds only after both followers have
  passed the obstacle by 2 m and held the recovered default formation.

## Lidar interface

The actor always receives 36 lidar sectors. Python initially ray-casts 36
directions directly. Gazebo may retain its 440 x 16 VLP-16 cloud, but the new
deployment observer must project and minimum-pool it to the same 36 sectors.

- Gazebo plugin range: 0.9--130 m.
- Existing mapping/preprocessing range: 0.9--20 m.
- Gaussian range noise: 0.008 m.
- 360-degree field of view.
- Gazebo sensor planar offset from `base_link`: `(0.20, 0.0)` m.

The 83-D per-agent observation is composed of 36 lidar sectors, 25 candidate
features, formation and teammate states, role, previous five-way action, and
the active Nav2-proxy goal/progress state.

Candidate path clearance covers both the transition to the current waypoint
and a 3 m forward corridor from that waypoint. This is necessary because a
formation-relative goal advances only about 0.1 m per one-second decision when
go1 travels at the configured pretraining speed.

The team reward has five semantic terms:

1. Task completion: `+50` on success and `-0.02` per decision.
2. Obstacle avoidance: continuous executed/path clearance costs, `-5` for a
   blocked candidate, and `-100` for an obstacle collision.
3. Inter-robot avoidance: continuous executed/path separation costs and `-100`
   for a pair collision.
4. Formation preservation: goal-pair spacing plus `-0.60` per unnecessary
   metre beyond the nearest safe candidate.  A two-metre detour is therefore
   penalized when one metre is safe, but not when two metres is necessary.
5. Forward progress: up to `+0.50` per decision to prevent standing still.

Continuous proximity penalties are clipped before squaring.  Success already
requires both followers to return to their default slots, so there is no
separate rejoin bonus or action-switch term.

## Algorithm

Actors emit five logits. Training uses straight-through Gumbel-Softmax and each
centralized critic consumes both observations and both five-dimensional
one-hot actions. Deployment uses `argmax`.

## Commands

Run interface and learner checks:

```bash
python -m waypoint_maddpg_v0.smoke_test
```

Run a tiny end-to-end training check:

```bash
python -m waypoint_maddpg_v0.train --smoke --device cpu
```

Start the default experiment:

```bash
python -m waypoint_maddpg_v0.train --total-steps 200000
```

Warm-start a curriculum fine-tune while expanding one random obstacle from the
default lanes to `|y| <= 3.5` m:

```bash
python -m waypoint_maddpg_v0.train \
  --init-checkpoint waypoint_maddpg_v0/runs/<run>/latest_model.pt \
  --total-steps 100000 --warmup-steps 2000 \
  --actor-lr 1e-4 --critic-lr 2e-4 \
  --initial-epsilon 0.25 --final-epsilon 0.05 \
  --epsilon-decay-steps 60000 \
  --max-obstacle-abs-y 3.5 --curriculum-steps 80000 \
  --shared-actor --device cuda
```

Use three internal rays per output sector without changing the model input:

```bash
python -m waypoint_maddpg_v0.train --sim-rays 108 --total-steps 200000
```

Evaluate a checkpoint:

```bash
python -m waypoint_maddpg_v0.evaluate \
  waypoint_maddpg_v0/runs/<run>/best_model.pt --episodes 50
```

Training outputs are written only below `waypoint_maddpg_v0/runs/`.
