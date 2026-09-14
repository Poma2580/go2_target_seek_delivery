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
- Recommended RL-first runs impose no obstacle-aware action filtering: the
  policy must learn when to hold, detour, and return from observation/reward.
- High-level decision period: 1 s.
- Nav2 proxy update period: 0.1 s.
- Follower maximum speed: 0.15 m/s.
- Leader pretraining speed: 0.10 m/s, providing recovery margin after a detour.
- Episode layout: 10% completely obstacle-free episodes; otherwise a clear
  approach, obstacle avoidance, then recovery in the default formation.
- Curriculum obstacles: stage 1 has one random axis-aligned 1.5 x 1.5 m box;
  stage 2 adds a random radius-1 m circle in the opposite follower lane.
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
4. Formation preservation in the current curriculum: `-1.20` per metre of
   lateral offset while the default corridor is clear and `-0.15` while it is
   blocked, `-1.00` per action change, `-1.50` for a two-step oscillation such
   as `2->3->2`, and `-0.50` for reversing the detour side while blocked.
5. Forward progress: up to `+0.50` per decision to prevent standing still.

Continuous proximity penalties are clipped before squaring. Success requires
both followers to learn to return to their default slots; there is no forced
return action or separate rejoin bonus.

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
python -m waypoint_maddpg_v0.train --total-steps 200000 \
  --adjacent-only-mask --direct-offset-formation-reward
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

## Reproducible convergence experiment

The convergence experiment uses three independent training seeds (`7`, `17`,
`27`). Each seed trains from scratch with one random square obstacle and then
fine-tunes its own best checkpoint with the random square plus random circle.
Both stages mix in 10% fully clear episodes so that returning to and holding
action 2 is learned explicitly instead of being imposed by a rule.
The 83-D observation, five actions, safety reward weights, 108-to-36-ray
preprocessing, and shared-actor settings are retained.

New runs use an RL-first action interface: the only action mask is adjacency.
Obstacle blockage no longer filters candidates, clear paths no longer force
action 2, and recovery no longer forces a step toward action 2. The policy sees
endpoint/path clearance and the blocked feature, then learns avoidance and
recovery from reward. Formation cost is proportional to the selected lateral
offset rather than distance beyond an oracle "nearest safe" action. Legacy
checkpoints retain their original heuristic-mask behavior through metadata
defaults. Every stage evaluates at step 0 before learning for a visible baseline.

Stage 1 trains for at most 150000 steps. It moves to stage 2 early only after
three consecutive fixed-set evaluations each reach at least 95% success, at
most 2% collisions, and at least 98% action-2 choices whenever the default
corridor is clear. A qualifying checkpoint is saved as `converged_model.pt`.
If the criterion is still unmet at 150000 steps, the seed moves to stage 2
anyway and initializes it from the best stage-1 checkpoint. Stage 2 always
trains for 200000 steps; its convergence status is evaluated at the final step.

Install TensorBoard once, then launch all six sequential training stages:

```bash
python3 -m pip install tensorboard
python3 -m waypoint_maddpg_v0.train_curriculum --device cuda
```

For every stage, TensorBoard records episode return, 50-episode moving reward,
success/collision rates, all five reward components, actor/critic losses,
exploration schedules, deterministic evaluation return and success rate,
clear-lane action-2 rate, switch/oscillation rates, clear/obstacle subset
success, clearances, action fractions, and evaluation episode length. Evaluation always
uses the same 100 seeds starting at seed 10000. An event directory, CSV files,
and a checkpoint are saved every 10000 steps.

Open all seeds and stages together with the command printed by the launcher:

```bash
tensorboard --logdir waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_<time> --port 6006
```

Run a short end-to-end curriculum check with:

```bash
python3 -m waypoint_maddpg_v0.train_curriculum --smoke --device cpu
```

## Gazebo online fine-tuning

The Gazebo fine-tuner warm-starts the current seed-27 stage-2 checkpoint and
uses the exact deployment `/scan` and `/odom` preprocessing.  Each episode
spawns one static red box, with independently randomized length and width in
`[0.8, 1.2]` m, 6--9 m ahead of the elected leader.  Its lateral position
randomly blocks the default lane of either follower.  The box is deleted and
respawned after success, collision, pair collision, or 130 decisions.

The fine-tune launcher fixes `go2_1` as leader.  City spawn poses therefore
start directly in the policy formation: `go2_1=(-15,4)`,
`go2_2=(-13,6)`, and `go2_3=(-13,2)`, all at yaw zero.  The two follower
commands are role-aware limited to 0.20 m/s.  Fine-tuning retains all reward
coefficients stored in the seed-27 checkpoint; only the follower motion scale
is changed from 0.15 to 0.20 m/s and the Gazebo stage uses one obstacle.

Build the ROS package after changing the interface:

```bash
cd go2_ws_v2
colcon build --packages-select go2_mapping_nav
source install/setup.bash
```

Start the two-stage headless curriculum (recommended):

```bash
./Scripts/start_three_go2_gazebo_finetune.sh
```

The default curriculum first runs 10,000 completely clear Gazebo steps, then
continues the same policy for 30,000 steps with one randomized box per episode.
Stage-1 replay remains in the 50,000-transition buffer during stage 2 to reduce
clear-lane forgetting.  Both stages use a one-second Nav2 goal refresh.

Use `--gui` to inspect the curriculum, or `--check` to validate the derived
launcher without starting Gazebo.  Checkpoints, `config.json`, and
`episodes.csv` are written below
`waypoint_maddpg_v0/runs/gazebo_finetune_seed27_<time>/`.  The stage boundary
saves `stage1_clear_model.pt`; completion saves
`stage2_one_obstacle_final_model.pt`.  Successful one-obstacle episodes also
compete for the deployment-oriented `best_model.pt`.  Interrupting with Ctrl-C
saves `latest_model.pt`.
