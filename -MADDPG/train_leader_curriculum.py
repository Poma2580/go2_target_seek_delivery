"""
Run the two-stage curriculum for leader_slot_tracking_v0.

Each stage continues from the previous stage's best_model.pt.  The environment
uses the target-free 25-dimensional observation layout across all stages.
"""

import argparse
import os
import sys

from train import parse_args as parse_train_args
from train import train


DEFAULT_STAGE_STEPS = {
    1: 300_000,
    2: 500_000,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--end-stage", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--stage-steps",
        type=str,
        default="",
        help="Comma-separated total timesteps per stage, e.g. 300000,500000",
    )
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--hidden-sizes", type=str, default="128,128")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--min-noise", type=float, default=0.03)
    parser.add_argument("--use-noise-decay", action="store_true", default=True)
    parser.add_argument("--shared-actor", action="store_true", default=False)
    parser.add_argument("--create-gif", action="store_true")
    parser.add_argument("--stage-transition-success-rate", type=float, default=0.95)
    parser.add_argument("--stage-transition-patience", type=int, default=2)
    parser.add_argument("--stage-transition-min-steps", type=int, default=3000)
    return parser.parse_args()


def make_train_args(curriculum_args, stage, total_timesteps, pretrained_model_path):
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        args = parse_train_args()
    finally:
        sys.argv = argv
    args.env_name = "leader_slot_tracking_v0"
    args.training_stage = stage
    args.total_timesteps = int(total_timesteps)
    args.batch_size = curriculum_args.batch_size
    args.max_steps = curriculum_args.max_steps
    args.warmup_steps = curriculum_args.warmup_steps
    args.eval_interval = curriculum_args.eval_interval
    args.hidden_sizes = curriculum_args.hidden_sizes
    args.actor_lr = curriculum_args.actor_lr
    args.critic_lr = curriculum_args.critic_lr
    args.noise_scale = curriculum_args.noise_scale
    args.min_noise = curriculum_args.min_noise
    args.use_noise_decay = curriculum_args.use_noise_decay
    args.shared_actor = curriculum_args.shared_actor
    args.create_gif = curriculum_args.create_gif
    args.early_stop_success_rate = (
        curriculum_args.stage_transition_success_rate
        if stage < curriculum_args.end_stage
        else None
    )
    args.early_stop_patience = curriculum_args.stage_transition_patience
    args.early_stop_min_steps = curriculum_args.stage_transition_min_steps
    args.pretrained_model_path = pretrained_model_path
    return args


def parse_stage_steps(text):
    if not text:
        return DEFAULT_STAGE_STEPS.copy()
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError("--stage-steps must contain exactly two comma-separated integers")
    return {stage: values[stage - 1] for stage in range(1, 3)}


def main():
    curriculum_args = parse_args()
    if curriculum_args.start_stage > curriculum_args.end_stage:
        raise ValueError("--start-stage must be <= --end-stage")

    stage_steps = parse_stage_steps(curriculum_args.stage_steps)
    pretrained = curriculum_args.pretrained_model_path

    for stage in range(curriculum_args.start_stage, curriculum_args.end_stage + 1):
        print(f"\n===== Leader-relative curriculum stage {stage} / {curriculum_args.end_stage} =====")
        args = make_train_args(curriculum_args, stage, stage_steps[stage], pretrained)
        _, experiment_name = train(args)
        run_dir = os.path.join("runs", args.env_name, args.algo, experiment_name)
        best_model = os.path.join(run_dir, "best_model.pt")
        final_model = os.path.join(run_dir, "model.pt")
        pretrained = best_model if os.path.exists(best_model) else final_model
        print(f"Stage {stage} finished. Next pretrained model: {pretrained}")


if __name__ == "__main__":
    main()
