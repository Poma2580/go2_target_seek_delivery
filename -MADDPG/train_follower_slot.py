"""
Convenience launcher for the Go2 follower-slot MARL task.

This trains only the two followers. The leader is controlled inside the
environment by a fixed pedestrian-following rule.
"""

from train import parse_args, train
import sys


def main():
    args = parse_args()
    args.env_name = "follower_slot_tracking_v0"
    if not any(arg == "--training-stage" or arg.startswith("--training-stage=") for arg in sys.argv):
        args.training_stage = 1
    if args.total_timesteps == int(2e6):
        args.total_timesteps = int(8e5)
    if args.max_steps == 100:
        args.max_steps = 180
    if args.warmup_steps == 50000:
        args.warmup_steps = 10000
    if args.noise_scale == 0.4:
        args.noise_scale = 0.25
    if args.min_noise == 0.05:
        args.min_noise = 0.03
    if args.eval_interval == 10000:
        args.eval_interval = 10000
    train(args)


if __name__ == "__main__":
    main()
