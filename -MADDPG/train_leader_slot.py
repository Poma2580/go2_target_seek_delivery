"""
Convenience launcher for the target-free leader-relative Go2 follower task.

This trains only go2/go3.  The learned policy does not observe the pedestrian;
it tracks slots inferred from go1 pose/yaw and formation dimensions.
"""

import sys

from train import parse_args, train


def main():
    args = parse_args()
    args.env_name = "leader_slot_tracking_v0"
    if not any(arg == "--training-stage" or arg.startswith("--training-stage=") for arg in sys.argv):
        args.training_stage = 1
    if args.total_timesteps == int(2e6):
        args.total_timesteps = int(8e5)
    if args.max_steps == 100:
        args.max_steps = 250
    if args.warmup_steps == 50000:
        args.warmup_steps = 10000
    if args.noise_scale == 0.4:
        args.noise_scale = 0.25
    if args.min_noise == 0.05:
        args.min_noise = 0.03
    train(args)


if __name__ == "__main__":
    main()
