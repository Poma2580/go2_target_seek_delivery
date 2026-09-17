"""Run one reproducible seed through the one-to-two-obstacle curriculum."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_SEED = 27


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stage1-min-steps", type=int, default=200_000)
    parser.add_argument("--stage1-steps", type=int, default=250_000)
    parser.add_argument("--stage2-steps", type=int, default=400_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Forwarded test/debug option; full experiments should leave logging enabled.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run both stages with train.py's tiny smoke-test settings.",
    )
    return parser.parse_args()


def run_stage(common, run_dir, extra, smoke):
    command = [
        sys.executable,
        "-m",
        "waypoint_maddpg_v0.train",
        *common,
        "--run-dir",
        str(run_dir),
        *extra,
    ]
    if smoke:
        command.append("--smoke")
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    if args.stage1_min_steps < 0:
        raise ValueError("--stage1-min-steps must be nonnegative")
    if args.stage1_steps < args.stage1_min_steps:
        raise ValueError("--stage1-steps must be at least --stage1-min-steps")
    if args.stage2_steps <= 0:
        raise ValueError("--stage2-steps must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        Path(__file__).resolve().parent
        / "runs"
        / f"single_seed_obstacle_curriculum_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)

    for seed in (args.seed,):
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        stage1_dir = seed_root / "stage1_one_obstacle"
        stage2_dir = seed_root / "stage2_two_obstacles"
        common = [
            "--seed",
            str(seed),
            "--eval-seed",
            str(args.eval_seed),
            "--eval-interval",
            str(args.eval_interval),
            "--eval-episodes",
            str(args.eval_episodes),
            "--sim-rays",
            "108",
            "--obstacle-y-min",
            "-3.5",
            "--obstacle-y-max",
            "3.5",
            "--shared-actor",
            "--adjacent-only-mask",
            "--direct-offset-formation-reward",
            "--empty-episode-probability",
            "0.10",
            "--clear-default-offset-weight",
            "1.20",
            "--blocked-default-offset-weight",
            "0.15",
            "--switch-weight",
            "1.00",
            "--oscillation-weight",
            "1.50",
            "--over-avoidance-weight",
            "1.00",
            "--extreme-avoidance-weight",
            "3.00",
            "--device",
            args.device,
        ]
        if args.no_tensorboard:
            common.append("--no-tensorboard")

        # Stage 1 preserves the original from-scratch hyperparameters and
        # learns the task with one randomly positioned square obstacle.
        run_stage(
            common,
            stage1_dir,
            [
                "--obstacle-count",
                "1",
                "--total-steps",
                str(args.stage1_steps),
                "--warmup-steps",
                "5000",
                "--actor-lr",
                "3e-4",
                "--critic-lr",
                "5e-4",
                "--initial-epsilon",
                "0.9",
                "--final-epsilon",
                "0.05",
                "--epsilon-decay-steps",
                "150000",
                "--min-steps-before-stop",
                "100" if args.smoke else str(args.stage1_min_steps),
                "--early-stop-success-rate",
                "0.0" if args.smoke else "0.95",
                "--early-stop-max-collision-rate",
                "1.0" if args.smoke else "0.02",
                "--early-stop-max-pair-collision-rate",
                "1.0",
                "--early-stop-min-clear-default-rate",
                "0.0" if args.smoke else "0.98",
                "--early-stop-min-empty-success-rate",
                "0.0",
                "--early-stop-min-obstacle-success-rate",
                "0.0",
                "--early-stop-max-switch-rate",
                "1.0",
                "--early-stop-max-oscillation-rate",
                "1.0",
                "--early-stop-consecutive-evals",
                "1" if args.smoke else "3",
            ],
            args.smoke,
        )
        converged_checkpoint = stage1_dir / "converged_model.pt"
        transition_checkpoint = converged_checkpoint
        if not transition_checkpoint.exists():
            transition_checkpoint = stage1_dir / "best_model.pt"
            print(
                f"\nSeed {seed} reached the stage-1 step limit without meeting "
                "the transition criterion; continuing to stage 2 from its "
                "best stage-1 checkpoint.",
                flush=True,
            )

        # Stage 2 starts from the qualifying stage-1 policy, or the best policy
        # at the step limit, adds the other-lane circle, and fine-tunes it.
        run_stage(
            common,
            stage2_dir,
            [
                "--obstacle-count",
                "2",
                "--total-steps",
                str(args.stage2_steps),
                "--warmup-steps",
                "2000",
                "--actor-lr",
                "5e-5",
                "--critic-lr",
                "1e-4",
                "--initial-epsilon",
                "0.2",
                "--final-epsilon",
                "0.03",
                "--epsilon-decay-steps",
                "60000",
                "--init-checkpoint",
                str(transition_checkpoint),
                "--min-steps-before-stop",
                str(args.stage2_steps),
                "--early-stop-success-rate",
                "0.0" if args.smoke else "0.95",
                "--early-stop-max-collision-rate",
                "1.0" if args.smoke else "0.02",
                "--early-stop-max-pair-collision-rate",
                "1.0" if args.smoke else "0.0",
                "--early-stop-min-clear-default-rate",
                "0.0" if args.smoke else "0.98",
                "--early-stop-min-empty-success-rate",
                "0.0" if args.smoke else "0.95",
                "--early-stop-min-obstacle-success-rate",
                "0.0" if args.smoke else "0.95",
                "--early-stop-max-switch-rate",
                "1.0" if args.smoke else "0.05",
                "--early-stop-max-oscillation-rate",
                "1.0" if args.smoke else "0.01",
                "--early-stop-consecutive-evals",
                "1",
            ],
            args.smoke,
        )

    print(f"\nCurriculum complete: {output_root}")
    print(f"TensorBoard: tensorboard --logdir {output_root} --port 6006")


if __name__ == "__main__":
    main()
