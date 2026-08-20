# plot_rewards.py
"""Plot training reward curves and evaluation success-rate curves.

This script keeps the original reward plotting behavior based on agent_rewards.npy,
and adds a TensorBoard-log parser for eval/success_rate.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import utils.plotting as plotting
from utils.env import get_env_info


SUCCESS_RATE_TAG = "eval/success_rate"


def setup_chinese_font():
    """Configure Matplotlib font fallback for Chinese plot labels."""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = preferred_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def display_algorithm_name(algo_name: str) -> str:
    """Translate known algorithm display names while preserving MADDPG."""
    return {
        "MADDPG-Approx": "MADDPG-近似版",
    }.get(algo_name, algo_name)


setup_chinese_font()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot and compare algorithm results")
    parser.add_argument("--mode", type=str, choices=["single", "compare"], default="single",
                        help="Plot mode: single algorithm or comparison")
    parser.add_argument("--env-name", type=str, required=True, help="Environment name")

    parser.add_argument("--algo-name", type=str, default="MADDPG",
                        help="Name of first algorithm/run")
    parser.add_argument("--rewards-path", type=str,
                        help="Path to first algorithm agent_rewards.npy")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory for the first run. If omitted, dirname(--rewards-path) is used.")

    parser.add_argument("--algo2-name", type=str, default="MADDPG-近似版",
                        help="Name of second algorithm/run for compare mode")
    parser.add_argument("--rewards2-path", type=str,
                        help="Path to second algorithm agent_rewards.npy for compare mode")
    parser.add_argument("--log2-dir", type=str, default=None,
                        help="TensorBoard log directory for the second run. If omitted, dirname(--rewards2-path) is used.")

    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for running average of reward curves")
    parser.add_argument("--success-window-size", type=int, default=1,
                        help="Window size for smoothing eval success-rate curves. Use 1 to keep raw eval curve.")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Directory to save plots")
    parser.add_argument("--target-score", type=int, default=None,
                        help="Target score for single mode reward plot")
    parser.add_argument("--no-success-plot", action="store_true",
                        help="Disable plotting eval/success_rate from TensorBoard logs")
    return parser.parse_args()


def infer_log_dir(rewards_path: Optional[str], log_dir: Optional[str]) -> Optional[str]:
    if log_dir:
        return log_dir
    if rewards_path:
        return os.path.dirname(os.path.abspath(rewards_path))
    return None


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or len(values) == 0:
        return values
    window_size = min(window_size, len(values))
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(values, kernel, mode="valid")


def find_event_files(log_dir: str) -> List[str]:
    event_files: List[str] = []
    if not log_dir or not os.path.isdir(log_dir):
        return event_files
    for root, _, files in os.walk(log_dir):
        for filename in files:
            if filename.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, filename))
    return sorted(event_files)


def load_success_rate_from_tensorboard(log_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load eval/success_rate scalars from TensorBoard event files.

    Returns:
        steps: shape [N]
        values: shape [N], converted to percent, i.e. 0.86 -> 86.0
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is required to read eval/success_rate from event files. "
            "Install it with: pip install tensorboard"
        ) from exc

    event_files = find_event_files(log_dir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")

    all_points: List[Tuple[int, float]] = []
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        if SUCCESS_RATE_TAG not in ea.Tags().get("scalars", []):
            continue
        for scalar_event in ea.Scalars(SUCCESS_RATE_TAG):
            all_points.append((int(scalar_event.step), float(scalar_event.value) * 100.0))

    if not all_points:
        raise ValueError(
            f"Found event files under {log_dir}, but none contains scalar tag '{SUCCESS_RATE_TAG}'."
        )

    # If the same step appears more than once, keep the last value.
    merged: Dict[int, float] = {}
    for step, value in sorted(all_points, key=lambda x: x[0]):
        merged[step] = value

    steps = np.array(sorted(merged.keys()), dtype=np.int64)
    values = np.array([merged[int(step)] for step in steps], dtype=np.float64)
    return steps, values


def plot_success_rate_single(log_dir: str, output_dir: str, algo_name: str, success_window_size: int = 1):
    try:
        steps, success_rates = load_success_rate_from_tensorboard(log_dir)
    except Exception as exc:
        print(f"[Warning] Could not plot success rate for {algo_name}: {exc}")
        return None

    plot_steps = steps
    plot_values = moving_average(success_rates, success_window_size)
    if success_window_size > 1 and len(success_rates) >= success_window_size:
        plot_steps = steps[success_window_size - 1:]

    plt.figure(figsize=(8, 5), dpi=150)
    display_name = display_algorithm_name(algo_name)
    plt.plot(plot_steps, plot_values, label=display_name)
    plt.xlabel("训练时间步")
    plt.ylabel("评估成功率（%）")
    plt.title(f"{display_name} 评估成功率")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "eval_success_rate.png")
    plt.savefig(output_path)
    plt.close()

    np.save(os.path.join(output_dir, "eval_success_rate.npy"),
            {"steps": steps, "success_rate_percent": success_rates, "log_dir": log_dir})
    print(f"Success-rate plot saved to: {output_path}")
    return output_path


def plot_success_rate_compare(algo_log_dirs: Dict[str, str], output_dir: str, success_window_size: int = 1):
    loaded = {}
    for algo_name, log_dir in algo_log_dirs.items():
        try:
            loaded[algo_name] = load_success_rate_from_tensorboard(log_dir)
        except Exception as exc:
            print(f"[Warning] Could not load success rate for {algo_name}: {exc}")

    if not loaded:
        print("[Warning] No success-rate data loaded; skip compare success-rate plot.")
        return None

    plt.figure(figsize=(8, 5), dpi=150)
    raw_data = {}
    for algo_name, (steps, values) in loaded.items():
        plot_steps = steps
        plot_values = moving_average(values, success_window_size)
        if success_window_size > 1 and len(values) >= success_window_size:
            plot_steps = steps[success_window_size - 1:]
        plt.plot(plot_steps, plot_values, label=display_algorithm_name(algo_name))
        raw_data[algo_name] = {
            "steps": steps,
            "success_rate_percent": values,
            "log_dir": algo_log_dirs[algo_name],
        }

    plt.xlabel("训练时间步")
    plt.ylabel("评估成功率（%）")
    plt.title("评估成功率对比")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "eval_success_rate_compare.png")
    plt.savefig(output_path)
    plt.close()

    np.save(os.path.join(output_dir, "eval_success_rate_compare.npy"), raw_data)
    print(f"Success-rate comparison plot saved to: {output_path}")
    return output_path


def main():
    args = parse_args()

    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        apply_padding=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "single":
        if not args.rewards_path:
            raise ValueError("Single mode requires --rewards-path")
        if not os.path.exists(args.rewards_path):
            raise ValueError(f"Rewards file not found: {args.rewards_path}")

        output_dir = f"{args.output_dir}/{args.env_name}/{args.algo_name}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Original reward plot: preserved.
        agent_rewards = np.load(args.rewards_path, allow_pickle=True)
        plotting.plot_rewards_single_env(
            agents,
            agent_rewards,
            output_dir,
            env_name=args.env_name,
            target_score=args.target_score,
            window_size=args.window_size,
        )

        log_dir = infer_log_dir(args.rewards_path, args.log_dir)
        if not args.no_success_plot and log_dir:
            plot_success_rate_single(
                log_dir=log_dir,
                output_dir=output_dir,
                algo_name=args.algo_name,
                success_window_size=args.success_window_size,
            )

        config_info = {
            "env_name": args.env_name,
            "algo_name": args.algo_name,
            "rewards_path": args.rewards_path,
            "log_dir": log_dir,
            "window_size": args.window_size,
            "success_window_size": args.success_window_size,
            "target_score": args.target_score,
            "num_agents": num_agents,
            "timestamp": timestamp,
        }
        np.save(os.path.join(output_dir, "plot_config.npy"), config_info)
        print(f"Single mode plots saved to: {output_dir}")

    elif args.mode == "compare":
        if not args.rewards_path or not args.rewards2_path:
            raise ValueError("Compare mode requires --rewards-path and --rewards2-path")
        for path in [args.rewards_path, args.rewards2_path]:
            if not os.path.exists(path):
                raise ValueError(f"Rewards file not found: {path}")

        output_dir = f"{args.output_dir}/compare/{args.env_name}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Original reward comparison plot: preserved.
        algo_paths = {
            args.algo_name: args.rewards_path,
            args.algo2_name: args.rewards2_path,
        }
        plotting.compare_algorithms(args.env_name, algo_paths, output_dir, window_size=args.window_size)

        log_dir_1 = infer_log_dir(args.rewards_path, args.log_dir)
        log_dir_2 = infer_log_dir(args.rewards2_path, args.log2_dir)
        if not args.no_success_plot:
            algo_log_dirs = {}
            if log_dir_1:
                algo_log_dirs[args.algo_name] = log_dir_1
            if log_dir_2:
                algo_log_dirs[args.algo2_name] = log_dir_2
            plot_success_rate_compare(
                algo_log_dirs=algo_log_dirs,
                output_dir=output_dir,
                success_window_size=args.success_window_size,
            )

        compare_config = {
            "env_name": args.env_name,
            "algo1_name": args.algo_name,
            "algo1_path": args.rewards_path,
            "algo1_log_dir": log_dir_1,
            "algo2_name": args.algo2_name,
            "algo2_path": args.rewards2_path,
            "algo2_log_dir": log_dir_2,
            "window_size": args.window_size,
            "success_window_size": args.success_window_size,
            "timestamp": timestamp,
        }
        np.save(os.path.join(output_dir, "compare_config.npy"), compare_config)
        print(f"Compare mode plots saved to: {output_dir}")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
