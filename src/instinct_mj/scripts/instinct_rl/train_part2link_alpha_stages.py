"""Run staged G1 Sitting Part2Link alpha training in InstinctMJ."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TASK = "Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0"
DEFAULT_ALPHAS = "1.0,0.8,0.5,0.2,0.0"
DEFAULT_EXPERIMENT = "g1_interaction_part2link_transformer"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Sequentially train Part2Link alpha stages with one alpha value loaded per process."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--iterations-per-stage", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--initial-load-run", default=None)
    parser.add_argument("--checkpoint", default="model_.*.pt")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def _parse_alpha_list(raw: str) -> list[str]:
    alphas = [item.strip() for item in raw.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alphas must contain at least one alpha value")
    return alphas


def _log_root(experiment_name: str) -> Path:
    return Path("logs") / "instinct_rl" / experiment_name


def _latest_run(log_root: Path, before: set[Path]) -> str:
    candidates = [path for path in log_root.iterdir() if path.is_dir() and path not in before]
    if not candidates:
        raise RuntimeError(f"No new run directory was created under {log_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def _stage_env(alpha: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SITTING_PART2LINK_ALPHA_VALUES"] = alpha
    return env


def _train_command(
    args: argparse.Namespace,
    previous_run: str | None,
    passthrough: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "instinct_mj.scripts.instinct_rl.train",
        args.task,
        "--num-envs",
        str(args.num_envs),
        "--agent.max_iterations",
        str(args.iterations_per_stage),
        "--agent.experiment_name",
        str(args.experiment_name),
    ]
    if previous_run is not None:
        command += [
            "--agent.resume",
            "true",
            "--agent.load_run",
            str(previous_run),
            "--agent.load_checkpoint",
            str(args.checkpoint),
        ]
    command += passthrough
    return command


def main() -> int:
    args, passthrough = _parse_args()
    alphas = _parse_alpha_list(args.alphas)
    previous_run = args.initial_load_run
    log_root = _log_root(args.experiment_name)

    for stage_idx, alpha in enumerate(alphas, start=1):
        command = _train_command(args, previous_run, passthrough)
        before = set(log_root.iterdir()) if log_root.exists() else set()
        print(f"[stage {stage_idx}/{len(alphas)}] alpha={alpha}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, env=_stage_env(alpha), check=True)
            previous_run = _latest_run(log_root, before)
            print(f"[stage {stage_idx}/{len(alphas)}] completed: {previous_run}")
        else:
            previous_run = f"<stage_{stage_idx}_run>"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
