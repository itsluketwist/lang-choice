"""The command line interface entry-point for the experiment pipeline.

Usage:
    run --model deepseek-r1 \\
        --model-config config/models.yaml \\
        --inference greedy \\
        --inference-config config/inference.yaml \\
        [--context none] \\
        [--mode default] \\
        [--debug]

The run command is wired in pyproject.toml as run = "src.cli:main".
All pipeline logic lives in src/run.py.
"""

import argparse

from src.run import run_experiment


def _parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a language-choice reasoning experiment.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model key from the model config YAML.",
    )
    parser.add_argument(
        "-mc",
        "--model-config",
        default="config/models.yaml",
        help="Path to the model config YAML (default: config/models.yaml).",
    )
    parser.add_argument(
        "-i",
        "--inference",
        default="greedy",
        help="Inference preset key from the inference config YAML (default: greedy).",
    )
    parser.add_argument(
        "-ic",
        "--inference-config",
        default="config/inference.yaml",
        help="Path to the inference config YAML (default: config/inference.yaml).",
    )
    parser.add_argument(
        "--context",
        default="none",
        help="Prior-context condition for implementation prompts (default: none).",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "overwrite", "update", "evaluate"],
        default="default",
        help=(
            "default: generate only if the output file doesn't exist (default). "
            "overwrite: ignore existing results and regenerate everything. "
            "update: top up existing results until each prompt has the required "
            "number of valid responses. "
            "evaluate: skip generation, run evaluation/analysis on existing files only."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Limit to 2 prompts per split and write to output/debug/.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and run the experiment pipeline."""
    args = _parse_args()
    run_experiment(
        model=args.model,
        model_config=args.model_config,
        inference=args.inference,
        inference_config=args.inference_config,
        context_condition=args.context,
        mode=args.mode,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
