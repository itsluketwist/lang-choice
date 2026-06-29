#!/usr/bin/env bash

echo
echo ">>> Submitting experiment jobs to slurm >>>"

###############################################################################
#                              experiment configuration                        #
###############################################################################

# models to evaluate (keys from config/models.yaml)
models=(
    # deepseek
    "deepseek-flash"
    "deepseek-pro"
    # mistral
    "mistral-medium"
    "mistral-small"
    "ministral-14b"
    "codestral"
    # openai
    "gpt-5-4-mini"
    "gpt-5-4"
    # anthropic
    "claude-sonnet"
    "claude-haiku"
    # google
    "gemini-3.5-flash"
    "gemini-3.1-flash-lite"
    # internal
    "kimi-k26"
    "kimi-k27-code"
    "gemma4-26b"
    "gemma4-e4b"
    # olmo
    "olmo-7b"
    "olmo-32b"
    # nvidia nemotron
    "nemotron-7b"
    "nemotron-32b"
    "code-nemotron-7b"
    "code-nemotron-32b"
    # qwen (huggingface)
    "qwen3-8b"
    "qwen3-14b"
    "qwen3-32b"
)

# inference preset (key from config/inference.yaml)
inference="default"

# context conditions for implementation prompts
# recommendation prompts always use no context
context_conditions=(
    "none"
    # "neutral"
    # "python"
    # "non_python"
    # "contradictory"
)

# run mode: default | overwrite | update | evaluate
mode="default"

###############################################################################
#                              submit jobs                                     #
###############################################################################

echo "Models:     ${models[*]}"
echo "Inference:  $inference"
echo "Contexts:   ${context_conditions[*]}"
echo "Mode:       $mode"
echo

for model in "${models[@]}"; do
    for context in "${context_conditions[@]}"; do
        echo "Submitting: $model | context=$context"

        sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=cc-${model}-${context}
#SBATCH --output=/users/%u/code/code-choice/logs/${model}/${context}-%j.out
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=04:00:00

source ./scripts/setup_job.sh

run \
    --model $model \
    --inference $inference \
    --context $context \
    --mode $mode

echo "Done: $model | context=$context"
EOF

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done
done

echo
echo "All jobs submitted!"
echo
