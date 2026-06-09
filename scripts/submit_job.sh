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
    # moonshot ai
    "kimi-thinking"
    "kimi-k26"
    # mistral
    "mistral-medium"
    "mistral-small"
    # openai
    "gpt-54"
    "o4-mini"
    # anthropic
    "claude-sonnet"
    "claude-haiku"
    # google
    "gemini-flash"
    "gemma-31b"
    # qwen
    "qwen37-max"
    "qwen3-6-27b"
    # nvidia
    "nemotron-ultra"
    "nemotron-nano"
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

# force regeneration even if output files already exist
update=false

###############################################################################
#                              submit jobs                                     #
###############################################################################

update_flag=""
if [ "$update" = true ]; then
    update_flag="--update"
fi

echo "Models:     ${models[*]}"
echo "Inference:  $inference"
echo "Contexts:   ${context_conditions[*]}"
echo "Update:     $update"
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
    $update_flag

echo "Done: $model | context=$context"
EOF

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done
done

echo
echo "All jobs submitted!"
echo
