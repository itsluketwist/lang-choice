#!/usr/bin/env bash

echo
echo ">>> Submitting ablation jobs to slurm >>>"

###############################################################################
#                              experiment configuration                        #
###############################################################################

# api models used for the ablations (keys from config/models.yaml).
# open-weight models are generated separately.
ablation_models=(
    # openai
    "gpt-5-4"
    "gpt-5-4-mini"
    # google
    "gemini-3.5-flash"
    "gemini-3.1-flash-lite"
    # mistral
    "mistral-small"
    "codestral"
)

# 1. python control area — adds the 12 control prompts to each existing
#    default run. this is the only block that passes --include-control.
run_control="true"

# 2. decoding sensitivity — the temperature presets to run for each model.
#    each model's existing def-* run covers the third point of the sweep.
run_temperatures="true"
temperature_runs=(
    "gpt-5-4 temp-0.3 temp-0.6"
    "gpt-5-4-mini temp-0.3 temp-0.6"
    "gemini-3.5-flash temp-0.3 temp-0.6"
    "gemini-3.1-flash-lite temp-0.3 temp-0.6"
    "mistral-small temp-0.6 temp-1.0"
    "codestral temp-0.6 temp-1.0"
)

# 3. two-stage deliberation — replays the saved recommendations as a first turn,
#    then asks for the implementation.
run_two_stage="true"

###############################################################################
#                              submit jobs                                     #
###############################################################################

# submit one job: submit <model> <job suffix> <arguments for the run command>
submit () {
    model="$1"
    suffix="$2"
    shift 2

    echo "Submitting: $model | $suffix"

    sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=cc-${model}-${suffix}
#SBATCH --output=/users/%u/code/lang-choice/logs/${model}/${suffix}-%j.out
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=04:00:00

source ./scripts/setup_job.sh

run --model $model $@

echo "Done: $model | $suffix"
EOF

    # small delay between submissions to avoid scheduler overload
    sleep 0.5
}

echo "Models:     ${ablation_models[*]}"
echo "Control:    $run_control"
echo "Sweep:      $run_temperatures"
echo "Two-stage:  $run_two_stage"
echo

# --- 1. python control area ---------------------------------------------------
# update mode only generates the prompts that are missing, i.e. the control ones
if [ "$run_control" = "true" ]; then
    for model in "${ablation_models[@]}"; do
        submit "$model" "control" --inference default --mode update --include-control
    done
fi

# --- 2. decoding sensitivity --------------------------------------------------
if [ "$run_temperatures" = "true" ]; then
    for entry in "${temperature_runs[@]}"; do
        read -ra parts <<< "$entry"
        model="${parts[0]}"
        for preset in "${parts[@]:1}"; do
            submit "$model" "$preset" --inference "$preset" --mode default
        done
    done
fi

# --- 3. two-stage deliberation ------------------------------------------------
if [ "$run_two_stage" = "true" ]; then
    for model in "${ablation_models[@]}"; do
        submit "$model" "two-stage" --inference default --two-stage --mode default
    done
fi

echo
echo "All jobs submitted!"
echo
