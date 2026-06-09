#!/usr/bin/env bash

echo "============================================================"
echo "Setting up environment!"
echo "============================================================"
echo

echo "Running on node: $(hostname)"
echo "Date/time: $(date)"
echo
echo

# display cpu and memory info
echo "CPU model: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPU cores: $(nproc)"
echo "Total memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Available memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo

echo "Loading modules..."
module load python/3.11.6-gcc-13.2.0
echo "Modules loaded:"
module list
echo

echo "Loading virtual environment..."
source .venv/bin/activate
echo

echo "Loading environment variables from .env..."
if [ -f .env ]; then
    source .env
else
    echo "WARNING: .env file not found! Create one with OPENROUTER_API_KEY."
fi
echo

echo "Environment setup, starting job..."

echo
echo
