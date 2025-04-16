#!/bin/bash

# Arguments to be passed to the python command
model="gpt-4o"
episode=10
domains=("office")
scenes=("allensville" "shelbiana" "parole")

# Loop through the models and execute the command
for d in "${domains[@]}"; do
    for s in "${scenes[@]}"; do
        python baselines/llmgenplan.py -m "$model" -e "$episode" -d "$d" -s "$s"
    done
done
