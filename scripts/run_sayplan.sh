#!/bin/bash

# Arguments to be passed to the python command
model="gpt-4o"
episode=50
domains=("clean" "dining" "pc" "office")
scenes=("allensville" "shelbiana" "parole")

# Loop through the models and execute the command
for d in "${domains[@]}"; do
    for s in "${scenes[@]}"; do
        python baselines/sayplan.py -m "$model" -e "$episode" -d "$d" -s "$s"
    done
done
