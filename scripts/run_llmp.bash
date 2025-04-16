#!/bin/bash

# Arguments to be passed to the python command
model="gpt-4o"
episode=50
experiment="problem"
domains=("clean" "dining" "pc" "office")
scenes=("allensville" "shelbiana" "parole")
timelimit=60

# Loop through the models and execute the command
for d in "${domains[@]}"; do
    for s in "${scenes[@]}"; do
        python delta.py -m "$model" -e "$episode" -d "$d" -s "$s" --experiment "$experiment" --max-time "$timelimit"
    done
done
