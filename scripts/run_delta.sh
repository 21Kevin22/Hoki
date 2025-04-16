#!/bin/bash

# Arguments to be passed to the python command
# "gpt-35-turbo-16k", "gpt-4", "gpt-4-turbo", "gpt-4o"
# "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"
model="gpt-4o"
episode=50
domain=("clean" "dining" "pc" "office")
scene=("allensville" "shelbiana" "parole")
timelimit=60

# Loop through the models and execute the command
for d in "${domain[@]}"; do
    for s in "${scene[@]}"; do
        python delta.py -m "$model" -e "$episode" -d "$d" -s "$s" --max-time "$timelimit"
    done
done
