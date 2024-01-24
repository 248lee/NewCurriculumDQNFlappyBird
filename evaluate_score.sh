#!/bin/bash

while true; do
    # Read content from last_old_time.txt
    content=$(python evaluate_score.py)

    # Append content to usage.txt along with a timestamp
    echo "$(date): $content"

    # Sleep for 10 minutes
    sleep 600
done
