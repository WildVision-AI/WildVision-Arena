#!/bin/bash

# Infinite loop to run the command every 10 minutes
while true; do
  # Change directory and run the python script
  python scripts/update_space_url.py
  # Wait for 30 seconds before the next run
  sleep 30 
done
