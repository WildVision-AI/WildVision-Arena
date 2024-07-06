#!/bin/bash

# List bore processes
echo "Checking for running 'bore' processes..."
bore_processes=$(pgrep -af bore)

if [[ -z "$bore_processes" ]]; then
    echo "No 'bore' processes are currently running."
else
    echo "Running 'bore' processes:"
    echo "$bore_processes"
fi