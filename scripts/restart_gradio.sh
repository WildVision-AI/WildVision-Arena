#!/bin/bash
ELO_DIR="/home/yuchenl/Arena-Elo"
PKL_FILE="$ELO_DIR/results/20240702/elo_results.pkl"
CSV_FILE="$ELO_DIR/results/20240702/leaderboard.csv"

FIRST_TIME=true
# Infinite loop to restart the command every 10 minutes
while true; do
    if [ "$FIRST_TIME" = true ]; then
        # Start the Python command in the background
        # python -m arena.serve.gradio_web_server_multi --share --controller-url http://0.0.0.0:8888 --elo-results-file $PKL_FILE --leaderboard-table-file $CSV_FILE &
        python -m arena.serve.gradio_web_server_multi --share --port 5679 --controller-url http://0.0.0.0:8888 --elo-results-file $PKL_FILE --leaderboard-table-file $CSV_FILE  &
        
        # Get the PID of the last background process
        PID=$!
        FIRST_TIME=false
        echo "PID: $PID"
        # Wait for 1 hour
        sleep 10800
    else
        # Start the Python command in the background
        # python -m arena.serve.gradio_web_server_multi --share --controller-url http://0.0.0.0:8888 --elo-results-file $PKL_FILE --leaderboard-table-file $CSV_FILE &
        python -m arena.serve.gradio_web_server_multi --share --port 5678 --controller-url http://0.0.0.0:8888 --elo-results-file $PKL_FILE --leaderboard-table-file $CSV_FILE  &
        
        # Get the PID of the last background process
        PID_NEW=$!
        sleep 90
        echo "Killing OLD  PID: $PID"
        # Kill the Python command
        kill $PID 
        PID=$PID_NEW
        sleep 10800
    fi 
done
