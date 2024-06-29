#!/bin/bash

FIRST_TIME=true
# Infinite loop to restart the command every 10 minutes
while true; do
    if [ "$FIRST_TIME" = true ]; then
        # Start the Python command in the background
        bash update_elo_rating_local.sh
        
        # Get the PID of the last background process
        FIRST_TIME=false
        # Wait for 1 hour
        sleep 36000
    else
        # Start the Python command in the background
        bash update_elo_rating_local.sh
        
        sleep 36000
    fi 
done
