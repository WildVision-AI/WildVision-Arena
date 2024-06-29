#!/bin/bash

BORE_LOG_FOLDER="/home/yuchenl/BORE_LOG/"
mkdir -p $BORE_LOG_FOLDER
# Define an output file for storing the PORT mapping
output_file="${BORE_LOG_FOLDER}/bore_port_mappings.txt"
> "$output_file"  # Clear or create the file before writing to it.

# Define an associative array with your local port variables.
declare -A local_ports=(
    [COGVLM]=31006
    [MINICPM]=31007
    [LLAVA7B]=31009
    [LLAVA34B]=31008
    [BLIP]=31012
    [UFORM]=31011
    [TINYLLAVA]=31015
    [DEEPSEEK]=31010
    [BUNNY]=31013
)

# Loop through the associative array
for key in "${!local_ports[@]}"; do
    local_port="${local_ports[$key]}"
    
    # Construct the command to launch bore and route local ports, using a unique log file for each.
    bore_log_file="${BORE_LOG_FOLDER}/bore_output_${key}.log"
    # bore local $local_port --to 34.19.37.54 &> $bore_log_file &
    bore local $local_port --to 34.19.37.54 > $bore_log_file 2>&1 &
    pid=$!

    echo "Launching bore for $key with PID $pid..."
    
    # Disown the process so it’s not killed when script exits
    disown $pid
     
    sleep 5  # Wait to ensure the command has time to output.
    
    # Extract the remote port using grep and Perl-compatible regex
    if remote_port=$(awk -F "=" '/remote_port/ {print $2}' $bore_log_file | head -n 1); then
        echo "Captured remote port $remote_port for $key"
        # Write the mapping to the output file.
        echo "PORT_${key}=${remote_port}" >> $output_file
    else
        echo "Failed to capture remote port for $key"
    fi

    # Clean-up: Optionally delete the temporary log file.
    # rm -f $bore_log_file
done

echo "Port mappings have been saved to $output_file"
cat $output_file