#!/bin/bash

NUM_EXP=${1:-1} # Default to 1 if not provided

echo "Running all specified experiments with NUM_EXP=$NUM_EXP"

# Array of scripts to run
scripts=(
  "autorun_er_ad_reciprocators.sh"
  "autorun_er_reciprocators.sh"
  "autorun_ipd_ad_reciprocators.sh"
  "autorun_ipd_reciprocators.sh"
  "autorun_staghunt_ad_lio.sh"
  "autorun_staghunt_ad_reciprocators.sh"
  "autorun_staghunt_lio.sh"
  "autorun_staghunt_reciprocators.sh"
)

# Navigate to scripts directory to ensure relative paths work
cd "$(dirname "$0")"

# Loop through each script
for script in "${scripts[@]}"; do
  log_file="${script%.sh}.log"
  echo "Executing $script ... Output will be saved to $log_file"
  
  # Run the script sequentially and redirect stdout and stderr to the log file
  bash "$script" "$NUM_EXP" > "$log_file" 2>&1
  
  echo "Completed $script."
done

echo "All specified scripts have been executed."
