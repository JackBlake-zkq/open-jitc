#!/bin/bash

# Ensure GPU_ID is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <GPU_ID>"
  exit 1
fi

GPU_ID=$1

# Read the PCI ID from lscpi (This is assuming there will be one "NVIDIA" line per GPU in a multi-GPU setup)
PCI_ID=$(lspci | grep -i nvidia | sed -n "${GPU_ID}p" | awk '{print $1}')

# Ensure PCI read
if [ -z "$PCI_ID" ]; then
  echo "Error: GPU_ID $GPU_ID not found."
  exit 1
fi

echo "GPU_ID: $GPU_ID"
echo "Associated PCI ID: $PCI_ID"

# Sleep for 300 seconds * (GPU_ID + 1)
SLEEP_TIME=$((300 * (GPU_ID + 1)))
echo "Sleeping for $SLEEP_TIME seconds..."
sleep "$SLEEP_TIME"

# Choose 1/3 to be recoverable and 2/3 to be unrecoverable, can make this selection more random later if needed
if (( GPU_ID % 3 != 0 )); then
  # Recoverable error using dcgmi
  echo "Simulating recoverable error..."
  dcgmi test --inject --gpuid "$GPU_ID" -f 319 -v 4
else
  # Unrecoverable error by setting command register to 0
  echo "Simulating unrecoverable error..."
  setpci -s "$PCI_ID" command=0
fi
