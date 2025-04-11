s=$(dcgmi group -c SingleGPU | tr -dc '0-9')
group_id=${s: -1}
dcgmi group -g $group_id -a $0
dcgmi policy --reg -g $group_id &
while true; do sleep 1 && nvidia-smi --id=$0 | grep -E "Segmentation fault|No devices were found|Unable to determine the device handle for"; done &
sleep infinity
