s=$(dcgmi group -c SingleGPU | tr -dc '0-9')
group_id=${s: -1}
dcgmi group -g $group_id -a $0
dcgmi policy --reg -g $group_id &
while true; do sleep 1 && nvidia-smi | grep -E "Segmentation fault|No devices were found|Unable to determine the device handle for" --id=0; done &
sleep infinity
