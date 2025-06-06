dcgmi policy --set 0,0 -e -p -n -x
group_id=$(dcgmi group -c SingleGPU | tr -dc '0-9')
dcgmi group -g $group_id -a $1
dcgmi policy --reg -g $group_id &
while true; do sleep 1 && nvidia-smi --id=$1 | grep -E "Segmentation fault|No devices were found|Unable to determine the device handle for"; done &
sleep infinity
