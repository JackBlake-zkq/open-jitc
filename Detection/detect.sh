dcgmi policy --set 0,0 -e -p -n -x

for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    s=$(dcgmi group -c SingleGPU | tr -dc '0-9')
    group_id=${s: -1}
    dcgmi group -g $group_id -a $gpu_id
    dcgmi policy --reg -g $group_id | python failure_notifier.py $gpu_id &
    while true; do sleep 1 && nvidia-smi --id=0; done | python failure_notifier.py $gpu_id &
done
