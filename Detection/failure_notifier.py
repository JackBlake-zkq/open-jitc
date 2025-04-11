import sys
import socket
import subprocess
import torch
import os

UNRECOVERABLE_FAILURES = [
    "Segmentation fault",
    "No devices were found",
    "Unable to determine the device handle for",
]
RECOVERABLE_FAILURES = ["ECC error"]


if len(sys.argv) != 2:
    print("Usage: python failure_notifier.py <n_connections>")
    sys.exit(1)

n_connections = int(sys.argv[1])

result = subprocess.run("dcgmi policy --set 0,0 -e -p -n -x".split(" "))
if result.returncode != 0:
    print("Failed to set DCGM policy")
    sys.exit(1)


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 25565))
server_socket.listen(n_connections)
connections = []
for _ in range(n_connections):
    conn, addr = server_socket.accept()
    connections.append(conn)

num_gpus = torch.cuda.device_count()
gpu_ids = list(range(num_gpus))

procs = []
for gpu_id in gpu_ids:
    proc = subprocess.Popen(["sh", "detect_single_gpu.sh", str(gpu_id)], capture_output=True)
    procs.append(proc)

for proc in procs:
    pid = os.fork()
    if pid == 0:
        while True:
            line = proc.stdout.readline()
            if any([failure in line for failure in UNRECOVERABLE_FAILURES]) in line:
                for conn in connections:
                    conn.sendall(b'0')
                    conn.close()
            if any([failure in line for failure in RECOVERABLE_FAILURES])  in line:
                for conn in connections:
                    conn.sendall(b'1')


while all([proc.poll() is None for proc in procs]):
    pass
    
for conn in connections:
    conn.close()

server_socket.close()