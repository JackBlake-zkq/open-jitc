import argparse
import os
import time
import signal
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import psutil
from collections import deque
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import model as mdl  # Your VGG11
import socket
import subprocess
import select
import shutil
import sys
from copy import deepcopy

# print("GIL Enabled:", sys._is_gil_enabled())

# --- Configuration ---
torch.set_num_threads(4)
seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)
log_iter = 20
device = torch.device("cpu")  # Will be overridden per rank
jit_checkpoint_dir = "/tmp/jit_checkpoints"

os.makedirs("output", exist_ok=True)
os.makedirs(jit_checkpoint_dir, exist_ok=True)

app_log_path = "/tmp/app_0.log"
app_log_file = open(app_log_path, "w")

stop = False
in_opt_step = False
watchdog_thread = None
addrs = []
connections = []
raw_model, optimizer, epoch, batch_idx, ddp_model = None, None, 0, 0, None

# --- Globals for signal handling ---

# def handle_sigusr1(signum, frame):
#     time.sleep(9999)

# signal.signal(signal.SIGUSR1, handle_sigusr1)

def forcibly_kill_process():
    os.kill(os.getpid(), signal.SIGKILL)

def handle_failure():
    global stop, app_log_file
    app_log_file.write("failed\n")
    app_log_file.flush()
    stop = True
    if not in_opt_step:
        # os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(1)
        checkpoint_state()


def master_send_failure_to_clients(skip=[]):
    global connections
    for conn in connections:
        if conn in skip:
            continue
        conn.sendall("failed".encode('utf-8'))
    print("Master sent failure to clients")

def master_recv_and_forward_failures():
    global connections, stop
    ready, _, _ = select.select(connections, [], [], 1)
    for conn in ready:
        try:
            data = conn.recv(1024).decode('utf-8')
        except:
            print("Connection closed")
            connections.remove(conn)
            continue
        if "failed" in data:
            print(f"Master received failure signal: {data}")
            master_send_failure_to_clients(skip=[conn])
            handle_failure()
        
def send_failure_to_master():
    global client_socket
    if client_socket:
        client_socket.sendall("failed".encode('utf-8'))
        print("Client sent failure to master")


def recv_failure_from_master():
    global client_socket, stop
    ready, _, _ = select.select([client_socket], [], [], 1)
    if ready:
        data = ready[0].recv(1024).decode('utf-8')
        if "failed" in data:
            handle_failure()


# --- Watchdog ---
def setup_watchdog(stop_event, rank):
    def watchdog():
        global client_socket, connections
        while True:
            if stop_event.is_set():
                print("Stop event set!")
                if rank == 0:
                    master_send_failure_to_clients()
                    forcibly_kill_process()
                else:
                    send_failure_to_master()
                    forcibly_kill_process()
            if rank == 0:
                master_recv_and_forward_failures()
            else:
                recv_failure_from_master()
            time.sleep(0.1)
        # with open(f'/tmp/interceptor_0.log', 'r') as f:
        #     for line in f:
        #         if "Allreduce hang detected" in line:
        #             print("Allreduce hang detected")
        #             checkpoint_state()
        #             forcibly_kill_process()

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    return watchdog_thread

def master_consolidate_checkpoints():
    global jit_checkpoint_dir, addrs
    print("Consolidating checkpoints")
    newest_path = f"{jit_checkpoint_dir}/newest.cp"
    if(os.path.exists(newest_path)):
        os.remove(newest_path)
    for i,addr in enumerate(addrs):
        try:
            subprocess.run(['scp', f'{addr[0]}:{jit_checkpoint_dir}/jit.cp', f'{jit_checkpoint_dir}/jit_{i}.cp'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during scp: {e}")
    print("Got files from other ranks")

    checkpoint_fnames = os.listdir(jit_checkpoint_dir)
    newest_name = checkpoint_fnames[0]
    newest = torch.load(f"{jit_checkpoint_dir}/{newest_name}")
    for fname in checkpoint_fnames:
        checkpoint = torch.load(f"{jit_checkpoint_dir}/{fname}")
        if checkpoint['epoch'] > newest['epoch'] or checkpoint['batch_idx'] > newest['batch_idx']:
            newest_name = fname
            newest = checkpoint
    newest_path = f"{jit_checkpoint_dir}/newest.cp"
    print(f"Best checkpoint: {newest_name}")
    shutil.copy(f"{jit_checkpoint_dir}/{newest_name}", newest_path)
    print("Found best checkpoint")

    for addr in addrs:
        try:
            subprocess.run(['scp', newest_path, f'{addr[0]}:{newest_path}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during scp: {e}")

    print("Sent best checkpoint to other ranks")
    

def checkpoint_state():
    global optimizer, epoch, batch_idx, raw_model, ddp_model
    print("Checkpointing state")
    path = f"{jit_checkpoint_dir}/jit.cp"
    cp = deepcopy(raw_model).cpu()
    print("deep copy done")
    print(cp.state_dict())
    torch.save({
            'model_state': cp.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'batch_idx': batch_idx,
    }, path)

def recover_state():
    global jit_checkpoint_dir, raw_model, optimizer, epoch, batch_idx
    print("Recovering state")
    path = f"{jit_checkpoint_dir}/newest.cp"
    while not os.path.exists(path):
        time.sleep(1)
    checkpoint = torch.load(path, map_location=device)
    raw_model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    os.remove(path)
    print("Recovered state")
    return checkpoint['epoch'], checkpoint['batch_idx']

# --- Training ---
def train_model(model, train_loader, optimizer, criterion, epoch, rank, watchdog_stop_event, sampler):
    model.train()
    start_time = time.time()
    log_iter_start = time.time()
    global args, stop, in_opt_step, watchdog_thread
    try:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= stop_iter:
                break
            if stop:
                checkpoint_state()
                sys.exit(1)

            in_opt_step = False

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            
            if args.error_before_opt_step and batch_idx == 20 and epoch == 0:
                raise RuntimeError("Simulated error before all_reduce")

            optimizer.zero_grad()

            loss.backward()

            in_opt_step = True

            if stop:
                watchdog_thread.join()
                sys.exit(1)

            optimizer.step()

            # Logging
            elapsed_time = time.time() - start_time
            start_time = time.time()

            if (batch_idx + 1) % log_iter == 0 and rank == 0:
                iter_time = time.time() - log_iter_start
                print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx + 1} | Loss {loss.item():.4f} | Time {iter_time:.2f}")
                log_iter_start = time.time()
    except BaseException as e:
        watchdog_stop_event.set()
        print("Caught exception:", e)


# --- Testing ---
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print(f"Test set: Average loss {test_loss/len(test_loader):.4f}, Accuracy {correct}/{len(test_loader.dataset)}")

# --- DDP Environment Setup ---
def init_process(master_ip, rank, size, backend='nccl'):
    dist.init_process_group(backend, init_method=f"tcp://{master_ip}:6585", rank=rank, world_size=size)

def run(rank, size, from_checkpoint):
    print("Using Checkpoint" if from_checkpoint else "Not using Checkpoint")
    global device, addrs, checkpointer, raw_model, ddp_model, optimizer, epoch, batch_idx, watchdog_thread
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26]),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26]),
        ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    raw_model = mdl.VGG11().to(device)
    ddp_model = DDP(raw_model, device_ids=[0] if torch.cuda.is_available() else None)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    if from_checkpoint:
        if rank == 0:
            master_consolidate_checkpoints()
        epoch, batch_idx = recover_state()

    watchdog_stop_event = threading.Event()
    watchdog_thread = setup_watchdog(watchdog_stop_event, rank)
    sampler.set_epoch(epoch)

    for epoch in range(num_epochs):
        train_model(ddp_model, train_loader, optimizer, criterion, epoch, rank, watchdog_stop_event, sampler)

    if rank == 0:
        test_model(ddp_model, test_loader, criterion)

# --- Main Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', type=str, default='127.0.0.1')
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--stop_iter', type=int, default=40)
    parser.add_argument('--total_batch_size', type=int, default=256)
    parser.add_argument('--error_before_opt_step', action='store_true', default=False, help='Simulate error before optimizer step')
    parser.add_argument('--all_reduce_timeout', type=int, default=10, help='Timeout for a single batch in seconds')
    parser.add_argument('--from_checkpoint', action='store_true', default=False, help='Load from checkpoint')
    args = parser.parse_args()

    client_socket = None
    addrs = []
    if args.rank == 0:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((args.master_ip, 18080))
        server_socket.listen(args.num_nodes - 1)
        connections = []
        for i in range(1, args.num_nodes):
            print(f"Waiting for connection from node {i}...")   
            client_socket, addr = server_socket.accept()
            print(f"Connected to node {i} at {addr}")
            addrs.append(addr)
            connections.append(client_socket)
            # client_socket.setblocking(False)
    else:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to master at {args.master_ip}...")
        client_socket.connect((args.master_ip, 18080))
        print(f"Connected to master at {args.master_ip}")
        client_socket.setblocking(False)


    global batch_size, num_epochs, stop_iter, log_file_name
    batch_size = args.total_batch_size // args.num_nodes
    num_epochs = args.epoch
    stop_iter = args.stop_iter
    log_file_name = f"timelog_{num_epochs}_{stop_iter}_{args.num_nodes}_{batch_size}.csv"

    with open(f'output/{log_file_name}', 'w') as f:
        f.write("epoch,iteration,elapsed_time\n")

    init_process(args.master_ip, args.rank, args.num_nodes)
    run(args.rank, args.num_nodes, args.from_checkpoint)
