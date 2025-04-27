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

# --- Globals for signal handling ---
interrupted_by_sigusr1 = False

def handle_sigusr1(signum, frame):
        global interrupted_by_sigusr1
        interrupted_by_sigusr1 = True
        print("Received SIGUSR1, triggering checkpoint")
        raise RuntimeError("Checkpoint triggered")

signal.signal(signal.SIGUSR1, handle_sigusr1)

def notify_main_thread_of_failure():
    os.kill(os.getpid(), signal.SIGUSR1)
    time.sleep(1)
    print("Notified main thread of failure")

def master_send_failure_to_clients(skip=[]):
    global connections
    for conn in connections:
        if conn in skip:
            continue
        conn.sendall("failed".encode('utf-8'))
    print("Master sent failure signal to clients")

def master_recv_and_forward_failures():
    global connections
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
            print("Master notifying main thread of failure")
            notify_main_thread_of_failure()
        
def send_failure_to_master():
    global client_socket
    if client_socket:
        client_socket.sendall("failed".encode('utf-8'))
        print("Client sent failure signal to master")

def recv_failure_from_master():
    ready, _, _ = select.select([client_socket], [], [], 1)
    if ready:
        data = ready[0].recv(1024).decode('utf-8')
        if "failed" in data:
            notify_main_thread_of_failure()


# --- Watchdog ---
def setup_watchdog(rank):
    def watchdog():
        global client_socket, connections
        while True:
            if rank == 0:
                master_recv_and_forward_failures()
            else:
                recv_failure_from_master()
            time.sleep(1)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    return watchdog_thread

class Checkpointer:
    def __init__(self, model, cp_dir, addrs):
        self.model = model
        self.cp_dir = cp_dir
        self.addrs = addrs

    def master_consolidate_checkpoints(self):
        newest_path = f"{self.cp_dir}/newest.cp"
        os.remove(newest_path)
        for i,addr in enumerate(self.addrs):
            try:
                subprocess.run(['scp', f'{addr}:{self.cp_dir}/jit.cp', f'{self.cp_path}/jit_{i}.cp'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during scp: {e}") 

        checkpoint_fnames = os.listdir(self.cp_dir)
        newest = checkpoint_fnames[0]
        for fname in checkpoint_fnames:
            checkpoint = torch.load(f"{self.cp_dir}/{fname}")
            if checkpoint['epoch'] > newest['epoch'] or checkpoint['batch_idx'] > newest['batch_idx']:
                newest = checkpoint
        newest_path = f"{self.cp_dir}/newest.cp"
        os.move(f"{self.cp_dir}/{newest}", newest_path)
    
        for addr in self.addrs:
            try:
                subprocess.run(['scp', newest_path, f'{addr}:{newest_path}'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during scp: {e}")
        

    def checkpoint_state(self, model, optimizer, epoch, batch_idx):
        path = f"{self.cp_dir}/jit.cp"
        torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx,
        }, path)

    def recover_state(self, model, optimizer):
        path = f"{self.cp_dir}/newest.cp"
        while not os.path.exists(path):
            time.sleep(1)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        os.remove(path)
        return checkpoint['epoch'], checkpoint['batch_idx']

# --- Training ---
def train_model(model, train_loader, optimizer, criterion, epoch, rank, checkpointer, sampler):
    model.train()
    setup_watchdog(rank)
    sampler.set_epoch(epoch)
    start_time = time.time()
    log_iter_start = time.time()
    global args
    try:
        with open(f'output/{log_file_name}', 'a+') as f:
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= stop_iter:
                    break

                data, target = data.to(device), target.to(device)

                output = model(data)

                loss = criterion(output, target)

                if args.error_before_all_reduce and batch_idx == 20 and epoch == 0:
                    raise RuntimeError("Simulated error before all_reduce")

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # Logging
                elapsed_time = time.time() - start_time
                f.write(f"{epoch},{batch_idx + 1},{elapsed_time:.4f}\n")
                start_time = time.time()

                if (batch_idx + 1) % log_iter == 0 and rank == 0:
                    iter_time = time.time() - log_iter_start
                    print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx + 1} | Loss {loss.item():.4f} | Time {iter_time:.2f}")
                    log_iter_start = time.time()
    except BaseException as e:
        global interrupted_by_sigusr1
        print("Caught exception:", e)
        print("was signal interrupted:", interrupted_by_sigusr1)
        if not interrupted_by_sigusr1:
            if rank == 0:
                master_send_failure_to_clients()
            else:
                send_failure_to_master()
        else:
            print(f"Rank {rank} | Saving checkpoint...")
            checkpointer.checkpoint_state(model, optimizer, epoch, batch_idx)
        exit(1)


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
    global device
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

    model = mdl.VGG11().to(device)
    ddp_model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    global addrs
    checkpointer = Checkpointer(ddp_model, jit_checkpoint_dir, addrs)
    if from_checkpoint:
        if rank == 0:
            checkpointer.master_consolidate_checkpoints()
        checkpointer.recover_state(ddp_model, optimizer)

    for epoch in range(num_epochs):
        train_model(ddp_model, train_loader, optimizer, criterion, epoch, rank, checkpointer, sampler)

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
    parser.add_argument('--error_before_all_reduce', action='store_true', default=False, help='Simulate error before all_reduce')
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
