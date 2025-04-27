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

# --- Configuration ---
torch.set_num_threads(4)
seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)
log_iter = 20
device = torch.device("cpu")  # Will be overridden per rank
jit_checkpoint_dir = "jit_checkpoints"
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

# --- Watchdog ---
def setup_watchdog(stop_event):
    def watchdog():
        while not stop_event.is_set():
            time.sleep(1)
            print("Watchdog triggered! Sending SIGUSR1.")
            os.kill(os.getpid(), signal.SIGUSR1)
    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    return watchdog_thread

class Checkpointer:
    def __init__(self, model):
        self.model = model

    def checkpoint_state(self, model, optimizer, epoch, batch_idx):
        ckpt_path = os.path.join(jit_checkpoint_dir, "checkpoint.pt")
        torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'operation_log': self.operation_log
        }, ckpt_path)
        return ckpt_path

    def recover_state(self, checkpoint_path, model, optimizer):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.operation_log = checkpoint['operation_log']
        return checkpoint['epoch'], checkpoint['batch_idx']

# --- Training ---
def train_model(model, train_loader, optimizer, criterion, epoch, rank, checkpointer, sampler):
    model.train()
    stop_event = threading.Event()
    watchdog = setup_watchdog(stop_event)
    sampler.set_epoch(epoch)
    start_time = time.time()
    log_iter_start = time.time()
    try:
        with open(f'output/{log_file_name}', 'a+') as f:
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= stop_iter:
                    break

                checkpointer.log_operation('data_loader', batch_idx=batch_idx)
                data, target = data.to(device), target.to(device)

                checkpointer.log_operation('forward_pass')
                output = model(data)

                checkpointer.log_operation('loss_computation')
                loss = criterion(output, target)

                checkpointer.log_operation('backward_pass')
                optimizer.zero_grad()
                loss.backward()

                checkpointer.log_operation('optimizer_step')
                optimizer.step()

                # Logging
                elapsed_time = time.time() - start_time
                f.write(f"{epoch},{batch_idx + 1},{elapsed_time:.4f}\n")
                start_time = time.time()

                if (batch_idx + 1) % log_iter == 0 and rank == 0:
                    iter_time = time.time() - log_iter_start
                    print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx + 1} | Loss {loss.item():.4f} | Time {iter_time:.2f}")
                    log_iter_start = time.time()
    except RuntimeError as e:
        if interrupted_by_sigusr1:
            print(f"Rank {rank} | Saving checkpoint due to signal...")
            checkpointer.checkpoint_state(model, optimizer, epoch, batch_idx)
    finally:
        stop_event.set()

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
def init_process(master_ip, rank, size, fn, backend='nccl'):
    dist.init_process_group(backend, init_method=f"tcp://{master_ip}:6585", rank=rank, world_size=size)
    fn(rank, size)

def run(rank, size):
    global device
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
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
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    checkpointer = Checkpointer(ddp_model)

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
    args = parser.parse_args()

    if args.rank == 0:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((args.master_ip, 8080))
        server_socket.listen(args.num_nodes - 1)
        connections = []
        for i in range(1, args.num_nodes):
            client_socket, addr = server_socket.accept()
            connections.append(client_socket)
            client_socket.setblocking(False)
    else:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((args.master_ip, 8080))
        client_socket.setblocking(False)


    global batch_size, num_epochs, stop_iter, log_file_name
    batch_size = args.total_batch_size // args.num_nodes
    num_epochs = args.epoch
    stop_iter = args.stop_iter
    log_file_name = f"timelog_{num_epochs}_{stop_iter}_{args.num_nodes}_{batch_size}.csv"

    with open(f'output/{log_file_name}', 'w') as f:
        f.write("epoch,iteration,elapsed_time\n")

    init_process(args.master_ip, args.rank, args.num_nodes, run)
