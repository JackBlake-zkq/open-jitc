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
from datetime import datetime, timedelta
import torchvision.models as models

# print("GIL Enabled:", sys._is_gil_enabled())

# --- Configuration ---
# torch.set_num_threads(4)
seed = 2021
torch.manual_seed(seed)
np.random.seed(seed)
log_iter = 1
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

def forcibly_kill_process():
    os.kill(os.getpid(), signal.SIGKILL)

def inform_interceptor_to_switch_streams():
    global app_log_file
    app_log_file.write("failed\n")
    app_log_file.flush()
    time.sleep(1)

def handle_failure():
    """Return true if thread should return"""
    global stop, in_opt_step
    stop = True
    if not in_opt_step:
        checkpoint_state()
        forcibly_kill_process()
        return False
    else:
        print("In optimizer step, waiting for opt step to finish before checkpointing")
        return True
def master_send_failure_to_clients(skip=[]):
    global connections
    for conn in connections:
        if conn in skip:
            continue
        conn.sendall("failed".encode('utf-8'))
    print("Master sent failure to clients")

def master_recv_and_forward_failures():
    """Return true iff thread should return"""
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
            return handle_failure()
        
def send_failure_to_master():
    global client_socket
    if client_socket:
        client_socket.sendall("failed".encode('utf-8'))
        print("Client sent failure to master")


def recv_failure_from_master():
    """Return true iff thread should return"""
    global client_socket, stop
    ready, _, _ = select.select([client_socket], [], [], 1)
    if ready:
        data = ready[0].recv(1024).decode('utf-8')
        if "failed" in data:
            return handle_failure()


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
                if master_recv_and_forward_failures():
                    return
            else:
                if recv_failure_from_master():
                    return
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


# def handle_sigabrt(signum, frame):
#     print("SIGABRT received")
#     checkpoint_state()
#     sys.exit(1)

# # Register the handler
# signal.signal(signal.SIGABRT, handle_sigabrt)

def master_consolidate_checkpoints():
    start = time.time()
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
    elapsed = time.time() - start
    print(f"Consolidation took {elapsed:.2f} seconds")
    

def checkpoint_state():
    inform_interceptor_to_switch_streams()
    start = time.time()
    global optimizer, epoch, batch_idx, raw_model, ddp_model
    print("Checkpointing state")
    path = f"{jit_checkpoint_dir}/jit.cp"
    cpu_model = raw_model.cpu()
    print("move to cpu done")
    torch.save({
            'model_state': cpu_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'batch_idx': batch_idx,
    }, path)
    print("Done checkpointing")
    elapsed = time.time() - start
    print(f"Checkpointing took {elapsed:.2f} seconds")
    
def recover_state():
    start = time.time()
    global jit_checkpoint_dir, raw_model, optimizer, epoch, batch_idx
    print("Recovering state")
    path = f"{jit_checkpoint_dir}/newest.cp"
    prev_size = 0
    checkpoint = torch.load(path, map_location=device)
    raw_model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    os.remove(path)
    print("Recovered state")
    elapsed = time.time() - start
    print(f"Recovering took {elapsed:.2f} seconds")
    return checkpoint['epoch'], checkpoint['batch_idx']


# --- Training ---
def train_model(model, train_loader, optimizer, criterion, epoch, rank, watchdog_stop_event, sampler):
    model.train()
    log_iter_start = time.time()
    global args, stop, in_opt_step, watchdog_thread
    try:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= stop_iter:
                break
            in_opt_step = False
            if stop:
                print("Stop detected, beggining of batch, will checkpoint")
                checkpoint_state()
                sys.exit(1)


            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            
            if args.error_before_opt_step and batch_idx == 1 and epoch == 0:
                raise RuntimeError("Simulated error before all_reduce")

            optimizer.zero_grad()

            loss.backward()

            # gradient commmunication using all_reduce
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= args.num_nodes
            dist.barrier()
            print("After all_reduce")
            in_opt_step = True

            if stop:
                print("Stop set, after all_reduce")
                watchdog_thread.join()
                print("Watchdog returned, will checkpoint at beginning of next batch")

            print("Entering optimizer step")

            if args.error_during_opt_step and batch_idx == 1 and epoch == 0:
                raise RuntimeError("Simulated error after all_reduce")

            optimizer.step()

            if (batch_idx + 1) % log_iter == 0 and rank == 0:
                iter_time = time.time() - log_iter_start
                print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx + 1} | Loss {loss.item():.4f} | Avg Iter Time {iter_time / log_iter:.2f}")
                log_iter_start = time.time()
    except BaseException as e:
        watchdog_stop_event.set()
        print("Caught exception:", e)
        watchdog_thread.join()
        


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


def run(rank, size, from_checkpoint, model_name):
    print("Using Checkpoint" if from_checkpoint else "Not using Checkpoint")
    global device, addrs, raw_model, ddp_model, optimizer, epoch, batch_idx, watchdog_thread
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    train_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26]),
        ]
    
    resize = [transforms.Resize((224, 224))]
    if model_name == 'Vit-H':
        train_transforms = resize + train_transforms
    transform_train = transforms.Compose(train_transforms)

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26]),
        ]
    if model_name == 'Vit-H':
        test_transforms = resize + test_transforms

    transform_test = transforms.Compose(test_transforms)

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    if model_name == 'VGG11':
        raw_model = mdl.VGG11().to(device)
    elif model_name == 'ResNet152':
        raw_model = models.resnet152(weights=None, num_classes=10).to(device)
    elif model_name == 'VGG19':
        raw_model = models.vgg19(weights=None, num_classes=10).to(device)
    
    model_param_bytes = sum(p.element_size() * p.nelement() for p in raw_model.parameters())
    print(f"Model Size (GB): {model_param_bytes / (1024*1024*1024)}")
    # ddp_model = DDP(raw_model, device_ids=[0] if torch.cuda.is_available() else None)
    optimizer = optim.SGD(raw_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    if from_checkpoint:
        if rank == 0:
            master_consolidate_checkpoints()
            dist.barrier()
        else:
            dist.barrier()
        epoch, batch_idx = recover_state()

    watchdog_stop_event = threading.Event()
    watchdog_thread = setup_watchdog(watchdog_stop_event, rank)
    sampler.set_epoch(epoch)

    for epoch in range(num_epochs):
        train_model(raw_model, train_loader, optimizer, criterion, epoch, rank, watchdog_stop_event, sampler)

    if rank == 0:
        test_model(raw_model, test_loader, criterion)

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
    parser.add_argument('--error_during_opt_step', action='store_true', default=False, help='Simulate error during optimizer step')
    parser.add_argument('--all_reduce_timeout', type=int, default=10, help='Timeout for a single batch in seconds')
    parser.add_argument('--from_checkpoint', action='store_true', default=False, help='Load from checkpoint')
    parser.add_argument('--model', type=str, default='VGG11', choices=['VGG11', 'ResNet152', 'VGG19'], help='Model to use')
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
    else:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to master at {args.master_ip}...")
        client_socket.connect((args.master_ip, 18080))
        print(f"Connected to master at {args.master_ip}")
        client_socket.setblocking(False)


    global batch_size, num_epochs, stop_iter
    batch_size = args.total_batch_size // args.num_nodes
    num_epochs = args.epoch
    stop_iter = args.stop_iter


    dist.init_process_group("nccl", init_method=f"tcp://{args.master_ip}:6585", rank=args.rank, world_size=args.num_nodes)
        # dist.init_process_group("nccl", init_method=f"tcp://{args.master_ip}:6585", rank=args.rank, world_size=args.num_nodes, timeout=timedelta(seconds=args.all_reduce_timeout))
    run(args.rank, args.num_nodes, args.from_checkpoint, args.model)
