# User Level JITC

## Running instructions

From this directory, run:
```bash
sudo make
```
This creates the `tracer.so` to use with `LD_PRELOAD` e.g. before your training command, add:

```bash
LD_PRELOAD="abosulte_path_to/tracer.so"
```

e.g.

```bash
LD_PRELOAD="/home/$USER/open-jitc/UserLevel/CUDATracePreload/tracer.so" python3 main.py --master-ip 10.128.0.14 --all_reduce_timeout 100 --num-nodes 2  --rank 0
```

For all options, run:
```bash
python3 main.py -h
```
Note the `--all_reduce_timeout` and `--from_checkpoint` options are very relevant to JITC.

Additionally, make sure that the master can ssh into all other ranks via `ssh ip_address`. It should not need to specify the user or provide a password. An easy way to do this is ssh agent with something like this in your ssh config file:

```
Host vm_name
  HostName vm_ip
  User username
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
```

You'll need to modify `main.py` directly to use different models, datasets, loss functions, hyper parameters, etc.

## Experiments we've done

We have tried this using each of the model options that are already included in `main.py`, along with the dataset (CIFAR10) and optimizer that are currently there.

For each, we ensure that it works with both errors before and during the optimizer step using the flags `--error_before_opt_step` and `--error_during_opt_step` respectively.

We also test with errors that are more real and uncatchable:

Firstly, a simulate  corrupting CUDA drivers:
```bash
export LIBTORCH_CUDA_PATH=$(sudo find / -name 'libtorch_cuda.so' | head -n 1)
echo $LIBTORCH_CUDA_PATH
cp $LIBTORCH_CUDA_PATH .
ls
sudo shred $LIBTORCH_CUDA_PATH
```
This should cause a seg fault for the rank it's done on, which cannot be caught in the training script, so recovery will happen after the all reduce times out.

After failure, to recover:
```bash
cp libtorch_cuda.so $LIBTORCH_CUDA_PATH
```
You could imagine starting from a fresh image to have the same effect, which is probably what you would do in practice.

For a simulated hardware failure, we corrupt a register on the PCIe using:

First run:
```bash
lspci
```
Note whichever port has your GPU.
```bash
sudo setpci -s 00:03.0 command=0
```
Replace `00:03.0` with your GPU's handle.

This will make your GPU completely inaccessible until the host is restarted e.g. via `sudo reboot`

