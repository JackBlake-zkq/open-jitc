# How to run

You can run Oobleck examples in two ways: run in standalone (using one node without fault tolerance),
or across multiple nodes with fault tolerance capability.

## Run in standalone

> For more configurable arguments, use `python run_singlenode.py --help`. Following example includes minimial necessary information.

This mode executes workers in this node. As only one node training is possible, Oobleck fault tolerance working in node granularity cannot be used.

**Use `run_singlenode.py` script.** If you want to run `run_gpt2.py`:
```bash
python run_singlenode.py --tag <tag> --num_agents 2 --num_gpus_per_agent 2 run_gpt2.py --tp_size 2
```
which uses 4 GPUs.

Oobleck profiles the given model and uses the profile result to generate pipeline templates. The profile result is cached in `<base_dir>/<tag>/profile` (where default `<base_dir>` is `/tmp/oobleck`, which can be overridden).

`--num_gpus_per_agent` should be the same with `tp_size`.

## Run full Oobleck

This mode executes a master daemon, agents in each node, and workers to run training.

Before running a script, we need to prepare an MPI-style hostfile. A hostfile looks like:
```
<ip_or_hostname1> slots=N port=xx
<ip_or_hostname2> slots=N devices=0,1 port=yy
<ip_or_hostname2> slots=N devices=2,3 port=yy
```
that specifies nodes to be used for training. The master daemon accesses each daemon via ssh (`<ip_or_hostname>:<port>`) to run agents and workers, thus **all nodes must be accessible without password from the node where the master daemon will be running**.

The `slots` field indicates the number of GPUs per agent. All agent must have the same number of slots currently.

The `devices` field is optional; if not specified, Oobleck automatically uses the first N devices starting from 0.
It allows to spawn multiple agents on the same node.

> An executable of `python` should be located in the same path of that in the master node and oobleck should be properly configured in all remote nodes.
> 
> A script must also be in the same path on all nodes.

Now, **use `oobleck.elastic.run` module.** If you want to run `run_gpt2.py`:
```bash
python -m oobleck.elastic.run --hostfile <path_to_hostfile> --tag <tag> run_gpt2.py --tp_size 2
```

Each agent's output will be forwarded to `<base_dir>/<tag>/agent<index>.log`.