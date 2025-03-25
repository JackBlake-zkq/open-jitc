<h1 align="center">Oobleck<br>
Resilient Distributed Training Framework</h1>

Oobleck is a large-model training framework with fast fault recovery support utilizing the concept of *pipeline templates*.

It is the first training framework that realizes:

- **Dynamic reconfiguration**: Oobleck can reconfigure distributed training configurtation without restart after failures.
- **Pipeline template instantiation**: Oobleck pre-generates a set of pipeline templates, and then combine their instantiated pipelines to form a distributed execution plan. The same set of pipeline templates is reused and different pipelines are instantiated after failures.

## Getting Started

### Install

Use `pip` to install Oobleck:
```
pip install oobleck
```

Oobleck relies on [`cornstarch`](https://github.com/Symbioticlab/cornstarch) for pipeline template and [`Colossal-AI`](https://github.com/hpcaitech/ColossalAI) for training backend.
Optionally, install [`apex`](https://github.com/nvidia/apex), [`xformers`](https://github.com/facebookresearch/xformers) and [`flash-attn`](https://github.com/Dao-AILab/flash-attention) to boost throughput (follow instructions in each README).

### Run

Please refer to [this README](./examples/README.md).

### Cluster Management

Oobleck provides a command line interface (CLI) that manages the cluster. Use `oobleck` to access the master agent:

```
$ oobleck --ip <master_ip> --port <master_port> <command> <command_options>
```
where master port can be found in `stdout` of running:

```
| INFO     | __main__:serve:430 - Running master service on port 45145
```

Currently you can see the list of agents and send a request to gracefully terminate an agent:

```
$ oobleck --ip <master_ip> --port <master_port> get_agent_list
=== Agents ===
[0] IP: node1:10000 Status: up (device indices: 0,1)
[1] IP: node1:10000 Status: up (device indices: 2,3)
[2] IP: node2:10000 Status: up (device indices: 0,1)
[3] IP: node2:10000 Status: up (device indices: 2,3)
==============

$ oobleck --ip <master_ip> --port <master_port> kill_agent --agent_index 2
| INFO     | __main__:KillAgent:340 - Terminating agent 2 on node1:10000
```

## Citation

```bibtex
@inproceedings{oobleck-sosp23,
    title     = {Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates},
    author    = {Jang, Insu and Yang, Zhenning and Zhang, Zhen and Jin, Xin and Chowdhury, Mosharaf},
    booktitle = {ACM SIGOPS 29th Symposium of Operating Systems and Principles (SOSP '23)},
    year      = {2023},
}
```