From this directory, run:
```bash
sudo make
```
This creates the `tracer.so` to use with `LD_PRELOAD` e.g.

```bash
LD_PRELOAD="abosulte_path_to/tracer.so" TORCH_NCCL_ASYNC_ERROR_HANDLING=0 python run_singlenode.py --tag tag --num_agents 1 --num_gpus_per_agent 1 run_gpt2.py --tp_size 1
```

Also make sure to create a directory called `log` in the directory you are running Oobleck from.
