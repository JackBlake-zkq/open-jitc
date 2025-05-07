From this directory, run:
```bash
sudo make
```
This creates the `tracer.so` to use with `LD_PRELOAD` e.g. before your training command, add:

```bash
LD_PRELOAD="abosulte_path_to/tracer.so"
```

Also make sure to create a directory called `log` in the directory you are running the training code from.
