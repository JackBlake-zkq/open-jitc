To compile `cuda_logger.cpp`, you'll first have to find the installation location of `cuda_runtime.h` in your VM. You can do this by running the following command:

```bash
find /usr -name cuda_runtime.h 2>/dev/null
```
After running this command, you should get a one line response with something like `/usr/local/cuda/include/cuda_runtime.h`

Using this file, you can compile `cuda_logger.cpp` by running the following:

```bash
g++ -shared -fPIC -o libcuda_logger.so cuda_logger.cpp -I/usr/local/cuda/include -ldl
```

Where `/usr/local/cuda/include` is replaced with the folder you found by using the above commands.

To use the log when running Oobleck, you add `LD_PRELOAD=./libcuda_logger.so` inline with the command you're running.

(Note: the path to `libcuda_logger.so` should be the absolute path, which can be found by running `realpath libcuda_logger.so`)

For Oobleck's `run_singlenode.py` example, you would run the following:

```bash
LD_PRELOAD=/home/user/open-jitc/TransparentJITC/libcuda_logger.so TORCH_NCCL_ASYNC_ERROR_HANDLING=0 python run_singlenode.py --tag tag --num_agents 1 --num_gpus_per_agent 1 run_gpt2.py --tp_size 1
```

The `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` was necessary for me, but may not be needed.

To listen to the log, open a separate terminal and go to the directory Oobleck was run from. There, you should see a file called `cuda_trace.log`, which you can listen to in real time by running:

```bash
tail -f cuda_trace.log
```
