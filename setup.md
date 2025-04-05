Hardware: L4 GPU(s)
OS Image: Deep Learning on Linux
OS Image Version: Deep Learning VM with CUDA 12.2 M126

Commands to setup machine:
```bash
pip install oobleck
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm
git clone https://github.com/SymbioticLab/Oobleck.git
```

Make sure you forward your ssh agent so that you can ssh from one vm to the other

Running oobleck
```bash
python -m oobleck.elastic.run --hostfile hostfile --tag test Oobleck/examples/run_gpt2.py --tp_size 1
```