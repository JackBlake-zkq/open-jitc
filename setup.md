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
```