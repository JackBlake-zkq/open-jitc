sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm