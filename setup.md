Hardware: L4 GPU(s)
OS Image: Deep Learning on Linux
OS Image Version: Deep Learning VM with CUDA 12.2 M126

Install DCGM:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
sudo systemctl --now enable nvidia-dcgm
```

Pull repo and install stuff for building Oobleck:
```bash
git clone https://github.com/JackBlake-zkq/open-jitc.git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
pip install build
```

Building Oobleck. From Oobleck directory:
```bash
python -m build
```

Make sure you forward your ssh agent so that you can ssh from one vm to the other. E.g. in your config file:

```
Host oobleck-master
  HostName vm_ip
  User username
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
```

Hostfile for GCP setup with 1 GPU per machine:
```bash
private_ip_1 slots=1 port=22
private_ip_2 slots=1 port=22
```

Running Oobleck, from root dir:
```bash
python -m oobleck.elastic.run --hostfile hostfile --tag test open-jitc/Oobleck/examples/run_gpt2.py --tp_size 1
```