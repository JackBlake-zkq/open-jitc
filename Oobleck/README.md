
Standard Oobleck:
```bash
pip install oobleck
```

Setup to build our modified Oobleck:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env" 
pip install build flash-attn
```

Building our modified Oobleck. From Oobleck directory:
```bash
python -m build
pip install dist/oobleck-0.1.1.tar.gz
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
python -m oobleck.elastic.run --hostfile hostfile --tag test examples/run_gpt2.py --tp_size 1
```

Cleanup between runs:
```bash
rm -rf /tmp/oobleck
sudo killall python
```