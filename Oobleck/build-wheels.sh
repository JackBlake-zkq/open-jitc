#!/bin/bash
# Run: docker run --rm -v `pwd`:/io quay.io/pypa/manylinux_2_28_x86_64 bash /io/build-wheels.sh
# Reference: https://setuptools-rust.readthedocs.io/en/latest/building_wheels.html
set -ex

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Compile wheels
for PYBIN in /opt/python/cp310*/bin; do
    rm -rf /io/build/
    "${PYBIN}/pip" install -U setuptools setuptools-rust wheel build
    "${PYBIN}/python" -m build --wheel --sdist --outdir /io/dist/ --no-isolation /io/
done

# Bundle external shared libraries into the wheels
for whl in /io/dist/*cp310*.whl; do
    auditwheel repair "$whl" -w /io/dist/
done

# Remove x86_64 packages
for whl in /io/dist/*linux_x86_64.whl; do
    rm "$whl"
done