# Connector

The connector compiles the JAX model to HLO using python and provides an interface to
evaluate the model in C++ via a shared library.


## Building Connector

It is best to use an official docker container for building, e.g., from TensorFlow.
To enable GPU support, we first install the [NVIDIA docker support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Additionally, ensure that you are part of the docker group.

To download and run the container:

```bash
docker run --name xla --gpus all -it -w /connector -it -d --rm -v $PWD:/connector -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/build:latest-python3.11 bash
```
To compile the binary for testing:

```bash
docker exec xla ./configure.py --backend=CUDA --host_compiler=CLANG
docker exec xla bazel build -c opt --spawn_strategy=sandboxed --experimental_repo_remote_exec --cxxopt='-std=c++17' --host_cxxopt='-std=c++17' :libmain.so
```

## Building Lammps Plugin

```bash
docker run --name lammps_plugin -w /mnt/lammps_plugin/build -it -d --rm -v $PWD:/mnt lammps-build bash
docker exec lammps_plugin cmake -D LAMMPS_HEADER_DIR=../../lammps/src ../cmake
docker exec lammps_plugin make
```

```bash
docker run --name lammps -w /mnt/lammps/build -it -d --rm -v $PWD:/mnt lammps-build bash
docker exec lammps cmake -D PKG_PLUGIN=yes ../cmake
docker exec lammps make
```

## Test the plugin

First, copy the HLO instruction into the build folder:

```bash
cp connector/fn_hlo.txt lammps/build/fn_hlo.txt
```

Then, we can run LAMMPS inside the container:

```bash
docker exec lammps ./lmp -i input.lmp
```
