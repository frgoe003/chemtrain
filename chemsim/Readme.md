# Connector

The connector compiles the JAX model to HLO using python and provides an interface to
evaluate the model in C++ via a shared library.


## Building Connector

### Prepare Docker Container (Optional)

It is best to use an official docker container for building, e.g., from TensorFlow.
We use the TensorFlow docker container and install some additional dependencies.
To build the container, use:

```bash
docker build -t jaxconnector . 
```

To enable GPU support, we first install the [NVIDIA docker support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Additionally, ensure that you are part of the docker group.

To download and run the container:

```bash
docker run --name xla --gpus all -it -w /connector/connector -it -d --rm -v $PWD:/connector -e HOST_PERMS="$(id -u):$(id -g)" jaxconnector bash
```

### Build Connector

The connector relies on the PJRT plugin to for GPU support.
To build the plugin for an NVIDIA gpu:


Next, the connector can be built using the following command:

## Building Lammps Plugin

```bash
cmake -D LAMMPS_HEADER_DIR=../../../lammps/src ../lammps_plugin
make
```

Note: When changing the connector, you need to recompile the connector and the plugin using:

```bash
cmake --build . --clean-first
```

To build lammps with plugin support, run:

```bash
cmake -D PKG_PLUGIN=yes ../cmake
make
```

## "Installing" LAMMPS and the plugin

To "install" LAMMPS and the plugin, we can create a script to set the
correct environment variables. The script should look like this:

__activate:__ 
```bash
#! /bin/bash

export PATH=/home/paul/nn_prior/external/lammps/build:$PATH
export LAMMPS_PLUGIN_PATH=/home/paul/nn_prior/external/chemtrain/chemsim/build
export JCN_PJRT_PATH=/home/paul/nn_prior/external/chemtrain/chemsim/lib
```

Calling the script with ``source ./activate`` will set all necessary variables
to discover the LAMMPS executable, the plugin, and the PJRT library.


