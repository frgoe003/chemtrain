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
docker run --name lammps_plugin -w /mnt/lammps_plugin/build -it -d --rm -v $PWD:/mnt jaxconnector bash
docker exec lammps_plugin cmake -D LAMMPS_HEADER_DIR=../../lammps/src ../cmake
docker exec lammps_plugin make
```

Note: When changing the connector, you need to recompile the connector and the plugin using:

```bash
docker exec lammps_plugin cmake --build . --clean-first
```

To build lammps with plugin support, run:

```bash
docker run --name lammps -w /mnt/lammps/build -it -d --rm -v $PWD:/mnt jaxconnector bash
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

## Execute an example

```bash
docker run --name lmp_example --gpus all -w /mnt/example/alanine_dipeptide -it -d --rm -v $PWD:/mnt jaxconnector bash
docker exec lmp_example pip install "chemtrain[all]" 
docker exec lmp_example ../../lammps/build/lmp -i input.lmp
docker exec lmp_example ../../lammps/build/lmp -i in.lammps
```


## You want to acces your own files?

You can get back ownership of the docker output by running:

```bash
docker exec lammps chown -R `stat -c "%u:%g" /mnt` /mnt
```
