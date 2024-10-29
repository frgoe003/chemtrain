# Getting Started

This document will walk through the steps to export a simple model from
JAX to LAMMPS.


## Export Model

For this example, we define a simple binary Lennard Jones potential.

```python
from jax import numpy as jnp

from chemtrain.deploy import exporter, utils, graphs

from jax_md_mod import custom_energy, custom_space
from jax_md import partition

class LennardJonesExport(exporter.Exporter):

    # We use the neighborlist from LAMMPS to define the graph
    graph_type = graphs.SimpleSparseNeighborList

    # The force function will be derived automatically from the energy function
    def energy_fn(self, pos, species, graph):

        # LAMMPS deals with periodic boundary conditions via ghost atoms
        displacement_fn, _ = custom_space.nonperiodic_general(
            0.0, fractional_coordinates=False)
        
        neighbors = partition.NeighborList(
            jnp.stack((graph.senders, graph.receivers)),
            pos, None, None, graph.senders.size, partition.Sparse,
            None, None, None
        )
        
        # We expect inputs to have the units A and kcal/mol. The species are
        # zero-based indices.
        apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
            displacement_fn, 0.0, species, sigma=[3.156, 3.5], epsilon=[0.6, 0.8])

        return apply_fn(pos, neighbors)

model = LennardJonesExport()
    
# Compile the model to StableHLO and save the serialized protobuffer
model.export()
model.save("model.ptb")
```

## Run Simulation

The following LAMMPS script will run a simulation with the exported model.

```text
# Note: Plugins are loaded automatically if the paths in the environment
#       variables are set correctly

# 1) Basic settings. The units defined here must correspond to the units used
#    in the model.
units real
dimension 3
atom_style atomic
boundary p p p

neighbor 	2.0 bin
neigh_modify 	every 1 delay 0 check yes once no

# Might be necessary if message-passing NN is used
# comm_modify cutoff 1.0

# 2) Create random positions
region simulation_box block -50 50 -50 50 -50 50

create_box 1 simulation_box

create_atoms 1 random 15000 341341 simulation_box
create_atoms 2 random 15000 341342 simulation_box

# 3) Simulation settings
mass 1 18
mass 2 20

# Loads the previously exported model. Numbers after the backend ("cuda12") are
# multipliers to adjust the buffers. The first number corresponds to the extra
# capacity for ghost atoms and the second number corresponds to the extra
# capacity for edges
pair_style jaxnn
pair_coeff * * model.ptb cuda12 1.1 1.1

# 4) Visualization
thermo 10
thermo_style custom step temp pe ke etotal press

# 5) Run
min_style quickmin
minimize 1.0e-3 1.0e-5 10000 10000

neigh_modify 	every 10 delay 0 check yes

# 6) Visualization
thermo 50
dump mydmp all atom 100 dump.lammpstrj

# Must reset the velocity or the simulation explodes instantaneously
velocity all create 100.0 4928459 rot no dist gaussian

# 7) Run
fix         1 all nve
fix         2 all langevin 300.0 300.0 $(100. * dt) 1530917

timestep 1
run 10000
```
