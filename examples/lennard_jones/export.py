import functools

import numpy as onp

import jax_md_mod


import tomli

from jax import debug


from e3nn_jax import scatter_sum
import jax
from jax import numpy as jnp, nn, tree_util

import haiku as hk

from chemtrain.deploy import exporter, utils

from jax_md_mod import custom_energy
from jax_md import partition, space

def main():

    exporter = build_export_model()()

    mlir_str = exporter.export()

    with open("lennard_jones.mlir", "w") as f:
        f.write(mlir_str)


def build_export_model():

    class LennardJonesExport(exporter.Exporter):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def energy_fn(self, pos, species, graph):

            neighbors = partition.NeighborList(
                jnp.stack((graph.senders, graph.receivers)),
                pos, None, None, graph.senders.size, partition.Sparse,
                None, None, None
            )

            # debug.print("Positions {}", pos)

            assert neighbors.idx.shape[0] == 2, "Wrong shape"

            sigma = jnp.asarray([3.165, 1.0])
            epsilon = jnp.asarray([1.0, 0.0])

            # Will apply epsilon = 0 to the energy
            # species = jnp.where(mask, species, 1)

            apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
                lambda ra, rb, **kwargs: rb - ra, None, None,
                sigma=3.165, epsilon=1.0, r_onset=4.0, r_cutoff=5.0,
                initialize_neighbor_list=False
            )

            return apply_fn(pos, neighbors)

    return LennardJonesExport

if __name__ == "__main__":
    main()
