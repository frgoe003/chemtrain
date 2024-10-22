import functools

import numpy as onp

import jax_md_mod


import tomli

from jax import debug


from e3nn_jax import scatter_sum
import jax
from jax import numpy as jnp, nn, tree_util

import haiku as hk

from chemtrain.deploy import exporter, utils, graphs

from jax_md_mod import custom_energy
from jax_md import partition, space

def main():

    exporter = build_export_model()()

    export_module = exporter.export()

    print(export_module)

    with open("lennard_jones.ptb", "wb") as f:
        f.write(export_module.SerializeToString())


def build_export_model():

    class LennardJonesExport(exporter.Exporter):

        graph_type = graphs.SimpleSparseNeighborList

        def energy_fn(self, pos, species, graph):

            neighbors = partition.NeighborList(
                jnp.stack((graph.senders, graph.receivers)),
                pos, None, None, graph.senders.size, partition.Sparse,
                None, None, None
            )

            assert neighbors.idx.shape[0] == 2, "Wrong shape"
            print(neighbors.idx.shape)

            box = onp.eye(3) / 10.0 # Convert to Angstrom

            apply_fn =

            return apply_fn(pos, neighbors)

    return LennardJonesExport

if __name__ == "__main__":
    main()
