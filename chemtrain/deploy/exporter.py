
import abc

import jax
from jax import numpy as jnp, export, tree_util, Array

import chex
import jraph

from typing import Dict, NamedTuple

import jax_md_mod
from jax_md import util as md_util

from scipy.stats import reciprocal

from . import graphs



class Exporter(metaclass=abc.ABCMeta):

    # Use the default graph containing the full neighbor indices
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    @abc.abstractmethod
    def energy_fn(self, position, species, graph):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            species: (N) Array of atoms species.
            graph: Graph representation of the neighborhood around atoms.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    def _energy_fn(self, position, species, ghost_mask, *graph_args):
        graph, build_statistics = self.graph_type.create_from_args(position, species, ghost_mask,*graph_args)
        per_atom_energies = self.energy_fn(position, species, graph)

        # Attention: Force is negative gradient of potential
        total_neg_energy = jnp.float32(-1.0) * md_util.high_precision_sum(
            per_atom_energies)
        local_energy = md_util.high_precision_sum(
            ghost_mask * per_atom_energies)

        # Differentiate w.r.t. the total potential in the box, but exclude
        # ghost atom contributions to the total potential
        aux = local_energy, *build_statistics

        return total_neg_energy, aux

    def export(self):
        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        force_and_energy_fn = jax.grad(self._energy_fn, argnums=0, has_aux=True)

        # TODO: For different types of graphs, we need different input arguments.
        #       for now, we just use the neighbor idx as before
        scope = export.SymbolicScope()
        n_atoms, = export.symbolic_shape("n_atoms", scope=scope)

        position_def = jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32)
        ghost_mask_def = jax.ShapeDtypeStruct((n_atoms,), jnp.bool)
        species_def = jax.ShapeDtypeStruct((n_atoms,), jnp.int32)
        graph_def = self.graph_type.create_symbolic_input_format(n_atoms, scope)

        print(f"Shapes defs are: \n"
              f"\tpositions: {position_def}\n"
              f"\tghost_mask: {ghost_mask_def}\n"
              f"\tspecies: {species_def}\n"
              f"\tgraph_def: {graph_def}")

        exp: export.Exported = export.export(
            jax.jit(force_and_energy_fn), platforms=["cuda"]
        )(position_def, species_def, ghost_mask_def, *graph_def)

        return exp.mlir_module()


if __name__ == "__main__":

    import allegro_jax

