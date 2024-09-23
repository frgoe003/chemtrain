
import abc

import jax
from jax import numpy as jnp, export, tree_util, Array

import chex
import jraph

from typing import Dict, NamedTuple

from scipy.stats import reciprocal


@chex.dataclass
class DenseNeighborGraph:

    idx = None
    species = None

    @staticmethod
    def create_symbolic_format(max_atoms, scope):
        max_neighbors = export.symbolic_shape("max_neighbors", scope=scope)

        idx = jax.ShapeDtypeStruct((max_atoms, max_neighbors), jnp.int32)
        species = jax.ShapeDtypeStruct((max_atoms,), jnp.int32)

        return DenseNeighborGraph(idx, species)


class NeighborList(NamedTuple, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    @staticmethod
    def create_symbolic_input_format(max_atoms, scope):
        """Creates a symbolic representation of the graph.

        Args:
            max_atoms: The maximum number of atoms, including ghost atoms and
                padding atoms.
            scope: The scope to add more symbolic variables.

        The variables should begin with "graph_".

        Returns:
            Returns a symbolic representation of the graph.

        """

    @staticmethod
    def create_from_args(position, species, *args):
        """Creates the neighbor list from inputs to the exported function."""


class SimpleSparseNeighborList(NamedTuple):

    senders: Array
    receivers: Array

    @staticmethod
    def create_symbolic_input_format(max_atoms, scope):
        # We do not need
        max_neighbors = export.symbolic_shape("graph_max_neighbors", scope=scope)

        senders = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)
        receivers = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)

        return senders, receivers

    @staticmethod
    def create_from_args(position, species, senders, receivers):
        graph = SimpleSparseNeighborList(
            senders=senders, receivers=receivers
        )
        return graph


class Exporter(metaclass=abc.ABCMeta):

    # Use the default graph containing the full neighbor indices
    graph_type: NeighborList = SimpleSparseNeighborList

    @abc.abstractmethod
    def energy_fn(self, position, graph):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            graph: Graph representation of the neighborhood around atoms.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    def _energy_fn(self, position, graph, ghost_mask):
        # TODO: Maybe do some preprocessing
        per_atom_energies = self.energy_fn(position, graph)
        return jnp.sum(per_atom_energies), jnp.sum(per_atom_energies * ghost_mask)

    def export(self):
        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        force_and_energy_fn = jax.grad(self._energy_fn, argnums=0, has_aux=True)

        # TODO: For different types of graphs, we need different input arguments.
        #       for now, we just use the neighbor idx as before
        scope = export.SymbolicScope()
        n_atoms = export.symbolic_shape("n_atoms", scope=scope)

        position_def = jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32)
        ghost_mask_def = jax.ShapeDtypeStruct((n_atoms,), jnp.bool)
        species_def = jax.ShapeDtypeStruct((n_atoms,), jnp.int32)
        graph_def = self.graph_type.create_symbolic_input_format(n_atoms, scope)

        exp: export.Exported = export.export(
            jax.jit(force_and_energy_fn), platforms=["cpu", "CUDA"]
        )(position_def, species_def, *graph_def, ghost_mask_def)

        mlir_str = "\n".join([
            line for line in exp.mlir_module().splitlines()
            if not line.lstrp().startswith("#")
        ])

        return mlir_str


if __name__ == "__main__":

    import allegro_jax

