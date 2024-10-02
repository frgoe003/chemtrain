import abc
import functools

import numpy as onp

import jax
from jax import export, numpy as jnp, lax
from numpy.ma.core import product
from sympy.codegen.cnodes import static

import jax_md_mod
from jax_md import partition, dataclasses

from typing import NamedTuple

class NeighborList(NamedTuple):

    @staticmethod
    @abc.abstractmethod
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
    def create_from_args(position, species, ghost_mask, *args):
        """Creates the neighbor list from inputs to the exported function."""


class SimpleSparseNeighborList(NeighborList):

    senders: jax.Array
    receivers: jax.Array

    @staticmethod
    def create_symbolic_input_format(max_atoms, scope):
        # We do not need
        max_neighbors, = export.symbolic_shape("graph_max_neighbors", scope=scope)

        senders = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)
        receivers = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)

        return senders, receivers

    @staticmethod
    def create_from_args(position, species, ghost_mask, *args):
        graph = SimpleSparseNeighborList(*args)
        return graph, True


class DeviceSparseNeighborList(NamedTuple):

    @staticmethod
    def create_symbolic_input_format(max_atoms, scope):
        max_neighbors, = export.symbolic_shape("graph_max_neighbors", scope=scope)

        return

    @staticmethod
    def create_from_args(positions, species, ghost_mask, *args):
        partition.neighbor_list()


@dataclasses.dataclass
class NeighborListStatistics:

    min_cell_capacity: int
    cell_too_small: bool
    max_neighbors: int


@jax.jit
def compute_cell_list(position, id_buffer, cutoff, mask=None, eps=1e-3):
    """Assigns particle IDs into a 3D grid.

    This implementation follows the JAX, M.D. implementation, but aims to
    support building a cell list by only using shape information from the
    input arguments.

    Args:
        position: The position of the atom.
        id_buffer: Determines the dimensions of the grid and the cell
            capacities. Shape (nx, ny, nz, c) correponds to the numbers of
            cells in x,y,z dimensions and the maximum capacity per cell c.
        cutoff: Cutoff to check the dimensions of the cells. If the cell
            dimensions are smaller than the cutoff, increases the box size
            to enlarge the cells. Has the downside that cells will get fuller
            than usual, but will still yield correct neighbor list results.
        mask: Specifies whether particles should be ignored (mask = 0)
        eps: Tolerance increasing the box and cells to avoid wrong classification

    Returns:
        Returns a tuple with updated particle ids per grid and a dataclass
        containing statistics of the build.

    """
    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=bool)

    *cell_counts, capacity = id_buffer.shape

    # Shift the positions to be in the range [0, box]. First, we shift
    # the masked particles positions to not have an influence on the range.
    # Then we shift the positions to be positive.
    mean_position = jnp.mean(mask * position.T, axis=1, keepdims=True)
    position = jnp.where(mask, position.T, mean_position).T
    position -= jnp.min(position, axis=0, keepdims=True)

    box = jnp.diag(jnp.max(position, axis=0) + eps * cutoff)

    # Generally, the minimum cell dimension must be larger than the cutoff,
    # such that all potential neighbors are contained in the neighboring cells.
    # Potential workaround: Increase box dimension such that smallest cell size
    # is as large as the cutoff. Will work if cell capacity is big enough
    cell_sizes = jnp.diag(box) / jnp.asarray(cell_counts)
    min_cell_size = jnp.min(cell_sizes)
    cell_too_small = (min_cell_size < cutoff)

    # Scale the box dimensions such that all cell sizes are larger than the cutoff
    cell_sizes *= 1 + (cell_sizes < cutoff) * ((cutoff - cell_sizes) / cell_sizes)

    # Get the cell ids for each particle in every dimension (n, x_id, y_id, z_id)
    # and transfrom into flat ids
    nx, ny, nz = cell_counts
    cell_ids = jnp.int32(jnp.floor(position / cell_sizes[jnp.newaxis, :]))
    cell_ids = jnp.sum(cell_ids * jnp.asarray([[nz * ny, nz, 1]]), axis=-1)

    # We can now count how often a particle appears in each cell
    cell_occupancy = jax.ops.segment_sum(jnp.ones_like(cell_ids), cell_ids, cell_ids.size)
    min_cell_capacity = jnp.max(cell_occupancy)

    # We sort the particles along their cell id to obtain, e.g.
    # the cell id array (0, 0, 0, 1, 1, 2, 3, ...). If the capacity is
    # sufficiently large, each segment should be no longer than the capacity.
    # We now create a second array that with repeating numbers 0 ... capacity,
    # such that within segment each number appears at most once.
    sort_idx = jnp.argsort(cell_ids)
    particle_ids = jnp.arange(position.shape[0])
    unique_id_per_segment = jnp.mod(jnp.arange(position.shape[0]), capacity)

    max_cell_ids = 1
    for n_in_dim in cell_counts:
        max_cell_ids *= n_in_dim

    new_id_buffer = jnp.full((max_cell_ids + 1, capacity), position.shape[0])
    new_id_buffer = new_id_buffer.at[cell_ids[sort_idx], unique_id_per_segment].set(particle_ids[sort_idx])
    new_id_buffer = new_id_buffer[:-1, :].reshape(id_buffer.shape)

    statistics = NeighborListStatistics(min_cell_capacity, cell_too_small, 0)
    return new_id_buffer, statistics
