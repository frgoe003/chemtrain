
from jax import ops, numpy as jnp

from jax_md_mod.model import layers
from jax_md.util import high_precision_sum

def edge_to_atom_energies(n_atoms, per_edge_energies, senders, receivers):
    """Assigns energies of edges to per-atom energies.

    Args:
        n_atoms: Number of atoms
        per_edge_energies: Energies of each edge
        senders: Sender atoms of the edge
        receivers: Receiver atoms of the edge

    Returns:
        Returns the energies per atoms. Edge energies are split equally among
        the sender and receiver atoms.

    """
    per_atom_energies = jnp.concatenate([
        layers.high_precision_segment_sum(
            per_edge_energies, senders, num_segments=n_atoms),
        layers.high_precision_segment_sum(
            per_edge_energies, receivers, num_segments=n_atoms)
    ], axis=-1)
    # High-precision average over sender and receiver energies
    per_atom_energies = 0.5 * high_precision_sum(per_atom_energies, axis=-1)
    return per_atom_energies
