# MIT License
#
# Copyright (c) 2022 mace-jax
# Copyright (c) 2026 Multiscale Modeling of Fluid Materials, TU Munich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Loads a MACE model from PyTorch via MACE-JAX."""

from typing import Dict, Any, Tuple, Callable

import jax
import jax.numpy as jnp

from jax_md_mod import custom_partition
from jax_md import space, partition

from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from mace_jax.modules import models as mace_jax_models

from mace_jax.cli import mace_jax_from_torch

from . import utils


class JaxMACE(mace_jax_models.ScaleShiftMACE):
    """MACE-JAX model matching chemtrain's expected __call__ signature."""

    def __call__(
        self,
        vectors,
        senders,
        receivers,
        species,
        mask,
        *,
        num_species: int,

    ) -> jnp.ndarray:
        batch = jnp.zeros(species.shape, dtype=jnp.int32)

        num_atoms_arange = jnp.arange(species.size)
        # TODO: How to deal with "heads"?
        node_heads = jnp.zeros_like(batch)

        # Data defines the following:
        #
        # 'node_attrs': one-hot encoding of species -> encoding of chemtrain type species
        # 'node_attrs_index': class label of species -> chemtrain type species
        # 'edge_index': [2, n_edges] array of senders and receivers
        #
        # For more detauls on deriving data from a graph, see
        # https://github.com/ACEsuit/mace-jax/blob/7e9d467d1701290b6606a20ff2c625c27e973254/mace_jax/tools/gin_model.py#L234

        lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        edge_index = jnp.stack([senders, receivers], axis=0)
        node_attrs = jax.nn.one_hot(
            species,
            num_classes=num_species,
            dtype=vectors.dtype,
        )
        node_attrs = node_attrs * mask[:, None]
        # Cuequivariance implementation raises an error if species with
        # values outside [0, num_species-1] are given.
        species = jnp.clip(species, 0, num_species - 1)

        node_e0 = self.atomic_energies_fn(node_attrs)[num_atoms_arange, node_heads]

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            self._atomic_numbers,
            node_attrs_index=species,
        )

        if self.pair_repulsion:
            pair_node_energy = self.pair_repulsion_fn(
                lengths,
                node_attrs,
                edge_index,
                self._atomic_numbers,
                node_attrs_index=species,
            )
        else:
            pair_node_energy = jnp.zeros_like(node_e0)

        node_energies_list = [pair_node_energy]
        node_feats_list: list[jnp.ndarray] = []

        for idx, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):

            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(idx == 0),
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
                node_attrs_index=species,
            )

            node_feats_list.append(node_feats)

        for idx, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else idx
            node_energies_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_inter_es = jnp.sum(jnp.stack(node_energies_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        node_energy = node_e0 + node_inter_es

        # Only necessary to return energies per node (for now)
        return node_energy * mask


def load_foundational_model(family: str = "mp",
                            version: str = "medium-0b3"):
    """Loads a foundational MACE model from PyTorch.
    
    Args:
        family: Model family to load, e.g., "mp" or "off".
        version: Model version to load, e.g., "medium-0b3".
    
    Returns:
        A tuple of the loaded PyTorch model and its configuration dictionary.

    """

    torch_model = mace_jax_from_torch._load_torch_model_from_foundations(
            family, version)
    torch_model.eval()

    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if 'error' in config:
        raise RuntimeError(config['error'])

    return torch_model, config


class SpeciesMapping():
    """Identity mapping for species."""

    def __call__(self, species: jnp.ndarray, config: Dict) -> jnp.ndarray:
        del config # Unused

        return species


class AtomicNumberMapping(SpeciesMapping):
    """Maps atomic numbers to MACE-JAX species."""

    def __init__(self, max_number: int = 100):
        self.max_number = max_number


    def __call__(self, species: jnp.ndarray, config: Dict) -> jnp.ndarray:

        atomic_numbers = jnp.asarray(config["atomic_numbers"], dtype=jnp.int32)

        # Create lookup table, mapping from atomic number to index
        lookup_table = jnp.argmax(
            jnp.arange(self.max_number)[:, None] + 1 == atomic_numbers[None, :],
            axis=-1
        )

        return lookup_table[species - 1]


def mace_jax_neighborlist(config: Dict[str, Any],
                          torch_model: Any,
                          displacement: space.DisplacementFn,
                          max_edge_multiplier: float = 1.25,
                          per_particle: bool = False,
                          scale_pos: float = 0.1,
                          scale_pot: float = 96.485,
                          species_mapping: SpeciesMapping = SpeciesMapping(),
                          cueq_config: CuEquivarianceConfig = None,
                          ) -> Tuple[Any, Callable]:
    """MACE model for property prediction.

    Args:
        config: Configuration dictionary for the MACE model.
        torch_model: The PyTorch MACE model to convert.
        displacement: Jax_md displacement function
        max_edge_multiplier: Multiplier to limit the maximum number of
            edges per particle.
        max_edges: Expected maximum of valid edges.
        per_particle: Return per-particle energies instead of total energy.
        scale_pos: Scaling factor for positions, i.e., to convert units.
        scale_pot: Scaling factor for potentials, i.e., to convert units.
        species_mapping: Mapping for species to model-compatible indices.
        cueq_config: Configuration for CuEquivariance optimizations.

    Returns:
        Returns a tuple of parameters and an apply function.
    
    """

    jax_model, variables, template_data = mace_jax_from_torch.convert_model(
        torch_model, config, cueq_config=cueq_config)

    print(f"Called with cuex config: {cueq_config}")

    cueq_enabled = False if cueq_config is None else cueq_config.enabled

    del template_data # Unused

    # We need a different __call__ method
    jax_model.__class__ = JaxMACE

    r_cutoff = jnp.array(config["r_max"], dtype=jnp.float32) * scale_pos
    edges_per_particle = config["avg_num_neighbors"] * max_edge_multiplier

    def _apply_fn(params, senders, receivers, edge_feats, node_feats):
        vectors, = edge_feats
        species, mask = node_feats

        return jax_model.apply(
            params, vectors, senders, receivers, species, mask,
            num_species=config['num_elements'],
        )

    # Cuequivariance does not support batching yet
    if cueq_enabled:
        _apply_fn = utils.batch_apply_fn(_apply_fn)

    def apply_fn(params: Any,
                 position: jax.Array,
                 neighbor: partition.NeighborList,
                 species: jax.Array = None,
                 mask: jax.Array = None,
                 **dynamic_kwargs):
        assert species is not None, "Species must be provided."
        species = species_mapping(species, config)

        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement, r_cutoff, position, neighbor, species,
            mask, edges_per_particle=edges_per_particle, sort=True,
            **dynamic_kwargs
        )

        vectors /= scale_pos

        per_atom_energies = _apply_fn(
            params, senders, receivers, (vectors,), (species, mask)
        )
        per_atom_energies *= scale_pot

        if per_particle:
            return per_atom_energies
        else:
            return jnp.sum(per_atom_energies)

    return jax.tree.map(jnp.asarray, variables), jax.jit(apply_fn)
