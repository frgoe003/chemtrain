"""Loads a MACE model from PyTorch via MACE-JAX."""


from pathlib import Path

import json

from typing import Dict, Any, Tuple, Callable

import jax
import jax.numpy as jnp

from jax_md_mod import custom_partition
from jax_md import space, partition

from flax import serialization

from mace_jax.modules import models as mace_jax_models
from mace_jax.tools.scatter import scatter_sum as mace_jax_scatter_sum

from mace_jax.cli import mace_jax_from_torch

from . import utils


class JaxMACE(mace_jax_models.ScaleShiftMACE):

    def __call__(
        self,
        # data: dict[str, jnp.ndarray],
        # *,
        # lammps_mliap: bool = False,
        # lammps_class: Any | None = None,
        # compute_node_feats: bool = True,
        vectors,
        senders,
        receivers,
        species,
        mask,
        *,
        num_species: int,

    ) -> jnp.ndarray:
        # ctx = prepare_graph(
        #     data,
        #     lammps_mliap=lammps_mliap,
        #     lammps_class=data.get('lammps_class', lammps_class),
        # )

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


        # num_atoms_arange = ctx.num_atoms_arange
        # node_heads = ctx.node_heads
        # interaction_kwargs = ctx.interaction_kwargs

        # lammps_class = interaction_kwargs.lammps_class
        # lammps_natoms = interaction_kwargs.lammps_natoms
        
        
        n_real = int(num_atoms_arange.shape[0])

        # if lammps_class is not None:
        #     n_real = int(lammps_natoms[0])
        
        
        # node_attrs = data['node_attrs']
        # need_node_attrs_index = self.pair_repulsion or self.distance_transform in {
        #     'Agnesi',
        #     'Soft',
        # }

        # if self.cueq_config is not None and getattr(self.cueq_config, 'enabled', False):
        #     need_node_attrs_index = need_node_attrs_index or bool(
        #         getattr(self.cueq_config, 'optimize_all', False)
        #         or getattr(self.cueq_config, 'optimize_symmetric', False)
        #     )

        # Replaced by "species" input
        # node_attrs_index = data.get('node_attrs_index')
        # if node_attrs_index is None:
        #     node_attrs_index = data.get('node_type')
        # if node_attrs_index is None:
        #     node_attrs_index = data.get('species')
        # if node_attrs_index is not None and getattr(node_attrs_index, 'ndim', 1) != 1:
        #     node_attrs_index = None
        # if node_attrs_index is None and need_node_attrs_index:
        #     node_attrs_index = jnp.argmax(node_attrs, axis=1)
        # if node_attrs_index is not None:
        #     node_attrs_index = jnp.asarray(node_attrs_index, dtype=jnp.int32)

        node_e0 = self.atomic_energies_fn(node_attrs)[num_atoms_arange, node_heads]

        # Not necessary to pool per-graph energies
                
        # e0 = scatter_sum(
        #     src=node_e0,
        #     index=data['batch'],
        #     dim=0,
        #     dim_size=ctx.num_graphs,
        #     indices_are_sorted=True,
        # ).astype(ctx.vectors.dtype)

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
            # if lammps_class is not None:
            #     pair_node_energy = pair_node_energy[:n_real]
        else:
            pair_node_energy = jnp.zeros_like(node_e0)

        # if self._embedding_specs:
        #     # Map back to the original embedding feature names
        #     embedding_features = {
        #         name: 
        #           node_attrs if name == "node_attrs"
        #           else edge_attrs if name == "edge_attrs"
        #           else None
        #         for name in self.embedding_specs.keys()
        #     }
        #     node_feats += self.joint_embedding(batch, embedding_features)
        #     if self.use_embedding_readout:
        #         embedding_node_energy = self.embedding_readout(
        #             node_feats, node_heads
        #         ).squeeze(-1)
        #         e0 += mace_jax_scatter_sum(
        #             src=embedding_node_energy,
        #             index=batch,
        #             dim=0,
        #             dim_size=ctx.num_graphs,
        #             indices_are_sorted=True,
        #         )

        node_energies_list = [pair_node_energy]
        node_feats_list: list[jnp.ndarray] = []

        # node_attrs_full = node_attrs
        # node_attrs_index_full = node_attrs_index

        for idx, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            # if lammps_class is not None and idx > 0:
            #     node_feats = _apply_lammps_exchange(
            #         node_feats, lammps_class, lammps_natoms
            #     )

            # node_attrs_slice = node_attrs_full
            # node_attrs_index_slice = node_attrs_index_full
            # if lammps_class is not None and idx > 0:
            #     node_attrs_slice = node_attrs_slice[:n_real]
            #     if node_attrs_index_slice is not None:
            #         node_attrs_index_slice = node_attrs_index_slice[:n_real]

            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                # n_real=n_real if lammps_class is not None else None,
                first_layer=(idx == 0),
            )
            # if lammps_class is not None and idx == 0:
            #     node_attrs_slice = node_attrs_slice[:n_real]
            #     if node_attrs_index_slice is not None:
            #         node_attrs_index_slice = node_attrs_index_slice[:n_real]
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
                node_attrs_index=species,
            )
            # if lammps_class is not None:
            #     node_feats = node_feats[:n_real]

            node_feats_list.append(node_feats)

        for idx, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else idx
            node_energies_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        # node_feats_out = None
        # if compute_node_feats:
        #     node_feats_out = (
        #         jnp.concatenate(node_feats_list, axis=-1)
        #         if node_feats_list
        #         else node_feats
        #     )

        node_inter_es = jnp.sum(jnp.stack(node_energies_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)

        # inter_e = scatter_sum(
        #     node_inter_es,
        #     index=data['batch'],
        #     dim=-1,
        #     dim_size=ctx.num_graphs,
        #     indices_are_sorted=True,
        # )

        # total_energy = e0 + inter_e
        
        node_energy = node_e0 + node_inter_es

        # Only necessary to return energies per node (for now)
        return node_energy * mask

        # contributions = jnp.stack((e0, inter_e), axis=-1)
        # return {
        #     'energy': total_energy,
        #     'node_energy': node_energy,
        #     'contributions': contributions,
        #     'node_feats': node_feats_out,
        #     'interaction_energy': inter_e,
        #     'displacement': ctx.displacement,
        #     'lammps_natoms': ctx.interaction_kwargs.lammps_natoms,
        # }


def load_foundational_model(outdir: Path = Path("./models"),
                            family: str = "mp",
                            version: str = "medium-0b3"):
    # outdir.mkdir(parents=True, exist_ok=True)
    
    torch_model = mace_jax_from_torch._load_torch_model_from_foundations(
            family, version
    )
    torch_model.eval()

    # output_path = outdir / f"{family}-{version}-jax.npz"
    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if 'error' in config:
        raise RuntimeError(config['error'])

    return torch_model, config

    # # params_bytes = serialization.to_bytes(variables)
    # # output_path.write_bytes(params_bytes)
    # # print(f'Serialized JAX parameters written to {output_path}')

    # # Persist config alongside parameters.
    # # config_path = output_path.with_suffix('.json')
    # # config_path.write_text(json.dumps(mace_jax_from_torch._serialize_for_json(config), indent=2))
    # # print(f'Config written to {config_path}')

    # # print(f'Config: {config}')
    # # print(f'Type of JAX model: {type(jax_model)}, {jax_model}')
    # # print(f'Type of variables: {type(variables)}, {variables}')
    # # print(f'Template data: {template_data}')
    # # print(f"Embedding features: {jax_model.embedding_specs}")

    # print(f"Loaded model with template {jax.tree.map(jnp.shape, template_data)}")

    # jax_model.__class__ = JaxMACE

    # print(f"Loaded foundational MACE-JAX model successfully: {jax_model}.")
    # # help(jax_model.__call__)

    # return jax_model, config, variables


def mace_jax_neighborlist(config: Dict[str, Any],
                          torch_model: Any,
                          displacement: space.DisplacementFn,
                          max_edge_multiplier: float = 1.25,
                          per_particle: bool = False,
                          positive_species: bool = False,
                          scale_pos: float = 0.1,
                          scale_pot: float = 96.485,
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
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        scale_pos: Scaling factor for positions, i.e., to convert units.
        scale_pot: Scaling factor for potentials, i.e., to convert units.

    Returns:
        Returns a tuple of parameters and an apply function.
    
    """

    jax_model, variables, template_data = mace_jax_from_torch.convert_model(
        torch_model, config)
    
    del template_data # Unused

    # We need a different __call__ method
    jax_model.__class__ = JaxMACE

    r_cutoff = jnp.array(config["r_max"], dtype=jnp.float32) * scale_pos
    edges_per_particle = config["avg_num_neighbors"] * max_edge_multiplier

    @utils.batch_apply_fn
    def _apply_fn(params, senders, receivers, edge_feats, node_feats):
        vectors, = edge_feats
        species, mask = node_feats

        return jax_model.apply(
            params, vectors, senders, receivers, species, mask,
            num_species=config['num_elements'],
        )

    def apply_fn(params: Any,
                 position: jax.Array,
                 neighbor: partition.NeighborList,
                 species: jax.Array = None,
                 mask: jax.Array = None,
                 **dynamic_kwargs):
        if species is None:
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
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


if __name__ == '__main__':
    _, config = load_foundational_model()
    
    print(f"Loaded foundational model with config: {config}")
