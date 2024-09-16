"""Constructs an HLO module for alanine dipeptide in an implicit solvent."""

import jax
import jax.numpy as jnp
from jax import tree_util, debug



from jax_md_mod.model import neural_networks, prior
from jax_md_mod import io, custom_space

print("Imported jax_md_mod")

from jax_md import space, partition, quantity

import mdtraj

import numpy as onp

print("...Loading...")

box, r_init, _, _ = io.load_box("alanine_heavy_2_7nm.gro")

n_species = 5
r_cut = 0.5

displacement_fn, shift_fn = space.periodic_general(
    box, fractional_coordinates=True)

# Initialize a box tensor but do not use fractional coordinates
box_tensor, _ = custom_space.init_fractional_coordinates(box)
inv_box = jnp.asarray(onp.linalg.inv(box_tensor))

neighbor_fn = partition.neighbor_list(
    displacement_fn, box_tensor, r_cut, disable_cell_list=True,
    fractional_coordinates=True, capacity_multiplier=1.5
)

nbrs_init = neighbor_fn.allocate(r_init / box, extra_capacity=1)

nbrs_init_idx = onp.ones((10, 9), dtype=int)
for i in range(10):
    for j in range(10):
        if i < j:
            nbrs_init_idx[i,j - 1] = j
        if i > j:
            nbrs_init_idx[i,j] = j

nbrs_init = nbrs_init.set(idx=jnp.asarray(nbrs_init_idx))

print(f"The neighborlist has max. {nbrs_init.idx.shape[1]} neighbors per atom.")

# Load the pretrained parameters

params = onp.load("alanine_dipeptide_re_params.pkl", allow_pickle=True)
params = tree_util.tree_map(jnp.asarray, params)

prior_energy = prior.init_prior_potential(displacement_fn, nonbonded_type="repulsion")

force_field = prior.ForceField.load_ff("alanine_heavy.toml")

top = mdtraj.load_topology("alanine_heavy_2_7nm.gro")

_mapping = force_field.mapping(by_name=True)
def mapping(name="", residue="", **kwargs):
    if residue == "NME" and name =="C":
        return _mapping(name="CH3", **kwargs)
    if name == "CB":
        return _mapping(name="CH3", **kwargs)
    else:
        return _mapping(name=name, **kwargs)

topology = prior.Topology.from_mdtraj(top, mapping)

species = topology.get_atom_species()
masses, *_ = force_field.get_nonbonded_params(species)[0].T

print(f"Species: {species}")
print(f"Masses: {masses}")

prior_energy_fn = prior_energy(topology, force_field)

_, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
    displacement_fn, r_cut, n_species, embed_size=32,
)

def energy_fn(position, neighbor_idx):
    # We transfrom from angstrom back to nm and into fractional coordinates
    n_atoms, dim = position.shape
    _, max_neighbors = neighbor_idx.shape

    position *= jax.lax.broadcast_in_dim(jnp.asarray([[0.1]]), (n_atoms, 3), (0, 1))
    position = jnp.dot(inv_box, position.T).T

    # We use the neighbor list built by LAMMPS
    nbrs = nbrs_init.set(idx=neighbor_idx)
    nbrs = nbrs.set(reference_position=position)

    pot = 0.0
    pot += gnn_energy_fn(params, position, neighbor=nbrs, species=species)
    # pot += prior_energy_fn(position, neighbor=nbrs)

    # Lammps unit system real expects kcal/mol instead of kJ/mol
    pot /= 4.184

    return pot

# This must be defined
force_fn = quantity.force(energy_fn)
