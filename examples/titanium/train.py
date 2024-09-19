import os
import functools

from pathlib import Path

import tomli_w

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as onp

import jax
from jax import tree_util, lax, random

import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from jax_md_mod import io, custom_quantity, custom_space, custom_energy
from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity
from jax.experimental import mesh_utils

from jax_md_mod.model import layers, neural_networks, prior

import mdtraj

import optax

from collections import OrderedDict


import matplotlib.pyplot as plt

import haiku as hk
import chex
import copy
import contextlib

from chemtrain.data import preprocessing
from chemtrain.ensemble import sampling
from chemtrain import quantity, trainers, util as chem_util
from chemtrain.trainers import ForceMatching


import data_utils

def get_default_config():
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.3,
            edge_multiplier=1.15,
        ),
        optimizer=OrderedDict(
            init_lr=0.001,
            lr_decay=0.01,
            epochs=250,
            batch=25,
            cache=8
        ),
        gammas=OrderedDict(
            virial=4e-6,
            U=1e-6,
            F=1e-2,
        )
    )


def main():

    config = get_default_config()
    out_dir = create_out_dir(config)

    dataset = data_utils.download_dataset("./")

    displacement_fn, _ = space.periodic_general(1.0, fractional_coordinates=True)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    max_neighbor, max_edges, max_triplets, nbrs_init = data_utils.estimate_edge_and_triplet_count(
        dataset, displacement_fn, r_cutoff=config["model"]["r_cutoff"], capacity_multiplier=1.25
    )

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbor.max()},"
          f"\tMax. edges: {max_edges.max()},"
          f"\tMax. triplets: {max_triplets.max()}")

    max_edges = int(max_edges.max() * config["model"]["edge_multiplier"])
    max_triplets = int(max_triplets.max() * config["model"]["edge_multiplier"] ** 2)

    config["model"]["max_edges"] = max_edges
    config["model"]["max_triplets"] = max_triplets
    config["model"]["max_neighbor"] = nbrs_init.idx.shape[1]

    energy_fn_template, init_params = define_model(config, dataset, nbrs_init, max_edges, max_triplets)

    optimizer = init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"],
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        additional_targets={
            'virial': custom_quantity.init_virial_stress_tensor(
                energy_fn_template, reference_box=None, include_kinetic=False)
        },
        weights_keys={
            'virial': 'virial_weights'
        }
    )

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    predictions = trainer_fm.predict(dataset['training'], batch_size=config["optimizer"]["batch"])

    assert not onp.any(onp.isnan(predictions["U"])), "Predicted NaN energies"
    assert not onp.any(onp.isnan(predictions["F"])), "Predicted NaN forces"

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"])

    save_results(config, out_dir, trainer_fm)


def define_model(config, dataset, nbrs_init, max_edges, max_triplets):
    """Initializes a concrete model for a system given path to model parameters."""

    # Set up NN model
    r_init = jnp.asarray(dataset['training']['R'][0])
    species_init = jnp.ones(r_init.shape[0], dtype=jnp.int32)
    box_init = jnp.asarray(dataset['training']['box'][0])

    n_species = 10

    fractional = True
    displacement_fn, shift_fn = space.periodic_general(
        box_init, fractional_coordinates=fractional)

    key = random.PRNGKey(21)
    mlp_init = {
        'b_init': hk.initializers.Constant(0.),
        'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
    }

    init_fn, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
        displacement_fn, config["model"]["r_cutoff"], n_species, embed_size=32,
        init_kwargs=mlp_init, max_edges=max_edges, max_triplets=max_triplets
    )

    # Load a pretrained model
    init_params = init_fn(key, r_init, nbrs_init, species=species_init, box=box_init)

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, **dynamic_kwargs):
            assert 'box' in dynamic_kwargs.keys(), 'box not in dynamic_kwargs'

            # We only have one type of particle
            species = jnp.ones(pos.shape[0], dtype=int)

            gnn_energy = gnn_energy_fn(
                energy_params, pos, neighbor, species=species, **dynamic_kwargs
            )

            return gnn_energy

        return energy_fn

    return energy_fn_template, init_params


def init_optimizer(config, dataset):

    transition_steps = int(
        config["optimizer"]["epochs"] * dataset['training']['U'].size
    ) // config["optimizer"]["batch"]

    lr_schedule_fm = optax.exponential_decay(
        config["optimizer"]["init_lr"], transition_steps, config["optimizer"]["lr_decay"])
    optimizer_fm = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True)
    )

    return optimizer_fm


def create_out_dir(config):
    def _get_hash(subdict):
        tmpstr = ""
        for value in subdict.values():
            if isinstance(value, dict):
                tmpstr += _get_hash(value)
                continue

            try:
               tmpstr += hash(value)
            except TypeError:
               tmpstr += str(value)

        return tmpstr

    id_str = hash(_get_hash(config))
    name = f"titanium_r_cutoff_{config['model']['r_cutoff']}_{id_str}"

    out_dir = Path("output") / name
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return out_dir


def save_results(config, out_dir, trainer: ForceMatching):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "best_params.pkl", ".pkl", best=True)
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=True)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")


if __name__ == "__main__":
    main()

