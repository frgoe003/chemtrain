import os
import functools
import sys

import argparse

from pathlib import Path

import tomli_w

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

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

import e3nn_jax

import data_utils, train_utils

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            type="Allegro",
            r_cutoff=args.cutoff,
            edge_multiplier=1.15,
            model_kwargs=OrderedDict(
                max_ell=3,
                n_irreps=128,
                irreps="0o + 1o + 1e + 2e + 2o + 3o + 3e",
                mlp_n_hidden=1024,
                mlp_n_layers=3,
                n_radial_basis=8,
                num_layer=3,
                p=6,
            ),
            no_shift=True
        ),
        optimizer=OrderedDict(
            init_lr=args.lr,
            lr_decay=0.01,
            epochs=args.epochs,
            batch=args.batch,
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
    out_dir = train_utils.create_out_dir(config)

    dataset = data_utils.download_dataset("./")

    # Estimate per-particle shift
    config["model"]["energy_shift"] = onp.mean(
        dataset["training"]["U"] / dataset["training"]["R"].shape[1])

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

    energy_fn_template, init_params = train_utils.define_model(config, dataset, nbrs_init, max_edges, max_triplets)

    optimizer = train_utils.init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
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

    train_utils.save_training_results(config, out_dir, trainer_fm)


if __name__ == "__main__":
    main()

