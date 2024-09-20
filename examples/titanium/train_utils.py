from pathlib import Path
import uuid
import datetime

import tomli_w

import numpy as onp

from jax import tree_util, lax, random

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity

from jax_md_mod.model import layers, neural_networks, prior

import optax

import haiku as hk
from chemtrain.trainers import ForceMatching

import data_utils


def define_model(config, dataset, nbrs_init, max_edges, max_triplets):
    """Initializes a concrete model for a system given path to model parameters."""

    pot_shift = config["model"].get("energy_shift")

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
    init_params['learnable_shift'] = 0.0

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, **dynamic_kwargs):
            assert 'box' in dynamic_kwargs.keys(), 'box not in dynamic_kwargs'

            if pot_shift is not None:
                shift = pot_shift * pos.shape[0]
            else:
                shift = 0.0

            # Remove this parameter again from the dictionary
            params = {key: value for key, value in energy_params.items()
                      if key != "learnable_shift"}
            shift += energy_params.get('learnable_shift', 0.0) * pos.shape[0]

            # We only have one type of particle
            species = jnp.ones(pos.shape[0], dtype=int)

            gnn_energy = gnn_energy_fn(
                params, pos, neighbor, species=species, **dynamic_kwargs
            )

            return gnn_energy + shift

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
    now = datetime.datetime.now()

    name = f"titanium_r_cutoff_{config['model']['r_cutoff']}_{now.year}_{now.month}_{now.day}_{uuid.uuid4()}"

    out_dir = Path("output") / name
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return out_dir


def save_training_results(config, out_dir, trainer: ForceMatching):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "best_params.pkl", ".pkl", best=True)
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=True)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")

def save_predictions(out_dir, name, predictions):
    predictions = tree_util.tree_map(
        onp.asarray, predictions
    )

    onp.savez(out_dir / f"{name}.npz", **predictions)

def plot_predictions(predictions, reference_data, out_dir, name):
    scale_energy = 96.4853722  # [eV] ->   [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5),
                                        layout="constrained")

    fig.suptitle("Predictions on Testset")

    mae = onp.mean(onp.abs(
        predictions['U'] - reference_data['U'])) / scale_energy / 256
    ax1.set_title(f"Energy (MAE: {mae * 1000:.1f} meV/atom)")
    ax1.plot(reference_data['U'] / scale_energy / 256,
             predictions['U'] / scale_energy / 256, "*")
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    mae = onp.mean(onp.abs(predictions['F'] - reference_data[
        'F'])) / scale_energy * scale_pos
    ax2.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")
    ax2.plot(reference_data['F'][::50].ravel() / scale_energy * scale_pos,
             predictions['F'][::50].ravel() / scale_energy * scale_pos,
             "*")
    ax2.set_xlabel("Ref. F [eV/A]")
    ax2.set_ylabel("Pred. F [eV/A]")

    mae = onp.mean(onp.abs(predictions['virial'] - reference_data[
        'virial'])) / scale_energy * (scale_pos ** 3)
    ax3.set_title(f"Virial (MAE: {mae * 1000:.1f} meV/A^3)")
    ax3.plot(reference_data['virial'][
                 reference_data['type'] == 0].ravel() / scale_energy * (
                     scale_pos ** 3), predictions['virial'][
                 reference_data['type'] == 0].ravel() / scale_energy * (
                     scale_pos ** 3), "*")
    ax3.set_xlabel("Ref. W [eV/A^3]")
    ax3.set_ylabel("Pred. W [eV/A^3]")

    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")

