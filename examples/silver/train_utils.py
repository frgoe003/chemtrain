from pathlib import Path
import uuid
import datetime

import tomli_w

import numpy as onp
from flax.core.lift import switch

import functools

import jax
from jax import tree_util, lax, random, nn

import jax.numpy as jnp

import matplotlib.pyplot as plt
from jax.example_libraries.optimizers import nesterov

from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity

from jax_md_mod.model import layers, neural_networks, prior

import optax

import haiku as hk

import e3nn_jax

from chemtrain.trainers import ForceMatching
from jax_md_mod import custom_space

from chemutils.models.mace import MACE
from chemutils.models.allegro import Allegro

def define_model(config, dataset, nbrs_init, max_edges, max_triplets):
    """Initializes a concrete model for a system given path to model parameters."""

    pot_shift = config["model"].get("energy_shift")

    # Set up NN model
    r_init = jnp.asarray(dataset['training']['R'][0])
    species_init = jnp.ones(r_init.shape[0], dtype=jnp.int32)
    box_init = jnp.asarray(dataset['training']['box'][0])

    key = random.PRNGKey(21)

    fractional = True
    displacement_fn, shift_fn = space.periodic_general(
        box_init, fractional_coordinates=fractional)

    model = config["model"].get("type", "DimeNet")
    if model == "DimeNet":
        n_species = 10

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

    elif model == "Allegro":
        n_species = 10

        model_kwargs = config["model"].get("model_kwargs", {})
        default_kwargs = dict(
            max_ell = model_kwargs.get("max_ell", 3),
            irreps = model_kwargs.get("n_irreps", 128) * e3nn_jax.Irreps(
                model_kwargs.get("irreps", "0o + 1o + 1e + 2e + 2o + 3o + 3e")),
            mlp_n_hidden = model_kwargs.get("hidden_dim", 1024),
            mlp_n_layers = model_kwargs.get("n_layer", 3),
            n_radial_basis = model_kwargs.get("n_radial_basis", 8),
            output_irreps = e3nn_jax.Irreps("0e"),
            num_layers = model_kwargs.get("num_layers", 1),  # 3,
            p=model_kwargs.get("p", 6),
        )

        @hk.transform
        def haiku_model(pos, neighbor, box=None, **dynamic_kwargs):
            allegro_model = Allegro(
                avg_num_neighbors=max_edges / r_init.shape[0],
                radial_cutoff=config["model"]["r_cutoff"],  # In nm
                **default_kwargs,
            )

            # Create a neighbor list with maximum capacity first
            dense_idx = neighbor.idx
            senders = jnp.arange(dense_idx.shape[0]).repeat(dense_idx.shape[1])
            receivers = dense_idx.ravel()

            # Sort the indices of the receivers (invalids will be last)
            # and only keep a pre-defined amount
            _, sorted_idx = lax.top_k(-receivers, max_edges)
            senders = senders[sorted_idx]
            receivers = receivers[sorted_idx]

            # Mask out all invalid neighbors
            mask = receivers < pos.shape[0]

            # Assemble the Allegro Haiku Model
            species = jnp.ones(pos.shape[0], dtype=int)
            node_attrs = nn.one_hot(species, n_species)

            displacements = jax.vmap(
                functools.partial(displacement_fn, box=box)
            )(pos[senders, :], pos[receivers, :])
            displacements = jnp.where(mask[:, None], displacements, config["model"]["r_cutoff"])

            vectors = e3nn_jax.IrrepsArray("1o", displacements)

            maybe_energy = allegro_model(node_attrs, vectors, senders, receivers).array
            maybe_energy = (maybe_energy.T * mask).T

            return util.high_precision_sum(maybe_energy)


        init_params = haiku_model.init(key, r_init, nbrs_init, box=box_init)
        init_params['learnable_shift'] = 0.0

        def energy_fn_template(energy_params):

            def energy_fn(pos, neighbor, rng=None, **dynamic_kwargs):
                if pot_shift is not None:
                    shift = pot_shift * pos.shape[0]
                else:
                    shift = 0.0

                if rng is not None:
                    print(f"Are we trainig?")
                    is_training=True
                else:
                    rng = random.PRNGKey(21)
                    is_training=False

                # Remove this parameter again from the dictionary
                params = {key: value for key, value in energy_params.items()
                          if key != "learnable_shift"}
                shift += energy_params.get('learnable_shift', 0.0) * pos.shape[0]

                gnn_energy = haiku_model.apply(params, rng, pos, neighbor, is_training=is_training, **dynamic_kwargs)

                # Disable the learned energy shift
                if config["model"].get("no_shift", False):
                    return gnn_energy

                return gnn_energy + shift

            return energy_fn

    elif model == "MACE":

        def bessel_basis(length, max_length, number: int):
            return e3nn_jax.bessel(length, number, max_length)

        def soft_envelope(
            length, max_length, arg_multiplicator: float = 2.0,
            value_at_origin: float = 1.2):
            return e3nn_jax.soft_envelope(
                length,
                max_length,
                arg_multiplicator=arg_multiplicator,
                value_at_origin=value_at_origin,
            )

        n_species = 10

        @hk.transform
        def haiku_model(pos, neighbor, box=None, **dynamic_kwargs):
            mace_model = MACE(
                # Irreps of the output, default 1x0e
                r_max = config["model"]["r_cutoff"],
                num_interactions = 2,
                # Number of interactions (layers), default 2
                hidden_irreps = e3nn_jax.Irreps("128x0e + 128x1o"),  # 256x0e or 128x0e + 128x1o
                readout_mlp_irreps = e3nn_jax.Irreps("16x0e"),
                # Hidden irreps of the MLP in last readout, default 16x0e
                avg_num_neighbors = max_edges / r_init.shape[0],
                num_species = n_species,
                output_irreps = e3nn_jax.Irreps("0e"),
                radial_basis = functools.partial(bessel_basis, number=8),
                radial_envelope = soft_envelope,
            )
            # Create a neighbor list with maximum capacity first
            dense_idx = neighbor.idx
            senders = jnp.arange(dense_idx.shape[0]).repeat(
                dense_idx.shape[1])
            receivers = dense_idx.ravel()

            # Sort the indices of the receivers (invalids will be last)
            # and only keep a pre-defined amount
            _, sorted_idx = lax.top_k(-receivers, max_edges)
            senders = senders[sorted_idx]
            receivers = receivers[sorted_idx]

            # Mask out all invalid neighbors
            mask = receivers < pos.shape[0]

            # Assemble the Allegro Haiku Model
            species = jnp.ones(pos.shape[0], dtype=int)

            displacements = jax.vmap(
                functools.partial(displacement_fn, box=box)
            )(pos[senders, :], pos[receivers, :])
            displacements = jnp.where(
                mask[:, None], displacements, 2* config["model"]["r_cutoff"])

            vectors = e3nn_jax.IrrepsArray("1o", displacements)

            maybe_energy = mace_model(vectors, species, senders, receivers).array
            # maybe_energy = (maybe_energy.T * mask).T

            return util.high_precision_sum(maybe_energy)

        init_params = haiku_model.init(key, r_init, nbrs_init, box=box_init)

        def energy_fn_template(energy_params):

            def energy_fn(pos, neighbor, rng=None, **dynamic_kwargs):

                if rng is not None:
                    print(f"Are we trainig?")
                    is_training=True
                else:
                    rng = random.PRNGKey(21)
                    is_training=False

                gnn_energy = haiku_model.apply(
                    energy_params, rng, pos, neighbor, is_training=is_training, **dynamic_kwargs)

                return gnn_energy

            return energy_fn

    else:
        raise NotImplementedError(f"Model {model} not implemented.")

    return energy_fn_template, init_params


def init_optimizer(config, dataset):

    transition_steps = int(
        config["optimizer"]["epochs"] * dataset['training']['U'].size
    ) // config["optimizer"]["batch"]

    lr_schedule_fm = optax.exponential_decay(
        config["optimizer"]["init_lr"], transition_steps, decay_rate=0.33, end_value=config["optimizer"]["lr_decay"])
    optimizer_fm = optax.chain(
        # optax.scale_by_adam(
        #     b1=0.5,
        #     b2=0.9,
        #     eps=1e-8,
        #     eps_root=1e-16,
        #     nesterov=True,
        # ),
        optax.scale_by_belief(0.5, 0.9),
        optax.transforms.add_decayed_weights(config["optimizer"].get("weight_decay", 1e-2)),
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
        # optax.add_noise(1e-4, 0.55, 11)
    )

    return optimizer_fm


def dropout_key_split(trainer: ForceMatching, *args, **kwargs):
    # Update the dropout key
    params = trainer.params
    params["key"] = random.split(params["key"])
    trainer.params = params


def create_out_dir(config):
    now = datetime.datetime.now()

    model = config["model"].get("type", "DimeNet")
    name = f"silver_{model}_r_cutoff_{config['model']['r_cutoff']}_{now.year}_{now.month}_{now.day}_{uuid.uuid4()}"

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

    fig.suptitle("Predictions")

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

