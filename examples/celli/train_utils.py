import datetime
import functools
import uuid
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import mdtraj

from cycler import cycler
import matplotlib.pyplot as plt

import numpy as onp

import optax
import tomli_w

from sklearn import linear_model

from jax import tree_util, random

from jax_md_mod import custom_electrostatics
from jax_md_mod import custom_partition
from jax_md_mod.model import neural_networks

from chemtrain.ensemble import sampling
from chemtrain.trainers import ForceMatching, Difftre

from chemutils.models import painn
from chemutils.models.allegro import allegro_qeq_neighborlist_pp
from chemutils.models.mace import mace_qeq_neighborlist_pp
from chemutils.models.nequip import nequip_neighborlist_pp
from chemutils.visualize import molecule


def define_model(config,
                 dataset=None,
                 nbrs_init=None,
                 max_edges=None,
                 per_particle=False,
                 avg_num_neighbors=1.0,
                 positive_species=False,
                 displacement_fn=None,
                 exclude_correction=False,
                 exclude_electrostatics=False,
                 fractional_coordinates=True,
                 box=None,
                 max_triplets=None,
                 ):
    """Initializes a concrete model for a system given path to model parameters."""

    # Requirement to capture all species in the dataset
    n_species = 100
    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            assert 'species' in dynamic_kwargs.keys(), 'species not in dynamic_kwargs'

            if "mask" not in dynamic_kwargs:
                dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)


            pot, charges = gnn_energy_fn(
                energy_params, pos, neighbor, **dynamic_kwargs
            )

            dipole = jnp.sum(
                jnp.where(dynamic_kwargs["mask"][:, None], charges[:, None] * pos, 0.0),
                axis=0
            )

            if mode == "with_aux":
                return pot, {'charge': charges, 'dipole': dipole}
            else:
                return pot

        return energy_fn

    charge_eq_energy = custom_electrostatics.charge_eq_energy_neighborlist(
        displacement_fn, r_onset=config["model"]["coulomb_onset"],
        r_cutoff=config["model"]["coulomb_cutoff"],
        method="direct", solver="direct",
    )

    if dataset is None:
        return energy_fn_template
    
    model_type = config["model"].get("type")
    print(f"Run model {model_type}")
    if model_type == "MACE":
        init_fn, gnn_energy_fn = mace_qeq_neighborlist_pp(
            displacement_fn, charge_eq_energy, config["model"]["r_cutoff"],
            n_species,
            max_edges=max_edges, output_irreps="1x0e",
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy_and_charge",
            positive_species=positive_species,
            **config["model"]["model_kwargs"]
        )
    elif model_type == "AllegroQeq":
        init_fn, gnn_energy_fn = allegro_qeq_neighborlist_pp(
            displacement_fn, charge_eq_energy, config["model"]["r_cutoff"],
            n_species,
            max_edges=max_edges, output_irreps="1x0e",
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy_and_charge",
            positive_species=positive_species,
            **config["model"]["model_kwargs"]
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented.")


    # Set up NN model
    r_init = jnp.asarray(dataset['training']['R'][0])
    species_init = jnp.asarray(dataset['training']['species'][0])
    mask_init = jnp.asarray(dataset['training']['mask'][0])

    nbrs_init = nbrs_init.update(r_init, mask=mask_init)

    key = random.PRNGKey(11)

    try:
        top = molecule.topology_from_neighbor_list(nbrs_init, species_init)
        fig = molecule.plot_molecule(r_init, top)
        fig.savefig("molecule.pdf", bbox_inches="tight")

    except NotImplementedError:
        print("Could not plot molecule")

    # Load a pretrained model
    init_params = init_fn(
        key, r_init, nbrs_init, species=species_init,
        mask=mask_init, total_charge=jnp.asarray(dataset['training']['total_charge'][0]),
        radius=jnp.asarray(dataset['training']['radius'][0]),
    )

    # Infer the per-species energies from the training set
    spec = onp.arange(1, dataset['training']['species'].max() + 1)
    #
    # # Get a matrix with number of unique species for each sample
    counts = onp.sum(
        spec[None, None, :] == dataset['training']['species'][:, :, None],
        axis=1)
    # counts = counts[:, jnp.any(counts, axis=0)]

    # # Solve for the mean potential contribution
    model = linear_model.Ridge(
        alpha=1e-6, fit_intercept=False, positive=False, solver="svd"
    )

    model.fit(-counts, dataset["training"]['U'])
    per_species_energy = -model.coef_

    print("Init params with learned shifts")
    init_params['atomic_energy_layer/~/embed']['embeddings'] = (
        init_params['atomic_energy_layer/~/embed']['embeddings'].at[spec - 1, :].set(
            onp.reshape(per_species_energy, (spec.size, 1))
        )
    )

    print(f"Initial energy is {jax.jit(energy_fn_template(init_params))(r_init, nbrs_init, mask=mask_init, species=species_init, total_charge=0, radius=jnp.asarray(dataset['training']['radius'][0]))}")

    return energy_fn_template, init_params


symbol_to_radius = {
    "H": 32, "He": 46, "Li": 133, "Be": 102, "B": 85, "C": 75, "N": 71,
    "O": 63, "F": 64, "Ne": 67, "Na": 155, "Mg": 139, "Al": 126, "Si": 116,
    "P": 111, "S": 103, "Cl": 99, "Ar": 96, "K": 196, "Ca": 171, "Sc": 148,
    "Ti": 136, "V": 134, "Cr": 122, "Mn": 119, "Fe": 116, "Co": 111,
    "Ni": 110, "Cu": 112, "Zn": 118, "Ga": 124, "Ge": 121, "As": 121,
    "Se": 116, "Br": 114, "Kr": 117, "Rb": 210, "Sr": 185, "Y": 163,
    "Zr": 154, "Nb": 147, "Mo": 138, "Tc": 128, "Ru": 125, "Rh": 125,
    "Pd": 120, "Ag": 128, "Cd": 136, "In": 142, "Sn": 140, "Sb": 140,
    "Te": 136, "I": 133, "Xe": 131, "Cs": 232, "Ba": 196, "La": 180,
    "Ce": 163, "Pr": 176, "Nd": 174, "Pm": 173, "Sm": 172, "Eu": 168,
    "Gd": 169, "Tb": 168, "Dy": 167, "Ho": 166, "Er": 165, "Tm": 164,
    "Yb": 170, "Lu": 162, "Hf": 152, "Ta": 146, "W": 137, "Re": 131,
    "Os": 129, "Ir": 122, "Pt": 123, "Au": 124, "Hg": 133, "Tl": 144,
    "Pb": 144, "Bi": 151, "Po": 145, "At": 147, "Rn": 142, "Fr": 223,
    "Ra": 201, "Ac": 186, "Th": 175, "Pa": 169, "U": 170, "Np": 171,
    "Pu": 172, "Am": 166, "Cm": 166, "Bk": 168, "Cf": 168, "Es": 165,
    "Fm": 167, "Md": 173, "No": 176, "Lr": 161, "Rf": 157, "Db": 149,
    "Sg": 143, "Bh": 141, "Hs": 134, "Mt": 129, "Ds": 128, "Rg": 121,
    "Cn": 122, "Nh": 136, "Fl": 143, "Mc": 162, "Lv": 175, "Ts": 165,
    "Og": 157
}


def add_radii(dataset):
    radius_lookup = jnp.asarray([1.0] + [
        symbol_to_radius[mdtraj.element.Element.getByAtomicNumber(i).symbol]
        for i in range(1, onp.max(dataset["training"]["species"]) + 1)
    ])

    for key in dataset.keys():
        dataset[key].pop("box")
        dataset[key]["radius"] = radius_lookup[dataset[key]["species"]] / 1000.

    return dataset


def init_optimizer(config, dataset, key="optimizer"):

    num_samples = 1
    if 'U' in dataset['training']:
        num_samples = dataset['training']['U'].shape[0]
    elif 'dF' in dataset['training']:
        num_samples = dataset['training']['dF'].shape[0]
    else:
        exit()

    transition_steps = int(
        config[key]["epochs"] * num_samples
    ) // config[key]["batch"]

    if config[key].get("power") == "exponential":
        lr_schedule_fm = optax.exponential_decay(
            config[key]["init_lr"],
            transition_steps,
            config[key]["lr_decay"],
        )
    else:
        lr_schedule_fm = optax.polynomial_schedule(
            config[key]["init_lr"],
            config[key]["lr_decay"] * config[key]["init_lr"],
            config[key].get("power", 2.0),
            transition_steps,
        )

    print(f"Decay LR with power {config[key].get('power', 2.0)}")

    transforms = []

    if config[key].get("normalize"):
        transforms.append(optax.scale_by_param_block_norm())

    if config[key]["type"] == "ADAM":
        transforms.append(optax.scale_by_adam(
            b1=config[key]["optimizer_kwargs"]["b1"],
            b2=config[key]["optimizer_kwargs"]["b2"],
            eps=config[key]["optimizer_kwargs"]["eps"],
            eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
            nesterov=True,
        ))
    elif config[key]["type"] == "AdaBelief":
        transforms.append(optax.scale_by_belief(
            b1=config[key]["optimizer_kwargs"]["b1"],
            b2=config[key]["optimizer_kwargs"]["b2"],
            eps=config[key]["optimizer_kwargs"]["eps"],
            eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
        ))
    else:
        raise NotImplementedError(f"Optimizer {config[key]['type']} not implemented.")

    weight_decay = config[key].get("weight_decay")
    if weight_decay is not None:
        transforms.append(optax.transforms.add_decayed_weights(weight_decay))

    optimizer_fm = optax.chain(
        *transforms,
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


def create_out_dir(config, tag=None):
    now = datetime.datetime.now()
    if tag is not None:
        tag = f"_{tag}"
    else:
        tag = ""

    model = config["model"].get("type", "NequIP")
    name = f"Qeq{tag}_{model}_{now.year}_{now.month}_{now.day}_{uuid.uuid4()}"

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
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=False)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")


def save_predictions(out_dir, name, predictions):
    predictions = tree_util.tree_map(
        onp.asarray, predictions
    )

    onp.savez(out_dir / f"{name}.npz", **predictions)


def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")


def plot_predictions(predictions, reference_data, out_dir, name):
    # Simplifies comparison to reported values
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]
    scale_charge = 11.7871

    cmap = plt.get_cmap('tab20')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    fig.suptitle("Predictions")
    pred_u_per_a = predictions['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy
    ref_u_per_a = reference_data['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy

    mae = onp.mean(onp.abs(pred_u_per_a - ref_u_per_a))
    ax1.set_title(f"Energy (MAE: {mae * 1000:.1f} meV/atom)")
    ax1.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    
    ax1.scatter(ref_u_per_a , pred_u_per_a, c=reference_data["total_charge"])
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    if "F" in predictions:
        # Select only the atoms that are not masked
        pred_F = predictions['F'].reshape((-1, 3))[
                 reference_data['mask'].ravel(), :] / scale_energy * scale_pos
        ref_F = reference_data['F'].reshape((-1, 3))[
                reference_data['mask'].ravel(), :] / scale_energy * scale_pos

        mae = onp.mean(onp.abs(pred_F - ref_F))
        ax2.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")
        ax2.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
        ax2.plot(ref_F.ravel(), pred_F.ravel(), ".")
        ax2.set_xlabel("Ref. F [eV/A]")
        ax2.set_ylabel("Pred. F [eV/A]")

    # Select only the atoms that are not masked
    pred_charge = predictions['charge'].reshape((-1,))[reference_data['mask'].ravel()] / scale_charge
    ref_charge = reference_data['charge'].reshape((-1,))[reference_data['mask'].ravel()] / scale_charge

    mae = onp.mean(onp.abs(pred_charge - ref_charge))
    ax3.set_title(f"Charge (MAE: {mae * 1000:.1f} me)")
    ax3.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    ax3.plot(ref_charge.ravel(), pred_charge.ravel(), "*")
    ax3.set_xlabel("Ref. Q [e]")
    ax3.set_ylabel("Pred. Q [e]")
    ax3.legend(loc="lower right", prop={'size': 5})

    fig.savefig( out_dir / f"{name}.tiff", bbox_inches="tight")
