import argparse
import os
import sys

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import numpy as onp

import jax

from jax_md_mod import custom_quantity
from jax_md import partition, space

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import trainers

from chemutils.datasets import spice

import train_utils

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
            # type="AllegroQeq",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="64x1o + 32x2e + 16x2o",
            #     # hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o + 4x3e + 4x3o",
            #     # embed_dim=32,
            #     # max_radius=0.5, # Relative to the cutoff
            #     # min_radius=0.01,
            #     mlp_n_hidden=128,
            #     mlp_n_layers=2,
            #     embed_dim=(32, 64, 128),
            #     max_ell=2,
            #     num_layers=(2, 1),
            #     charge_embed_dim=64,
            #     charge_embed_layers=2,
            #     learn_radius=True,
            # ),
            # exclude_electrostatics=False,
            type="MACE",
            model_kwargs=OrderedDict(
                hidden_irreps="128x0e + 32x1o",
                max_ell=2,
                num_interactions=2,
                readout_mlp_irreps="16x0e",
                correlation=3,
                charge_embed_n_hidden=128,
                charge_embed_n_layers=2,
                learn_radius=True,
                qeq=1,
            ),
            exclude_electrostatics=False,
            coulomb_cutoff="max",
            coulomb_onset=0.9 * 0.7,
        ),
        optimizer=OrderedDict(
            init_lr=3e-3,
            lr_decay=1e-2,
            epochs=args.epochs,
            batch=16,
            cache=1000,
            power="exponential",
            weight_decay=1e-2,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.95,
                b2=0.995,
                eps=1e-8,
            )
        ),
        dataset=OrderedDict(
            subsets=[
                "SPICE PubChem Set",  # Regex matching the subset names
                "Amino Acid Ligand",
                "SPICE Dipeptides",
                "DES370K",
                "Ion Pairs"
            ],
            total_charge='total_charge', # Use all samples if commented out
            max_samples=10000 # Use all samples if commented out
        ),
        gammas=OrderedDict(
            U=1e-3,
            F=1e-3,
            charge=0.0,
            dipole=1e3,
        ),
    )

def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config)

    dataset, info = spice.download_spice(
        "/home/ga27pej/Datasets",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False
    )
    dataset = spice.process_dataset(dataset)

    if config["model"]["coulomb_cutoff"] == "max":
        coulomb_onset = 1.1 * (dataset["training"]["R"].max() - dataset["training"]["R"].min())
        coulomb_cutoff = 1.1 * coulomb_onset
        config["model"]["coulomb_onset"] = coulomb_onset
        config["model"]["coulomb_cutoff"] = coulomb_cutoff

    # Update the loaded subset information
    config["dataset"]["subsets"] = list(info["subsets"].values())
    print(f"Dataset information: {info}")

    # Add bond-radii to the dataset
    train_utils.add_radii(dataset)

    displacement_fn, _ = space.free()
    if config["model"]["type"] == "DimeNetPP":
        nbrs_format = partition.Dense

        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors
            ) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, 0.0,
            config["model"]["r_cutoff"], mask_key="mask",
            format=nbrs_format, count_triplets=True
        )

    elif config["model"]["type"] in ["AllegroQeq", "PaiNN", "MACE"]:
        nbrs_format = partition.Sparse

        # Infer the number of neighbors within the model cutoff
        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors,
                   ) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, 0.0,
            config["model"]["r_cutoff"], mask_key="mask",
            format=nbrs_format, count_triplets=True
        )

        if not config["model"]["exclude_electrostatics"]:
            # Infer the neighbor list size for the longer electrostatic interactions. Not required if
            # the electrostatic interactions are excluded
            nbrs_init, _ = graphs.allocate_neighborlist(
                dataset["training"], displacement_fn, 0.0,
                config["model"]["coulomb_cutoff"], mask_key="mask",
                format=nbrs_format,
            )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=True,
        displacement_fn=displacement_fn,
        exclude_correction=False,
        exclude_electrostatics=config["model"]["exclude_electrostatics"],
        fractional_coordinates=False,
    )

    if config["model"]["exclude_electrostatics"]:
        config["gammas"]["charge"] = 0.0  # Pseudo-predictions do not contribute to the error

    optimizer = train_utils.init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        energy_fn_has_aux=True,
        additional_targets={
            "charge": custom_quantity.get_aux("charge"),
            "dipole": custom_quantity.get_aux("dipole")
        },
        weights_keys={
            "F": "F_weight",
            "charge": "charge_weight",
            "dipole": "dipole_weight"
        },
        log_file=out_dir / "training.log",
        checkpoint_path=out_dir / "checkpoints"
    )

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"], checkpoint_freq=10)

    trainer_fm.save_trainer(out_dir / "trainer.pkl", format=".pkl")

    train_utils.plot_convergence(trainer_fm, out_dir)


    test_predictions = trainer_fm.predict(dataset['testing'],
                                          batch_size=config["optimizer"][
                                                         "batch"] // len(
                                              jax.devices()))
    train_predictions = trainer_fm.predict(dataset['training'],
                                           batch_size=config["optimizer"][
                                                          "batch"] // len(
                                               jax.devices()))
    validation_predictions = trainer_fm.predict(
        dataset["validation"], trainer_fm.best_params,
        batch_size=config["optimizer"]["batch"],
    )

    rmse_U_test = onp.sqrt(onp.mean((test_predictions['U'] / onp.sum(
        dataset['testing']['mask'], axis=1) - dataset['testing']['U'] / onp.sum(
        dataset['testing']['mask'], axis=1)) ** 2)) / 96.49  # TOD>
    print(f"Energy Test RMSE: {rmse_U_test * 1000:.1f} (meV/atom)")
    rmse_U_train = onp.sqrt(onp.mean((train_predictions['U'] / onp.sum(
        dataset['training']['mask'], axis=1) - dataset['training'][
                                          'U'] / onp.sum(
        dataset['training']['mask'], axis=1)) ** 2)) / 96.49  # T>
    print(f"Energy Training RMSE: {rmse_U_train * 1000:.1f} (meV/atom)")
    mae_U_test = onp.mean(
        onp.abs(test_predictions['U'] - dataset['testing']['U'])) / 96.49
    print(f"Energy Test MAE: {mae_U_test * 1000:.1f} (meV)")
    mae_U_train = onp.mean(
        onp.abs(train_predictions['U'] - dataset['training']['U'])) / 96.49
    print(f"Energy Training MAE: {mae_U_train * 1000:.1f} (meV)")
    trainer_fm.save_trainer(out_dir / "trainer.pkl", format=".pkl")
    trainer_fm.save_energy_params(out_dir / "best_params.pkl",
                                  save_format=".pkl", best=True)

    train_utils.plot_predictions(validation_predictions, dataset["validation"],
                                 out_dir,
                                 f"preds_validation")
    train_utils.plot_predictions(test_predictions, dataset["testing"], out_dir,
                                 f"preds_testing")
    train_utils.plot_predictions(train_predictions, dataset["training"],
                                 out_dir, f"preds_training")


if __name__ == "__main__":
    main()

