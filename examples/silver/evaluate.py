from pathlib import Path
import argparse
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
import tomli

import numpy as onp

from jax import numpy as jnp, tree_util

from jax_md_mod import custom_quantity
from jax_md import space, partition

import matplotlib.pyplot as plt

from chemtrain import trainers
from chemtrain.deploy import exporter, graphs

import train_utils
from chemutils.datasets import utils as data_utils
from chemutils.datasets import silver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=0)
    args = parser.parse_args()

    # assert (args.a == 1) and (args.b == 1) and (args.c == 1), f"Supercells not supported"

    out_dir = Path(args.directory)
    with open(out_dir / "config.toml", "rb") as f:
        config = tomli.load(f)

    # with jax.default_device(jax.devices("cpu")[0]):
    #     # Use the same script to also test how big one can scale the supercell
    #     dataset = data_utils.download_dataset('./')
    #     if args.n_samples > 0:
    #         for split in dataset.keys():
    #             dataset[split] = {key: arr[0:args.n_samples, ...] for key, arr in dataset[split].items()}

    #    dataset = data_utils.make_supercell(dataset, args.a, args.b, args.c)
    scale_energy = 96.4853722  # [eV] ->   [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    dataset = silver.download_and_prepare_dataset("./", scale_R=scale_pos, scale_U=scale_energy)
    displacement_fn, _ = space.periodic_general(1.0, fractional_coordinates=True)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    # with jax.default_device(jax.devices("cpu")[0]):
    max_neighbor, max_edges, max_triplets, nbrs_init = data_utils.estimate_edge_and_triplet_count(
        dataset, displacement_fn, r_cutoff=config["model"]["r_cutoff"], capacity_multiplier=1.25
    )

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbor.max()},"
          f"\tMax. edges: {max_edges.max()},"
          f"\tMax. triplets: {max_triplets.max()}")

    max_edges = int(max_edges.max() * config["model"]["edge_multiplier"])
    max_triplets = int(max_triplets.max() * config["model"]["edge_multiplier"] ** 2)

    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, max_triplets
    )

    trainer_fm = trainers.ForceMatching(
        init_params, None, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"],
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
    )

    energy_params = onp.load(
        out_dir / "best_params.pkl", allow_pickle=True
    )
    energy_params = tree_util.tree_map(
        jnp.asarray, energy_params
    )

    print(tree_util.tree_structure(init_params))
    print(tree_util.tree_structure(energy_params))

    for key in init_params.keys():
        assert key in energy_params.keys(), (
            f"Key {key} not contained in the loaded parameters."
        )

    # Compute a new batch size
    batch_size = max([
        config["optimizer"]["batch"] // (args.a * args.b * args.c), 1
    ])

    class Export(exporter.Exporter):

        graph_type = graphs.SimpleSparseNeighborList

        def energy_fn(self, pos, species, graph):

            neighbors = partition.NeighborList(
                jnp.stack((graph.senders, graph.receivers)),
                pos, None, None, graph.senders.size, partition.Sparse,
                None, None, None
            )

            assert neighbors.idx.shape[0] == 2, "Wrong shape"
            print(neighbors.idx.shape)

            pos /= 10.0

            apply_fn = energy_fn_template(energy_params)

            return apply_fn(pos, neighbors, export=True, per_particle=True) / 4.184 # Convert to kcal/mol

    module = Export().export()
    print(module)

    with open(out_dir / "model.ptb", "wb") as f:
        f.write(module.SerializeToString())

    predictions = trainer_fm.predict(
        dataset["validation"], energy_params, batch_size=batch_size,
    )

    train_utils.save_predictions(out_dir, f"preds_validation_a={args.a}_b={args.b}_c={args.c}", predictions)
    train_utils.plot_predictions(predictions, dataset["validation"], out_dir, f"preds_validation_a={args.a}_b={args.b}_c={args.c}")


    plt.show()


if __name__ == "__main__":
    main()
