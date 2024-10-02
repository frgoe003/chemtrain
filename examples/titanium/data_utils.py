import zipfile
import functools
from urllib import request
from pathlib import Path

import numpy as onp
import os

if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

from jax import numpy as jnp, tree_util

from jax_md_mod import custom_space
from jax_md_mod.model import sparse_graph
from jax_md import quantity as snapshot_quantity, space, partition

from chemtrain.data import preprocessing


def load_subset(data_dir, train_ratio=0.7, val_ratio=0.1):
    """Loads a subset of the data from disk.

    Args:
        data_dir: Path to the directory containing the data.
        train_ratio: Fraction of data to use as training set.
        val_ratio: Fraction of data to use as validation set.

    Returns:
        Returns training, validation, and testing splits of the subset of
        data.

    """
    box = onp.load(data_dir / 'box.npy', allow_pickle=True)
    coord = onp.load(data_dir / 'coord.npy', allow_pickle=True)
    energy = onp.load(data_dir / 'energy.npy', allow_pickle=True)
    force = onp.load(data_dir / 'force.npy', allow_pickle=True)
    virial = onp.load(data_dir / 'virial.npy', allow_pickle=True)
    type = onp.load(data_dir / 'types.npy', allow_pickle=True)

    # We reshape the data to a standard format
    n_samples = box.shape[0]

    # Transpose box tensor to conform to JAX-MD format
    dataset = dict(
        box=onp.reshape(box, (n_samples, 3, 3)).swapaxes(1, 2),
        R=onp.reshape(coord, (n_samples, -1, 3)),
        U=onp.reshape(energy, (n_samples,)),
        type=onp.reshape(type, (n_samples,)),
        F=onp.reshape(force, (n_samples, -1, 3)),
        virial=onp.reshape(virial, (n_samples, 3, 3))
    )

    # Do not shuffle to use same splits as in the paper
    splits = preprocessing.train_val_test_split(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=False)

    return splits


def get_train_val_test_set(dir_files):
    """Loads multiple datasets and combines them to a large one.

    Args:
        dir_files: List of paths to data-directories.

    Returns:
        Returns training, validation, and testing split of the single large
        dataset.

    """

    # Initialize arrays to store the data
    dataset = dict(
        training=dict(box=[], R=[], U=[], F=[], virial=[], type=[]),
        validation=dict(box=[], R=[], U=[], F=[], virial=[], type=[]),
        testing=dict(box=[], R=[], U=[], F=[], virial=[], type=[])
    )

    # Load the data from all provided files
    for i in range(len(dir_files)):
        train_split, val_split, test_split = load_subset(dir_files[i])

        for k in dataset['training'].keys():
            dataset['training'][k].append(train_split[k])
            dataset['validation'][k].append(val_split[k])
            dataset['testing'][k].append(test_split[k])

    # Concatenate to single arrays
    for split in dataset.keys():
        for quantity in dataset[split].keys():
            dataset[split][quantity] = onp.concatenate(dataset[split][quantity],
                                                       axis=0)

    return dataset


def scale_dataset(dataset, scale_U=1.0, scale_R=1.0, fractional=True):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
            Each split is again a dictionary, containing the keys
            ``["box", "R", "U", "F", "virial", "type"]``.
        scale_U: Unit conversion factor for the energy
        scale_R: Unit conversion factor for lengths
        fractional: Whether to scale the dataset in fractional coordinates.

    Returns:
        Returns all splits of the data in the correct units.

    """

    scale_F = scale_U / scale_R
    for split in dataset.keys():

        _, scale_fn = custom_space.init_fractional_coordinates(
            dataset[split]['box'][0])
        vmap_scale_fn = jax.vmap(lambda R, box: scale_fn(R, box=box),
                                 in_axes=(0, 0))

        if fractional:
            dataset[split]['R'] = vmap_scale_fn(dataset[split]['R'],
                                                dataset[split]['box'])
        else:
            dataset[split]['R'] = dataset[split]['R'] * scale_R

        dataset[split]['box'] *= scale_R
        dataset[split]['U'] *= scale_U
        dataset[split]['F'] *= scale_F

        # Scale virial by volume as in the chemntrain implementation and invert
        # the sign.
        volumes = jax.vmap(snapshot_quantity.volume, (None, 0))(3,
                                                                dataset[split][
                                                                    'box'])
        dataset[split]['virial'] *= -scale_U / volumes[:, None, None]

        # Only bulk virials should contribute to the loss. Correct for
        # the reduction in the number of samples
        type = dataset[split]['type']
        virial_weights = (1 - type) / onp.mean(1 - type)

        assert onp.all(
            virial_weights >= 0), "Virial weights should be positive."
        assert onp.isclose(onp.mean(virial_weights), 1.0)

        dataset[split]['virial_weights'] = virial_weights

    return dataset

def download_dataset(root="./"):
    """Downloads the training data and stores it in the provided root directory."""

    # Load data from the link provided in the paper

    url = "https://github.com/tummfm/Fused-EXP-DFT-MLP/raw/main/Dataset/Data_DFT_and_Exp.zip"

    data_dir = Path(root) / "_data"

    print(f"Create directory at {data_dir}")

    data_dir.mkdir(exist_ok=True)

    if not (data_dir / "TI_DFT_EXP").exists():
        request.urlretrieve(url, data_dir / "TI_DFT_EXP.zip")

    with zipfile.ZipFile(data_dir / "TI_DFT_EXP.zip") as zip_f:
        zip_f.extractall(data_dir / "TI_DFT_EXP")

    dft_path = data_dir / "TI_DFT_EXP" / "Data DFT and EXP" / "DFT_data" / "InitAndBulk_256atoms_curatedData.zip"
    with zipfile.ZipFile(dft_path) as zip_f:
        zip_f.extractall(dft_path.parent / "InitAndBulk_256atoms_curatedData")

    exp_path = data_dir / "TI_DFT_EXP" / "Data DFT and EXP" / "Exp_data" / "Exp_Boxes_AtomPositions" / "ExperimentalLattice_Boxes_AtomPositions.zip"
    with zipfile.ZipFile(exp_path) as zip_f:
        zip_f.extractall(
            exp_path.parent / "ExperimentalLattice_Boxes_AtomPositions")

    data_list = [
        dft_path.parent / "InitAndBulk_256atoms_curatedData" / "261022_AllInitAndBulk_256atoms_with_types_curatedData"]

    predef_weights = onp.load(data_list[0] / 'types.npy')
    predef_weights = jnp.array(predef_weights)

    scale_energy = 96.4853722  # [eV] ->   [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    dataset = get_train_val_test_set(data_list)
    dataset = scale_dataset(dataset, scale_R=scale_pos, scale_U=scale_energy,
                            fractional=True)

    return dataset

def estimate_edge_and_triplet_count(dataset, displacement_fn, r_cutoff=0.5, capacity_multiplier=1.25):
    """Iterates through all data of all splits and determines the minimum dimensions of the graph.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        displacement_fn: Function to compute displacement between particles.
        r_cutoff: Cutoff radius within particles are considered neighbors
        capacity_multiplier: Factor to estimate max. number of neighbors based
            on the computes number of neighbors.

    Returns:
        Returns arrays containing the maximum number of neighbors, edges, and
        triplets of the graph and a sufficiently large neighbor list for the
        given dataset.

    """

    @jax.jit
    def compute_and_check_overflow(position, neighbor, box):
        neighbor = neighbor.update(position, box=box)
        dynamic_displacement = functools.partial(displacement_fn, box=box)

        # Compute a graph and return the estimated number of weights and triplets
        graph, _ = sparse_graph.sparse_graph_from_neighborlist(
            dynamic_displacement, position, neighbor, r_cutoff
        )

        max_triplets = jnp.int32(jnp.ceil(graph.n_triplets))
        max_edges = jnp.int32(jnp.ceil(graph.n_edges))

        return max_edges, max_triplets, neighbor.did_buffer_overflow

    n_samples = 0
    for split in dataset.keys():
        n_samples += dataset[split]['box'].shape[0]

    all_max_neighbors, all_max_edges, all_max_triplets = onp.zeros((3, n_samples), dtype=int)

    neighbor = None
    overflow = False

    neighbor_fn = partition.neighbor_list(
        displacement_fn, 1.0, r_cutoff, capacity_multiplier=capacity_multiplier,
        disable_cell_list=False, dr_threshold=0.0, fractional_coordinates=True
    )

    def compute_and_check(position, neighbor, box, overflow):
        if neighbor is None or overflow:
            print(f"Re-compute the neighborlist for samples {idx} in split {split}")
            neighbor = neighbor_fn.allocate(position, box=box)

            return compute_and_check(position, neighbor, box, False)

        return *compute_and_check_overflow(position, neighbor, box), neighbor

    n_iter = 0
    for split in dataset.keys():
        for idx in range(dataset[split]['box'].shape[0]):
            box, position = dataset[split]['box'], dataset[split]['R']

            # assert onp.all(position <= 1.0), f"Fractional coordinates are wrong."

            max_edges, max_triplets, overflow, neighbor = compute_and_check(
                jnp.asarray(position[idx]), neighbor, jnp.asarray(box[idx]), overflow
            )

            all_max_edges[n_iter] = int(max_edges)
            all_max_triplets[n_iter] = int(max_triplets)
            all_max_neighbors[n_iter] = int(neighbor.idx.shape[1])


    return all_max_neighbors, all_max_edges, all_max_triplets, neighbor


def make_supercell(dataset, a=1, b=1, c=1, fractional=True):
    """Transforms the boxes of the dataset into supercells.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        a, b, c: Repetitions of the cells in the directions of the base
            vectors.
        fractional: Whether positions are given in fractional coordinates.

    Returns:
        Returns the dataset tiled into supercells.

    """

    assert fractional, "Not implemented in real space."

    # Move all particles into the original boxes
    for split in dataset.keys():
        dataset[split]["R"] = onp.mod(dataset[split]["R"], 1.0)

    # For all samples and all particles, we add all combinations of the multiples
    # of a, b, and c.
    @functools.partial(jax.vmap, in_axes=(0, None, None, None)) # All samples
    @functools.partial(jax.vmap, in_axes=(0, None, None, None)) # All particles
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0)) # All c repetitions
    @functools.partial(jax.vmap, in_axes=(None, None, 0, None)) # All b repetitions
    @functools.partial(jax.vmap, in_axes=(None, 0, None, None)) # All a repetitions
    def tile_positions_and_forces(subset, a_off, b_off, c_off):
        subset["R"] += jnp.asarray([a_off, b_off, c_off])
        # We do not need to changes forces, returned as is
        return subset

    for split in dataset.keys():
        a_off = jnp.arange(a)
        b_off = jnp.arange(b)
        c_off = jnp.arange(c)

        # We also add the forces to replicate them
        tiled_subset = tile_positions_and_forces(
            {"R": dataset[split]["R"], "F": dataset[split]["F"]},
            a_off, b_off, c_off
        )

        # We now combine the replicas of the box into a single large one.
        # Additionally, we have to scale back into fractional coordinates
        n_samples = tiled_subset["R"].shape[0]
        dataset[split]["R"] = tiled_subset["R"].reshape((n_samples, -1, 3))
        dataset[split]["R"] *= jnp.asarray([1/a, 1/b, 1/c])
        dataset[split]["F"] = tiled_subset["F"].reshape((n_samples, -1, 3))

        # Virial normalized by volume is intensive, internal energy and box
        # are extensive

        dataset[split]["U"] *= a * b * c
        dataset[split]["box"] *= onp.asarray([a, b, c])[:, None]

    # Ensure that dataset entries are still numpy arrays
    return tree_util.tree_map(onp.asarray, dataset)


if __name__ == "__main__":
    # Test the supercell creator

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    dataset = download_dataset("./")
    dataset = make_supercell(dataset, 1, 1, 1)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    idx = 11
    position = onp.dot(dataset["training"]["box"][idx], dataset["training"]["R"][idx].T).T

    ax.scatter(position[:, 0], position[:, 1], position[:, 2])

    # Scatter some pseudo-points to make axes equal
    min_max = [position.min(), position.max()]
    ax.scatter(min_max, min_max, min_max)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
