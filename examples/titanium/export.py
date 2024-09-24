import functools

import numpy as onp

import jax_md_mod

import e3nn_jax
from allegro_jax import AllegroHaiku

import tomli


from e3nn_jax import scatter_sum
import jax
from jax import numpy as jnp, nn, tree_util

import haiku as hk

from chemtrain.deploy import exporter, utils

import train_utils, data_utils

def main():

    out_dir = "output/titanium_Allegro_r_cutoff_0.5_2024_9_23_782aff92-1267-4b61-9181-65bb0b3e749f"

    with open(out_dir + "/config.toml", "rb") as f:
        config = tomli.load(f)

    exporter = build_export_model(config)(out_dir + "/best_params.pkl")

    print(f"Start exporting")
    mlir_str = exporter.export()

    with open(out_dir + "allegro.mlir", "w") as f:
        f.write(mlir_str)


def init_model_apply(config):
    n_species = 10

    model_kwargs = config["model"].get("model_kwargs", {})
    default_kwargs = dict(
        max_ell=model_kwargs.get("max_ell", 3),
        irreps=model_kwargs.get("n_irreps", 128) * e3nn_jax.Irreps(
            model_kwargs.get("irreps", "0o + 1o + 1e + 2e + 2o + 3o + 3e")),
        mlp_n_hidden=model_kwargs.get("hidden_dim", 1024),
        mlp_n_layers=model_kwargs.get("n_layer", 3),
        n_radial_basis=model_kwargs.get("n_radial_basis", 8),
        output_irreps=e3nn_jax.Irreps("0e"),
        num_layers=model_kwargs.get("num_layers", 1),  # 3,
        p=model_kwargs.get("p", 6),
    )

    @hk.without_apply_rng
    @hk.transform
    def haiku_model(pos, species, graph, **dynamic_kwargs):
        allegro_model = AllegroHaiku(
            avg_num_neighbors=config["model"]["max_edges"] / 256,
            # Attention: Hard-coded
            radial_cutoff=config["model"]["r_cutoff"] * 10.,  # In angstrom
            **default_kwargs,
        )

        # Mask out all invalid neighbors
        mask = graph.receivers < pos.shape[0]

        # Assemble the Allegro Haiku Model
        node_attrs = nn.one_hot(species, n_species)

        # Note: There is no need to define a displacement function for a
        #       LAMMPS export as there is no minimum image convention
        # We set the distance between masked atoms to the cutoff to ensure
        # a stable computation
        displacements = pos[graph.receivers, :] - pos[graph.senders, :]
        displacements = jnp.where(
            mask[:, None], displacements, jnp.full_like(mask[:, None], config["model"]["r_cutoff"])
        )

        vectors = e3nn_jax.IrrepsArray("1o", displacements)

        # Mask the per-edge energies
        edge_energy = allegro_model(
            node_attrs, vectors, graph.senders, graph.receivers).array
        edge_energy = (edge_energy.T * mask).T

        per_atom_energy = utils.edge_to_atom_energies(
            pos.shape[0], edge_energy, graph.senders, graph.receivers
        )
        return per_atom_energy

    return haiku_model.apply


def build_export_model(config):

    class AllegroExport(exporter.Exporter):

        apply_fn = staticmethod(init_model_apply(config))
        energy_params = None

        def __init__(self, params_path, *args, **kwargs):
            super().__init__(*args, **kwargs)

            energy_params = onp.load(params_path, allow_pickle=True)
            energy_params = tree_util.tree_map(
                jnp.asarray, energy_params
            )

            self.energy_params = energy_params

        def energy_fn(self, pos, species, graph):
            energy_params = self.energy_params
            print(f"Energy params are {energy_params}")

            gnn_energy = self.apply_fn(energy_params, pos, species, graph)
            return gnn_energy

    return AllegroExport

if __name__ == "__main__":
    main()
