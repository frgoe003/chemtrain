import mdtraj

import numpy as onp

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax_md_mod import io, custom_quantity
from jax_md import space

box, *_ = io.load_box("alanine_heavy_2_7nm.gro")

top = mdtraj.load_topology("alanine_heavy_2_7nm.gro")
r_frac = mdtraj.load_lammpstrj("alanine_dipeptide.lammpstrj", top).xyz / box

assert onp.all(onp.logical_and(r_frac >= 0, r_frac <= 1)), "Fractional coordinates are not within the box."
# r_frac = r_frac[10:5800, ...]

displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)


def postprocess_fn(positions):
    # Compute the dihedral angles
    dihedral_idxs = jnp.array([[1, 3, 4, 6], [3, 4, 6, 8]])  # 0: phi    1: psi
    batched_dihedrals = jax.vmap(
        custom_quantity.dihedral_displacement, (0, None, None)
    )

    dihedral_angles = batched_dihedrals(positions, displacement_fn, dihedral_idxs)

    return dihedral_angles.T


def plot_1d_dihedral(ax, angles, labels, bins=60, degrees=True,
                     xlabel='$\phi$ in deg', ylabel=True):
    """Plot  1D histogram splines for a dihedral angle. """
    color = ['#368274', '#0C7CBA', '#C92D39', 'k']
    line = ['-', '-', '-', '--']

    n_models = len(angles)
    for i in range(n_models):
        if degrees:
            angles_conv = angles[i]
            hist_range = [-180, 180]
        else:
            angles_conv = onp.rad2deg(angles[i])
            hist_range = [-onp.pi, onp.pi]

        # Compute the histogram
        hist, x_bins = jnp.histogram(angles_conv, bins=bins, density=True, range=hist_range)
        width = x_bins[1] - x_bins[0]
        bin_center = x_bins + width / 2

        ax.plot(
            bin_center[:-1], hist, label=labels[i], color=color[i],
            linestyle=line[i], linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel('Density')

    return ax

def plot_histogram_free_energy(ax, phi, psi, kbt, degrees=True, ylabel=False, title=""):
    """Plot 2D free energy histogram for alanine from the dihedral angles."""
    cmap = plt.get_cmap('viridis')

    if degrees:
        phi = jnp.deg2rad(phi)
        psi = jnp.deg2rad(psi)

    h, x_edges, y_edges = jnp.histogram2d(phi, psi, bins=60, density=True)

    h = jnp.log(h) * -(kbt / 4.184)
    x, y = onp.meshgrid(x_edges, y_edges)

    cax = ax.pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax.set_xlabel('$\phi$ [rad]')
    if ylabel:
        ax.set_ylabel('$\psi$ [rad]')
    ax.set_title(title)

    return ax, cax

phi, psi = postprocess_fn(r_frac)

print(f"Phi angles ({phi.size}): {phi}")
print(f"Psi angles ({psi.size}): {psi}")

labels = ["100", "1000", "10000", "All"]

fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", figsize=(9, 3), sharey=True)
ax1 = plot_1d_dihedral(ax1, [phi[:1000, ...], phi[:10000, ...], phi[:100000, ...], phi], labels, xlabel="$\phi\ [deg]$")
ax2 = plot_1d_dihedral(ax2, [psi[:1000, ...], psi[:10000, ...], psi[:100000, ...], psi], labels, xlabel="$\psi\ [deg]$", ylabel=False)


fig, ax1 = plt.subplots(1, 1, layout="constrained", figsize=(3, 3), sharey=True)
ax1, cax = plot_histogram_free_energy(ax1, phi, psi, 2.5, ylabel=True, title="LAMMPS")

cbar = fig.colorbar(cax)
cbar.set_label('Free Energy (kcal/mol)')

plt.show()