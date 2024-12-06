import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp

from jax_md import energy, smap, space, util as md_util


def structure_factor(g, R, q=1):
    if isinstance(q, jnp.ndarray):
        q = q[None, :]
    return md_util.high_precision_sum(
        q * jnp.exp(1j * jnp.einsum('id,jd->ij', g, R)),
        axis=1
    )


def shielded_interaction(dr, charge, alpha, alpha_max=None):
    """Gaussian (shielded) charge interaction."""
    # Safety: Avoid division by zero
    mask = dr > 1e-2
    dr = jnp.where(mask, dr, 1e-2)

    pot = mask * charge * jsp.special.erf(alpha * dr) / dr

    if alpha_max is not None:
        print(f"Apply shielding")
        pot -= mask * charge * jsp.special.erf(alpha_max * dr) / dr

    return pot


def shielded_self(charge, radii):
    """Gaussian (shielded) self-interaction."""
    return jnp.sum(jnp.square(charge) / (2 * radii * jnp.sqrt(jnp.pi)))


def core_interaction(charge, chi, idmp):
    """Core interaction."""
    return jnp.sum(charge * chi + jnp.square(charge) * idmp / 2)


def coulomb_recip_ewald(charge,
                        side_length,
                        alpha: float,
                        n_vectors: int):
  def energy_fn(position, **kwargs):
    dim = position.shape[-1]
    V = side_length**dim

    dg = 2 * onp.pi / side_length
    # Just to make the sum inclusive.
    g_max = n_vectors * dg
    g_range = jnp.arange(0, n_vectors) * dg
    g_range = jnp.concatenate((-g_range[::-1], g_range[1:]))

    gx, gy, gz = jnp.meshgrid(g_range, g_range, g_range)
    g = jnp.reshape(jnp.stack((gx, gy, gz), axis=-1), (-1, dim))
    g2 = jnp.sum(g**2, axis=-1)
    mask = (g2 < g_max**2) & (g2 > 1e-7)
    g2 = jnp.where(mask, g2, 1.0)

    Z = (4 * jnp.pi) / V

    S = structure_factor(g, position, charge)
    S2 = jnp.float32(jnp.conj(S) * S)

    return Z * md_util.high_precision_sum(jnp.exp(-g2 / (4*alpha**2)) / g2 * S2 * mask)
  return energy_fn


def shielded_interaction_neighbor_list(displacement_fn, r_onset, r_cutoff, box=None, alpha=2.0, method="reciprocal"):
    """Gaussian (shielded) charge interaction."""

    def energy_fn(position, neighbor, charge, radii, chi=None, idmp=None, **dynamic_kwargs):
        if method == "direct":
            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
            )
            pot = 0.0
        elif method == "reciprocal":
            _box = dynamic_kwargs.get("box", box)

            assert _box is not None, "Box must be provided for reciprocal space calculation."

            recip_fn = lambda pos, charge, **kwargs: energy.coulomb_recip_pme(charge, _box, onp.int32(30), fractional_coordinates=True, alpha=alpha)(pos, **kwargs)

            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
                alpha_max=alpha
            )
            pot = recip_fn(position, charge, **dynamic_kwargs)
            pot -= shielded_self(charge, 1 / (2 * alpha)) # Correct for the self-interaction added in reciprocal space

            # jax.debug.print("Reciprocal energy: {}", pot)
            # jax.debug.print("Reciprocal gradient: {}", jax.grad(recip_fn, argnums=1)(position, charge, **dynamic_kwargs))
            # jax.debug.print("Reciprocal hessian: {}", jax.hessian(recip_fn, argnums=1)(position, charge, **dynamic_kwargs))


        # jax.debug.print("Real space energy: {}", _energy_fn(position, neighbor))
        # jax.debug.print("Self-interaction energy: {}", shielded_self(charge, radii))

        pot += shielded_self(charge, radii)
        pot += _energy_fn(position, neighbor)

        # Add electronegativity and hardness terms
        if chi is not None and idmp is not None:
            print(f"Add core interaction")
            pot += core_interaction(charge, chi, idmp)

        return pot

    return energy_fn


def charge_eq_energy_neighborlist(displacement, r_onset, r_cutoff, max_radii=None, interaction="shielded", method="direct", electrostatics="direct"):
    """Charge equilibration energy function."""

    if interaction == "shielded":
        total_energy_fn = shielded_interaction_neighbor_list(displacement, r_onset, r_cutoff, method=electrostatics)
    else:
        raise ValueError(f"Unknown interaction {interaction}")

    def energy_fn(position, neighbor, radii=None, chi=None, idmp=None, mask=None, total_charge=None, charges=None, **dynamic_kwargs):
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=bool)
        if total_charge is None:
            total_charge = 0.0
            print(f"No total charge specified. Total charge will be set to {total_charge}")
        else:
            print(f"Total charge specified: {total_charge}")

        n_particles = mask.size

        charge = jnp.zeros(n_particles)
        if method == "direct":
            # Count number of particles
            A = jnp.zeros((n_particles + 1, n_particles + 1))

            # Set last row (charge neutrality) to mask
            A = A.at[-1, :-1].set(mask)

            # Set row for muliplier to mask
            A = A.at[:-1, -1].set(mask)

            # Set diagonal entries to hessian
            A = A.at[:-1, :-1].set(
                jax.hessian(total_energy_fn, argnums=2)(
                    position, neighbor, charge, radii, chi, idmp,
                    **dynamic_kwargs
                )
            )

            # jax.debug.print("Coulomb matrix: {}", A)

            # Charge neutrality constraint (for now)
            b = jnp.concatenate((-chi, jnp.full((1,), total_charge))).reshape((-1, 1))

            # Solve the linear system with lagrange multipliers
            charges = jnp.linalg.solve(A, b)[:-1, 0]

        elif method == "CG":

            def linear_operator(x):
                charge = x[:-1, 0]
                mult = x[-1, 0]

                Ax = jnp.concatenate([
                    jax.grad(total_energy_fn, argnums=2)(
                        position, neighbor, charge, radii, chi, idmp,
                        **dynamic_kwargs
                    ) + mult * mask,
                    jnp.sum(mask * charge).reshape((1,))
                ]).reshape((-1, 1))
                return Ax

            # Initial guess
            x0 = jnp.zeros(n_particles + 1).reshape((-1, 1))
            b = jnp.concatenate((-chi, jnp.full((1,), total_charge))).reshape((-1, 1))

            sol, _ = jsp.sparse.linalg.cg(linear_operator, b, x0=x0, tol=1e-8)
            charges = sol[:-1, 0]

            # raise NotImplementedError("CG method not implemented yet.")
        else:
            raise ValueError(f"Unknown method {method}")

        charges = jnp.where(mask, charges, 0.0)

        qeq_energy = total_energy_fn(
            position, neighbor, charge=charges, radii=radii, chi=chi, idmp=idmp,
            **dynamic_kwargs
        )

        # Only include electrostatic energy
        # qeq_energy = total_energy_fn(
        #     position, neighbor, charge=charges, radii=radii, **dynamic_kwargs)
        return qeq_energy, charges

    return energy_fn
