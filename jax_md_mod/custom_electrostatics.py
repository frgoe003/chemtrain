import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jax_md import energy, smap, space




def shielded_interaction(dr, charge, alpha):
    """Gaussian (shielded) charge interaction."""
    # Safety: Avoid division by zero
    mask = dr > 1e-2
    dr = jnp.where(mask, dr, 1e-2)

    return mask * charge * jsp.special.erf(alpha * dr) / dr


def shielded_self(charge, radii):
    """Gaussian (shielded) self-interaction."""
    return jnp.sum(jnp.square(charge) / (2 * radii * jnp.sqrt(jnp.pi)))


def core_interaction(charge, chi, idmp):
    """Core interaction."""
    return jnp.sum(charge * chi + jnp.square(charge) * idmp / 2)


def shielded_interaction_neighbor_list(displacement_fn, r_onset, r_cutoff):
    """Gaussian (shielded) charge interaction."""

    def energy_fn(position, neighbor, charge, radii, chi=None, idmp=None, **dynamic_kwargs):
        _energy_fn = smap.pair_neighbor_list(
            energy.multiplicative_isotropic_cutoff(
                shielded_interaction, r_onset, r_cutoff
            ),
            space.metric(displacement_fn),
            charge=(lambda q1, q2: q1 * q2, charge),
            alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
        )

        pot = shielded_self(charge, radii)
        pot += _energy_fn(position, neighbor, **dynamic_kwargs)

        # Add electronegativity and hardness terms
        if chi is not None and idmp is not None:
            pot += core_interaction(charge, chi, idmp)

        return pot

    return energy_fn


def charge_eq_energy_neighborlist(displacement, r_onset, r_cutoff, interaction="shielded", method="direct"):
    """Charge equilibration energy function."""

    if interaction == "shielded":
        total_energy_fn = shielded_interaction_neighbor_list(displacement, r_onset, r_cutoff)
    else:
        raise ValueError(f"Unknown interaction {interaction}")

    def energy_fn(position, neighbor, radii, chi=None, idmp=None, mask=None, total_charge=None, **dynamic_kwargs):
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=bool)
        if total_charge is None:
            total_charge = 0.0

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
                    position, neighbor, charge, radii, chi, idmp
                )
            )

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
                        position, neighbor, charge, radii, chi, idmp
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

        qeq_energy = total_energy_fn(position, neighbor, charge=charges, radii=radii, chi=chi, idmp=idmp)
        return qeq_energy, charges

    return energy_fn
