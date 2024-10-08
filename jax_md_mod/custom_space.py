# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom functions simplifying the handling of fractional coordinates."""
from typing import Union, Tuple, Callable

from jax_md import space, util
import jax.numpy as jnp
from jax import vmap

Box = Union[float, util.Array]


def _rectangular_boxtensor(box: Box) -> Box:
    """Transforms a 1-dimensional box to a 2D box tensor."""
    spatial_dim = box.shape[0]
    return jnp.eye(spatial_dim).at[jnp.diag_indices(spatial_dim)].set(box)


def init_fractional_coordinates(box: Box) -> Tuple[Box, Callable]:
    """Returns a 2D box tensor and a scale function that projects positions
    within a box in real space to the unit-hypercube as required by fractional
    coordinates.

    Args:
        box: A 1 or 2-dimensional box

    Returns:
        A tuple (box, scale_fn) of a 2D box tensor and a scale_fn that scales
        positions in real-space to the unit hypercube.
    """
    if box.ndim != 2:
        box = _rectangular_boxtensor(box)

    def scale_fn(positions, **kwargs):
        _box = kwargs.get('box', box)
        if _box.ndim != 2:
            _box = _rectangular_boxtensor(_box)
        inv_box = jnp.linalg.inv(_box)
        return jnp.dot(inv_box, positions.T).T

    return box, scale_fn


def open_general(box: Box,
                fractional_coordinates: bool = True,
                wrapped: bool = True) -> space.Space:
    """Open boundary conditions on a parallelepiped."""
    inv_box = space.inverse(box)

    def displacement_fn(Ra, Rb, perturbation=None, **kwargs):
        _box, _inv_box = box, inv_box

        if 'box' in kwargs:
            _box = kwargs['box']

            if not fractional_coordinates:
                _inv_box = space.inverse(_box)

        if 'new_box' in kwargs:
            _box = kwargs['new_box']

        if not fractional_coordinates:
            Ra = space.transform(_inv_box, Ra)
            Rb = space.transform(_inv_box, Rb)

        dR = space.pairwise_displacement(Ra, Rb)
        dR = space.transform(_box, dR)

        if perturbation is not None:
            dR = space.raw_transform(perturbation, dR)

        return dR

    def u(R, dR):
        return R + dR

    def shift_fn(R, dR, **kwargs):
        if not fractional_coordinates and not wrapped:
            return R + dR

        _box, _inv_box = box, inv_box
        if 'box' in kwargs:
            _box = kwargs['box']
            _inv_box = space.inverse(_box)

        if 'new_box' in kwargs:
            _box = kwargs['new_box']

        dR = space.transform(_inv_box, dR)
        if not fractional_coordinates:
            R = space.transform(_inv_box, R)

        R = u(R, dR)

        if not fractional_coordinates:
            R = space.transform(_box, R)
        return R

    return displacement_fn, shift_fn
