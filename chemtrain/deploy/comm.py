# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Feature communication used by communicating deployed models.

The Python implementation is an identity operation. During export the two
primitives lower to FFI calls which are implemented by chemtrain-deploy and
LAMMPS. Keeping the eager implementation as an identity is useful for model
initialisation and single-domain testing.
"""

from __future__ import annotations

import math

import jax
from jax import numpy as jnp
from jax._src import core
from jax._src.interpreters import mlir


FORWARD_TARGET = "chemtrain_deploy.gather_forward"
REVERSE_TARGET = "chemtrain_deploy.gather_reverse"
CUSTOM_CALL_TARGETS = (FORWARD_TARGET, REVERSE_TARGET)


def _make_primitive(name: str, target: str):
    """Create a same-shape communication primitive.

    This intentionally follows the OpenEquivariance pattern:
    define the implementation via jax.ffi.ffi_call and lower it using
    mlir.lower_fun. This lets JAX own the StableHLO custom-call shape
    lowering, including dynamic result-shape operands.
    """

    primitive = core.Primitive(name)

    def impl(buffer):
        result_shape = jax.ShapeDtypeStruct(buffer.shape, buffer.dtype)
        call = jax.ffi.ffi_call(
            target,
            result_shape,
            has_side_effect=True,
        )
        return call(buffer)

    def abstract_eval(buffer):
        return core.ShapedArray(buffer.shape, buffer.dtype)

    # Outside a compiled deployment this operation represents communication
    # over an already-global array and is therefore the identity. Keeping the
    # eager path free of FFI also makes reference/numerical tests possible.
    primitive.def_impl(lambda buffer: buffer)
    primitive.def_abstract_eval(abstract_eval)

    lowering = mlir.lower_fun(impl, multiple_results=False)

    # Match the export platform used in exporter.py.
    mlir.register_lowering(primitive, lowering, platform="cuda")

    # Harmless if unused, useful if this exporter is later used on ROCm.
    mlir.register_lowering(primitive, lowering, platform="rocm")

    # CPU lowering is useful for inspecting and testing the exported custom
    # call without attempting to execute an unregistered host handler.
    mlir.register_lowering(primitive, lowering, platform="cpu")

    return primitive


_forward_p = _make_primitive("chemtrain_gather_forward", FORWARD_TARGET)
_reverse_p = _make_primitive("chemtrain_gather_reverse", REVERSE_TARGET)


@jax.custom_vjp
def _communicate(buffer):
    return _forward_p.bind(buffer)


def _communicate_fwd(buffer):
    return _forward_p.bind(buffer), None


def _communicate_bwd(_, cotangent):
    return (_reverse_p.bind(cotangent),)


_communicate.defvjp(_communicate_fwd, _communicate_bwd)


def gather(tree):
    """Gather atom-leading floating-point arrays with one FFI call.

    Leaves are packed into a single ``[n_atoms, packed_width]`` matrix. The
    reverse-mode transpose is a reverse communication call at the matching
    point in backpropagation.
    """

    leaves, treedef = jax.tree.flatten(tree)
    if not leaves:
        raise ValueError("comm.gather requires a non-empty pytree")

    arrays = [jnp.asarray(leaf) for leaf in leaves]
    first = arrays[0]

    if first.ndim < 1:
        raise ValueError("comm.gather leaves must have an atom-leading axis")

    if not jnp.issubdtype(first.dtype, jnp.floating):
        raise TypeError("comm.gather supports floating-point arrays only")

    for array in arrays[1:]:
        if array.ndim < 1:
            raise ValueError(
                "comm.gather leaves must have an atom-leading axis"
            )

        if array.dtype != first.dtype:
            raise TypeError("comm.gather leaves must have the same dtype")

        if array.shape[0] != first.shape[0]:
            raise ValueError(
                "comm.gather leaves must have the same atom-leading size"
            )

    widths = [math.prod(array.shape[1:]) for array in arrays]

    packed = jnp.concatenate(
        [
            array.reshape((array.shape[0], width))
            for array, width in zip(arrays, widths)
        ],
        axis=1,
    )

    communicated = _communicate(packed)

    if communicated.shape != packed.shape:
        raise ValueError(
            "comm.gather changed shape from "
            f"{packed.shape} to {communicated.shape}"
        )

    unpacked = []
    start = 0
    for array, width in zip(arrays, widths):
        unpacked.append(
            communicated[:, start:start + width].reshape(array.shape)
        )
        start += width

    return jax.tree.unflatten(treedef, unpacked)
