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

"""Exporting potential models to MLIR."""

import abc
import functools
import re

import jax
from jax import numpy as jnp, export, lax

from typing import Dict, NamedTuple, Any, List, Tuple, Callable, NoReturn

import jax_md_mod
from jax_md import util as md_util, space

from . import comm, graphs, util
from ._protobuf import model_pb2 as model_proto


OPENEQUIVARIANCE_CUSTOM_CALLS = (
    "conv_forward",
    "conv_backward",
    "conv_double_backward",
)

COMMUNICATION_CUSTOM_CALLS = comm.CUSTOM_CALL_TARGETS

_JAX_INTERNAL_CUSTOM_CALLS = frozenset(
    {"shape_assertion", "stablehlo.dynamic_top_k"}
)
_CUSTOM_CALL_RE = re.compile(
    r'\bstablehlo\.custom_call\s+(?:@(?:"([^"]+)"|([\w.$-]+))|"([^"]+)")'
)


def stablehlo_custom_call_targets(module: bytes | str) -> frozenset[str]:
    """Return exact custom-call target names from a StableHLO module."""
    if isinstance(module, bytes):
        module = module.decode("utf-8")
    return frozenset(
        next(value for value in match if value)
        for match in _CUSTOM_CALL_RE.findall(module)
    )


def validate_custom_calls(
    module: bytes | str, allowed_external_calls=()
) -> None:
    """Reject external calls not explicitly enabled for this export path."""
    targets = stablehlo_custom_call_targets(module)
    allowed = frozenset(allowed_external_calls) | _JAX_INTERNAL_CUSTOM_CALLS
    unknown = targets - allowed
    if unknown:
        raise ValueError(
            "Export contains unsupported custom-call targets: "
            + ", ".join(sorted(unknown))
        )


def validate_openequivariance_custom_calls(module: bytes | str) -> None:
    """Reject cuEquivariance and unapproved external calls in an OEQ export."""
    validate_custom_calls(module, OPENEQUIVARIANCE_CUSTOM_CALLS)


def _openequivariance_disabled_checks():
    """Allow only the OpenEquivariance convolution FFI calls during export."""
    return tuple(
        export.DisabledSafetyCheck.custom_call(target)
        for target in OPENEQUIVARIANCE_CUSTOM_CALLS
    )


def _communication_disabled_checks():
    return tuple(
        export.DisabledSafetyCheck.custom_call(target)
        for target in COMMUNICATION_CUSTOM_CALLS
    )


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to an MLIR module.

    To export a potential model, subclass this class, select an appropriate
    graph type and define the energy function:

    Usage:

        >>> import jax.numpy as jnp
        >>> from jax_md_mod import custom_energy
        >>> from jax_md import partition, space
        >>> from chemtrain.deploy import exporter, graphs
        >>> class LennardJonesExport(exporter.Exporter):
        ...
        ...     graph_type = graphs.SimpleSparseNeighborList
        ...     r_cutoff = 5.0
        ...     unit_style = "real"
        ...     nbr_order = [1, 1]
        ...
        ...     def energy_fn(self, pos, species, graph):
        ...
        ...         neighbors = partition.NeighborList(
        ...             jnp.stack((graph.senders, graph.receivers)),
        ...             pos, None, None, graph.senders.size, partition.Sparse,
        ...             None, None, None
        ...         )
        ...
        ...         assert neighbors.idx.shape[0] == 2, "Wrong shape"
        ...
        ...         displacement_fn, _ = space.free()
        ...         apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
        ...             displacement_fn, None, None,
        ...             sigma=jnp.asarray([3.165]), epsilon=jnp.asarray([1.0]),
        ...             r_onset=4.0, r_cutoff=5.0,
        ...             initialize_neighbor_list=False,
        ...             per_particle=True # Important for export
        ...         )
        ...
        ...         return apply_fn(pos, neighbors, species=species)
        ...
        >>> model = LennardJonesExport()
        >>> try:
        ...     model.save("out.ptb")
        ... except AssertionError as e:
        ...     # We need to call the export method first
        ...     print(f"Error: {e}")
        Error: Model has not been exported yet. Please call `export()` first.
        >>>
        >>> model.export()


    Attributes:
        graph_type: Specifies the required neighborhood representation and
            how to generate it from the input data.
            See ref:`chemtrain.deploy.graphs`.
        nbr_order: List of two integers specifying the number of neighbors
            required for the newton and non-newton setting to correctly
            compute forces.
        r_cutoff: Cutoff radius for the potential.
        unit_style: Specifies the units in which the potential requires
            positions and returns energies. The force units depend solely
            on the length and energy units.
        has_aux: If True, the energy function returns additional quantities
            besides the potential energy as dictionary.

    """

    # Use the default graph containing the full neighbor indices
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    # Order to which neighbors are required for a correct force computation
    # in the newton and the non-newton setting
    nbr_order: List[int] = [1, 1]

    r_cutoff: float
    unit_style: str = "real"
    has_aux: bool = False

    _symbols: List[str] = None
    _constraints: List[str] = None
    _init_fns: List[Callable] = None
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, species, graph):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            species: (N) Array of atoms species.
            graph: Graph representation of the neighborhood around atoms.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    @staticmethod
    @util.define_symbols("n_atoms")
    def _define_position_shapes(n_atoms, **kwargs):
        shape_defs = (
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_atoms,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32), # n_local
            jax.ShapeDtypeStruct((), jnp.int32), # n_ghost
            jax.ShapeDtypeStruct((), jnp.bool_), # newton flag
        )

        return shape_defs

    def _add_shapes(self, init_fn):
        init_fn(self._symbols, self._constraints, self._init_fns)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        # Reset
        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes


    def _energy_fn(self, position, species, n_local, n_ghost, newton, *graph_args):
        # Expects particles to be sorted by local, ghost, and padding atoms

        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        local_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff, self._neighbor_orders(), position, species,
            local_mask, valid_mask, newton, *graph_args)
        graph = lax.stop_gradient(graph)

        @functools.partial(jax.grad, has_aux=True)
        def force_and_aux(pos):
            out = self.energy_fn(pos, species, graph)
            if self.has_aux:
                per_atom_energies, aux = out
            else:
                per_atom_energies = out
                aux = {}

            assert per_atom_energies.shape == local_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {local_mask.shape}."
            )

            # Attention: Force is negative gradient of potential.
            # Depending on the newton flag, we either compute:
            # - the gradient of the _total potential_ w.r.t. the _local atoms_
            # - the gradient of the _local potential_ w.r.t. _all atoms_
            # The latter case equals newton=true and requires additional
            # communication to sum up the forces on the ghost atoms.
            total_energy = md_util.high_precision_sum(
                jnp.where(valid_mask, per_atom_energies, jnp.float32(0.0)))
            local_energy = md_util.high_precision_sum(
                jnp.where(local_mask, per_atom_energies, jnp.float32(0.0))
            )

            force_energy = jnp.where(newton, local_energy, total_energy)
            force_energy = jnp.negative(force_energy)

            # Differentiate w.r.t. the total potential in the box, but exclude
            # ghost atom contributions to the total potential

            return force_energy, (per_atom_energies, aux)

        force, (energy, aux) = force_and_aux(position)
        predictions = dict(U=energy, F=force, **aux)

        for key, value in predictions.items():
            assert value.shape[0] == local_mask.size, (
                f"Wrong shape for prediction {key}. All model outputs "
                f"must be per-atom quantities."
            )

        return predictions, build_statistics

    def _neighbor_orders(self):
        """Return neighbor orders used to construct and serialize the graph."""
        return self.nbr_order

    def export(self) -> None:
        """Exports the potential model to an MLIR module."""

        self._export()

    def export_openequivariance(self) -> None:
        """Export a model containing OpenEquivariance convolution FFI calls.

        The regular :meth:`export` path intentionally retains JAX's strict
        custom-call safety checks. This opt-in path permits only the three
        OpenEquivariance convolution targets registered by chemtrain-deploy.
        """

        self._export(
            disabled_checks=_openequivariance_disabled_checks(),
            validate_openequivariance=True,
        )

    def _export(
        self, *, disabled_checks=(), validate_openequivariance: bool = False
    ) -> None:
        """Shared export implementation with explicitly scoped safety checks."""

        # Create a new context for each export
        self._symbols: List[str] = []
        self._constraints: List[str] = []
        self._init_fns: List[Callable] = []

        proto = model_proto.Model()

        proto.neighbor_list.cutoff = self.r_cutoff
        proto.unit_style = self.unit_style

        neighbor_orders = self._neighbor_orders()
        assert len(neighbor_orders) == 2, (
            "The nbr_order must contain the order of required neighbors for "
            "the newton and non-newton setting."
        )
        proto.neighbor_list.nbr_order.extend(neighbor_orders)
        self.graph_type.set_properties(proto)

        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        self._add_shapes(self._define_position_shapes)
        self._add_shapes(self.graph_type.create_symbolic_input_format)

        export_fn = jax.jit(self._energy_fn)

        shapes = self._create_shapes()

        exp: export.Exported = export.export(
            export_fn,
            platforms=["cuda"],
            disabled_checks=disabled_checks,
        )(*shapes)

        # Reconstruct the output to save the returned statistics and...
        predictions, statistics = exp.out_tree.unflatten(exp.out_avals)

        proto.neighbor_list.statistics_keys.extend(statistics.keys())
        proto.quantities.extend(predictions.keys())

        mlir_module = exp.mlir_module()
        if validate_openequivariance:
            validate_openequivariance_custom_calls(mlir_module)
        proto.mlir_module = mlir_module

        self._proto = proto

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        return str(self._proto)

    def save(self, file: str) -> None:
        """Save the exported protocol buffer to ``file``."""
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())


def stablehlo_custom_call_count(module: bytes | str, target: str) -> int:
    """Count StableHLO custom calls by target name."""
    if isinstance(module, bytes):
        module = module.decode("utf-8")

    count = 0

    count += len(re.findall(
        rf'call_target_name\s*=\s*"{re.escape(target)}"',
        module,
    ))

    count += len(re.findall(
        rf'\bstablehlo\.custom_call\s+@(?:"{re.escape(target)}"|{re.escape(target)})\b',
        module,
    ))

    return count


def infer_communication_buffer_width(module: bytes | str) -> int:
    """Infer the packed communication buffer width from StableHLO text."""
    if isinstance(module, bytes):
        module = module.decode("utf-8")

    targets = "|".join(
        re.escape(target) for target in COMMUNICATION_CUSTOM_CALLS
    )

    widths: list[int] = []

    quoted_blocks = re.findall(
        rf'(?s)'
        rf'"stablehlo\.custom_call"\(.*?'
        rf'call_target_name\s*=\s*"({targets})".*?'
        rf'\)\s*:\s*\((.*?)\)\s*->\s*(?:\((.*?)\)|(tensor<[^>]+>))',
        module,
    )

    for _, operand_types, result_tuple_types, result_single_type in quoted_blocks:
        block = " ".join(
            part for part in (
                operand_types,
                result_tuple_types,
                result_single_type,
            )
            if part
        )
        widths.extend(
            int(width)
            for width in re.findall(
                r'tensor<[^>]*x(\d+)xf(?:32|64)>',
                block,
            )
        )

    direct_blocks = re.findall(
        rf'(?s)'
        rf'stablehlo\.custom_call\s+@(?:"(?:{targets})"|(?:{targets}))'
        rf'.*?'
        rf'\)\s*:\s*\((.*?)\)\s*->\s*(?:\((.*?)\)|(tensor<[^>]+>))',
        module,
    )

    for operand_types, result_tuple_types, result_single_type in direct_blocks:
        block = " ".join(
            part for part in (
                operand_types,
                result_tuple_types,
                result_single_type,
            )
            if part
        )
        widths.extend(
            int(width)
            for width in re.findall(
                r'tensor<[^>]*x(\d+)xf(?:32|64)>',
                block,
            )
        )

    if not widths:
        raise ValueError(
            "Could not infer the packed communication width from the "
            "exported module"
        )

    return max(widths)


class CommunicationExporter(Exporter):
    """Exporter for models containing :func:`chemtrain.deploy.comm.gather`.

    Communication synchronizes ghost features at message-passing boundaries.
    Newton-on requires a one-cutoff halo because forces are reverse summed;
    Newton-off computes all influencing ghost energies locally and retains the
    communication-free ``2 * num_interactions`` halo. A provisional export
    counts communication sites to infer the latter order.
    """

    def export(self) -> None:
        self._export_communicating()

    def export_openequivariance(self) -> None:
        self._export_communicating(openequivariance=True)

    def _neighbor_orders(self):
        return self._inferred_neighbor_orders

    def _export_communicating(self, openequivariance: bool = False) -> None:
        communication_checks = _communication_disabled_checks()
        oeq_checks = (
            _openequivariance_disabled_checks() if openequivariance else ()
        )
        checks = communication_checks + oeq_checks

        self._inferred_neighbor_orders = [1, 1]
        super()._export(disabled_checks=checks)

        provisional_module = self._proto.mlir_module
        gather_count = stablehlo_custom_call_count(
            provisional_module, comm.FORWARD_TARGET
        )

        # A gather is inserted before every interaction after the first, so
        # num_interactions = gather_count + 1 for this interface.
        self._inferred_neighbor_orders = [1, 2 * (gather_count + 1)]
        super()._export(disabled_checks=checks)

        final_module = self._proto.mlir_module
        final_gather_count = stablehlo_custom_call_count(
            final_module, comm.FORWARD_TARGET
        )

        if final_gather_count != gather_count:
            raise ValueError(
                "Communication structure changed while inferring neighbor "
                f"orders: provisional gather_count={gather_count}, "
                f"final_gather_count={final_gather_count}"
            )

        allowed = set(COMMUNICATION_CUSTOM_CALLS)
        if openequivariance:
            allowed.update(OPENEQUIVARIANCE_CUSTOM_CALLS)

        validate_custom_calls(final_module, allowed)

        self._proto.uses_communication = True
        self._proto.communication_buffer_width = (
            infer_communication_buffer_width(final_module)
        )
        self.gather_count = gather_count
