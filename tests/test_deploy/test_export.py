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
import itertools

import jax
from jax import numpy as jnp, Array

from chemtrain.deploy import comm, graphs, exporter

from jax_md_mod import custom_partition
from jax_md import space, partition, util as md_util

import numpy as onp

import pytest


def model_neighborlist_pp(displacement: space.DisplacementFn,
                          r_cutoff: float,
                          positions_test: jnp.ndarray = None,
                          neighbor_test: partition.NeighborList = None,
                          max_edge_multiplier: float = 1.25,
                          max_edges=None,
                          avg_num_neighbors: float = None,
                          ):
    """Export test model."""
    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        _avg_num_neighbors, _ = custom_partition.test_graph_statistics(
            displacement, positions_test, neighbor_test,
            r_cutoff, max_edge_multiplier=max_edge_multiplier
        )

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors

    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement, r_cutoff, position, neighbor, species,
            mask, max_edges=max_edges, **dynamic_kwargs
        )

        vectors /= r_cutoff

        pot = (jnp.linalg.norm(vectors, axis=-1) - 1.0) ** 2
        return jax.ops.segment_sum(pot, senders, num_segments=position.shape[0])

    return jax.jit(model)





class TestExport:

    @pytest.fixture(scope="function")
    def setup_export(self):
        class ExportedModel(exporter.Exporter):

            graph_type = graphs.SimpleSparseNeighborList
            r_cutoff = 5.0
            unit_style = "real"
            nbr_order = [1, 1]

            def __init__(self, max_edge_multiplier=None, max_edges=None):
                self.max_edge_multiplier = max_edge_multiplier
                self.max_edges = max_edges

            def energy_fn(self, pos, species,
                          graph: graphs.SimpleSparseNeighborList):
                neighbors = graph.to_neighborlist()
                displacement_fn, _ = space.free()

                model = model_neighborlist_pp(
                    displacement_fn, self.r_cutoff,
                    max_edges=self.max_edges, avg_num_neighbors=20.0
                )

                pot = model(pos, neighbors, species=species)
                return pot

        yield ExportedModel

    def test_no_max_edges(self, tmp_path, setup_export):
        model = setup_export(max_edges=None)

        model.export()
        targets = exporter.stablehlo_custom_call_targets(model._proto.mlir_module)
        assert "shape_assertion" in targets
        assert "stablehlo.dynamic_top_k" in targets
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_export_paths_scope_openequivariance_checks(
            self, monkeypatch, setup_export):
        model = setup_export(max_edges=None)
        calls = []

        monkeypatch.setattr(
            model,
            "_export",
            lambda **kwargs: calls.append(kwargs.get("disabled_checks", ())),
        )

        model.export()
        model.export_openequivariance()

        assert calls[0] == ()
        assert tuple(str(check) for check in calls[1]) == tuple(
            f"custom_call:{target}"
            for target in exporter.OPENEQUIVARIANCE_CUSTOM_CALLS
        )

    def test_openequivariance_custom_call_validation(self):
        mlir = """
          %0 = stablehlo.custom_call @shape_assertion(%arg0)
          %1 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0)
          %2 = stablehlo.custom_call @conv_forward(%arg0)
        """
        exporter.validate_openequivariance_custom_calls(mlir)
        assert exporter.stablehlo_custom_call_targets(mlir) == {
            "shape_assertion", "stablehlo.dynamic_top_k",
            "conv_forward",
        }

    @pytest.mark.parametrize(
        "target", ["tp_forward", "tp_backward", "tp_double_backward",
                   "cuequivariance_tp", "unknown_ffi"]
    )
    def test_openequivariance_custom_call_validation_rejects_unknown(self, target):
        mlir = f"stablehlo.custom_call @{target}(%arg0)"
        with pytest.raises(ValueError, match=target):
            exporter.validate_openequivariance_custom_calls(mlir)

    def test_symbolic_max_edges(self, tmp_path, setup_export):
        class ExportedModelSymbolic(setup_export):

            def __init__(self):
                super().__init__(max_edge_multiplier=0.5, max_edges=None)

        model = ExportedModelSymbolic()

        model.export()
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_static_max_edges(self, tmp_path, setup_export):
        # This test ensures that the maximum number of edges during export
        # must be symbolic. If it is static, the number of edges would be
        # also fixed during execution.

        class ExportedModelStatic(setup_export):
            def __init__(self):
                super().__init__(max_edge_multiplier=None, max_edges=10)

        model = ExportedModelStatic()

        with pytest.raises(TypeError, match="max_edges must be symbolic"):
            model.export()

        with pytest.raises(AssertionError, match="has not been exported yet"):
            model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_communication_export_infers_neighbor_orders(self, setup_export):
        class CommunicatingModel(exporter.CommunicationExporter, setup_export):
            def energy_fn(self, pos, species, graph):
                energy = super().energy_fn(pos, species, graph)
                return comm.gather(comm.gather(energy))

        model = CommunicatingModel(max_edges=None)
        model.export()

        assert model.gather_count == 2
        assert list(model._proto.neighbor_list.nbr_order) == [1, 6]
        assert model._proto.uses_communication
        targets = exporter.stablehlo_custom_call_targets(
            model._proto.mlir_module
        )
        assert set(comm.CUSTOM_CALL_TARGETS) <= targets
