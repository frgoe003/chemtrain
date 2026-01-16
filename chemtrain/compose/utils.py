
# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Utilities to connect models to chemtrain."""

import numpy as onp

import jax
import jax.numpy as jnp
from jax import lax

from typing import Protocol, Any, Tuple


class ApplyFn(Protocol):
    """GNN apply function protocol."""
    def __call__(
        self,
        params: Any,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_features: Tuple[jnp.ndarray],
        node_features: Tuple[jnp.ndarray],
    ) -> jnp.ndarray: ...



def batch_apply_fn(_apply_fn: ApplyFn) -> ApplyFn:
    """Write custom vmap rules for the apply function.

    Instead of batching over graphs, combines all graphs into a supergraph.
    This step avoids vmapping operations in the neural network.

    Args:
        apply_fn: Mapping from (params, vectors, senders, receivers, species, mask)
            to per-particle energies.


    Returns:
        A wrapped apply function with custom vmap rules that avoids vmapping
        in the neural network.

    """

    def apply_fn(*args, **kwargs):
        res = _apply_fn(*args, **kwargs)
        return res


    def wrapped(params, senders, receivers, edge_features, node_features):
        # Check inputs. Node features are important to correctly rewrite invalid
        # indices in the supergraph.
        assert len(node_features) > 0, "At least one node feature array is required."
        
        return wrapped_apply_fn(
            params, senders, receivers, edge_features, node_features
        )


    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def wrapped_apply_fn(params, senders, receivers, edge_features, node_features):
        return apply_fn(params, senders, receivers, edge_features, node_features)
        
    def wrapped_fun_fwd(*args):
        y = wrapped_apply_fn(*args)
        return y, args

    @jax.custom_batching.custom_vmap
    def wrapped_fun_bwd(res, y_bar):        
        _, vjp_fn = jax.vjp(apply_fn, *res)

        return vjp_fn(y_bar)
    
    @wrapped_fun_bwd.def_vmap
    def wrapped_fun_bwd_batch(
            axis_size, in_batched, res, b_y_bar):
        
        args_batched, y_bar_batched = in_batched
        bparams, *inputs_batched = args_batched
        params, *graph_args = res
        
        flattened_graph, graph_shapes = flatten_graph(
            axis_size, inputs_batched, *graph_args
        )
        senders_flat, receivers_flat, edge_features_flat, node_features_flat = \
            flattened_graph
        num_graphs, num_edges, natoms = graph_shapes
 
        if not y_bar_batched:
            # Tile the cotangent if it is not batched
            if b_y_bar.shape[0] == natoms: 
                y_bar_flat = jnp.tile(b_y_bar, (num_graphs,))
            else:
                y_bar_flat = b_y_bar
        else:
            y_bar_flat = jnp.reshape(b_y_bar, (-1,))

        # TODO: Can we also just apply the vjp to the batched fwd function to
        #       save some operations?
        
        # TODO: Correct the param batching logic here....
        #       Params could in principle be batched INSIDE the jvp function.

        # Let jax figure out the vjp on the flattened supergraph

        if jax.tree.leaves(bparams)[0]:
            _, vjp_fn = jax.vjp(
                lambda *args: lax.map(
                    lambda p: wrapped_apply_fn(p, *args[1:]), args[0]
                ),
                params, senders_flat, receivers_flat, edge_features_flat,
                node_features_flat
            )
        else:
            _, vjp_fn = jax.vjp(
                apply_fn, params, senders_flat, 
                receivers_flat, edge_features_flat, node_features_flat
            )

        grads = vjp_fn(y_bar_flat)
        
        g_params, g_senders, g_receivers, g_edge_features, g_node_features = grads

        # Reshape the gradients back to the batched shape
        out_grads = (
            g_params,
            g_senders.reshape((num_graphs, num_edges)),
            g_receivers.reshape((num_graphs, num_edges)),
            tuple(
                g_edge_feat.reshape((num_graphs, num_edges) + g_edge_feat.shape[1:])
                for g_edge_feat in g_edge_features
            ),
            tuple(
                g_node_feat.reshape((num_graphs, natoms) + g_node_feat.shape[1:])
                for g_node_feat in g_node_features
            )
        )

        # Return which of the outputs are batched
        out_batched = (
            bparams, True, True,
            (True,) * len(g_edge_features),
            (True,) * len(g_node_features)
        )

        return out_grads, out_batched

    wrapped_apply_fn.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)

    @wrapped_apply_fn.def_vmap
    def wrapped_fun_batch(
            axis_size, in_batched, params, senders, receivers, edge_features, node_features):

        bparams, *inputs_batched = in_batched

        # Find out whether any of the parameters is batched
        batched_params = jax.tree.reduce(
            onp.logical_or, bparams, onp.bool(False)
        )
 
        # TODO: Is is possible that both params and graphs are batched at the
        #       same time?
        #       It seems possible. In that case, we just tile all parameters
        #       and apply a complete lax.map.

        if batched_params:
            print("Found batched parameters.")
            # Ensure that all params are in the batched shape
            args_tiled = jax.tree.map(
                lambda l, b: jnp.tile(l, (axis_size,) + (1,) * (l.ndim - 1))
                if not b else l,
                (params, senders, receivers, edge_features, node_features),
                (bparams, *inputs_batched),
            )

            energies = lax.map(
                lambda args: wrapped_apply_fn(*args), args_tiled
            )

            return energies, True
        else:
            
            flattened_graph, graph_shapes = flatten_graph(
                axis_size, inputs_batched, senders, receivers, edge_features,
                node_features
            )
            senders, receivers, edge_features_flat, node_features_flat = \
                flattened_graph
            num_graphs, num_edges, natoms = graph_shapes


            energies_flat = wrapped_apply_fn(
                params, senders, receivers, edge_features_flat,
                node_features_flat
            )

            # Unflatten the results
            return energies_flat.reshape((axis_size, -1)), True

    return wrapped


def flatten_graph(axis_size, in_batched, senders, receivers, edge_features, node_features):
        """Flattens batched graphs into a supergraph."""

        bsenders, breceivers, bedge_features, bnode_features = in_batched

        num_graphs = axis_size
        num_edges = (
            senders.shape[1] if bsenders
            else senders.shape[0]
        )
        natoms = (
            node_features[0].shape[1] if bnode_features[0]
            else node_features[0].shape[0]
        )

        if bsenders:
            assert breceivers, (
                "If vectors are batched, senders and receivers must be batched."
            )

        else:
            assert not breceivers, (
                "If vectors are not batched, senders and receivers must not be batched."
            )

            senders = jnp.tile(
                senders[None, :], (num_graphs, 1) + (1,) * (senders.ndim -1)
            )
            receivers = jnp.tile(
                receivers[None, :], (num_graphs, 1) + (1,) * (receivers.ndim -1)
            )            

        # Relabel the senders and receiver indices. Offset the senders
        # by the number of atoms in previous graphs.
        senders = jnp.where(
            senders.ravel() < natoms,
            senders.ravel() + natoms * jnp.repeat(jnp.arange(num_graphs), num_edges),
            num_graphs * natoms
        )
        receivers = receivers.reshape((-1, 2))
        receivers = jnp.where(
            receivers.ravel() < natoms,
            receivers.ravel() + natoms * jnp.repeat(jnp.arange(num_graphs), num_edges),
            num_graphs * natoms
            )

        edge_features_flat = ()
        for b, feat in zip(bedge_features, edge_features):
            if b:
                edge_features_flat += (jnp.reshape(feat, (-1,) + feat.shape[2:]),)
            else:
                edge_features_flat += (jnp.tile(
                    feat[None, :], (num_graphs, 1) + (1,) * (feat.ndim -1)
                ).ravel(),)

        node_features_flat = ()
        for b, feat in zip(bnode_features, node_features):
            if b:
                node_features_flat += (jnp.reshape(feat, (-1,) + feat.shape[2:]),)
            else:
                node_features_flat += (jnp.tile(
                    feat[None, :], (num_graphs, 1) + (1,) * (feat.ndim -1)
                ).ravel(),)

        return (senders, receivers, edge_features_flat, node_features_flat), (num_graphs, num_edges, natoms)
