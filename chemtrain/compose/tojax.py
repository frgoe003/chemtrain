"""Loads MACE torch checkpoints through the toJax patching path."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import numpy as np

import jax
import jax.core as jcore
import jax.numpy as jnp

from jax_md import partition, space
from jax_md_mod import custom_partition

from mace_jax.cli import mace_jax_from_torch
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

from tojax import tojax
from tojax.data import tojax_data
import tojax.patches as tojax_patches
from tojax.patches import patch_module
from tojax.wrapper import TensorWrapper, jax_dtype, unwrap, wrap

from .mace_jax import AtomicNumberMapping, SpeciesMapping

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import e3nn.nn._extract as e3nn_extract
import e3nn.o3 as e3nn_o3
import mace.modules.models as mace_models
import mace.modules.utils as mace_utils
import torch


def _to_jax_array(value):
    if isinstance(value, TensorWrapper):
        return unwrap(value)
    if isinstance(value, torch.Tensor):
        return tojax_data(value)
    return value


class _RankPreservingE3nnLinear(torch.nn.Module):
    """JAX path for e3nn Linear that avoids flattening dynamic leading axes."""

    def __init__(self, module: e3nn_o3.Linear, fallback: torch.nn.Module):
        super().__init__()
        self.irreps_in = module.irreps_in
        self.irreps_out = module.irreps_out
        self.instructions = tuple(module.instructions)
        self.shared_weights = module.shared_weights
        self.fallback = fallback

    def forward(self, features, weight=None, bias=None):
        if not isinstance(features, TensorWrapper):
            return self.fallback(features, weight, bias)

        if not self.shared_weights:
            raise NotImplementedError(
                "toJAX rank-preserving e3nn Linear currently requires shared weights."
            )

        x = unwrap(features)
        w_flat = _to_jax_array(weight)
        b_flat = None if bias is None else _to_jax_array(bias)
        leading_shape = x.shape[:-1]

        x_blocks = []
        for segment, mul_ir in zip(self.irreps_in.slices(), self.irreps_in):
            block = x[..., segment.start : segment.stop]
            if mul_ir.ir.dim == 1:
                x_blocks.append(block)
            else:
                x_blocks.append(
                    block.reshape(leading_shape + (mul_ir.mul, mul_ir.ir.dim))
                )

        out_blocks = [[] for _ in self.irreps_out]
        flat_weight_index = 0
        flat_bias_index = 0
        for ins in self.instructions:
            out_ir = self.irreps_out[ins.i_out]
            if ins.i_in == -1:
                if b_flat is None:
                    continue
                block_size = out_ir.dim
                b = b_flat[flat_bias_index : flat_bias_index + block_size]
                flat_bias_index += block_size
                out = ins.path_weight * b.reshape((1,) * len(leading_shape) + (block_size,))
                out_blocks[ins.i_out].append(jnp.broadcast_to(out, leading_shape + (block_size,)))
                continue

            in_ir = self.irreps_in[ins.i_in]
            path_nweight = int(np.prod(ins.path_shape))
            w = w_flat[flat_weight_index : flat_weight_index + path_nweight]
            w = w.reshape(ins.path_shape)
            flat_weight_index += path_nweight

            if in_ir.ir.dim == 1:
                out = jnp.einsum("uw,...u->...w", w, x_blocks[ins.i_in])
                out_blocks[ins.i_out].append(ins.path_weight * out)
            else:
                out = jnp.einsum("uw,...ui->...wi", w, x_blocks[ins.i_in])
                out = ins.path_weight * out
                out_blocks[ins.i_out].append(out.reshape(leading_shape + (out_ir.dim,)))

        outputs = []
        for pieces, out_ir in zip(out_blocks, self.irreps_out):
            if out_ir.mul == 0:
                continue
            if pieces:
                total = pieces[0]
                for piece in pieces[1:]:
                    total = total + piece
            else:
                total = jnp.zeros(leading_shape + (out_ir.dim,), dtype=x.dtype)
            outputs.append(total)

        if len(outputs) == 1:
            return TensorWrapper(outputs[0])
        return TensorWrapper(jnp.concatenate(outputs, axis=-1))


def _patch_e3nn_linear_rank_preserving(module: e3nn_o3.Linear):
    module._compiled_main = _RankPreservingE3nnLinear(module, module._compiled_main)
    return module


tojax_patches._PATCHES[e3nn_o3.Linear] = _patch_e3nn_linear_rank_preserving


if not getattr(TensorWrapper, "_chemtrain_mace_compat", False):
    def _new_zeros(self, *size, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.zeros(
                size,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    def _new_ones(self, *size, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.ones(
                size,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    def _new_full(self, size, fill_value, *, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.full(
                size,
                fill_value,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    TensorWrapper.new_zeros = _new_zeros
    TensorWrapper.new_ones = _new_ones
    TensorWrapper.new_full = _new_full
    TensorWrapper._chemtrain_mace_compat = True


if not getattr(e3nn_extract.Extract, "_chemtrain_mace_compat", False):

    def _extract_forward(self, x):
        is_wrapped = isinstance(x, TensorWrapper)
        data = unwrap(x) if is_wrapped else x

        outs = []
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            pieces = []
            for s_out, i_in in zip(irreps_out.slices(), ins):
                del s_out
                i_start = self.irreps_in[:i_in].dim
                i_len = self.irreps_in[i_in].dim
                pieces.append(data[..., i_start : i_start + i_len])

            if is_wrapped:
                outs.append(jnp.concatenate(pieces, axis=-1))
            else:
                outs.append(torch.cat(pieces, dim=-1))

        if len(outs) == 1:
            return wrap(outs[0]) if is_wrapped else outs[0]
        return tuple(wrap(out) for out in outs) if is_wrapped else tuple(outs)

    e3nn_extract.Extract.forward = _extract_forward
    e3nn_extract.Extract._chemtrain_mace_compat = True



import mace.modules.irreps_tools as irreps_tools
if getattr(irreps_tools, "_chemtrain_mace_compat_mask", True):
    def new_mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Memory-light mirror of MACE's torch mask_head implementation."""

        if isinstance(x, TensorWrapper):
            x_arr = unwrap(x)
            width = x_arr.shape[1]
            h = unwrap(head) if isinstance(head, TensorWrapper) else head
            h = jnp.asarray(unwrap(h), dtype=jnp.int32)

            grouped = jnp.reshape(x_arr, (x_arr.shape[0], num_heads, width // num_heads))
            head_ids = jnp.arange(num_heads, dtype=jnp.int32)
            if h.ndim == 0 or h.size == 1:
                head_mask = head_ids == jnp.reshape(h, ())[None]
                head_mask = jnp.reshape(head_mask, (1, num_heads, 1))
            else:
                head_mask = head_ids[None, :] == jnp.reshape(h, (-1, 1))
                head_mask = head_mask[:, :, None]

            grouped = grouped * head_mask.astype(x_arr.dtype)
            return TensorWrapper(jnp.reshape(grouped, x_arr.shape))

        width = x.shape[1]
        grouped = x.reshape(x.shape[0], num_heads, width // num_heads)
        head_ids = torch.arange(num_heads, device=x.device)
        if isinstance(head, torch.Tensor) and head.numel() != 1:
            head_mask = head_ids[None, :] == head.reshape(-1, 1).to(device=x.device)
            head_mask = head_mask[:, :, None]
        else:
            head_scalar = head.reshape(()) if isinstance(head, torch.Tensor) else head
            head_mask = (head_ids == head_scalar).reshape(1, num_heads, 1)
        return (grouped * head_mask.to(dtype=x.dtype)).reshape(x.shape)
    irreps_tools.mask_head = new_mask_head
    import mace.modules.blocks as blocks
    blocks.mask_head = new_mask_head
    irreps_tools._chemtrain_mace_compat_mask = False

def load_foundational_model(family: str = "mp", version: str = "medium-0b3"):
    """Load a foundation MACE torch model and extract its conversion config."""

    torch_model = mace_jax_from_torch._load_torch_model_from_foundations(
        family, version
    )
    torch_model = torch_model.to(dtype=torch.float32)
    torch_model.eval()

    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])

    return torch_model, config


def tojax_vectors_from_torch(
    config: Dict[str, Any],
    torch_model: Any,
    *,
    per_particle: bool = False,
    scale_pot: float = 96.485,
    species_mapping: SpeciesMapping = SpeciesMapping(),
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    head: str | None = None,
) -> Tuple[Any, Callable]:
    """Wrap a torch MACE model as a vector-first JAX callable via toJax."""

    del cueq_config, use_custom_batch_fn

    torch_model = patch_module(deepcopy(torch_model).to(dtype=torch.float32))
    torch_model = torch_model.to(dtype=torch.float32)
    torch_model.eval()

    heads = tuple(str(h) for h in (config.get("heads") or ("Default",)))
    head_name = heads[0] if head is None else str(head)
    if head_name not in heads:
        raise ValueError(
            f"Requested head '{head_name}' not present in model heads {heads}."
        )
    head_index = heads.index(head_name)
    num_species = len(torch_model.atomic_numbers)

    def _predict(vectors, senders, receivers, species, mask):
        num_atoms = species.shape[0]

        node_attrs = torch.nn.functional.one_hot(species, num_species).to(vectors.dtype)
        node_attrs = node_attrs * mask[:, None]

        data = {
            # "vectors": vectors,
            "node_attrs": node_attrs,
            "node_attrs_index": species,
            "species": species,
            "edge_index": torch.stack((senders, receivers), dim=0),
            "batch": torch.zeros((num_atoms,), dtype=torch.int32),
            "natoms": num_atoms,
            "ptr": torch.tensor([0, num_atoms], dtype=torch.int32),
            "positions": torch.zeros((num_atoms, 3), dtype=vectors.dtype),
            "unit_shifts": torch.zeros_like(vectors),
            "shifts": vectors, # torch.zeros_like(vectors),
            "cell": torch.zeros((1, 3, 3), dtype=vectors.dtype),
            "head": torch.tensor([head_index], dtype=torch.int32),
            "lammps_class": None,
        }

        out = torch_model(
            data,
            compute_force=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        return out["node_energy"] * mask

    _apply_fn = jax.jit(tojax(_predict))

    def apply_fn(
        params: Any,
        vectors: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        species: jax.Array,
        mask: jax.Array | None = None,
    ):
        del params

        if isinstance(species_mapping, AtomicNumberMapping) and not isinstance(
            species, jcore.Tracer
        ):
            atomic_numbers = np.asarray(config["atomic_numbers"], dtype=np.int32)
            species_np = np.asarray(species, dtype=np.int32)
            valid = np.isin(species_np, atomic_numbers)
            if not np.all(valid):
                invalid = sorted({int(value) for value in species_np[~valid]})
                raise ValueError(
                    "Species contains atomic numbers not supported by the MACE model: "
                    f"{invalid}. Supported atomic numbers are "
                    f"{atomic_numbers.tolist()}."
                )

        if mask is None:
            mask = jnp.ones(species.shape[0], dtype=jnp.bool_)

        mapped_species = species_mapping(species, config)
        per_atom_energies = _apply_fn(
            vectors,
            senders,
            receivers,
            mapped_species,
            mask,
        )
        per_atom_energies *= scale_pot

        if per_particle:
            return per_atom_energies
        return jnp.sum(per_atom_energies)

    return None, apply_fn


def tojax_neighborlist_from_torch(
    config: Dict[str, Any],
    torch_model: Any,
    displacement: space.DisplacementFn,
    max_edge_multiplier: float = 1.25,
    per_particle: bool = False,
    scale_pos: float = 0.1,
    scale_pot: float = 96.485,
    species_mapping: SpeciesMapping = SpeciesMapping(),
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    head: str | None = None,
) -> Tuple[Any, Callable]:
    """Compose a torch MACE checkpoint with a chemtrain neighborlist frontend."""

    variables, apply_fn = tojax_vectors_from_torch(
        config,
        torch_model,
        per_particle=per_particle,
        scale_pot=scale_pot,
        species_mapping=species_mapping,
        cueq_config=cueq_config,
        use_custom_batch_fn=use_custom_batch_fn,
        head=head,
    )

    r_cutoff = jnp.asarray(config["r_max"], dtype=jnp.float32) * scale_pos
    edges_per_particle = float(config["avg_num_neighbors"]) * float(max_edge_multiplier)

    def apply_neighbor_fn(
        params: Any,
        position: jax.Array,
        neighbor: partition.NeighborList,
        species: jax.Array = None,
        mask: jax.Array = None,
        **dynamic_kwargs,
    ):
        assert species is not None, "Species must be provided."
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement,
            r_cutoff,
            position,
            neighbor,
            species,
            mask,
            edges_per_particle=None, # edges_per_particle,
            sort=True,
            **dynamic_kwargs,
        )

        vectors /= scale_pos
        return apply_fn(
            params,
            vectors,
            senders,
            receivers,
            species,
            mask=mask,
        )

    return variables, apply_neighbor_fn


mace_jax_neighborlist_from_torch = tojax_neighborlist_from_torch


__all__ = [
    "AtomicNumberMapping",
    "SpeciesMapping",
    "load_foundational_model",
    "mace_jax_neighborlist_from_torch",
    "tojax_neighborlist_from_torch",
    "tojax_vectors_from_torch",
]
