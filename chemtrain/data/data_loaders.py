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
import functools
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, NamedTuple, Optional, cast

try:
    import mpi4py
except ImportError:
    mpi4py = None

import h5py
import jax
from jax import numpy as jnp, random
import numpy as onp

from jax_sgmc.data import numpy_loader, core

from chemtrain.data.preprocessing import train_val_test_split
from chemtrain import util
from chemtrain import config as chemtrain_config

PyTree = Any


class HDF5ParallelDataLoader(numpy_loader.NumpyDataLoader):
    """DataLoader that can be used in distributed settings and reads data from HDF5 files.

    This DataLoader is designed to be used in distributed settings, where multiple
    processes are running in parallel. It ensures that each process gets a different
    subset of the data, and that the data is shuffled differently for each process.

    Args:
        file: HDF file containing the entries of the dataset as root datasets.
        strict_order: Whether to strictly enforce the order of the data.
            If False, the indices for the batches are redistributed.

    """

    def __init__(self, file, strict_order: bool = False):
        # The sample is necessary to return the observations in the correct format.
        super().__init__()

        if isinstance(file, h5py.File):
            self._dataset = file
        else:
            self._dataset = h5py.File(name=file, mode="r")

        root_datasets = {
            key: val.shape[0] for key, val in self._dataset.items()
            if isinstance(val, h5py.Dataset)
        }

        assert len(set(root_datasets.values())) == 1, \
            "All datasets in the HDF5 file must have the same length."

        self._observation_count = list(root_datasets.values())[0]
        self._keys = list(root_datasets.keys())

        self._format_cache = {
            key: jax.ShapeDtypeStruct(
                dtype=onp.dtype(cast(h5py.Dataset, self._dataset[key]).dtype),
                shape=tuple(int(s) for s in cast(h5py.Dataset, self._dataset[key]).shape[1:]),
            )
            for key in self._keys
        }

        self._strict_order = strict_order

        # Clone to use communicator with mpi4py
        comm = util.get_communicator()
        if comm is not None:
            self._comm = comm.Clone()
        else:
            self._comm = None

    def is_root(self):
        if self._comm is None:
            return False
        return self._comm.Get_rank() == 0

    def get_batches(self, chain_id: int) -> PyTree:
        """Draws a batch from a chain.

        Args:
        chain_id: ID of the chain, which holds the information about the form of
            the batch and the process of assembling.

        Returns:
        Returns a superbatch as registered by :func:`register_random_pipeline` or
        :func:`register_ordered_pipeline` with `cache_size` batches holding
        `mb_size` observations.

        """
        # Data slicing is the same for all methods of random and ordered access,
        # only the indices for slicing differ. The method _get_indices find the
        # correct method for the chain.

        if self._comm is None:
            selections_idx, selections_mask = self._get_indices(chain_id)
        else:
            if self.is_root():
                selections_idx, selections_mask = self._get_indices(chain_id)

                # We need to slice the data across the batch dimension and not the
                # cache dimension.
                selections_idx = onp.ascontiguousarray(
                    onp.asarray(selections_idx, dtype=onp.int32).swapaxes(0, 1)
                )
                selections_mask = onp.ascontiguousarray(
                    onp.asarray(selections_mask, dtype=onp.bool_).swapaxes(0, 1)
                )
                batch_size, cache_size = selections_idx.shape
            else:
                selections_idx = None
                selections_mask = None
                batch_size, cache_size = None, None

            batch_size = self._comm.bcast(batch_size, root=0)
            cache_size = self._comm.bcast(cache_size, root=0)
            world_size = self._comm.Get_size()

            if batch_size is None or cache_size is None:
                raise RuntimeError("Failed to broadcast batch metadata from root process.")
            if batch_size < world_size:
                raise ValueError(
                    f"mb_size ({batch_size}) must be >= number of MPI processes ({world_size})."
                )
            if batch_size % world_size != 0:
                raise ValueError(
                    f"mb_size ({batch_size}) must be divisible by number of MPI processes ({world_size})."
                )

            slice_size = batch_size // world_size

            recv = onp.empty((slice_size, cache_size), dtype=onp.int32)
            recv_mask = onp.empty((slice_size, cache_size), dtype=onp.bool_)

            # Note: mpi4py Scatter/Scatterv write into the receive buffer and
            # return None.
            self._comm.Scatter(selections_idx, recv, root=0)
            self._comm.Scatter(selections_mask, recv_mask, root=0)

            selections_idx = recv
            selections_mask = recv_mask

        selections_idx = onp.asarray(selections_idx, dtype=onp.int32)
        selections_mask = onp.asarray(selections_mask, dtype=onp.bool_)
        selections_idx = selections_idx.swapaxes(0, 1)
        selections_mask = selections_mask.swapaxes(0, 1)

        restore_shape = selections_idx.shape
        unique, restore = onp.unique(selections_idx.ravel(), return_inverse=True)

        # Slice the data and transform into pytree
        def _read_leaf(leaf_name: str) -> jax.Array:
            dataset = 
            per_observation_shape = tuple(int(s) for s in dataset.shape[1:])
            target_shape = tuple(selections_idx.shape) + per_observation_shape

            restored = dataset[unique][restore]
            return jnp.asarray(restored.reshape(target_shape))

        selected_observations = {
            leaf_name: jnp.asarray(
                cast(h5py.Dataset, self._dataset[leaf_name])
                [unique][restore].reshape(restore_shape + self._format_cache[leaf_name].shape)
            )  for leaf_name in self._keys
        }

        return selected_observations, jnp.array(selections_mask, dtype=jnp.bool_)

    def save_state(self, chain_id: int) -> PyTree:
        raise NotImplementedError("Saving of the DataLoader state is not supported.")

    def load_state(self, chain_id: int, data) -> None:
        raise NotImplementedError("Loading of the DataLoader state is not supported.")

    @property
    def _format(self):
        """Returns shape and dtype of a single observation."""
        return self._format_cache

    @property
    def static_information(self):
        """Returns information about total samples count and batch size. """
        information = {
            "observation_count": self._observation_count
        }
        return information

    def close(self):
        self._dataset.close()





class DataLoaders(NamedTuple):
    train_loader: core.DataLoader
    val_loader: core.DataLoader
    test_loader: core.DataLoader


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False):
    """Splits dataset and initializes dataloaders.

    If the validation or test ratios are 0, returns None for the respective
    dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
            returns batches with the same kwargs as provided in dataset.
        train_ratio: Fraction of dataset to use for training.
        val_ratio: Fraction of dataset to use for validation.
        shuffle: Whether to shuffle data before splitting into train-val-test.

    Returns:
        Returns a tuple ``(train_loader, val_loader, test_loader)`` of
        NumpyDataLoaders.

    """
    def init_subloader(data_subset):
        if data_subset is None:
            loader = None
        else:
            loader = numpy_loader.NumpyDataLoader(**data_subset, copy=False)
        return loader

    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio, val_ratio, shuffle=shuffle)
    train_loader = init_subloader(train_set)
    val_loader = init_subloader(val_set)
    test_loader = init_subloader(test_set)
    return DataLoaders(train_loader, val_loader, test_loader)


def init_batch_functions(data_loader: core.HostDataLoader,
                         mb_size: int,
                         cache_size: int = 1,
                         *,
                         prefetch: bool = False,
                         ) -> core.RandomBatch:
    """Initializes reference data access outside jit-compiled functions.

    Randomly draw batches from a given dataset on the host or the device.
    If ``rng_seed=<seed>`` is passed to the ``init_fn``, a ``jax.random.PRNGKey``,
    will be added to the batch.

    Args:
        data_loader: Reads data from storage.
        cache_size: Number of batches in the cache. A larger number is
            faster, but requires more memory.
        mb_size: Size of the data batch.

    Returns:
      Returns a tuple of functions to initialize a new reference data state, get
      a minibatch from the reference data state and release the data loader after
      the last computation.
    """

    hcb_format, mb_information = data_loader.batch_format(
        cache_size, mb_size=mb_size)
    mask_shape = (cache_size, mb_size)

    prefetch_futures: Dict[int, Future] = {}
    executor = None
    if prefetch:
        executor = ThreadPoolExecutor(max_workers=1)

    def _chain_id_as_int(chain_id) -> int:
        if isinstance(chain_id, (int, onp.integer)):
            return int(chain_id)
        return int(jax.device_get(chain_id))

    def _prefetch_once(chain_id: int):
        return data_loader.get_batches(chain_id)

    def _submit_prefetch(chain_id: int) -> None:
        if not prefetch:
            return
        if executor is None:
            raise RuntimeError("Prefetch requested but executor is not initialized.")

        prefetch_futures[chain_id] = executor.submit(_prefetch_once, chain_id)

    def init_fn(random: bool = True, rng_seed=None, **kwargs) -> core.CacheState:

        if random:
            chain_id = data_loader.register_random_pipeline(
                cache_size=cache_size, mb_size=mb_size, **kwargs
            )
        else:
            chain_id = data_loader.register_ordered_pipeline(
                cache_size=cache_size, mb_size=mb_size, **kwargs
            )

        initial_state, initial_mask = data_loader.get_batches(chain_id)
        if initial_mask is None:
            initial_mask = jnp.ones((cache_size, mb_size), dtype=jnp.bool_)

        _submit_prefetch(chain_id)

        initial_internal_state = {}
        if rng_seed is not None:
            initial_internal_state['rng'] = jax.random.PRNGKey(rng_seed)

        inital_cache_state = core.CacheState(
            cached_batches=initial_state,
            cached_batches_count=jnp.array(cache_size),
            current_line=jnp.array(0),
            chain_id=jnp.array(chain_id),
            valid=initial_mask,
            state=initial_internal_state,
        )

        return inital_cache_state

    def _new_cache_fn(state: core.CacheState,
                      ) -> core.CacheState:
        chain_id = _chain_id_as_int(state.chain_id)

        if prefetch:
            future = prefetch_futures.pop(chain_id, None)
            if future is None:
                new_data, masks = data_loader.get_batches(chain_id)
            else:
                new_data, masks = future.result()

            _submit_prefetch(chain_id)
        else:
            new_data, masks = data_loader.get_batches(chain_id)

        if masks is None:
            # Assume all samples to be valid.
            masks = jnp.ones(mask_shape, dtype=jnp.bool_)

        new_state = core.CacheState(
            cached_batches_count=state.cached_batches_count,
            cached_batches=new_data,
            current_line=jnp.array(0),
            chain_id=state.chain_id,
            valid=masks,
            callback_uuid=state.callback_uuid,
            state=state.state
        )

        return new_state
        
    @jax.jit
    def _split_batch(data_state: core.CacheState):
        current_line = jnp.mod(
            data_state.current_line, data_state.cached_batches_count)

        # Read the current line from the cache and add the mask containing
        # information about the validity of the individual samples
        mini_batch = util.tree_get_single(data_state.cached_batches, current_line)
        mask = data_state.valid[current_line, :]

        # Add a random key if required
        internal_state = data_state.state
        if 'rng' in internal_state.keys():
            key, split = random.split(internal_state['rng'])
            mini_batch['rng'] = random.split(split, mb_information.batch_size)
            internal_state['rng'] = key

        current_line = current_line + 1

        new_state = core.CacheState(
            cached_batches=data_state.cached_batches,
            cached_batches_count=data_state.cached_batches_count,
            current_line=current_line,
            chain_id=data_state.chain_id,
            valid=data_state.valid,
            state=internal_state
        )
        
        info = core.MiniBatchInformation(
            observation_count = mb_information.observation_count,
            batch_size = mb_information.batch_size,
            mask = mask)
            
        return new_state, mini_batch, info

    def batch_fn(data_state: core.CacheState,
                 information: bool = False,
                 device_count: int = 1,
                 ) -> core.Batch:
        """Draws a new random batch.

        Args:
            data_state: State with cached samples
            information: Whether to return batch information
            device_count: Number of parallel programs calling the batch function

        Returns:
            Returns the new data state and the next batch. Optionally an additional
            struct containing information about the batch can be returned.

        """
        # Refresh the cache if necessary, after all cached batches have been used.
        if data_state.current_line == data_state.cached_batches_count:
            data_state = _new_cache_fn(data_state)

        new_state, mini_batch, info = _split_batch(data_state)

        if information:
            return new_state, (mini_batch, info)
        else:
            return new_state, mini_batch

    def release():
        for future in prefetch_futures.values():
            future.cancel()
        prefetch_futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    return init_fn, batch_fn, release
