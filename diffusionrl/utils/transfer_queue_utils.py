import asyncio
import inspect
import logging
import math
import os
import threading
import time
from functools import wraps
from typing import Any, Callable

import ray
import torch
from ray.actor import ActorHandle
from tensordict import NonTensorData, TensorDict
from transfer_queue import (
    AsyncTransferQueueClient,
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
    get_placement_group,
    process_zmq_server_info,
)
from transfer_queue.storage.clients.mooncake_client import RegisterBufferType

from diffusionrl.cmdline.transfer_queue import build_transfer_queue_config_from_args
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.types.transfer_queue_batch_meta import TQBatchMetaBatched

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", "WARN"))

_TRANSFER_QUEUE_GLOBAL_CONFIG = None
_TRANSFER_QUEUE_BACKEND_CONFIG = None
_TRANSFER_QUEUE_CLIENT = None
_DATA_SYSTEM_STORAGE_UNITS = None
_DATA_SYSTEM_CONTROLLER = None
_DEFAULT_PARTITION_ID = "train_partition"
_TRANSFER_QUEUE_UTILS_DEBUG = int(os.getenv("TRANSFER_QUEUE_UTILS_DEBUG", 0))

# =========================================================
# attr utils
# =========================================================


def _split_path(path: str):
    return path.split(".")


def _is_int(s: str) -> bool:
    return s.isdigit()


def has_path(obj: Any, path: str) -> bool:
    try:
        _traverse(obj, path)
        return True
    except (AttributeError, KeyError, IndexError, TypeError):
        return False


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    try:
        return _traverse(obj, path)
    except (AttributeError, KeyError, IndexError, TypeError):
        return default


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = _split_path(path)
    if not parts:
        return

    cur = obj
    for p in parts[:-1]:
        if isinstance(cur, dict):
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        elif isinstance(cur, list) and _is_int(p):
            idx = int(p)
            cur = cur[idx]
        else:
            if not hasattr(cur, p):
                setattr(cur, p, {})
            cur = getattr(cur, p)

    last = parts[-1]

    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list) and _is_int(last):
        cur[int(last)] = value
    else:
        setattr(cur, last, value)


def _traverse(obj: Any, path: str) -> Any:
    cur = obj
    for p in _split_path(path):
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list) and _is_int(p):
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur


def _get_local_ip():
    import socket

    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    return ip


# =========================================================
# transferqueue method
# =========================================================


def is_transferqueue_enabled():
    return get_transferqueue_client() is not None


def get_transferqueue_controller():
    from transfer_queue.interface import _TRANSFER_QUEUE_CONTROLLER

    return _TRANSFER_QUEUE_CONTROLLER


def update_single_controller_tq_config(tq_global_config, tq_backend_config, args):
    if not tq_global_config.get("enable", False):
        return tq_global_config, tq_backend_config
    if tq_global_config.get("storage_backend") == "MooncakeStorageManager":
        assert "zero_copy" in tq_backend_config
        tq_backend_config["zero_copy"].update(
            {
                "tensor_buffer_size_gb": args.transfer_queue.mooncake_backend_config.zero_copy_single_controller_tensor_buffer_size_gb,
                "bytes_buffer_size_gb": args.transfer_queue.mooncake_backend_config.zero_copy_single_controller_bytes_buffer_size_gb,
            }
        )
    return tq_global_config, tq_backend_config


def create_transferqueue_client(
    client_id: str,
    tq_global_config: dict,
    tq_backend_config: dict,
    sync: bool = False,
) -> "AsyncTransferQueueClient | TransferQueueClient":
    global _TRANSFER_QUEUE_CLIENT
    global _TRANSFER_QUEUE_GLOBAL_CONFIG
    global _TRANSFER_QUEUE_BACKEND_CONFIG
    if not tq_global_config.get("enable", False):
        return None
    if tq_global_config.get("storage_backend") == "MooncakeStorageManager":
        # NOTE Mooncake must specify MC_TCP_SINDOADRESS, otherwise it will identify randomly available IP addresses
        os.environ["MC_TCP_BIND_ADDRESS"] = os.getenv("LOCAL_IP", _get_local_ip())
        tq_backend_config["local_hostname"] = os.getenv("LOCAL_IP", _get_local_ip())
    logger.info(f"create_transferqueue_client, backend config: {tq_backend_config}")
    if _TRANSFER_QUEUE_CLIENT is None:
        if sync:
            _TRANSFER_QUEUE_CLIENT = TransferQueueClient(client_id, tq_backend_config.get("controller_info"))
        else:
            _TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, tq_backend_config.get("controller_info"))
        _TRANSFER_QUEUE_CLIENT.initialize_storage_manager(
            manager_type=tq_global_config.get("storage_backend"), config=tq_backend_config
        )
        _TRANSFER_QUEUE_GLOBAL_CONFIG = tq_global_config
        _TRANSFER_QUEUE_BACKEND_CONFIG = tq_backend_config
    else:
        logger.warning("transferqueue_client is already exists!")

    return _TRANSFER_QUEUE_CLIENT


def get_transferqueue_client() -> "AsyncTransferQueueClient | TransferQueueClient":
    global _TRANSFER_QUEUE_CLIENT
    return _TRANSFER_QUEUE_CLIENT


def tq_profiler(name=""):
    def decorator(func):
        global _TRANSFER_QUEUE_UTILS_DEBUG
        if not _TRANSFER_QUEUE_UTILS_DEBUG:
            return func

        @wraps(func)
        def wrapper_inner(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"[{name}] {func.__name__} Runtime: {end_time - start_time:.6f} s, timestamp: f{end_time}")
            return result

        @wraps(func)
        async def wrapper_async_inner(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"[{name}] {func.__name__} Runtime: {end_time - start_time:.6f} s, timestamp: f{end_time}")
            return result

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    # Use a temporary event loop in a new thread because event
    # loop may already exist in server mode
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta dataproto converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def reset_zero_copy_buffer_free():
    tq_client = get_transferqueue_client()
    if tq_client is None:
        return
    _run_async_in_temp_loop(tq_client.async_reset_zero_copy_all_keys_free, RegisterBufferType.GET_BYTES)
    _run_async_in_temp_loop(tq_client.async_reset_zero_copy_all_keys_free, RegisterBufferType.GET_TENSOR)
    _run_async_in_temp_loop(tq_client.async_reset_zero_copy_all_keys_free, RegisterBufferType.PUT_TENSOR)
    _run_async_in_temp_loop(tq_client.async_reset_zero_copy_all_keys_free, RegisterBufferType.PUT_BYTES)


def reset_actors_zero_copy_buffer_free(actors: list[ActorHandle]):
    refs = [actor.reset_zero_copy_buffer_free.remote() for actor in actors]
    ray.get(refs)


def clear_partition(partition_id: str = _DEFAULT_PARTITION_ID):
    tq_client = get_transferqueue_client()
    if tq_client is None:
        return
    tq_client.clear_partition(partition_id)


def stack_data(data_ret):
    # mooncake zero_copy has manager_merge_to_tensordict config, if set this False, return list[Tensor]
    if isinstance(data_ret, list):
        if all(isinstance(tensor, torch.Tensor) for tensor in data_ret):
            return torch.stack(data_ret, dim=0)
        else:
            # if put list[object] to TQ, not need to stack
            return data_ret
    elif isinstance(data_ret, torch.Tensor):
        return data_ret
    else:
        raise ValueError(f"stack_data error, check data_ret cls: {type(data_ret)}")


def init_remote_actor_transferqueue_client(actors: list[ActorHandle], tq_global_config, tq_backend_config):
    refs = [
        actor.init_transferqueue_client.remote(tq_global_config=tq_global_config, tq_backend_config=tq_backend_config)
        for actor in actors
    ]
    ray.get(refs)


def init_transferqueue(args):
    tq_global_config, tq_backend_config = build_transfer_queue_config_from_args(args)
    if not tq_global_config.get("enable", False):
        return tq_global_config, None
    # 1. initialize TransferQueueStorage
    if tq_global_config.get("storage_backend") == "AsyncSimpleStorageManager":
        train_data_size = (
            tq_backend_config["train_micro_batch_size"]
            * tq_backend_config["num_global_batch"]
            * tq_backend_config["rollout_n"]
            * tq_backend_config["item_size_one_batch"]
        )

        total_storage_size = train_data_size
        global _DATA_SYSTEM_STORAGE_UNITS
        _DATA_SYSTEM_STORAGE_UNITS = {}
        num_data_storage_units = (
            tq_backend_config["num_data_storage_units_per_node"] * tq_backend_config["storage_units_nnodes"]
        )
        storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)
        for storage_unit_rank in range(num_data_storage_units):
            storage_node = SimpleStorageUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
            ).remote(storage_unit_size=math.ceil(total_storage_size / num_data_storage_units))
            _DATA_SYSTEM_STORAGE_UNITS[storage_unit_rank] = storage_node
        data_system_storage_unit_infos = process_zmq_server_info(_DATA_SYSTEM_STORAGE_UNITS)
    elif tq_global_config.get("storage_backend") == "MooncakeStorageManager":
        pass
    else:
        raise NotImplementedError(
            f"Currently not support {tq_global_config.get('storage_backend')} backend in TransferQueue"
        )

    # 2. Initialize TransferQueueController (single controller only)
    global _DATA_SYSTEM_CONTROLLER
    _DATA_SYSTEM_CONTROLLER = TransferQueueController.remote()

    # 3. register controller & storage and prepare necessary information
    data_system_controller_info = process_zmq_server_info(_DATA_SYSTEM_CONTROLLER)
    tq_backend_config["controller_info"] = data_system_controller_info

    # Adapt the backend configuration format for TransferQueue, and populate required dependent parameters.
    if tq_global_config.get("storage_backend") == "AsyncSimpleStorageManager":
        tq_backend_config["storage_unit_infos"] = data_system_storage_unit_infos
    elif tq_global_config.get("storage_backend") == "MooncakeStorageManager":
        # NOTE: as a pure client set global_segment_size=0
        tq_backend_config["global_segment_size"] = int(tq_backend_config.get("global_segment_size_gb") * 1024**3)

        tq_backend_config["local_buffer_size"] = int(tq_backend_config.get("local_buffer_size_gb") * 1024**3)
    else:
        raise NotImplementedError(
            f"Currently not support {tq_global_config.get('storage_backend')} backend in TransferQueue"
        )

    return tq_global_config, tq_backend_config


# =========================================================
# function adapter for interacting with tq
# =========================================================


def _batchmeta_to_data(data):
    return _run_async_in_temp_loop(_batchmeta_to_data_async, data)


async def _batchmeta_to_data_async(data):
    get_fn = dispatch_get_fn(data)
    if get_fn:
        return await get_fn(data)
    return data


@tq_profiler(name="TQ utils")
def _data_to_transferqueue(data):
    return _run_async_in_temp_loop(_data_to_transferqueue_async, data)


async def _data_to_transferqueue_async(data):
    if data is None:
        return data

    if isinstance(data, (int, float, str, bytes)):
        return data

    if isinstance(data, (list, tuple)):
        return type(data)([await _data_to_transferqueue_async(obj) for obj in data])

    if isinstance(data, dict):
        return {k: await _data_to_transferqueue_async(v) for k, v in data.items()}

    put_fn = dispatch_put_fn(data)
    if put_fn:
        global _DEFAULT_PARTITION_ID
        return await put_fn(data, _DEFAULT_PARTITION_ID)
    return data


def tqbridge(get=False, put=False):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if get and is_transferqueue_enabled():
                args = [_batchmeta_to_data(arg) for arg in args]
                kwargs = {k: _batchmeta_to_data(v) for k, v in kwargs.items()}
            output = func(*args, **kwargs)
            if put and is_transferqueue_enabled():
                output = _data_to_transferqueue(output)
            return output

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if get and is_transferqueue_enabled():
                args = [await _batchmeta_to_data_async(arg) for arg in args]
                kwargs = {k: await _batchmeta_to_data_async(v) for k, v in kwargs.items()}
            output = await func(*args, **kwargs)
            if put and is_transferqueue_enabled():
                output = await _data_to_transferqueue_async(output)
            return output

        wrapper_inner = inner
        wrapper_async_inner = async_inner

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator


@tqbridge(get=True, put=False)
def resolve_batch_from_tq(data):
    if isinstance(data, TQBatchMetaBatched):
        return data.data
    return data


# =========================================================
# define cls transfer queue get/put register
# =========================================================

TRANSFER_QUEUE_GET_CLS_REGISTER = {}
TRANSFER_QUEUE_PUT_CLS_REGISTER = {}


def register_put_get_cls(cls, get=None, put=None):
    global TRANSFER_QUEUE_GET_CLS_REGISTER
    global TRANSFER_QUEUE_PUT_CLS_REGISTER
    if get:
        TRANSFER_QUEUE_GET_CLS_REGISTER[cls] = get
    if put:
        TRANSFER_QUEUE_PUT_CLS_REGISTER[cls] = put


def dispatch_get_fn(obj):
    global TRANSFER_QUEUE_GET_CLS_REGISTER
    get_fn = None
    for cls, fn in TRANSFER_QUEUE_GET_CLS_REGISTER.items():
        if isinstance(obj, cls):
            get_fn = fn
    return get_fn


def dispatch_put_fn(obj):
    global TRANSFER_QUEUE_PUT_CLS_REGISTER
    put_fn = None
    for cls, fn in TRANSFER_QUEUE_PUT_CLS_REGISTER.items():
        if isinstance(obj, cls):
            put_fn = fn
    return put_fn


# =========================================================
# define cls transfer queue get/put method
# =========================================================


@tq_profiler(name="TQ utils")
async def _tq_put_batch_meta_batched(data: TQBatchMetaBatched, partition_id: str):
    assert isinstance(data, TQBatchMetaBatched)
    assert isinstance(data.data, torch.Tensor)
    tq_client = get_transferqueue_client()
    shape = data.data.shape if isinstance(data.data, torch.Tensor) else [len(data.data)]
    data_key = data._data_key
    metas = await tq_client.async_put(
        data=TensorDict({data_key: data.data}, batch_size=shape[0]), partition_id=partition_id
    )
    data.batch_meta = metas
    data.reset_data()
    return data


@tq_profiler(name="TQ utils")
async def _tq_get_batch_meta_batched(data: TQBatchMetaBatched):
    assert isinstance(data, TQBatchMetaBatched)
    data_key = data._data_key
    tq_client = get_transferqueue_client()
    tensor_data = await tq_client.async_get_data(data.batch_meta)
    data.data = stack_data(tensor_data[data_key])
    data.reset_batch_meta()
    return data


@tq_profiler(name="TQ utils")
async def _tq_put_rollout_response(data: RolloutResponse, partition_id):
    assert isinstance(data, RolloutResponse)
    tq_client = get_transferqueue_client()

    def add_data_dict(x, key, data_dict: dict):
        if isinstance(x, torch.Tensor):
            data_dict[key] = x
        elif isinstance(x, list):
            data_dict[key] = NonTensorData(x)
        return data_dict

    def create_data_batched(batch_meta: BatchMeta, orin_data: torch.Tensor | list, field_name: str):
        assert isinstance(orin_data, torch.Tensor) or isinstance(orin_data, list), (
            f"cant create type: {type(orin_data)}"
        )
        shape = orin_data.shape if isinstance(orin_data, torch.Tensor) else [len(orin_data)]
        data_con = TQBatchMetaBatched(
            batch_meta=batch_meta.select_fields([field_name]),
            data=None,
            _shape=shape,
            _data_key=field_name,
        )
        return data_con

    data_dict = {}
    samples = data.samples
    attrs_fields = {
        "samples.latents": "latents",
        "samples.trajectories.data": "trajectories",
        "samples.decoded_images": "decoded_images",
        "samples.decoded_videos": "decoded_videos",
        "samples.forward_context.prompt_embeds": "fc_prompt_embeds",
        "samples.forward_context.negative_pooled_prompt_embeds": "fc_negative_pooled_prompt_embeds",
        "samples.forward_context.negative_prompt_embeds": "fc_negative_prompt_embeds",
        "samples.forward_context.pooled_prompt_embeds": "fc_pooled_prompt_embeds",
    }
    for attr_key, field_name in attrs_fields.items():
        if has_path(data, attr_key) and get_path(data, attr_key) is not None:
            data_dict = add_data_dict(get_path(data, attr_key), field_name, data_dict)

    batch_size = samples.latents.shape[0]
    tq_put_tensor = TensorDict(data_dict, batch_size=batch_size).cpu()
    batch_meta = await tq_client.async_put(
        data=tq_put_tensor,
        partition_id=partition_id,
    )

    for attr_key, field_name in attrs_fields.items():
        if has_path(data, attr_key) and get_path(data, attr_key) is not None:
            set_path(data, attr_key, create_data_batched(batch_meta, get_path(data, attr_key), field_name))
    return data


@tq_profiler(name="TQ utils")
async def _tq_get_training_batch(data: TrainingBatch):
    assert isinstance(data, TrainingBatch)
    tq_client = get_transferqueue_client()

    batch_metas = []

    def add_batch_metas(x, batch_metas: list):
        if isinstance(x, TQBatchMetaBatched):
            batch_metas.append(x.batch_meta)
        return batch_metas

    attrs_fields = {
        "trajectory_store.data": "trajectories",
        "forward_context.prompt_embeds": "fc_prompt_embeds",
        "forward_context.negative_pooled_prompt_embeds": "fc_negative_pooled_prompt_embeds",
        "forward_context.negative_prompt_embeds": "fc_negative_prompt_embeds",
        "forward_context.pooled_prompt_embeds": "fc_pooled_prompt_embeds",
    }

    for attr_key, field_name in attrs_fields.items():
        if has_path(data, attr_key) and get_path(data, attr_key) is not None:
            batch_metas = add_batch_metas(get_path(data, attr_key), batch_metas)

    # union all batch meta, get all data in once async_get_data
    batch_union: BatchMeta = batch_metas[0] if len(batch_metas) > 0 else BatchMeta(samples=[])
    for item in batch_metas[1:]:
        batch_union = batch_union.union(item)

    tensor_dict_data = await tq_client.async_get_data(batch_union)

    for attr_key, field_name in attrs_fields.items():
        if has_path(data, attr_key) and field_name in tensor_dict_data:
            set_path(data, attr_key, stack_data(tensor_dict_data[field_name]))
    return data


register_put_get_cls(RolloutResponse, get=None, put=_tq_put_rollout_response)
register_put_get_cls(TrainingBatch, get=_tq_get_training_batch, put=None)
register_put_get_cls(TQBatchMetaBatched, get=_tq_get_batch_meta_batched, put=_tq_put_batch_meta_batched)
