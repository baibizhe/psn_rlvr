# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import ast
import gc
import logging
import os
import time
from typing import Any, Generator, Optional
import zlib

import ray
import torch
import zmq
from packaging import version as vs
from torch.distributed.device_mesh import DeviceMesh
from torch.multiprocessing.reductions import reduce_tensor

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator
from verl.workers.rollout.vllm_rollout.utils import TensorMetadata, get_device_uuid

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        self.zmq_context = zmq.Context()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

        self.use_param_noise = bool(self.config.get("use_param_noise", False))
        self.noise_sigma = float(self.config.get("noise_sigma", 0.0) or 0.0)
        raw_noise_target_modules = self.config.get("noise_target_modules", None)
        self.noise_target_modules = self._normalize_noise_target_modules(raw_noise_target_modules)
        self._noise_plan_logged = False

        if self.replica_rank == 0 and self.rollout_rank == 0:
            if self.noise_target_modules:
                logger.info(
                    "[Noise Debug][Target Modules] raw=%r -> normalized=%s | match_rule=any(substr in param_name)",
                    raw_noise_target_modules,
                    self.noise_target_modules,
                )
            else:
                logger.info(
                    "[Noise Debug][Target Modules] raw=%r -> normalized=None => apply_to=ALL floating-point params",
                    raw_noise_target_modules,
                )

        
        self.param_noise_base_seed = self._normalize_noise_seed(
            self.config.get("param_noise_base_seed", None)
        )
        self.noise_step_counter = 0
        self.noise_update_counter = 0
        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(
                "[Noise Debug][Rollout Init] use_param_noise=%s sigma=%s target_modules=%s base_seed=%s",
                self.use_param_noise,
                self.noise_sigma,
                self.noise_target_modules,
                self.param_noise_base_seed,
            )

    def set_noise_sigma(self, new_sigma: float):
        prev_sigma = self.noise_sigma
        self.noise_sigma = float(new_sigma)
        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info("[Noise Debug][Rollout Sigma] %s -> %s", prev_sigma, self.noise_sigma)

    @staticmethod
    def _normalize_noise_seed(seed: Optional[int | str]) -> Optional[int]:
        if seed is None:
            return None
        if isinstance(seed, str):
            normalized = seed.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_noise_target_modules(value: Any) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value if str(v)]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v)]
            except (ValueError, SyntaxError):
                pass
            return [seg.strip() for seg in stripped.split(",") if seg.strip()]
        return None

    def _should_apply_noise(self, name: str, tensor: torch.Tensor) -> bool:
        if not self.use_param_noise or self.noise_sigma <= 0.0:
            return False
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return False
        if not self.noise_target_modules:
            return True
        return any(target in name for target in self.noise_target_modules)

    def _next_noise_seed(self) -> Optional[int]:
        if self.param_noise_base_seed is None:
            return None
        step_seed = self.param_noise_base_seed + self.noise_step_counter
        self.noise_step_counter += 1
        return step_seed

    def _apply_param_noise(self, name: str, weight: torch.Tensor, step_seed: Optional[int]) -> torch.Tensor:
        # if not self._should_apply_noise(name, weight):
        #     logger.info('should not  apply noise %s ',str(name))
        #     return weight
        # logger.info('should apply noise !!! %s',str(name))
        device = weight.device
        base_cpu = weight.detach().cpu()
        base_f32 = base_cpu.to(torch.float32)
        gen = None
        if step_seed is not None:
            per_param_seed = (step_seed ^ zlib.adler32(name.encode())) % (2**31 - 1)
            gen = torch.Generator(device="cpu").manual_seed(int(per_param_seed))
        noise = torch.randn(base_f32.shape, dtype=torch.float32, generator=gen) * float(self.noise_sigma)
        noisy = (base_f32 + noise).to(dtype=base_cpu.dtype)
        if device.type != "cpu":
            noisy = noisy.to(device)
        return noisy

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if self.rollout_rank != 0:
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            self.server_handle = ray.get_actor(f"vllm_server_{self.replica_rank}_{self.node_rank}")

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})

    @torch.no_grad()
    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers."""
        start_time = time.time()
        use_noise = self.use_param_noise and self.noise_sigma > 0.0

        step_seed = self._next_noise_seed() if use_noise else None
        log_noise_plan_once = (
        use_noise
            and (not self._noise_plan_logged)
            and self.replica_rank == 0
            and self.rollout_rank == 0
        )
        noise_plan_max = int(os.getenv("VERL_NOISE_PLAN_MAX_NAMES", "200"))  # -1 打印全部
        noise_plan_names = []
        noise_plan_total = 0

        self.noise_update_counter += 1
        noise_applied_param_count = 0
        if self.replica_rank == 0 and self.rollout_rank == 0 and self.use_param_noise:
            logger.info(
                "[Noise Debug][Rollout Update] idx=%s use_noise=%s sigma=%s step_seed=%s",
                self.noise_update_counter,
                use_noise,
                self.noise_sigma,
                step_seed,
            )
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        # build communication buffer
        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        bucket_size = int(bucket_size_mb) << 20
        s = self.zmq_context.socket(zmq.REQ)
        s.bind(self.zmq_handle)

        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:0")
            handle = reduce_tensor(buffer)
            s.send_pyobj(handle)
        else:
            import uuid
            from multiprocessing import shared_memory

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=bucket_size)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": bucket_size}
            s.send_pyobj(comm_metadata)

        s.recv()

        # send bucket weights
        offset = 0
        bucket_meta: dict[str, TensorMetadata] = {}
        dtype = PrecisionType.to_dtype(self.config.dtype)
        async for name, weight in ensure_async_iterator(weights):
            if use_noise:
                if self._should_apply_noise(name, weight):
                    noise_applied_param_count += 1
                weight = self._apply_param_noise(name, weight, step_seed)
                if log_noise_plan_once:
                    noise_plan_total += 1
                    if noise_plan_max < 0 or len(noise_plan_names) < noise_plan_max:
                        noise_plan_names.append(name)
            # model parameters are in fp32 full precision
            weight = weight.to(dtype, non_blocking=True)

            # fill the tensor bucket
            if offset + weight.nbytes > bucket_size:
                get_torch_device().synchronize()
                s.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                s.recv()
                bucket_meta = {}
                offset = 0

            # TODO: slice embedding layer weight into chunks
            assert offset + weight.nbytes <= bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                f"Please increase rollout.update_weights_bucket_megabytes({bucket_size_mb} MB)."
            )
            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        # send the last bucket
        get_torch_device().synchronize()
        s.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
        s.recv()

        # clean up
        s.close()
        del buffer
        if shm is not None:
            shm.close()
            shm.unlink()
            del shm
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
        if future is not None:
            await future

        # reset prefix cache after updating weights
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(
                "update_weights done, time cost: %.2fs | noise_enabled=%s | noisy_param_tensors=%s",
                time.time() - start_time,
                use_noise,
                noise_applied_param_count,
            )
        if log_noise_plan_once:
            truncated = (noise_plan_max >= 0 and noise_plan_total > noise_plan_max)
            logger.info(
                "[Noise Debug][Plan Once] target_modules=%s | total_noisy_params=%d | showing=%d%s\n%s",
                self.noise_target_modules,
                noise_plan_total,
                len(noise_plan_names),
                " (TRUNCATED; set VERL_NOISE_PLAN_MAX_NAMES=-1 to log all)" if truncated else "",
                "\n".join(noise_plan_names) if noise_plan_names else "(none)",
            )
            self._noise_plan_logged = True


    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ServerAdapter uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/volcengine/verl/issues/4682 for more details."
        )
