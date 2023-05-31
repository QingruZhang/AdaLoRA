# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
Utilities for the Trainer and TFTrainer class. Should be independent from PyTorch and TensorFlow.
"""

import copy
import gc
import inspect
import os
import random
import re
import time
import tracemalloc
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from .file_utils import (
    ExplicitEnum,
    is_sagemaker_distributed_available,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_tpu_available,
)


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # ^^ safe to call this function even if cuda is not available
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
    if is_tf_available():
        tf.random.set_seed(seed)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see :class:`~transformers.Trainer.hyperparameter_search`).

    Parameters:
        run_id (:obj:`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (:obj:`float`):
            The objective that was obtained for this run.
        hyperparameters (:obj:`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_samples_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())


def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available

    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }


def default_hp_space_ray(trial) -> Dict[str, float]:
    from .integrations import is_ray_tune_available

    assert is_ray_tune_available(), "This function needs ray installed: `pip " "install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"


default_hp_space = {
    HPSearchBackend.OPTUNA: default_hp_space_optuna,
    HPSearchBackend.RAY: default_hp_space_ray,
}


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    elif is_sagemaker_distributed_available():
        import smdistributed.dataparallel.torch.distributed as dist

        return dist.get_world_size()
    elif local_rank != -1 and is_torch_available():
        import torch

        return torch.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = 1 / (runtime / num_samples)
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    return result


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example ::

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        code ...
        metrics = {"train_runtime": 10.5}
        self._memory_tracker.stop_and_update_metrics(metrics)

    At the moment gpu tracking is only for pytorch, but can be extended to support tensorflow.

    Understanding the reports:

    - ``*_alloc_delta`` - is the difference in the used/allocated memory counter between the end and the start of the
      stage - it can be negative if a function released more memory than it allocated.

    - ``*_peaked_delta`` - is any extra memory that was consumed and then freed - relative to the current allocated
      memory counter - it is never negative.

    So when you look at the metrics of any stage you add up ``alloc_delta`` + ``peaked_delta`` and you know how much
    memory was needed to complete that stage.

    The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
    main process does the bulk of work, but it could be not quite so if model parallel is used and then other gpus may
    use a different amount of gpu RAM. Perhaps in the future this tracker will evolve to measure those too.

    Note that this tracker doesn't account for memory allocations outside of :class:`~transformers.Trainer`'s
    ``__init__``, ``train``, ``evaluate`` and ``predict`` calls.

    Because ``evaluation`` calls may happen during ``train``, we can't handle nested invocations because
    ``torch.cuda.max_memory_allocated`` is a single counter, so if it gets reset by a nested eval call, ``train``'s
    tracker will report incorrect info. If this `pytorch issue <https://github.com/pytorch/pytorch/issues/16266>`__
    gets resolved it will be possible to change this class to be re-entrant. Until then we will only track the outer
    level of ``train``, ``evaluate`` and ``predict`` methods. Which means that if ``eval`` is called during ``train``,
    it's the latter that will account for its memory usage and that of the former.

    This also means that if any other tool that is used along the :class:`~transformers.Trainer` calls
    ``torch.cuda.reset_peak_memory_stats``, the gpu peak memory stats could be invalid. And the
    :class:`~transformers.Trainer` will disrupt the normal behavior of any such tools that rely on calling
    ``torch.cuda.reset_peak_memory_stats`` themselves.

    """

    # map trainer methods to metrics prefix
    stages = {
        "__init__": "init",
        "train": "train",
        "evaluate": "eval",
        "predict": "test",
    }

    def __init__(self, skip_memory_metrics=False):
        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None

        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False
        self.skip_memory_metrics = skip_memory_metrics

    def derive_stage(self):
        """ derives the stage/caller name automatically """
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )

    def start(self):
        """ start tracking for the caller's stage """
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        self.cur_stage = stage

        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()

        gc.collect()

        # gpu
        if self.torch is not None:
            self.gpu[self.cur_stage] = {}
            self.gpu[self.cur_stage]["alloc"] = self.torch.cuda.memory_allocated()
            self.gpu[self.cur_stage]["peaked"] = 0

        # cpu
        self.cpu[self.cur_stage] = {}
        tracemalloc.start()

    def stop(self, stage):
        """ stop tracking for the passed stage """

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        if self.torch is not None:
            self.torch.cuda.empty_cache()

        gc.collect()

        # gpu
        if self.torch is not None:
            mem_cur = self.torch.cuda.memory_allocated()
            # this is the difference between the start and the end allocated memory
            self.gpu[self.cur_stage]["alloc"] = mem_cur - self.gpu[self.cur_stage]["alloc"]  # can be negative
            # this is the difference if any between the start and the peak
            self.gpu[self.cur_stage]["peaked"] = max(0, self.torch.cuda.max_memory_allocated() - mem_cur)

        # cpu
        cpu_mem_used_delta, cpu_mem_used_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # reset accounting
        self.cpu[self.cur_stage]["alloc"] = cpu_mem_used_delta  # can be negative
        self.cpu[self.cur_stage]["peaked"] = max(0, cpu_mem_used_peak - cpu_mem_used_delta)

        # reset - cycle finished
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        """ stop tracking for the passed stage """
        if self.skip_memory_metrics:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]

    def stop_and_update_metrics(self, metrics=None):
        """ combine stop + update in one call for simpler code """
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)


def denumpify_detensorize(metrics):
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    if isinstance(metrics, (list, tuple)):
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        return metrics.item()
    elif is_torch_available() and isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        return metrics.item()
    return metrics


class ShardedDDPOption(ExplicitEnum):
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"
