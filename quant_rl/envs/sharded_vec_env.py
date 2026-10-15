"""Multiprocess VecEnv that keeps several bundle environments in each process.

Unlike :class:`SubprocVecEnv`, this backend uses one pipe and process per shard,
not one per logical environment. Environments inside a shard are still stepped
independently and in the same stable order, so the policy's logical rollout and
episode semantics are unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _shard_worker(
    remote: Any,
    parent_remote: Any,
    env_fns_wrapper: CloudpickleWrapper,
) -> None:
    """Run a contiguous shard of environments behind one pipe."""
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = [_patch_env(env_fn()) for env_fn in env_fns_wrapper.var]
    reset_infos: list[dict[str, Any]] = [{} for _ in envs]
    results: list[Any] = []

    try:
        while True:
            try:
                command, data = remote.recv()
            except EOFError:
                break

            try:
                if command == "get_spaces":
                    remote.send((envs[0].observation_space, envs[0].action_space))
                elif command == "step":
                    results = []
                    for env_idx, (env, action) in enumerate(zip(envs, data, strict=True)):
                        observation, reward, terminated, truncated, info = env.step(action)
                        done = bool(terminated or truncated)
                        info = dict(info)
                        info["TimeLimit.truncated"] = bool(truncated and not terminated)
                        if done:
                            info["terminal_observation"] = observation
                            observation, reset_infos[env_idx] = env.reset()
                        results.append((observation, reward, done, info, reset_infos[env_idx]))
                    remote.send(results)
                elif command == "reset":
                    results = []
                    for env_idx, (seed, options) in enumerate(data):
                        env = envs[env_idx]
                        maybe_options = {"options": options} if options else {}
                        observation, reset_infos[env_idx] = env.reset(seed=seed, **maybe_options)
                        results.append((observation, reset_infos[env_idx]))
                    remote.send(results)
                elif command == "render":
                    remote.send([env.render() for env in envs])
                elif command == "get_attr":
                    indices, attr_name = data
                    remote.send([envs[idx].get_wrapper_attr(attr_name) for idx in indices])
                elif command == "has_attr":
                    indices, attr_name = data
                    values = []
                    for idx in indices:
                        try:
                            envs[idx].get_wrapper_attr(attr_name)
                            values.append(True)
                        except AttributeError:
                            values.append(False)
                    remote.send(values)
                elif command == "set_attr":
                    indices, attr_name, value = data
                    for idx in indices:
                        setattr(envs[idx], attr_name, value)
                    remote.send(None)
                elif command == "env_method":
                    indices, method_name, method_args, method_kwargs = data
                    remote.send(
                        [
                            envs[idx].get_wrapper_attr(method_name)(*method_args, **method_kwargs)
                            for idx in indices
                        ]
                    )
                elif command == "is_wrapped":
                    indices, wrapper_class = data
                    remote.send([is_wrapped(envs[idx], wrapper_class) for idx in indices])
                elif command == "close":
                    for env in envs:
                        env.close()
                    remote.send(None)
                    break
                else:
                    raise NotImplementedError(f"unsupported VecEnv worker command: {command}")
            except BaseException:
                remote.send({"worker_error": traceback.format_exc()})
    finally:
        remote.close()


def _stack_observations(
    observations: Sequence[VecEnvObs], observation_space: spaces.Space[Any]
) -> VecEnvObs:
    if isinstance(observation_space, spaces.Dict):
        dict_observations = [
            cast(dict[str, np.ndarray[Any, Any]], observation) for observation in observations
        ]
        if not dict_observations:
            raise TypeError("Dict observation space received a non-dict observation")
        return {
            key: np.stack([observation[key] for observation in dict_observations])
            for key in observation_space.spaces
        }
    if isinstance(observation_space, spaces.Tuple):
        tuple_observations = [
            cast(tuple[np.ndarray[Any, Any], ...], observation) for observation in observations
        ]
        if not tuple_observations:
            raise TypeError("Tuple observation space received a non-tuple observation")
        return tuple(
            np.stack([observation[idx] for observation in tuple_observations])
            for idx in range(len(observation_space.spaces))
        )
    return np.stack(cast(Sequence[np.ndarray[Any, Any]], observations))


class ShardedBundleVecEnv(VecEnv):
    """Subprocess VecEnv with multiple bundle environments per worker process."""

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env[Any, Any]]],
        *,
        n_shards: int,
        start_method: str | None = None,
    ) -> None:
        if not env_fns:
            raise ValueError("ShardedBundleVecEnv requires at least one environment")
        if n_shards < 1 or n_shards > len(env_fns):
            raise ValueError("n_shards must be between 1 and the number of environments")

        self.waiting = False
        self.closed = False
        self._shard_ranges: list[tuple[int, int]] = []
        base, remainder = divmod(len(env_fns), n_shards)
        start = 0
        for shard_idx in range(n_shards):
            width = base + (1 if shard_idx < remainder else 0)
            self._shard_ranges.append((start, start + width))
            start += width

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {start_method}")
        context: Any = mp.get_context(start_method)
        pipes = [context.Pipe() for _ in self._shard_ranges]
        self.remotes = [parent for parent, _ in pipes]
        self.processes = []
        for shard_idx, ((parent_remote, work_remote), (begin, end)) in enumerate(
            zip(pipes, self._shard_ranges, strict=True)
        ):
            process = context.Process(
                target=_shard_worker,
                args=(work_remote, parent_remote, CloudpickleWrapper(env_fns[begin:end])),
                daemon=True,
                name=f"bundle-env-shard-{shard_idx}",
            )
            process.start()
            work_remote.close()
            self.processes.append(process)

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self._receive(self.remotes[0])
        super().__init__(len(env_fns), observation_space, action_space)

    @staticmethod
    def _receive(remote: Any) -> Any:
        result = remote.recv()
        if isinstance(result, dict) and "worker_error" in result:
            raise RuntimeError(f"sharded environment worker failed:\n{result['worker_error']}")
        return result

    @staticmethod
    def _receive_all(remotes: Sequence[Any]) -> list[Any]:
        results = [remote.recv() for remote in remotes]
        for result in results:
            if isinstance(result, dict) and "worker_error" in result:
                raise RuntimeError(f"sharded environment worker failed:\n{result['worker_error']}")
        return results

    @property
    def num_shards(self) -> int:
        return len(self.remotes)

    @property
    def logical_envs_per_shard(self) -> list[int]:
        return [end - start for start, end in self._shard_ranges]

    def _group_indices(self, indices: VecEnvIndices) -> list[list[tuple[int, int]]]:
        groups: list[list[tuple[int, int]]] = [[] for _ in self.remotes]
        for env_idx in self._get_indices(indices):
            if env_idx < 0 or env_idx >= self.num_envs:
                raise IndexError(f"environment index {env_idx} is outside [0, {self.num_envs})")
            for shard_idx, (start, end) in enumerate(self._shard_ranges):
                if start <= env_idx < end:
                    groups[shard_idx].append((env_idx, env_idx - start))
                    break
        return groups

    def _requested_indices(self, indices: VecEnvIndices) -> list[int]:
        result = list(self._get_indices(indices))
        for env_idx in result:
            if env_idx < 0 or env_idx >= self.num_envs:
                raise IndexError(f"environment index {env_idx} is outside [0, {self.num_envs})")
        return result

    def reset(self) -> VecEnvObs:
        for remote, (start, end) in zip(self.remotes, self._shard_ranges, strict=True):
            remote.send(
                ("reset", list(zip(self._seeds[start:end], self._options[start:end], strict=True)))
            )
        results = self._receive_all(self.remotes)
        observations: list[VecEnvObs] = []
        reset_infos: list[dict[str, Any]] = []
        for shard_results in results:
            for observation, reset_info in shard_results:
                observations.append(observation)
                reset_infos.append(reset_info)
        self.reset_infos = reset_infos
        self._reset_seeds()
        self._reset_options()
        return _stack_observations(observations, self.observation_space)

    def step_async(self, actions: np.ndarray[Any, Any]) -> None:
        if self.waiting:
            raise RuntimeError("step_async called while a previous step is pending")
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, received {len(actions)}")
        for remote, (start, end) in zip(self.remotes, self._shard_ranges, strict=True):
            remote.send(("step", actions[start:end]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        if not self.waiting:
            raise RuntimeError("step_wait called without a pending step")
        try:
            results = self._receive_all(self.remotes)
        finally:
            self.waiting = False
        observations: list[VecEnvObs] = []
        rewards: list[Any] = []
        dones: list[bool] = []
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any]] = []
        for shard_results in results:
            for observation, reward, done, info, reset_info in shard_results:
                observations.append(observation)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                reset_infos.append(reset_info)
        self.reset_infos = reset_infos
        return (
            _stack_observations(observations, self.observation_space),
            np.asarray(rewards),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    self._receive(remote)
                except RuntimeError:
                    pass
            self.waiting = False
        for remote in self.remotes:
            if remote.closed:
                continue
            remote.send(("close", None))
        for remote in self.remotes:
            if not remote.closed:
                try:
                    self._receive(remote)
                except (EOFError, BrokenPipeError):
                    pass
                remote.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray[Any, Any] | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"render mode is {self.render_mode!r}; expected 'rgb_array' for get_images()"
            )
            return [None for _ in range(self.num_envs)]
        for remote in self.remotes:
            remote.send(("render", None))
        return [image for remote in self.remotes for image in self._receive(remote)]

    def has_attr(self, attr_name: str) -> bool:
        groups = self._group_indices(None)
        for remote, group in zip(self.remotes, groups, strict=True):
            remote.send(("has_attr", (tuple(local for _, local in group), attr_name)))
        results = [value for remote in self.remotes for value in self._receive(remote)]
        return all(results)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        groups = self._group_indices(indices)
        for remote, group in zip(self.remotes, groups, strict=True):
            remote.send(("get_attr", (tuple(local for _, local in group), attr_name)))
        responses = self._receive_all(self.remotes)
        values: dict[int, Any] = {}
        for group, response in zip(groups, responses, strict=True):
            values.update(
                {env_idx: value for (env_idx, _), value in zip(group, response, strict=True)}
            )
        return [values[env_idx] for env_idx in self._requested_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        groups = self._group_indices(indices)
        for remote, group in zip(self.remotes, groups, strict=True):
            remote.send(("set_attr", (tuple(local for _, local in group), attr_name, value)))
        for remote in self.remotes:
            self._receive(remote)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        groups = self._group_indices(indices)
        for remote, group in zip(self.remotes, groups, strict=True):
            remote.send(
                (
                    "env_method",
                    (
                        tuple(local for _, local in group),
                        method_name,
                        method_args,
                        method_kwargs,
                    ),
                )
            )
        responses = self._receive_all(self.remotes)
        values: dict[int, Any] = {}
        for group, response in zip(groups, responses, strict=True):
            values.update(
                {env_idx: value for (env_idx, _), value in zip(group, response, strict=True)}
            )
        return [values[env_idx] for env_idx in self._requested_indices(indices)]

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper[Any, Any, Any, Any]],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        groups = self._group_indices(indices)
        for remote, group in zip(self.remotes, groups, strict=True):
            remote.send(("is_wrapped", (tuple(local for _, local in group), wrapper_class)))
        responses = self._receive_all(self.remotes)
        values: dict[int, bool] = {}
        for group, response in zip(groups, responses, strict=True):
            values.update(
                {env_idx: value for (env_idx, _), value in zip(group, response, strict=True)}
            )
        return [values[env_idx] for env_idx in self._requested_indices(indices)]
