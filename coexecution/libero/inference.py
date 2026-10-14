"""Prepared CUDA inference, separated from observations and deployment machinery.

This module imports no model framework. Production callers establish CUDA,
eval/inference mode, input readiness and serial policy ownership before calling
the engine. The engine neither transfers nor synchronizes tensors. It returns
the original component output with that component's lifetime contract; callers
that retain outputs must snapshot them outside their measurement boundary.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedInputs:
    """One observation's resident input trees and separate output-adapter state.

    The container is immutable; contained tensors remain caller-owned. Do not
    mutate them while an asynchronous CUDA call is in flight. ``prepare_*``
    adapters below own mutable input copies so later collation cannot replace
    this frame. Registered immutable metadata belongs to the preparation scope;
    keep that scope open and its bindings resident while using these inputs.
    No visual feature or other neural output is retained here.
    """

    backbone: Any
    action: Any
    decode_state: Any
    observation_id: str

    def __post_init__(self):
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("A nonempty observation_id is required")


def _options(options):
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("Inference options must be a mapping or None")


class PreparedInferenceEngine:
    """The unchanged pinned backbone -> action-head inference body.

    ``stage(name)`` is an optional instrumentation context factory. Clean runs
    leave it unset: no profiler, CUDA events, recording, IO, decode, cache or
    synchronization is introduced by this boundary. Options pass through by
    identity, leaving model-specific validation to the pinned action head.
    """

    def __init__(self, model, *, stage=None):
        if stage is not None and not callable(stage):
            raise TypeError("stage must be a context factory or None")
        self.model = model
        self.stage = stage

    def vlm(self, backbone_inputs):
        if self.stage is None:
            return self.model.backbone(backbone_inputs)
        with self.stage("vlm"):
            return self.model.backbone(backbone_inputs)

    def action(self, features, action_inputs, options=None):
        _options(options)
        if self.stage is None:
            return self.model.action_head.get_action(features, action_inputs, options)
        with self.stage("action"):
            return self.model.action_head.get_action(features, action_inputs, options)

    def full(self, inputs, options=None):
        _options(options)
        return self.action(self.vlm(inputs.backbone), inputs.action, options)


def prepare_observation(policy, observation, *, observation_id):
    """Collate, transfer and own an input snapshot, outside model execution.

    Clone collated inputs before preparing: static VLM adapters register the
    identities of their returned metadata, which must not be cloned afterward.
    The pinned preparation keeps mutable pixels/state owned by this snapshot
    while its scope owns immutable registered metadata. CUDA copies use the
    caller's stream; the runner waits before its independent model timer.
    """
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("A nonempty observation_id is required")
    collated, states = policy._collate_observation(observation)
    owned_inputs, states = copy.deepcopy((collated["inputs"], states))
    backbone, action = policy.model.prepare_input(owned_inputs)
    return PreparedInputs(backbone, action, states, observation_id)


def prepare_observations(policy, observations, observation_ids, *, limit=16):
    """Bound resident input count before any collation/transfer work begins."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Prepared observation limit must be a positive integer")
    if not 0 < len(observations) <= limit:
        raise ValueError("Prepared observation count exceeds its positive limit")
    if len(observations) != len(observation_ids):
        raise ValueError("One identity is required per observation")
    if any(not isinstance(value, str) or not value for value in observation_ids):
        raise ValueError("Nonempty observation identities are required")
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("Prepared observation identities must be unique")
    return tuple(
        prepare_observation(policy, observation, observation_id=identity)
        for observation, identity in zip(observations, observation_ids)
    )


def decode_prediction(policy, prediction, inputs):
    """Convert a prediction using its observation's decoder state, outside CUDA."""
    return policy._decode_action(prediction, inputs.decode_state)
