"""Evaluate the public Playground ONNX expert with an equivalent JAX MLP."""

from pathlib import Path

import jax
import jax.numpy as jnp
import onnx
from onnx import numpy_helper


def load_expert(path: str | Path):
    model = onnx.load(str(path))
    tensors = {
        item.name: jnp.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    nodes = list(model.graph.node)
    assert [n.op_type for n in nodes] == [
        "Sub",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Sigmoid",
        "Mul",
        "Gemm",
        "Split",
        "Tanh",
    ], "Unexpected expert architecture; conversion must be reviewed"
    mean = tensors[nodes[0].input[1]]
    inv_std = tensors[nodes[1].input[1]]
    layers = []
    for node in nodes:
        if node.op_type != "Gemm":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert not attrs.get("transA", 0) and not attrs.get("transB", 0)
        assert attrs.get("alpha", 1) == 1 and attrs.get("beta", 1) == 1
        layers.append((tensors[node.input[1]], tensors[node.input[2]]))

    def policy(observation):
        x = (observation - mean) * inv_std
        for weight, bias in layers[:-1]:
            x = jax.nn.silu(x @ weight + bias)
        weight, bias = layers[-1]
        return jnp.tanh((x @ weight + bias)[..., :16])

    return policy
