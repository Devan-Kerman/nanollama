"""Working version of get_paths.py"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ueaj import model
from flax.nnx import rnglib as rng
from flax import nnx
from ueaj import opt
import optax
from jax import numpy as jnp

print("Loading model...")
tensor_config = model.ParamConfig("", group=nnx.Param).with_dtype(jnp.bfloat16)
config = model.LlamaConfig(
    vocab_size=128256,
    model_d=768,
    num_layers=12,
    tensor_config=tensor_config,
    layer_config=model.TransformerLayerConfig(
        model_d=768,
        use_gated_mlp=False,
        attention_config=model.AttentionConfig(
            _fused=False,
            model_d=768,
            kq_d=64,
            v_head_d=64,
            kv_heads=6,
            kv_q_ratio=2,
            rope_theta=10_000.0,
            param_config=tensor_config
        ),
        mlp_config=model.MLPConfig(
            model_d=768,
            hidden_d=768*4,
            param_config=tensor_config,
            activation_fn=lambda x: jnp.where(x < 0, -.0625, 1) * x * x
        ),
        norm_config=model.RMSNormConfig(
            model_d=768,
            scale="centered"
        )
    ),
)

instance = nnx.eval_shape(lambda: model.LlamaModel(config, rngs=rng.Rngs(0)))
state = nnx.state(instance, nnx.Param)
print("State from eval_shape model:", state)
