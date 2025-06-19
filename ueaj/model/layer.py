import functools
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import jax
from flax import nnx
from flax.nnx import rnglib as rng
import jax.numpy as jnp
from transformer_engine.jax.attention import SequenceDescriptor

from ueaj.model.attention.soft_attn import SoftmaxAttention, AttentionConfig
from ueaj.model.mlp import MLP, GMLP, MLPConfig
from ueaj.model.rmsnorm import RMSNorm, RMSNormConfig
from ueaj.model.ueajsum import ParamConfig
from ueaj.utils.argutils import either


@dataclass(frozen=True)
class TransformerLayerConfig:
	"""Configuration for a transformer layer combining attention and MLP blocks."""
	model_d: int

	norm_config: RMSNormConfig
	attention_config: AttentionConfig
	mlp_config: MLPConfig

	# Private RMSNorm configurations
	_attention_norm_config: Optional[RMSNormConfig] = None
	_mlp_norm_config: Optional[RMSNormConfig] = None

	# Whether to use gated MLP
	use_gated_mlp: bool = False

	@property
	def attn_norm_config(self) -> RMSNormConfig:
		"""Get attention normalization config, defaulting to standard RMSNorm."""
		return either(
			self._attention_norm_config,
			self.norm_config
		)

	@property
	def mlp_norm_config(self) -> RMSNormConfig:
		"""Get MLP normalization config, defaulting to standard RMSNorm."""
		return either(
			self._mlp_norm_config,
			self.norm_config
		)

	def with_attention_config(self, config: AttentionConfig):
		"""Update attention configuration."""
		return replace(self, attention_config=config)

	def with_mlp_config(self, config: MLPConfig):
		"""Update MLP configuration."""
		return replace(self, mlp_config=config)

	def with_attention_norm(self, config: RMSNormConfig):
		"""Update attention normalization configuration."""
		return replace(self, _attention_norm_config=config)

	def with_mlp_norm(self, config: RMSNormConfig):
		"""Update MLP normalization configuration."""
		return replace(self, _mlp_norm_config=config)

	def with_gated_mlp(self, use_gated: bool = True):
		"""Enable or disable gated MLP."""
		return replace(self, use_gated_mlp=use_gated)

	def validate(self):
		"""Validate that all dimensions are consistent."""
		assert self.attention_config.model_d == self.model_d, \
			f"Attention model_d {self.attention_config.model_d} != {self.model_d}"
		assert self.mlp_config.model_d == self.model_d, \
			f"MLP model_d {self.mlp_config.model_d} != {self.model_d}"
		if self._attention_norm_config:
			assert self._attention_norm_config.model_d == self.model_d, \
				f"Attention norm model_d {self._attention_norm_config.model_d} != {self.model_d}"
		if self._mlp_norm_config:
			assert self._mlp_norm_config.model_d == self.model_d, \
				f"MLP norm model_d {self._mlp_norm_config.model_d} != {self.model_d}"


class TransformerLayer(nnx.Module):
	"""A transformer layer with attention and MLP blocks."""

	def __init__(self, config: TransformerLayerConfig, rngs: rng.Rngs):
		super().__init__()

		# Validate configuration
		config.validate()

		# Initialize attention components
		self.attn = SoftmaxAttention(config.attention_config, rngs)
		self.attn_norm = RMSNorm(config.attn_norm_config)

		# Initialize MLP components
		mlp_class = GMLP if config.use_gated_mlp else MLP
		self.mlp = mlp_class(config.mlp_config, rngs)
		self.mlp_norm = RMSNorm(config.mlp_norm_config)

		self.config = config

	def __call__(self, x, **kwargs):
		"""
		Forward pass through the transformer layer.

		Args:
			x: Input tensor of shape (batch, sequence, model_d)
			**kwargs: Additional arguments passed to attention (e.g., rope, sequence_descriptor)

		Returns:
			Output tensor of same shape as input
		"""
		# Attention block with residual connection
		# Use TransformerEngine workaround for attention normalization
		x += self.attn(self.attn_norm(x), **kwargs)
		x += self.mlp(self.mlp_norm(x))
		return x

	def bwd(self, x: jax.Array, dx: jax.Array, **kwargs):
		"""
		Backward pass through the transformer layer.

		Args:
			x: Input tensor of shape (batch, sequence, model_d)
			**kwargs: Additional arguments passed to attention (e.g., rope, sequence_descriptor)

		Returns:
			Output tensor of same shape as input
		"""
		graph_def, params, etc = nnx.split(self, nnx.Param, ...)

		def g_def(params, x):
			return nnx.merge(graph_def, params, etc)(x, **kwargs)

		output, callback = jax.vjp(g_def, params, x)
		dparams, dx = callback(dx)

		return output, (dparams, dx)

if __name__ == "__main__":
	"""Test TransformerLayer.bwd."""
	# Create layer config with all necessary settings
	model_d = 2048
	tensor_config = ParamConfig("", group=nnx.Param).with_dtype(jnp.bfloat16)# .with_grad_dtype(jnp.float32)

	# Load config using transformers
	attention_config = AttentionConfig(
		model_d=model_d,
		kq_d=64,
		v_head_d=64,
		kv_heads=8,
		kv_q_ratio=1,
		rope_theta=500_000.,
		param_config=tensor_config
	)

	layer_config = TransformerLayerConfig(
		model_d=model_d,
		use_gated_mlp=True,
		attention_config=attention_config, # .with_o(attention_config.o_config.with_initializer(nnx.initializers.lecun_normal())),
		mlp_config=MLPConfig(
			model_d=model_d,
			hidden_d=1024,
			param_config=tensor_config
		),
		norm_config=RMSNormConfig(
			model_d=model_d,
			_scale_dtype=tensor_config.dtype,
			scale="centered"
		),
	)

	kwargs = {
		# 'mask': jnp.ones((8192, 8192), dtype=jnp.bool),
		'sequence_descriptor': SequenceDescriptor.from_seqlens(jnp.array([8192]*4)),
	}

	@nnx.jit
	@nnx.value_and_grad(argnums=(0, 1, 3))
	def layers(layer1, layer2, x, y):
		x = layer1(x, **kwargs)
		x = layer2(x, **kwargs)
		return jnp.sum((x - y)**2) / 2

	# def layers(layer, x, y):
	# 	gdef, params, etc = nnx.split(layer, nnx.Param, ...)
	# 	@jax.jit
	# 	@functools.partial(jax.value_and_grad, argnums=(0, 1))
	# 	def inner(params, x, y):
	# 		layer = nnx.merge(gdef, params, etc)(x, **kwargs)
	# 		return jnp.sum((layer - y)**2) / 2
	# 	return inner(params, x, y)

	layer1 = TransformerLayer(layer_config, rngs=rng.Rngs(0))
	layer2 = TransformerLayer(layer_config, rngs=rng.Rngs(0))

	x = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	dh = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	output, (dlayer1, dlayer2, dx) = layers(layer1, layer2, x, x+dh)

	print(output - x)
	print(dlayer1)
	print(dlayer2)
	print(dx - dh)