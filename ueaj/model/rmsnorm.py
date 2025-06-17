from dataclasses import dataclass
from typing import Literal

import jax
from flax import nnx
from jax import typing
from jax import numpy as jnp

from ueaj import utils

@dataclass(frozen=True)
class RMSNormConfig:
	model_d: int
	scale_dtype: typing.DTypeLike

	_accum_dtype: typing.DTypeLike = None

	scale: Literal["uncentered", "centered", "none"] = "centered"

	@property
	def accum_dtype(self):
		return self._accum_dtype or utils.DEFAULT_ACCUM_TYPE.value


class RMSNorm(nnx.Module):
	def __init__(self, config):
		super().__init__(config)
		self.scale = config.scale
		self.accum_dtype = config.accum_dtype

		initializer: nnx.Initializer | None = None
		if config.scale == "uncentered":
			initializer = nnx.initializers.ones
		elif config.scale == "centered":
			initializer = nnx.initializers.zeros
		elif config.scale == "none":
			initializer = None

		if initializer is not None:
			self.scale = nnx.Param(
				initializer(key=jax.random.PRNGKey(0), shape=(config.model_d,), dtype=config.scale_dtype)
			)
		else:
			self.scale = None

	def __call__(self, x):
		input_dtype = x.dtype

		var = jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=self.accum_dtype)
		x = x * jax.lax.rsqrt(var + 1e-06)

		if self.scale == "none":
			return x.astype(input_dtype)

		x, scale = utils.promote_fp8(x, self.scale)
		scale = jnp.expand_dims(scale, axis=-1)

		if self.scale == "uncentered":
			return (x * scale).astype(input_dtype)
		elif self.scale == "centered":
			return (x * (1 + scale)).astype(input_dtype)
		else:
			raise NotImplementedError(f"Unknown scaling method: {self.scale}")