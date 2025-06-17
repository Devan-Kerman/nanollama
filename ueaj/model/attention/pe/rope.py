from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Tuple

from flax.core import FrozenDict
from jax import numpy as jnp
import jax
from flax import nnx


class PositionalEmbedding(nnx.Variable):
	pass

@dataclass(frozen=True)
class RoPEConfig:
	rope_theta: float
	rope_d: int
	value_dtype: Any

	def compute_embedding(self, pos: jax.Array) -> jax.Array:
		half = self.rope_d // 2
		freq_seq = jnp.arange(half)
		inv_freq = 1.0 / (self.rope_theta ** (freq_seq / half))
		angles = jnp.einsum("...l,h->...lh", pos, inv_freq)
		sin_part = jnp.sin(angles).astype(dtype=self.value_dtype)
		cos_part = jnp.cos(angles).astype(dtype=self.value_dtype)
		return jnp.array([sin_part, cos_part])

@contextmanager
def with_rope_caches(module: nnx.Module, indexes: jax.Array):
	populate_rope_caches(module, indexes)
	try:
		yield
	finally:
		clear_rope_caches(module)

def populate_rope_caches(module: nnx.Module, indexes: jax.Array, caches: FrozenDict | dict | None = None):
	caches = dict(caches) if caches is not None else {}
	for iter_module in module.iter_modules():
		if isinstance(iter_module, RoPE):
			if iter_module.config not in caches:
				embeds = iter_module.config.compute_embedding(indexes)
				caches[iter_module.config] = embeds
			else:
				embeds = caches[iter_module.config]

			iter_module.cache.value = embeds
	return FrozenDict(caches)


def clear_rope_caches(module: nnx.Module):
	for iter_module in module.iter_modules():
		if isinstance(iter_module, RoPE):
			iter_module.cache.value = None


class RoPE(nnx.Module):
	def __init__(self, config: RoPEConfig):
		super().__init__()
		self.config = config
		self.cache = PositionalEmbedding(None)

	def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
		if self.cache.value is None:
			embeds = self.config.compute_embedding(jnp.arange(x.shape[1]))
			self.cache.value = embeds
		else:
			embeds = self.cache.value

		return self.invoke(x, embeds)

	def invoke(self, x: jax.Array, embeds: Tuple[jax.Array, jax.Array] | jax.Array) -> jax.Array:
		batch_size, seq_len, model_d = x.shape[0], x.shape[1], x.shape[-1]
		half = model_d // 2
		x1 = x[..., :half]
		x2 = x[..., half:]

		sin_part, cos_part = embeds

		if seq_len < sin_part.shape[1]:
			sin_part = sin_part[:seq_len, ...]
			cos_part = cos_part[:seq_len, ...]

		extra_dims = len(x1.shape) - 3

		cos_part = cos_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))
		sin_part = sin_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))

		out1 = x1 * cos_part - x2 * sin_part
		out2 = x2 * cos_part + x1 * sin_part

		return jnp.concatenate([out1, out2], axis=-1)

if __name__ == "__main__":
	rope = RoPE(RoPEConfig(0.1, 512, jnp.float32))
	rope(jnp.zeros((2, 3, 512)))
	rope(jnp.zeros((2, 3, 4, 512)))
	rope(jnp.zeros((2, 3, 4, 5, 512)))
