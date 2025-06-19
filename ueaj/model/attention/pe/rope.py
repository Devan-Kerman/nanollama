from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Tuple

from flax.core import FrozenDict
from jax import numpy as jnp
import jax
from flax import nnx


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

class RoPE(nnx.Module):
	def __init__(self, config: RoPEConfig):
		super().__init__()
		self.config = config

	def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
		embeds = self.config.compute_embedding(jnp.arange(x.shape[1]))
		return self.invoke(x, embeds)
		
	def compute_freqs(self, position_ids: jax.Array) -> jax.Array:
		"""Compute RoPE frequencies for given position IDs."""
		return self.config.compute_embedding(position_ids)

	def invoke(self, x: jax.Array, embeds: Tuple[jax.Array, jax.Array] | jax.Array) -> jax.Array:
		batch_size, seq_len, model_d = x.shape[0], x.shape[1], x.shape[-1]
		half = model_d // 2
		x1 = x[..., :half]
		x2 = x[..., half:]

		sin_part, cos_part = embeds

		# Handle different embed shapes
		if sin_part.ndim == 2:  # Shape: (seq_len, half)
			if seq_len < sin_part.shape[0]:
				sin_part = sin_part[:seq_len, ...]
				cos_part = cos_part[:seq_len, ...]
			
			extra_dims = len(x1.shape) - 3
			cos_part = cos_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))
			sin_part = sin_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))
		elif sin_part.ndim == 3:  # Shape: (batch_size, seq_len, half)
			# Already in the right shape for broadcasting
			extra_dims = len(x1.shape) - 3
			if extra_dims > 0:
				cos_part = cos_part.reshape(cos_part.shape[:2] + (1,) * extra_dims + (half,))
				sin_part = sin_part.reshape(sin_part.shape[:2] + (1,) * extra_dims + (half,))

		out1 = x1 * cos_part - x2 * sin_part
		out2 = x2 * cos_part + x1 * sin_part

		return jnp.concatenate([out1, out2], axis=-1)

if __name__ == "__main__":
	rope = RoPE(RoPEConfig(0.1, 512, jnp.float32))
	rope(jnp.zeros((2, 3, 512)))
	rope(jnp.zeros((2, 3, 4, 512)))
	rope(jnp.zeros((2, 3, 4, 5, 512)))
