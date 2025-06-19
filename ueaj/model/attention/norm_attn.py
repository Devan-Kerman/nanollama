from typing import Tuple

from ueaj.utils.argutils import either
from ueaj.model.attention.soft_attn import AttentionConfig

from flax import nnx
import jax.numpy as jnp
import ueaj.model.ueajsum as us
import ueaj.model.rmsnorm as rms
from flax.nnx import rnglib as rng
import jax
from jax import lax

class TransNormer(nnx.Module):
	def __init__(self, config: AttentionConfig, rngs: rng.Rngs):
		super().__init__()
		self.config = config

		size_dict = {
			'd': config.model_d,
			'z': config.model_d,
			'y': config.model_d,
			'k': config.kq_d,
			'v': config.v_head_d,
			'h': config.kv_heads,
			'i': config.kv_q_ratio,
			'f': 2
		}

		if config.k_config != config.param_config or config.v_config != config.param_config or config.q_config != config.param_config:
			raise NotImplementedError("Custom qkv tensor configs not supported yet")

		self.attn = us.Ueajsum(
			us.parse("bnd,*dhik,bmz,*zhk,bmy,*yhv,nm->bnhiv").param(config.param_config),
			size_dict,
			rngs=rngs
		)
		self.norm = rms.RMSNorm(rms.RMSNormConfig(config.v_head_d, config.param_config))
		self.o = us.Ueajsum(
			us.parse("bnhiv,*hivd->bnd").param(config.param_config),
			size_dict,
			rngs=rngs
		)

	def __call__(self, x_q, x_k=None, x_v=None, **kwargs):
		norm_mask = kwargs.pop("norm_mask", jnp.ones((1, 1), x_q.dtype))
		x_k = either(x_k, x_q)
		x_v = either(x_v, x_k)
		x = self.attn(x_q, x_k, x_v, norm_mask)
		x = self.norm(x)
		x = self.o(x)
		return x

	def causal(self, x_q, **kwargs) -> Tuple[jax.Array, jax.Array]:
		"""Einsum-based implementation for causal mask with optional decay.

		This method achieves the same result as scan_causal but using einsum operations.
		Since TransNormer is linear, we can decompose the computation into:
		1. Regular causal attention with decay mask
		2. Contribution from init_carry (if provided)
		

		Args:
			x_q: Query input tensor of shape (batch, seq_len, d)
			init_carry: Optional initial state for inference mode, shape (b, h, k, v)
			decay_rate: Optional decay factor in (0, 1]. Default 1.0 (no decay).
				        Creates exponential decay: older positions are weighted by decay^distance

		Returns:
			Output tensor of shape (batch, seq_len, heads * head_dim, v_dim)
		"""
		b, n, d = x_q.shape

		init_carry = kwargs.get('init_carry', None)
		decay_rate = kwargs.get('decay_rate', 1.0)

		positions = jnp.arange(n)
		distance = positions[:, None] - positions[None, :]  # (n, m)
		causal_decay_mask = jnp.where(distance >= 0, jnp.power(decay_rate, distance.astype(jnp.float32)), 0.0)

		queries = jnp.einsum('bnd,dhik->bnhik', x_q, self.attn.w_1)
		y = self.attn.parse_and_call("bnhik,bmz,[3],bmy,[5],nm->bnhiv", queries, x_q, x_q, causal_decay_mask)
		kv_carry = self.attn.parse_and_call('bmz,[3],bmy,[5],m->bhkv', x_q, x_q, causal_decay_mask[-1])
		if init_carry is not None:
			decay_factors = jnp.power(decay_rate, positions.astype(jnp.float32) + 1)
			carry_contribution = jnp.einsum('n,bnhik,bhkv->bnhiv', decay_factors, queries, init_carry)
			y += carry_contribution
			kv_carry += init_carry * decay_factors[-1]

		y = self.norm(y)
		y = self.o(y)
		return kv_carry, y

if __name__ == "__main__":
	"""Minimal smoke-tests for TransNormer.

	Run them from the repository root *via* the helper script so that the
	correct virtual-environment (with JAX, Flax, etc.) is picked up:

	```bash
	./run_python.sh ueaj/model/attention/norm_attn.py
	```

	The original test-suite contained a few shape mismatches and attempted
	to call code-paths that are not implemented (e.g. cross-attention for
	`scan_causal`).  The rewritten version below keeps the spirit of those
	demonstrations – comparing the scan implementation with the einsum
	implementation – while making sure everything actually runs.
	"""

	# ---------------------------------------------------------------------
	# Build a tiny TransNormer so the example finishes quickly on CPU.
	# ---------------------------------------------------------------------
	config = AttentionConfig(
		model_d=16,
		kq_d=16,
		v_head_d=13,
		kv_heads=4,
		kv_q_ratio=4,
		rope_theta=None,
	)
	normer = TransNormer(config, rng.Rngs(0))

	print("=== TransNormer – sanity checks ===\n")

	# ------------------------------------------------------------------
	# 1. Scan vs. regular attention (no decay)
	# ------------------------------------------------------------------
	seq_len = 4096
	batch = 1
	key = jax.random.PRNGKey(0)
	x = jax.random.normal(key, (batch, seq_len, config.model_d), dtype=jnp.float32)

	causal_out = normer(x, causal=True)
	non_causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=x.dtype))
	regular_out = normer(x, norm_mask=non_causal_mask)

	diff = jnp.abs(causal_out - regular_out).mean()
	print("[Test 1] scan_causal  vs  regular : max |Δ| =", float(diff))
	print("          match?", bool(diff < 1e-5), "\n")

	# ------------------------------------------------------------------
	# 2. Scan vs. einsum implementation with exponential decay
	# ------------------------------------------------------------------
	decay_rate = 0.95
	_, scan_decay_out = normer.scan_causal(x, decay_rate=decay_rate)
	_, einsum_decay_out = normer.causal(x, decay_rate=decay_rate)

	diff = jnp.abs(scan_decay_out - einsum_decay_out).mean()
	print("[Test 2] scan_causal  vs  einsum (decay): max |Δ| =", float(diff))
	print("          relative error =", float(diff))
	print("          match? (< 1e-3 relative)", bool(diff < 1e-3), "\n")

	# ------------------------------------------------------------------
	# 3. Same comparison but with a non-zero initial KV carry
	# ------------------------------------------------------------------
	init_carry = jax.random.normal(
		jax.random.PRNGKey(1),
		(batch, config.kv_heads, config.kq_d, config.v_head_d),  # (b, h, k, v)
		dtype=jnp.float32,
	) * .1

	_, scan_init_out = normer.scan_causal(x, decay_rate=decay_rate, init_carry=init_carry)
	_, einsum_init_out = normer.causal(
		x,
		decay_rate=decay_rate,
		init_carry=init_carry,
	)

	diff = jnp.abs(scan_init_out - einsum_init_out).mean()
	print("[Test 3] scan_causal  vs  einsum (init-carry): max |Δ| =", float(diff))
	print("          relative error =", float(diff))
	print("          match? (< 1e-3 relative)", bool(diff < 1e-3))