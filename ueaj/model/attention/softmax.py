from abc import ABC
from dataclasses import dataclass, replace

from flax import nnx
from flax.nnx import rnglib as rng
import ueaj.model.ueajsum as us
from ueaj.utils import *
from kvax.ops import (
	create_attention_mask,
	flash_attention,
)
from kvax.utils import (
	attention_specs,
	permute_tokens_context_parallelism,
	unpermute_tokens_context_parallelism,
)

import pe

@dataclass(frozen=True)
class AttentionConfig:
	model_d: int

	kq_d: int
	v_head_d: int

	kv_heads: int
	kv_q_ratio: int

	rope_theta: float | None

	param_config: us.ParamConfig = us.ParamConfig("", group=nnx.Param)

	_k_config: us.ParamConfig | None = None
	_v_config: us.ParamConfig | None = None
	_q_config: us.ParamConfig | None = None
	_o_config: us.ParamConfig | None = None

	@property
	def k_config(self):
		return either(self._k_config, self.param_config)

	@property
	def v_config(self):
		return either(self._v_config, self.param_config)

	@property
	def q_config(self):
		return either(self._q_config, self.param_config)

	@property
	def o_config(self):
		return either(self._o_config, self.param_config.with_initializer(nnx.initializers.zeros))

	def with_down(self, config: us.ParamConfig):
		return replace(self, _k_config=config)

	def with_up(self, config: us.ParamConfig):
		return replace(self, _v_config=config)

	def with_q(self, config: us.ParamConfig):
		return replace(self, _q_config=config)

	def with_o(self, config: us.ParamConfig):
		return replace(self, _o_config=config)

class SoftmaxAttention(nnx.Module, ABC):
	def __init__(self, config: AttentionConfig, rngs: rng.Rngs):
		super().__init__()
		self.config = config

		size_dict = {'d': config.model_d, 'k': config.kq_d, 'v': config.v_head_d, 'h': config.kv_heads, 'i': config.kv_q_ratio}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		self.k = make_ueajsum(us.parse("bnd,*dhk->bnhk").param(config.k_config))
		self.v = make_ueajsum(us.parse("bnd,*dhv->bnhv").param(config.v_config))
		self.q = make_ueajsum(us.parse("bnd,*dhik->bnhik").param(config.q_config))
		self.o = make_ueajsum(us.parse("bnhiv,*hivd->bnd").param(config.o_config))

		if config.rope_theta is not None:
			self.rope = pe.RoPE(pe.RoPEConfig(config.rope_theta, config.kq_d, jnp.float32))
		else:
			self.rope = None


	def __call__(self, x, **kwargs):
		k = self.k(x)
		v = self.v(x)
		q = self.q(x)

		if self.rope:
			if 'rope' in kwargs:
				rope = kwargs['rope']
				del kwargs['rope']
				k = self.rope.invoke(k, rope)
				q = self.rope.invoke(q, rope)
			else:
				k = self.rope(k)
				q = self.rope(q)

		out = flash_attention(query=q.reshape(), key=k, value=v, **kwargs)

		return self.o(out)
