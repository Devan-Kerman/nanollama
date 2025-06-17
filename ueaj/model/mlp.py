from dataclasses import dataclass, replace
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import ueajsum as us
from ueaj.utils import *


@dataclass
class MLPConfig:
	model_d: int
	hidden_d: int

	activation_fn: Callable[[jax.Array], jax.Array] = nnx.swish

	param_config: us.ParamConfig = us.ParamConfig("", group=nnx.Param)

	_gate_config: us.ParamConfig | None = None
	_up_config: us.ParamConfig | None = None
	_down_config: us.ParamConfig | None = None

	@property
	def up_config(self):
		return either(self._up_config, self.param_config)

	@property
	def down_config(self):
		return either(self._down_config, self.param_config.with_initializer(nnx.initializers.zeros))

	@property
	def gate_config(self):
		return either(self._gate_config, self.param_config)

	def with_gate(self, config: us.ParamConfig):
		return replace(self, _gate_config=config)

	def with_up(self, config: us.ParamConfig):
		return replace(self, _up_config=config)

	def with_down(self, config: us.ParamConfig):
		return replace(self, _down_config=config)


class MLP(nnx.Module):
	def __init__(self, config: MLPConfig, rngs: rng.Rngs):
		super().__init__()
		size_dict = {'d': config.model_d, 'h': config.hidden_d}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		self.up_proj = make_ueajsum(
			us.parse("bnd,*dh->bnh").param(config.up_config)
		)
		self.down_proj = make_ueajsum(
			us.parse("bnh,*dh->bnd").param(config.down_config)
		)

		self.activation_fn = config.activation_fn

	def __call__(self, x):
		x = self.up_proj(x)
		x = self.activation_fn(x)
		x = self.down_proj(x)
		return x


class GMLP(nnx.Module):
	def __init__(self, config: MLPConfig, rngs: rng.Rngs):
		super().__init__()
		size_dict = {'d': config.model_d, 'h': config.hidden_d, 'i': 2}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		self.fused_proj = make_ueajsum(
			us.parse("bnd,*dhi->bnhi").param(config.up_config)
		)
		self.down_proj = make_ueajsum(
			us.parse("bnh,bnh,*hd->bnd").param(config.down_config)
		)
		self.activation_fn = config.activation_fn

	def __call__(self, x):
		fused = self.fused_proj(x)

		up, gate = fused[:, :, :, 0], fused[:, :, :, 1]
		gate = self.activation_fn(gate)

		if x.dtype not in LOW_PRECISION:
			x = self.down_proj(up, gate)
		else:
			s = jnp.max(jnp.abs(up), axis=(0, 1), keepdims=True) + 1
			x = (s * up) * gate
			x = x / s
			x = self.down_proj.parse_and_call("bnh,[2]->bnd", x)

		return x


if __name__ == "__main__":
	config = MLPConfig(
		model_d=16,
		hidden_d=32,
		activation_fn=nnx.relu,
	)
	x = jnp.ones((1, 1, 16))
	m = MLP(config.with_down(config.param_config.with_initializer(nnx.initializers.lecun_normal())), rngs=rng.Rngs(0))
	gm = GMLP(config.with_down(config.param_config.with_initializer(nnx.initializers.lecun_normal())), rngs=rng.Rngs(0))

	print(m(x))
	print(gm(x))