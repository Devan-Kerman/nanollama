import itertools
from typing import Sequence, Mapping

import jax
# todo ueajsum to einsum translation
#	1. grab arguments and configs in order
#	2. add dummies for missing shapes, or reshape output
#	3. update mixsum and varsum

from flax import nnx
from flax.nnx import rnglib as rng

from . import config as cfg, parser
from .mixsum import mixsum

class Ueajsum(nnx.Module):
	def __init__(self, config: cfg.UeajsumConfig, shape_map: Mapping[str, int], rngs: rng.Rngs):
		super().__init__()

		self._default = config

		for k, v in config.kwarg_configs.items():
			if isinstance(v, cfg.ParamConfig):
				shape = tuple(map(lambda x: shape_map[x], v.shape))
				tensor = v.group(v.initializer(key=rngs.params(), shape=shape).astype(v.dtype))
				setattr(self, k, tensor)

		for i, v in enumerate(config.arg_configs):
			if isinstance(v, cfg.ParamConfig):
				shape = tuple(map(lambda x: shape_map[x], v.shape))
				tensor = v.group(v.initializer(key=rngs.params(), shape=shape).astype(v.dtype))
				setattr(self, f"w_{i}", tensor)


	def __call__(self, *args, **kwargs):
		return self.invoke(self._default, False, *args, **kwargs)

	def parse(self, expr: str) -> cfg.UeajsumConfig:
		return parser.parse(expr, no_instantiation=True, **self._default_arg_dict(pairs=False))

	def parse_and_call(self, expr: str, *args, **kwargs):
		return self.invoke(self.parse(expr), False, *args, **kwargs)

	def invoke(self, terms: cfg.UeajsumConfig, override: bool, *args: jax.Array, **kwargs: jax.Array) -> jax.Array:
		arg_dict = self._build_args(terms, override, args, kwargs)

		accumulator = None
		for expr in terms.sums:
			dims = set()
			tensors = []
			configs = []

			for arg in expr:
				tensor, config = arg_dict[arg]
				tensors.append(tensor)
				configs.append(config)
				dims.update(config.shape)

			output_shape = "".join([v for v in terms.result_config.shape if v in dims])

			output = mixsum(configs, terms.result_config.with_shape(output_shape), *tensors)

			old_shape = output.shape
			new_shape = []
			idx = 0
			for v in terms.result_config.shape:
				if v in output_shape:
					new_shape.append(old_shape[idx])
					idx += 1
				else:
					new_shape.append(1)

			output = output.reshape(new_shape)

			if accumulator is None:
				accumulator = output
			else:
				accumulator += output

		return accumulator

	def _default_arg_dict(self, pairs: bool = True):
		arg_dict = {}

		for k, v in enumerate(self._default.arg_configs):
			if isinstance(v, cfg.ParamConfig):
				arg_dict[k] = (getattr(self, f"w_{k}"), v) if pairs else v

		for k, v in self._default.kwarg_configs.items():
			if isinstance(v, cfg.ParamConfig):
				arg_dict[k] = (getattr(self, k), v) if pairs else v

		return arg_dict

	def _build_args(self, terms: cfg.UeajsumConfig, override: bool, args: Sequence[jax.Array], kwargs: Mapping[str, jax.Array]):
		arg_dict = self._default_arg_dict()

		c = 0
		for arg in args:
			while c in arg_dict:
				c += 1
			arg_dict[c] = (arg, None)

		for k, v in kwargs.items():
			arg_dict[k] = (v, None)

		for k, v in itertools.chain(enumerate(terms.arg_configs), terms.kwarg_configs.items()):
			if k not in arg_dict:
				raise ValueError(f"Missing argument {k}")
			entry = arg_dict[k]
			if override or entry[1] is None:
				arg_dict[k] = (entry[0], v)

		return arg_dict