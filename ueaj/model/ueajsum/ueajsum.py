import itertools
from typing import Sequence, Mapping

import jax
# todo ueajsum to einsum translation
#	1. grab arguments and configs in order
#	2. add dummies for missing shapes, or reshape output
#	3. update mixsum and varsum

from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model.ueajsum import config as cfg
from ueaj.model.ueajsum import parser
from ueaj.model.ueajsum.mixsum import mixsum

class Ueajsum(nnx.Module):
	def __init__(self, config: cfg.UeajsumConfig, shape_map: Mapping[str, int], rngs: rng.Rngs):
		super().__init__()

		self.config = config

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
		return self.invoke(self.config, False, *args, **kwargs)

	def parse(self, expr: str) -> cfg.UeajsumConfig:
		kwargs = self._default_arg_dict(pairs=False)
		args = [None] * (max([i for i in kwargs.keys() if isinstance(i, int)])+1)
		for k, v in dict(kwargs).items():
			if isinstance(k, int):
				args[k] = v
				del kwargs[k]
		return parser.parse(expr, True, *args, **kwargs)

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

	def get_argument_axes_info(self, terms: cfg.UeajsumConfig = None):
		"""
		Determines the reducing and non-reducing axes for each argument
		based on a Ueajsum configuration.

		Args:
			terms: A UeajsumConfig object. If None, `self.config` is used.

		Returns:
			A dictionary mapping each argument identifier (int for positional,
			str for keyword) to a dictionary containing:
			- 'reducing_axes': A sorted list of axes that are reduced.
			- 'non_reducing_axes': A sorted list of axes that are not reduced.
		"""
		if terms is None:
			terms = self.config

		arg_configs = {}
		for i, term_config in enumerate(terms.arg_configs):
			if term_config:
				arg_configs[i] = term_config
		for k, term_config in terms.kwarg_configs.items():
			if term_config:
				arg_configs[k] = term_config

		output_axes = set(terms.result_config.shape)
		axes_info = {}

		for arg_ref, arg_config in arg_configs.items():
			arg_shape_axes = set(arg_config.shape)
			
			non_reducing = arg_shape_axes.intersection(output_axes)
			reducing = arg_shape_axes.difference(output_axes)

			axes_info[arg_ref] = {
				'reducing_axes': reducing,
				'non_reducing_axes': non_reducing,
			}
			
		return axes_info

	def _default_arg_dict(self, pairs: bool = True):
		arg_dict = {}

		for k, v in enumerate(self.config.arg_configs):
			if isinstance(v, cfg.ParamConfig):
				param = getattr(self, f"w_{k}")
				# Extract value from nnx.Param if needed
				param_value = param.value if hasattr(param, 'value') else param
				arg_dict[k] = (param_value, v) if pairs else v

		for k, v in self.config.kwarg_configs.items():
			if isinstance(v, cfg.ParamConfig):
				param = getattr(self, k)
				# Extract value from nnx.Param if needed
				param_value = param.value if hasattr(param, 'value') else param
				arg_dict[k] = (param_value, v) if pairs else v

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
			if v is None:
				continue
			if k not in arg_dict:
				raise ValueError(f"Missing argument {k}")
			entry = arg_dict[k]
			if override or entry[1] is None:
				arg_dict[k] = (entry[0], v)

		return arg_dict