import os
import typing

import jax
from jax import config
from jax import numpy as jnp

T = typing.TypeVar('T')

class Value(typing.Generic[T]):
	__slots__ = ("_name", "value")

	_name: str
	value: T

	def __init__(self, name: str, default: T):
		self._name = name
		self._set(default)

	def _set(self, value: T) -> None:
		self.value = value


def make_config(name: str, default: T, *args, **kwargs) -> Value[T]:
	holder = Value(name, default)
	config.add_option(
		name,
		holder,
		str,
		args,
		kwargs
	)
	return holder

backend = jax.default_backend()
DEFAULT_FP8_PROMO_TYPE = make_config(
	'default_fp8_promotion_type',
	jax.dtypes.canonicalize_dtype(os.environ.get('DEFAULT_F8_PROMO_TYPE', jnp.bfloat16 if backend == 'tpu' else jnp.float16)),
	help='Set the default float8 promotion type for when promotion is needed.'
)

DEFAULT_ACCUM_TYPE = make_config(
	'default_accumulation_type',
	jax.dtypes.canonicalize_dtype(os.environ.get('DEFAULT_ACCUM_TYPE', jnp.bfloat16 if backend == 'tpu' else jnp.float32)),
	help='Set the default accumulation type for rmsnorm among other things.'
)
