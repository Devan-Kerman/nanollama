import jax.lax
from jax import numpy as jnp
from . import config

def either(a, b):
	return a if a is not None else b

special_types = set(map(jax.dtypes.canonicalize_dtype, (
	jnp.float8_e4m3fn,
	jnp.float8_e5m2,
	jnp.float8_e4m3fnuz,
	jnp.float8_e5m2fnuz,
	jnp.float8_e4m3b11fnuz,
)))

def promote_fp8(*args):
	types = {a.dtype for a in args}

	if len(types) > 1 and any(t in special_types for t in types):
		return [
			(
				a.astype(config.DEFAULT_FP8_PROMO_TYPE.value)
				if a.dtype in special_types
				else a
			) for a in args
		]
	else:
		return args


if __name__ == "__main__":
	print(
		jnp.einsum(
			"bh,h->b",
			*promote_fp8(jnp.ones((2, 3), dtype=jnp.float8_e4m3fn),jnp.ones(3, dtype=jnp.float32)),
			precision=jax.lax.DotAlgorithmPreset.F32_F32_F32
		)
	)
