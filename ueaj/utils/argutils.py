from typing import List

import jax.lax
from jax import numpy as jnp
from ueaj.utils import config

def either(a, b):
	return a if a is not None else b


special_types = set(map(jax.dtypes.canonicalize_dtype, (
	jnp.float8_e4m3fn,
	jnp.float8_e5m2,
	jnp.float8_e4m3fnuz,
	jnp.float8_e5m2fnuz,
	jnp.float8_e4m3b11fnuz,
)))

def promote_fp8(*args) -> List[jax.Array]:
	has_fp8, gcd_type = False, None
	for arg in args:
		if arg.dtype in special_types:
			has_fp8 = True
		elif gcd_type is None:
			gcd_type = arg.dtype
		else:
			gcd_type = jnp.promote_types(gcd_type, arg.dtype)

	if has_fp8 and gcd_type is not None:
		return [
			(
				jax.lax.convert_element_type(a, jnp.dtype(gcd_type))
				if a.dtype in special_types
				else a
			) for a in args
		]
	else:
		return list(args)


if __name__ == "__main__":
	print(
		jnp.einsum(
			"bh,h->b",
			promote_fp8(jnp.ones((2, 3), dtype=jnp.float8_e4m3fn),jnp.ones(3, dtype=jnp.float32)),
			precision=jax.lax.DotAlgorithmPreset.F32_F32_F32
		)
	)
