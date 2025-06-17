from jax import numpy as jnp
import jax

LOW_PRECISION = set(map(jax.dtypes.canonicalize_dtype, (
	jnp.float8_e4m3fn,
	jnp.float8_e5m2,
	jnp.float8_e4m3fnuz,
	jnp.float8_e5m2fnuz,
	jnp.float8_e4m3b11fnuz,
)))