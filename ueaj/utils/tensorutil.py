import functools

import jax
import jax.numpy as jnp
from jax import lax
from typing import Any, Callable, Union, Tuple, Optional
import jax.tree_util as tree

def chunked_scan(
	f: Callable,
	init: Any,
	xs: Any,
	chunk_size: int,
	axis: Union[int, Tuple[int, ...], dict, None] = 0,
	out_axis: Union[int, Tuple[int, ...], dict, None] = 0,
) -> Tuple[Any, Any]:
	def transpose(x, ax, fwd=True):
		if isinstance(ax, int):
			from_, to = (0, ax) if fwd else (ax, 0)
			return jax.tree.map(
				lambda x: jnp.moveaxis(x, from_, to),
				x
			)
		return jax.tree.map(functools.partial(transpose, ax=axis, fwd=fwd), xs)

	xs_t = transpose(xs, axis)
	leaves = jax.tree.leaves(xs_t)
	if not leaves:
		return init, xs

	scan_length = leaves[0].shape[0]
	for i, leaf in enumerate(leaves):
		assert leaf.shape[0] == scan_length, f"Scan length mismatch: {leaf.shape[0]} != {scan_length} for leaf {i}"

	# Split into chunks
	num_full_chunks = scan_length // chunk_size
	remainder = scan_length % chunk_size

	xs_scan = jax.tree.map(
		lambda t: t[:num_full_chunks * chunk_size].reshape(
			(-1, chunk_size) + t.shape[1:]
		),
		xs_t
	)

	carry, ys = jax.lax.scan(
		lambda carry, x: f(carry, transpose(x, axis, fwd=False)),
		init,
		xs_scan,
	)

	if out_axis != 0:
		ys = transpose(ys, out_axis, fwd=True)

	if remainder > 0:
		xs_rem = jax.tree.map(
			lambda t: t[-remainder:],
			xs_t
		)

		carry, ys_rem = f(carry, transpose(xs_rem, axis, fwd=False))
		if out_axis != 0:
			ys_rem = transpose(ys_rem, out_axis, fwd=True)

		if ys_rem.ndim == 0:
			ys_rem = ys_rem.reshape((1,))

		ys = jax.tree.map(
			lambda y, y_rem: jnp.concatenate((y, y_rem), axis=out_axis),
			ys,
			ys_rem
		)
		return carry, ys
	return carry, ys

def test():
	x = jnp.ones((16, 123, 3))
	y = jnp.zeros((16, 123, 3))
	from flax import nnx
	from flax.nnx import rnglib as rng

	layer = nnx.Linear(3, 3, use_bias=False, rngs=rng.Rngs(0))

	def f(layer, carry, x):
		test = layer(x[0])
		print("carry", carry.shape)
		print("x", x[0].shape)
		return jnp.array([carry[0] + test.sum(), carry[1] + x[1].sum()]), carry[0] + carry[1]

	carry = jnp.zeros((2,))
	@nnx.grad
	def test(layer, x, y, carry):
		return chunked_scan(functools.partial(f, layer), carry, (x, y), 8, axis=1, out_axis=0)[0][0]

	print(test(layer, x, y, carry))