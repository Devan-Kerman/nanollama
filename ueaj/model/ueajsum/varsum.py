from dataclasses import dataclass
from functools import reduce
from operator import mul, or_
from typing import Sequence, Optional, Counter, Tuple, List

import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike
from ueaj import utils

def accumulation_type(precision, result_type):
	if isinstance(precision, lax.DotAlgorithm):
		preferred_element_type = precision.accumulation_type
	elif isinstance(precision, lax.DotAlgorithmPreset):
		preferred_element_type = precision.accumulation_type
	elif isinstance(precision, tuple) or isinstance(precision, str) or isinstance(precision, lax.Precision):
		preferred_element_type = jnp.float32
	else:
		preferred_element_type = None

	if preferred_element_type is None:
		preferred_element_type = result_type
	return preferred_element_type

def parse_einsum(expr: str) -> Tuple[List[str], str]:
	"""Return the input‑subscript list and output subscripts from an einsum expr."""
	packed = expr.replace(" ", "")
	if "->" in packed:
		lhs, rhs = packed.split("->")
	else:
		lhs, rhs = packed, ""

	return lhs.split(","), rhs

@dataclass(frozen=True)
class EinsumKwargs:
	precision: lax.PrecisionLike
	result_type: DTypeLike | None

	def var_kwargs(self):
		"""
		Compute kwargs when rescaling, in which case we want to avoid down-casting until after rescaling
		"""
		return {
			"precision": self.precision,
			"preferred_element_type": accumulation_type(self.precision, self.result_type)
		}

	def ein_kwargs(self):
		return {
			"precision": self.precision,
			"preferred_element_type": self.result_type
		}

DEFAULT = EinsumKwargs(
	precision = None,
	result_type = None
)

def var_einsum(
	expr: str,
	*operands: jnp.ndarray,
	input_var: Optional[Sequence[float]] = None,
	einsum_kwargs: EinsumKwargs
) -> jnp.ndarray:
	operands = utils.promote_fp8(*operands)

	if input_var is None or all(v is None for v in input_var) or len(operands) == 0:
		return jnp.einsum(expr, *operands, **einsum_kwargs.ein_kwargs())

	assert len(operands) == len(input_var), "operand count and input var count don't match"
	assert not any(v is None for v in input_var), "All input variances must be specified"

	inputs, output = parse_einsum(expr)

	assert len(inputs) == len(operands), "operand count and ein expr don't match"

	shapes = {}
	for i, (input, shape) in enumerate(zip(inputs, map(jnp.shape, operands))):
		for c, n in zip(input, shape):
			if c in shapes and shapes[c] != n:
				raise ValueError(f"Shape mismatch for argument {i}, dimension {c}: {shapes[c]} != {n}")
			shapes[c] = n

	output_cs = Counter(output)
	outer_loops = reduce(or_, [Counter(input) - output_cs for input in inputs], Counter())

	scale = reduce(mul, input_var, 1)
	for c in outer_loops.elements():
		scale *= shapes[c]

	var_kwargs = einsum_kwargs.var_kwargs()
	summand = jnp.einsum(expr, *operands, **var_kwargs)

	scale_type = var_kwargs["preferred_element_type"]
	if scale_type is None:
		scale_type = summand.dtype
	scale = jnp.sqrt(scale).astype(scale_type)

	final_type = einsum_kwargs.result_type if einsum_kwargs.result_type else summand.dtype
	return (summand / scale).astype(final_type)