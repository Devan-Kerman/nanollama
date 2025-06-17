from .config import *
from .varsum import *

def mixsum(
	operand_specs: Sequence[ArgumentConfig],
	output_spec: ArgumentConfig,
	*operands: jnp.ndarray
) -> jnp.ndarray:
	"""
	Einsum with custom per‑tensor gradient precision and variance tracking.
	"""
	expr = ",".join(f"{spec.shape}" for spec in operand_specs) + f"->{output_spec.shape}"

	operand_var = [spec.variance for spec in operand_specs]
	output_var = output_spec.variance

	assert all((spec.variance is None) == (output_var is None) for spec in operand_specs), "Output variance must be specified if any operand variance is specified"

	@jax.custom_vjp
	def _einsum(*ops):
		return var_einsum(
			expr,
			*ops,
			input_var=operand_var,
			einsum_kwargs=EinsumKwargs(
				precision=output_spec.precision,
				result_type=output_spec.dtype
			)
		)

	def _fwd(*ops):
		y = var_einsum(
			expr,
			*ops,
			input_var=operand_var,
			einsum_kwargs=EinsumKwargs(
				precision=output_spec.precision,
				result_type=output_spec.dtype
			)
		)
		return y, ops  # save primals

	def _bwd(ops, g):
		grads = []
		for i, (op_i, op_spec_i) in enumerate(zip(ops, operand_specs)):
			other_ops = [ops[j] for j in range(len(ops)) if j != i]
			other_subs = [op_spec_j.shape for j, op_spec_j in enumerate(operand_specs) if j != i]

			input_var = None
			if output_var is not None:
				input_var = [operand_var[j] for j in range(len(ops)) if j != i] + [output_var]

			# build grad expression:  (other_subs..., out_subs) -> in_subs[i]
			grad_expr = f"{','.join(other_subs + [output_spec.shape])}->{op_spec_i.shape}"

			grad_i = var_einsum(
				grad_expr,
				*other_ops,
				g,
				input_var=input_var,
				einsum_kwargs=EinsumKwargs(
					precision=op_spec_i.precision,
					result_type=op_spec_i.grad_dtype
				)
			)
			grads.append(grad_i)
		return tuple(grads)

	_einsum.defvjp(_fwd, _bwd)
	return _einsum(*operands)

