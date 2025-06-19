from test_memory import memory_report


def compile_mlp_bwd():
	import functools
	import jax
	import jax.numpy as jnp
	from flax import nnx
	from flax.nnx import rnglib as rng
	from ueaj.model.mlp import GMLP, MLPConfig
	from ueaj.model.rmsnorm import RMSNormConfig, RMSNorm
	from ueaj.model.ueajsum import ParamConfig
	print("Initialized JAX")

	mlp = nnx.eval_shape(
		lambda: GMLP(
			MLPConfig(
				model_d=2048,
				hidden_d=8192,
			),
			rngs=rng.Rngs(0)
		)
	)
	rms = nnx.eval_shape(lambda: RMSNorm(RMSNormConfig(model_d=2048).with_accum_dtype(jnp.bfloat16)))

	sample_x = jax.ShapeDtypeStruct(shape=(16, 4096, 2048), dtype=jnp.bfloat16)

	# @functools.partial(jax.grad, argnums=(2,))
	def mlp_diff(graph_defs, x, params):
		mlp_graph, rms_graph = graph_defs
		mlp_params, rms_params = params
		mlp = nnx.merge(mlp_graph, mlp_params)
		rms = nnx.merge(rms_graph, rms_params)
		x += mlp(rms(x))
		return x

	# return jnp.square(x-y).sum(dtype=jnp.bfloat16)

	# compile function and let memory report print memory consumption of pass
	mlp_graph, mlp_params = nnx.split(mlp)
	rms_graph, rms_params = nnx.split(rms)

	graph_defs = (mlp_graph, rms_graph)
	params = (mlp_params, rms_params)

	def function(graph_defs, params, x, y):
		output, callback = jax.vjp(functools.partial(mlp_diff, graph_defs, x), params)
		return callback(y - x)

	# Create sample inputs
	x = sample_x
	y = sample_x  # Same shape as x for the loss computation

	# AOT compile the function
	lowered = jax.jit(function).lower(graph_defs, params, x, y)
	print("Lowered")
	print(lowered.as_text())

	compiled = lowered.compile()
	print("Compiled")
	print(compiled.as_text())

	print("Other Analysis:")
	import pprint
	pprint.pprint(compiled.cost_analysis())
	mem_analysis = compiled.memory_analysis()
	print(mem_analysis)

	print("\n============= Memory Report ==============")
	print(f"Total memory usage: {mem_analysis.temp_size_in_bytes / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
	import shutil

	shutil.rmtree("/tmp/jax_cache", ignore_errors=True)
	# .5GB for the input
	# 2GB for hidden state checkpoint
	# .5GB for output state checkpoint
	# ~= 3.5GB total
	print("Compiling mlp bwd pass")
	with memory_report():
		compile_mlp_bwd()
