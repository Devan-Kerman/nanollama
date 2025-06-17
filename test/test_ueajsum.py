import pytest

from ueaj.model.ueajsum import *

def get_rngs():
	return nnx.Rngs(0)


def test_simple_matmul():
	rngs = get_rngs()
	config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(
			ArgumentConfig(shape="ij"),
			ArgumentConfig(shape="jk"),
		),
		kwarg_configs=FrozenDict(),
		result_config=ArgumentConfig(shape="ik"),
		sums=((0, 1),)
	)

	model = Ueajsum(config, {}, rngs=rngs)

	a = jnp.ones((2, 3))
	b = jnp.ones((3, 4))

	result = model(a, b)
	expected = jnp.einsum("ij,jk->ik", a, b)

	assert jnp.allclose(result, expected)
	assert result.shape == (2, 4)


def test_complex_summation():
	rngs = get_rngs()
	config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(),
		kwarg_configs=FrozenDict(
			{
				'x': ArgumentConfig(shape="ab"),
				'y': ArgumentConfig(shape="bc"),
				'z': ArgumentConfig(shape="ac"),
			}
		),
		result_config=ArgumentConfig(shape="ac"),
		sums=(('x', 'y'), ('z',))
	)
	model = Ueajsum(config, {}, rngs=rngs)

	x = jnp.ones((2, 3))
	y = jnp.ones((3, 4))
	z = jnp.full((2, 4), 2.0)

	result = model(config, False, x=x, y=y, z=z)
	expected = jnp.einsum("ab,bc->ac", x, y) + z

	assert jnp.allclose(result, expected)
	assert result.shape == (2, 4)


def test_with_parameters():
	rngs = get_rngs()

	param_config = ParamConfig(
		shape="jk",
	).with_dtype(jnp.float32).with_initializer(nnx.initializers.ones)

	config = UeajsumConfig(
		no_instantiation=False,
		arg_configs=(
			ArgumentConfig(shape="ij"),
		),
		kwarg_configs=FrozenDict(
			{
				'w': param_config
			}
		),
		result_config=ArgumentConfig(shape="ik"),
		sums=((0, 'w'),)
	)

	model = Ueajsum(config, {'j': 10, 'k': 20}, rngs=rngs)

	assert hasattr(model, 'w')
	assert model.w.shape == (10, 20)

	a = jnp.ones((30, 10))

	result = model(a)
	expected = jnp.einsum("ij,jk->ik", a, jnp.ones(model.w.shape))

	assert jnp.allclose(result, expected)


def test_override_logic():
	rngs = get_rngs()
	config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(
			ArgumentConfig(shape="ij").with_dtype(jnp.float32),
		),
		kwarg_configs=FrozenDict(),
		result_config=ArgumentConfig(shape="ij"),
		sums=((0,),)
	)
	model = Ueajsum(config, {}, rngs=rngs)

	a = jnp.ones((2, 3), dtype=jnp.float16)

	override_config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(
			ArgumentConfig(shape="ij").with_dtype(jnp.float16),
		),
		kwarg_configs=FrozenDict(),
		result_config=ArgumentConfig(shape="ij").with_shape(jnp.float16),
		sums=((0,),)
	)

	args_dict_no_override = model._build_args(config, False, [a], {})
	assert args_dict_no_override[0][1].dtype == jnp.float32

	args_dict_with_override = model._build_args(override_config, True, [a], {})
	assert args_dict_with_override[0][1].dtype == jnp.float16


def test_gradient():
	rngs = get_rngs()
	config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(),
		kwarg_configs=FrozenDict(
			{
				'x': ArgumentConfig(shape="ij", variance=None).with_grad_dtype(jnp.float32),
				'y': ArgumentConfig(shape="jk", variance=None).with_grad_dtype(jnp.float32),
			}
		),
		result_config=ArgumentConfig(shape="ik", variance=None),
		sums=(('x', 'y'),)
	)
	model = Ueajsum(config, {}, rngs=rngs)

	x = jnp.ones((2, 3))
	y = jnp.ones((3, 4))

	def loss_fn(params):
		res = model(x=params['x'], y=params['y'])
		return jnp.sum(res)

	grad_fn = jax.grad(loss_fn)
	grads = grad_fn({'x': x, 'y': y})

	expected_grad_x = jnp.ones_like(x) * y.shape[1]
	expected_grad_y = jnp.ones_like(y) * x.shape[0]

	assert jnp.allclose(grads['x'], expected_grad_x)
	assert jnp.allclose(grads['y'], expected_grad_y)
	assert grads['x'].dtype == jnp.float32
	assert grads['y'].dtype == jnp.float32


def test_missing_argument():
	rngs = get_rngs()
	config = UeajsumConfig(
		no_instantiation=True,
		arg_configs=(
			ArgumentConfig(shape="ij"),
			ArgumentConfig(shape="jk"),
		),
		kwarg_configs=FrozenDict(),
		result_config=ArgumentConfig(shape="ik"),
		sums=((0, 1),)
	)
	model = Ueajsum(config, {}, rngs=rngs)
	a = jnp.ones((2, 3))

	with pytest.raises(ValueError, match="Missing argument 1"):
		model(a)


def test_parsing():
	print()
	config = parse("ij,*jk->ik").fp8_params().group_map(lambda c: c.with_initializer(nnx.initializers.ones), nnx.Param)

	assert config.arg_configs[0].shape == "ij"
	assert config.arg_configs[1].shape == "jk"
	assert config.result_config.shape == "ik"

	model = Ueajsum(config, {'j': 10, 'k': 20}, rngs=get_rngs())

	a = jax.random.normal(jax.random.PRNGKey(0), (30, 10)).astype(jnp.float8_e4m3fn)
	b = jnp.ones((10, 20), dtype=jnp.float8_e4m3fn)

	result = model(a)

	expected = jnp.einsum("ij,jk->ik", a, b)

	assert jnp.allclose(result, expected)

def test_mup():
	config = parse("ij,*jk->ik").unit().fp8_params().map_result(lambda c: c.with_dtype(jnp.float32))

	assert config.arg_configs[0].shape == "ij"
	assert config.arg_configs[1].shape == "jk"
	assert config.result_config.shape == "ik"

	model = Ueajsum(config, {'j': 100, 'k': 200}, rngs=get_rngs())

	a = jax.random.normal(jax.random.PRNGKey(0), (300, 100)).astype(jnp.float8_e4m3fn)

	result = model(a)

	assert abs(result.std() - 1.0) < .1


def test_everything():
	model = Ueajsum(
		parse("ij,*w=jk + *b=k -> ik")
			.fp8_params()
			.group_map(dtype(jnp.float8_e4m3fn), nnx.LoRAParam)
			.unit()
			.fp32_grads()
			.map_arg('b', lambda c: c.with_initializer(nnx.initializers.zeros)),
		{'j': 100, 'k': 200, 'u': 64},
		rngs=get_rngs()
	)

	print(model)
	state = nnx.state(model)
	for k, v in state.items():
		tensor = v.value
		print(f"{k}: {float(tensor.mean(dtype=jnp.float32)):.2f}, {float(tensor.std(dtype=jnp.float32)):.2f} {tensor.dtype}")

	a = jax.random.normal(jax.random.PRNGKey(0), (300, 100)).astype(jnp.float8_e4m3fn)

	result = model(a)

	print(result)

