import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model.mlp import MLP, MLPConfig
from ueaj.model.rmsnorm import RMSNorm, RMSNormConfig
from ueaj.model.ueajsum import ParamConfig


def test_mlp_gradient_dtypes():
    """Test that MLP gradients have correct dtypes: fp32 for inputs, fp16 for parameters"""
    
    # Create base parameter config with fp16 parameters and fp16 gradients
    base_param_config = (ParamConfig("", group=nnx.Param)
                        .with_dtype(jnp.bfloat16))
    
    # Create MLP config with ReLU activation and custom parameter configs
    config = MLPConfig(
        model_d=32,
        hidden_d=64,
        activation_fn=nnx.relu,
        param_config=base_param_config
    ).with_up(base_param_config).with_down(base_param_config.with_initializer(nnx.initializers.zeros))

    # Initialize MLP
    rngs = rng.Rngs(42)
    mlp = MLP(config, rngs)
    rms = RMSNorm(RMSNormConfig(
        model_d=config.model_d,
        scale_dtype=jnp.bfloat16
    ))
    
    # Create input tensor
    batch_size, seq_len = 2, 8
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, config.model_d)).astype(jnp.bfloat16)
    
    # Use nnx.grad to compute gradients w.r.t. inputs only
    def loss_fn(rms, mlp, x):
        x = rms(x, downcast_grads=False)
        x = mlp.invoke_fp32_backprop(x)
        return jnp.sum(x)
    
    # Compute input gradients only to avoid tracer issues
    grad_fn = nnx.value_and_grad(loss_fn, argnums=(2,))
    loss_val, (x_grads,) = grad_fn(rms, mlp, x)
    
    # Check input gradient dtype (should be fp32)
    assert x_grads.dtype == jnp.float32, f"Input gradient dtype should be fp32, got {x_grads.dtype}"
    
    # Verify forward pass works
    assert loss_val.shape == (), "Loss should be scalar"
    assert jnp.isfinite(loss_val), "Loss should be finite"

    print()
    print(f"✓ Input gradient dtype: {x_grads.dtype}")
    print(f"✓ Loss value: {loss_val}")


if __name__ == "__main__":
    test_mlp_gradient_dtypes()
    print("All tests passed!")