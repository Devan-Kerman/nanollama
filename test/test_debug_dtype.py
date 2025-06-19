"""
Test script to demonstrate the debug_dtype_astype function.
"""
import jax
import jax.numpy as jnp
from ueaj.utils.gradutils import debug_dtype


def test_debug_dtype():
    """Test the debug dtype function with various dtype conversions."""
    print("Testing debug_dtype_astype function\n")
    
    # Create test data in different dtypes
    x_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    x_f16 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float16)
    
    # Define a simple function that uses the debug dtype conversion
    def forward_fn(x):
        # Convert float32 to float16 in forward pass
        x_f16 = debug_dtype(x, jnp.float16, name="input_to_f16")
        
        # Do some computation
        y = x_f16 * 2.0
        
        # Convert back to float32
        y_f32 = debug_dtype(y, jnp.float32, name="output_to_f32")
        
        return jnp.sum(y_f32)
    
    print("=== Forward Pass ===")
    # Compute forward pass
    result = forward_fn(x_f32)
    print(f"Forward result: {result}\n")
    
    print("=== Backward Pass ===")
    # Compute gradients
    grad_fn = jax.grad(forward_fn)
    grads = grad_fn(x_f32)
    print(f"Gradients: {grads}")
    print(f"Gradient dtype: {grads.dtype}\n")
    
    # Test with mixed precision scenario
    print("=== Mixed Precision Test ===")
    
    @jax.custom_vjp
    def mixed_precision_op(x):
        # Simulate a mixed precision operation
        x_bf16 = debug_dtype(x, jnp.bfloat16, name="mixed_prec_input")
        return x_bf16 ** 2
    
    def mixed_fwd(x):
        y = mixed_precision_op.fun(x)
        return y, x
    
    def mixed_bwd(x, g):
        # Gradient comes in as float32 but input was bfloat16
        print(f"[Mixed Precision BWD] Received gradient dtype: {g.dtype}")
        # Compute gradient: d/dx(x^2) = 2x
        grad = 2 * x * g
        return (grad,)
    
    mixed_precision_op.defvjp(mixed_fwd, mixed_bwd)
    
    # Test mixed precision
    x_test = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    
    def test_mixed(x):
        y = mixed_precision_op(x)
        return jnp.sum(y)
    
    print("Forward pass:")
    result = test_mixed(x_test)
    print(f"Result: {result}\n")
    
    print("Backward pass:")
    grad_mixed = jax.grad(test_mixed)(x_test)
    print(f"Final gradient: {grad_mixed}")
    print(f"Final gradient dtype: {grad_mixed.dtype}")


if __name__ == "__main__":
    test_debug_dtype()