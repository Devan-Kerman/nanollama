import jax
import jax.numpy as jnp
from jax import lax, dtypes
from typing import Any


def _normalize_dtype(dtype: Any):
    """Convert various dtype formats to a JAX-compatible dtype"""
    return dtypes.canonicalize_dtype(dtype)


def astype_fwd_noop_bwd(x: jax.Array, dtype: Any) -> jax.Array:
    """
    Custom astype that preserves gradient dtype in backwards pass.
    
    Forward pass: converts x from its original dtype to target dtype
    Backward pass: keeps gradients in whatever dtype they come in (no recasting)
    
    This avoids issues where fp8->fp16 casting in forward pass would
    try to cast fp16 gradients back to fp8 in backward pass.
    """
    # Convert dtype outside the custom_vjp to avoid argument issues
    jax_dtype = _normalize_dtype(dtype)
    
    @jax.custom_vjp
    def _astype_preserve_grad(x_inner):
        return lax.convert_element_type(x_inner, jax_dtype)
    
    def _fwd(x_inner):
        # Forward: convert to target dtype, don't save anything (we won't need it)
        y = lax.convert_element_type(x_inner, jax_dtype)
        return y, None
    
    def _bwd(_, g):
        # Backward: return gradient as-is, don't cast back to original dtype
        return (g,)
    
    _astype_preserve_grad.defvjp(_fwd, _bwd)
    return _astype_preserve_grad(x)


def noop_fwd_astype_bwd(x: jax.Array, dtype: Any) -> jax.Array:
    """
    Custom astype that does nothing in forward pass but casts in backward pass.
    
    Forward pass: returns x unchanged (no dtype conversion)
    Backward pass: casts gradients to the specified dtype
    
    This is the opposite of astype_fwd_noop_bwd.
    """
    # Convert dtype outside the custom_vjp to avoid argument issues
    jax_dtype = _normalize_dtype(dtype)
    
    @jax.custom_vjp
    def _noop_fwd_astype_bwd(x_inner):
        return x_inner
    
    def _fwd(x_inner):
        # Forward: return as-is, save original dtype for backward
        return x_inner, None
    
    def _bwd(orig_dtype, g):
        # Backward: cast gradient to target dtype
        return (lax.convert_element_type(g, jax_dtype),)
    
    _noop_fwd_astype_bwd.defvjp(_fwd, _bwd)
    return _noop_fwd_astype_bwd(x)

def identity_grad(x: jax.Array, lambda_: float | jax.Array = .1):
    @jax.custom_vjp
    def _igrad(x: jax.Array):
        return x

    def _igrad_fwd(x: jax.Array):
        return x, x

    def _igrad_bwd(resid: jax.Array, grad: jax.Array):
        return (grad + lambda_ * resid,)

    _igrad.defvjp(_igrad_fwd, _igrad_bwd)
    return _igrad(x)


def debug_dtype(x: jax.Array, name: str = "tensor") -> jax.Array:
    """
    Debug version of astype that prints dtype information in forward and backward passes.
    
    Forward pass: converts x from its original dtype to target dtype and prints info
    Backward pass: keeps gradients in whatever dtype they come in and prints info
    
    Args:
        x: Input array
        dtype: Target dtype for forward pass
        name: Optional name for the tensor being tracked (for clearer debug output)
    
    Returns:
        Array with converted dtype
    """
    @jax.custom_vjp
    def _debug_astype(x_inner):
        return x_inner
    
    def _fwd(x):
        # Forward: convert to target dtype
        print(f"[FWD] {name}: dtype = {x.dtype}")
        return x, None
    
    def _bwd(_, g):
        # Backward: return gradient as-is and print debug info
        print(f"[BWD] {name}: dtype = {g.dtype}")
        return (g,)
    
    _debug_astype.defvjp(_fwd, _bwd)
    return _debug_astype(x)

def debug_grad_flow(x: jax.Array, name: str = "tensor") -> jax.Array:
    @jax.custom_vjp
    def _debug_astype(x_inner):
        return x_inner

    def _fwd(x):
        # Forward: convert to target dtype
        return x, None

    def _bwd(_, g):
        # Backward: return gradient as-is and print debug info
        jax.debug.print("[BWD] {}: {}", name, jnp.sqrt(jnp.square(g).mean()))
        return (g,)

    _debug_astype.defvjp(_fwd, _bwd)
    return _debug_astype(x)


def te_gradient_workaround(x: jax.Array) -> jax.Array:
    """
    Workaround for TransformerEngine CUDA illegal memory access error.
    
    This error occurs when normalized tensors are passed to TE's fused attention
    during JIT compilation. The custom VJP breaks the problematic gradient pattern
    while preserving correct gradients.
    """
    @jax.custom_vjp
    def _identity_with_grad(x):
        return x
    
    def _fwd(x):
        return x, x.sum()
    
    def _bwd(_, g):
        # Pass gradient through normally
        return (g,)
    
    _identity_with_grad.defvjp(_fwd, _bwd)
    return _identity_with_grad(x)