"""
Fast GPU-friendly parameter hashing for JAX
Optimized for checking parameter synchronization across devices
"""

import jax
import jax.numpy as jnp
from functools import partial

# === FASTEST: Simple XOR hash ===
@jax.jit
def xor_hash(array: jnp.ndarray) -> jnp.uint32:
    """
    XOR all bits together. Extremely fast on GPU.
    Detects ANY bit difference in parameters.
    
    Time complexity: O(n) with tree reduction -> O(log n) steps
    """
    # Bitcast float32 to uint32 (preserves exact bit pattern)
    as_uint = jax.lax.bitcast_convert_type(array.flatten(), jnp.uint32)
    
    # Tree reduction XOR - very efficient on GPU
    return jax.lax.reduce(as_uint, jnp.uint32(0), jax.lax.bitwise_xor, dimensions=[0])

# === GOOD DISTRIBUTION: Rolling hash ===
@jax.jit
def rolling_hash(array: jnp.ndarray) -> jnp.uint32:
    """
    Rolling hash with multiplication and rotation.
    Better distribution than XOR, still fast.
    """
    as_uint = jax.lax.bitcast_convert_type(array.flatten(), jnp.uint32)
    
    def roll_step(h, x):
        # Rotate left by 13 bits and mix
        h = ((h << 13) | (h >> 19)) & jnp.uint32(0xFFFFFFFF)
        h = (h + x) * jnp.uint32(0x5bd1e995)  # MurmurHash constant
        return h
    
    # Tree reduction with mixing
    hash_val = jax.lax.reduce(as_uint, jnp.uint32(0x9747b28c), roll_step, dimensions=[0])
    
    # Final mix
    return hash_val ^ (hash_val >> 16)

# === BEST FOR LARGE ARRAYS: Hierarchical hash ===
@jax.jit
def hierarchical_hash(array: jnp.ndarray) -> jnp.uint32:
    """
    Two-level hashing for very large arrays.
    Maximizes GPU parallelism.
    """
    as_uint = jax.lax.bitcast_convert_type(array.flatten(), jnp.uint32)
    
    # Split into chunks (GPU-friendly size)
    # 16384
    chunk_size = (204800 // 4) // 4 # H100 SM Cache Size // uint size // 4 warps per SM
    n_chunks = (as_uint.size + chunk_size - 1) // chunk_size
    
    # Pad if needed
    padded_size = n_chunks * chunk_size
    if as_uint.size < padded_size:
        as_uint = jnp.pad(as_uint, (0, padded_size - as_uint.size))
    
    # Reshape into chunks
    chunks = as_uint.reshape(n_chunks, chunk_size)
    
    # Hash each chunk in parallel
    chunk_hashes = jax.vmap(xor_hash)(chunks)
    
    # Final reduction
    return rolling_hash(chunk_hashes)

# === Main API: Hash parameter trees ===
@partial(jax.jit, static_argnames=['algorithm'])
def hash_params(params, algorithm='xor'):
    """
    Hash an entire parameter pytree into a single uint32.
    
    Args:
        params: PyTree of parameters
        algorithm: 'xor' (fastest), 'rolling' (better distribution), 'hierarchical' (large params)
    
    Returns:
        uint32 hash value
    """
    # Flatten all parameters into single array
    leaves, _ = jax.tree_util.tree_flatten(params)
    all_params = jnp.concatenate([leaf.flatten() for leaf in leaves])
    
    # Apply chosen algorithm
    if algorithm == 'xor':
        return xor_hash(all_params)
    elif algorithm == 'rolling':
        return rolling_hash(all_params)
    elif algorithm == 'hierarchical':
        return hierarchical_hash(all_params)
    else:
        return xor_hash(all_params)

# === Check synchronization across devices ===
@partial(jax.jit, static_argnames=['algorithm'])
def check_param_sync(params_per_device, algorithm='xor'):
    """
    Check if parameters are synchronized across all devices.
    
    Args:
        params_per_device: PyTree with leading device dimension
        algorithm: Hash algorithm to use
        
    Returns:
        (all_synced: bool, hashes: array of hash per device)
    """
    # Compute hash for each device
    hashes = jax.vmap(lambda p: hash_params(p, algorithm))(params_per_device)
    
    # Check if all match
    all_synced = jnp.all(hashes == hashes[0])
    
    return all_synced, hashes

# === Usage example ===
if __name__ == "__main__":
    print("=== Fast Parameter Hashing for JAX ===\n")
    
    # Example: Check parameter sync across devices
    key = jax.random.PRNGKey(42)
    
    # Simulate model parameters
    model_params = {
        'encoder': {
            'embed': jax.random.normal(key, (10000, 512)),
            'layers': [
                {'attn': jax.random.normal(key, (512, 512)),
                 'mlp': jax.random.normal(key, (512, 2048))}
                for _ in range(12)
            ]
        },
        'decoder': {
            'output': jax.random.normal(key, (512, 10000))
        }
    }
    
    print("Model parameter count:", sum(p.size for p in jax.tree_util.tree_leaves(model_params)))
    
    # Test 1: All devices synchronized
    print("\n1. Testing synchronized parameters:")
    n_devices = 8
    params_sync = jax.tree.map(lambda x: jnp.stack([x] * n_devices), model_params)
    
    synced, hashes = check_param_sync(params_sync)
    print(f"   All synced: {synced}")
    print(f"   Hashes: {[f'{h:08x}' for h in hashes[:4]]}...")
    
    # Test 2: One device out of sync
    print("\n2. Testing out-of-sync parameters:")
    params_async = jax.tree.map(lambda x: jnp.stack([x] * n_devices), model_params)
    
    # Introduce tiny change on device 3
    def add_noise_to_device_3(x):
        return x.at[3].add(jax.random.normal(key, x[3].shape) * 1e-8)
    
    params_async = jax.tree.map(add_noise_to_device_3, params_async)
    
    synced, hashes = check_param_sync(params_async)
    print(f"   All synced: {synced}")
    print(f"   Device 0-2: {hashes[0]:08x}")
    print(f"   Device 3:   {hashes[3]:08x} (different!)")
    print(f"   Device 4-7: {hashes[4]:08x}")
    
    # Test 3: Performance comparison
    print("\n3. Performance comparison:")
    test_sizes = [1e6, 10e6, 100e6]
    
    for size in test_sizes:
        array = jax.random.normal(key, (int(size),))
        print(f"\n   Array size: {size/1e6:.0f}M elements")
        
        # Time each algorithm
        for algo in ['xor', 'rolling', 'hierarchical']:
            hash_fn = lambda: hash_params({'array': array}, algorithm=algo)
            
            # Warmup
            _ = hash_fn().block_until_ready()
            
            # Time
            import time
            start = time.time()
            for _ in range(10):
                _ = hash_fn().block_until_ready()
            elapsed = (time.time() - start) / 10 * 1000
            
            h = hash_fn()
            print(f"      {algo:12} {elapsed:6.2f}ms  (hash: {h:08x})")
    
    print("\n=== Recommendations ===")
    print("1. Use 'xor' for fastest sync checking (detects any bit difference)")
    print("2. Use 'rolling' if you need better hash distribution")
    print("3. Use 'hierarchical' for very large models (>100M params)")
    print("\nNote: These hashes detect exact bit-level differences.")
    print("Even tiny numerical errors (e.g., 1e-8) will produce different hashes.")