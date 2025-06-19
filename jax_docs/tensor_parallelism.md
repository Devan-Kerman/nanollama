# JAX Tensor Parallelism, SPMD, and Sharding Documentation

This document compiles comprehensive documentation about tensor parallelism, SPMD (Single Program Multiple Data), and sharding in JAX, gathered from official JAX documentation and tutorials.

## Table of Contents
1. [Introduction to Parallel Programming](#introduction-to-parallel-programming)
2. [Distributed Arrays and Automatic Parallelization](#distributed-arrays-and-automatic-parallelization)
3. [Core Concepts](#core-concepts)
4. [Sharding and NamedSharding](#sharding-and-namedsharding)
5. [Automatic Parallelization with jax.jit](#automatic-parallelization-with-jaxjit)
6. [Manual Parallelism with shard_map](#manual-parallelism-with-shard_map)
7. [pjit and Modern JAX](#pjit-and-modern-jax)
8. [Practical Examples](#practical-examples)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices and Tips](#best-practices-and-tips)

## Introduction to Parallel Programming

JAX provides three modes of parallel computation for Single-Program Multi-Data (SPMD) programming:

1. **Automatic Parallelism via `jax.jit()`**
   - Compiler chooses optimal computation strategy
   - Write global-view programs
   - Compiler handles data partitioning and communication

2. **Explicit Sharding**
   - Sharding becomes part of the array's JAX-level type
   - Shardings are propagated and queryable during tracing
   - Compiler constrained by user-supplied shardings

3. **Manual Parallelism with `jax.shard_map()`**
   - Programmer writes per-device code
   - Enables explicit communication collectives
   - Requires manual specification of shard sizes and assembly

## Distributed Arrays and Automatic Parallelization

JAX introduces `jax.Array` for representing arrays across multiple devices. This unified datatype can represent arrays with physical storage spanning multiple devices, making parallelism a core feature of JAX.

### Key Features:
- Computation follows data sharding
- Compiler decides intermediate and output shardings
- Automatically inserts communication operations when needed

### Basic Example:
```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Create a mesh with 8 devices arranged in a 4x2 grid
mesh = jax.make_mesh((4, 2), ('a', 'b'))

# Create a sharded array
x = jnp.arange(32).reshape(8, 4)
sharded_x = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))

# Computation is automatically parallelized
y = jnp.sin(sharded_x)  # Runs in parallel across devices
```

## Core Concepts

### SPMD (Single Program Multiple Data)
SPMD refers to a parallel computation technique where:
- The same computation (e.g., forward pass of a neural net) is run on different input data
- Different inputs run in parallel on different devices (e.g., TPUs, GPUs)
- Originally implemented via `jax.pmap()`, now integrated into `jax.jit()`

### jax.Array
- Immutable array data structure
- Represents arrays with physical storage spanning one or multiple devices
- Has an associated `jax.sharding.Sharding` object
- Makes parallelism a core feature of JAX

### Computation Follows Data
The XLA compiler includes heuristics for optimizing computations across multiple devices. In the simplest cases, these heuristics boil down to "computation follows data":
- Based on input data sharding, compiler decides shardings for intermediates and outputs
- Parallelizes evaluation automatically
- Inserts communication operations as necessary

## Sharding and NamedSharding

### NamedSharding
A NamedSharding is a pair of:
1. A Mesh of devices
2. A PartitionSpec describing how to shard an array across that mesh

```python
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Create a mesh
devices = jax.devices()
mesh = Mesh(devices, ('x',))

# Create a PartitionSpec
spec = PartitionSpec('x')  # Shard along the x-axis of the mesh

# Create a NamedSharding
sharding = NamedSharding(mesh, spec)
```

### Mesh
A Mesh is a multidimensional NumPy array of JAX devices where:
- Each axis has a name (e.g., 'x', 'y')
- Can represent logical organization of physical devices

```python
# 2D mesh with named axes
mesh = jax.make_mesh((4, 2), ('data', 'model'))
```

### PartitionSpec
Describes how input dimensions are partitioned across mesh dimensions:
- `None`: dimension is not partitioned
- Axis name: dimension is sharded across that mesh axis
- Tuple of axes: dimension is sharded across multiple axes

```python
# Examples:
P('x', 'y')      # First dim on x-axis, second on y-axis
P('x', None)     # First dim on x-axis, second replicated
P(('x', 'y'),)   # First dim sharded across both x and y
```

## Automatic Parallelization with jax.jit

Modern JAX achieves parallel computation by passing sharded data to `jax.jit()`-compiled functions:

```python
@jax.jit
def parallel_fn(x, w):
    return jnp.dot(x, w)

# With sharded inputs, computation is automatically parallelized
with mesh:
    x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
    w_sharded = jax.device_put(w, NamedSharding(mesh, P(None, 'model')))
    
    # This runs in parallel across devices
    result = parallel_fn(x_sharded, w_sharded)
```

### Sharding Constraints
Use `jax.lax.with_sharding_constraint()` to specify intermediate shardings:

```python
@jax.jit
def constrained_fn(x):
    # Specify that intermediate should be sharded a certain way
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('x', None)))
    return x @ x.T
```

## Manual Parallelism with shard_map

`shard_map` provides explicit control over parallelization:

### Key Features:
- Write per-device code explicitly
- Use collective communication primitives
- More control than automatic parallelization

### Basic Example:
```python
from functools import partial
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, 
         in_specs=(P('i', 'j'), P('j', 'k')),
         out_specs=P('i', 'k'))
def manual_matmul(a_block, b_block):
    # Each device computes its local block
    c_partial = jnp.dot(a_block, b_block)
    
    # Use collective to sum across devices
    c_block = jax.lax.psum(c_partial, 'j')
    return c_block
```

### Collective Operations
Available collectives include:
- `psum`: Parallel sum across devices
- `all_gather`: Gather values from all devices
- `psum_scatter`: Sum and scatter results
- `all_to_all`: All-to-all communication

## pjit and Modern JAX

### Historical Context
`pjit` (partitioned jit) was originally a separate function for multi-device parallelism. As of recent JAX versions:
- `pjit` is now equivalent to `jax.jit`
- Use `jax.jit` with `in_shardings` and `out_shardings` parameters

### Modern Usage:
```python
# Old pjit style (deprecated)
from jax.experimental.pjit import pjit
f = pjit(fn, in_shardings=..., out_shardings=...)

# Modern jax.jit style (preferred)
f = jax.jit(fn, in_shardings=..., out_shardings=...)
```

### Key Capabilities:
- Automatic partitioning across devices
- SPMD execution model
- Works on CPU, GPU, and TPU
- Supports multi-process platforms (TPU pods)

## Practical Examples

### Example 1: Data Parallel Training
```python
# Shard batch dimension across devices
def data_parallel_loss(params, batch):
    # Batch is sharded across devices
    predictions = model.apply(params, batch['inputs'])
    return jnp.mean((predictions - batch['targets'])**2)

with mesh:
    # Shard data across 'data' axis
    batch = jax.device_put(batch, NamedSharding(mesh, P('data', None)))
    
    # Replicate parameters
    params = jax.device_put(params, NamedSharding(mesh, P(None)))
    
    # Compute loss in parallel
    loss = jax.jit(data_parallel_loss)(params, batch)
```

### Example 2: Model Parallel Linear Layer
```python
@jax.jit
def model_parallel_linear(x, w):
    # x: [batch, features]
    # w: [features, outputs] sharded on outputs dimension
    with mesh:
        # Shard weights across model dimension
        w = jax.device_put(w, NamedSharding(mesh, P(None, 'model')))
        
        # Computation automatically parallelized
        return jnp.dot(x, w)
```

### Example 3: 2D Parallel Matrix Multiplication
```python
# Shard both matrices optimally for parallel matmul
def parallel_matmul_2d(a, b):
    with Mesh(jax.devices(), ('x', 'y')):
        # a: [M, K] sharded as [x, y]
        # b: [K, N] sharded as [y, None]
        a_sharded = jax.device_put(a, NamedSharding(mesh, P('x', 'y')))
        b_sharded = jax.device_put(b, NamedSharding(mesh, P('y', None)))
        
        return jnp.dot(a_sharded, b_sharded)
```

## Advanced Topics

### Sharp Bits and Caveats

#### Random Number Generation
Random number generation with sharding requires special consideration:
```python
# Use threefry_partitionable flag for better RNG behavior
jax.config.update('jax_threefry_partitionable', True)

@jax.jit
def parallel_random(key, shape):
    # RNG automatically handles sharding
    return jax.random.normal(key, shape)
```

#### Nested Parallelism
Combining different parallelism strategies:
```python
# Combine data and model parallelism
with Mesh(devices.reshape(dp_size, mp_size), ('data', 'model')):
    # Can now shard along both dimensions
    x = jax.device_put(x, NamedSharding(mesh, P('data', None)))
    w = jax.device_put(w, NamedSharding(mesh, P(None, 'model')))
```

### Performance Optimization

#### Overlapping Computation and Communication
Use `shard_map` for fine-grained control:
```python
@partial(shard_map, mesh=mesh, ...)
def optimized_fn(x):
    # Start async communication
    future = jax.lax.all_gather_async(x, 'x')
    
    # Do local computation while communicating
    local_result = expensive_local_op(x)
    
    # Wait for communication
    gathered = future.wait()
    
    return combine(local_result, gathered)
```

#### Choosing Sharding Strategies
Consider:
- Memory usage per device
- Communication patterns
- Computation-to-communication ratio
- Hardware topology

### Integration with Flax

Flax provides utilities for sharding model parameters:
```python
from flax import linen as nn
from flax.training import train_state

class ParallelModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Specify parameter sharding
        kernel_init = nn.with_partitioning(
            nn.initializers.lecun_normal(),
            ('data', 'model')
        )
        
        x = nn.Dense(features=1024, 
                     kernel_init=kernel_init)(x)
        return x
```

## Best Practices and Tips

### 1. Start Simple
Begin with automatic parallelization via `jax.jit` before moving to manual control with `shard_map`.

### 2. Profile and Monitor
```python
# Check sharding of arrays
jax.debug.visualize_array_sharding(array)
jax.debug.inspect_array_sharding(array)
```

### 3. Use Sharding Constraints Wisely
Only add constraints where necessary - let the compiler optimize when possible.

### 4. Consider Memory Constraints
Ensure each device has sufficient memory for its shard:
```python
shard_size = total_size // num_devices
print(f"Each device needs {shard_size} elements")
```

### 5. Test Scaling
Verify correctness with different device counts:
```python
# Test with different mesh shapes
for devices in [1, 2, 4, 8]:
    mesh = Mesh(jax.devices()[:devices], ('x',))
    # Run tests...
```

### 6. Hardware Topology Awareness
Consider physical device connectivity:
```python
# Create mesh respecting hardware topology
mesh = jax.make_mesh((local_devices, hosts), ('local', 'global'))
```

## Debugging and Troubleshooting

### Visualizing Shardings
```python
# Visualize how an array is sharded
from jax.debug import visualize_array_sharding
visualize_array_sharding(sharded_array)
```

### Common Issues:

1. **Incompatible Shardings**
   - Ensure operations have compatible input shardings
   - Use `with_sharding_constraint` to reshape shardings

2. **Memory Errors**
   - Check per-device memory usage
   - Consider different sharding strategies

3. **Communication Overhead**
   - Profile to identify communication bottlenecks
   - Optimize sharding to minimize cross-device transfers

## Conclusion

JAX's tensor parallelism and SPMD capabilities provide flexible and powerful tools for distributed computation. The key is understanding:
- How data sharding drives parallel computation
- When to use automatic vs. manual parallelization
- How to optimize for specific hardware configurations

Start with automatic parallelization for simplicity, and move to manual control with `shard_map` when you need fine-grained optimization.

## Sources

[source](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
[source](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)
[source](https://docs.jax.dev/en/latest/sharded-computation.html)
[source](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
[source](https://jax.readthedocs.io/en/latest/jax.sharding.html)
[source](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html)