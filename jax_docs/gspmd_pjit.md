# JAX GSPMD, pjit, and Automatic Parallelization Documentation

## Overview

JAX provides powerful automatic parallelization capabilities through GSPMD (General and Scalable Parallelization for ML Computation Graphs) and the `jit` function (formerly `pjit`). This documentation covers distributed arrays, automatic parallelization, and the various sharding modes available in JAX.

## Key Concepts

### GSPMD (General and Scalable Parallelization for ML Computation Graphs)

GSPMD is the parallelization system that enables automatic parallelization in JAX, implemented as an extension to XLA (JAX's compiler). The key principle is that you only need to specify how you want the input and output of your code to be partitioned, and the compiler will:

1. Partition everything inside automatically
2. Compile inter-device communications
3. Optimize computations across multiple devices

Reference paper: [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)

### Important Note: pjit is now jit

**NOTE**: The `pjit` function is now equivalent to `jax.jit` - please use `jax.jit` instead. The pjit functionality has been merged into the regular `jax.jit`.

## Core Principle: Computation Follows Data

When you explicitly shard data with `jax.device_put` and apply functions to that data, the compiler attempts to parallelize the computation and decide the output sharding automatically.

## Three Modes of Parallel Computation

JAX supports three modes of parallel computation:

| Mode | View | Explicit Sharding | Explicit Collectives | Description |
|------|------|-------------------|---------------------|-------------|
| **Automatic** | Global | ❌ | ❌ | The compiler chooses the optimal computation strategy ("the compiler takes the wheel") |
| **Explicit** | Global | ✅ | ❌ | Sharding of each array is part of JAX-level type, compiler is constrained by user-supplied shardings |
| **Manual** | Per-device | ✅ | ✅ | Full manual control using `jax.shard_map()` with explicit communication collectives |

### 1. Automatic Sharding via `jax.jit()`

The compiler chooses the optimal computation strategy. This is the simplest mode where you let JAX handle all parallelization decisions.

```python
@jax.jit
def f_elementwise(x):
    return 2 * jnp.sin(x) + 1

# The computation is automatically distributed based on input sharding
result = f_elementwise(sharded_array)
```

### 2. Explicit Sharding

Similar to automatic sharding but with explicit control over array sharding:

- Sharding becomes part of the array's JAX-level type
- Shardings are propagated at the JAX level and queryable at trace time
- The compiler turns whole-array programs into per-device programs (e.g., turning `jnp.sum` into `psum`)
- The compiler is heavily constrained by user-supplied shardings

```python
mesh = jax.make_mesh((2, 4), ("X", "Y"))
sharded_array = jax.device_put(array, jax.NamedSharding(mesh, P("X", "Y")))
```

### 3. Fully Manual Sharding with `shard_map()`

- Enables per-device code
- Explicit communication collectives
- Full control over parallelization strategy

## How pjit/jit Works

To use `jit` for parallel computation, you need to provide:

1. **A mesh specification**: Maps logical devices on a 2D (or higher-D) mesh to physical devices
2. **Sharding specs**: For all input and output tensors
3. **Sharding constraints**: For select intermediate tensors (optional)

**Important**: JAX doesn't require manual insertion of collective operations. It uses a constraint-based model where you specify sharding constraints for memory-intensive tensors, and it automatically determines the sharding pattern.

## Key Components

### jax.Array

- Unified array object that can represent data across multiple devices
- Contains a `jax.sharding.Sharding` object describing data distribution
- Enables automatic parallelization based on data placement

### Sharding Types

1. **NamedSharding**: Allows specifying device mesh and partition specifications
2. **PositionalSharding**: Based on device positions
3. **Partial/Full replication**: Arrays can be partially or fully replicated across devices

### Example: Creating a Sharded Array

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# Create a device mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=('data',))

# Create a sharding specification
sharding = NamedSharding(mesh, P('data'))

# Create and shard an array
array = jnp.arange(1000)
sharded_array = jax.device_put(array, sharding)
```

## Automatic Parallelization Features

### Output Sharding Propagation

If not specified, `jax.jit()` uses GSPMD's sharding propagation to determine output sharding automatically.

### Supported Operations

- **Elementwise operations**: Automatically parallelized based on input sharding
- **Matrix multiplication**: Compiler determines optimal sharding strategy
- **Reductions**: Automatically inserts necessary collective operations
- **Complex neural networks**: Supports both data and model parallelism

### Example: Automatic Parallelization

```python
@jax.jit
def matrix_multiply(a, b):
    return jnp.dot(a, b)

# With sharded inputs, the computation is automatically parallelized
result = matrix_multiply(sharded_a, sharded_b)
```

## Neural Network Parallelization

JAX supports various parallelization strategies for neural networks:

### 1. Batch Data Parallelism

Shard data across the batch dimension:

```python
data_sharding = NamedSharding(mesh, P('data', None))
```

### 2. Model Tensor Parallelism

Shard model parameters across devices:

```python
model_sharding = NamedSharding(mesh, P(None, 'model'))
```

### 3. Automatic Gradient Computation

Gradients are automatically computed across sharded devices:

```python
@jax.jit
def loss_fn(params, batch):
    # Loss computation
    return loss

grad_fn = jax.grad(loss_fn)
grads = grad_fn(sharded_params, sharded_batch)
```

## Sharding Constraints

Use `jax.lax.with_sharding_constraint` to specify intermediate value shardings:

```python
@jax.jit
def f(x):
    y = heavy_computation(x)
    # Constrain y to specific sharding
    y = jax.lax.with_sharding_constraint(y, sharding_spec)
    return final_computation(y)
```

## Performance Benefits

- Significant speedup compared to single-device computation
- Automatic communication and computation splitting across devices
- Efficient memory usage through distributed storage
- Overlapping of computation and communication

## Sharp Bits and Caveats

### Random Number Generation

Random number generation requires special handling in distributed settings:

```python
# Use jax_threefry_partitionable flag for partitionable random values
jax.config.update('jax_threefry_partitionable', True)
```

### Memory Considerations

- Be aware of memory usage patterns when sharding large arrays
- Consider replication vs. sharding trade-offs
- Monitor device memory usage

### Debugging

- Use `jax.debug.visualize_array_sharding()` to inspect sharding
- Check intermediate shardings with constraints
- Monitor communication patterns

## Best Practices

1. **Start with automatic sharding**: Let the compiler make decisions initially
2. **Profile and optimize**: Add explicit constraints where needed
3. **Consider communication costs**: Minimize cross-device communication
4. **Use appropriate mesh shapes**: Match your hardware topology
5. **Test at small scale**: Verify correctness before scaling up

## Example: Complete Parallel Training Loop

```python
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

# Setup device mesh
devices = jax.devices()
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define shardings
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))

@jax.jit
def train_step(params, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['images'])
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['labels']
            )
        )
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = optimizer.update(grads, params)
    return params, loss

# Shard data and parameters
params = jax.device_put(params, model_sharding)
batch = jax.device_put(batch, data_sharding)

# Training automatically parallelized
params, loss = train_step(params, batch)
```

## Migration from pmap

If migrating from `pmap` to `jit` with sharding:

1. Replace `pmap` with `jit`
2. Define appropriate mesh and shardings
3. Remove explicit collective operations (they're automatic now)
4. Update batch dimension handling

## Additional Resources

- [Introduction to parallel programming in JAX](https://docs.jax.dev/en/latest/sharded-computation.html)
- [Distributed arrays and automatic parallelization tutorial](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [jax.experimental.pjit module documentation](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html)
- [GSPMD Paper](https://arxiv.org/abs/2105.04663)

## Sources

[Source 1](https://docs.jax.dev/en/latest/sharded-computation.html)
[Source 2](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)