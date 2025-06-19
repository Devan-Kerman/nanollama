# JAX Collective Operations and Communication Primitives

JAX provides a comprehensive set of collective operations and communication primitives for parallel and distributed computing. These operations are primarily used with `jax.pmap()` (parallel map) and `jax.experimental.shard_map` for implementing SPMD (Single Program Multiple Data) parallelism across multiple devices (GPUs/TPUs).

## Core Collective Operations

### 1. psum (Parallel Sum)

Computes an all-reduce sum on `x` over the pmapped axis `axis_name`.

```python
jax.lax.psum(x, axis_name, *, axis_index_groups=None)
```

**Parameters:**
- `x`: array(s) with a mapped axis named `axis_name`
- `axis_name`: hashable Python object used to name a pmapped axis
- `axis_index_groups`: optional list of lists containing axis indices for partial reductions

**Returns:**
- Array(s) with the same shape as `x` containing the sum across all replicas

**Example:**
```python
# With 4 XLA devices
x = jnp.arange(4)
result = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
# Returns: [6 6 6 6]  (sum of [0, 1, 2, 3])
```

### 2. pmean (Parallel Mean)

Computes an all-reduce mean over the pmapped axis.

```python
jax.lax.pmean(x, axis_name, *, axis_index_groups=None)
```

**Parameters:**
- Same as `psum`

**Returns:**
- Array(s) with the same shape as `x` containing the mean across all replicas

**Example:**
```python
# With 4 XLA devices
x = jnp.arange(4)
result = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
# Returns: [1.5 1.5 1.5 1.5]  (mean of [0, 1, 2, 3])
```

### 3. pmax (Parallel Maximum)

Computes an all-reduce max over the pmapped axis.

```python
jax.lax.pmax(x, axis_name, *, axis_index_groups=None)
```

**Parameters:**
- Same as `psum`

**Returns:**
- Array(s) with the same shape as `x` containing the maximum across all replicas

**Notes:**
- Supports `axis_index_groups` for partial reductions
- On TPUs, all groups must be the same size
- Groups must cover all indices exactly once

### 4. pmin (Parallel Minimum)

Computes an all-reduce min along the axis.

```python
jax.lax.pmin(x, axis_name, *, axis_index_groups=None)
```

**Parameters:**
- Same as `psum`

**Returns:**
- Array(s) with the same shape as `x` containing the minimum across all replicas

### 5. ppermute (Parallel Permute)

Performs a collective permutation of data across devices.

```python
jax.lax.ppermute(x, axis_name, perm)
```

**Parameters:**
- `x`: array(s) with a mapped axis named `axis_name`
- `axis_name`: hashable Python object used to name a pmapped axis
- `perm`: list of (source_index, destination_index) pairs encoding the permutation

**Returns:**
- Permuted array(s) according to the specified permutation

**Notes:**
- No two pairs should have the same source or destination index
- Indices not corresponding to a destination are filled with zeros

**Example:**
```python
# Rotate data among 4 devices
perm = [(0, 1), (1, 2), (2, 3), (3, 0)]
result = jax.pmap(lambda x: jax.lax.ppermute(x, 'i', perm), axis_name='i')(x)
```

### 6. all_gather

Gathers values from all devices along a new axis.

```python
jax.lax.all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False)
```

**Parameters:**
- `x`: array(s) with a mapped axis named `axis_name`
- `axis_name`: hashable Python object used to name a pmapped axis
- `axis_index_groups`: optional groups for partial gather
- `axis`: axis along which to gather (default: 0)
- `tiled`: whether to use a tiled layout (default: False)

**Returns:**
- Array with gathered values from all devices along the specified axis

**Notes:**
- The output has an additional axis of size equal to the number of devices
- `all_gather` has a `pbroadcast` fused into it for efficiency

### 7. psum_scatter

Like `psum` but each device retains only part of the result.

```python
jax.lax.psum_scatter(x, axis_name, *, scatter_dimension=0, tiled=False)
```

**Parameters:**
- `x`: array(s) with a mapped axis named `axis_name`
- `axis_name`: hashable Python object used to name a pmapped axis
- `scatter_dimension`: dimension along which to scatter (default: 0)
- `tiled`: whether to use a tiled layout (default: False)

**Returns:**
- Partial sum result for each device

**Notes:**
- More efficient than `psum(x, axis_name)[axis_index(axis_name)]`
- Often used in combination with `all_gather` for efficient `psum` implementation

### 8. pshuffle

Performs a collective shuffle according to a permutation.

```python
jax.lax.pshuffle(x, axis_name, perm)
```

**Parameters:**
- Same as `ppermute`

**Returns:**
- Shuffled array(s) according to the permutation

**Notes:**
- Implemented as a wrapper around `ppermute`

### 9. all_to_all

Performs an all-to-all communication pattern.

```python
jax.lax.all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None)
```

**Parameters:**
- `x`: array(s) with a mapped axis named `axis_name`
- `axis_name`: hashable Python object used to name a pmapped axis
- `split_axis`: axis to split along for sending
- `concat_axis`: axis to concatenate along after receiving
- `axis_index_groups`: optional groups for partial communication

**Returns:**
- Transformed array after all-to-all communication

## Utility Functions

### axis_index

Returns the index along the mapped axis.

```python
jax.lax.axis_index(axis_name)
```

**Parameters:**
- `axis_name`: hashable Python object used to name a pmapped axis

**Returns:**
- Integer index of the current device along the named axis

**Example:**
```python
# Get device index within pmap
def f(x):
    idx = jax.lax.axis_index('i')
    return x * idx

result = jax.pmap(f, axis_name='i')(jnp.ones(4))
# Returns: [0. 1. 2. 3.]
```

## Advanced Features

### Axis Index Groups

Many collective operations support `axis_index_groups` for performing partial reductions over subsets of devices:

```python
# Perform psum over pairs of devices
groups = [[0, 1], [2, 3]]
result = jax.pmap(
    lambda x: jax.lax.psum(x, 'i', axis_index_groups=groups),
    axis_name='i'
)(x)
```

### Nested Parallelism

JAX supports nested `pmap` calls with different axis names:

```python
@partial(jax.pmap, axis_name='i')
@partial(jax.pmap, axis_name='j')
def f(x):
    x = jax.lax.psum(x, 'i')  # Sum across 'i' axis
    x = jax.lax.pmean(x, 'j')  # Mean across 'j' axis
    return x
```

### Integration with shard_map

Modern JAX also provides `shard_map` which represents devices as a Mesh and allows collective communication operations:

```python
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices, axis_names=('x', 'y'))

@partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P())
def f(x):
    return jax.lax.psum(x, ('x', 'y'))
```

## Implementation Details

### Efficient Algorithms

- One efficient algorithm for computing `psum(x, axis_name)` is to perform a `psum_scatter` followed by an `all_gather`:
  ```python
  # Equivalent to psum but potentially more efficient
  result = all_gather(psum_scatter(x, axis_name), axis_name)
  ```

### Variance Types

- JAX has variants like `all_gather_invariant` which lowers to the same operation as `all_gather` but has different device variance type
- `all_gather` has a `pbroadcast` fused into it, whereas `all_gather_invariant` does not

### Transposition and Gradients

- Collective operations have well-defined transposes for automatic differentiation
- The transpose of `psum` is effectively an identity operation
- The transpose of `psum_scatter` is `all_gather`

## Best Practices

1. **Use appropriate collectives**: Choose the right collective for your use case (e.g., `pmean` for averaging gradients)

2. **Consider memory usage**: Operations like `all_gather` increase memory usage proportionally to the number of devices

3. **Minimize communication**: Batch operations when possible to reduce communication overhead

4. **Profile performance**: Use JAX's profiling tools to identify communication bottlenecks

5. **Handle device failures**: Consider using `axis_index_groups` for fault tolerance in large-scale deployments

## Common Patterns

### Data Parallelism
```python
# Parallel training step with gradient averaging
@jax.pmap
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    grads = jax.lax.pmean(grads, 'batch')  # Average gradients
    params = update(params, grads)
    return params, loss
```

### Model Parallelism
```python
# Split model across devices
@partial(jax.pmap, axis_name='model')
def forward(x, params):
    # Each device handles part of the model
    local_out = local_forward(x, params)
    # Gather results from all devices
    return jax.lax.all_gather(local_out, 'model')
```

### Pipeline Parallelism
```python
# Pipeline stages with ppermute
@partial(jax.pmap, axis_name='pipe')
def pipeline_step(x, stage_params):
    # Process local stage
    x = stage_fn(x, stage_params)
    # Pass to next stage
    perm = [(i, (i + 1) % n_stages) for i in range(n_stages)]
    return jax.lax.ppermute(x, 'pipe', perm)
```

[source](https://jax.readthedocs.io/en/latest/jax.lax.html#parallel-operators)