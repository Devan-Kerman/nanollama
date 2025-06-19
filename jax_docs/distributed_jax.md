# Distributed JAX and JAX Parallelism Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [JAX Distributed Module](#jax-distributed-module)
3. [Multi-Process Programming](#multi-process-programming)
4. [Sharding and Distributed Arrays](#sharding-and-distributed-arrays)
5. [Parallel Primitives](#parallel-primitives)
6. [Advanced Topics](#advanced-topics)
7. [Debugging and Visualization](#debugging-and-visualization)
8. [References and Tutorials](#references-and-tutorials)

## Introduction

JAX provides multiple approaches to distributed and parallel computing, from simple data parallelism to sophisticated sharding strategies. The framework supports execution across multiple devices (GPUs, TPUs) and multiple hosts, with automatic optimization of communication patterns.

## JAX Distributed Module

### Overview

The `jax.distributed` module enables multi-host execution on GPU clusters and Cloud TPU pods. The distributed system serves multiple roles:
- Allows JAX processes to discover each other and share topology information
- Performs health checking, ensuring all processes shut down if any process dies
- Enables distributed checkpointing

### Initialization

The `jax.distributed.initialize()` function must be called before any JAX computations when running on multi-host systems.

#### Required Parameters:
- `coordinator_address`: IP address of process 0 and a port (e.g., '10.0.0.1:1234')
- `num_processes`: Total number of processes in the cluster
- `process_id`: Unique identifier for each process (0 to num_processes-1)

#### Example: Two-GPU Setup

```python
# On process 0 (coordinator):
jax.distributed.initialize(
    coordinator_address='10.0.0.1:1234',
    num_processes=2,
    process_id=0
)

# On process 1:
jax.distributed.initialize(
    coordinator_address='10.0.0.1:1234',
    num_processes=2,
    process_id=1
)
```

#### Automatic Configuration

For TPU, Slurm, or Open MPI environments, all arguments are optional and will be chosen automatically if omitted.

### Process Management Functions

```python
# Get current process index
process_idx = jax.process_index()  # Returns 0 on single-process, varies on multi-process

# Get total number of processes
num_processes = jax.process_count()  # Returns number of JAX processes in backend

# Check device properties
devices = jax.devices()
local_devices = jax.local_devices()
```

## Multi-Process Programming

### SPMD Model

JAX follows the Single Program Multiple Data (SPMD) model:
- Same functions must run in the same order on all processes
- Functions can be interspersed with single-process operations
- On multi-process platforms (TPU pods), computations run across all available devices

### Data Distribution

#### Host-Local to Global Arrays

```python
from jax.experimental.multihost_utils import host_local_array_to_global_array

# Convert host-local data to globally sharded array
global_array = host_local_array_to_global_array(
    local_data,
    global_mesh,
    pspecs
)
```

#### Process-Local Data

```python
# Create distributed arrays from process-local data
array = jax.make_array_from_process_local_data(
    sharding,
    local_data
)
```

## Sharding and Distributed Arrays

### Core Concepts

#### Sharding
A `Sharding` describes how an array is laid out across multiple devices. Key properties:
- **Fully addressable**: Current process can address all devices in the sharding
- **is_fully_addressable**: Equivalent to "is_local" in multi-process JAX

#### NamedSharding
Expresses sharding using named axes, consisting of:
- **Mesh**: Multidimensional NumPy array of JAX devices with named axes
- **PartitionSpec**: Tuple describing how dimensions map to mesh axes

### Creating Shardings

```python
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Create a 2D mesh of devices
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), ('x', 'y'))

# Define how to partition data
# First dimension sharded on 'x', second on 'y'
pspec = PartitionSpec('x', 'y')

# Create the sharding
sharding = NamedSharding(mesh, pspec)
```

### PartitionSpec Examples

```python
# Shard first dimension across 'x' axis
PartitionSpec('x', None)

# Shard both dimensions
PartitionSpec('x', 'y')

# Replicate first dimension, shard second across both axes
PartitionSpec(None, ('x', 'y'))

# Fully replicated
PartitionSpec()
```

### Sharding Constraints

```python
# Apply sharding constraints within computations
@jax.jit
def f(x):
    # Ensure intermediate has specific sharding
    x = jax.lax.with_sharding_constraint(x, sharding)
    return x @ x.T
```

## Parallel Primitives

### pmap (Parallel Map)

Traditional data-parallel primitive:

```python
@jax.pmap
def parallel_fn(x):
    return jax.lax.psum(x, axis_name='i')

# Execute across devices
result = parallel_fn(data)  # data shape: (num_devices, ...)
```

### pjit (Now Equivalent to jit)

**Note**: `pjit` is now equivalent to `jax.jit`. Use `jit` with sharding specifications:

```python
@jax.jit
def f(x, y):
    return x @ y

# With explicit sharding
sharded_f = jax.jit(f, in_shardings=(sharding1, sharding2), 
                    out_shardings=sharding_out)
```

### shard_map (Experimental)

Maps functions over shards of data with explicit control:

```python
from jax.experimental.shard_map import shard_map

@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=PartitionSpec('x', 'y'),
    out_specs=PartitionSpec('x', None)
)
def sharded_fn(x_shard):
    # Function operates on shards
    return x_shard.sum(axis=1)
```

### xmap (Experimental)

Named-axis programming model for easier parallelism:

```python
from jax.experimental.maps import xmap

# Define named axes
f = xmap(
    lambda x: x.sum(),
    in_axes=['batch', 'features'],
    out_axes=['batch']
)
```

Key benefits of xmap:
- Self-documenting with named axes
- Easier to avoid errors
- Can replace pmap for multi-dimensional hardware meshes
- Scales from laptop CPU to TPU supercomputers

## Advanced Topics

### Mixed Precision and Sharding

```python
# Combine sharding with dtype specifications
@jax.jit
def mixed_precision_fn(x):
    x = x.astype(jax.numpy.bfloat16)
    x = jax.lax.with_sharding_constraint(x, sharding)
    return x @ x.T
```

### Custom Collectives

```python
# All-reduce across devices
jax.lax.psum(x, axis_name='devices')

# All-gather
jax.lax.all_gather(x, axis_name='devices')

# Permutation collective
jax.lax.ppermute(x, axis_name='devices', perm=[(0, 1), (1, 0)])
```

### Distributed Checkpointing

JAX's distributed system supports checkpointing across multiple hosts:

```python
# Save checkpoint (coordinated across processes)
# Implementation depends on checkpointing library used
```

## Debugging and Visualization

### Inspect Sharding

```python
# Inspect array sharding
jax.debug.inspect_array_sharding(array)

# Works inside jit-compiled functions
@jax.jit
def f(x):
    jax.debug.inspect_array_sharding(x)
    return x * 2
```

### Visualize Sharding

```python
# Rich visualization of sharding
jax.debug.visualize_sharding(sharding)

# Visualize array's sharding
jax.debug.visualize_array_sharding(array)
```

### Print from All Processes

```python
# Debug print that works across processes
jax.debug.print("Process {}: value = {}", 
                jax.process_index(), value)
```

## Best Practices

### 1. Initialize Early
Always call `jax.distributed.initialize()` before any JAX operations in multi-host settings.

### 2. Use Sharding Specifications
Be explicit about data layout to optimize communication:

```python
# Good: Explicit sharding
@jax.jit
def f(x):
    return x @ x.T

sharded_f = jax.jit(f, in_shardings=sharding, out_shardings=sharding)

# Less optimal: Let JAX infer
@jax.jit
def f(x):
    return x @ x.T
```

### 3. Minimize Communication
Structure computations to minimize cross-device communication:

```python
# Prefer local reductions when possible
@shard_map(..., in_specs=P('x', 'y'), out_specs=P('x'))
def local_reduce(x_shard):
    return x_shard.sum(axis=1)  # Reduction within shard
```

### 4. Use Appropriate Parallelism Level
- `pmap`: Simple data parallelism
- `jit` with sharding: Flexible SPMD parallelism
- `shard_map`: Fine-grained control over sharding
- `xmap`: Named-axis model for complex parallelism patterns

## Common Patterns

### Data Parallel Training

```python
# Replicate model, shard data
model_sharding = NamedSharding(mesh, PartitionSpec())
data_sharding = NamedSharding(mesh, PartitionSpec('devices', None))

@jax.jit
def train_step(model, batch):
    model = jax.lax.with_sharding_constraint(model, model_sharding)
    batch = jax.lax.with_sharding_constraint(batch, data_sharding)
    # ... training logic
    return updated_model
```

### Model Parallel Inference

```python
# Shard model across devices
model_sharding = NamedSharding(mesh, PartitionSpec('model_parallel', None))

@jax.jit
def inference(model, inputs):
    model = jax.lax.with_sharding_constraint(model, model_sharding)
    # ... inference logic
    return outputs
```

## References and Tutorials

### Official Documentation
- [JAX Distributed Module](https://jax.readthedocs.io/en/latest/jax.distributed.html)
- [JAX Sharding Module](https://jax.readthedocs.io/en/latest/jax.sharding.html)
- [Distributed Arrays and Automatic Parallelization Tutorial](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

### Key Tutorials
- [Introduction to Parallel Programming](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
- [SPMD Multi-device Parallelism with shard_map](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html)
- [xmap Tutorial](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html)

### API References
- [`jax.distributed.initialize()`](https://jax.readthedocs.io/en/latest/jax.distributed.html#jax.distributed.initialize)
- [`jax.sharding.NamedSharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
- [`jax.sharding.PartitionSpec`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.PartitionSpec)
- [`jax.experimental.shard_map.shard_map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.shard_map.shard_map.html)

## Glossary

- **SPMD**: Single Program Multiple Data - programming model where same program runs on all devices
- **Mesh**: Multi-dimensional array of devices with named axes
- **PartitionSpec**: Specification of how array dimensions map to mesh axes
- **Sharding**: Description of how data is distributed across devices
- **Collective**: Operation involving communication across multiple devices (e.g., all-reduce)
- **Shard**: Portion of data residing on a single device

[source](https://jax.readthedocs.io/en/latest/)