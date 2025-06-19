# JAX Multi-Process and Multi-Host Documentation

## Overview

JAX provides powerful capabilities for distributed computing across multiple processes and hosts, enabling you to scale your computations across GPU clusters, Cloud TPU pods, and other multi-device environments. This documentation covers the key concepts, APIs, and best practices for using JAX in multi-process environments.

## Table of Contents

1. [Introduction](#introduction)
2. [Multi-Process Programming Model](#multi-process-programming-model)
3. [Initialization](#initialization)
4. [Local vs Global Devices](#local-vs-global-devices)
5. [Running Multi-Process Computations](#running-multi-process-computations)
6. [Distributed Data Loading](#distributed-data-loading)
7. [Key APIs and Functions](#key-apis-and-functions)
8. [Best Practices and Warnings](#best-practices-and-warnings)
9. [Examples](#examples)

## Introduction

JAX uses a "multi-controller" programming model where each JAX Python process runs independently, sometimes referred to as a Single Program, Multiple Data (SPMD) model. Generally, the same JAX Python program is run in each process, with only slight differences between each process's execution (e.g., different processes will load different input data).

An important requirement of multi-process environments in JAX is direct communication links between accelerators, such as:
- High-speed interconnects for Cloud TPUs
- NCCL for GPUs

These links allow collective operations to run across multiple processes' worth of accelerators with high performance.

**Important**: You must manually run your JAX program on each host! JAX doesn't automatically start multiple processes from a single program invocation.

## Multi-Process Programming Model

### Key Concepts

- **Controller**: A Python process running JAX computations
- **Local Devices**: Devices physically attached to a specific host that a process can directly address and launch computations on
- **Global Devices**: All devices across all processes

### Process Assignment Assumptions

- **GPU with Slurm/Open MPI**: One process is started per GPU (each process assigned one visible local device)
- **Other configurations**: One process is started per host (each process assigned all local devices)

## Initialization

To set up multi-process JAX, you need to call `jax.distributed.initialize()` at the start of each process:

```python
import jax

# Manual initialization with explicit parameters
jax.distributed.initialize(
    coordinator_address="192.168.0.1:1234",
    num_processes=2,
    process_id=0
)
```

### Automatic Initialization

On Cloud TPU, Slurm, and Open MPI environments, you can simply call:

```python
jax.distributed.initialize()
```

Default values for the arguments will be chosen automatically.

### What `initialize()` Does

- Prepares JAX for execution on multi-host GPU and Cloud TPU
- Allows JAX processes to discover each other and share topology information
- Performs health checking, ensuring all processes shut down if any process dies
- Sets up distributed checkpointing capabilities

**Note**: `initialize()` must be called before performing any JAX computations.

## Local vs Global Devices

### Local Devices

A process's local devices are those that it can directly address and launch computations on:
- **GPU cluster**: Each host can only launch computations on directly attached GPUs
- **Cloud TPU pod**: Each host can only launch computations on the 8 TPU cores attached directly to that host

```python
# Get local devices
local_devices = jax.local_devices()
print(f"Process {jax.process_index()} has {len(local_devices)} local devices")
```

### Global Devices

The global devices are the devices across all processes. A computation can span devices across processes and perform collective operations via the direct communication links between devices.

```python
# Get all global devices
global_devices = jax.devices()
print(f"Total global devices: {len(global_devices)}")
```

## Running Multi-Process Computations

Programming multiple processes from JAX usually looks just like programming a single process, just with more devices! The main exceptions are around data coming in or out of JAX.

### Basic Multi-Process Example

```python
import jax
import jax.numpy as jnp

# Initialize the distributed environment
jax.distributed.initialize()

# Create data on local devices
xs = jnp.ones(jax.local_device_count())

# Run a parallel computation across all devices
result = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(f"Process {jax.process_index()} result: {result}")
```

### Creating Process-Spanning Arrays

There are three main approaches to create arrays that span multiple processes:

#### 1. Load Full Array and Shard

```python
# Each process loads the full array
full_array = load_my_data()

# Create a sharding specification
mesh = jax.sharding.Mesh(jax.devices(), ('devices',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices'))

# Shard the array across all devices
sharded_array = jax.device_put(full_array, sharding)
```

#### 2. Load Process-Local Data

```python
# Each process loads its portion of the data
local_data = load_my_data_shard(jax.process_index())

# Create array from process-local data
global_array = jax.make_array_from_process_local_data(
    sharding, local_data
)
```

#### 3. Manual Per-Device Arrays

```python
# Create arrays on each local device
per_device_arrays = [
    create_array_for_device(d) for d in jax.local_devices()
]

# Combine into global array
global_array = jax.make_array_from_single_device_arrays(
    shape, sharding, per_device_arrays
)
```

## Distributed Data Loading

### Key Approaches

1. **Load Global Data in Each Process**: Simple but potentially memory-intensive
2. **Per-Device Data Pipeline**: Each device loads its own data
3. **Consolidated Per-Process Pipeline**: Each process loads data for its local devices
4. **Load and Reshard**: Load conveniently and reshard during computation

### Data Loading Example

```python
import tensorflow_datasets as tfds

# Shard dataset across processes
ds = tfds.load('mnist', split='train')
ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

# Create mesh for data parallelism
mesh_shape = (jax.device_count() // 2, 2)
devices = np.array(jax.devices()).reshape(mesh_shape)
mesh = jax.sharding.Mesh(devices, ["model_replicas", "data_parallelism"])

# Create sharding for data
data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, "data_parallelism")
)
```

### Replication Strategies

- **Full Replication**: All devices have complete data copy
- **Partial Replication**: Multiple copies of data, each potentially sharded

## Key APIs and Functions

### Core Distributed Functions

```python
# Initialize distributed JAX
jax.distributed.initialize(
    coordinator_address=None,  # Required unless auto-detected
    num_processes=None,        # Required unless auto-detected
    process_id=None,          # Required unless auto-detected
    local_device_ids=None,    # Optional: restrict local devices
    initialization_timeout=300 # Optional: timeout in seconds
)

# Get process information
process_id = jax.process_index()  # Current process index
num_processes = jax.process_count()  # Total number of processes

# Device information
local_devices = jax.local_devices()  # Devices on this process
global_devices = jax.devices()       # All devices across processes
```

### Array Creation Functions

```python
# Create array from process-local data
array = jax.make_array_from_process_local_data(
    sharding, data, global_shape=None
)

# Create array from per-device arrays
array = jax.make_array_from_single_device_arrays(
    shape, sharding, device_arrays
)

# Check if array is fully addressable
is_local = array.is_fully_addressable  # True if all shards are local
```

### Sharding and Mesh Creation

```python
# Create a device mesh
mesh = jax.sharding.Mesh(devices, axis_names)

# Create named sharding
sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(...)
)

# Create positional sharding
sharding = jax.sharding.PositionalSharding(devices)
```

## Best Practices and Warnings

### Critical Warning: Synchronization

**All processes must run the same computations in the same order!** Otherwise, you risk:
- Deadlocks
- Hangs
- Incorrect results

### Common Pitfalls to Avoid

1. **Different-shaped inputs across processes**
   ```python
   # BAD: Different shapes on different processes
   if jax.process_index() == 0:
       data = jnp.ones((10, 5))
   else:
       data = jnp.ones((8, 5))  # Different shape!
   ```

2. **Non-deterministic operations**
   ```python
   # BAD: Random order might differ across processes
   for key in dict.keys():  # Dictionary iteration order not guaranteed
       process(key)
   ```

3. **Conditional execution based on process**
   ```python
   # BAD: Only some processes execute
   if jax.process_index() == 0:
       result = jax.pmap(fn)(data)  # Deadlock!
   ```

### Best Practices

1. **Always initialize before any JAX operations**
   ```python
   jax.distributed.initialize()
   # Now safe to use JAX
   ```

2. **Use consistent random seeds**
   ```python
   key = jax.random.PRNGKey(42 + jax.process_index())
   ```

3. **Log from single process to avoid clutter**
   ```python
   if jax.process_index() == 0:
       print("Training started...")
   ```

4. **Verify device setup**
   ```python
   print(f"Process {jax.process_index()}: "
         f"{len(jax.local_devices())} local devices, "
         f"{len(jax.devices())} total devices")
   ```

## Examples

### Complete Multi-Process Training Example

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax

# Initialize distributed JAX
jax.distributed.initialize()

# Define a simple model
def model(params, x):
    return jnp.dot(x, params['w']) + params['b']

def loss_fn(params, x, y):
    pred = model(params, x)
    return jnp.mean((pred - y) ** 2)

# Initialize parameters
key = random.PRNGKey(42)
params = {
    'w': random.normal(key, (10, 1)),
    'b': random.normal(key, (1,))
}

# Create optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Create sharding for parameters
mesh = jax.sharding.Mesh(jax.devices(), ('devices',))
param_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec()
)

# Shard parameters across devices
params = jax.tree_map(lambda x: jax.device_put(x, param_sharding), params)

# Training step
@jit
def train_step(params, opt_state, batch):
    x, y = batch
    grads = grad(loss_fn)(params, x, y)
    
    # Average gradients across all devices
    grads = jax.tree_map(lambda g: jax.lax.pmean(g, 'devices'), grads)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop (simplified)
for epoch in range(10):
    # Load batch (each process loads different data)
    x = random.normal(key, (32, 10))
    y = random.normal(key, (32, 1))
    batch = (x, y)
    
    # Training step
    params, opt_state = train_step(params, opt_state, batch)
    
    # Log from first process only
    if jax.process_index() == 0 and epoch % 5 == 0:
        print(f"Epoch {epoch} completed")
```

### Multi-Host Data Pipeline Example

```python
import jax
import numpy as np
from typing import Iterator

class DistributedDataLoader:
    def __init__(self, data, batch_size, mesh, data_sharding):
        self.data = data
        self.batch_size = batch_size
        self.mesh = mesh
        self.data_sharding = data_sharding
        
        # Shard data across processes
        total_samples = len(data)
        samples_per_process = total_samples // jax.process_count()
        start_idx = jax.process_index() * samples_per_process
        end_idx = start_idx + samples_per_process
        
        self.local_data = data[start_idx:end_idx]
        
    def __iter__(self) -> Iterator:
        num_local_batches = len(self.local_data) // self.batch_size
        
        for i in range(num_local_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            local_batch = self.local_data[start:end]
            
            # Create global batch from local data
            global_batch = jax.make_array_from_process_local_data(
                self.data_sharding, local_batch
            )
            
            yield global_batch

# Usage
jax.distributed.initialize()

# Create mesh and sharding
devices = np.array(jax.devices()).reshape(-1)
mesh = jax.sharding.Mesh(devices, ('batch',))
data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec('batch')
)

# Create distributed data loader
data = np.random.randn(10000, 784)  # Example data
dataloader = DistributedDataLoader(data, batch_size=32, 
                                 mesh=mesh, 
                                 data_sharding=data_sharding)

# Iterate over distributed batches
for batch in dataloader:
    # batch is now a jax.Array sharded across all devices
    print(f"Batch shape: {batch.shape}, "
          f"is distributed: {not batch.is_fully_addressable}")
```

### TPU Pod Example

```python
import jax
import jax.numpy as jnp

# On Cloud TPU, initialization is automatic
jax.distributed.initialize()

# Verify TPU setup
print(f"Process {jax.process_index()} of {jax.process_count()}")
print(f"Local devices: {jax.local_devices()}")

# Create computation mesh
devices = jax.devices()
mesh_shape = (len(devices) // 8, 8)  # Typical TPU pod structure
devices_array = np.array(devices).reshape(mesh_shape)
mesh = jax.sharding.Mesh(devices_array, ('dp', 'mp'))

# Define sharding strategies
# Data parallel sharding
dp_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec('dp', None)
)

# Model parallel sharding
mp_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, 'mp')
)

# Mixed sharding
mixed_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec('dp', 'mp')
)
```

## Platform-Specific Notes

### Cloud TPU
- On TPU, calling `jax.distributed.initialize()` is optional but recommended
- Enables additional checkpointing and health checking features
- Automatic device detection and process coordination

### GPU Clusters
- Requires explicit coordinator address unless using Slurm/Open MPI
- NCCL backend for collective operations
- One process per GPU is the default configuration

### CPU-Only Multi-Process
- `jax.distributed.initialize()` does not provide MPI.COMM_WORLD functionality
- For CPU-only parallelism, consider extensions like mpi4jax or alpa

## Troubleshooting

### Common Issues and Solutions

1. **Initialization Timeout**
   ```python
   # Increase timeout for slow network setup
   jax.distributed.initialize(initialization_timeout=600)
   ```

2. **Device Visibility**
   ```python
   # Restrict visible devices
   jax.distributed.initialize(local_device_ids=[0, 1])
   ```

3. **Debugging Deadlocks**
   ```python
   # Add logging to identify where processes diverge
   print(f"Process {jax.process_index()} at checkpoint A")
   # ... computation ...
   print(f"Process {jax.process_index()} at checkpoint B")
   ```

4. **Memory Issues**
   ```python
   # Use process-local data loading instead of full replication
   # Bad: each process loads full dataset
   # data = load_full_dataset()
   
   # Good: each process loads its shard
   data = load_dataset_shard(jax.process_index(), jax.process_count())
   ```

## Performance Considerations

1. **Communication Overhead**: Minimize cross-process communication by:
   - Batching collective operations
   - Using appropriate sharding strategies
   - Leveraging data locality

2. **Load Balancing**: Ensure equal work distribution:
   - Use consistent batch sizes across processes
   - Balance data shards evenly
   - Consider dynamic load balancing for irregular workloads

3. **Memory Efficiency**:
   - Shard large arrays instead of replicating
   - Use gradient accumulation for large models
   - Stream data instead of loading all at once

## Additional Resources

- [JAX Distributed Arrays and Automatic Parallelization Tutorial](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [JAX Sharding Module Documentation](https://jax.readthedocs.io/en/latest/jax.sharding.html)
- [Cloud TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture)
- [JAX GitHub Discussions on Multi-Host Training](https://github.com/jax-ml/jax/discussions)

---

[source](https://jax.readthedocs.io/en/latest/multi_process.html)