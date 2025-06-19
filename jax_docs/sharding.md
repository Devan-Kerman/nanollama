# JAX Sharding Documentation

## Overview

JAX provides a comprehensive sharding API for distributed array computation across multiple devices. The sharding system allows you to specify how data should be partitioned across devices for parallel computation.

## Core Sharding Classes

### jax.sharding.Sharding

Base class for all sharding specifications. A `Sharding` describes how an array is laid out across devices.

#### Key Properties:
- **is_fully_addressable**: A sharding is fully addressable if the current process can address all of the devices named in the Sharding. `is_fully_addressable` is equivalent to "is_local" in multi-process JAX.
- **is_fully_replicated**: A sharding is fully replicated if each device has a complete copy of the entire data.

### jax.sharding.NamedSharding

A `NamedSharding` expresses sharding using named axes. It is a pair of a `Mesh` of devices and `PartitionSpec` which describes how to shard an array across that mesh.

```python
class jax.sharding.NamedSharding(mesh, spec)
```

#### Parameters:
- **mesh**: A `Mesh` object describing the device mesh
- **spec**: A `PartitionSpec` describing how to partition the array

#### Example:
```python
from jax.sharding import Mesh, PartitionSpec as P
import jax
import numpy as np

# Create a 2x4 mesh of devices
mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))

# Create a partition spec
spec = P('x', 'y')

# Create a NamedSharding
named_sharding = jax.sharding.NamedSharding(mesh, spec)
```

### jax.sharding.PositionalSharding

A `PositionalSharding` expresses sharding using device positions. A `NamedSharding` may be equivalent to a `PositionalSharding` if both place the same shards of the array on the same devices.

### jax.sharding.Mesh

A `Mesh` is a multidimensional NumPy array of JAX devices, where each axis of the mesh has a name.

```python
class jax.sharding.Mesh(devices, axis_names, *, axis_types=None)
```

#### Parameters:
- **devices**: A NumPy array of JAX devices
- **axis_names**: Names for each axis of the device array
- **axis_types**: Optional types for each axis

#### Example:
```python
import jax
import numpy as np
from jax.sharding import Mesh

# Create a 2D mesh with axis names 'x' and 'y'
devices = np.array(jax.devices()).reshape(2, 4)
mesh = Mesh(devices, ('x', 'y'))
```

### jax.sharding.PartitionSpec

Tuple describing how to partition an array across a mesh of devices. Each element is either `None`, a string, or a tuple of strings.

```python
class jax.sharding.PartitionSpec(*partitions)
```

#### Description:
- Each element describes how an input dimension is partitioned across zero or more mesh dimensions
- `None` means the dimension is not partitioned
- A string names the mesh axis across which the dimension is partitioned
- A tuple of strings means the dimension is partitioned across multiple mesh axes

#### Example:
```python
from jax.sharding import PartitionSpec as P

# First dimension sharded across 'x', second across 'y'
spec1 = P('x', 'y')

# First dimension sharded across 'x', second not sharded
spec2 = P('x', None)

# First dimension sharded across both 'x' and 'y'
spec3 = P(('x', 'y'), None)
```

## Mesh Utilities

### jax.experimental.mesh_utils

The `jax.experimental.mesh_utils` module provides utilities for creating device meshes.

#### Key Functions:

##### create_device_mesh()
Creates a device mesh from available devices. This is typically used to create meshes in a convenient way.

```python
from jax.experimental import mesh_utils

# Create a mesh with specific shape
mesh_shape = (2, 4)
devices = mesh_utils.create_device_mesh(mesh_shape)
```

## Usage with pjit

The `pjit` (partitioned JIT) function allows you to specify input and output shardings for JIT-compiled functions.

```python
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec

@pjit(in_shardings=PartitionSpec('x', 'y'), 
      out_shardings=PartitionSpec('x', None))
def f(x):
    return jnp.sum(x, axis=1)

with Mesh(devices, ('x', 'y')):
    result = f(input_array)
```

## Sharding Constraints

### jax.lax.with_sharding_constraint

Applies a sharding constraint to an intermediate value in a computation.

```python
import jax.lax

def f(x):
    # Constrain x to be sharded according to spec
    x = jax.lax.with_sharding_constraint(x, spec)
    return x @ x.T
```

## Debugging and Inspection

### jax.debug.inspect_array_sharding

Enables inspecting array sharding inside JIT-ted functions.

```python
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec

x = jnp.arange(8, dtype=jnp.float32)

def f_(x):
    x = jnp.sin(x)
    jax.debug.inspect_array_sharding(x, callback=print)
    return jnp.square(x)

f = pjit(f_, in_shardings=PartitionSpec('dev'), 
         out_shardings=PartitionSpec('dev'))

with Mesh(jax.devices(), ('dev',)):
    f.lower(x).compile()
    # Output: NamedSharding(mesh={'dev': 8}, partition_spec=PartitionSpec(('dev',),))
```

## Making Arrays from Process-Local Data

### jax.make_array_from_process_local_data

Creates a distributed array from process-local data in multi-process environments.

```python
# In a multi-process setting
local_data = process_local_array_data()
global_array = jax.make_array_from_process_local_data(
    local_data, 
    sharding=named_sharding
)
```

## Complete Example: Matrix Multiplication with Sharding

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

# Create a device mesh
devices = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices, ('x', 'y'))

# Define partition specs
# A is sharded along rows (x axis)
# B is sharded along columns (y axis)
# Output is sharded along both dimensions
a_spec = P('x', None)
b_spec = P(None, 'y')
c_spec = P('x', 'y')

# Define the computation
@pjit(in_shardings=(a_spec, b_spec), out_shardings=c_spec)
def matmul(a, b):
    return jnp.dot(a, b)

# Use within mesh context
with mesh:
    # Create input arrays
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (1024, 512))
    b = jax.random.normal(key, (512, 256))
    
    # Compute sharded matrix multiplication
    c = matmul(a, b)
    
    # Inspect sharding
    jax.debug.visualize_array_sharding(c)
```

## Advanced Sharding Patterns

### 1. Data Parallelism
```python
# Shard batch dimension across all devices
data_parallel_spec = P('devices', None, None)
```

### 2. Model Parallelism
```python
# Shard model parameters across devices
model_parallel_spec = P(None, 'devices')
```

### 3. Pipeline Parallelism
```python
# Shard layers across device groups
pipeline_spec = P('pipeline', None, None)
```

### 4. Hybrid Parallelism
```python
# Combine data and model parallelism
hybrid_spec = P('data', 'model')
```

## Best Practices

1. **Use Named Axes**: Always use descriptive names for mesh axes
2. **Context Managers**: Use `Mesh` as a context manager for cleaner code
3. **Validate Shardings**: Check `is_fully_addressable` before operations
4. **Debugging**: Use `inspect_array_sharding` and `visualize_array_sharding` liberally
5. **Performance**: Consider communication patterns when designing shardings

## Additional Resources

- **shard_map**: Lower-level API for explicit per-device computations
- **multihost_utils**: Utilities for multi-host synchronization
- **sharding_impls**: Internal implementation details

[source](https://jax.readthedocs.io/en/latest/jax.sharding.html)