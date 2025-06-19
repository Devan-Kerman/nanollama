# JAX Pallas Documentation

## Overview

Pallas is an extension to JAX that enables writing custom kernels for GPU and TPU. It aims to provide fine-grained control over the generated code, combined with the high-level ergonomics of JAX tracing and the jax.numpy API.

Pallas allows you to use the same JAX functions and APIs but operates at a lower level of abstraction. Specifically, Pallas requires users to think about memory access and how to divide up computations across multiple compute units in a hardware accelerator. On GPUs, Pallas lowers to Triton and on TPUs, Pallas lowers to Mosaic.

**Important Note**: Pallas is experimental and is changing frequently. You can expect to encounter errors and unimplemented cases.

## Table of Contents

1. [Quickstart](#quickstart)
2. [Core Concepts](#core-concepts)
3. [Grids and BlockSpecs](#grids-and-blockspecs)
4. [Memory Hierarchy](#memory-hierarchy)
5. [TPU Backend](#tpu-backend)
6. [GPU Backends](#gpu-backends)
7. [Software Pipelining](#software-pipelining)
8. [API Reference](#api-reference)
9. [Design Principles](#design-principles)
10. [Examples](#examples)

## Quickstart

### Simple Vector Addition Example

First, let's write a kernel that adds two vectors:

```python
from functools import partial
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y
```

To invoke the kernel from a JAX computation, use the `pallas_call` higher-order function:

```python
@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

# Example usage
result = add_vectors(jnp.arange(8), jnp.arange(8))
```

### Key Concepts

1. **Kernel Functions**: A "kernel" is a program that runs as an atomic unit of execution on an accelerator, without any interaction with the host.

2. **References (`Ref`)**: Kernel functions operate on `Ref` objects representing mutable memory buffers, not regular JAX arrays.

3. **Memory Management**: Part of writing Pallas kernels is thinking about how to take big arrays in HBM (DRAM) and express computations on "blocks" that fit in SRAM.

## Core Concepts

### Kernel Programming Model

Pallas kernels are defined as Python functions that operate on references (`Ref` objects) rather than arrays:

```python
def kernel_function(x_ref, y_ref, output_ref):
    # Read from refs
    x_value = x_ref[...]
    y_value = y_ref[...]
    
    # Perform computation
    result = x_value + y_value
    
    # Write to output ref
    output_ref[...] = result
```

### Grid Programming

A grid is a tuple of integers that specifies an iteration space. The kernel function runs once for each element in the grid:

```python
# 1D grid: runs kernel 8 times
grid = (8,)

# 2D grid: runs kernel 20 times (4 * 5)
grid = (4, 5)

# Use in pallas_call
result = pl.pallas_call(kernel, out_shape=..., grid=grid)(inputs)
```

### Memory Spaces

Pallas exposes different memory hierarchies on accelerators:

- **HBM/DRAM**: Main memory, largest but slowest
- **SRAM/VMEM**: Faster but smaller on-chip memory
- **Registers**: Fastest but very limited

## Grids and BlockSpecs

### Understanding Grids

A grid defines how a kernel function is executed multiple times:

- A 1D grid `(n,)` is conceptually equivalent to:
  ```python
  for i in range(n):
      kernel(...)
  ```

- A 2D grid `(n, m)` is equivalent to:
  ```python
  for i in range(n):
      for j in range(m):
          kernel(...)
  ```

### BlockSpecs

BlockSpecs define how to slice inputs for each kernel invocation:

```python
# Example BlockSpec
block_spec = pl.BlockSpec(
    block_shape=(64, 128),  # Size of each block
    index_map=lambda i, j: (i, j)  # How to map grid indices to blocks
)

# Use in pallas_call
result = pl.pallas_call(
    kernel,
    grid=(10, 5),
    in_specs=[block_spec],
    out_specs=block_spec,
    out_shape=...
)(input_array)
```

### Working with Grid Indices

Inside kernels, you can access the current grid position:

```python
def kernel(x_ref, o_ref):
    i = pl.program_id(0)  # Get current index along first grid axis
    j = pl.program_id(1)  # Get current index along second grid axis
    
    # Process block based on grid position
    o_ref[...] = x_ref[...] * i + j
```

## Memory Hierarchy

### TPU Memory Hierarchy

1. **HBM (High Bandwidth Memory)**: Main memory where inputs typically reside
2. **VMEM (Vector Memory)**: 16MB+ of on-chip SRAM for vector operations
3. **SMEM (Scalar Memory)**: Low-latency memory for 32-bit scalar values
4. **Scalar Registers**: Fastest storage for individual values

### GPU Memory Hierarchy

1. **Global Memory (GMEM)**: Largest but slowest, accessible by all threads
2. **Shared Memory (SMEM)**: Fast memory shared within a streaming multiprocessor
3. **Register Memory**: Fastest, smallest, thread-private storage
4. **Tensor Memory (TMEM)**: On Blackwell GPUs, for MMA accumulator storage

## TPU Backend

### TPU Architecture

TPUs are specialized machine learning accelerators that operate as sequential machines with very wide vector registers. Key characteristics:

- Sequential execution with wide SIMD operations
- Asynchronous background operations for memory transfers
- Grid processing in lexicographic order
- Memory reuse optimization for consecutive grid indices

### TPU Constraints

1. **Block Shape Requirements**:
   - Blocks must have rank ≥ 1
   - Last two dimensions must be divisible by 8 and 128 respectively
   - Or equal to the full array dimensions

2. **Supported Data Types**:
   - float32, bfloat16
   - int8, int16, int32
   - uint8, uint16, uint32
   - bool

3. **Control Flow**:
   - Limited support for `cond`, `fori_loop`, and `for_loop`
   - Loops are fully unrolled during compilation

### TPU Example

```python
@functools.partial(
    pl.pallas_call,
    out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
    grid=(4, 4),
    dimension_semantics=("parallel", "parallel")  # Enable multicore
)
def tpu_matmul_kernel(x_ref, y_ref, o_ref):
    # Each invocation handles a 128x128 block
    i, j = pl.program_id(0), pl.program_id(1)
    
    # Load blocks from VMEM
    x_block = x_ref[i * 128:(i + 1) * 128, :]
    y_block = y_ref[:, j * 128:(j + 1) * 128]
    
    # Compute and store result
    o_ref[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = jnp.dot(x_block, y_block)
```

## GPU Backends

### Triton Backend

The default GPU backend that lowers Pallas to Triton IR:

```python
import jax.experimental.pallas.triton as plgpu

@plgpu.triton_call(
    out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
    grid=(8,)
)
def triton_kernel(x_ref, y_ref, o_ref):
    idx = pl.program_id(0) * 128
    o_ref[idx:idx+128] = x_ref[idx:idx+128] + y_ref[idx:idx+128]
```

### Mosaic GPU Backend

Experimental backend providing lower-level control:

```python
import jax.experimental.pallas.mosaic_gpu as plgpu

@functools.partial(
    plgpu.kernel,
    out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
    grid=(2,)
)
def mosaic_kernel(x_ref, y_ref):
    # Thread corresponds to CUDA warpgroup
    block_slice = pl.ds(pl.program_id(0) * 128, 128)
    y_ref[block_slice] = x_ref[block_slice] + 1
```

### GPU Memory Spaces

```python
# Specify memory spaces
x_in_smem = pl.load(x_ref, indices, memory_space=plgpu.SMEM)
pl.store(o_ref, indices, value, memory_space=plgpu.GMEM)
```

### Synchronization

```python
# Barrier synchronization
barrier = plgpu.Barrier(num_threads=32)
barrier.wait()

# Ensure memory visibility
plgpu.commit_smem()
```

## Software Pipelining

Software pipelining overlaps memory transfers with computation to maximize hardware utilization:

### Basic Pipelining Example

```python
from jax.experimental.pallas import pipelines

@functools.partial(
    pl.pallas_call,
    out_shape=jax.ShapeDtypeStruct((1024, 1024), jnp.float32),
    grid=(8, 8),
    in_specs=[pl.BlockSpec((128, 128), lambda i, j: (i, j))],
    out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
    compiler_params=pipelines.PipelineParams(
        num_stages=3,
        prefetch=True
    )
)
def pipelined_kernel(x_ref, o_ref):
    # Kernel automatically benefits from pipelining
    o_ref[...] = jnp.exp(x_ref[...])
```

### Pipelining Benefits

1. **Latency Hiding**: Overlaps memory transfers with computation
2. **Throughput Improvement**: Keeps compute units busy
3. **Automatic Management**: Pallas handles buffer allocation and scheduling

### Platform-Specific Pipelining

- **TPU**: Supports double-buffering by default
- **GPU**: Can specify multiple pipeline stages via compiler parameters

## API Reference

### Core Functions

#### `pallas_call`

```python
jax.experimental.pallas.pallas_call(
    kernel,
    out_shape,
    *,
    grid_spec=None,
    grid=(),
    in_specs=NoBlockSpec,
    out_specs=NoBlockSpec,
    scratch_shapes=(),
    input_output_aliases={},
    debug=False,
    interpret=False,
    name=None,
    compiler_params=None,
    cost_estimate=None,
    backend=None
)
```

Invokes a Pallas kernel on inputs.

#### `program_id`

```python
jax.experimental.pallas.program_id(axis: int) -> jax.Array
```

Returns the kernel's position along the specified grid axis.

#### `num_programs`

```python
jax.experimental.pallas.num_programs(axis: int) -> jax.Array
```

Returns the grid size along the specified axis.

### Memory Operations

#### `load`

```python
jax.experimental.pallas.load(
    ref: Ref,
    idx: tuple[indexing expressions],
    *,
    memory_space: MemorySpace = None
) -> jax.Array
```

Loads data from a reference at the given indices.

#### `store`

```python
jax.experimental.pallas.store(
    ref: Ref,
    idx: tuple[indexing expressions],
    value: jax.Array,
    *,
    memory_space: MemorySpace = None
) -> None
```

Stores a value to a reference at the given indices.

### Atomic Operations

```python
# Atomic addition
pl.atomic_add(ref, idx, value)

# Compare and swap
old_value = pl.atomic_cas(ref, idx, expected, new_value)

# Other atomic operations
pl.atomic_max(ref, idx, value)
pl.atomic_min(ref, idx, value)
pl.atomic_and(ref, idx, value)
pl.atomic_or(ref, idx, value)
pl.atomic_xor(ref, idx, value)
```

### Classes

#### `BlockSpec`

```python
class BlockSpec:
    block_shape: tuple[int, ...]
    index_map: Callable[..., tuple[int, ...]]
    memory_space: Optional[MemorySpace] = None
```

Specifies how arrays are partitioned into blocks.

#### `GridSpec`

```python
class GridSpec:
    grid: tuple[int, ...]
    in_specs: Sequence[BlockSpec]
    out_specs: Union[BlockSpec, Sequence[BlockSpec]]
    scratch_specs: Sequence[ShapeDtypeStruct]
```

Encodes complete grid iteration specification.

### Utility Functions

#### `debug_print`

```python
pl.debug_print(fmt: str, *args)
```

Prints debug information from within kernels.

#### `when`

```python
pl.when(condition: bool):
    # Code executed only when condition is true
```

Conditional execution within kernels.

## Design Principles

### Core Philosophy

1. **"Pallas is just JAX, with some extensions"**: Maintains JAX's programming model while adding kernel-specific capabilities

2. **Unified Interface**: Single API that targets multiple backends (Triton for GPU, Mosaic for TPU)

3. **Memory Explicitness**: Requires explicit management of memory hierarchies

4. **Transformation Support**: Compatible with JAX transformations like `vmap` and `grad`

### Architecture Components

1. **Reference Types (`Ref`)**: More precise control over memory access than regular arrays

2. **Pallas-specific Primitives**: `load`, `store`, and atomic operations

3. **Backend Lowering**: 
   - GPU → Triton IR
   - TPU → Mosaic

### Development Features

1. **Emulation Mode**: Debug kernels on any XLA-supported platform

2. **Python Closures**: Support for higher-order functions in kernels

3. **NumPy Compatibility**: Familiar array programming interface

## Examples

### Matrix Multiplication with Activation

```python
def matmul_kernel(x_ref, y_ref, o_ref, *, activation):
    x = x_ref[...]
    y = y_ref[...]
    result = jnp.dot(x, y)
    
    if activation == "relu":
        result = jnp.maximum(result, 0)
    elif activation == "silu":
        result = result * jax.nn.sigmoid(result)
    
    o_ref[...] = result

@functools.partial(jax.jit, static_argnames=['activation'])
def matmul(x, y, *, activation=None):
    return pl.pallas_call(
        functools.partial(matmul_kernel, activation=activation),
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(2, 2),
        in_specs=[
            pl.BlockSpec((x.shape[0] // 2, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], y.shape[1] // 2), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec((x.shape[0] // 2, y.shape[1] // 2), lambda i, j: (i, j))
    )(x, y)
```

### Batched Operations with vmap

```python
# Define kernel for single matrix multiply
single_matmul = functools.partial(
    pl.pallas_call,
    matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32),
    grid=(2, 2),
    in_specs=[...],
    out_specs=...
)

# Batch over multiple matrix multiplies
batched_matmul = jax.vmap(single_matmul)

# Use with batched inputs
x = jnp.ones((10, 256, 256))  # Batch of 10 matrices
y = jnp.ones((10, 256, 256))
result = batched_matmul(x, y)  # Shape: (10, 256, 256)
```

### Reduction Operations

```python
def sum_reduction_kernel(x_ref, o_ref):
    # Read input block
    x = x_ref[...]
    
    # Perform reduction
    sum_val = jnp.sum(x, axis=-1, keepdims=True)
    
    # Store result
    o_ref[...] = sum_val

def parallel_sum(x):
    return pl.pallas_call(
        sum_reduction_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype),
        grid=(x.shape[0] // 128,),
        in_specs=pl.BlockSpec((128, x.shape[1]), lambda i: (i * 128, 0)),
        out_specs=pl.BlockSpec((128, 1), lambda i: (i * 128, 0))
    )(x)
```

### Custom Attention Mechanism

```python
def attention_kernel(q_ref, k_ref, v_ref, o_ref, *, block_size):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    
    # Compute attention scores
    scores = jnp.dot(q, k.T) / jnp.sqrt(k.shape[-1])
    
    # Apply softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Compute output
    o_ref[...] = jnp.dot(attn_weights, v)

def blocked_attention(q, k, v, block_size=64):
    seq_len, d_model = q.shape
    
    return pl.pallas_call(
        functools.partial(attention_kernel, block_size=block_size),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(seq_len // block_size,),
        in_specs=[
            pl.BlockSpec((block_size, d_model), lambda i: (i * block_size, 0)),
            pl.BlockSpec(k.shape, lambda i: (0, 0)),  # Full K matrix
            pl.BlockSpec(v.shape, lambda i: (0, 0)),  # Full V matrix
        ],
        out_specs=pl.BlockSpec((block_size, d_model), lambda i: (i * block_size, 0))
    )(q, k, v)
```

## Best Practices

### Performance Optimization

1. **Block Size Selection**:
   - Choose block sizes that fit in SRAM/shared memory
   - Align with hardware requirements (e.g., 8x128 for TPU)
   - Consider memory bandwidth vs compute ratio

2. **Memory Access Patterns**:
   - Minimize global memory accesses
   - Reuse data in faster memory spaces
   - Use coalesced memory access on GPU

3. **Grid Design**:
   - Balance parallelism with memory constraints
   - Use sequential dimensions on TPU for predictable access
   - Leverage parallel dimensions for multicore execution

### Debugging Tips

1. **Start Simple**: Begin with small examples and gradually increase complexity

2. **Use Interpret Mode**: `interpret=True` in `pallas_call` for debugging

3. **Print Debugging**: Use `pl.debug_print` to inspect values

4. **Verify Shapes**: Ensure block shapes and grid dimensions align correctly

### Common Pitfalls

1. **Incorrect Block Shapes**: Ensure TPU blocks follow 8x128 divisibility rules

2. **Memory Space Confusion**: Be explicit about which memory space you're using

3. **Grid Index Errors**: Carefully map grid indices to array indices

4. **Read-Only Inputs**: Remember input buffers should not be modified

## Platform-Specific Considerations

### TPU Best Practices

1. Use `dimension_semantics` for multicore parallelization
2. Leverage memory reuse by ordering grid iterations appropriately
3. Prefer bfloat16 for optimal performance
4. Minimize control flow complexity

### GPU Best Practices

1. Optimize for warp-level parallelism
2. Use shared memory for frequently accessed data
3. Consider memory coalescing for global accesses
4. Leverage tensor cores for matrix operations

## Future Directions

Pallas is actively developed with ongoing improvements:

1. **Enhanced Control Flow**: Better support for dynamic computation
2. **Automatic Optimization**: Improved automatic pipelining and scheduling
3. **Backend Expansion**: Support for additional accelerators
4. **Higher-Level Abstractions**: Simplified APIs for common patterns

## Additional Resources

- [Pallas Changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Triton Documentation](https://triton-lang.org/)
- [TPU Programming Guide](https://cloud.google.com/tpu/docs)

[source](https://jax.readthedocs.io/en/latest/pallas/index.html)