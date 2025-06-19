# JAX Documentation Collection

This directory contains comprehensive documentation about JAX's distributed and parallel computing capabilities, gathered from the official JAX documentation.

## Contents

1. **[Pallas](./pallas.md)** - JAX's kernel language for writing custom GPU/TPU kernels
   - Core concepts and programming model
   - Memory hierarchy and optimization techniques
   - Examples and best practices

2. **[Distributed JAX](./distributed_jax.md)** - Overview of JAX's distributed computing capabilities
   - Multi-process programming
   - Sharding and distributed arrays
   - Parallel primitives (pmap, pjit, shard_map)

3. **[Multiprocess JAX](./multiprocess_jax.md)** - Detailed guide to multi-process JAX
   - Initialization and setup
   - Distributed data loading
   - Platform-specific configurations

4. **[Tensor Parallelism](./tensor_parallelism.md)** - SPMD and sharding in JAX
   - Automatic parallelization
   - Manual parallelism with shard_map
   - Practical examples

5. **[GSPMD and pjit](./gspmd_pjit.md)** - Automatic parallelization features
   - GSPMD (General Sparse Parallel Mesh Distribution)
   - Modern JAX parallelization with jax.jit
   - Migration from pmap

6. **[Sharding](./sharding.md)** - JAX sharding API reference
   - NamedSharding and PositionalSharding
   - Mesh utilities
   - Debugging and visualization

7. **[Collective Operations](./collective_operations.md)** - Communication primitives
   - All collective operations (psum, all_gather, etc.)
   - Integration with pmap and shard_map
   - Common parallel patterns

## Quick Start

For beginners, we recommend starting with:
1. [Distributed JAX](./distributed_jax.md) for an overview
2. [Tensor Parallelism](./tensor_parallelism.md) for understanding core concepts
3. [GSPMD and pjit](./gspmd_pjit.md) for modern JAX parallelization

## Sources

All documentation is sourced from:
- [JAX Official Documentation](https://jax.readthedocs.io/en/latest/)
- [JAX GitHub Repository](https://github.com/jax-ml/jax)

Each file contains specific source links at the end.