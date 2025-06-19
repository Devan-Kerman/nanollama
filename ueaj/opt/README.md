# Optimizer Management System

A flexible optimizer configuration system for JAX/Flax models with NumPy-style slicing syntax.

## Features

- **Path-based parameter matching**: Target specific parameters by their path in the model
- **Slice notation**: Use `:` to create separate optimizers per element
- **Batched grouping**: Use `opt.batch[...]` to group parameters together
- **Override semantics**: Later patterns override earlier ones
- **Function-based configuration**: Pass functions that configure based on module/parameter properties
- **Mixed precision support**: Different optimizers can use different precisions

## Basic Usage

```python
from ueaj import opt
import optax

# Create optimizer manager
opt_mgr = opt.OptimizerManager()

# Configure optimizers
opt_mgr[...] = optax.adam(1e-4)  # Default
opt_mgr['layers', :, 'attn'] = optax.adamw(5e-5)  # Per-layer attention
opt_mgr['layers', opt_mgr.batch[0:4], 'mlp'] = optax.adamw(1e-4)  # Batch layers 0-3

# Build optimizer
optimizer = opt_mgr.build(model)
opt_state = optimizer.init(params)
```

## Slicing Patterns

### Basic Path Matching
```python
opt_mgr['encoder'] = optax.adam(1e-3)
opt_mgr['decoder', 'output'] = optax.adamw(5e-4)
```

### List/Array Indexing
```python
# Individual elements
opt_mgr['layers', 0] = optax.sgd(1e-3)
opt_mgr['layers', 1] = optax.adam(1e-4)

# Slice notation (creates separate optimizers)
opt_mgr['layers', :] = optax.adam(1e-4)  # One optimizer per layer
opt_mgr['layers', 2:5] = optax.adamw(5e-5)  # Layers 2-4
```

### Batched Grouping
```python
# Group multiple layers with one optimizer
opt_mgr['layers', opt_mgr.batch[0:8]] = optax.adam(1e-4)
opt_mgr['layers', opt_mgr.batch[8:16]] = optax.adam(5e-5)

# Batch named parameters
opt_mgr[opt_mgr.batch[['q', 'k', 'v']]] = optax.adam(1e-4)
```

### Pattern Matching
```python
# List of names
opt_mgr[['encoder', 'decoder']] = optax.adam(1e-4)

# Nested paths
opt_mgr['layers', :, 'attn', 'qkv'] = optax.adamw(5e-5)

# Ellipsis matches everything
opt_mgr[...] = optax.adam(1e-4)  # Default fallback
```

## Advanced Features

### Function-based Configuration
```python
def configure_by_layer(modules, shape_dtype):
    """Configure based on module hierarchy."""
    for i, module in enumerate(modules):
        if hasattr(module, 'layer_idx'):
            lr = 1e-4 * (0.95 ** module.layer_idx)
            return optax.adamw(lr)
    return optax.adam(1e-4)

opt_mgr['layers', :] = configure_by_layer
```

### Override Behavior
```python
# Later assignments override earlier ones
opt_mgr[...] = optax.adam(1e-4)  # Default
opt_mgr['layers'] = optax.adamw(5e-5)  # Override for layers
opt_mgr['layers', 0] = optax.sgd(1e-3)  # Override for layer 0
```

## Integration with Training

```python
# Standard Flax/Optax workflow
@jax.jit
def train_step(params, opt_state, batch):
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

## Implementation Notes

- The system uses optax's `multi_transform` under the hood
- Parameters are matched based on their path in the model structure
- Slicing with `:` creates separate optimizer states per element
- Batching with `opt.batch[...]` creates a single shared optimizer state
- Function configurations receive the module hierarchy and shape/dtype info