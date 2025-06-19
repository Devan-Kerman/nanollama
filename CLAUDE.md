# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NanoLLaMA** is a JAX/Flax-based transformer implementation that bridges experimental numerical precision research with production-ready LLaMA model inference. The project uses a custom Einstein summation operation called "ueajsum" that supports variance tracking, mixed precision, and parametrized tensor operations. Originally a research project exploring efficient and numerically stable transformer architectures, it now includes full support for loading and running pre-trained LLaMA models from HuggingFace.

## Key Architecture Components

### UeajSum System
- **Core Module**: `ueaj/model/ueajsum/ueajsum.py` - Main `Ueajsum` class that handles parametrized Einstein summations
- **Configuration**: `ueaj/model/ueajsum/config.py` - Contains `UeajsumConfig`, `ArgumentConfig`, and `ParamConfig` classes for specifying tensor operations with variance tracking and precision controls
- **Parser**: `ueaj/model/ueajsum/parser.py` - Parses string expressions like `"ij,*jk->ik"` into `UeajsumConfig` objects
- **Variance Tracking**: `ueaj/model/ueajsum/varsum.py` - Implements `var_einsum` function that rescales outputs based on input variances
- **Mixed Precision**: `ueaj/model/ueajsum/mixsum.py` - Implements `mixsum` with custom VJP for per-tensor gradient precision

### Expression Syntax
- `*` prefix: marks parameters (e.g., `*w=jk` creates a parameter named `w` with shape `jk`)
- `&` prefix: marks LoRA parameters
- `+` separates terms in summations
- `=` assigns names to tensors (e.g., `w=jk`)
- Standard einsum notation for shapes

### Model Architecture

#### Transformer Layers (`ueaj/model/layer.py`)
- **TransformerLayer**: Combines attention and MLP blocks with residual connections
- **TransformerLayerConfig**: Configuration for layer components with support for:
  - Separate attention and MLP normalization configs
  - Gated MLP option (GMLP)
  - Flexible configuration chaining

#### Attention Mechanisms (`ueaj/model/attention/`)
- **SoftmaxAttention** (`soft_attn.py`): Standard scaled dot-product attention with:
  - Support for fused attention operations via Transformer Engine
  - RoPE (Rotary Position Embeddings) support with custom position IDs
  - Sliding window attention
  - Mixed precision control for K, Q, V, and O projections
  - GPU-optimized fused kernels and CPU fallback
  - KV caching removed for simplified implementation

- **SimpleAttention** (`simple_attn.py`): CPU-friendly attention implementation:
  - Inherits from SoftmaxAttention but without transformer engine dependencies
  - Uses standard JAX operations (jnp.einsum, jax.nn.softmax)
  - Explicitly casts to bfloat16 for computation
  - Useful for debugging and CPU inference

- **TransNormer** (`norm_attn.py`): Novel linear attention variant that:
  - Implements attention without softmax using direct K-V outer product accumulation
  - Features scan-based causal implementation for efficient autoregressive generation
  - Uses the expression: `"bnd,bmz,*zhk,*dhik,nm,*dhv->bmhiv"`
  - Includes RMSNorm after attention computation

#### MLP Components (`ueaj/model/mlp.py`)
- **MLP**: Standard feedforward network with up/down projections
  - Configurable activation functions (default: swish)
  - Support for mixed precision gradients via `invoke_fp32_backprop` method
- **GMLP**: Gated MLP variant with fused gate/up projections
  - Special handling for low-precision dtypes with scaling
  - Enhanced numerical stability for low-precision formats

#### Normalization (`ueaj/model/rmsnorm.py`)
- RMSNorm implementation with configurable scaling methods:
  - "standard": scale = sqrt(d)
  - "expected": scale = d (matches LLaMA 3 implementation)
  - "none": scale = 1
- Note: Default epsilon is 1e-6 (LLaMA 3 uses 1e-5)

#### Position Embeddings (`ueaj/model/attention/pe/`)
- RoPE (Rotary Position Embeddings) implementation with configurable theta
- Enhanced to handle multiple tensor shapes (2D, 3D, 4D, 5D inputs)
- Improved broadcasting for tensors with arbitrary extra dimensions

#### LLaMA Model (`ueaj/model/llama.py`)
- **LlamaModel**: Complete implementation supporting pre-trained LLaMA models
  - Supports all LLaMA model sizes (1B, 8B, 70B, 405B)
  - Loading from HuggingFace model IDs or local paths
  - Configurable dtypes for parameters and computation
  - KV cache support for efficient autoregressive generation
  - Tied/untied embedding options
  - Integration with transformer engine sequence descriptors
- **LlamaConfig**: Configuration dataclass with:
  - Automatic loading from HuggingFace config.json
  - Support for grouped query attention (GQA)
  - Configurable RoPE theta values

#### Weight Loading (`ueaj/model/weight_loader.py`)
- **WeightMapper**: Clean regex-based weight mapping system
  - Maps HuggingFace weight names to model attributes
  - Handles complex tensor reshaping for:
    - Attention projections (Q, K, V, O) with GQA support
    - MLP projections (gate, up, down) for gated architectures
    - Layer normalization weights
    - Embeddings and output projections
  - Type-aware weight conversion to configured dtypes
  - Support for safetensors format

### Utility Components

#### Tensor Scaling (`ueaj/utils/tensor_scaling.py`)
- Precision-aware parameter update algorithm using binary search
- Addresses numerical precision limitations in low-precision formats (e.g., FP8)
- Ensures target learning rate is achieved despite quantization effects
- **Note**: Moved from root directory to `ueaj/utils/` for better organization

#### Testing Infrastructure
- `test/test_ueajsum.py`: Tests for the core ueajsum functionality
- `test/test_mlp.py`: Tests for MLP gradient dtype behavior
- Focuses on verifying mixed precision gradient flow
- `test_cache/`: Directory for caching downloaded models during testing

### Text Generation (`sample_llama.py`)
- Command-line tool for text generation using LLaMA models
- **Sampling Methods**:
  - Temperature-based sampling
  - Top-k filtering
  - Top-p (nucleus) sampling
  - Min-p filtering (dynamic probability threshold)
  - Greedy decoding (temperature=0)
- **Features**:
  - Automatic model downloading from HuggingFace
  - KV caching for faster generation
  - Early stopping on EOS token
  - Configurable precision modes
  - Position-aware generation with cache

### Dependencies
- JAX with CUDA support (`jax[cuda12]`)
- Flax for neural network modules
- Datasets library
- Transformer Engine for JAX (provides fused attention kernels)

### Project Structure

```
nanollama/
├── jax_docs/              # JAX documentation reference
├── ueaj/                  # Main package
│   ├── data/              # Data utilities
│   ├── llama/             # LLaMA-specific code (empty)
│   ├── model/             # Model implementations
│   │   ├── attention/     # Attention mechanisms
│   │   │   ├── pe/        # Position embeddings
│   │   │   ├── norm_attn.py
│   │   │   ├── simple_attn.py
│   │   │   └── soft_attn.py
│   │   ├── ueajsum/       # Custom einsum implementation
│   │   ├── layer.py       # Transformer layers
│   │   ├── llama.py       # LLaMA model
│   │   ├── mlp.py         # MLP components
│   │   ├── rmsnorm.py     # Normalization
│   │   └── weight_loader.py
│   ├── models/            # Additional models (empty)
│   └── utils/             # Utility functions
├── test/                  # Test suite
├── test_cache/            # Model download cache
├── weights/               # Model weights
│   ├── llama-3.2-1b/      # Full LLaMA 3.2 1B model
│   └── mock_llama/        # Mock weights for testing
├── sample_llama.py        # Text generation script
├── requirements.txt       # Dependencies
├── run_python.sh          # Python execution wrapper
├── CLAUDE.md              # This file
└── BACKLOG.md             # Development roadmap
```

## Common Commands

**IMPORTANT**: Always use the `run_python.sh` script to run Python commands to ensure the correct virtual environment is used.

**Note**: The script now sets `JAX_COMPILATION_CACHE_DIR` for caching JAX compilations.

### Running Python Scripts
```bash
./run_python.sh script.py                     # Run a Python script
./run_python.sh -m module                     # Run a Python module
./run_python.sh -c "print('Hello')"          # Run Python command
```

### Testing
```bash
./run_python.sh -m pytest test/test_ueajsum.py                    # Run all ueajsum tests
./run_python.sh -m pytest test/test_ueajsum.py::test_simple_matmul  # Run specific test
./run_python.sh -m pytest test/test_mlp.py                       # Run MLP tests
./run_python.sh -m pytest test/                                  # Run all tests
```

### Development
```bash
./run_python.sh -c "from ueaj.model.ueajsum import *; print('Import successful')"  # Quick import test
./run_python.sh ueaj/model/attention/norm_attn.py      # Test TransNormer implementation
./run_python.sh ueaj/model/attention/soft_attn.py      # Test SoftmaxAttention
```

### Text Generation
```bash
# Basic text generation
./run_python.sh sample_llama.py --prompt "Once upon a time" --max-new-tokens 100

# With specific model
./run_python.sh sample_llama.py --model meta-llama/Llama-3.2-1B --temperature 0.8

# With different sampling methods
./run_python.sh sample_llama.py --prompt "The future of AI" --top-k 50 --top-p 0.9

# Greedy decoding
./run_python.sh sample_llama.py --prompt "Complete this" --temperature 0
```

## Key Development Patterns

### Configuration Chaining
The system uses method chaining for configuration:
```python
config = parse("ij,*jk->ik").fp8_params().unit().bf16_grads()
```

### Variance Tracking
- Set `variance=None` to disable scaling
- Use `.unit()` to set all variances to 1.0 for unit variance initialization
- Variance affects both initialization and forward/backward pass scaling

### Mixed Precision
- `.fp8_params()` - sets parameters to FP8
- `.bf16_grads()` - sets gradients to bfloat16
- `.fp32_grads()` - sets gradients to FP32
- Separate control over parameter dtype and gradient dtype

### Attention Implementation Notes
- The project now includes three attention implementations:
  - **SoftmaxAttention**: GPU-optimized with transformer engine support
  - **SimpleAttention**: CPU-friendly fallback without dependencies
  - **TransNormer**: Novel linear attention variant
- Attention modules use ueajsum for flexible einsum operations with automatic mixed precision
- Fused kernels are preferred on GPU for efficiency
- KV caching is supported in the LLaMA model for efficient autoregressive generation

## Current Development Status

The project has evolved from experimental research to include production-ready LLaMA inference:

1. **Completed Features**:
   - Full LLaMA model implementation with HuggingFace compatibility
   - Weight loading from pre-trained models
   - Text generation with advanced sampling methods
   - Three attention implementations for different use cases
   - Mixed precision support throughout the model

2. **Original Research Tracks** (from BACKLOG.md):
   - **Ether Sampling**: Integration with Gemma model for inference
   - **Nano Llama**: Core model development including:
     - Completing ueajsum layer implementation
     - Comprehensive test coverage
     - Baseline transformer model
     - Training scripts with tensor statistics logging
   - **Einstein Project**: Advanced features including:
     - Better scan operators
     - Manual backpropagation for layer looping
     - Momentum-based self-prediction

## Environment Configuration

- venv is in ~/venvs/jax-packages
- Supports both GPU (with fused kernels) and CPU execution
- Uses JAX's default backend detection for platform-specific optimizations
- JAX compilation cache configured for faster subsequent runs

## Security Note

**WARNING**: The `sample_llama.py` script contains a hardcoded HuggingFace token. This should be replaced with environment variable usage for production deployments.