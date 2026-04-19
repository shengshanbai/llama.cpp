# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work. This project has strict policies regarding AI-generated contributions.

## Project Overview

llama.cpp is a C/C++ implementation for LLM inference with minimal dependencies. It provides:
- `llama` library (`include/llama.h`) - core inference API
- `ggml` tensor library (`ggml/include/ggml.h`) - underlying computation engine
- Multiple backend support (CPU, CUDA, Metal, Vulkan, HIP, SYCL, etc.)
- Tools: `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, quantize tools

## Build Commands

```bash
# Basic CPU build
cmake -B build
cmake --build build --config Release -j 8

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# CUDA support
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (MacOS, enabled by default)
cmake -B build  # Metal auto-enabled on macOS
cmake -B build -DGGML_METAL=OFF  # disable Metal

# Vulkan
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release

# Static library build
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release
```

## Running Tests

```bash
# Build tests (tests are built as part of main cmake build)
cmake --build build

# Run all tests with ctest
cd build && ctest --output-on-failure

# Run specific test
cd build && ctest -R test-name --output-on-failure

# Run single test binary directly
./build/bin/test-sampling
```

## Key Tools

Build output executables are in `build/bin/`:
- `llama-cli` - main CLI tool for inference
- `llama-server` - OpenAI-compatible HTTP server
- `llama-bench` - benchmark performance
- `llama-perplexity` - measure model quality metrics
- `llama-quantize` - quantize models to smaller sizes
- `llama-gguf-split` - split/merge GGUF files

## Architecture Overview

### Core Library Structure

```
src/
├── llama.cpp          # Main entry point, top-level API implementation
├── llama-*.cpp/h      # Modular components (adapter, batch, context, etc.)
└── models/            # Model-specific implementations (gemma.cpp, qwen.cpp, etc.)

ggml/
├── include/           # Public headers (ggml.h, ggml-backend.h, gguf.h)
└── src/               # Backend implementations
    ├── ggml-cpu/      # CPU backend
    ├── ggml-cuda/     # CUDA backend
    ├── ggml-metal/    # Metal backend (Apple)
    ├── ggml-vulkan/   # Vulkan backend
    └── ggml-hip/      # HIP backend (AMD)
```

### Key Concepts

- **Batch**: Input tokens for `llama_encode`/`llama_decode` - see `llama_batch` struct in `include/llama.h`
- **Context**: `llama_context` holds inference state, KV cache, and active sequences
- **Sampler**: Token sampling chain (`llama_sampler`) - see sampling API in `include/llama.h`
- **Memory/KV Cache**: `llama_memory_t` - manages key-value cache for attention
- **Model**: `llama_model` - loaded GGUF weights and metadata

### Server Architecture

`llama-server` (`tools/server/`) consists of:
- `server_context` - main inference state with all active slots
- `server_slot` - manages individual parallel inference requests
- `server_routes` - HTTP request routing middleware
- `server_queue`/`server_response` - thread-safe task/result queues

Server runs on a single thread; HTTP workers handle JSON parsing/tokenization.

## Coding Conventions

From `CONTRIBUTING.md`:

- Use `snake_case` for functions, variables, types
- Naming optimizes for longest common prefix: `number_small`, `number_big` (not `small_number`, `big_number`)
- Enum values: uppercase with prefix (e.g., `LLAMA_VOCAB_TYPE_SPM`)
- C/C++ filenames: lowercase with dashes (`llama-context.cpp`)
- Python filenames: lowercase with underscores
- 4 spaces indentation, brackets on same line
- Use sized integer types (`int32_t`) in public API
- Declare structs as `struct foo {}` (not `typedef struct foo {} foo`)
- Use `clang-format` (clang-tools v15+) for formatting new code
- Avoid third-party dependencies, templates, fancy STL constructs

### Naming Pattern

General pattern: `<class>_<method>` with `<method>` being `<action>_<noun>`:
- `llama_model_init()` - class: "llama_model", method: "init"
- `llama_sampler_chain_remove()` - class: "llama_sampler_chain", method: "remove"

### Matrix Multiplication Note

Tensors store data in row-major order. `ggml_mul_mat(ctx, A, B)` computes: C = B A^T (unconventional ordering).

## Running CI Locally

Before submitting PRs, run full CI:

```bash
mkdir tmp
# CPU-only
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Important Files to Reference

- `docs/build.md` - detailed build instructions for all backends
- `docs/development/HOWTO-add-model.md` - adding new model support
- `tools/server/README.md` - server user documentation
- `tools/server/README-dev.md` - server architecture for developers
- `CONTRIBUTING.md` - contribution guidelines and coding conventions
- `AGENTS.md` - AI usage policy (READ THIS FIRST)

## Model Testing

Test with small models available in the project:
```bash
# Download test model (used in CI)
# Stories15M is commonly used for quick testing
llama-cli -m stories15M-q4_0.gguf -p "Once upon a time"
```

## Quantization Types

Supported quantization types are defined in `enum llama_ftype` in `include/llama.h`. Common types:
- `Q4_0`, `Q4_1` - 4-bit quantization
- `Q5_0`, `Q5_1` - 5-bit quantization
- `Q8_0` - 8-bit quantization
- `IQ2_XXS`, `IQ3_XS`, `IQ4_NL` - improved quantization variants
- `BF16`, `F16` - 16-bit formats