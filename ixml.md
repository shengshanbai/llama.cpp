# Iluvatar (天数智芯) AI 加速卡编译指南

本文档记录 llama.cpp 在天数智芯（Iluvatar）AI 加速卡上的编译方法和已知限制。

## 系统信息

### 硬件架构
- **GPU 架构**: ivcore11
- **最大线程块线程数**: 4096
- **SDK**: CoreX SDK 4.4.0

### 编译器
- **路径**: `/usr/local/corex/bin/`
- **CUDA 兼容编译器**: `clang` / `clang++` (Clang 18.1.8)
- **CUDA Toolkit**: 10.2.89 (兼容接口)
- **llc**: LLVM 静态编译器，将 LLVM IR 编译为 GPU 机器码

### 关头文件路径
- `/usr/local/corex/include/`
- 包含标准 CUDA 头文件和天数智芯专用头文件（如 `iluvatar_fp16.hpp`, `iluvatar_bf16.hpp`）

### 库文件路径
- `/usr/local/corex/lib64/`
- 包含 `libcublas.so`, `libcudart.so`, `libcuda.so` 等 CUDA 兼容库

## 编译状态

### ✅ CPU 后端编译成功

CPU 后端可正常编译并运行，支持 AVX/AVX2/SSE4.2/F16C/FMA/BMI2 指令集优化。

```bash
# 设置环境变量
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH

# 配置 CMake（仅 CPU 后端）
cmake -B build \
  -DCMAKE_C_COMPILER=/usr/local/corex/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/corex/bin/clang++ \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build --config Release -j 8
```

### ✅ CUDA 后端编译成功

经过源码修改，CUDA 后端现已可以成功编译。以下是所需的修改和注意事项。

## CUDA 后端编译方法

```bash
# 设置环境变量
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:$LD_LIBRARY_PATH

# 配置 CMake（启用 CUDA 后端）
cmake -B build \
  -DCMAKE_C_COMPILER=/usr/local/corex/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/corex/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/clang \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build --config Release -j 8
```

## 源码修改说明

为适配 CoreX SDK 的限制，以下源码文件已修改：

### 1. CMakeLists.txt (`ggml/src/ggml-cuda/CMakeLists.txt`)

**编译器检测和标志设置**：
```cmake
# 检测 Iluvatar 编译器
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" OR CMAKE_CUDA_COMPILER_ID STREQUAL "ILUVATAR")
    # Clang CUDA compiler (Iluvatar CoreX) - use ivcore language
    set(CUDA_FLAGS --cuda-path=/usr/local/corex)
    list(APPEND CUDA_FLAGS -x ivcore)
    list(APPEND CUDA_FLAGS -std=c++17)
    add_compile_definitions(GGML_USE_ILUVATAR __ILUVATAR__)
else()
    # NVIDIA nvcc compiler
    set(CUDA_FLAGS -use_fast_math -extended-lambda)
endif()
```

**排除问题文件**：
```cmake
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" OR CMAKE_CUDA_COMPILER_ID STREQUAL "ILUVATAR")
    # 排除导致 llc crash 的文件
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "fattn.*\\.cu$")  # Flash attention - CFG annotation 问题
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "mmq\\.cu$")      # Matrix mul quant - SGPR spill 问题
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "mmvq\\.cu$")     # Matrix mul vec quant - SGPR spill 问题
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "mmf\\.cu$")      # Matrix mul f - SGPR spill 问题
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "mmid\\.cu$")     # Matrix mul id - SGPR spill 问题
    list(FILTER GGML_SOURCES_CUDA EXCLUDE REGEX "mmv\\.cu$")      # Matrix mul vec - SGPR spill 问题
    # 添加 stub 实现
    list(APPEND GGML_SOURCES_CUDA "iluvatar-stubs.cu")
endif()
```

**Iluvatar 特定编译选项**：
```cmake
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" OR CMAKE_CUDA_COMPILER_ID STREQUAL "ILUVATAR")
    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fno-optimize-sibling-calls>")
    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-mllvm -iluvatar-function-calls=true>")
    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-mllvm --iluvatar-spill-sgpr-to-smem>")
endif()
```

### 2. common.cuh (`ggml/src/ggml-cuda/common.cuh`)

**修复 `no_device_code` 函数**：CoreX SDK 的 `[[noreturn]]` 属性与 `__builtin_unreachable()` 不兼容，创建简化版本：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
static __device__ void no_device_code(const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {
    printf("no device code. kernel requires higher GPU architecture than %d.\n", arch);
    printf("file: %s, line: %d, function: %s, architectures: %s\n", file_name, line, function_name, arch_list);
    while(1) {}  // Iluvatar: 使用无限循环替代 __builtin_unreachable()
}
#else
[[noreturn]]
static __device__ void no_device_code(...) {
    ...
    __builtin_unreachable();
}
#endif
```

### 3. convert.cu (`ggml/src/ggml-cuda/convert.cu`)

**创建 Iluvatar 专用 kernel**：简化 Q8_0 dequantize kernel 避免 CFG annotation 问题：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
// Iluvatar: 简化版本无循环/共享内存，避免 CFG annotation 问题
static __global__ void dequantize_block_q8_0_f16_iluvatar(const void * vx, half * y) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const block_q8_0 * block = (const block_q8_0 *)vx + i / QK8_0;
    const half d = block->d;
    const int8_t q = block->qs[i % QK8_0];
    y[i] = __hmul(d, __int2half_rn(q));
}
#endif
```

### 4. cumsum.cu (`ggml/src/ggml-cuda/cumsum.cu`)

**禁用 CUB 路径**：CoreX SDK 不支持 CUB 库的完整功能：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
    // Iluvatar: 禁用 CUB 路径，使用简单 fallback kernel
    cumsum_kernel<<<...>>>(...);
#else
    // NVIDIA: 使用 CUB 优化路径
    if (use_cub && ne00 >= 1024) {
        cumsum_cub_kernel<<<...>>>(...);
    }
#endif
```

**移除 `#pragma unroll`**：防止 CFG annotation 问题：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
    // Iluvatar CoreX SDK: no loop unrolling
    for (int i = 0; i < UNROLL_FACTOR; i++) { ... }
#else
#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) { ... }
#endif
```

### 5. softmax.cu (`ggml/src/ggml-cuda/softmax.cu`)

**禁用 cooperative_groups**：CoreX SDK 不提供 `cooperative_groups/reduce.h`：
```cpp
#if !defined(GGML_USE_ILUVATAR) && !defined(__ILUVATAR__)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif
```

**禁用 cooperative launch 路径**：
```cpp
#if !defined(GGML_USE_ILUVATAR) && !defined(__ILUVATAR__)
    if (supports_cooperative_launch && ...) {
        cudaLaunchCooperativeKernel(...);
    } else {
#endif
        soft_max_f32<false, 0, 0><<<...>>>(...);
#if !defined(GGML_USE_ILUVATAR) && !defined(__ILUVATAR__)
    }
#endif
```

### 6. solve_tri.cu (`ggml/src/ggml-cuda/solve_tri.cu`)

**禁用 fast kernel**：避免 SGPR spill 导致的 llc crash：
```cpp
#if !defined(GGML_USE_ILUVATAR) && !defined(__ILUVATAR__)
// Fast kernel causes llc crash on Iluvatar
template <int n_template, int k_template>
static __global__ void solve_tri_f32_fast(...) { ... }
static void solve_tri_f32_cuda(...) { ... }
#endif

#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
    // Iluvatar: always use cublas
    solve_tri_f32_cublas(ctx, ...);
#else
    if (n <= MAX_N_FAST && k <= MAX_K_FAST) {
        solve_tri_f32_cuda(...);
    } else {
        solve_tri_f32_cublas(...);
    }
#endif
```

### 7. ggml-cuda.cu (`ggml/src/ggml-cuda/ggml-cuda.cu`)

**修复 cudaStreamWaitEvent API**：CoreX SDK 需要 3 个参数（flags）：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)
    CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
#else
    CUDA_CHECK(cudaStreamWaitEvent(stream, event));
#endif
```

### 8. iluvatar-stubs.cu (`ggml/src/ggml-cuda/iluvatar-stubs.cu`)

**新增文件**：为排除的 kernel 提供 stub 实现，确保链接成功：
```cpp
#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)

// Flash attention stubs
void ggml_cuda_flash_attn_ext(...) { GGML_ABORT("Flash attention not supported on Iluvatar"); }
bool ggml_cuda_flash_attn_ext_supported(...) { return false; }

// MMVQ/MMID stubs
int get_mmvq_mmid_max_batch(...) { return 0; }
void ggml_cuda_mul_mat_vec_q(...) { GGML_ABORT("MMVQ not supported on Iluvatar"); }

// MMQ stubs
bool ggml_cuda_should_use_mmq(...) { return false; }
void ggml_cuda_mul_mat_q(...) { GGML_ABORT("MMQ not supported on Iluvatar"); }

// MMF stubs
bool ggml_cuda_should_use_mmf(...) { return false; }
void ggml_cuda_mul_mat_f(...) { GGML_ABORT("MMF not supported on Iluvatar"); }

// MMVF stubs
bool ggml_cuda_should_use_mmvf(...) { return false; }

#endif
```

## 已知限制

### 排除的 Kernel（使用 cuBLAS fallback）

以下优化的 CUDA kernel 因 CoreX SDK llc 编译器问题被排除：

| Kernel 类型 | 文件 | 问题 | Fallback |
|------------|------|------|----------|
| Flash Attention | fattn*.cu | CFG annotation 错误 | 标准 attention |
| Matrix Mul Quant | mmq.cu | SGPR spill crash | cuBLAS |
| Matrix Mul Vec Quant | mmvq.cu | SGPR spill crash | cuBLAS |
| Matrix Mul F | mmf.cu | SGPR spill crash | cuBLAS |
| Matrix Mul ID | mmid.cu | SGPR spill crash | cuBLAS |
| Matrix Mul Vec | mmv.cu | SGPR spill crash | cuBLAS |

### 功能影响

- **Flash Attention**: 不支持，使用标准 attention 实现
- **量化矩阵乘法**: 使用 cuBLAS 代替优化 kernel，性能可能下降
- **Loop Unroll**: 部分代码禁用 `#pragma unroll` 防止 CFG annotation 问题
- **Cooperative Groups**: 不支持 grid synchronization

### API 差异

CoreX SDK 与 NVIDIA CUDA 的 API 差异：
- `cudaStreamWaitEvent`: 必须提供 flags 参数（NVIDIA 可选）

## NCCL/ixCCL 支持

天数智芯 ixCCL 通信库接口兼容 NCCL v2.24.3，支持多卡大模型推理通信优化。

**文件位置**：
- 头文件: `/usr/local/corex/include/nccl.h`
- 库文件: `/usr/local/corex/lib64/libnccl.so` → `libnccl.so.3.0.0.2243` (约 195MB)

编译时启用 NCCL 支持：
```bash
cmake -B build \
  -DCMAKE_C_COMPILER=/usr/local/corex/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/corex/bin/clang++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/clang \
  -DCMAKE_CUDA_ARCHITECTURES=ivcore11 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=ON \
  -DNCCL_INCLUDE_DIR=/usr/local/corex/include \
  -DNCCL_LIBRARY=/usr/local/corex/lib64/libnccl.so \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j 8
```

验证 NCCL 链接：
```bash
ldd build/bin/libggml-cuda.so | grep nccl
# 输出: libnccl.so.2 => /usr/local/corex/lib64/libnccl.so.2
```

NCCL 支持现已可用，可用于多卡大模型推理的 GPU 间通信优化。

## 运行示例

### CPU 后端运行
```bash
./build/bin/llama-cli -m model.gguf -p "Hello" -n 100
```

### CUDA 后端运行
```bash
./build/bin/llama-cli -m model.gguf -p "Hello" -ngl 99
```

## 天数智芯编译注意事项

根据天数智芯文档：

1. **Tail Call 优化**: GPU 不支持 Tail Call 和 Sibling Call 优化，需关闭
2. **Function Call**: 支持 Function Call，但不支持 Indirect Call、可变参数列表和 Lib Call
3. **Device 函数**: 不需要 inline 的 device 函数可添加 `__attribute__((noinline))`

## 参考资源

- CoreX SDK 文档: `/usr/local/corex/`
- CUDA 头文件: `/usr/local/corex/include/cuda_runtime.h`
- LLVM llc 选项: `/usr/local/corex/bin/llc --help`
- 技术支持: 天数智芯官方渠道

## 更新日志

- 2026-04-19: 完成 CPU 后端编译测试
- 2026-04-19: 尝试天数智芯文档推荐的编译选项，CFG annotation 错误仍存在
- 2026-04-19: 确认问题为 CoreX SDK llc 后端 bug，需要天数智芯修复
- 2026-04-19: 通过排除问题 kernel 和创建 stub 实现，成功编译 CUDA 后端
- 2026-04-19: 修改 convert.cu、cumsum.cu、softmax.cu、solve_tri.cu 适配 Iluvatar
- 2026-04-19: 修复 cudaStreamWaitEvent API 差异（CoreX SDK 需 flags 参数）
- 2026-04-19: CUDA 后端编译成功，所有工具可正常构建
- 2026-04-19: 启用 NCCL/ixCCL 支持，成功链接 libnccl.so，支持多卡推理通信优化