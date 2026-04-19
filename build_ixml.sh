#!/bin/bash
# 编译 llama.cpp for Iluvatar (天数智芯) AI 加速卡，支持 NCCL 多卡通信
# 参考: ixml.md

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Iluvatar 编译脚本 (支持 NCCL)${NC}"
echo -e "${GREEN}========================================${NC}"

# CoreX SDK 路径
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
COREX_BIN="${COREX_ROOT}/bin"
COREX_LIB="${COREX_ROOT}/lib64"
COREX_INCLUDE="${COREX_ROOT}/include"

# 检查 CoreX SDK 是否存在
if [ ! -d "${COREX_ROOT}" ]; then
    echo -e "${RED}错误: CoreX SDK 未找到于 ${COREX_ROOT}${NC}"
    echo "请设置 COREX_ROOT 环境变量或安装 CoreX SDK"
    exit 1
fi

if [ ! -f "${COREX_BIN}/clang" ]; then
    echo -e "${RED}错误: Clang 编译器未找到于 ${COREX_BIN}/clang${NC}"
    exit 1
fi

echo -e "${GREEN}[1/5] 设置环境变量${NC}"
export PATH="${COREX_BIN}:${PATH}"
export LD_LIBRARY_PATH="${COREX_LIB}:${LD_LIBRARY_PATH}"
echo "  PATH: ${COREX_BIN}"
echo "  LD_LIBRARY_PATH: ${COREX_LIB}"

# 并行编译任务数
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"

echo -e "${GREEN}[2/5] 检查 NCCL 支持${NC}"
NCCL_HEADER="${COREX_INCLUDE}/nccl.h"
NCCL_LIB="${COREX_LIB}/libnccl.so"

if [ -f "${NCCL_HEADER}" ] && [ -f "${NCCL_LIB}" ]; then
    echo -e "  ${GREEN}✓${NC} NCCL 头文件: ${NCCL_HEADER}"
    echo -e "  ${GREEN}✓${NC} NCCL 库文件: ${NCCL_LIB}"
    NCCL_CMAKE_OPTIONS=(
        -DGGML_CUDA_NCCL=ON
        -DNCCL_INCLUDE_DIR="${COREX_INCLUDE}"
        -DNCCL_LIBRARY="${NCCL_LIB}"
    )
else
    echo -e "  ${YELLOW}!${NC} NCCL 未找到，将不启用 NCCL 支持"
    NCCL_CMAKE_OPTIONS=()
fi

echo -e "${GREEN}[3/5] 配置 CMake${NC}"
echo "  编译器: ${COREX_BIN}/clang++"
echo "  架构: ivcore11"
echo "  构建类型: ${BUILD_TYPE}"
echo "  并行任务: ${JOBS}"
echo "  构建目录: ${BUILD_DIR}"

# 删除旧的构建目录（如果需要）
if [ "${CLEAN_BUILD}" = "1" ] && [ -d "${BUILD_DIR}" ]; then
    echo -e "${YELLOW}  清理旧构建目录...${NC}"
    rm -rf "${BUILD_DIR}"
fi

cmake -B "${BUILD_DIR}" \
    -DCMAKE_C_COMPILER="${COREX_BIN}/clang" \
    -DCMAKE_CXX_COMPILER="${COREX_BIN}/clang++" \
    -DCMAKE_CUDA_COMPILER="${COREX_BIN}/clang" \
    -DCMAKE_CUDA_ARCHITECTURES=ivcore11 \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_VMM=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "${NCCL_CMAKE_OPTIONS[@]}"

echo -e "${GREEN}[4/5] 编译${NC}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j "${JOBS}"

echo -e "${GREEN}[5/5] 验证编译结果${NC}"

# 检查主要可执行文件
BIN_DIR="${BUILD_DIR}/bin"
if [ -f "${BIN_DIR}/llama-cli" ]; then
    echo -e "  ${GREEN}✓${NC} llama-cli"
fi
if [ -f "${BIN_DIR}/llama-server" ]; then
    echo -e "  ${GREEN}✓${NC} llama-server"
fi
if [ -f "${BIN_DIR}/llama-bench" ]; then
    echo -e "  ${GREEN}✓${NC} llama-bench"
fi

# 检查 NCCL 链接
if [ -f "${NCCL_LIB}" ] && [ -f "${BIN_DIR}/libggml-cuda.so" ]; then
    echo ""
    echo -e "${GREEN}NCCL 链接验证:${NC}"
    ldd "${BIN_DIR}/libggml-cuda.so" 2>/dev/null | grep nccl || echo -e "  ${YELLOW}NCCL 未链接${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  编译完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "运行示例:"
echo "  # CPU 后端"
echo "  ${BIN_DIR}/llama-cli -m model.gguf -p \"Hello\" -n 100"
echo ""
echo "  # GPU 后端 (单卡)"
echo "  ${BIN_DIR}/llama-cli -m model.gguf -p \"Hello\" -ngl 99"
echo ""
echo "  # GPU 后端 (多卡)"
echo "  ${BIN_DIR}/llama-cli -m model.gguf -p \"Hello\" -ngl 99 -sm row"
echo ""