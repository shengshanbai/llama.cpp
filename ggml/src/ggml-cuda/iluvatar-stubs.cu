#include "common.cuh"
#include "ggml.h"

// Iluvatar stub implementations for excluded matrix multiplication kernels
// These kernels (mmq, mmvq, mmf, mmv, mmid, fattn) were excluded due to
// llc compiler issues with SGPR spill and CFG annotation errors.

#if defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)

// Flash attention stubs - disable for Iluvatar
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Flash attention disabled for Iluvatar - use fallback
    GGML_ABORT("Flash attention not supported on Iluvatar");
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    // Flash attention not supported on Iluvatar
    return false;
}

// MMVQ/MMID stubs - disable for Iluvatar, use cuBLAS fallback
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // Return 0 to disable MMVQ for mul_mat_id on Iluvatar
    return 0;
}

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMVQ not supported on Iluvatar");
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMVQ not supported on Iluvatar");
}

// MMQ stubs - disable for Iluvatar
bool ggml_cuda_should_use_mmq(ggml_type type, int cc, long ne01, long ne11) {
    // Disable MMQ for Iluvatar
    return false;
}

void ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMQ not supported on Iluvatar");
}

void ggml_cuda_op_mul_mat_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMQ not supported on Iluvatar");
}

// MMF stubs - disable for Iluvatar
bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size,
    const int64_t * scr0_ne, const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    // Disable MMF for Iluvatar
    return false;
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMF not supported on Iluvatar");
}

void ggml_cuda_op_mul_mat_vec_f(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMF not supported on Iluvatar");
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    // Disabled for Iluvatar - caller should use cuBLAS fallback
    GGML_ABORT("MMF not supported on Iluvatar");
}

// MMVF stubs - disable for Iluvatar
bool ggml_cuda_should_use_mmvf(ggml_type type, int cc, const long * ne, const size_t * nb, long nrows_dst) {
    // Disable MMVF for Iluvatar
    return false;
}

#endif // defined(GGML_USE_ILUVATAR) || defined(__ILUVATAR__)