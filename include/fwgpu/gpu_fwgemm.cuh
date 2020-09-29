#ifndef FWGPU_GPU_FWGEMM_CUH
#define FWGPU_GPU_FWGEMM_CUH

#include <limits>

namespace fwgpu {

template <typename T>
__global__ auto gpu_fwgemm_naive(
    int m,
    int n,
    int k,
    T const *__restrict__ A,
    int lda,
    T const *__restrict__ B,
    int ldb,
    T *__restrict__ dist,
    int lddist,
    int *__restrict__ parent,
    int parent_offset,
    bool do_epilogue_min = true,
    void *stream         = nullptr) -> void {
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = blockIdx.x * blockDim.x + threadIdx.x;

  int col_idx = ty;
  while (col_idx < n) {
    int row_idx = tx;
    while (row_idx < m) {
      // initialize accumulators
      auto runnign_min_dist = std::numeric_limits<T>::infinity();
      int running_parent    = 0; // this initialization value does not matter

      // FW main loop
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        // calculate the distance between col_idx->row_idx by going through k_idx
        T curr_dist = A[(k_idx * lda) + row_idx] + B[(col_idx * ldb) + k_idx];
        if (curr_dist < runnign_min_dist) {
          runnign_min_dist = curr_dist;
          running_parent   = k_idx + parent_offset;
        }
      }

      // store final output
      if (do_epilogue_min) {
        if (runnign_min_dist < dist[(col_idx * lddist) + row_idx]) {
          dist[(col_idx * lddist) + row_idx]   = runnign_min_dist;
          parent[(col_idx * lddist) + row_idx] = running_parent;
        }
        else {
          dist[(col_idx * lddist) + row_idx]   = runnign_min_dist;
          parent[(col_idx * lddist) + row_idx] = running_parent;
        }
      }

      row_idx += gridDim.x * blockDim.x;
    }
    col_idx += gridDim.y * blockDim.y;
  }
}

} // namespace fwgpu

#endif // FWGPU_GPU_FWGEMM_HPP
