#ifndef FWGPU_BENCH_UBENCH
#define FWGPU_BENCH_UBENCH

#include "fwgpu/Matrix.hpp"
#include "fwgpu/internal/utils.cuh"
#include "fwgpu/utils.hpp"

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/layout/matrix.h"

#include <iostream>
#include <tuple>
#include <type_traits>

namespace fwgpu {
namespace internal {

/// Thread-level matrix multiply-accumulate
template <typename Mma>
__global__ void kernel(
    typename Mma::ElementC *D,
    typename Mma::ElementA const *A,
    typename Mma::ElementB const *B,
    typename Mma::ElementC const *C,
    int itrs) {
  auto ptr_D
      = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> *>(D);
  auto ptr_A
      = reinterpret_cast<cutlass::Array<typename Mma::ElementA, Mma::Shape::kMK> const *>(
          A);
  auto ptr_B
      = reinterpret_cast<cutlass::Array<typename Mma::ElementB, Mma::Shape::kKN> const *>(
          B);
  auto ptr_C
      = reinterpret_cast<cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> const *>(
          C);

  Mma mma;

  auto a = *ptr_A;
  auto b = *ptr_B;
  auto c = *ptr_C;

  cutlass::Array<typename Mma::ElementC, Mma::Shape::kMN> d;

#pragma unroll 1
  for (int i = 0; i < itrs; ++i) {
    mma(d, a, b, c);
  }

  *ptr_D = d;
}

// Structure to compute the matrix product
template <
    // Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape,
    // Data type of A elements
    typename ElementA,
    // Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    // Data type of B elements
    typename ElementB,
    // Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    // Element type of C matrix
    typename ElementC,
    // Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    // Thread level SemiRing MMA operator type
    typename SemiRingOp>
struct UBench {
  /// Thread-level matrix multiply-accumulate operator
  using Mma = cutlass::gemm::thread::Mma<
      Shape,    //
      ElementA, //
      LayoutA,  //
      ElementB, //
      LayoutB,  //
      ElementC, //
      LayoutC,  //
      SemiRingOp>;

  using FwgpuLayoutA = typename std::conditional<
      std::is_same<LayoutA, cutlass::layout::ColumnMajor>::value,
      fwgpu::ColumnMajor,
      fwgpu::RowMajor>::type;

  using FwgpuLayoutB = typename std::conditional<
      std::is_same<LayoutB, cutlass::layout::ColumnMajor>::value,
      fwgpu::ColumnMajor,
      fwgpu::RowMajor>::type;

  using FwgpuLayoutC = typename std::conditional<
      std::is_same<LayoutC, cutlass::layout::ColumnMajor>::value,
      fwgpu::ColumnMajor,
      fwgpu::RowMajor>::type;

  fwgpu::Matrix<ElementA, FwgpuLayoutA> matrix_A;
  fwgpu::Matrix<ElementB, FwgpuLayoutB> matrix_B;
  fwgpu::Matrix<ElementC, FwgpuLayoutC> matrix_C;
  fwgpu::Matrix<ElementC, FwgpuLayoutC> matrix_D;
  cudaEvent_t m_start_evnt;
  cudaEvent_t m_end_evnt;
  std::tuple<ElementA *, ElementB *, ElementC *, ElementC *> m_dptrs;

  // Default constructor inits all data to keep kernel timinig pure.
  UBench()
      : matrix_A(Shape::kM, Shape::kK, 1, ElementA(1), ElementA(100))
      , matrix_B(Shape::kK, Shape::kN, 2, ElementB(1), ElementB(100))
      , matrix_C(Shape::kM, Shape::kN, 3, ElementC(1), ElementC(100))
      , matrix_D(Shape::kM, Shape::kN, ElementC(0))

  {
    cudaEventCreate(&m_start_evnt);
    cudaEventCreate(&m_end_evnt);
    m_dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(
        matrix_A, matrix_B, matrix_C, matrix_D);
  }

  ~UBench() {
    cudaEventDestroy(m_start_evnt);
    cudaEventDestroy(m_end_evnt);
    fwgpu::internal::dealloc_device_gemm_mats(m_dptrs);
  }

  // Runs the benchmark and returns the duration as a float
  auto run(int itrs = 1) -> float {
    // std::cout << matrix_A << std::endl;
    // std::cout << matrix_B << std::endl;
    // std::cout << matrix_C << std::endl;
    // std::cout << matrix_D << std::endl;
    // launch and time kernel
    float duration = 0.0f;
    cudaEventRecord(m_start_evnt);
    kernel<Mma><<<32, 32>>>(
        std::get<3>(m_dptrs), // d
        std::get<0>(m_dptrs), // a
        std::get<1>(m_dptrs), // b
        std::get<2>(m_dptrs), // c
        itrs);
    cudaEventRecord(m_end_evnt);
    cudaEventSynchronize(m_end_evnt);
    cudaEventElapsedTime(&duration, m_start_evnt, m_end_evnt);
    fwgpu::memcpy_d2h(matrix_D.get_buf(), std::get<3>(m_dptrs), matrix_D.bytesize());

    // std::cout << matrix_A << std::endl;
    // std::cout << matrix_B << std::endl;
    // std::cout << matrix_C << std::endl;
    // std::cout << matrix_D << std::endl;

    return duration;
  }
};

} // namespace internal
} // namespace fwgpu

#endif
