#include "benchmark/benchmark.h"
#include "ubench.cuh"

static void uBM_FFMA_8x8x1(benchmark::State &state) {
  auto itrs   = 1280;
  using Shape = cutlass::gemm::GemmShape<8, 8, 8>;
  fwgpu::internal::UBench<
      Shape,                               //
      float, cutlass::layout::ColumnMajor, //
      float, cutlass::layout::ColumnMajor, //
      float, cutlass::layout::ColumnMajor, //
      cutlass::arch::OpMultiplyAdd>        //
      bench;
  for (auto _ : state) {
    auto millis = bench.run(itrs);
    state.SetIterationTime(millis / 1000.0f);
  }

  double flops_per_itr = 2 * Shape::kM * Shape::kN * Shape::kK * itrs;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(uBM_FFMA_8x8x1)->UseManualTime();

// TEST(SM50_Sgemm_thread, col_row_4x4x2) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<4, 4, 2>, float, cutlass::layout::ColumnMajor, float,
//       cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Sgemm_thread, row_col_4x4x2) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<4, 4, 2>, float, cutlass::layout::RowMajor, float,
//       cutlass::layout::ColumnMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Sgemm_thread, col_row_4x5x3) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<4, 5, 3>, float, cutlass::layout::ColumnMajor, float,
//       cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }
// TEST(SM50_Sgemm_thread, col_row) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, float, cutlass::layout::ColumnMajor, float,
//       cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Sgemm_thread, row_col) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, float, cutlass::layout::RowMajor, float,
//       cutlass::layout::ColumnMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Sgemm_thread, col_col) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, float, cutlass::layout::ColumnMajor, float,
//       cutlass::layout::ColumnMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Sgemm_thread, row_row) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, float, cutlass::layout::RowMajor, float,
//       cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor>()
//       .run();
// }

// /////////////////////////////////////////////////////////////////////////////////////////////////

// TEST(SM50_Dgemm_thread, col_row) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, double, cutlass::layout::ColumnMajor, double,
//       cutlass::layout::RowMajor, double, cutlass::layout::ColumnMajor>()
//       .run();
// }

// TEST(SM50_Dgemm_thread, row_col) {
//   test::gemm::thread::Testbed<
//       cutlass::gemm::GemmShape<8, 8, 1>, double, cutlass::layout::RowMajor, double,
//       cutlass::layout::ColumnMajor, double, cutlass::layout::ColumnMajor>()
//       .run();
// }
