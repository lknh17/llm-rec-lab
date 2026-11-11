// S3 第 5 课：CUDA 矩阵乘的 5 步演进。C = A @ B，全部行优先（row-major）。
//
//   A: [M, K]   B: [K, N]   C: [M, N]
//   v1 naive        FP32  每线程 1 个输出，threadIdx.x -> 行（相邻线程读 A 的不同行：不合并）
//   v2 coalesced    FP32  每线程 1 个输出，threadIdx.x -> 列（相邻线程读 B 的相邻列：合并）
//   v3 smem tiling  FP32  32x32 的 A/B 小块先搬进 shared memory，块内复用 32 次
//   v4 reg tiling   FP32  128x128 的 block tile，每线程算 8x8 个输出，A/B 的值复用在寄存器里
//   v5 wmma         FP16 输入、FP32 累加，nvcuda::wmma 16x16x16 走 Tensor Core
//
// 写法参考 Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"（kernel 1/2/3/5），
// 区别：这里每个版本都带边界检查，任意 M、N、K 都正确（v5 例外，要求 16 的倍数，见 v5 注释）。
// 下标都用 int：要求 M·K、K·N、M·N 都小于 2^31（4096² = 1.6e7，远在范围内）。
// 由 matmul_ext.py 用 torch.utils.cpp_extension.load_inline 编译；Mac 上不能编译运行。

#include <cstdint>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// --8<-- [start:v1_naive]
// v1：最直白的写法。一个线程算 C 的一个元素，block 是 32x32 = 1024 个线程。
// 启动：grid = (ceil(M/32), ceil(N/32))，block = (32, 32)
// __global__：这是一个 kernel，在 GPU 上由 grid × block 个线程各执行一遍函数体，返回值必须是 void
// const float* A：指向 A 首元素（显存里）的只读指针，A[i] 是第 i 个元素；二维下标要自己拍平成 row*K + k
__global__ void matmul_v1_naive_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
  // blockIdx / blockDim / threadIdx 是内建变量，各有 .x .y：本 block 在 grid 里的坐标 / block 的形状（这里 32x32）/
  // 本线程在 block 里的坐标。blockIdx.x 相当于 Triton 的 tl.program_id(0)；threadIdx 在 Triton 里没有对应物
  // 全局行号：第 blockIdx.x 个 block 管 32 行，块内第 threadIdx.x 行
  // 关键问题在这里：threadIdx.x 是 warp 内变化最快的维度（一个 warp = threadIdx.x 0..31，threadIdx.y 相同）
  // 所以一个 warp 的 32 个线程拿到的是 32 个不同的 row、同一个 col
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  // 边界检查：M、N 不是 32 的倍数时，最后一圈 block 里有线程落在矩阵外，直接不干活
  if (row < M && col < N) {
    // 累加器在寄存器里，FP32
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A[row][k] 的地址 = row·K + k：warp 里 32 个线程的 row 不同，地址相隔 K·4 字节
      //   -> 32 个线程各占一个 32 字节 sector，搬 1024 字节只用 128 字节（合并效率 12.5%）
      // B[k][col] 的地址 = k·N + col：warp 里 col 相同 -> 32 个线程读同一个数，广播，1 个 sector
      acc += A[row * K + k] * B[k * N + col];
    }
    // 写 C[row][col]：同样是 32 个不同的行，写也不合并
    C[row * N + col] = acc;
  }
}
// --8<-- [end:v1_naive]

// --8<-- [start:v2_coalesced]
// v2：和 v1 只差两行——把 threadIdx.x 映射到列。计算量、访存次数一模一样，搬运的 sector 少 6.6 倍。
// 启动：grid = (ceil(N/32), ceil(M/32))，block = (32, 32)
__global__ void matmul_v2_coalesced_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
  // 现在一个 warp 的 32 个线程：row 相同、col 连续
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A[row][k]：warp 内地址相同 -> 广播，1 个 sector
      // B[k][col]：warp 内地址连续 32 个 float = 128 字节 = 4 个 sector，完美合并
      // 每个 k，一个 warp 只搬 5 个 sector（v1 是 33 个）
      acc += A[row * K + k] * B[k * N + col];
    }
    // 写 C 也变成连续的 128 字节
    C[row * N + col] = acc;
  }
}
// --8<-- [end:v2_coalesced]

// --8<-- [start:v3_smem]
// --8<-- [start:v3_setup]
// v3：shared memory 分块。block 负责 C 的一个 32x32 小块，沿 K 每次搬 A 的 [32, 32] 和 B 的 [32, 32] 进 smem。
// 每个从全局内存读进来的数，在 smem 里被 32 个线程复用 -> 全局读次数降到 v2 的 1/16。
// 启动：grid = (ceil(N/32), ceil(M/32))，block = (32, 32)；smem = 2 × 32 × 32 × 4 = 8 KB
// constexpr：编译期常量，作用类似 Triton 的 tl.constexpr，数组大小必须用它
constexpr int V3_TILE = 32;

__global__ void matmul_v3_smem_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
  // __shared__：整个 block 共享的片上 SRAM，生命周期 = block，延迟约 30 个周期（HBM 是几百个）
  __shared__ float As[V3_TILE][V3_TILE];  // A 的 [32 行, 32 个 k]
  __shared__ float Bs[V3_TILE][V3_TILE];  // B 的 [32 个 k, 32 列]
  const int tx = threadIdx.x;  // 块内列，warp 内连续
  const int ty = threadIdx.y;  // 块内行
  const int row = blockIdx.y * V3_TILE + ty;
  const int col = blockIdx.x * V3_TILE + tx;
  float acc = 0.0f;
  // --8<-- [end:v3_setup]

  // --8<-- [start:v3_loop]
  for (int k0 = 0; k0 < K; k0 += V3_TILE) {
    // 第一步：1024 个线程合作搬两块，每线程搬 A、B 各一个数
    // A[row][k0+tx]：warp 内 tx 连续 -> 合并；越界（M 或 K 不整除）填 0，不影响点积结果
    // cond ? x : y 等于 Python 的 x if cond else y，作用同 Triton 的 tl.load(..., mask, other=0)
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
    // B[k0+ty][col]：同样 warp 内连续
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
    // 栅栏：等 block 内所有线程都写完 smem，才能开始读别人搬的数
    // 注意：所有线程都必须走到这里，所以上面不能对越界线程提前 return
    __syncthreads();

    // 第二步：在 smem 里做 32 次乘加
    for (int kk = 0; kk < V3_TILE; ++kk) {
      // As[ty][kk]：warp 内 ty 相同 -> 同一个地址，广播，无 bank conflict
      // Bs[kk][tx]：warp 内 tx 连续 -> 32 个连续 word 落在 32 个不同 bank，无冲突
      acc += As[ty][kk] * Bs[kk][tx];
    }
    // 第二个栅栏：等所有线程算完，才能在下一轮覆盖 As/Bs
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}
// --8<-- [end:v3_loop]
// --8<-- [end:v3_smem]

// --8<-- [start:v4_regtile]
// --8<-- [start:v4_header]
// v4：寄存器分块（Boehm kernel 5 "2D blocktiling"）。
// block tile [BM, BN] = [128, 128]，每次沿 K 走 BK = 8；256 个线程，每线程算 TM x TN = 8x8 = 64 个输出。
// 为什么快：v3 每做 1 次乘加要读 2 次 smem；这里每个 kk 读 8 + 8 个数进寄存器，做 64 次乘加，smem 读减少 4 倍，
// 而且 block tile 从 32 变成 128，全局读次数再降 4 倍。
// 启动：grid = (ceil(N/128), ceil(M/128))，block = 256；smem = (128·8 + 8·128) × 4 = 8 KB
constexpr int V4_BM = 128, V4_BN = 128, V4_BK = 8, V4_TM = 8, V4_TN = 8;
constexpr int V4_THREADS = (V4_BM / V4_TM) * (V4_BN / V4_TN);  // 16 × 16 = 256

// __launch_bounds__ 告诉编译器 block 最多 256 线程，它据此决定每线程能用多少寄存器
__global__ void __launch_bounds__(V4_THREADS) matmul_v4_regtile_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
// --8<-- [end:v4_header]
  // --8<-- [start:v4_setup]
  __shared__ float As[V4_BM * V4_BK];  // [128, 8]，行优先
  __shared__ float Bs[V4_BK * V4_BN];  // [8, 128]，行优先
  const int block_row = blockIdx.y * V4_BM;  // 本 block 的 C 小块左上角
  const int block_col = blockIdx.x * V4_BN;
  const int tid = threadIdx.x;  // 0..255，一维 block

  // 计算时的分工：16x16 个线程排成网格，线程 (thread_row, thread_col) 负责 C 小块里的 8x8 子块
  const int thread_col = tid % (V4_BN / V4_TN);  // 0..15
  const int thread_row = tid / (V4_BN / V4_TN);  // 0..15

  // 搬运时的分工（和计算分工无关）：A 小块 128x8 = 1024 个数，256 线程每人搬 4 个
  // 线程按 (a_row, a_col) 排成 32x8，一次搬 32 行，循环 4 次；warp 内 8 个一组读连续 32 字节，合并
  const int a_row = tid / V4_BK;  // 0..31
  const int a_col = tid % V4_BK;  // 0..7
  constexpr int A_ROW_STEP = V4_THREADS / V4_BK;  // 32
  // B 小块 8x128：线程排成 2x128，一次搬 2 行，循环 4 次；warp 内 32 个连续 float，合并
  const int b_row = tid / V4_BN;  // 0..1
  const int b_col = tid % V4_BN;  // 0..127
  constexpr int B_ROW_STEP = V4_THREADS / V4_BN;  // 2

  // 64 个 FP32 累加器 + 8 + 8 个操作数缓存，全部在寄存器里（下面的循环全展开，下标是编译期常数）
  // 这就是 v4 每线程要 100 多个寄存器的原因：occupancy 会因此降到 25% 左右（第 4 课的计算器）
  // float acc[64] = {0.0f}：每个线程自己的长度 64 的局部数组，全部初始化为 0（相当于 tl.zeros）
  float acc[V4_TM * V4_TN] = {0.0f};
  float reg_a[V4_TM];
  float reg_b[V4_TN];
  // --8<-- [end:v4_setup]

  // --8<-- [start:v4_mainloop]
  // --8<-- [start:v4_load]
  for (int k0 = 0; k0 < K; k0 += V4_BK) {
    // 搬 A 的 [128, 8] 小块：全局 (block_row + r, k0 + a_col)，越界填 0
    for (int r = a_row; r < V4_BM; r += A_ROW_STEP) {
      const int gr = block_row + r, gc = k0 + a_col;
      As[r * V4_BK + a_col] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // 搬 B 的 [8, 128] 小块：全局 (k0 + r, block_col + b_col)
    for (int r = b_row; r < V4_BK; r += B_ROW_STEP) {
      const int gr = k0 + r, gc = block_col + b_col;
      Bs[r * V4_BN + b_col] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();
    // --8<-- [end:v4_load]

    // --8<-- [start:v4_compute]
    // 外积累加：每个 kk 取 A 的一列 8 个数、B 的一行 8 个数，做 8x8 = 64 次乘加
    // #pragma unroll：让编译器把紧跟的循环完全展开成直线代码，数组下标都变成常数，acc/reg_a/reg_b 才能放进寄存器
#pragma unroll
    for (int kk = 0; kk < V4_BK; ++kk) {
      // A 子块第 kk 列：As[(thread_row·8 + i)·8 + kk]
      // warp 内只有 2 个不同的 thread_row，两个地址相差 64 个 word，落在同一个 bank -> 2 路冲突
#pragma unroll
      for (int i = 0; i < V4_TM; ++i) reg_a[i] = As[(thread_row * V4_TM + i) * V4_BK + kk];
      // B 子块第 kk 行：Bs[kk·128 + thread_col·8 + j]
      // warp 内 16 个不同的 thread_col，地址步长 8 个 word：thread_col 0/4/8/12 同一个 bank -> 按标量读是 4 路冲突
      // （编译器若合并成 128 位向量读，路数会变；以 ncu 的 bank conflict 指标为准，第 6 课）
#pragma unroll
      for (int j = 0; j < V4_TN; ++j) reg_b[j] = Bs[kk * V4_BN + thread_col * V4_TN + j];
      // 64 次 FFMA，全在寄存器里，不碰 smem
#pragma unroll
      for (int i = 0; i < V4_TM; ++i)
#pragma unroll
        for (int j = 0; j < V4_TN; ++j) acc[i * V4_TN + j] += reg_a[i] * reg_b[j];
    }
    __syncthreads();
  }
  // --8<-- [end:v4_compute]
  // --8<-- [end:v4_mainloop]

  // --8<-- [start:v4_store]
  // 写回 8x8 个结果；越界的不写
  // 这里每条 store 指令 warp 内地址步长 8 个 float（32 字节），不完美合并；Boehm 的 kernel 6 用 float4 向量写解决
#pragma unroll
  for (int i = 0; i < V4_TM; ++i) {
#pragma unroll
    for (int j = 0; j < V4_TN; ++j) {
      const int gr = block_row + thread_row * V4_TM + i;
      const int gc = block_col + thread_col * V4_TN + j;
      if (gr < M && gc < N) C[gr * N + gc] = acc[i * V4_TN + j];
    }
  }
  // --8<-- [end:v4_store]
}
// --8<-- [end:v4_regtile]

// --8<-- [start:v5_wmma]
// --8<-- [start:v5_setup]
// v5：Tensor Core。nvcuda::wmma 是 warp 级 API：一个 warp 的 32 个线程合作完成一次 16x16x16 的矩阵乘加，
// 输入 FP16、累加 FP32。编译到 Ampere/Hopper 上会变成若干条 mma.sync（m16n8k16）指令。
// 分工：block = 4 个 warp 排成 2x2，每个 warp 算 C 的 32x32（2x2 个 16x16 fragment），block tile 64x64。
// 维度要求：M、N、K 都是 16 的倍数（fragment 不能部分越界）；A、B 的起始地址 32 字节对齐（wrapper 里检查）。
// 简化：fragment 直接从全局内存加载（load_matrix_sync），没有 shared memory 分块和流水线，
// 所以它的算术强度只有约 16 FLOP/B，会是带宽瓶颈。真正的 Tensor Core GEMM（cuBLAS、CUTLASS）
// 会先用 cp.async / TMA 把大块搬进 smem，再从 smem 喂给 Tensor Core（本课最后一节）。
// 启动：grid = (ceil(N/64), ceil(M/64))，block = 128
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int V5_WARPS_M = 2, V5_WARPS_N = 2;  // block 内 warp 排布
constexpr int V5_FRAGS_M = 2, V5_FRAGS_N = 2;  // 每个 warp 负责的 fragment 排布
constexpr int V5_BM = V5_WARPS_M * V5_FRAGS_M * WM;  // 64
constexpr int V5_BN = V5_WARPS_N * V5_FRAGS_N * WN;  // 64

// half 是 CUDA 的 FP16 类型
__global__ void matmul_v5_wmma_kernel(int M, int N, int K, const half* A, const half* B, float* C) {
  // 之后的 wmma::xxx 就是 nvcuda::wmma::xxx
  using namespace nvcuda;
  // 第几个 warp（0..3），以及它在 2x2 排布里的位置；warpSize 是内建常量 32，block 是一维的 128 个线程
  // 线程 0-31 是 warp 0，32-63 是 warp 1，依此类推
  const int warp_id = threadIdx.x / warpSize;
  const int warp_m = warp_id / V5_WARPS_N;
  const int warp_n = warp_id % V5_WARPS_N;
  // 本 warp 负责的 32x32 子块左上角（全局坐标）
  const int warp_row = blockIdx.y * V5_BM + warp_m * V5_FRAGS_M * WM;
  const int warp_col = blockIdx.x * V5_BN + warp_n * V5_FRAGS_N * WN;
  // --8<-- [end:v5_setup]

  // --8<-- [start:v5_fragments]
  // fragment 是"分散在 32 个线程寄存器里的一个 16x16 小矩阵"，每个线程具体存哪几个元素由硬件决定，对程序员不透明
  // 累加器 2x2 个：每个 16x16 FP32 = 256 个数 / 32 线程 = 每线程 8 个寄存器，共 32 个
  // 模板参数 fragment<用途, M, N, K, 元素类型[, 布局]>：用途是 matrix_a（左乘数）/ matrix_b（右乘数）/ accumulator，
  // 16,16,16 是一次 MMA 的形状，row_major 表示 A、B 在内存里按行优先存放
  wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[V5_FRAGS_M][V5_FRAGS_N];
  wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[V5_FRAGS_M];
  wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[V5_FRAGS_N];
  // fill_fragment：把累加器清零，相当于 tl.zeros
#pragma unroll
  for (int i = 0; i < V5_FRAGS_M; ++i)
#pragma unroll
    for (int j = 0; j < V5_FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  // 本 warp 的每个 fragment 是否在矩阵内（M、N 是 16 的倍数，所以 fragment 要么全在、要么全不在）
  // 这些判断只依赖 warp 编号，warp 内 32 个线程结果相同；*_sync 函数要求整个 warp 一起执行，不能在 warp 内分叉
  bool row_ok[V5_FRAGS_M], col_ok[V5_FRAGS_N];
#pragma unroll
  for (int i = 0; i < V5_FRAGS_M; ++i) row_ok[i] = warp_row + i * WM < M;
#pragma unroll
  for (int j = 0; j < V5_FRAGS_N; ++j) col_ok[j] = warp_col + j * WN < N;
  // --8<-- [end:v5_fragments]

  // --8<-- [start:v5_mainloop]
  for (int k0 = 0; k0 < K; k0 += WK) {
    // 加载 A 的 16x16：起点 A[warp_row + i·16][k0]，ldm = K 是行与行之间相隔的元素数
#pragma unroll
    for (int i = 0; i < V5_FRAGS_M; ++i)
      if (row_ok[i]) wmma::load_matrix_sync(a_frag[i], A + (warp_row + i * WM) * K + k0, K);
    // 加载 B 的 16x16：起点 B[k0][warp_col + j·16]，ldm = N
#pragma unroll
    for (int j = 0; j < V5_FRAGS_N; ++j)
      if (col_ok[j]) wmma::load_matrix_sync(b_frag[j], B + k0 * N + warp_col + j * WN, N);
    // mma_sync(d, a, b, c) 计算 d = a·b + c；d、c 都传 acc 就是 acc += a·b，对应 Triton 的 acc += tl.dot(a, b)
    // 2x2 次 Tensor Core 乘加：acc[i][j] += a[i] · b[j]，每次 16·16·16·2 = 8192 FLOPs
    // A 的每个 fragment 用了 2 次、B 的用了 2 次：这就是 warp 级的"寄存器分块"
#pragma unroll
    for (int i = 0; i < V5_FRAGS_M; ++i)
#pragma unroll
      for (int j = 0; j < V5_FRAGS_N; ++j)
        if (row_ok[i] && col_ok[j]) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
  }
  // 写回 FP32 结果，ldm = N；store_matrix_sync 是 load 的反方向，mem_row_major 指定按行优先写
#pragma unroll
  for (int i = 0; i < V5_FRAGS_M; ++i)
#pragma unroll
    for (int j = 0; j < V5_FRAGS_N; ++j)
      if (row_ok[i] && col_ok[j])
        wmma::store_matrix_sync(C + (warp_row + i * WM) * N + warp_col + j * WN, acc[i][j], N, wmma::mem_row_major);
}
// --8<-- [end:v5_mainloop]
// --8<-- [end:v5_wmma]

// ---------------------------------------------------------------- 主机端封装：检查输入、算 grid、启动 kernel

static void check_inputs(const torch::Tensor& a, const torch::Tensor& b, at::ScalarType dtype) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "输入必须在 GPU 上");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "形状必须是 [M, K] @ [K, N]");
  TORCH_CHECK(a.scalar_type() == dtype && b.scalar_type() == dtype, "dtype 不对");
  // kernel 按行优先、无额外步长来算地址；转置过的视图必须先 .contiguous()
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "输入必须是连续的（行优先）");
}

// --8<-- [start:launch]
// 主机端函数（在 CPU 上执行）：收 torch 张量、返回 torch 张量，matmul_ext.py 把它暴露成 Python 函数
torch::Tensor matmul_v1(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b, at::kFloat);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  // 分配输出（同 torch.empty），a.options() 表示和 a 同 dtype、同设备；auto 让编译器自己推断类型
  auto c = torch::empty({M, N}, a.options());
  // dim3(x, y)：block 里 32x32 个线程；grid 的 x 管行方向、y 管列方向（这是 v1 的映射）
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  // <<<grid, block, 动态 smem 字节数, stream>>>：用 PyTorch 当前的 stream，才能和前后的 torch 算子正确排序
  // data_ptr<float>()：取张量首元素的 float* 指针，传给 kernel（Triton 里这一步是自动的）
  matmul_v1_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  // kernel 启动是异步的；这里只检查"启动参数是否合法"（比如资源超限），运行时错误要等同步时才暴露
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
// --8<-- [end:launch]

torch::Tensor matmul_v2(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b, at::kFloat);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  // v2 的映射：grid.x 管列方向
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  matmul_v2_coalesced_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor matmul_v3(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b, at::kFloat);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  dim3 block(V3_TILE, V3_TILE);
  dim3 grid(CEIL_DIV(N, V3_TILE), CEIL_DIV(M, V3_TILE));
  matmul_v3_smem_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor matmul_v4(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b, at::kFloat);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  dim3 grid(CEIL_DIV(N, V4_BN), CEIL_DIV(M, V4_BM));
  matmul_v4_regtile_kernel<<<grid, V4_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor matmul_v5(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b, at::kHalf);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "v5 要求 M、N、K 都是 16 的倍数");
  // load_matrix_sync 要求每个 fragment 的起始地址 256 位（32 字节）对齐；K、N 是 16 的倍数时偏移都是 32 字节的倍数
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 32 == 0 && reinterpret_cast<uintptr_t>(b.data_ptr()) % 32 == 0,
              "v5 要求输入起始地址 32 字节对齐");
  // 输出 FP32：Tensor Core 的累加精度直接写出，不再转回 FP16
  auto c = torch::empty({M, N}, a.options().dtype(at::kFloat));
  dim3 grid(CEIL_DIV(N, V5_BN), CEIL_DIV(M, V5_BM));
  matmul_v5_wmma_kernel<<<grid, V5_WARPS_M * V5_WARPS_N * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, reinterpret_cast<const half*>(a.data_ptr<at::Half>()), reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      c.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
