#include <stdio.h>
#include <vector>
#include <algorithm>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#define BASE_THREAD_NUM 256
#define TILE 4
typedef float scalar_t; // scalar_t 可能是 float 或者 double 等其他类型
const uint32_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray
{
  scalar_t *ptr;
  uint32_t size;
  CudaArray(uint32_t size) : size(size)
  {
    cudaError_t ret = cudaMalloc((void **)&ptr, ELEM_SIZE * size);
    if (ret != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(ret));
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
};

struct CudaDims
{
  dim3 grid, block;
};

CudaDims CudaOneDim(uint32_t size)
{
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dims;
  uint32_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;

  dims.grid = dim3(num_blocks, 1, 1);
  dims.block = dim3(BASE_THREAD_NUM, 1, 1);

  return dims;
}

// cuda kernel function 不支持 std 容器，因此只能将容器转成普通数组
// 后面需要用到 shape strides 数组
#define MAX_VEC_SIZE 8
struct CudaVec
{
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x)
{
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE)
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (uint32_t i = 0; i < x.size(); i++)
  {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val)
{
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ uint32_t index_transform(uint32_t index, CudaVec shape, CudaVec strides, uint32_t offset)
{
  /**
   * index = i * shape[1] * shape[2] + j * shape[2] + k
   * k = index % shape[2] / 1
   * j = index % (shape[1] * shape[2]) / shape[2]
   * i = index % (shape[0] * shape[1] * shape[2]) / shape[1] * shape[2]
   * 利用公式 (a % (kd)) / d = (a / d) % k 变形下
   * k = index % shape[2]
   * j = (index / shape[2]) % shape[1]
   * i = (index / (shape[1] * shape[2])) % shape[0]
   */
  // 从 index 推出 for 循环遍历的 [i,j,k...]
  // uint32_t idxs[MAX_VEC_SIZE];
  // uint32_t cur_size, pre_size = 1;
  // for (int32_t i = shape.size - 1; i >= 0; i--) // 从最内层开始
  // {
  //   cur_size = pre_size * shape.data[i];
  //   idxs[i] = index % cur_size / pre_size;
  //   pre_size = cur_size;
  // }

  // // 计算最后索引
  // uint32_t comp_idx = offset;
  // for (uint32_t i = 0; i < shape.size; i++)
  // {
  //   comp_idx += strides.data[i] * idxs[i];
  // }

  // 两个 for 循环可以合并
  uint32_t comp_idx = offset;
  uint32_t index_ = index;
  for (int32_t i = shape.size - 1; i >= 0; i--) // 从最内层开始
  {
    uint32_t l = shape.data[i];
    comp_idx += (index_ % l) * strides.data[i];
    index_ /= l;
  }
  return comp_idx;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, uint32_t size, CudaVec shape,
                              CudaVec strides, uint32_t offset)
{
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size)
    out[gid] = a[index_transform(gid, shape, strides, offset)];
  /// END SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, uint32_t offset)
{
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, uint32_t size, CudaVec shape,
                                   CudaVec strides, uint32_t offset)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[index_transform(gid, shape, strides, offset)] = a[gid];
}

void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, uint32_t offset)
{
  /**
   * Set items in a (non-compact) array using CUDA. you will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(uint32_t size, scalar_t val, scalar_t *out, CudaVec shape,
                                    CudaVec strides, uint32_t offset)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[index_transform(gid, shape, strides, offset)] = val;
}

void ScalarSetitem(const uint32_t size, scalar_t val, CudaArray *out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, uint32_t offset)
{
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
__global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] * b[gid];
}
void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] * val;
}
void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] / b[gid];
}
void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] / val;
}
void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwisePowerKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = pow(a[gid], b[gid]);
}
void EwisePower(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwisePowerKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = pow(a[gid], val);
}
void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = max(a[gid], b[gid]);
}
void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = max(a[gid], val);
}
void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] == b[gid];
}
void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] == val;
}
void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] >= b[gid];
}
void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] >= val;
}
void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = log(a[gid]);
}
void EwiseLog(const CudaArray &a, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = exp(a[gid]);
}
void EwiseExp(const CudaArray &a, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = tanh(a[gid]);
}
void EwiseTanh(const CudaArray &a, CudaArray *out)
{
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Matrix Multiplication operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel_naive(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M,
                                   uint32_t N, uint32_t P)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < M && j < P)
  {
    scalar_t t = 0;
    for (int k = 0; k < N; k++)
    {
      t += a[i * N + k] * b[k * P + j];
    }
    out[i * P + j] = t;
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
            uint32_t P)
{
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  // 对 M 和 P 进行并行
  // dim3 grid((M + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, (P + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM);
  // dim3 block(BASE_THREAD_NUM, BASE_THREAD_NUM); // max thread in one block 1024
  dim3 block((M + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, (P + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM);
  dim3 grid(BASE_THREAD_NUM, BASE_THREAD_NUM);
  MatmulKernel_naive<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

// https://github.com/CalebDu/DLsys-hw3/blob/main/src/ndarray_backend_cuda.cu
__global__ void MatmulKernel_tile(const scalar_t *a, const scalar_t *b, scalar_t *out,
                                  uint32_t M, uint32_t N, uint32_t P)
{
  uint32_t bidx = blockIdx.x, bidy = blockIdx.y, tidx = threadIdx.x,
           tidy = threadIdx.y;
  int x_range = static_cast<int>(bidx + 1) * TILE - M,
      y_range = static_cast<int>(bidy + 1) * TILE - P;
  if (x_range > 0)
  {
    a -= x_range * N;
    out -= x_range * P;
  }
  if (y_range > 0)
  {
    b -= y_range;
    out -= y_range;
  }
  a += bidx * TILE * N;
  b += bidy * TILE;
  out += (bidx * TILE) * P + (bidy * TILE);
  __shared__ scalar_t smemA[TILE][TILE], smemB[TILE][TILE];
  scalar_t accumu = 0.0f;
  for (int i = 0; i < N; i += TILE)
  {
    smemA[tidx][tidy] = (tidy + i < N) ? a[(tidx)*N + (tidy + i)] : 0.0f;
    smemB[tidx][tidy] = (tidx + i < N) ? b[(tidx + i) * P + tidy] : 0.0f;
    __syncthreads();
    for (int j = 0; j < TILE; j++)
    {
      accumu += smemA[tidx][j] * smemB[j][tidy];
    }
    __syncthreads();
  }
  out[tidx * P + tidy] = accumu;
}

void MatmulTiled(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                 uint32_t P)
{
  cudaMemset(out->ptr, 0, out->size * ELEM_SIZE);
  dim3 block(TILE, TILE);
  dim3 grid((M - 1) / TILE + 1, (P - 1) / TILE + 1);
  if (M < TILE || P < TILE || N < TILE)
  {
    MatmulKernel_naive<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  else
  {
    MatmulKernel_tile<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, uint32_t reduce_size, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    uint32_t offset = gid * reduce_size;
    scalar_t reduce_max = a[offset];
    for (int j = 1; j < reduce_size; j++)
    {
      reduce_max = max(reduce_max, a[j + offset]);
    }
    out[gid] = reduce_max;
  }
}

void ReduceMax(const CudaArray &a, CudaArray *out, uint32_t reduce_size)
{
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, uint32_t reduce_size, uint32_t size)
{
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
  {
    uint32_t offset = gid * reduce_size;
    scalar_t reduce_sum = 0;
    for (int j = 0; j < reduce_size; j++)
    {
      reduce_sum += a[j + offset];
    }
    out[gid] = reduce_sum;
  }
}

void ReduceSum(const CudaArray &a, CudaArray *out, uint32_t reduce_size)
{
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

PYBIND11_MODULE(ndarray_backend_cuda, m)
{

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<uint32_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                       uint32_t offset)
        {
    std::vector<uint32_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](uint32_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == nullptr) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_array", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("ewise_power", EwisePower);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}