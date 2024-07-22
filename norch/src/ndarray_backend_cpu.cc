#include <iostream>
#include <vector>
#include <algorithm>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t; // scalar_t 可能是 float 或者 double 等其他类型
const uint32_t ELEM_SIZE = sizeof(scalar_t);

struct AlignedArray
{
  scalar_t *ptr;
  uint32_t size;
  AlignedArray(uint32_t size) : size(size)
  {
    uint32_t ret = posix_memalign((void **)&ptr, ALIGNMENT, ELEM_SIZE * size);
    if (ret != 0)
      throw std::bad_alloc();
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
};

void Fill(AlignedArray *out, scalar_t val)
{
  /**
   * Fill the values of an aligned array with val
   */
  for (uint32_t i = 0; i < out->size; i++)
  {
    out->ptr[i] = val;
  }
}

enum strided_index_mode
{
  INDEX_OUT,
  INDEX_IN,
  SET_VAL
};

void _strided_index_setter(const AlignedArray *a, AlignedArray *out, std::vector<int32_t> shape,
                           std::vector<int32_t> strides, int32_t offset, strided_index_mode mode, int val = -1)
{
  int depth = shape.size();
  std::vector<int32_t> loop(depth, 0);
  int cnt = 0;
  while (true)
  {
    // inner loop
    int index = offset;
    for (int i = 0; i < depth; i++)
    {
      index += strides[i] * loop[i];
    }
    switch (mode)
    {
    case INDEX_OUT:
      out->ptr[index] = a->ptr[cnt++];
      break;
    case INDEX_IN:
      out->ptr[cnt++] = a->ptr[index];
      break;
    case SET_VAL:
      out->ptr[index] = val;
      break;
    }

    // increment
    loop[depth - 1]++;

    // carry
    int idx = depth - 1;
    while (loop[idx] == shape[idx])
    {
      if (idx == 0)
      {
        // overflow
        return;
      }
      loop[idx--] = 0;
      loop[idx]++;
    }
  }
}

void Compact(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, int32_t offset)
{
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN YOUR SOLUTION
  _strided_index_setter(&a, out, shape, strides, offset, INDEX_IN);
  /// END YOUR SOLUTION
}

void EwiseSetitem(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, int32_t offset)
{
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  _strided_index_setter(&a, out, shape, strides, offset, INDEX_OUT);
  /// END YOUR SOLUTION
}

void ScalarSetitem(const uint32_t size, scalar_t val, AlignedArray *out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, int32_t offset)
{
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN YOUR SOLUTION
  _strided_index_setter(nullptr, out, shape, strides, offset, SET_VAL, val);
  /// END YOUR SOLUTION
}

void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] + val;
  }
}

/**
 * In the code the follows, use the above template to create analogous element-wise
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
void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}
void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}
void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void EwisePower(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::pow(a.ptr[i], b.ptr[i]);
  }
}
void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}
void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] == b.ptr[i];
  }
}
void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] == val;
  }
}

void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] >= b.ptr[i];
  }
}
void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] >= val;
  }
}

void EwiseLog(const AlignedArray &a, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray &a, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray &a, AlignedArray *out)
{
  for (uint32_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}
/// END YOUR SOLUTION

void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
            uint32_t p)
{
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < m; i++)
  {
    for (uint32_t j = 0; j < p; j++)
    {
      scalar_t t = 0;
      for (uint32_t k = 0; k < n; k++)
      {
        t += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
      out->ptr[i * p + j] = t;
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const scalar_t *__restrict__ a,
                       const scalar_t *__restrict__ b,
                       scalar_t *__restrict__ out)
{

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const scalar_t *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const scalar_t *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (scalar_t *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < TILE; i++)
  {
    for (uint32_t j = 0; j < TILE; j++)
    {
      scalar_t t = 0;
      for (uint32_t k = 0; k < TILE; k++)
      {
        t += a[i * TILE + k] * b[k * TILE + j];
      }
      out[i * TILE + j] += t;
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                 uint32_t n, uint32_t p)
{
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  memset(out->ptr, 0, out->size * ELEM_SIZE); // only for 0 init !!
  for (uint32_t i = 0; i < m / TILE; i++)
  {
    for (uint32_t j = 0; j < p / TILE; j++)
    {
      for (uint32_t k = 0; k < n / TILE; k++)
      {
        AlignedDot(&a.ptr[i * n * TILE + k * TILE * TILE], // [i,k] * [k,j] = [i,j] 为起点 TILE * TILE 的小矩阵
                   &b.ptr[k * p * TILE + j * TILE * TILE],
                   &out->ptr[i * p * TILE + j * TILE * TILE]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray &a, AlignedArray *out, uint32_t reduce_size)
{
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   * Example:
   *  a = (2,3,4,2) 原始数组 shape
   *  reduce = (4,2) reduce axes
   *  out = (2,3) reduce 后的数组 shape
   *  由于 compact 处理后，reduce 是连在一起的，因此可以直接认为是
   *  对 (6, 8) 的一个矩阵，reduce 成 (6)
   *  reduce_size = 8, out_size = 6
   */

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < out->size; i++)
  {
    scalar_t max = a.ptr[i * reduce_size];
    for (uint32_t j = 1; j < reduce_size; j++) // 可以从 1 开始
    {
      max = std::max(max, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray &a, AlignedArray *out, uint32_t reduce_size)
{
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < out->size; i++)
  {
    scalar_t sum = 0;
    for (uint32_t j = 0; j < reduce_size; j++)
    {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
  }
  /// END SOLUTION
}

PYBIND11_MODULE(ndarray_backend_cpu, m)
{
  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array").def(py::init<uint32_t>()).def("ptr", &AlignedArray::ptr_as_int).def_readonly("size", &AlignedArray::size);
  m.def("from_array", [](py::array_t<scalar_t> a, AlignedArray *out)
        { std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE); });
  m.def("to_numpy", [](AlignedArray &out, std::vector<uint32_t> shape, std::vector<uint32_t> strides, uint32_t offset)
        { std::vector<uint32_t> strides_byte = strides;
    std::transform(strides_byte.begin(), strides_byte.end(), strides_byte.begin(), [](uint32_t v)
                   { return v * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, strides_byte, out.ptr + offset); });

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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}