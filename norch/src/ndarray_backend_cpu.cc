#include <iostream>
#include <vector>
#include <algorithm>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#define ALIGNMENT 256
typedef float scalar_t; // scalar_t 可能是 float 或者 double 等其他类型
const size_t ELEMENT_SIZE = sizeof(scalar_t);

struct AlignedArray
{
  scalar_t *ptr;
  size_t size;
  AlignedArray(size_t size) : size(size)
  {
    int ret = posix_memalign((void **)&ptr, ALIGNMENT, ELEMENT_SIZE * size);
    if (ret != 0)
      throw std::bad_alloc();
  }
  ~AlignedArray() { free(ptr); }
};

void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
{
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
{
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++)
  {
    out->ptr[i] = a.ptr[i] + val;
  }
}

PYBIND11_MODULE(libndarray_backend_cpu, m)
{
  py::class_<AlignedArray>(m, "Array").def(py::init<size_t>());
  m.def("from_array", [](py::array_t<scalar_t> a, AlignedArray *out)
        { std::memcpy(out->ptr, a.request().ptr, out->size * ELEMENT_SIZE); });
  m.def("to_numpy", [](AlignedArray &out, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset)
        { std::vector<size_t> strides_byte = strides;
    std::transform(strides_byte.begin(), strides_byte.end(), strides_byte.begin(), [](size_t v)
                   { return v * ELEMENT_SIZE; });
    return py::array_t<scalar_t>(shape, strides_byte, out.ptr + offset); });
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
}