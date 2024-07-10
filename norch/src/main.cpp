#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "iostream"

namespace py = pybind11;

void add(const float *a, const float *b, float *c, int n)
{
  std::cout << n << std::endl;
  for (int i = 0; i < n; i++)
  {
    c[i] = a[i] + b[i];
  }
}

PYBIND11_MODULE(libnorch, m)
{
  m.def("add", [](py::array_t<float, py::array::c_style> a, py::array_t<float, py::array::c_style> b, py::array_t<float, py::array::c_style> c)
        { add(static_cast<const float *>(a.request().ptr),
              static_cast<const float *>(b.request().ptr),
              static_cast<float *>(c.request().ptr),
              c.size()); });
}
