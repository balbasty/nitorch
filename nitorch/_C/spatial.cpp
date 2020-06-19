#include <torch/extension.h>
#include "pushpull.h"
#include "conv.h"

using namespace ni;
namespace pya = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::enum_<BoundType>(m, "BoundType")
    .value("replicate", BoundType::Replicate)
    .value("dct1",      BoundType::DCT1)
    .value("dct2",      BoundType::DCT2)
    .value("dst1",      BoundType::DST1)
    .value("dst2",      BoundType::DST2)
    .value("dft",       BoundType::DFT)
    .value("sliding",   BoundType::Sliding)
    .value("zero",      BoundType::Zero)
    .export_values();

  py::enum_<InterpolationType>(m, "InterpolationType")
    .value("nearest",   InterpolationType::Nearest)
    .value("linear",    InterpolationType::Linear)
    .value("quadratic", InterpolationType::Quadratic)
    .value("cubic",     InterpolationType::Cubic)
    .value("fourth",    InterpolationType::FourthOrder)
    .value("fifth",     InterpolationType::FifthOrder)
    .value("sixth",     InterpolationType::SixthOrder)
    .value("seventh",   InterpolationType::SeventhOrder)
    .export_values();

  m.def("grid_pull",           &ni::grid_pull,           "GridPull");
  m.def("grid_pull_backward",  &ni::grid_pull_backward,  "GridPull backward");
  m.def("grid_push",           &ni::grid_push,           "GridPush");
  m.def("grid_push_backward",  &ni::grid_push_backward,  "GridPush backward");
  m.def("grid_count",          &ni::grid_count,          "GridCount");
  m.def("grid_count_backward", &ni::grid_count_backward, "GridCount backward");
  m.def("grid_grad",           &ni::grid_grad,           "GridGrad");
  m.def("grid_grad_backward",  &ni::grid_grad_backward,  "GridGrad backward");

  m.def("conv",                &ni::conv,                "Conv");
}
