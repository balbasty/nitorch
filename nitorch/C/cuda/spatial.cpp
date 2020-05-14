#include <torch/extension.h>
#include <pybind11/stl.h>
#include "../pushpull.h"

using at::Tensor;
using c10::IntArrayRef;
using namespace ni;
namespace py = pybind11;
using namespace py::literals;

using TensorRef = const Tensor &;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::enum_<BoundType>(m, "BoundType")
    .value("replicate", BoundType::Replicate)
    .value("dct1",      BoundType::DCT1)
    .value("dct2",      BoundType::DCT2)
    .value("dst1",      BoundType::DST1)
    .value("dst2",      BoundType::DST2)
    .value("dft",       BoundType::DFT)
    .value("zero",      BoundType::Zero)
    .value("sliding",   BoundType::Sliding).
    export_values();

  py::enum_<InterpolationType>(m, "InterpolationType")
    .value("nearest",   InterpolationType::Nearest)
    .value("linear",    InterpolationType::Linear)
    .value("quadratic", InterpolationType::Quadratic)
    .value("cubic",     InterpolationType::Cubic)
    .value("fourth",    InterpolationType::FourthOrder)
    .value("fifth",     InterpolationType::FifthOrder)
    .value("sixth",     InterpolationType::SixthOrder)
    .value("seventh",   InterpolationType::SeventhOrder).
    export_values();

  m.def("grid_pull_cpu",          &grid_pull_cpu,        "GridPull (CPU)",
        "input"_a, "grid"_a, "bound"_a, "interpolation"_a, "extrapolate"_a);
  m.def("grid_pull_backward_cpu", &grid_pull_backward_cpu, "GridPull backward (CPU)",
        "grad"_a, "input"_a, "grid"_a, "bound"_a, "interpolation"_a, "extrapolate"_a);
  m.def("grid_push_cpu",          &grid_push_cpu,          "GridPush (CPU)",
        "input"_a, "grid"_a, "source_size"_a, "bound"_a, "interpolation"_a, "extrapolate"_a);
  m.def("grid_push_backward_cpu", &grid_push_backward_cpu, "GridPush backward (CPU)",
        "grad"_a, "input"_a, "grid"_a, "bound"_a, "interpolation"_a, "extrapolate"_a);
}