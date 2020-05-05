#include <torch/extension.h>
#include <ATen/NativeFunctions.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pull3d_cpu", &at::native::grid_sampler_3d_cpu, "Pull 3D (CPU)");
  m.def("pull3d_cuda", &at::native::grid_sampler_3d_cuda, "Pull 3D (CUDA)");
  m.def("push3d_cpu", &at::native::grid_sampler_3d_backward_cpu, "Push 3D (CPU)");
  m.def("push3d_cuda", &at::native::grid_sampler_3d_backward_cuda, "Push 3D (CUDA)");
  m.def("pull2d_cpu", &at::native::grid_sampler_2d_cpu, "Pull 2D (CPU)");
  m.def("pull2d_cuda", &at::native::grid_sampler_2d_cuda, "Pull 2D (CUDA)");
  m.def("push2d_cpu", &at::native::grid_sampler_2d_backward_cpu, "Push 2D (CPU)");
  m.def("push2d_cuda", &at::native::grid_sampler_2d_backward_cuda, "Push 2D (CUDA)");
}