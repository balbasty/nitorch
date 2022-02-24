#include <torch/extension.h>
#include "src/pushpull.h"
#include "src/regulariser.h"
#include "src/regulariser_grid.h"
#include "src/relax.h"
#include "src/relax_grid.h"
#include "src/precond.h"
#include "src/precond_grid.h"
#include "src/resize.h"
#include "src/pcg.h"
#include "src/fmg.h"
#include "src/multires.h"

using namespace ni;
namespace pya = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  const char *bound_doc =
    "Boundary conditions describe how to deal with indices that fall \n"
    "outside of the lattice or field-of-view.\n"
    "\n"
    "NiTorch implements boundary conditions that match those of common \n"
    "discrete transforms, such as the Fourier, Sine and Cosine \n"
    "transforms. There is a one-to-one correspondence between some of \n"
    "these boundary conditions and those used in scipy's ndimage module. \n"
    "Both conventions are implemented as aliases and can be used \n"
    "arbitrarily.\n"
    "\n"
    "|  nitorch  |      scipy       |              torch              |\n"
    "| --------- | ---------------- | ------------------------------- |\n"
    "| replicate | nearest          | border                          |\n"
    "| dct1      | mirror           | reflection, align_corners=False |\n"
    "| dct2      | reflect          | reflection, align_corners=True  |\n"
    "| fft       | wrap             |                                 |\n"
    "| zero      | constant, cval=0 | zero                            |\n"
    "| dst1      |                  |                                 |\n"
    "| dst2      |                  |                                 |\n"
    "\n"
    "References\n"
    "----------\n"
    "..[1] https://en.wikipedia.org/wiki/Discrete_cosine_transform\n"
    "..[2] https://en.wikipedia.org/wiki/Discrete_sine_transform\n"
    "..[3] https://docs.scipy.org/doc/scipy/reference/generated/"
    "scipy.ndimage.map_coordinates.html\n";


  py::enum_<BoundType>(m, "BoundType", bound_doc)
    .value("replicate",   BoundType::Replicate, " a  a  a | a b c d |  d  d  d")
    .value("nearest",     BoundType::Replicate, " a  a  a | a b c d |  d  d  d")
    .value("dct1",        BoundType::DCT1,      " d  c  b | a b c d |  c  b  a")
    .value("mirror",      BoundType::DCT1,      " d  c  b | a b c d |  c  b  a")
    .value("dct2",        BoundType::DCT2,      " c  b  a | a b c d |  d  c  b")
    .value("reflect",     BoundType::DCT2,      " c  b  a | a b c d |  d  c  b")
    .value("dst1",        BoundType::DST1,      "-b -a  0 | a b c d |  0 -d -c")
    .value("antimirror",  BoundType::DST1,      "-b -a  0 | a b c d |  0 -d -c")
    .value("dst2",        BoundType::DST2,      "-c -b -a | a b c d | -d -c -b")
    .value("antireflect", BoundType::DST2,      "-c -b -a | a b c d | -d -c -b")
    .value("dft",         BoundType::DFT,       " b  c  d | a b c d |  a  b  c")
    .value("wrap",        BoundType::DFT,       " b  c  d | a b c d |  a  b  c")
    .value("zero",        BoundType::Zero,      " 0  0  0 | a b c d |  0  0  0")
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

  py::enum_<GridAlignType>(m, "GridAlignType")
    .value("edge",      GridAlignType::Edge)
    .value("center",    GridAlignType::Center)
    .value("first",     GridAlignType::First)
    .value("last",      GridAlignType::Last)
    .export_values();

  m.def("grid_pull",           &ni::grid_pull,           "GridPull");
  m.def("grid_pull_backward",  &ni::grid_pull_backward,  "GridPull backward");
  m.def("grid_push",           &ni::grid_push,           "GridPush");
  m.def("grid_push_backward",  &ni::grid_push_backward,  "GridPush backward");
  m.def("grid_count",          &ni::grid_count,          "GridCount");
  m.def("grid_count_backward", &ni::grid_count_backward, "GridCount backward");
  m.def("grid_grad",           &ni::grid_grad,           "GridGrad");
  m.def("grid_grad_backward",  &ni::grid_grad_backward,  "GridGrad backward");

  m.def("regulariser",               &ni::regulariser,                "Field regulariser");
  m.def("regulariser_backward",      &ni::regulariser_backward,       "Field regulariser backward");
  m.def("regulariser_grid",          &ni::regulariser_grid,           "Grid regulariser");
  m.def("regulariser_grid_backward", &ni::regulariser_grid_backward,  "Grid regulariser backward");
  m.def("relax",                     &ni::relax,                      "Field relax");
  m.def("relax_grid",                &ni::relax_grid,                 "Grid relax");
  m.def("precond",                   &ni::precond,                    "Field preconditioner");
  m.def("precond_grid",              &ni::precond_grid,               "Grid preconditioner");
  m.def("resize",                    &ni::resize,                     "resize");
  m.def("resize_backward",           &ni::resize_backward,            "resize backward");
  m.def("prolongation",              &ni::prolongation,               "prolongation");
  m.def("prolongation_backward",     &ni::prolongation_backward,      "prolongation backward");
  m.def("restriction",               &ni::restriction,                "restriction");
  m.def("restriction_backward",      &ni::restriction_backward,       "restriction backward");
  m.def("fmg",                       &ni::fmg,                        "Field Full MultiGrid");
  m.def("fmg_grid",                  &ni::fmg_grid,                   "Grid Full MultiGrid");
  m.def("pcg",                       &ni::pcg,                        "Field Preconditioned Conjugate Gradient");
  m.def("fmg_prolongation",          &ni::fmg_prolongation,            "FMG prolongation");
  m.def("fmg_restriction",           &ni::fmg_restriction,             "FMG restriction");
}
