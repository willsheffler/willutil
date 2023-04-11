/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../cpp/extern']
cfg['compiler_args'] = ['-std=c++2a', '-w', '-Ofast']
cfg['dependencies'] = ['_orientations.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/

#include "willutil/sampling/_orientations.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Geometry>

namespace willutil {
namespace sampling {
namespace orientations {

namespace py = pybind11;

PYBIND11_MODULE(_orientations, m) {
  m.def("read_karney_orientations", &read_karney_orientations, R"pbdoc(
        docstring in sampling/orientations.pybind.cpp
    )pbdoc",
        py::call_guard<py::gil_scoped_release>());
}
} // namespace orientations
} // namespace sampling
} // namespace willutil