/*cppimport
<%


cfg['include_dirs'] = ['../../..','../extern']

cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/types.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/

// cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "willutil/cpp/rms/qcp.hpp"

namespace py = pybind11;

namespace willutil {
namespace rms {
namespace qcp {

PYBIND11_MODULE(qcp, m) {
  m.def("qcp_rms_float", &qcp_rmsd<float>);
  m.def("qcp_rms_double", &qcp_rmsd<double>);
}

} // namespace qcp
} // namespace rms
} // namespace willutil