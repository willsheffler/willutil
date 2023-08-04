/*cppimport
<%


cfg['include_dirs'] = ['../../..','../extern']

cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['qcp.hpp']

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

template <typename F>
py::tuple qcp_rmsd_align(RowMatrixX<F> xyz1, RowMatrixX<F> xyz2) {
  Matrix<double, 3, 3> R;
  Matrix<double, 3, 1> c1, c2, T;
  double rms;
  {
    py::gil_scoped_release release;
    rms = qcp_rmsd_impl(xyz1, xyz2, R.data(), c1.data(), c2.data());
    T = R * -c1 + c2;
  }
  return py::make_tuple(rms, R, T);
}

template <typename F, typename I>
py::array_t<F> qcp_rmsd_regions(RowMatrixX<F> xyz1_in, RowMatrixX<F> xyz2_in,
                                Matrix<I, 1, Dynamic> sizes,
                                RowMatrixX<I> offsets) {
  F *rms = new F[offsets.rows()];
  {
    Matrix<F, Dynamic, 3> xyz1 = xyz1_in.block(0, 0, xyz1_in.rows(), 3);
    Matrix<F, Dynamic, 3> xyz2 = xyz2_in.block(0, 0, xyz2_in.rows(), 3);

    py::gil_scoped_release release;
    if (sizes.rows() != 1 || sizes.cols() != offsets.cols() ||
        sizes.sum() != xyz2.rows())
      throw std::runtime_error("bad sizes or offsets");

    int nseg = sizes.cols();
    xyz2.rowwise() -= xyz2.colwise().mean();

    F sqnorm2 = xyz2.array().square().sum();
    Matrix<I, 1, Dynamic> offsets2(nseg);
    offsets2.fill(0);
    for (int i = 0; i < nseg - 1; ++i)
      offsets2(0, i + 1) = offsets2(0, i) + sizes(0, i);

    for (int ioff = 0; ioff < offsets.rows(); ++ioff) {
      Matrix<F, 1, 3> cen(0, 0, 0);

      for (int iseg = 0; iseg < nseg; ++iseg) {
        cen += xyz1.block(offsets(ioff, iseg), 0, sizes(0, iseg), 3)
                   .colwise()
                   .sum();
      }
      cen /= xyz2.rows();

      F sqnorm = 0;
      Matrix<F, 3, 3> iprod;
      iprod << 0, 0, 0, 0, 0, 0, 0, 0, 0;
      for (int iseg = 0; iseg < nseg; ++iseg) {
        Matrix<F, Dynamic, 3> block1 =
            xyz1.block(offsets(ioff, iseg), 0, sizes(0, iseg), 3).rowwise() -
            cen;
        // cout << "block " << iseg << " ---" << endl;
        // cout << block1 << endl;
        // cout << "------" << endl;
        // cout << block1 << endl;
        // cout << "------" << endl;
        // std::exit(-1);
        auto block2 = xyz2.block(offsets2(0, iseg), 0, sizes(0, iseg), 3);
        iprod += block1.transpose() * block2;
        sqnorm += block1.array().square().sum();
      }

      double E0 = (sqnorm + sqnorm2) / 2;
      double rmsd;
      double A[9];
      for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj)
          A[3 * ii + jj] = iprod(ii, jj);
      FastCalcRMSDAndRotation(NULL, A, &rmsd, E0, xyz2.rows(), -1);

      // cout << "CEN1 " << cen << endl;
      // cout << "sqnorm1 " << sqnorm << endl;
      // cout << "sqnorm2 " << sqnorm2 << endl;

      rms[ioff] = rmsd;
    }
  }
  py::capsule free_when_done(
      rms, [](void *f) { delete[] reinterpret_cast<F *>(f); });
  return py::array_t<F>({offsets.rows()}, {sizeof(F)}, rms, free_when_done);
}

PYBIND11_MODULE(qcp, m) {
  m.def("qcp_rms_float", &qcp_rmsd<float>);
  m.def("qcp_rms_align_float", &qcp_rmsd_align<float>);
  m.def("qcp_rms_regions_f4i4", &qcp_rmsd_regions<float, int32_t>);
  m.def("qcp_rms_double", &qcp_rmsd<double>);
  m.def("qcp_rms_align_double", &qcp_rmsd_align<double>);
}

} // namespace qcp
} // namespace rms
} // namespace willutil