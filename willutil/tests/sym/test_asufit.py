import pytest
import numpy as np
import willutil as wu

ic.configureOutput(includeContext=True)

def main():
   test_asufit_oct()
   # test_asufit_icos()
   ic('TEST asufit DONE')

@pytest.mark.xfail()
def test_asufit_oct():
   sym = 'oct'
   symaxes = [3, 2]
   assert len(symaxes)
   # fname = wu.tests.testdata.test_data_path('pdb/x012.pdb')
   # fname = '/home/sheffler/src/BFF/rf_diffusion/diffuser000Ainit.pdb'
   # pdb = wu.pdb.readpdb(fname)
   # pdb = pdb.subfile(atomnames=['CA'], chains=['A'])
   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   # xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz = wu.hrandpoint(500, std=10)
   xyz[:, :3] += 40 * (wu.sym.axes(sym)[2] + wu.sym.axes(sym)[3])[:3]
   # wu.showme(xyz)
   ax2 = wu.sym.axes(sym)[2]
   ax3 = wu.sym.axes(sym)[3]
   frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   lever = wu.hrog(xyz) * 1.5
   '''

   '''
   with wu.Timer():
      ic('symfit')
      np.random.seed(7)
      mc = wu.sym.asufit(
         sym,
         xyz,
         symaxes=[3, 2],
         frames=frames,
         showme=True,
         contactfrac=0.7,
         contactdist=12,
         cartsd=1,
         temperature=1,
         resetinterval=100,
         correctionfactor=1.5,
         iterations=1000,
         driftpenalty=0.0,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=4,
         usebvh=True,
      )
   assert np.allclose(
      mc.bestconfig,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_icos():
   fname = wu.tests.testdata.test_data_path('pdb/x012.pdb')
   pdb = wu.pdb.readpdb(fname)
   pdb = pdb.subfile(atomnames=['CA'], chains=['A'])
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz[:, :3] += wu.hcom(xyz)[:3]
   ax2 = wu.sym.axes('icos')[2]
   ax3 = wu.sym.axes('icos')[3]
   frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   lever = wu.hrog(xyz) * 1.5
   with wu.Timer():
      ic('symfit')
      np.random.seed(7)
      mc = wu.sym.asufit('icos', xyz, symaxes=[3, 2], frames=frames, showme=False, contactfrac=0.3, contactdist=12,
                         cartsd=0.5, temperature=2, resetinterval=10, correctionfactor=1.5, iterations=30,
                         driftpenalty=0.1, anglepenalty=0.1, thresh=0.0, spreadpenalty=0.1, biasradial=2, usebvh=True)
      ref = np.array([[0.99880172, 0.01937787, 0.04494025, -3.38724054],
                      [-0.02529149, 0.99052266, 0.13500072, 1.91723184],
                      [-0.04189831, -0.13597556, 0.98982583, 2.10409862], [0., 0., 0., 1.]])
      assert np.allclose(ref, mc.bestconfig)
      ic('test_asufit_icos PASS!')

if __name__ == '__main__':
   main()