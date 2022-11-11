import numpy as np
import willutil as wu

ic.configureOutput(includeContext=True)

def main():
   test_asufit()

def test_asufit():
   # fname = wu.tests.testdata.test_data_path('pdb/x012.pdb')
   # fname = '/home/sheffler/src/willutil/willutil/tests/testdata/pdb/x012.pdb'
   fname = '/home/sheffler/src/BFF/rf_diffusion/diffuser000Ainit.pdb'
   pdb = wu.pdb.readpdb(fname)
   pdb = pdb.subfile()
   pdb = pdb.subfile(atomnames=['CA'], chains=['A'])
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T

   xyz[:, :3] += wu.hcom(xyz)[:3]

   # axstan = np.arcsin(0.3647274834674778)
   # xyz = wu.hrandpoint(400, (50, 0, 10), 10)

   # frames = wu.sym.frames('icos')
   ax2 = wu.sym.axes('icos')[2]
   ax3 = wu.sym.axes('icos')[3]

   frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   lever = wu.hrog(xyz) * 1.5
   with wu.Timer():
      ic('symfit')
      mc = wu.sym.asufit(
         'icos',
         xyz,
         nf1=2,
         nf2=3,
         frames=frames,
         showme=True,
         contactfrac=0.3,
         contactdist=12,
         cartsd=0.5,
         temperature=0.5,
         resetinterval=100,
         correctionfactor=1.5,
         iterations=1000,
         driftpenalty=0.0,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=1,
         biasradial=2,
         usebvh=True,
      )

   # wu.showme(wu.hpoint(xyz))
   # wu.showme(wu.hxform(xdock, xyz, homogout=True))

if __name__ == '__main__':
   main()