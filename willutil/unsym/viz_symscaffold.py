import tempfile
import numpy as np
from willutil.unsym.symscaffold import SymScaffIcos
from willutil.viz.pymol_viz import pymol_load
import willutil as wu

@pymol_load.register(SymScaffIcos)
def pymol_viz_pdbfile(
   scaf,
   state,
   name='scaf',
   showcen=True,
   **kw,
):

   cen0 = wu.hnormalized([.21, .15, 1])
   ax2all = wu.sym.axes('icos', nfold=2, all=True)
   ax3all = wu.sym.axes('icos', nfold=3, all=True)
   ax5all = wu.sym.axes('icos', nfold=5, all=True)
   ax2all = np.concatenate([ax2all, -ax2all])
   ax3all = np.concatenate([ax3all, -ax3all])
   ax5all = np.concatenate([ax5all, -ax5all])
   ax2 = ax2all[np.argmin(wu.hangle(ax2all, cen0))]
   ax3 = ax3all[np.argmin(wu.hangle(ax3all, cen0))]
   ax5 = ax5all[np.argmin(wu.hangle(ax5all, cen0))]
   # cen = wu.hnormalized(ax2 + ax3 + ax5)
   cen = cen0

   pt2 = wu.hnormalized(ax2 * 0.90 + cen)
   pt3 = wu.hnormalized(ax3 * 0.90 + cen)
   pt5 = wu.hnormalized(ax5 * 0.50 + cen)

   scale = 28
   pts = wu.hpoint([
      cen * scale,
      pt2 * scale,
      pt3 * scale,
      pt5 * scale,
   ])
   ic(pts)
   pointnames = ['CEN', 'EDGE', 'VERT', 'FACE']
   frames = wu.sym.frames('icos')
   elem = np.concatenate([np.tile('H', (60, 1)), np.where(~scaf.occ, 'C', 'H')], axis=1)
   ic(scaf.occ.shape)
   ic(elem.shape)

   sympts = wu.hxformpts(frames, pts)
   ic(sympts.shape)
   ic(str(elem[1, 2]))

   # with tempfile.TemporaryDirectory() as td:
   if True:
      td = '/home/sheffler/tmp/'
      with open(f'{td}/SymScaffIcos.pdb', 'w') as out:
         for iframe in range(len(frames)):
            for ipoint in range(len(pts)):
               if not showcen and pointnames[ipoint] == 'CEN': continue
               idx = 10 * iframe + ipoint
               out.write(wu.pdb.pdb_format_atom(
                  ia=idx,
                  x=sympts[iframe, ipoint, 0],
                  y=sympts[iframe, ipoint, 1],
                  z=sympts[iframe, ipoint, 2],
                  ir=iframe,
                  rn=f'I{iframe:02}',
                  an=pointnames[ipoint],
                  c=0,
                  elem=elem[iframe, ipoint],
               ))

      from pymol import cmd

      cmd.load(f'{td}/SymScaffIcos.pdb')

   ic(scaf.placed)

   for p in scaf.placed:
      p = [10 * f + a + 1 for f, a in p]
      for i in range(len(p)):
         cmd.bond(f'id {p[i-1]}', f'id {p[i]}')

   cmd.color('white', 'name CEN')
   cmd.color('green', 'name EDGE')
   cmd.color('cyan', 'name VERT')
   cmd.color('yellow', 'name FACE')
   cmd.show('sph')
   cmd.do('alter elem H, vdw=0.8')
   cmd.do('alter elem C, vdw=2.7')
   cmd.rebuild()

   cmd.hide('ev', 'not name FACE+VERT')

   # cmd.color('red', 'resi 0')
