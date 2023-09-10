import pytest, tempfile
import numpy as np
from numpy.random import rand
import willutil as wu
from opt_einsum import contract as einsum

pytest.skip(allow_module_level=True)

def main():

   #   for i in range(40, 160, 10):
   #      cellgeom = wu.sym.full_cellgeom('P1', [
   #         1,
   #         1.3,
   #         1.7,
   #         90,
   #         90,
   #         float(i),
   #      ], strict=False)
   #      # print(cellgeom)
   #      ok = helper_test_spacegroup_frames_pymol('P1', cellgeom, ncells=5, dump_pdbs=True)
   #      if ok:
   #         print(cellgeom)
   #
   #   assert 0

   # for spacegroup in ['P1']:
   for spacegroup in wu.sym.sg_all_chiral:
      lattype = wu.sym.latticetype(spacegroup)
      ncells = 5
      cellgeom = wu.sym.sg_nonsingular_cellgeom
      cellgeom = wu.sym.full_cellgeom(spacegroup, cellgeom, strict=False)
      helper_test_spacegroup_frames_pymol(spacegroup, cellgeom, ncells=ncells, dump_pdbs=True)

@pytest.mark.parametrize('spacegroup', reversed(wu.sym.sg_all_chiral))
def test_spacegroup_frames_pymol(spacegroup):
   lattype = wu.sym.latticetype(spacegroup)
   cellgeom = wu.sym.sg_nonsingular_cellgeom
   cellgeom = wu.sym.full_cellgeom(spacegroup, cellgeom, strict=False)
   ncells = 5
   if wu.sym.copies_per_cell(spacegroup) > 8:
      print('test_spacegroup_pymol.py skipping large unitcell', spacegroup)
      return
   assert helper_test_spacegroup_frames_pymol(spacegroup, cellgeom, ncells=ncells)

def helper_test_spacegroup_frames_pymol(spacegroup, cellgeom, ncells, dump_pdbs=False):
   pymol = pytest.importorskip('pymol')
   # ic(spacegroup, cellgeom, ncells)

   frames = wu.sym.sgframes(spacegroup, cellgeom='nonsingular', cells=ncells)
   unitcrd = wu.hpoint(np.array([
      [0.1, 0.2, 0.3],
      [0.3, 0.2, 0.1],
      [0.2, 0.1, 0.1],
   ]))
   latvec = wu.sym.lattice_vectors(spacegroup, cellgeom)
   asymcrd = wu.sym.applylatticepts(latvec, unitcrd)

   with tempfile.TemporaryDirectory() as tmpdir:
      pymol.cmd.delete('all')
      fname = f'{tmpdir}/test.pdb'
      wu.dumppdb(fname, asymcrd, spacegroup=spacegroup, cellgeom=cellgeom)
      pymol.cmd.load(fname)
      pymol.cmd.symexp('pref', 'test', 'all', 9e9)
      crdref = pymol.cmd.get_coords()
      if crdref is None: return False
      pymol.cmd.delete('all')
      crdtst = einsum('fij,cj->fci', frames, asymcrd).reshape(-1, 4)
      ok = len(crdref) == 27 * len(asymcrd) * wu.sym.copies_per_cell(spacegroup)
      if not ok: return False
      # return True
      # assert len(crdtst) == 64 * len(asymcrd) * wu.sym.copies_per_cell(spacegroup)
      # assert len(crdtst) == 125 * len(asymcrd) * wu.sym.copies_per_cell(spacegroup)

      if dump_pdbs:
         wu.dumppdb('crdtst.pdb', 10 * crdtst[:, :3])
         wu.dumppdb('crdref.pdb', 10 * crdref)
      # assert 0

      delta = np.sum((crdtst[None, :, :3] - crdref[:, None])**2, axis=-1)
      # ic(np.sum(~np.isclose(0, np.min(delta, axis=1))))
      ok = np.allclose(np.min(delta, axis=1), 0, atol=1e-4)

      # print(np.max(np.min(delta, axis=1), 0))
      return ok
      # return np.allclose(np.min(delta, axis=1), 0)

if __name__ == '__main__':
   main()
