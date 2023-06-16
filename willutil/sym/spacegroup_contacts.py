import itertools
import numpy as np
import willutil as wu

def check_if_symelems_complete(spacegroup, symelems, depth=20, radius=5, trials=10000, fudgefactor=0.9):
   from willutil.cpp.geom import expand_xforms_rand
   latticevec = wu.sym.lattice_vectors(spacegroup, 'nonsingular')

   generators = list()
   for unitelem in symelems:
      elem = unitelem.tolattice(latticevec)
      ops = elem.make_operators_screw()
      generators.append(ops)
      # wu.showme(wu.htrans([0.02, 0.03, 0.04]) @ ops)
   generators = np.concatenate(generators)

   frames, _ = expand_xforms_rand(generators, depth=depth, radius=radius, trials=trials)
   frames = wu.sym.tounitcell(latticevec, frames)
   x, y, z = frames[:, :3, 3].T
   nunitcell = np.sum((-0.001 < x) * (x < 1.999) * (-0.001 < y) * (y < 1.999) * (-0.001 < z) * (z < 1.999))
   nunitcell_target = 8 * wu.sym.copies_per_cell(spacegroup)

   if nunitcell >= nunitcell_target * fudgefactor:
      print(spacegroup)
      for e in symelems:
         print(e)
      print(frames.shape, nunitcell, nunitcell_target, flush=True)
      return True
   else:
      return False

def minimal_spacegroup_cover_symelems(spacegroup):

   allsymelems = wu.sym.symelems(spacegroup)
   max_combo = min(len(allsymelems), 5)
   generators = list()
   for ncombo in range(2, max_combo + 1):
      for combo in itertools.product(*[list(range(len(allsymelems)))] * ncombo):
         if any([combo[i] >= combo[i + 1] for i in range(ncombo - 1)]):
            continue
         genelems = [allsymelems[i] for i in combo]
         complete = check_if_symelems_complete(spacegroup, genelems)
         if complete:
            generators.append(genelems)

      if generators:
         break

   return generators