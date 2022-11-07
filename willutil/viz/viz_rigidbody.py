import numpy as np
from willutil.viz.pymol_viz import pymol_load, cgo_cyl, cgo_sphere, cgo_fan, cgo_cube, showcube
import willutil as wu
from willutil.rigid.rigidbody import RigidBody

@pymol_load.register(RigidBody)
def pymol_viz_RigidBody(
   body,
   state,
   name='rigidbody',
   addtocgo=None,
   make_cgo_only=False,
):
   import pymol
   v = pymol.cmd.get_view()
   state["seenit"][name] += 1

   cgo = list()
   wu.showme(body.coords, addtocgo=cgo)

   if addtocgo is None:
      pymol.cmd.load_cgo(cgo, f'{name}_{state["seenit"][name]}')
      pymol.cmd.set_view(v)
   else:
      addtocgo.extend(cgo)
   if make_cgo_only:
      return cgo
