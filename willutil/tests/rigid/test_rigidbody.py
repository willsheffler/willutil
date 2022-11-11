import pytest
import numpy as np
import willutil as wu

def main():
   test_rigidbody_contacts()
   test_rigidbody_bvh()
   test_rigidbody_moveby()
   test_rigidbody_parent()
   test_rigidbody_viz()

def test_rigidbody_contacts():
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   body1 = wu.RigidBody(xyz)
   body2 = wu.RigidBody(xyz)
   npoints = len(body1)
   assert np.allclose(body1.point_contact_count(body2), [npoints, npoints])
   cfrac = body1.contact_fraction(body2)
   assert np.allclose(cfrac, 1, 1)

   body2.moveby([1, 2, 3])
   assert np.allclose(body1.point_contact_count(body2), [npoints, npoints])
   body2.moveby([1, 2, 3])
   assert np.allclose(body1.point_contact_count(body2), [npoints, npoints])
   body2.moveby([1, 2, 3])
   assert np.allclose(body1.point_contact_count(body2), [695, 671])

def test_rigidbody_moveby():
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   body = wu.RigidBody(xyz)

   com1 = body.com()
   com1b = wu.hcom(body.coords)
   body.move_about_com(wu.hrot([1, 0, 0], 180))
   com2b = wu.hcom(body.coords)
   com2 = body.com()
   assert np.allclose(com1b, com2b)
   assert np.allclose(com1, com2)

def test_rigidbody_bvh():
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   body1 = wu.RigidBody(xyz)
   body2 = wu.RigidBody(parent=body1, xfromparent=wu.hrot([1, 0, 0], 180))

   body1.moveby([-5, -5, -5])
   assert body1.clashes(body2) == 0
   assert body1.contacts(body2) == 4
   assert len(body1.interactions(body2)) == 4

   body1.moveby([-5, -5, -5])
   assert body1.clashes(body2) == 243
   assert body1.contacts(body2) == 4319
   assert len(body1.interactions(body2)) == 4319
   # ic(body1.point_contact_count(body2))

   body1.moveby([-5, -5, -5])
   assert body1.clashes(body2) == 555
   assert body1.contacts(body2) == 9052
   assert len(body1.interactions(body2)) == 9052
   # ic(body1.point_contact_count(body2))

def test_rigidbody_parent():

   a = wu.RigidBody()
   xform = wu.hrot([1, 0, 0], 180)
   b = wu.RigidBody(parent=a, xfromparent=xform)
   a.position = wu.hrand()
   with pytest.raises(ValueError):
      b.position = np.eye(4)
   assert np.allclose(b.position, xform @ a.position)

   a.position = wu.hrand()
   xform2 = wu.hrot([1, 1, 0], 120)
   c = wu.RigidBody(parent=b, xfromparent=xform2)
   assert np.allclose(c.position, xform2 @ xform @ a.position)

def test_rigidbody_viz(noshow=True):
   pytest.importorskip('pymol')
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   body = wu.RigidBody(xyz)
   xform = wu.hrot([1, 0, 0], 180)
   body2 = wu.RigidBody(xyz, parent=body, xfromparent=xform)

   wu.showme([body, body2], headless=noshow)

if __name__ == '__main__':
   main()