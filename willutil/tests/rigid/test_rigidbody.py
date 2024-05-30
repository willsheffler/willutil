import pytest
import numpy as np
import willutil as wu

def main():
   test_rigidbody_cellsize_scale(False)
   test_rigidbody_contacts()
   test_rigidbody_bvh()
   test_rigidbody_moveby()
   test_rigidbody_parent()
   test_rigidbody_viz()

def test_rigidbody_cellsize_scale(showme=False):
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T

   body = wu.RigidBodyFollowers(coords=xyz, sym='L6_32', cellsize=80)
   if showme: wu.showme(body)
   # print(repr(body.coms()[:, :3]))

   coms0 = np.array([[1.67999986e+01, 9.69947657e+00, 2.95419880e+01],
                     [-1.67999924e+01, 9.69948730e+00, 2.95419880e+01],
                     [-6.19596624e-06, -1.93989639e+01, 2.95419880e+01],
                     [6.32000014e+01, -9.69947657e+00, 2.95419880e+01],
                     [8.00000062e+01, 1.93989639e+01, 2.95419880e+01],
                     [1.03200008e+02, 7.89815196e+01, 2.95419880e+01],
                     [-5.67999986e+01, -7.89815089e+01, 2.95419880e+01],
                     [-1.03200001e+02, 7.89815089e+01, 2.95419880e+01],
                     [-1.20000006e+02, -8.86809962e+01, 2.95419880e+01],
                     [1.03200008e+02, -5.95825450e+01, 2.95419880e+01],
                     [-2.32000076e+01, -7.89815196e+01, 2.95419880e+01],
                     [-1.20000006e+02, 4.98830684e+01, 2.95419880e+01],
                     [-1.03200001e+02, -5.95825557e+01, 2.95419880e+01],
                     [-5.67999986e+01, 5.95825557e+01, 2.95419880e+01],
                     [-3.99999938e+01, -4.98830684e+01, 2.95419880e+01],
                     [9.67999924e+01, -9.69948730e+00, 2.95419880e+01],
                     [-2.32000076e+01, 5.95825450e+01, 2.95419880e+01],
                     [-3.99999938e+01, 8.86809962e+01, 2.95419880e+01]])
   assert np.allclose(body.coms()[:, :3], coms0)

   body.cellsize *= 1.2
   if showme: wu.showme(body)
   coms1 = np.array([[1.67999986e+01, 9.69947657e+00, 2.95419880e+01],
                     [-1.67999924e+01, 9.69948730e+00, 2.95419880e+01],
                     [-6.19596624e-06, -1.93989639e+01, 2.95419880e+01],
                     [7.92000014e+01, -9.69947657e+00, 2.95419880e+01],
                     [9.60000062e+01, 1.93989639e+01, 2.95419880e+01],
                     [1.27200008e+02, 9.28379261e+01, 2.95419880e+01],
                     [-6.47999986e+01, -9.28379153e+01, 2.95419880e+01],
                     [-1.27200001e+02, 9.28379153e+01, 2.95419880e+01],
                     [-1.44000006e+02, -1.02537403e+02, 2.95419880e+01],
                     [1.27200008e+02, -7.34389515e+01, 2.95419880e+01],
                     [-3.12000076e+01, -9.28379261e+01, 2.95419880e+01],
                     [-1.44000006e+02, 6.37394749e+01, 2.95419880e+01],
                     [-1.27200001e+02, -7.34389622e+01, 2.95419880e+01],
                     [-6.47999986e+01, 7.34389622e+01, 2.95419880e+01],
                     [-4.79999938e+01, -6.37394749e+01, 2.95419880e+01],
                     [1.12799992e+02, -9.69948730e+00, 2.95419880e+01],
                     [-3.12000076e+01, 7.34389515e+01, 2.95419880e+01],
                     [-4.79999938e+01, 1.02537403e+02, 2.95419880e+01]])
   # print(repr(body.coms()[:, :3]))
   if showme: wu.showme(coms1)
   assert np.allclose(body.coms()[:, :3], coms1)

   body.cellsize /= 1.2
   if showme: wu.showme(body)
   assert np.allclose(body.coms()[:, :3], coms0)

   body.scale *= 1.123
   if showme: wu.showme(body)
   assert np.allclose(coms0 * 1.123, body.coms()[:, :3])

   body.scale_com_with_cellsize = True
   body.cellsize = 80
   if showme: wu.showme(body)
   assert np.allclose(body.coms()[:, :3], coms0)

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

   a = wu.RigidBody(wu.hrandpoint(10))
   xform = wu.hrot([1, 0, 0], 180)
   b = wu.RigidBody(parent=a, xfromparent=xform)
   a.position = wu.hrandsmall()
   assert np.allclose(b.position, xform @ a.position)
   # with pytest.raises(ValueError):
   b.position = np.eye(4)
   assert np.allclose(b.position, np.eye(4))
   assert np.allclose(a.position, wu.hinv(xform))

   a.position = wu.hrandsmall()
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
   body2 = wu.RigidBody(parent=body, xfromparent=xform)

   wu.showme([body, body2], headless=noshow)

if __name__ == '__main__':
   main()
