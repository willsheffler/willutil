import pytest
import willutil as wu

def main():
   test_notpose()

def test_notpose():
   pyro = pytest.importorskip('pyrosetta')
   pyro.init('-mute all')
   fname = wu.tests.testdata.test_data_path('pdb/1coi.pdb1.gz')

   nopo = wu.NotPose(fname)
   pose = pyro.pose_from_file(fname)
   assert len(pose.secstruct()) == len(nopo.secstruct())
   assert pose.sequence() == nopo.sequence()

if __name__ == '__main__':
   main()