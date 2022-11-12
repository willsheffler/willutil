import willutil as wu

class NotPose(object):
   def __init__(self, fname):
      super(NotPose, self).__init__()
      self.pdb = wu.pdb.readpdb(fname)

   def size(self):
      return self.len(self.pdb.df)

   def sequence(self):
      return self.pdb.seq

   def secstruct(self):
      return 'L' * self.size()
