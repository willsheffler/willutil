import willutil as wu

def test_fetch_pdb_metadata():
   meta = wu.pdb.fetch_pdb_metadata(limit=10, timeout=1)
   # assert len(meta.resl) == 10
   assert all(len(k) == 4 for k in meta.resl)
   assert all(isinstance(v, float) for v in meta.resl.values())

def main():
   test_fetch_pdb_metadata()
   pass

if __name__ == '__main__':
   main()