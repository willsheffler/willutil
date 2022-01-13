import willutil as wu

def test_pisces_lookup():
    _ = 'cullpdb_pc50.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains5505'
    assert wu.pdb.get_pisces_file(50, 1.5) == _
    assert wu.pdb.get_pisces_file(50, 1.6) == _
    assert wu.pdb.get_pisces_file(49, 1.6) == _
    _ = 'cullpdb_pc40.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains4938'
    assert wu.pdb.get_pisces_file(38, 1.6) == _
    _ = 'cullpdb_pc30.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains336'
    assert wu.pdb.get_pisces_file(28, 1.1) == _

def test_pisces_read():
    setname = 'cullpdb_pc15.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains256'
    df = wu.pdb.read_pisces(setname)
    assert df.PDBchain[b'4TXRA'] == b'4TXRA'
    assert df.code[b'4TXRA'] == b'4TXR'
    assert df.chain[b'4TXRA'] == b'A'

def test_get_pisces_set():
    pc = wu.pdb.get_pisces_set(maxresl=1.4, max_seq_ident=50)
    assert len(pc) == 1476

def main():
    # wu.pdb.download_pisces()
    fn = 'cullpdb_pc15.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains256'
    test_pisces_lookup()
    test_pisces_read()
    test_get_pisces_set()

if __name__ == '__main__':
    main()
