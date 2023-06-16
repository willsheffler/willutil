import willutil as wu
from willutil.sym.spacegroup_contacts import minimal_spacegroup_cover_symelems, check_if_symelems_complete

def main():
   # symelems = wu.sym.symelems('P6')
   # genelems = [symelems[i] for i in [0, 1, 3]]
   # complete = check_if_symelems_complete('P6', genelems)

   # minimal_spacegroup_cover_symelems('P6')
   minimal_spacegroup_cover_symelems('I4132')

   assert 0

# def test_sg_color_symel

if __name__ == '__main__':
   main()
