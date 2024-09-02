import willutil as wu
import numpy as np

def main():
    pdb = wu.readpdb('/home/sheffler/for/shunzhi/xtal_cartoon/f432_asu_orig.pdb')
    xyz = pdb.bb()
    input_T_cen = np.array([90, 90, 90])
    input_O_cen = np.array([180, 180, 180])
    cellsize = 4 * (input_O_cen[0] - input_T_cen[0])
    # ic(type(cellsize))
    # xyz -= input_O_cen

    sym = 'F432'
    elem = wu.sym.symelems(sym)
    ic(elem)
    x = wu.sym.Xtal(sym)

    # wu.sym.showsymelems(sym, wu.sym.symelems(sym), scan=0, weight=3, offset=0.0)
    # wu.showme(x)
    x.dump_pdb('/home/sheffler/for/shunzhi/xtal_cartoon/shunzhi_f432_asym.pdb', xyz, cellsize)

if __name__ == '__main__':
    main()
