import fire
import willutil as wu

def make_pymol_sg_diagram(sg, name=None):
    name = name or f'sg_{sg}.pse'
    x = wu.sym.Xtal(sg)
    wu.showme(x, save=name)

def symelems(sg):
    x = wu.sym.Xtal(sg)
    sg, info = wu.sym.xtalinfo(sg)
    print(f'canonical name {sg}')
    for se in info.symelems:
        print(se)

if __name__ == '__main__':
    fire.Fire()
