from pymol import cmd
import glob, os, random

"""
run /home/sheffler/src/willutil/examples/pymol_sym_onto_template.py; sym_onto_template_dirs('c4', '/digs/home/amijais/projects/guanine/redesign/C3s/fix_BB/v26/refined/asymm/C4', '/digs//home/amijais/projects/guanine/redesign/C3s/fix_BB/v26/refined/asymm/', 'testout')
"""


def sym_onto_template_files(sym, fsym, fasym, fnew):
    print(fsym, fasym)
    cmd.delete("all")
    # I dont know why these randomized names are necessary, but they are
    # must be some strange pymol bug with repeated selection names
    sym_selname = "sym" + str(random.random())[3:]
    asym_selname = "asym" + str(random.random())[3:]
    cmd.load(fsym, sym_selname)
    cmd.load(fasym, asym_selname)
    cmd.super(asym_selname, sym_selname)
    nsym = int(sym[1:])
    # try:
    makecx(asym_selname, n=nsym, name="symmetrized")
    cmd.save(fnew, "symmetrized")
    # except:
    # pass

    # assert 0
    cmd.delete("all")


def sym_onto_template_dirs(sym, sympath, asympath, outdir):
    os.makedirs(outdir, exist_ok=True)
    gsym = list(sorted(glob.glob(sympath + "/*.pdb")))
    gasym = list(sorted(glob.glob(asympath + "/*.pdb")))
    # print(gsym)
    assert len(gsym) == len(gasym)
    for fsym, fasym in zip(gsym, gasym):
        fnew = outdir + "/" + os.path.basename(fasym) + "_" + sym + ".pdb"
        sym_onto_template_files(sym, fsym, fasym, fnew)
