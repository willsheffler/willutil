import numpy as np
import willutil as wu

lattice_minmult = dict(
   TRICLINIC=2,
   MONOCLINIC=4,
   CUBIC=4,
   ORTHORHOMBIC=4,
   TETRAGONAL=8,
   HEXAGONAL=6,
)
lattice_trans_dof = dict(
   TRICLINIC=(1, 2, 3),
   MONOCLINIC=(1, 2, 3),
   CUBIC=(1, 1, 1),
   ORTHORHOMBIC=(1, 2, 3),
   TETRAGONAL=(1, 1, 3),
   HEXAGONAL=(1, 1, 3),
)
lattice_rot_dof = dict(
   TRICLINIC=3,
   MONOCLINIC=1,
   CUBIC=1,
   ORTHORHOMBIC=1,
   TETRAGONAL=1,
   HEXAGONAL=1,
)

sg_all_chiral = 'P1 P121 P1211 C121 P222 P2221 P21212 P212121 C2221 C222 F222 I222 I212121 P4 P41 P42 P43 I4 I41 P422 P4212 P4122 P41212 P4222 P42212 P4322 P43212 I422 I4122 P3 P31 P32 R3 P312 P321 P3112 P3121 P3212 P3221 R32 P6 P61 P65 P62 P64 P63 P622 P6122 P6522 P6222 P6422 P6322 P23 F23 I23 P213 I213 P432 P4232 F432 F4132 I432 P4332 P4132 I4132'.split()

sg_nframes = {'P1': 1, 'P121': 2, 'P1211': 2, 'C121': 4, 'P222': 4, 'P2221': 4, 'P21212': 4, 'P212121': 4, 'C2221': 8, 'C222': 8, 'F222': 16, 'I222': 8, 'I212121': 8, 'P4': 4, 'P41': 4, 'P42': 4, 'P43': 4, 'I4': 8, 'I41': 8, 'P422': 8, 'P4212': 8, 'P4122': 8, 'P41212': 8, 'P4222': 8, 'P42212': 8, 'P4322': 8, 'P43212': 8, 'I422': 16, 'I4122': 16, 'P3': 3, 'P31': 3, 'P32': 3, 'R3': 9, 'P312': 6, 'P321': 6, 'P3112': 6, 'P3121': 6, 'P3212': 6, 'P3221': 6, 'R32': 18, 'P6': 6, 'P61': 6, 'P65': 6, 'P62': 6, 'P64': 6, 'P63': 6, 'P622': 12, 'P6122': 12, 'P6522': 12, 'P6222': 12, 'P6422': 12, 'P6322': 12, 'P23': 12, 'F23': 48, 'I23': 24, 'P213': 12, 'I213': 24, 'P432': 24, 'P4232': 24, 'F432': 96, 'F4132': 96, 'I432': 48, 'P4332': 24, 'P4132': 24, 'I4132': 48}

two_iface_spacegroups = [
   'P212121',
   'P43212',
   'P3121',
   'P3221',
   'P6122',
   'P6522',
   'P41212',
   'I4',
   'P61',
   'R3',
   'P31',
   'P41',
   'P43',
   'P32',
   'P6',
   'P63',
   'P65',
   'I41',
   'P62',
   'P64',
   'P213',
   'I23',
   'I213',
   'P4132',
   'P4232',
   'P4332',
   'P432',
   'I432',
   'I4132',
   'F432',
   'F4132',
]

sg_niface_dict = {'P212121': 2, 'P43212': 2, 'P3121': 2, 'P3221': 2, 'P6122': 2, 'P6522': 2, 'P41212': 2, 'I4': 2, 'P61': 2, 'R3': 2, 'P31': 2, 'P41': 2, 'P43': 2, 'P32': 2, 'P6': 2, 'P63': 2, 'P65': 2, 'I41': 2, 'P62': 2, 'P64': 2, 'P213': 2, 'I23': 2, 'I213': 2, 'P4132': 2, 'P4232': 2, 'P4332': 2, 'P432': 2, 'I432': 2, 'I4132': 2, 'F432': 2, 'F4132': 2, 'P121': 3, 'C121': 3, 'C2221': 3, 'P21212': 3, 'P1': 3, 'I222': 3, 'I212121': 3, 'P42212': 3, 'I422': 3, 'P3112': 3, 'P6422': 3, 'R32': 3, 'P4212': 3, 'P4122': 3, 'P4322': 3, 'I4122': 3, 'P321': 3, 'P3212': 3, 'P622': 3, 'P6222': 3, 'P6322': 3, 'P42': 3, 'P4': 3, 'P3': 3, 'P23': 3, 'F23': 3, 'P2221': 4, 'C222': 4, 'F222': 4, 'P1211': 4, 'P312': 4, 'P422': 4, 'P4222': 4, 'P222': 5}

sg_triclinic = 'P1 P-1'.split()
sg_monoclinic = 'P121 P1211 C121 P1m1 P1c1 C1m1 C1c1 P12/m1 P121/m1 C12/m1 P12/c1 P121/c1 P121/n1 231 C12/c1'.split()
sg_orthorhombic = 'P222 P2221 P212S12 P21212 P212121 C2221 C222 F222 I222 I212121 Pmm2 Pmc21 Pcc2 Pma2 Pca21 Pnc2 Pmn21 Pba2 Pna21 Pnn2 Cmm2 Cmc21 Ccc2 Amm2 Abm2 Ama2 Aba2 Fmm2 Fdd2 Imm2 Iba2 Ima2 Pmmm Pnnn:2 Pccm Pban:2 Pmma Pnna Pmna Pcca Pbam Pccn Pbcm Pnnm Pmmn:2 Pbcn Pbca Pnma Cmcm Cmca Cmmm Cccm Cmma Ccca:2 Fmmm Fddd:2 Immm Ibam Ibca Imma'.split()
sg_tetragonal = 'P4 P41 P42 P43 I4 I41 P-4 I-4 P4/m P42/m P4/n:2 P42/n:2 I4/m I41/a:2 P422 P4212 P4122 P41212 P4222 P42212 P4322 P43212 I422 I4122 P4mm P4bm P42cm P42nm P4cc P4nc P42mc P42bc I4mm I4cm I41md I41cd P-42m P-42c P-421m P-421c P-4m2 P-4c2 P-4b2 P-4n2 I-4m2 I-4c2 I-42m I-42d P4/mmm P4/mcc P4/nbm:2 P4/nnc:2 P4/mbm P4/mnc P4/nmm:2 P4/ncc:2 P42/mmc P42/mcm P42/nbc:2 P42/nnm:2 P42/mbc P42/mnm P42/nmc:2 P42/ncm:2 I4/mmm I4/mcm I41/amd:2 I41/acd:2'.split()
sg_hexagonal = 'P3 P31 P32 R3 P-3 R-3:H P312 P321 P3112 P3121 P3212 P3221 R32 P3m1 P31m P3c1 P31c R3m:H R3c:H P-31m P-31c P-3m1 P-3c1 R-3m:H R-3c:H P6 P61 P65 P62 P64 P63 P-6 P6/m P63/m P622 P6122 P6522 P6222 P6422 P6322 P6mm P6cc P63cm P63mc P-6m2 P-6c2 P-62m P-62c P6/mmm P6/mcc P63/mcm P63/mmc'.split()
sg_cubic = 'P23 F23 I23 P213 I213 Pm-3 Pn-3:2 Fm-3 Fd-3:2 Im-3 Pa-3 Ia-3 P432 P4232 F432 F4132 I432 P4332 P4132 I4132 P-43m F-43m I-43m P-43n F-43c I-43d Pm-3m Pn-3n:2 Pm-3n Pn-3m:2 Fm-3m Fm-3c Fd-3m:2 Fd-3c:2 Im-3m Ia-3d'.split()

sg_lattice = dict()
for n in sg_triclinic:
   sg_lattice[n] = 'TRICLINIC'
for n in sg_monoclinic:
   sg_lattice[n] = 'MONOCLINIC'
for n in sg_orthorhombic:
   sg_lattice[n] = 'ORTHORHOMBIC'
for n in sg_tetragonal:
   sg_lattice[n] = 'TETRAGONAL'
for n in sg_hexagonal:
   sg_lattice[n] = 'HEXAGONAL'
for n in sg_cubic:
   sg_lattice[n] = 'CUBIC'

def get_spacegroup_pdbname():
   return {'P1': 'P 1', 'P-1': 'P -1', 'P2': 'P 2', 'P121': 'P 1 2 1', 'P1211': 'P 1 21 1', 'C121': 'C 1 2 1', 'P1m1': 'P 1 m 1', 'P1c1': 'P 1 c 1', 'C1m1': 'C 1 m 1', 'C1c1': 'C 1 c 1', 'P12/m1': 'P 1 2/m 1', 'P121/m1': 'P 1 21/m 1', 'C12/m1': 'C 1 2/m 1', 'P12/c1': 'P 1 2/c 1', 'P121/c1': 'P 1 21/c 1', 'P121/n1': 'P 1 21/n 1', '231': 'P 1 21/n 1', 'C12/c1': 'C 1 2/c 1', 'P222': 'P 2 2 2', 'P2221': 'P 2 2 21', 'P21212': 'P 21 21 2', 'P212121': 'P 21 21 21', 'C2221': 'C 2 2 21', 'C222': 'C 2 2 2', 'F222': 'F 2 2 2', 'I222': 'I 2 2 2', 'I212121': 'I 21 21 21', 'Pmm2': 'P m m 2', 'Pmc21': 'P m c 21', 'Pcc2': 'P c c 2', 'Pma2': 'P m a 2', 'Pca21': 'P c a 21', 'Pnc2': 'P n c 2', 'Pmn21': 'P m n 21', 'Pba2': 'P b a 2', 'Pna21': 'P n a 21', 'Pnn2': 'P n n 2', 'Cmm2': 'C m m 2', 'Cmc21': 'C m c 21', 'Ccc2': 'C c c 2', 'Amm2': 'A m m 2', 'Abm2': 'A b m 2', 'Ama2': 'A m a 2', 'Aba2': 'A b a 2', 'Fmm2': 'F m m 2', 'Fdd2': 'F d d 2', 'Imm2': 'I m m 2', 'Iba2': 'I b a 2', 'Ima2': 'I m a 2', 'Pmmm': 'P m m m', 'Pnnn:2': 'P n n n :2', 'Pccm': 'P c c m', 'Pban:2': 'P b a n :2', 'Pmma': 'P m m a', 'Pnna': 'P n n a', 'Pmna': 'P m n a', 'Pcca': 'P c c a', 'Pbam': 'P b a m', 'Pccn': 'P c c n', 'Pbcm': 'P b c m', 'Pnnm': 'P n n m', 'Pmmn:2': 'P m m n :2', 'Pbcn': 'P b c n', 'Pbca': 'P b c a', 'Pnma': 'P n m a', 'Cmcm': 'C m c m', 'Cmca': 'C m c a', 'Cmmm': 'C m m m', 'Cccm': 'C c c m', 'Cmma': 'C m m a', 'Ccca:2': 'C c c a :2', 'Fmmm': 'F m m m', 'Fddd:2': 'F d d d :2', 'Immm': 'I m m m', 'Ibam': 'I b a m', 'Ibca': 'I b c a', 'Imma': 'I m m a', 'P4': 'P 4', 'P41': 'P 41', 'P42': 'P 42', 'P43': 'P 43', 'I4': 'I 4', 'I41': 'I 41', 'P-4': 'P -4', 'I-4': 'I -4', 'P4/m': 'P 4/m', 'P42/m': 'P 42/m', 'P4/n:2': 'P 4/n :2', 'P42/n:2': 'P 42/n :2', 'I4/m': 'I 4/m', 'I41/a:2': 'I 41/a :2', 'P422': 'P 4 2 2', 'P4212': 'P 4 21 2', 'P4122': 'P 41 2 2', 'P41212': 'P 41 21 2', 'P4222': 'P 42 2 2', 'P42212': 'P 42 21 2', 'P4322': 'P 43 2 2', 'P43212': 'P 43 21 2', 'I422': 'I 4 2 2', 'I4122': 'I 41 2 2', 'P4mm': 'P 4 m m', 'P4bm': 'P 4 b m', 'P42cm': 'P 42 c m', 'P42nm': 'P 42 n m', 'P4cc': 'P 4 c c', 'P4nc': 'P 4 n c', 'P42mc': 'P 42 m c', 'P42bc': 'P 42 b c', 'I4mm': 'I 4 m m', 'I4cm': 'I 4 c m', 'I41md': 'I 41 m d', 'I41cd': 'I 41 c d', 'P-42m': 'P -4 2 m', 'P-42c': 'P -4 2 c', 'P-421m': 'P -4 21 m', 'P-421c': 'P -4 21 c', 'P-4m2': 'P -4 m 2', 'P-4c2': 'P -4 c 2', 'P-4b2': 'P -4 b 2', 'P-4n2': 'P -4 n 2', 'I-4m2': 'I -4 m 2', 'I-4c2': 'I -4 c 2', 'I-42m': 'I -4 2 m', 'I-42d': 'I -4 2 d', 'P4/mmm': 'P 4/m m m', 'P4/mcc': 'P 4/m c c', 'P4/nbm:2': 'P 4/n b m :2', 'P4/nnc:2': 'P 4/n n c :2', 'P4/mbm': 'P 4/m b m', 'P4/mnc': 'P 4/m n c', 'P4/nmm:2': 'P 4/n m m :2', 'P4/ncc:2': 'P 4/n c c :2', 'P42/mmc': 'P 42/m m c', 'P42/mcm': 'P 42/m c m', 'P42/nbc:2': 'P 42/n b c :2', 'P42/nnm:2': 'P 42/n n m :2', 'P42/mbc': 'P 42/m b c', 'P42/mnm': 'P 42/m n m', 'P42/nmc:2': 'P 42/n m c :2', 'P42/ncm:2': 'P 42/n c m :2', 'I4/mmm': 'I 4/m m m', 'I4/mcm': 'I 4/m c m', 'I41/amd:2': 'I 41/a m d :2', 'I41/acd:2': 'I 41/a c d :2', 'P3': 'P 3', 'P31': 'P 31', 'P32': 'P 32', 'R3': 'R 3 :H', 'P-3': 'P -3', 'R-3:H': 'R -3 :H', 'P312': 'P 3 1 2', 'P321': 'P 3 2 1', 'P3112': 'P 31 1 2', 'P3121': 'P 31 2 1', 'P3212': 'P 32 1 2', 'P3221': 'P 32 2 1', 'R32': 'R 3 2 :H', 'P3m1': 'P 3 m 1', 'P31m': 'P 3 1 m', 'P3c1': 'P 3 c 1', 'P31c': 'P 3 1 c', 'R3m:H': 'R 3 m :H', 'R3c:H': 'R 3 c :H', 'P-31m': 'P -3 1 m', 'P-31c': 'P -3 1 c', 'P-3m1': 'P -3 m 1', 'P-3c1': 'P -3 c 1', 'R-3m:H': 'R -3 m :H', 'R-3c:H': 'R -3 c :H', 'P6': 'P 6', 'P61': 'P 61', 'P65': 'P 65', 'P62': 'P 62', 'P64': 'P 64', 'P63': 'P 63', 'P-6': 'P -6', 'P6/m': 'P 6/m', 'P63/m': 'P 63/m', 'P622': 'P 6 2 2', 'P6122': 'P 61 2 2', 'P6522': 'P 65 2 2', 'P6222': 'P 62 2 2', 'P6422': 'P 64 2 2', 'P6322': 'P 63 2 2', 'P6mm': 'P 6 m m', 'P6cc': 'P 6 c c', 'P63cm': 'P 63 c m', 'P63mc': 'P 63 m c', 'P-6m2': 'P -6 m 2', 'P-6c2': 'P -6 c 2', 'P-62m': 'P -6 2 m', 'P-62c': 'P -6 2 c', 'P6/mmm': 'P 6/m m m', 'P6/mcc': 'P 6/m c c', 'P63/mcm': 'P 63/m c m', 'P63/mmc': 'P 63/m m c', 'P23': 'P 2 3', 'F23': 'F 2 3', 'I23': 'I 2 3', 'P213': 'P 21 3', 'I213': 'I 21 3', 'Pm-3': 'P m -3', 'Pn-3:2': 'P n -3 :2', 'Fm-3': 'F m -3', 'Fd-3:2': 'F d -3 :2', 'Im-3': 'I m -3', 'Pa-3': 'P a -3', 'Ia-3': 'I a -3', 'P432': 'P 4 3 2', 'P4232': 'P 42 3 2', 'F432': 'F 4 3 2', 'F4132': 'F 41 3 2', 'I432': 'I 4 3 2', 'P4332': 'P 43 3 2', 'P4132': 'P 41 3 2', 'I4132': 'I 41 3 2', 'P-43m': 'P -4 3 m', 'F-43m': 'F -4 3 m', 'I-43m': 'I -4 3 m', 'P-43n': 'P -4 3 n', 'F-43c': 'F -4 3 c', 'I-43d': 'I -4 3 d', 'Pm-3m': 'P m -3 m', 'Pn-3n:2': 'P n -3 n :2', 'Pm-3n': 'P m -3 n', 'Pn-3m:2': 'P n -3 m :2', 'Fm-3m': 'F m -3 m', 'Fm-3c': 'F m -3 c', 'Fd-3m:2': 'F d -3 m :2', 'Fd-3c:2': 'F d -3 c :2', 'Im-3m': 'I m -3 m', 'Ia-3d': 'I a -3 d'}

def get_spacegroup_tag():
   return {'P1': 'P1', 'P-1': 'Pminus1', 'P121': 'P121', 'P2': 'P121', 'P1211': 'P1211', 'P21': 'P1211', 'C121': 'C121', 'P1m1': 'P1m1', 'P1c1': 'P1c1', 'C1m1': 'C1m1', 'C1c1': 'C1c1', 'P12/m1': 'P12slashm1', 'P121/m1': 'P121slashm1', 'C12/m1': 'C12slashm1', 'P12/c1': 'P12slashc1', 'P121/c1': 'P121slashc1', "P121/n1": 'P121slashn1', "231": 'P121slashn1', 'C12/c1': 'C12slashc1', 'P222': 'P222', 'P2221': 'P2221', 'P21212': 'P21212', 'P212121': 'P212121', 'C2221': 'C2221', 'C222': 'C222', 'F222': 'F222', 'I222': 'I222', 'I212121': 'I212121', 'Pmm2': 'Pmm2', 'Pmc21': 'Pmc21', 'Pcc2': 'Pcc2', 'Pma2': 'Pma2', 'Pca21': 'Pca21', 'Pnc2': 'Pnc2', 'Pmn21': 'Pmn21', 'Pba2': 'Pba2', 'Pna21': 'Pna21', 'Pnn2': 'Pnn2', 'Cmm2': 'Cmm2', 'Cmc21': 'Cmc21', 'Ccc2': 'Ccc2', 'Amm2': 'Amm2', 'Abm2': 'Abm2', 'Ama2': 'Ama2', 'Aba2': 'Aba2', 'Fmm2': 'Fmm2', 'Fdd2': 'Fdd2', 'Imm2': 'Imm2', 'Iba2': 'Iba2', 'Ima2': 'Ima2', 'Pmmm': 'Pmmm', 'Pnnn:2': 'Pnnn__2', 'Pccm': 'Pccm', 'Pban:2': 'Pban__2', 'Pmma': 'Pmma', 'Pnna': 'Pnna', 'Pmna': 'Pmna', 'Pcca': 'Pcca', 'Pbam': 'Pbam', 'Pccn': 'Pccn', 'Pbcm': 'Pbcm', 'Pnnm': 'Pnnm', 'Pmmn:2': 'Pmmn__2', 'Pbcn': 'Pbcn', 'Pbca': 'Pbca', 'Pnma': 'Pnma', 'Cmcm': 'Cmcm', 'Cmca': 'Cmca', 'Cmmm': 'Cmmm', 'Cccm': 'Cccm', 'Cmma': 'Cmma', 'Ccca:2': 'Ccca__2', 'Fmmm': 'Fmmm', 'Fddd:2': 'Fddd__2', 'Immm': 'Immm', 'Ibam': 'Ibam', 'Ibca': 'Ibca', 'Imma': 'Imma', 'P4': 'P4', 'P41': 'P41', 'P42': 'P42', 'P43': 'P43', 'I4': 'I4', 'I41': 'I41', 'P-4': 'Pminus4', 'I-4': 'Iminus4', 'P4/m': 'P4slashm', 'P42/m': 'P42slashm', 'P4/n:2': 'P4slashn__2', 'P42/n:2': 'P42slashn__2', 'I4/m': 'I4slashm', 'I41/a:2': 'I41slasha__2', 'P422': 'P422', 'P4212': 'P4212', 'P4122': 'P4122', 'P41212': 'P41212', 'P4222': 'P4222', 'P42212': 'P42212', 'P4322': 'P4322', 'P43212': 'P43212', 'I422': 'I422', 'I4122': 'I4122', 'P4mm': 'P4mm', 'P4bm': 'P4bm', 'P42cm': 'P42cm', 'P42nm': 'P42nm', 'P4cc': 'P4cc', 'P4nc': 'P4nc', 'P42mc': 'P42mc', 'P42bc': 'P42bc', 'I4mm': 'I4mm', 'I4cm': 'I4cm', 'I41md': 'I41md', 'I41cd': 'I41cd', 'P-42m': 'Pminus42m', 'P-42c': 'Pminus42c', 'P-421m': 'Pminus421m', 'P-421c': 'Pminus421c', 'P-4m2': 'Pminus4m2', 'P-4c2': 'Pminus4c2', 'P-4b2': 'Pminus4b2', 'P-4n2': 'Pminus4n2', 'I-4m2': 'Iminus4m2', 'I-4c2': 'Iminus4c2', 'I-42m': 'Iminus42m', 'I-42d': 'Iminus42d', 'P4/mmm': 'P4slashmmm', 'P4/mcc': 'P4slashmcc', 'P4/nbm:2': 'P4slashnbm__2', 'P4/nnc:2': 'P4slashnnc__2', 'P4/mbm': 'P4slashmbm', 'P4/mnc': 'P4slashmnc', 'P4/nmm:2': 'P4slashnmm__2', 'P4/ncc:2': 'P4slashncc__2', 'P42/mmc': 'P42slashmmc', 'P42/mcm': 'P42slashmcm', 'P42/nbc:2': 'P42slashnbc__2', 'P42/nnm:2': 'P42slashnnm__2', 'P42/mbc': 'P42slashmbc', 'P42/mnm': 'P42slashmnm', 'P42/nmc:2': 'P42slashnmc__2', 'P42/ncm:2': 'P42slashncm__2', 'I4/mmm': 'I4slashmmm', 'I4/mcm': 'I4slashmcm', 'I41/amd:2': 'I41slashamd__2', 'I41/acd:2': 'I41slashacd__2', 'P3': 'P3', 'P31': 'P31', 'P32': 'P32', 'R3': 'R3', 'P-3': 'Pminus3', 'R-3:H': 'Rminus3__H', 'P312': 'P312', 'P321': 'P321', 'P3112': 'P3112', 'P3121': 'P3121', 'P3212': 'P3212', 'P3221': 'P3221', 'R32': 'R32', 'P3m1': 'P3m1', 'P31m': 'P31m', 'P3c1': 'P3c1', 'P31c': 'P31c', 'R3m:H': 'R3m__H', 'R3c:H': 'R3c__H', 'P-31m': 'Pminus31m', 'P-31c': 'Pminus31c', 'P-3m1': 'Pminus3m1', 'P-3c1': 'Pminus3c1', 'R-3m:H': 'Rminus3m__H', 'R-3c:H': 'Rminus3c__H', 'P6': 'P6', 'P61': 'P61', 'P65': 'P65', 'P62': 'P62', 'P64': 'P64', 'P63': 'P63', 'P-6': 'Pminus6', 'P6/m': 'P6slashm', 'P63/m': 'P63slashm', 'P622': 'P622', 'P6122': 'P6122', 'P6522': 'P6522', 'P6222': 'P6222', 'P6422': 'P6422', 'P6322': 'P6322', 'P6mm': 'P6mm', 'P6cc': 'P6cc', 'P63cm': 'P63cm', 'P63mc': 'P63mc', 'P-6m2': 'Pminus6m2', 'P-6c2': 'Pminus6c2', 'P-62m': 'Pminus62m', 'P-62c': 'Pminus62c', 'P6/mmm': 'P6slashmmm', 'P6/mcc': 'P6slashmcc', 'P63/mcm': 'P63slashmcm', 'P63/mmc': 'P63slashmmc', 'P23': 'P23', 'F23': 'F23', 'I23': 'I23', 'P213': 'P213', 'I213': 'I213', 'Pm-3': 'Pmminus3', 'Pn-3:2': 'Pnminus3__2', 'Fm-3': 'Fmminus3', 'Fd-3:2': 'Fdminus3__2', 'Im-3': 'Imminus3', 'Pa-3': 'Paminus3', 'Ia-3': 'Iaminus3', 'P432': 'P432', 'P4232': 'P4232', 'F432': 'F432', 'F4132': 'F4132', 'I432': 'I432', 'P4332': 'P4332', 'P4132': 'P4132', 'I4132': 'I4132', 'P-43m': 'Pminus43m', 'F-43m': 'Fminus43m', 'I-43m': 'Iminus43m', 'P-43n': 'Pminus43n', 'F-43c': 'Fminus43c', 'I-43d': 'Iminus43d', 'Pm-3m': 'Pmminus3m', 'Pn-3n:2': 'Pnminus3n__2', 'Pm-3n': 'Pmminus3n', 'Pn-3m:2': 'Pnminus3m__2', 'Fm-3m': 'Fmminus3m', 'Fm-3c': 'Fmminus3c', 'Fd-3m:2': 'Fdminus3m__2', 'Fd-3c:2': 'Fdminus3c__2', 'Im-3m': 'Imminus3m', 'Ia-3d': 'Iaminus3d'}  #, 'B11m': 'B11m'}

sg_pdbname = get_spacegroup_pdbname()
sg_from_pdbname = {v: k for k, v in sg_pdbname.items()}
sg_tag = get_spacegroup_tag()
sg_from_tag = {v: k for k, v in sg_tag.items()}

for i, (k, v) in enumerate(sg_tag.items()):
   if v in sg_lattice: sg_lattice[k] = sg_lattice[v]
   else: sg_lattice[v] = sg_lattice[k]
