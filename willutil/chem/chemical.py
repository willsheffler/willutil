from functools import lru_cache
import willutil as wu

aa1 = "ACDEFGHIKLMNPQRSTVWY"

aa3 = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
    "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
]

nucleic = ['ADE', 'CYT', 'GDP', 'GNP', 'GTP', 'GUA', 'THY', 'G', 'A', 'C', 'U', 'T']

aa123 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS",
             L="LEU", M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL",
             W="TRP", Y="TYR")

aa321 = dict(ALA="A", CYS="C", ASP="D", GLU="E", PHE="F", GLY="G", HIS="H", ILE="I", LYS="K",
             LEU="L", MET="M", ASN="N", PRO="P", GLN="Q", ARG="R", SER="S", THR="T", VAL="V",
             TRP="W", TYR="Y")

@lru_cache()
def rosetta_chem_data():
    return wu.storage.load_package_data('rosetta_residue_type_info.json.xz')
