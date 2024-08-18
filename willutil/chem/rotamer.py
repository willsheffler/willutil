import numpy as np
import dataclasses
import willutil as wu

@dataclasses.dataclass
class RotamerSet:
    coords: np.ndarray  # f4
    rotnum: np.ndarray  # i4
    atomnum: np.ndarray  # i4
    atomtype: np.ndarray  # i4
    resname: np.ndarray  # <U3
    atomname: np.ndarray  # <U4
    rosetta_atom_type_index: np.ndarray  # i4

    def __post_init__(self):
        self.rotnum -= 1
        self.nrots = self.rotnum.max()

    def __len__(self):
        return self.nrots

    def __getitem__(self, slice):
        if isinstance(slice, int):
            slice = self.rotnum == slice
        rs = RotamerSet(**{k: v[slice] for k, v in dataclasses.asdict(self).items()})
        rotnum = rs.rotnum[rs.atomnum == 1]
        idx = -np.ones(rotnum.max() + 1, dtype=int)
        idx[rotnum] = np.arange(len(rotnum))
        rs.rotnum = idx[rs.rotnum]
        return rs

    def rotamers(self, res):
        r = self[self.resname == res]
        natom = r.atomnum.max()
        rots = {f: v.reshape(-1, natom) for f, v in dataclasses.asdict(r).items()}
        del rots['atomnum']
        rots = Rotamers(**rots)
        rots.rotnum = np.ascontiguousarray(rots.rotnum[:, 0])
        rots.resname = np.ascontiguousarray(rots.resname[:, 0])
        rots.atomname = np.ascontiguousarray(rots.atomname[0])
        rots.atomtype = np.ascontiguousarray(rots.atomtype[0])
        rots.rosetta_atom_type_index = np.ascontiguousarray(rots.rosetta_atom_type_index[0])
        rots.coords = rots.coords.reshape(-1, natom, 3)
        return rots

@dataclasses.dataclass
class Rotamers:
    coords: np.ndarray  # f4
    rotnum: np.ndarray  # i4
    atomtype: np.ndarray  # i4
    resname: np.ndarray  # <U3
    atomname: np.ndarray  # <U4
    rosetta_atom_type_index: np.ndarray  # i4

    def __getitem__(self, slice):
        return Rotamers(**{k: v[slice] for k, v in dataclasses.asdict(self).items()})

    def items(self):
        return dataclasses.asdict(self).items()

    def align(self, atoms):
        assert len(atoms) > 2
        anames = [n.strip() for n in self.atomname]
        idx = [anames.index(a) for a in atoms]
        frames = wu.hframe(*self.coords[:, idx].swapaxes(0, 1))
        ic(frames.shape, self.coords.shape)
        self.coords = wu.hxform(wu.hinv(frames), self.coords, outerprod=False)

def get_rotamerset():
    data = wu.load_package_data('rotamer/rosetta_rots_1res')
    del data['resnum']
    del data['onebody']
    return RotamerSet(**data)
