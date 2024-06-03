import sys
import logging
import collections

xr = deferred_import.deferred_import("xarray")
import willutil as wu

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
import sys


def get_lig_counts(files_or_pattern, hetonly=True):
    rncount = collections.Counter()
    natom = collections.defaultdict(list)
    ligpdbs = collections.defaultdict(list)
    pdbligs = dict()
    skipres = set(_.encode() for _ in wu.chem.aa3 + wu.chem.nucleic + ["HOH"])

    for fname, pdb in tqdm.tqdm(
        wu.pdb.pdbread.gen_pdbs(
            files_or_pattern,
            cache=True,
            skip_errors=True,
        )
    ):
        try:
            # print('extract lig info', fname)
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            # print(pdb.df)

            # for a, b in pdb.df.groupby('ri').rn:
            # print(a, len(b))
            # assert 0
            df = pdb.df
            if hetonly:
                df = df[df.het]

            resnames = df.groupby("ri").rn.apply(lambda x: x.iloc[0])
            resnums = df.groupby("ri").ri.apply(lambda x: x.iloc[0])
            resnatom = df.groupby("ri").rn.count()

            # for i, t in df.iterrows():
            # print(t.ri, t.rn)
            for ri, rn, na in zip(resnums, resnames, resnatom):
                #    print(ri, rn, na)
                natom[rn].append(int(na))
            rncount.update(resnames)
            pdbligs[pdb.code] = {k.decode(): v for k, v in collections.Counter(resnames).items()}
            for rn in df.rn.unique():
                if rn not in skipres:
                    ligpdbs[rn].append(pdb.code)
        except Error as e:
            print("error on", fname)
            print(repr(e))

    natom = {k: (sum(v) / len(v)) for k, v in natom.items()}
    rncount = {k: v for k, v in rncount.items()}
    common = set(natom.keys()).intersection(set(rncount.keys()))
    natom = {k: natom[k] for k in common}
    rncount = {k: rncount[k] for k in common}
    df = pd.DataFrame(dict(count=rncount, natom=natom))
    df = df.sort_values("count", ascending=False)
    df = df.set_index(df.index.astype("U4"))
    ligpdbs = {k.decode(): v for k, v in ligpdbs.items()}

    wu.storage.save(df, "hetres_counts.pickle")
    wu.storage.save(ligpdbs, "hetres_pdbs.pickle")
    wu.storage.save(pdbligs, "pdb_hetres.pickle")


def main(files_or_pattern):
    with wu.Timer():
        print(sys.argv[:4])
        get_lig_counts(files_or_pattern, hetonly=False)
        # extract_het_pdbfiles(files_or_pattern)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main("/home/sheffler/data/rcsb/divided/??/*.pdb?.gz.het.pickle")
