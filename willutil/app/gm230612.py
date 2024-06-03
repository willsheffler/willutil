import willutil as wu


def hascyclic(sg):
    if sg == "P622":
        return True
    return any([e.iscyclic for e in wu.sym.sg_symelem_dict[sg]])


def hascompound(sg):
    if sg == "P622":
        return True
    return any([e.iscompound for e in wu.sym.sg_symelem_dict[sg]])


def hasscrew(sg):
    if sg == "P622":
        return True
    return any([e.isscrew for e in wu.sym.sg_symelem_dict[sg]])


def main():
    print(" ".join(wu.sym.two_iface_spacegroups))

    for lat in "TRICLINIC MONOCLINIC ORTHORHOMBIC TETRAGONAL HEXAGONAL CUBIC".split():
        print(lat)
        for sg in wu.sym.sg_all_chiral:
            if wu.sym.sg_lattice[sg] == lat:
                print(sg, end=" ")
        print()

    print("1face w/compound")
    oneifacecyclic = set()
    oneifacecompound = set()
    oneifacescrew = set()
    for sg, niface in wu.sym.sg_niface_dict.items():
        if niface == 2 and hascyclic(sg):
            oneifacecyclic.add(sg)

    for sg, niface in wu.sym.sg_niface_dict.items():
        if niface == 3 and hascompound(sg):
            oneifacecompound.add(sg)

    for sg, niface in wu.sym.sg_niface_dict.items():
        if niface == 2 and hasscrew(sg):
            oneifacescrew.add(sg)

    print("-" * 80)
    oneiface = oneifacecyclic.union(oneifacescrew).union(oneifacecompound)
    print("cyc", len(oneifacecyclic))
    print(" ".join(oneifacecyclic))
    print("compound", len(oneifacecompound))
    print(" ".join(oneifacecompound))
    print("screw", len(oneifacescrew - oneifacecyclic))
    print(" ".join(oneifacescrew - oneifacecyclic))

    print("-" * 80)

    twoifacecyclic = set()
    twoifacecompound = set()
    twoifacescrew = set()
    for sg, niface in wu.sym.sg_niface_dict.items():
        if sg in oneiface:
            continue
        if niface == 3 and hascyclic(sg):
            twoifacecyclic.add(sg)

    for sg, niface in wu.sym.sg_niface_dict.items():
        if sg in oneiface:
            continue
        if niface == 4 and hascompound(sg):
            twoifacecompound.add(sg)

    for sg, niface in wu.sym.sg_niface_dict.items():
        if sg in oneiface:
            continue
        if niface == 3 and hasscrew(sg):
            twoifacescrew.add(sg)

    twoiface = twoifacecyclic.union(twoifacescrew).union(twoifacecompound)
    print("cyc", len(twoifacecyclic))
    print(" ".join(twoifacecyclic))
    print("compound", len(twoifacecompound))
    print(" ".join(twoifacecompound))
    print("screw", len(twoifacescrew - twoifacecyclic))
    print(" ".join(twoifacescrew - twoifacecyclic))
    print("-" * 80)

    print(len(oneiface))
    print(len(twoiface))
    print(len(wu.sym.sg_all_chiral))

    for sg in wu.sym.sg_all_chiral:
        if sg in oneiface:
            continue
        if sg in twoiface:
            continue
        print(sg, wu.sym.sg_niface_dict[sg], hascompound(sg), hascyclic(sg))

    lattelems = defaultdict(set)
    for sg in wu.sym.sg_all_chiral:
        if sg == "P622":
            continue
        lat = wu.sym.latticetype(sg)
        for e in wu.sym.symelems(sg):
            lattelems[lat].add(e.label)
    for lat in lattelems:
        print(lat, lattelems[lat])

    print("C11")
    for k, v in wu.sym.sg_symelem_dict.items():
        if any([e.label == "C11" for e in v]):
            print(k, end=" ")
    print()

    for e in wu.sym.symelems("P222"):
        print(e)

    print(wu.hangle_degrees([1, 0, 0], [1, 1, 1]))
    print(wu.hangle_degrees([1, 1, 0], [1, 1, 1]))


if __name__ == "__main__":
    main()
