import deferred_import, json
import willutil as wu

pyro = deferred_import.deferred_import('pyrosetta')

def get_rosetta_chem_data():
    return wu.storage.load_package_data('rosetta_residue_type_info.json.xz')

def extract_rosetta_chem_data(
        typeset='fa_standard',
        aas=list(),
        store=True,
        **kw,
):
    pyro.init()
    chem_manager = pyro.rosetta.core.chemical.ChemicalManager
    rts = chem_manager.get_instance().residue_type_set(typeset)
    aas = [rts.name_map(aa) for aa in aas]
    aas = aas or rts.base_residue_types()
    rosresinfo = dict()
    for rtype in aas:
        aaname = rtype.name()
        rosresinfo[aaname] = rosrestype_info(rtype)

    if store:
        wu.storage.save_package_data(rosresinfo, 'rosetta_residue_type_info.json.xz')
        # x = wu.storage.load_package_data('rosetta_residue_type_info.json.xz')
        # print(len(x))
        # assert len(x) == len(rosresinfo)

    return rosresinfo

def sanitize_crappy_rosetta_types(stuff):
    rutil = pyro.rosetta.utility
    for k, v in stuff.items():
        # print('------------------')
        # print(k)
        if isinstance(v, (
                rutil.vector1_int,
                rutil.vector1_unsigned_long,
                rutil.vector1_utility_vector1_unsigned_long_std_allocator_unsigned_long_t,
                rutil.vector1_utility_vector1_int_std_allocator_int_t,
                rutil.vector1_std_pair_unsigned_long_unsigned_long_t,
        )):
            v = list(v)
        if isinstance(v, list) and len(v) and isinstance(v[0], (
                rutil.vector1_int,
                rutil.vector1_unsigned_long,
        )):
            v = [list(x) for x in v]

        if isinstance(v, pyro.rosetta.core.chemical.AtomType):
            v = atomtype_info(v)
            for a in v:
                if isinstance(v[a], rutil.vector1_std_string):
                    v[a] = list(v[a])
        try:
            json.dumps(v)
        except:
            print('--------------------------')
            print('kv', k, v)
            assert 0
        stuff[k] = v
        # print(v)
    return stuff

def atomtype_info(atype):
    return dict(
        atom_has_orbital=atype.atom_has_orbital(),
        atom_type_name=atype.atom_type_name(),
        element=atype.element(),
        get_all_properties=atype.get_all_properties(),
        hybridization=str(atype.hybridization()).split('.')[-1],
        is_acceptor=atype.is_acceptor(),
        is_aromatic=atype.is_aromatic(),
        is_donor=atype.is_donor(),
        is_h2o=atype.is_h2o(),
        is_haro=atype.is_haro(),
        is_heavyatom=atype.is_heavyatom(),
        is_hydrogen=atype.is_hydrogen(),
        is_polar_hydrogen=atype.is_polar_hydrogen(),
        is_repulsive=atype.is_repulsive(),
        is_virtual=atype.is_virtual(),
        lj_radius=atype.lj_radius(),
        lj_wdepth=atype.lj_wdepth(),
        lk_dgfree=atype.lk_dgfree(),
        lk_lambda=atype.lk_lambda(),
        lk_volume=atype.lk_volume(),
        name=atype.name(),
    )

def rosrestype_info(restype):
    info = wu.Bunch()
    info.resinfo = sanitize_crappy_rosetta_types(
        dict(
            natoms=restype.natoms(),
            nheavyatoms=restype.nheavyatoms(),
            n_hbond_acceptors=restype.n_hbond_acceptors(),
            n_hbond_donors=restype.n_hbond_donors(),
            last_backbone_atom=restype.last_backbone_atom(),
            first_sidechain_atom=restype.first_sidechain_atom(),
            first_sidechain_hydrogen=restype.first_sidechain_hydrogen(),
            nbr_atom=restype.nbr_atom(),
            nbr_radius=restype.nbr_radius(),
            attached_H_begin=restype.attached_H_begin(),
            attached_H_end=restype.attached_H_end(),
            Haro_index=restype.Haro_index(),
            Hpol_index=restype.Hpol_index(),
            Hpos_polar=restype.Hpos_polar(),
            Hpos_apolar=restype.Hpos_apolar(),
            Hpos_polar_sc=restype.Hpos_polar_sc(),
            accpt_pos=restype.accpt_pos(),
            accpt_pos_sc=restype.accpt_pos_sc(),
            path_distances=restype.path_distances(),
            mainchain_atoms=restype.mainchain_atoms(),
            nchi=restype.nchi(),
            n_nus=restype.n_nus(),
            n_rings=restype.n_rings(),
            ndihe=restype.ndihe(),
            n_proton_chi=restype.n_proton_chi(),
            chi_atoms=restype.chi_atoms(),
            nu_atoms=restype.nu_atoms(),
            ring_atoms=restype.ring_atoms(),
            root_atom=restype.root_atom(),
            atoms_with_orb_index=restype.atoms_with_orb_index(),
            lower_connect_id=restype.lower_connect_id(),
            upper_connect_id=restype.upper_connect_id(),
            actcoord_atoms=restype.actcoord_atoms(),
            num_bondangles=restype.num_bondangles(),
            mass=restype.mass(),
            has=restype.has_shadow_atoms(),
            all_bb_atoms=restype.all_bb_atoms(),
            all_sc_atoms=restype.all_sc_atoms(),
            last_controlling_chi=restype.last_controlling_chi(),
            atoms_last_controlled_by_chi=restype.atoms_last_controlled_by_chi(),
            n_polymeric_residue_connections=restype.n_polymeric_residue_connections(),
            n_non_polymeric_residue_connections=restype.n_non_polymeric_residue_connections(),
            is_sidechain_thiol=restype.is_sidechain_thiol(),
            is_disulfide_bonded=restype.is_disulfide_bonded(),
            is_sidechain_amine=restype.is_sidechain_amine(),
            is_alpha_aa=restype.is_alpha_aa(),
            is_beta_aa=restype.is_beta_aa(),
            is_gamma_aa=restype.is_gamma_aa(),
            is_water=restype.is_water(),
            is_virtualizable_by_packer=restype.is_virtualizable_by_packer(),
            is_oligourea=restype.is_oligourea(),
            is_aramid=restype.is_aramid(),
            is_ortho_aramid=restype.is_ortho_aramid(),
            is_meta_aramid=restype.is_meta_aramid(),
            is_para_aramid=restype.is_para_aramid(),
            is_mirrored_type=restype.is_mirrored_type(),
            is_n_methylated=restype.is_n_methylated(),
            is_TNA=restype.is_TNA(),
            is_PNA=restype.is_PNA(),
            is_NA=restype.is_NA(),
            is_terminus=restype.is_terminus(),
            has_polymer_dependent_groups=restype.has_polymer_dependent_groups(),
        ))

    info.resatominfo = dict()
    for atomno in range(1, restype.natoms() + 1):
        atomname = restype.atom_name(atomno)
        info.resatominfo[atomname] = sanitize_crappy_rosetta_types(
            dict(
                atomno=atomno,
                atom_type=restype.atom_type(atomno),
                path_distance=restype.path_distance(atomno),
                attached_H_begin=restype.attached_H_begin(atomno),
                attached_H_end=restype.attached_H_end(atomno),
                # mainchain_atom=restype.mainchain_atom(mainchain_index),
                atom_is_backbone=restype.atom_is_backbone(atomno),
                atom_is_sidechain=restype.atom_is_sidechain(atomno),
                atom_is_hydrogen=restype.atom_is_hydrogen(atomno),
                nbonds=restype.nbonds(atomno),
                number_bonded_hydrogens=restype.number_bonded_hydrogens(atomno),
                cut_bond_neighbor=restype.cut_bond_neighbor(atomno),
                nbrs=restype.nbrs(atomno),
                # is_proton_chi=restype.is_proton_chi(chino),
                # proton_chi_2_chi=restype.proton_chi_2_chi(proton_chi_id),
                # chi_2_proton_chi=restype.chi_2_proton_chi(chi_index),
                # proton_chi_samples=restype.proton_chi_samples(proton_chi),
                # proton_chi_extra_samples=restype.proton_chi_extra_samples(proton_chi),
                # chi_rotamers=restype.chi_rotamers(chino),
                # chi_atoms=restype.chi_atoms(chino),
                # nu_atoms=restype.nu_atoms(nu_index),
                # ring_atoms=restype.ring_atoms(ring_num),
                # ring_saturation_type=restype.ring_saturation_type(ring_num),
                # dihedral=restype.dihedral(dihe),
                dihedrals_for_atom=restype.dihedrals_for_atom(atomno),
                # bondangle=restype.bondangle(bondang),
                bondangles_for_atom=restype.bondangles_for_atom(atomno),
                # atoms_within_one_bond_of_a_residue_connection=restype.
                # atoms_within_one_bond_of_a_residue_connection(resconn),
                within1bonds_sets_for_atom=restype.within1bonds_sets_for_atom(atomno),
                # atoms_within_two_bonds_of_a_residue_connection=restype.
                # atoms_within_two_bonds_of_a_residue_connection(resconn),
                # atom_being_shadowed=restype.atom_being_shadowed(atom_shadowing),
                #          atom_name=restype.atom_name(atomno),
                #          atom_index=restype.atom_index(atomname),
                #          atom_type_index=restype.atom_type_index(atomno),
                #          element_type=restype.element_type(atomno),
                #          heavyatom_is_an_acceptor=restype.heavyatom_is_an_acceptor(atomno),
                #          atom_is_polar_hydrogen=restype.atom_is_polar_hydrogen(atomno),
                #          atom_is_aro_hydrogen=restype.atom_is_aro_hydrogen(atomno),
                #          formal_charge=restype.formal_charge(atomno),
                #          atom_charge=restype.atom_charge(atomno),
                #          ideal_xyz=restype.ideal_xyz(atomno),
                #          atom_properties=restype.atom_properties(atomno),
                #          last_controlling_chi=restype.last_controlling_chi(atomno),
                #          # atoms_last_controlled_by_chi=restype.atoms_last_controlled_by_chi(chi),
                #          mm_atom_type_index=restype.mm_atom_type_index(atomno),
                #          bonded_orbitals=restype.bonded_orbitals(atomno),
                #          atom_forms_residue_connection=restype.atom_forms_residue_connection(atomno),
                #          n_residue_connections_for_atom=restype.n_residue_connections_for_atom(atomno),
                #          residue_connection_id_for_atom=restype.residue_connection_id_for_atom(atomno),
                #          residue_connections_for_atom=restype.residue_connections_for_atom(atomno),
                #          within2bonds_sets_for_atom=restype.within2bonds_sets_for_atom(atomno),
            ))

    return info
    # residue_connection_is_polymeric=restype.residue_connection_is_polymeric(resconn_id),
    # path_distance=restype.path_distance(at1, at2),
    # get_metal_binding_atoms=restype.get_metal_binding_atoms(AtomIndices
    # & metal_binding_indices),
