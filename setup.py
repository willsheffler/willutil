from setuptools import setup, find_packages

setup(
   name='willutil', version='0.1', url='https://github.com/willsheffler/willutil',
   author='Will Sheffler', author_email='willsheffler@gmail.com',
   description='Common utility stuff extracted from various projects',
   packages=find_packages(include=['willutil', 'willutil.*']), install_requires=[
      'pytest',
      'tqdm',
      'yapf',
      'deferred_import',
      'numpy',
      'cppimport',
      'pandas',
      'xarray',
      'netcdf4',
   ], package_data={
      'willutil': [
         'data/rosetta_residue_type_info.json.xz',
         'data/pdb/meta/author.txt.xz',
         'data/pdb/meta/clust30.txt.xz',
         'data/pdb/meta/clust40.txt.xz',
         'data/pdb/meta/clust50.txt.xz',
         'data/pdb/meta/clust70.txt.xz',
         'data/pdb/meta/clust90.txt.xz',
         'data/pdb/meta/clust95.txt.xz',
         'data/pdb/meta/clust100.txt.xz',
         'data/pdb/meta/compound.txt.xz',
         'data/pdb/meta/entries.txt.xz',
         'data/pdb/meta/entrytypes.txt.xz',
         'data/pdb/meta/onhold.txt.xz',
         'data/pdb/meta/resl.txt.xz',
         'data/pdb/meta/seqres.txt.xz',
         'data/pdb/meta/source.txt.xz',
         'data/pdb/meta/xtal.txt.xz',
         'data/pdb/meta/lig/hetres_counts.pickle.xz',
         'data/pdb/meta/lig/hetres_pdbs.pickle.xz',
         'data/pdb/meta/lig/pdb_rescount.pickle.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains256.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains687.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains1718.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains2846.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains3582.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains4046.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains4574.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains4876.xz',
         'data/pdb/pisces/cullpdb_pc15.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains5042.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains285.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains868.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains2392.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains4222.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains5383.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains6108.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains6981.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains7469.xz',
         'data/pdb/pisces/cullpdb_pc20.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains7708.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains311.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1041.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains3253.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains6131.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains8109.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains9314.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains10682.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains11493.xz',
         'data/pdb/pisces/cullpdb_pc25.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains11822.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains336.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1161.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains3958.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains8074.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains10981.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains12746.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains14717.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains15857.xz',
         'data/pdb/pisces/cullpdb_pc30.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains16322.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains367.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1358.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains4938.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains10839.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains15334.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains18147.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains21333.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains23120.xz',
         'data/pdb/pisces/cullpdb_pc40.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains23880.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains389.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1476.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains5505.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains12541.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains18143.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains21718.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains25803.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains28115.xz',
         'data/pdb/pisces/cullpdb_pc50.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains29087.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains401.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1552.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains5860.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains13669.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains20003.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains24097.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains28855.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains31563.xz',
         'data/pdb/pisces/cullpdb_pc60.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains32707.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains412.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1611.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6149.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains14571.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains21452.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains25965.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains31238.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains34278.xz',
         'data/pdb/pisces/cullpdb_pc70.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains35564.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains418.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1642.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6397.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains15352.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains22736.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains27634.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains33395.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains36794.xz',
         'data/pdb/pisces/cullpdb_pc80.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains38233.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains425.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1686.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6702.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains16353.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains24413.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains29798.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains36260.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains40217.xz',
         'data/pdb/pisces/cullpdb_pc90.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains41933.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains432.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1725.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6901.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains17030.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains25625.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains31432.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains38520.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains42980.xz',
         'data/pdb/pisces/cullpdb_pc95.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains44920.xz',
         'tests/testdata/pdb/1coi.pdb1.gz',
         'tests/testdata/pdb/1pgx.pdb1.gz',
         'tests/testdata/pdb/1qys.pdb1.gz',
         'tests/testdata/pdb/respairdat10.nc',
         'tests/testdata/pdb/respairdat10_plus_xmap_rots.nc',
         'tests/testdata/pdb/3asl.pdb1.gz.pickle',
      ]
   })
