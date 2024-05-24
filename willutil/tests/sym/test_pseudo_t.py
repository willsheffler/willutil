import willutil as wu
from willutil import *
from icecream import ic

def test_t2():
   wu.sym.create_pseudo_t(7)

def test_t4():
   wu.sym.create_pseudo_t(4)

def test_pseudo_t_dist_min():
   asym = wu.sym.pseudo_t_start(2)
   loss, asym2 = wu.sym.min_pseudo_t_dist2(asym)
   ic(asym2.shape)
   # wu.showme(hscaled(0.1, asym))
   # wu.showme(hscaled(0.1, asym2))
   # wu.showme(hscaled(0.1, wu.sym.make('I', asym)))

def test_pseudo_t_env_min():
   asym = wu.sym.pseudo_t_start(2)
   loss, asym2 = wu.sym.min_pseudo_t_symerror(asym)
   ic(asym2.shape)
   # wu.showme(hscaled(0.1, asym))
   # wu.showme(hscaled(0.1, asym2))
   # wu.showme(hscaled(0.1, wu.sym.make('I', asym)))

def test_from_pdb():
   import numpy as np
   ficos = wu.sym.frames('I')
   axes = wu.sym.axes('I')

   # frames = np.load('willutil/data/pseudo_t/T2_4btg.npy')
   # frames = np.load('willutil/data/pseudo_t/T2_3iz3.npy')
   # frames = np.load('willutil/data/pseudo_t/T2_7cbp_D.npy')  # 222 res
   # frames = np.load('willutil/data/pseudo_t/T2_7cbp_E.npy')  # 215 res
   # frames = np.load('willutil/data/pseudo_t/T3_7cbp_K.npy')  # 501 res
   # frames = np.load('willutil/data/pseudo_t/T3_7cbp_T.npy')  # 75 res
   # frames = np.load('willutil/data/pseudo_t/T3_2wbh_A129.npy')
   # frames = np.load('willutil/data/pseudo_t/T3_6rrs_A129.npy')
   # frames = np.load('willutil/data/pseudo_t/T3_2tbv.npy')
   # frames = np.load('willutil/data/pseudo_t/T4_1ohf_A510.npy')

   # frames = np.load('willutil/data/pseudo_t/T4_6rrt_A128.npy')
   # frames = np.load('willutil/data/pseudo_t/T4_1qgt_A142.npy')
   # frames = np.load('willutil/data/pseudo_t/T7_6o3h.npy')
   # frames = np.load('willutil/data/pseudo_t/T7_1ohg_A200.npy')
   frames = np.load('willutil/data/pseudo_t/T9_8h89_J155.npy')
   # frames = np.load('willutil/data/pseudo_t/T13_2btv.npy')

   if 0:
      frames[:, :3, 3] -= frames[:, :3, 3].mean(0)
      dumppdb('test.pdb', hcart(frames) / 10)
      a2 = (-13.205300, -7.494900, 20.899100)
      # a3 = (-4.202000,3.476000,15.283333)
      a5 = (-25.814400, 0.964600, 15.965600)
      # print(hangle(a2, a5))
      # print(hangle(axes[2], axes[5]))
      xaln = wu.halign2(a2, a5, axes[2], axes[5], strict=True)
      frames = hxform(xaln, frames)
      # return
      # wu.storage.save_package_data(frames, 'pseudo_t/T13_2btv.npy')

   # frames = wu.sym.pseudo_t_start(2)
   asym = frames[0]
   wu.showme(frames, weight=10, xyzlen=[10, 7, 4])
   # wu.showme(wu.hxform(frames0, asym), weight=10, xyzlen=[10, 7, 4])
   wu.showme(wu.hxform(wu.sym.frames('I'), frames[0]), weight=10, xyzlen=[10, 7, 4])
   wu.showme(wu.hxform(wu.sym.frames('I'), frames[-1]), weight=10, xyzlen=[10, 7, 4])

def main():
   test_pseudo_t_env_min()
   # test_pseudo_t_start()
   # test_from_pdb()
   # test_t2()

if __name__ == '__main__':
   main()
