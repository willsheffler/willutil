import numpy as np
import willutil as wu

def main():
   test_hxformpts_bug()
   # test_homog_misc1()

def test_hxformpts_bug():
   pts6 = np.array([
      [39.50315091, 24.89703772, 28.00395481, 1.],
      [39.50315091, 24.89703772, 34.25316121, 1.],
      [45.7523573, 31.14624412, 34.25316121, 1.],
      [45.7523573, 31.14624412, 34.25316121, 1.],
      [45.7523573, 31.14624412, 40.50236761, 1.],
      [45.7523573, 31.14624412, 40.50236761, 1.],
   ])
   pts4 = np.array([[37.42008211, 26.98010652, 25.92088601, 1.], [37.42008211, 26.98010652, 25.92088601, 1.], [37.42008211, 26.98010652, 38.41929881, 1.], [37.42008211, 26.98010652, 38.41929881, 1.]])
   frames = np.array([[[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[1.11022302e-16, -1.11022302e-16, 1.00000000e+00, 0.00000000e+00], [1.00000000e+00, 1.11022302e-16, -1.11022302e-16, 0.00000000e+00], [-1.11022302e-16, 1.00000000e+00, 1.11022302e-16, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-2.22044605e-16, 1.00000000e+00, 2.22044605e-16, 0.00000000e+00], [2.22044605e-16, -2.22044605e-16, 1.00000000e+00, 0.00000000e+00], [1.00000000e+00, 2.22044605e-16, -2.22044605e-16, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-1.00000000e+00, -1.22464680e-16, 0.00000000e+00, 1.00000000e+00], [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, 5.00000000e-01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]])
   result = wu.hxformpts(frames, pts6)
   assert result.shape == (4, 6, 4)
   result = wu.hxformpts(frames, pts4, outerprod=True)
   assert result.shape == (4, 4, 4)
   ic(result.shape)

def test_homog_misc1():
   # coords = np.array([[
   # [0, 0, 0],
   # [2.03473740e+01, -1.49673691e+01, -1.06923075e+01],
   # [2.02065296e+01, -1.35257816e+01, -1.08585186e+01],
   # [1.87417812e+01, -1.31084919e+01, -1.08297768e+01], [1.83886948e+01, -1.20876551e+01, -1.02393579e+01]],
   # [[1.80053673e+01, -1.38227987e+01, -1.14688034e+01], [1.65830917e+01, -1.35046196e+01, -1.15079212e+01],
   # [1.59670610e+01, -1.35631762e+01, -1.01159725e+01], [1.51399183e+01, -1.27251148e+01, -9.75700474e+00]],
   # [[1.63503494e+01, -1.45062838e+01, -9.36852455e+00], [1.58219585e+01, -1.46048174e+01, -8.01326466e+00],
   # [1.61854095e+01, -1.33759928e+01, -7.18965292e+00], [1.53551931e+01, -1.28421154e+01, -6.45406771e+00]],
   # ]])

   coords = np.array([[[1.25488234, 14.92007065, 16.20718193], [0.44757202, 14.39806747, 15.11109829], [1.24185705, 14.3593111, 13.81184769], [1.06575823, 13.45800686, 12.9920845]], [[2.13055038, 15.18398857, 13.71091557], [2.98480177, 15.16873932, 12.52953529], [3.84040833, 13.90922737, 12.48469448], [4.06813955, 13.33961678, 11.41742992]], [[4.17481279, 13.42983627, 13.61832905], [4.98391581, 12.21820545, 13.67255116], [4.20570326, 11.01281738, 13.16082096], [4.73995495, 10.18437862, 12.42350483]], [[2.97270155, 10.92389774, 13.50240517], [2.16935158, 9.78858376, 13.06499481], [1.9380188, 9.82482529, 11.55979919], [2.01440787, 8.79779625, 10.88544941]], [[1.6794374, 10.96183777, 11.042346], [1.46338141, 11.09329224, 9.60649204], [2.71246052, 10.70861721, 8.82401562], [2.6337347, 10.0039854, 7.81770229]], [[3.82948875, 11.18543434, 9.25978088], [5.07415819, 10.83906078, 8.58416271], [5.37927151, 9.35278606, 8.71957684], [5.84616518, 8.71655369, 7.77480745]], [[5.06588507, 8.79284477, 9.82806206], [5.26728058, 7.35874033, 9.99664783], [4.40965223, 6.56357479, 9.0206213], [4.87456131, 5.59782314, 8.41518402]], [[3.21274424, 7.0016675, 8.8249979], [2.32254195, 6.32281828, 7.89100695], [2.81087494, 6.47376108, 6.45601892], [2.80254912, 5.51552868, 5.68330145]], [[3.18737173, 7.65393782, 6.1036334], [3.67324829, 7.89649153, 4.75058937], [4.93277359, 7.08888292, 4.46456861], [5.07461882, 6.50096369, 3.3923738]], [[5.77222872, 6.98848486, 5.43195724], [7.00198507, 6.2269125, 5.24935579], [6.71463346, 4.73511553, 5.13793755], [7.30852795, 4.03858376, 4.31483173]], [[5.78487492, 4.25007439, 5.91199493], [5.42441368, 2.84164119, 5.80236244], [4.78828812, 2.53938818, 4.45164776], [5.06585598, 1.50714207, 3.84107876]], [[4.00729609, 3.45846653, 3.9593606], [3.39541674, 3.27035737, 2.64945412], [4.44878626, 3.23289394, 1.54970288], [4.37464237, 2.41405082, 0.63353187]], [[5.42454529, 4.09881067, 1.65327168], [6.49185371, 4.11382818, 0.66016352], [7.30966187, 2.8296752, 0.71099865], [7.66898918, 2.26982856, -0.3247745]], [[7.59729958, 2.37013578, 1.89873612], [8.32098961, 1.11178303, 2.03459406], [7.540802, -0.0417614, 1.41727245], [8.10114384, -0.86709851, 0.69601941]], [[6.27290726, -0.05657578, 1.69035172], [5.43463278, -1.11527705, 1.14072978], [5.42305517, -1.07415938, -0.38197124], [5.55115509, -2.10575104, -1.04134929]], [[5.22198725, 0.07722619, -0.92251849], [5.18762255, 0.21926971, -2.37313175], [6.52756691, -0.15271951, -2.99491668], [6.57936192, -0.81613427, -4.03056526]], [[7.57158995, 0.1538555, -2.29846382], [8.90331268, -0.14524746, -2.81099486], [9.19033146, -1.64038002, -2.75959539], [9.76765251, -2.20369554, -3.68951249]], [[8.76265907, -2.27979279, -1.69305599], [8.97411728, -3.71864724, -1.59000683], [8.21717834, -4.46611691, -2.68031693], [8.75449944, -5.37812901, -3.30871964]], [[7.00060272, -4.08091354, -2.91645908], [6.2206583, -4.71434927, -3.97290683], [6.82885742, -4.44536638, -5.34337568], [6.89498758, -5.33693361, -6.18960571]], [[7.26200914, -3.23090792, -5.54966307], [7.90227127, -2.8838439, -6.81269407], [9.11692429, -3.76492643, -7.07490349], [9.31700516, -4.24703836, -8.18976021]], [[9.86565304, -3.99786711, -6.04705191], [11.04700947, -4.84045553, -6.1887455], [10.66372204, -6.27104187, -6.54505587], [11.28047085, -6.89178133, -7.41089344]], [[9.65533352, -6.7663641, -5.89712143], [9.20582104, -8.12129116, -6.19334316], [8.74654865, -8.24526024, -7.64045906], [9.08463001, -9.20824242, -8.32874393]], [[8.01441288, -7.27828598, -8.09287739], [7.55640602, -7.30064201, -9.47684479], [8.72865772, -7.22108269, -10.44634914], [8.76676369, -7.93408108, -11.44912148]], [[9.68653965, -6.41938686, -10.10545826], [10.86691761, -6.29989576, -10.95284557], [11.65506172, -7.60302353, -10.9864645], [12.09464836, -8.04658222, -12.04730797]], [[11.75862312, -8.21899128, -9.86054325], [12.49285507, -9.47650528, -9.78845024], [11.77347755, -10.57909298, -10.5547657], [12.40296936, -11.38134193, -11.24433613]], [[10.50677776, -10.55658627, -10.53348827], [9.74757004, -11.54153442, -11.29448509], [9.92974567, -11.33931065, -12.79327202], [10.08487415, -12.30179596, -13.5448885]], [[9.97388458, -10.1202755, -13.18280029], [10.19470215, -9.83743763, -14.59591007], [11.58602142, -10.2759037, -15.03456688], [11.75964832, -10.82755184, -16.1212635]], [[12.48724747, -10.19516659, -14.11239815], [13.84663391, -10.61863613, -14.42605209], [13.94307137, -12.13598061, -14.51990986], [14.62489605, -12.67135811, -15.39389896]], [[13.13298512, -12.7776556, -13.79194832], [13.13897896, -14.23536587, -13.81802464], [12.35053635, -14.76825047, -15.00749302], [12.73965645, -15.75646591, -15.62989712]], [[11.42422009, -14.04975128, -15.45797157], [10.64277077, -14.50667858, -16.60085869], [11.38838673, -14.270051, -17.90795708], [11.48963451, -15.16527367, -18.74682045]]])

   coords = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0], [12, 0, 0]])

   symframes = np.array([[[5.00000000e-01, -8.66025404e-01, 0.00000000e+00, 2.79414700e+01], [8.66025404e-01, 5.00000000e-01, 0.00000000e+00, -1.21448246e-12], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-1.00000000e+00, -4.21999173e-14, 0.00000000e+00, 2.79414700e+01], [4.21999173e-14, -1.00000000e+00, 0.00000000e+00, -1.28575567e-12], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, -8.66025404e-01, 0.00000000e+00, 4.19122050e+01], [8.66025404e-01, -5.00000000e-01, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-1.00000000e+00, -4.04121181e-14, 0.00000000e+00, -1.39707350e+01], [4.04121181e-14, -1.00000000e+00, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[1.00000000e+00, 4.11452054e-14, 0.00000000e+00, -4.19122050e+01], [-4.11452054e-14, 1.00000000e+00, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, 8.66025404e-01, 0.00000000e+00, -4.19122050e+01], [-8.66025404e-01, -5.00000000e-01, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, -8.66025404e-01, 0.00000000e+00, 4.19122050e+01], [8.66025404e-01, -5.00000000e-01, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[5.00000000e-01, 8.66025404e-01, 0.00000000e+00, -1.39707350e+01], [-8.66025404e-01, 5.00000000e-01, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, 8.66025404e-01, 0.00000000e+00, -4.19122050e+01], [-8.66025404e-01, -5.00000000e-01, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[1.00000000e+00, 3.95468244e-14, 0.00000000e+00, -4.19122050e+01], [-3.95468244e-14, 1.00000000e+00, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[1.00000000e+00, 3.77035141e-14, 0.00000000e+00, -2.47549681e-12], [-3.77035141e-14, 1.00000000e+00, 0.00000000e+00, 1.64894000e-12], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-1.00000000e+00, -3.63598041e-14, 0.00000000e+00, -1.39707350e+01], [3.63598041e-14, -1.00000000e+00, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[5.00000000e-01, -8.66025404e-01, 0.00000000e+00, -1.39707350e+01], [8.66025404e-01, 5.00000000e-01, 0.00000000e+00, -2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[5.00000000e-01, 8.66025404e-01, 0.00000000e+00, 2.79414700e+01], [-8.66025404e-01, 5.00000000e-01, 0.00000000e+00, 5.61165149e-13], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, 8.66025404e-01, 0.00000000e+00, -2.23353096e-13], [-8.66025404e-01, -5.00000000e-01, 0.00000000e+00, 1.63498044e-12], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[5.00000000e-01, 8.66025404e-01, 0.00000000e+00, -1.39707350e+01], [-8.66025404e-01, 5.00000000e-01, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[-5.00000000e-01, -8.66025404e-01, 0.00000000e+00, -1.39409856e-12], [8.66025404e-01, -5.00000000e-01, 0.00000000e+00, -1.00547669e-12], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], [[5.00000000e-01, -8.66025404e-01, 0.00000000e+00, -1.39707350e+01], [8.66025404e-01, 5.00000000e-01, 0.00000000e+00, 2.41980228e+01], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]])

   assert wu.hvalid(symframes)
   # ic(np.linalg.det(symframes[:, :3, :3]))
   # wu.showme(symframes)
   # ic(np.linalg.norm(symframes[:, :3, 3], axis=1))
   # wu.showme(coords.reshape(-1, 3))
   # ic(coords.shape)
   # ic(coords)
   x = wu.hxform(symframes, coords, homogout=True)
   # x = wu.hxform(symframes.swapaxes(1, 2), coords)
   # ic(x.shape)
   # ic(x[:, :, 3])
   # wu.showme(x.reshape(-1, 4))

if __name__ == '__main__':
   main()
