import numpy as np
import willutil as wu

# yapf: disable

test_asuslide_case2_coords=np.array(
         [[[2.04448719e+01, 2.04590130e+00, -7.86451101e-01], [1.90442734e+01, 2.13035393e+00, -1.18247569e+00],
           [1.83011150e+01, 8.41098309e-01, -8.56961012e-01]],
          [[1.37144899e+01, 4.78648305e-01, -1.25086498e+00], [1.50636101e+01, 9.52432752e-01, -9.66187358e-01],
           [1.56305122e+01, 1.73909616e+00, -2.14101624e+00]],
          [[1.75918198e+01, -8.64950776e-01, -7.49649704e-01], [1.75002670e+01, -3.52461100e-01, 6.12187982e-01],
           [1.77665672e+01, 1.14675069e+00, 6.55614972e-01]],
          [[1.55580797e+01, 1.95528269e+00, -1.42992640e+00], [1.57699385e+01, 1.42434037e+00, -8.87127519e-02],
           [1.48881025e+01, 2.09125072e-01, 1.68296754e-01]],
          [[1.59783316e+01, -6.10352516e-01, -2.12161803e+00], [1.47687187e+01, -6.22749507e-01, -1.30777967e+00],
           [1.50794821e+01, -9.99330640e-01, 1.35153398e-01]],
          [[1.76629028e+01, -7.19160140e-01, -2.33132744e+00], [1.74876633e+01, -1.90539098e+00, -1.50199580e+00],
           [1.60261860e+01, -2.33215332e+00, -1.45287609e+00]],
          [[1.76013470e+01, 1.70260119e+00, -8.24000180e-01], [1.64121342e+01, 1.41668558e+00, -3.04815918e-02],
           [1.51968317e+01, 2.15799570e+00, -5.72635651e-01]],
          [[1.70272846e+01, -1.86413333e-01, 6.01175606e-01], [1.56560392e+01, 2.97346532e-01, 4.94893312e-01],
           [1.50044661e+01, 4.11969393e-01, 1.86702585e+00]],
          [[1.57275219e+01, 2.27222466e+00, -3.47527601e-02], [1.53235273e+01, 8.82105470e-01, -2.07932234e-01],
           [1.62933044e+01, -6.41607866e-02, 4.88173574e-01]],
          [[1.42689705e+01, 1.40846801e+00, -1.38541543e+00], [1.52978678e+01, 8.28514338e-01, -5.30627608e-01],
           [1.57710819e+01, 1.82775211e+00, 5.17242789e-01]],
          [[1.15566099e+00, -7.10495901e+00, -6.86922789e-01], [1.95095277e+00, -6.69541693e+00, 4.64348495e-01],
           [1.06984472e+00, -6.44479990e+00, 1.68142653e+00]],
          [[1.63747902e+01, -7.43966877e-01, 1.06922865e+00], [1.60419922e+01, -1.57073331e+00, 2.22306776e+00],
           [1.63236084e+01, -3.04193616e+00, 1.94613230e+00]],
          [[1.27207232e+01, -3.06410766e+00, -7.97445029e-02], [1.21846161e+01, -2.11406589e+00, 8.87543023e-01],
           [1.20091572e+01, -7.34600961e-01, 2.65667289e-01]],
          [[1.34239864e+01, -2.02490973e+00, 2.40504289e+00], [1.46760950e+01, -1.31540394e+00, 2.63844180e+00],
           [1.45297470e+01, -2.93827742e-01, 3.75889444e+00]],
          [[1.56426487e+01, 2.43592978e+00, 1.29912287e-01], [1.43634911e+01, 1.82113814e+00, 4.63706195e-01],
           [1.45121899e+01, 3.19796026e-01, 6.74254894e-01]],
          [[1.27153530e+01, -9.27376032e-01, 4.42267132e+00], [1.21352863e+01, -1.84125817e+00, 3.44595575e+00],
           [1.28365564e+01, -3.19328403e+00, 3.47229457e+00]],
          [[1.40579987e+01, 5.59682488e-01, -3.52990665e-02], [1.44056883e+01, -2.48379350e-01, 1.12736511e+00],
           [1.36702023e+01, 2.31946945e-01, 2.37186933e+00]],
          [[1.43977518e+01, -4.23211956e+00, 4.25188255e+00], [1.46945057e+01, -3.22614789e+00, 3.23916388e+00],
           [1.34739094e+01, -2.93827534e+00, 2.37446022e+00]],
          [[1.30419035e+01, 1.48204237e-01, 2.93017077e+00], [1.43673410e+01, -2.69414961e-01, 3.37114143e+00],
           [1.54012966e+01, 8.19747627e-01, 3.11599779e+00]],
          [[1.54748268e+01, 2.44238526e-01, 3.49390078e+00], [1.61681461e+01, -5.96699953e-01, 4.46228409e+00],
           [1.70567150e+01, 2.35392526e-01, 5.37798882e+00]],
          [[1.45282927e+01, -1.10684454e+00, 2.26700950e+00], [1.57757120e+01, -1.53876889e+00, 2.88590145e+00],
           [1.69671230e+01, -7.86344349e-01, 2.30725813e+00]],
          [[1.21226444e+01, 6.69778809e-02, 4.80274582e+00], [1.29708862e+01, 9.92495418e-01, 4.06142998e+00],
           [1.25313692e+01, 2.43510461e+00, 4.27626705e+00]],
          [[1.57602301e+01, 3.09536719e+00, 5.20226622e+00], [1.46446114e+01, 2.66631317e+00, 4.36743975e+00],
           [1.33270950e+01, 3.24257541e+00, 4.86994600e+00]],
          [[1.25907583e+01, 3.58901113e-01, 5.46799946e+00], [1.29692335e+01, 1.72460365e+00, 5.12557983e+00],
           [1.43828239e+01, 2.03933263e+00, 5.59797239e+00]],
          [[1.45663586e+01, 2.52287030e-01, 4.73263264e+00], [1.53262463e+01, 1.19612575e+00, 5.54342556e+00],
           [1.47114983e+01, 1.35217237e+00, 6.92840862e+00]],
          [[1.55494556e+01, 1.79678833e+00, 1.79668581e+00], [1.42745857e+01, 1.67543352e+00, 2.49354243e+00],
           [1.39378843e+01, 2.16918856e-01, 2.77599883e+00]],
          [[1.47370625e+01, 3.61771464e+00, 4.67119646e+00], [1.40707531e+01, 3.41374159e+00, 5.95184374e+00],
           [1.50821991e+01, 3.22222376e+00, 7.07467222e+00]],
          [[1.40755539e+01, 2.70846343e+00, 3.05196214e+00], [1.54934177e+01, 2.46923256e+00, 2.81100273e+00],
           [1.63317795e+01, 3.67573309e+00, 3.21342301e+00]],
          [[1.48519239e+01, 2.43406609e-01, 4.73855972e+00], [1.52803288e+01, 1.61192846e+00, 5.00173330e+00],
           [1.43161573e+01, 2.31907892e+00, 5.94553137e+00]],
          [[1.67134724e+01, 3.07795572e+00, 4.39014769e+00], [1.52900791e+01, 3.38965321e+00, 4.43955517e+00],
           [1.44788609e+01, 2.39087677e+00, 3.62419319e+00]],
          [[1.32469740e+01, 2.40128255e+00, 2.59808040e+00], [1.43415184e+01, 1.57711363e+00, 2.09972453e+00],
           [1.53164635e+01, 1.22292626e+00, 3.21528769e+00]],
          [[1.46508389e+01, 2.84711504e+00, 6.83102226e+00], [1.53259449e+01, 1.62493491e+00, 6.41133118e+00],
           [1.48944321e+01, 4.37333226e-01, 7.26212120e+00]],
          [[1.78763657e+01, 4.58335304e+00, 3.65928864e+00], [1.75042896e+01, 3.67987871e+00, 4.74138212e+00],
           [1.82624855e+01, 2.36226892e+00, 4.64412737e+00]],
          [[1.53566246e+01, -4.17760760e-01, 4.86370134e+00], [1.44558506e+01, -9.06893373e-01, 5.90052128e+00],
           [1.47637587e+01, -2.63086259e-01, 7.24631071e+00]],
          [[1.72808132e+01, 9.27557588e-01, 2.58314133e+00], [1.69799862e+01, -3.62507433e-01, 3.19214559e+00],
           [1.80521622e+01, -7.59924114e-01, 4.19859219e+00]],
          [[1.73609295e+01, 2.20560217e+00, 4.42946243e+00], [1.76337605e+01, 1.39771020e+00, 5.61205196e+00],
           [1.84241390e+01, 1.45816043e-01, 5.25355864e+00]],
          [[1.69976387e+01, 1.35578132e+00, 8.76470280e+00], [1.63928261e+01, 1.93660438e-01, 8.12492847e+00],
           [1.68562393e+01, 5.67227378e-02, 6.68030787e+00]],
          [[1.41193724e+01, 3.74222025e-02, 4.49280119e+00], [1.42992315e+01, -1.15463018e+00, 3.67286301e+00],
           [1.52544308e+01, -2.13947964e+00, 4.33475780e+00]],
          [[1.37584724e+01, -2.34956235e-01, 6.54343128e+00], [1.49578590e+01, 2.53152966e-01, 5.87346649e+00],
           [1.46372261e+01, 7.68694818e-01, 4.47637606e+00]],
          [[1.31071100e+01, 1.64614415e+00, 5.47401333e+00], [1.42941351e+01, 8.16850185e-01, 5.30405235e+00],
           [1.48974600e+01, 9.96325254e-01, 3.91688752e+00]],
          [[1.36752462e+01, -1.18778908e+00, 7.90398121e+00], [1.42817822e+01, -9.32873368e-01, 6.60291910e+00],
           [1.57888184e+01, -1.15076649e+00, 6.64553308e+00]],
          [[1.33179522e+01, -1.10722148e+00, 4.57600307e+00], [1.46083632e+01, -1.75459766e+00, 4.77937984e+00],
           [1.46077948e+01, -2.59767008e+00, 6.04810333e+00]],
          [[1.91498356e+01, -4.61448431e-01, 5.55518389e+00], [1.89762592e+01, -1.63825381e+00, 6.39817953e+00],
           [1.75038128e+01, -1.88955379e+00, 6.69682837e+00]],
          [[1.32709084e+01, -3.21482944e+00, 3.87100720e+00], [1.40265913e+01, -4.10640621e+00, 2.99941492e+00],
           [1.47980556e+01, -3.32276511e+00, 1.94528115e+00]],
          [[1.77968731e+01, -1.68548107e+00, 1.80251098e+00], [1.89269466e+01, -2.55315661e+00, 1.49313009e+00],
           [1.90974827e+01, -3.63435268e+00, 2.55255008e+00]],
          [[1.64999161e+01, -2.16976690e+00, 4.05510998e+00], [1.74404163e+01, -1.77928364e+00, 3.01174355e+00],
           [1.70079746e+01, -2.31348443e+00, 1.65231025e+00]],
          [[1.76104298e+01, -4.27643108e+00, 2.97628093e+00], [1.74604168e+01, -4.07436371e+00, 1.54020858e+00],
           [1.60764961e+01, -3.53548670e+00, 1.20137358e+00]],
          [[1.70602264e+01, -3.19115281e+00, 6.74386799e-01], [1.75131989e+01, -1.99581933e+00, 1.37555397e+00],
           [1.76794243e+01, -2.26139116e+00, 2.86628652e+00]],
          [[2.00487671e+01, -3.18758249e+00, 2.26752090e+00], [2.01421471e+01, -1.86564600e+00, 2.87528610e+00],
           [2.08305378e+01, -8.77774060e-01, 1.94218385e+00]],
          [[1.81998348e+01, -1.40916014e+00, 1.86327958e+00], [1.77441101e+01, -3.90617251e-01, 9.24911857e-01],
           [1.84674168e+01, -5.08805156e-01, -4.10490245e-01]],
          [[1.84403019e+01, -2.84100151e+00, 2.78153706e+00], [1.81855946e+01, -1.40547597e+00, 2.77643585e+00],
           [1.72434521e+01, -1.00919247e+00, 3.90593028e+00]],
          [[1.82866192e+01, -3.27423334e+00, 1.36289847e+00], [1.71931973e+01, -2.33748055e+00, 1.59217501e+00],
           [1.76653767e+01, -8.95647824e-01, 1.45577931e+00]],
          [[1.86651955e+01, -2.45792583e-01, 2.46714973e+00], [1.99814129e+01, -8.70983243e-01, 2.51545334e+00],
           [2.00891514e+01, -2.00670910e+00, 1.50601971e+00]],
          [[1.88425293e+01, -7.87893891e-01, 2.96664405e+00], [1.97372074e+01, -2.09759176e-01, 1.97117805e+00],
           [1.89960747e+01, 7.63733983e-01, 1.06371748e+00]],
          [[1.71032124e+01, -2.67969823e+00, 2.45524979e+00], [1.71957989e+01, -1.47476149e+00, 1.63965392e+00],
           [1.78860264e+01, -3.48145068e-01, 2.39778757e+00]],
          [[1.82593174e+01, -1.43861508e+00, 3.35743070e+00], [1.75508995e+01, -1.47675633e+00, 4.63114023e+00],
           [1.75927086e+01, -1.21555246e-01, 5.32548380e+00]],
          [[1.40505114e+01, 2.14702189e-01, 1.81301367e+00], [1.54804230e+01, 8.22369754e-02, 1.56113088e+00],
           [1.57868462e+01, 1.60758540e-01, 7.10359663e-02]],
          [[1.59634094e+01, 1.54515350e+00, 8.33694875e-01], [1.71364460e+01, 2.02248049e+00, 1.11348271e-01],
           [1.69393616e+01, 3.45232415e+00, -3.75628173e-01]],
          [[1.40236216e+01, 3.41041946e+00, 2.93139648e+00], [1.44867601e+01, 2.23958302e+00, 2.19637060e+00],
           [1.59622116e+01, 2.36230612e+00, 1.83797991e+00]],
          [[2.12938328e+01, 2.79293847e+00, 2.63089705e+00], [2.00955124e+01, 2.36005044e+00, 3.33961558e+00],
           [1.98773079e+01, 3.18008184e+00, 4.60467911e+00]],
          [[2.01581993e+01, 1.24836659e+00, 4.49337053e+00], [1.89518585e+01, 2.06452131e+00, 4.55872011e+00],
           [1.77233219e+01, 1.20988894e+00, 4.84285593e+00]],
          [[1.89189510e+01, 2.02560997e+00, 3.44368315e+00], [1.75252361e+01, 1.61008310e+00, 3.34097648e+00],
           [1.73080902e+01, 2.46607676e-01, 3.98457289e+00]],
          [[1.83589687e+01, 2.82316017e+00, 1.97816336e+00], [1.80182114e+01, 1.45843375e+00, 2.36162305e+00],
           [1.83130379e+01, 1.21046424e+00, 3.83540606e+00]],
          [[1.54724836e+01, 2.08352399e+00, 2.24917221e+00], [1.68689384e+01, 2.48883605e+00, 2.14303398e+00],
           [1.76236153e+01, 2.20872593e+00, 3.43626285e+00]],
          [[2.07362633e+01, 4.34827375e+00, 1.20472574e+00], [1.99604225e+01, 4.37889957e+00, 2.43873310e+00],
           [1.90803776e+01, 3.14216352e+00, 2.56692600e+00]],
          [[2.05722103e+01, 3.94158554e+00, 5.43729305e+00], [1.92751770e+01, 4.09718704e+00, 4.78989077e+00],
           [1.85259323e+01, 2.77214742e+00, 4.73201752e+00]],
          [[1.75611382e+01, 2.28161573e+00, 7.67181635e-01], [1.79012814e+01, 1.80100405e+00, 2.10095477e+00],
           [1.93906288e+01, 1.95776820e+00, 2.37973976e+00]],
          [[2.02956753e+01, 2.63022780e+00, 3.54341078e+00], [1.88646469e+01, 2.78520751e+00, 3.77530026e+00],
           [1.84936810e+01, 4.24970293e+00, 3.97044039e+00]],
          [[2.11783752e+01, 1.23079884e+00, 4.62380934e+00], [1.99843655e+01, 1.95833254e+00, 4.21068573e+00],
           [1.88155880e+01, 1.67486858e+00, 5.14557934e+00]],
          [[1.56690235e+01, 4.20675898e+00, 3.48947358e+00], [1.63715649e+01, 2.95177817e+00, 3.72841644e+00],
           [1.78738136e+01, 3.11640644e+00, 3.53713584e+00]],
          [[1.78946648e+01, 1.24114513e+00, 7.09250116e+00], [1.92256527e+01, 1.15884840e+00, 6.50314903e+00],
           [2.01484013e+01, 2.87908107e-01, 7.34603548e+00]],
          [[1.80929413e+01, -5.76938968e-03, 6.02543163e+00], [1.85790005e+01, -4.30548489e-01, 7.33269930e+00],
           [1.76506271e+01, -1.46493399e+00, 7.95609236e+00]],
          [[1.47494974e+01, 5.69137812e-01, 6.27208090e+00], [1.55930662e+01, -4.16457176e-01, 6.93739653e+00],
           [1.63865986e+01, -1.23442125e+00, 5.92660999e+00]],
          [[1.71675301e+01, -2.14999199e-01, 6.80527735e+00], [1.59216852e+01, -9.70752597e-01, 6.75680113e+00],
           [1.59259539e+01, -2.10560012e+00, 7.77294350e+00]],
          [[1.81295471e+01, 8.19048584e-01, 7.25556755e+00], [1.91076450e+01, 1.89854085e+00, 7.19514227e+00],
           [1.87466869e+01, 3.02336860e+00, 8.15686703e+00]]], dtype=np.float32)



# yapf: enable