import numpy as np
import scipy.io as sio
from sto_adv_BA_algs import *
from sto_adv_BA_exp import *

delta, cc, C_w, C_3, C_init, C_gap, K, playtime, n = 0.1, 1, 16, 522, 100/9, 60, 20, 100, 6000
parameters = [delta, cc, C_w, C_3, C_init, C_gap]

er_SH_adv = np.zeros([4, ])
er_AdUCBE_adv = np.zeros([4, ])
er_S3BA_adv = np.zeros([4, ])

## Exp 1: easy game; quick convergence

loss_exp1 = sio.loadmat('./loss_exp1.mat')
loss_exp1 = loss_exp1['loss_exp1']

er_SH_adv[0] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp1, mu=[], var=[], turn_bud_to_N=True, verbose=1000)
er_AdUCBE_adv[0] = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp1, mu=[], var=[], turn_bud_to_N=True, verbose=500)
er_S3BA_adv[0] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp1, mu=[], var=[], turn_bud_to_N=True, verbose=500)
print(er_SH_adv[0], er_AdUCBE_adv[0], er_S3BA_adv[0])


## Exp 2: difficult game; quick convergence

loss_exp2 = sio.loadmat('./loss_exp2.mat')
loss_exp2 = loss_exp2['loss_exp2']

er_SH_adv[1] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp2, mu=[], var=[], turn_bud_to_N=True, verbose=1000)
er_AdUCBE_adv[1] = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp2, mu=[], var=[], turn_bud_to_N=True, verbose=500)
er_S3BA_adv[1] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp2, mu=[], var=[], turn_bud_to_N=True, verbose=500)
print(er_SH_adv[1], er_AdUCBE_adv[1], er_S3BA_adv[1])


## Exp 3: easy game; slow convergence

loss_exp3 = sio.loadmat('./loss_exp3.mat')
loss_exp3 = loss_exp3['loss_exp3']

er_SH_adv[2] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp3, mu=[], var=[], turn_bud_to_N=True, verbose=1000)
er_AdUCBE_adv[2] = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp3, mu=[], var=[], turn_bud_to_N=True, verbose=500)
er_S3BA_adv[2] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp3, mu=[], var=[], turn_bud_to_N=True, verbose=500)
print(er_SH_adv[2], er_AdUCBE_adv[2], er_S3BA_adv[2])


## Exp 4: difficult game; quick convergence

loss_exp4 = sio.loadmat('./loss_exp4.mat')
loss_exp4 = loss_exp4['loss_exp4']

er_SH_adv[3] = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp4, mu=[], var=[], turn_bud_to_N=True, verbose=1000)
er_AdUCBE_adv[3] = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp4, mu=[], var=[], turn_bud_to_N=True, verbose=500)
er_S3BA_adv[3] = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp4, mu=[], var=[], turn_bud_to_N=True, verbose=500)
print(er_SH_adv[3], er_AdUCBE_adv[3], er_S3BA_adv[3])


sio.savemat('./er_SH_adv.mat', {'er_SH_adv': er_SH_adv})
sio.savemat('./er_AdUCBE_adv.mat', {'er_AdUCBE_adv': er_AdUCBE_adv})
sio.savemat('./er_S3BA_adv.mat', {'er_S3BA_adv': er_S3BA_adv})
