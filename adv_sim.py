import numpy as np
from numpy import random
import pickle
import scipy.io as sio
import seaborn

from sto_adv_BA_algs import *
from sto_adv_BA_exp import *

#delta, cc, C_w, C_3, C_init, C_gap = 0.1, 1, 0.5, 1, 10, 2
delta, cc, C_w, C_3, C_init, C_gap = 0.1, 1, 16, 522, 100. / 9, 60  # only exp4 trigger
K, playtime, n = 20, 100, 6000
parameters = [delta, cc, C_w, C_3, C_init, C_gap]

er_SH_adv = np.zeros([4, ])
er_AdUCBE_adv = np.zeros([4, ])
er_S3BA_adv = np.zeros([4, ])

## Exp 1: easy game; quick convergence

loss_exp1 = sio.loadmat('./loss_exp1.mat')
loss_exp1 = loss_exp1['loss_exp1']

er_SH_adv[0], trigger_rate, trigger_time = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp1, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=1000)
print trigger_rate

er_AdUCBE_adv[0], trigger_rate, trigger_time = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp1, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate

er_S3BA_adv[0], trigger_rate, trigger_time = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp1, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate, np.mean(np.array(trigger_time))
pickle.dump(trigger_time, open('trigger_time_exp1.pkl', 'wb'))

print(er_SH_adv[0], er_AdUCBE_adv[0], er_S3BA_adv[0])



## Exp 2: difficult game; quick convergence

loss_exp2 = sio.loadmat('./loss_exp2.mat')
loss_exp2 = loss_exp2['loss_exp2']

er_SH_adv[1], trigger_rate, trigger_time = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp2, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=1000)
print trigger_rate

er_AdUCBE_adv[1], trigger_rate, trigger_time = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp2, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate

er_S3BA_adv[1], trigger_rate, trigger_time = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp2, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate, np.mean(np.array(trigger_time))
pickle.dump(trigger_time, open('trigger_time_exp2.pkl', 'wb'))

print(er_SH_adv[1], er_AdUCBE_adv[1], er_S3BA_adv[1])


## Exp 3: easy game; slow convergence

loss_exp3 = sio.loadmat('./loss_exp3.mat')
loss_exp3 = loss_exp3['loss_exp3']

er_SH_adv[2], trigger_rate, trigger_time = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp3, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=1000)
print trigger_rate

er_AdUCBE_adv[2], trigger_rate, trigger_time = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp3, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate

er_S3BA_adv[2], trigger_rate, trigger_time = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp3, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate, np.mean(np.array(trigger_time))
pickle.dump(trigger_time, open('trigger_time_exp3.pkl', 'wb'))

print(er_SH_adv[2], er_AdUCBE_adv[2], er_S3BA_adv[2])


## Exp 4: difficult game; slow convergence

loss_exp4 = sio.loadmat('./loss_exp4.mat')
loss_exp4 = loss_exp4['loss_exp4']

er_SH_adv[3], trigger_rate, trigger_time = Exp(alg=Successive_Halving, playtimes=playtime, c=[], other_alg_parameters=[], N=n, K=K,
                   loss_generate=False, losses=loss_exp4, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=1000)
print trigger_rate

er_AdUCBE_adv[3], trigger_rate, trigger_time = Exp(alg=Ad_UCBE, playtimes=playtime, c=cc, other_alg_parameters=[], N=n, K=K,
                       loss_generate=False, losses=loss_exp4, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate

er_S3BA_adv[3], trigger_rate, trigger_time = Exp(alg=S3_BA, playtimes=playtime, c=[], other_alg_parameters=parameters, N=n, K=K,
                     loss_generate=False, losses=loss_exp4, mu=[], var=[], best_arm=1, turn_bud_to_N=True, verbose=500)
print trigger_rate, np.mean(np.array(trigger_time))
pickle.dump(trigger_time, open('trigger_time_exp4.pkl', 'wb'))

print(er_SH_adv[3], er_AdUCBE_adv[3], er_S3BA_adv[3])


sio.savemat('./er_SH_adv.mat', {'er_SH_adv': er_SH_adv})
sio.savemat('./er_AdUCBE_adv.mat', {'er_AdUCBE_adv': er_AdUCBE_adv})
sio.savemat('./er_S3BA_adv.mat', {'er_S3BA_adv': er_S3BA_adv})


### draw trigger-time distribution
